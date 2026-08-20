"""Independent Captum implementation of the synthetic-shortcut audit.

This module intentionally depends only on general-purpose scientific Python,
PyTorch / torchvision, and Captum.  The experiment artifacts are shared with
the primary workflow, but none of its implementation is imported.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import captum
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import yaml
from captum.attr import FeatureAblation
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

RAW_COLUMNS = (
    "item_key",
    "model",
    "dataset",
    "sample_id",
    "gt_label",
    "region_key",
    "target_region_key",
    "is_control",
    "control_index",
    "perturbation",
    "seed_salt",
    "clean_margin",
    "perturbed_margin",
    "degradation",
    "source_area",
    "model_area",
    "status",
)


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation used for identities and hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = canonical_json(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_config(path: Path, repo_root: Path | None = None) -> dict[str, Any]:
    """Load and resolve one reference configuration."""

    config_path = path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != "captum-reference-v1":
        raise ValueError("expected schema_version='captum-reference-v1'")
    root = (repo_root or config_path.parents[3]).resolve()
    resolved = json.loads(json.dumps(config))
    resolved["repo_root"] = str(root)
    resolved["config_path"] = str(config_path)
    for group in ("manifests", "checkpoints"):
        resolved["data"][group] = {
            key: str((root / value).resolve())
            for key, value in resolved["data"][group].items()
        }
    resolved["data"]["dataset_stats"] = str(
        (root / resolved["data"]["dataset_stats"]).resolve()
    )
    _validate_config(resolved)
    return resolved


def _validate_config(config: Mapping[str, Any]) -> None:
    if config["device"] != "cuda":
        raise ValueError("the full reference experiment requires device: cuda")
    rows = int(config["regions"]["rows"])
    cols = int(config["regions"]["cols"])
    input_size = tuple(config["preprocessing"]["input_size"])
    if input_size[0] % rows or input_size[1] % cols:
        raise ValueError("grid must divide the source image exactly")
    if tuple(config["seed_salts"]) != (0, 1, 2):
        raise ValueError("reference seed_salts must remain [0, 1, 2]")
    if int(config["regions"]["controls_per_region"]) != 2:
        raise ValueError("reference controls_per_region must remain 2")
    required_ops = {
        "constant_fill",
        "mean_fill",
        "blur",
        "gaussian_noise",
        "patch_shuffle",
    }
    if set(config["perturbations"]) != required_ops:
        raise ValueError("reference config must declare all five perturbations")
    for group in ("manifests", "checkpoints"):
        for value in config["data"][group].values():
            if not Path(value).is_file():
                raise FileNotFoundError(value)
    if not Path(config["data"]["dataset_stats"]).is_file():
        raise FileNotFoundError(config["data"]["dataset_stats"])


class ManifestDataset(Dataset):
    """Minimal JSON-manifest image dataset."""

    def __init__(self, path: Path) -> None:
        document = json.loads(path.read_text(encoding="utf-8"))
        self.samples = tuple(document["samples"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        entry = self.samples[index]
        image = np.asarray(Image.open(entry["path"]).convert("RGB"), dtype=np.uint8).copy()
        return torch.from_numpy(image), int(entry["gt_label"]), str(entry["sample_id"])


class RawMarginModel(nn.Module):
    """Wrap a classifier so Captum sees source-space pixels and returns margins."""

    def __init__(
        self,
        model: nn.Module,
        output_size: Sequence[int],
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        super().__init__()
        self.model = model
        self.output_size = tuple(int(value) for value in output_size)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.forward_evaluations = 0

    def forward(self, raw_images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.forward_evaluations += int(raw_images.shape[0])
        images = raw_images.permute(0, 3, 1, 2).float().div(255.0)
        images = F.interpolate(images, size=self.output_size, mode="bilinear", align_corners=False)
        logits = self.model((images - self.mean) / self.std)
        labels = labels.long().view(-1, 1)
        gt = logits.gather(1, labels).squeeze(1)
        masked = logits.scatter(1, labels, float("-inf"))
        return gt - masked.max(dim=1).values


def load_model(config: Mapping[str, Any], model_name: str) -> RawMarginModel:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full Captum reference experiment")
    from torchvision.models import squeezenet1_0

    spec = config["model"]
    if spec["architecture"] != "squeezenet1_0":
        raise ValueError("only squeezenet1_0 is supported by this fixed experiment")
    classifier = squeezenet1_0(weights=None, num_classes=int(spec["num_classes"]))
    checkpoint = torch.load(
        config["data"]["checkpoints"][model_name], map_location="cpu", weights_only=True
    )
    classifier.load_state_dict(checkpoint[spec["checkpoint_state_dict_key"]])
    classifier.eval().cuda()
    preprocessing = config["preprocessing"]
    return RawMarginModel(
        classifier,
        preprocessing["output_size"],
        preprocessing["mean"],
        preprocessing["std"],
    ).eval().cuda()


def grid_feature_mask(height: int, width: int, rows: int, cols: int) -> torch.Tensor:
    """Return a channel-broadcastable integer mask with one ID per grid cell."""

    if height % rows or width % cols:
        raise ValueError("grid must divide image dimensions")
    mask = torch.empty((1, height, width, 1), dtype=torch.long)
    cell_h, cell_w = height // rows, width // cols
    for row in range(rows):
        for col in range(cols):
            mask[:, row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w] = (
                row * cols + col
            )
    return mask


def region_key(row: int, col: int) -> str:
    return f"grid::grid/r{row}/c{col}"


def matched_control_masks(
    *,
    sample_ids: Sequence[str],
    target_row: int,
    target_col: int,
    control_index: int,
    height: int,
    width: int,
    rows: int,
    cols: int,
    global_seed: int,
) -> tuple[torch.Tensor, list[str]]:
    """Build deterministic rigid translations of one target cell per sample."""

    cell_h, cell_w = height // rows, width // cols
    masks = torch.ones((len(sample_ids), height, width, 1), dtype=torch.long)
    keys: list[str] = []
    target = region_key(target_row, target_col)
    for index, sample_id in enumerate(sample_ids):
        rng = np.random.default_rng(
            stable_seed(global_seed, sample_id, target, control_index, "matched-control")
        )
        row_offset = int(rng.integers(0, height - cell_h + 1))
        col_offset = int(rng.integers(0, width - cell_w + 1))
        masks[index, row_offset : row_offset + cell_h, col_offset : col_offset + cell_w, 0] = 0
        keys.append(f"control:{target}:{control_index}@{row_offset},{col_offset}")
    return masks, keys


def perturbation_baseline(
    images: torch.Tensor,
    operator: str,
    params: Mapping[str, Any],
    *,
    channel_mean: Sequence[float],
    global_seed: int,
    sample_ids: Sequence[str],
    seed_salt: int,
) -> torch.Tensor:
    """Build one full-frame candidate; Captum composites feature groups from it."""

    arrays = images.cpu().numpy().astype(np.uint8, copy=False)
    candidates: list[np.ndarray] = []
    for array, sample_id in zip(arrays, sample_ids, strict=True):
        if operator == "constant_fill":
            candidate = np.full_like(array, float(params["value"]), dtype=np.uint8)
        elif operator == "mean_fill":
            values = np.asarray(channel_mean, dtype=np.uint8)
            candidate = np.broadcast_to(values, array.shape).copy()
        elif operator == "blur":
            candidate = cv2.GaussianBlur(
                array,
                ksize=(0, 0),
                sigmaX=float(params["sigma"]),
                sigmaY=float(params["sigma"]),
                borderType=cv2.BORDER_REFLECT_101,
            )
        elif operator == "gaussian_noise":
            rng = np.random.default_rng(
                stable_seed(global_seed, sample_id, operator, seed_salt)
            )
            noise = rng.normal(0.0, float(params["sigma"]), size=array.shape)
            candidate = np.clip(np.rint(array.astype(np.float64) + noise), 0, 255).astype(
                np.uint8
            )
        elif operator == "patch_shuffle":
            rng = np.random.default_rng(
                stable_seed(global_seed, sample_id, operator, seed_salt)
            )
            patch = int(params["patch_size"])
            tile_rows, tile_cols = array.shape[0] // patch, array.shape[1] // patch
            candidate = array.copy()
            permutation = rng.permutation(tile_rows * tile_cols)
            for destination, source in enumerate(permutation):
                dst_row, dst_col = divmod(destination, tile_cols)
                src_row, src_col = divmod(int(source), tile_cols)
                candidate[
                    dst_row * patch : (dst_row + 1) * patch,
                    dst_col * patch : (dst_col + 1) * patch,
                ] = array[
                    src_row * patch : (src_row + 1) * patch,
                    src_col * patch : (src_col + 1) * patch,
                ]
        else:
            raise ValueError(f"unknown perturbation: {operator}")
        candidates.append(candidate)
    return torch.from_numpy(np.stack(candidates)).to(images.device, dtype=torch.float32)


def _attribution_values(
    ablator: FeatureAblation,
    images: torch.Tensor,
    labels: torch.Tensor,
    baseline: torch.Tensor,
    feature_mask: torch.Tensor,
    feature_ids: Sequence[int],
    perturbations_per_eval: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return clean margins and one ablation value per requested feature/sample."""

    with torch.no_grad():
        clean = ablator.forward_func(images, labels)
        attribution = ablator.attribute(
            images,
            baselines=baseline,
            additional_forward_args=(labels,),
            feature_mask=feature_mask,
            perturbations_per_eval=perturbations_per_eval,
        )
    values = np.empty((images.shape[0], len(feature_ids)), dtype=np.float64)
    masks = feature_mask
    if masks.shape[0] == 1:
        masks = masks.expand(images.shape[0], -1, -1, -1)
    for batch_index in range(images.shape[0]):
        for feature_index, feature_id in enumerate(feature_ids):
            selected = masks[batch_index, :, :, 0] == feature_id
            values[batch_index, feature_index] = float(
                attribution[batch_index, :, :, 0][selected].flatten()[0].item()
            )
    return clean.detach().cpu().numpy().astype(np.float64), values


@dataclass
class RawStore:
    """Append-only Parquet part store with input-identity validation."""

    output_dir: Path
    identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        self.raw_dir = self.output_dir / "raw"
        self.manifest_path = self.output_dir / "run_manifest.json"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.identity_hash = sha256_bytes(canonical_json(self.identity).encode("utf-8"))
        if self.manifest_path.is_file():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest["identity_hash"] != self.identity_hash:
                raise RuntimeError("output belongs to a different config or input artifact set")
            self.status = str(manifest["status"])
            self.manifest_rows = int(manifest["rows"])
        else:
            self._write_manifest("incomplete", parts=0, rows=0)
            self.status = "incomplete"
            self.manifest_rows = 0
        self._refresh()
        if self.status == "complete" and self.row_count != self.manifest_rows:
            raise RuntimeError("complete manifest row count does not match raw parquet parts")

    def _refresh(self) -> None:
        frames = [pd.read_parquet(path) for path in sorted(self.raw_dir.glob("part-*.parquet"))]
        self.frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RAW_COLUMNS)
        if not self.frame.empty and self.frame["item_key"].duplicated().any():
            raise RuntimeError("raw cache contains duplicate item_key rows")
        self.keys = set(self.frame["item_key"].astype(str))
        self.part_count = len(frames)
        self.pending: list[Mapping[str, Any]] = []

    @property
    def row_count(self) -> int:
        return len(self.frame) + len(self.pending)

    def append(self, rows: Sequence[Mapping[str, Any]]) -> int:
        new_rows = [row for row in rows if str(row["item_key"]) not in self.keys]
        if not new_rows:
            return 0
        self.pending.extend(new_rows)
        self.keys.update(str(row["item_key"]) for row in new_rows)
        if len(self.pending) >= 4096:
            self.flush()
        return len(new_rows)

    def flush(self) -> None:
        if not self.pending:
            return
        frame = pd.DataFrame(self.pending, columns=RAW_COLUMNS)
        path = self.raw_dir / f"part-{self.part_count:06d}.parquet"
        frame.to_parquet(path, index=False)
        self.part_count += 1
        self.frame = pd.concat([self.frame, frame], ignore_index=True)
        self.pending = []
        self._write_manifest("incomplete", parts=self.part_count, rows=len(self.frame))

    def _write_manifest(self, status: str, *, parts: int, rows: int) -> None:
        document = {
            "schema_version": "captum-reference-raw-v1",
            "identity_hash": self.identity_hash,
            "identity": self.identity,
            "status": status,
            "parts": parts,
            "rows": rows,
            "updated_at": utc_now(),
        }
        self.manifest_path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    def complete(self, forward_evaluations: int) -> None:
        self.flush()
        self._write_manifest("complete", parts=self.part_count, rows=len(self.frame))
        self.status = "complete"
        path = self.output_dir / "execution_summary.json"
        path.write_text(
            json.dumps(
                {
                    "completed_at": utc_now(),
                    "raw_rows": len(self.frame),
                    "forward_evaluations_this_invocation": forward_evaluations,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def build_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    config_for_hash = {key: value for key, value in config.items() if key != "config_path"}
    artifacts = {
        group: {name: sha256_file(Path(path)) for name, path in config["data"][group].items()}
        for group in ("manifests", "checkpoints")
    }
    artifacts["dataset_stats"] = sha256_file(Path(config["data"]["dataset_stats"]))
    return {"resolved_config": config_for_hash, "artifact_sha256": artifacts}


def item_key(**values: Any) -> str:
    return sha256_bytes(canonical_json(values).encode("utf-8"))


def _row(
    *,
    model: str,
    dataset: str,
    sample_id: str,
    gt_label: int,
    region: str,
    target_region: str,
    is_control: bool,
    control_index: int | None,
    operator: str,
    seed_salt: int,
    clean_margin: float,
    degradation: float,
    source_area: int,
    model_area: int,
) -> dict[str, Any]:
    identity = {
        "model": model,
        "dataset": dataset,
        "sample_id": sample_id,
        "region_key": region,
        "target_region_key": target_region,
        "is_control": is_control,
        "control_index": control_index,
        "perturbation": operator,
        "seed_salt": seed_salt,
    }
    return {
        "item_key": item_key(**identity),
        **identity,
        "gt_label": gt_label,
        "clean_margin": clean_margin,
        "perturbed_margin": clean_margin - degradation,
        "degradation": degradation,
        "source_area": source_area,
        "model_area": model_area,
        "status": "complete",
    }


def _expected_keys(
    model: str,
    dataset: str,
    sample_ids: Sequence[str],
    regions: Sequence[tuple[str, str, bool, int | None]],
    operator: str,
    seed_salt: int,
) -> set[str]:
    return {
        item_key(
            model=model,
            dataset=dataset,
            sample_id=sample_id,
            region_key=region,
            target_region_key=target,
            is_control=is_control,
            control_index=control_index,
            perturbation=operator,
            seed_salt=seed_salt,
        )
        for sample_id in sample_ids
        for region, target, is_control, control_index in regions
    }


def _expected_paired_keys(
    model: str,
    dataset: str,
    sample_ids: Sequence[str],
    regions: Sequence[tuple[str, str, bool, int | None]],
    operator: str,
    seed_salt: int,
) -> set[str]:
    """Return one expected identity for each aligned sample/control region pair."""

    if len(sample_ids) != len(regions):
        raise ValueError("paired sample and region identities must have equal length")
    return {
        item_key(
            model=model,
            dataset=dataset,
            sample_id=sample_id,
            region_key=region,
            target_region_key=target,
            is_control=is_control,
            control_index=control_index,
            perturbation=operator,
            seed_salt=seed_salt,
        )
        for sample_id, (region, target, is_control, control_index) in zip(
            sample_ids, regions, strict=True
        )
    }


def run_audit(
    config: Mapping[str, Any], output_dir: Path, stop_after_items: int | None = None
) -> dict[str, Any]:
    """Execute or resume all configured Captum region audits."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the workspace")
    store = RawStore(output_dir, build_identity(config))
    start_rows = store.row_count
    if store.status == "complete":
        store.complete(0)
        return {
            "status": "complete",
            "raw_rows": store.row_count,
            "new_rows": 0,
            "forward_evaluations": 0,
        }
    total_forward = 0
    channel_mean = json.loads(Path(config["data"]["dataset_stats"]).read_text())["channel_mean"]
    rows_count, cols_count = int(config["regions"]["rows"]), int(config["regions"]["cols"])
    height, width = (int(value) for value in config["preprocessing"]["input_size"])
    model_height, model_width = (int(value) for value in config["preprocessing"]["output_size"])
    source_area = (height // rows_count) * (width // cols_count)
    model_area = (model_height // rows_count) * (model_width // cols_count)
    grid_mask_cpu = grid_feature_mask(height, width, rows_count, cols_count)
    stopped = False

    for run in config["audit_runs"]:
        model_name, dataset_name = run["model"], run["dataset"]
        dataset = ManifestDataset(Path(config["data"]["manifests"][run["manifest"]]))
        loader = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
        )
        wrapped = load_model(config, model_name)
        ablator = FeatureAblation(wrapped)
        operators = (
            tuple(config["perturbations"])
            if run["operators"] == "all"
            else tuple(run["operators"])
        )
        seeds = tuple(config["seed_salts"]) if run["operators"] == "all" else (0,)

        for images_cpu, labels_cpu, sample_ids_raw in loader:
            sample_ids = [str(value) for value in sample_ids_raw]
            labels = labels_cpu.cuda(non_blocking=True)
            images = images_cpu.cuda(non_blocking=True).float()
            target_regions = [
                (region_key(row, col), region_key(row, col), False, None)
                for row in range(rows_count)
                for col in range(cols_count)
            ]
            for operator in operators:
                for seed_salt in seeds:
                    baseline = perturbation_baseline(
                        images,
                        operator,
                        config["perturbations"][operator],
                        channel_mean=channel_mean,
                        global_seed=int(config["global_seed"]),
                        sample_ids=sample_ids,
                        seed_salt=int(seed_salt),
                    )
                    expected = _expected_keys(
                        model_name,
                        dataset_name,
                        sample_ids,
                        target_regions,
                        operator,
                        int(seed_salt),
                    )
                    if not expected.issubset(store.keys):
                        before = wrapped.forward_evaluations
                        clean, values = _attribution_values(
                            ablator,
                            images,
                            labels,
                            baseline,
                            grid_mask_cpu.cuda(),
                            tuple(range(rows_count * cols_count)),
                            int(config["perturbations_per_eval"]),
                        )
                        total_forward += wrapped.forward_evaluations - before
                        rows = [
                            _row(
                                model=model_name,
                                dataset=dataset_name,
                                sample_id=sample_id,
                                gt_label=int(labels_cpu[batch_index]),
                                region=region_key(row_index, col_index),
                                target_region=region_key(row_index, col_index),
                                is_control=False,
                                control_index=None,
                                operator=operator,
                                seed_salt=int(seed_salt),
                                clean_margin=float(clean[batch_index]),
                                degradation=float(values[batch_index, row_index * cols_count + col_index]),
                                source_area=source_area,
                                model_area=model_area,
                            )
                            for batch_index, sample_id in enumerate(sample_ids)
                            for row_index in range(rows_count)
                            for col_index in range(cols_count)
                        ]
                        stopped = _append_with_limit(store, rows, start_rows, stop_after_items)
                        if stopped:
                            break
                    if run["controls"]:
                        for target_row in range(rows_count):
                            for target_col in range(cols_count):
                                target = region_key(target_row, target_col)
                                for control_index in range(
                                    int(config["regions"]["controls_per_region"])
                                ):
                                    control_mask, control_keys = matched_control_masks(
                                        sample_ids=sample_ids,
                                        target_row=target_row,
                                        target_col=target_col,
                                        control_index=control_index,
                                        height=height,
                                        width=width,
                                        rows=rows_count,
                                        cols=cols_count,
                                        global_seed=int(config["global_seed"]),
                                    )
                                    regions = [
                                        (key, target, True, control_index) for key in control_keys
                                    ]
                                    expected = _expected_paired_keys(
                                        model_name,
                                        dataset_name,
                                        sample_ids,
                                        regions,
                                        operator,
                                        int(seed_salt),
                                    )
                                    if expected.issubset(store.keys):
                                        continue
                                    before = wrapped.forward_evaluations
                                    clean, values = _attribution_values(
                                        ablator,
                                        images,
                                        labels,
                                        baseline,
                                        control_mask.cuda(),
                                        (0,),
                                        2,
                                    )
                                    total_forward += wrapped.forward_evaluations - before
                                    rows = [
                                        _row(
                                            model=model_name,
                                            dataset=dataset_name,
                                            sample_id=sample_id,
                                            gt_label=int(labels_cpu[batch_index]),
                                            region=control_keys[batch_index],
                                            target_region=target,
                                            is_control=True,
                                            control_index=control_index,
                                            operator=operator,
                                            seed_salt=int(seed_salt),
                                            clean_margin=float(clean[batch_index]),
                                            degradation=float(values[batch_index, 0]),
                                            source_area=source_area,
                                            model_area=model_area,
                                        )
                                        for batch_index, sample_id in enumerate(sample_ids)
                                    ]
                                    stopped = _append_with_limit(
                                        store, rows, start_rows, stop_after_items
                                    )
                                    if stopped:
                                        break
                                if stopped:
                                    break
                            if stopped:
                                break
                    if stopped:
                        break
                if stopped:
                    break
            if stopped:
                break
        del wrapped, ablator
        torch.cuda.empty_cache()
        if stopped:
            break

    if stopped:
        store.flush()
        return {
            "status": "incomplete",
            "raw_rows": store.row_count,
            "new_rows": store.row_count - start_rows,
            "forward_evaluations": total_forward,
        }
    _run_accuracy(config, output_dir)
    _write_provenance(config, output_dir)
    store.complete(total_forward)
    return {
        "status": "complete",
        "raw_rows": store.row_count,
        "new_rows": store.row_count - start_rows,
        "forward_evaluations": total_forward,
    }


def _append_with_limit(
    store: RawStore,
    rows: Sequence[Mapping[str, Any]],
    start_rows: int,
    stop_after_items: int | None,
) -> bool:
    if stop_after_items is None:
        store.append(rows)
        return False
    remaining = stop_after_items - (store.row_count - start_rows)
    if remaining <= 0:
        return True
    store.append(rows[:remaining])
    return store.row_count - start_rows >= stop_after_items


@torch.no_grad()
def _run_accuracy(config: Mapping[str, Any], output_dir: Path) -> None:
    path = output_dir / "accuracy.json"
    if path.is_file():
        return
    result: dict[str, dict[str, float]] = {}
    for run in config["accuracy_runs"]:
        model_name = run["model"]
        wrapped = load_model(config, model_name)
        result[model_name] = {}
        for dataset_name in run["datasets"]:
            dataset = ManifestDataset(Path(config["data"]["manifests"][dataset_name]))
            loader = DataLoader(
                dataset,
                batch_size=int(config["batch_size"]),
                shuffle=False,
                num_workers=int(config["num_workers"]),
            )
            correct = total = 0
            for images, labels, _sample_ids in loader:
                images, labels = images.cuda().float(), labels.cuda()
                normalized = images.permute(0, 3, 1, 2).div(255.0)
                normalized = F.interpolate(
                    normalized,
                    size=wrapped.output_size,
                    mode="bilinear",
                    align_corners=False,
                )
                logits = wrapped.model((normalized - wrapped.mean) / wrapped.std)
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                total += int(labels.numel())
            result[model_name][dataset_name.removesuffix("_test")] = correct / total
        del wrapped
        torch.cuda.empty_cache()
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _git_sha(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_provenance(config: Mapping[str, Any], output_dir: Path) -> None:
    path = output_dir / "provenance.json"
    document = {
        "schema_version": "captum-reference-provenance-v1",
        "created_at": utc_now(),
        "git_sha": _git_sha(Path(config["repo_root"])),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "captum": captum.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pid": os.getpid(),
        "identity": build_identity(config),
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def load_raw(output_dir: Path) -> pd.DataFrame:
    paths = sorted((output_dir / "raw").glob("part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no raw parquet parts under {output_dir / 'raw'}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if frame["item_key"].duplicated().any():
        raise RuntimeError("duplicate raw item keys")
    return frame.sort_values("item_key").reset_index(drop=True)
