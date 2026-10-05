"""Independent Captum implementation of the ImageNet region audit (C1-std).

Adapted from ``experiments/reference_comparison/captum_baseline/workflow.py``
(the synthetic-shortcut reference). This module depends only on
general-purpose scientific Python, PyTorch, timm, and Captum; it imports
neither SSAT nor the synthetic reference.
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
from typing import Any, Mapping, Sequence

import captum
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import yaml
from captum.attr import FeatureAblation
from PIL import Image
from timm.data import resolve_model_data_config
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
    "clean_correct",
    "perturbed_margin",
    "degradation",
    "source_area",
    "model_area",
    "status",
)
OPERATORS = ("mean_fill", "blur", "gaussian_noise")


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
    if not isinstance(config, dict) or config.get("schema_version") != "captum-reference-imagenet-v1":
        raise ValueError("expected schema_version='captum-reference-imagenet-v1'")
    root = (repo_root or config_path.parents[4]).resolve()
    resolved = json.loads(json.dumps(config))
    resolved["repo_root"] = str(root)
    resolved["config_path"] = str(config_path)
    for key in ("image_root", "annotation"):
        resolved["data"][key] = str((root / resolved["data"][key]).resolve())
    _validate_config(resolved)
    return resolved


def _validate_config(config: Mapping[str, Any]) -> None:
    if config["device"] != "cuda":
        raise ValueError("the full reference experiment requires device: cuda")
    if config["preprocessing"]["geometry"] != "squash":
        raise ValueError("only the crop-free squash geometry is implemented")
    if int(config["regions"]["controls_per_region"]) < 2:
        raise ValueError("z against controls needs at least 2 controls per region")
    unknown = set(config["perturbations"]) - set(OPERATORS)
    if unknown:
        raise ValueError(f"unsupported perturbations: {sorted(unknown)}")
    if not Path(config["data"]["image_root"]).is_dir():
        raise FileNotFoundError(config["data"]["image_root"])
    if not Path(config["data"]["annotation"]).is_file():
        raise FileNotFoundError(config["data"]["annotation"])


def perturbation_variants(config: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    """Return every ``(operator, seed_salt)`` pair in config order."""

    return tuple(
        (operator, int(seed_salt))
        for operator, spec in config["perturbations"].items()
        for seed_salt in spec["seed_salts"]
    )


def read_annotation(path: Path) -> list[tuple[str, int]]:
    """Parse ``<relative_path> <label>`` lines."""

    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            relative_path, label = line.split()
            entries.append((relative_path, int(label)))
    return entries


def perturbation_baseline(
    array: np.ndarray,
    operator: str,
    params: Mapping[str, Any],
    *,
    global_seed: int,
    sample_id: str,
    seed_salt: int,
) -> np.ndarray:
    """Build one full-frame uint8 candidate; Captum composites feature groups from it."""

    if operator == "mean_fill":
        values = np.rint(np.asarray(params["value"], dtype=np.float64)).astype(np.uint8)
        return np.broadcast_to(values, array.shape).copy()
    if operator == "blur":
        return cv2.GaussianBlur(
            array,
            ksize=(0, 0),
            sigmaX=float(params["sigma"]),
            sigmaY=float(params["sigma"]),
            borderType=cv2.BORDER_REFLECT_101,
        )
    if operator == "gaussian_noise":
        rng = np.random.default_rng(stable_seed(global_seed, sample_id, operator, seed_salt))
        noise = rng.normal(0.0, float(params["sigma"]), size=array.shape)
        return np.clip(np.rint(array.astype(np.float64) + noise), 0, 255).astype(np.uint8)
    raise ValueError(f"unknown perturbation: {operator}")


class AnnotationDataset(Dataset):
    """ImageNet file-list dataset that also builds every perturbation candidate.

    Images keep their source size, so the loader returns lists instead of
    stacked batches. Candidates are built in the loader workers.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.root = Path(config["data"]["image_root"])
        self.samples = tuple(read_annotation(Path(config["data"]["annotation"])))
        self.variants = perturbation_variants(config)
        self.params = {name: spec["params"] for name, spec in config["perturbations"].items()}
        self.global_seed = int(config["global_seed"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, str, dict[tuple[str, int], np.ndarray]]:
        sample_id, label = self.samples[index]
        with Image.open(self.root / sample_id) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        candidates = {
            (operator, seed_salt): perturbation_baseline(
                array,
                operator,
                self.params[operator],
                global_seed=self.global_seed,
                sample_id=sample_id,
                seed_salt=seed_salt,
            )
            for operator, seed_salt in self.variants
        }
        return array, label, sample_id, candidates


def _collate_list(batch: list[Any]) -> list[Any]:
    return batch


class RawMarginModel(nn.Module):
    """Wrap a classifier so Captum sees source-space pixels and returns margins.

    The squash resize runs on the GPU. Its output is rounded to integer
    pixel values, as a resize of a uint8 image would be.
    """

    def __init__(
        self,
        model: nn.Module,
        output_size: Sequence[int],
        interpolation: str,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        super().__init__()
        self.model = model
        self.output_size = tuple(int(value) for value in output_size)
        self.interpolation = interpolation
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.forward_evaluations = 0

    def logits(self, raw_images: torch.Tensor) -> torch.Tensor:
        self.forward_evaluations += int(raw_images.shape[0])
        images = raw_images.permute(0, 3, 1, 2).float()
        images = F.interpolate(
            images, size=self.output_size, mode=self.interpolation, align_corners=False, antialias=True
        )
        images = images.round().clamp(0.0, 255.0).div(255.0)
        return self.model((images - self.mean) / self.std)

    def forward(self, raw_images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return margins(self.logits(raw_images), labels)


def margins(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Ground-truth logit minus the largest other logit."""

    labels = labels.long().view(-1, 1)
    gt = logits.gather(1, labels).squeeze(1)
    masked = logits.scatter(1, labels, float("-inf"))
    return gt - masked.max(dim=1).values


def load_model(config: Mapping[str, Any]) -> RawMarginModel:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full Captum reference experiment")
    spec = config["model"]
    classifier = timm.create_model(spec["timm_name"], pretrained=bool(spec["pretrained"]))
    preprocessing = config["preprocessing"]
    data_config = resolve_model_data_config(classifier)
    if (
        tuple(data_config["input_size"]) != (3, *preprocessing["output_size"])
        or data_config["interpolation"] != preprocessing["interpolation"]
        or not np.allclose(data_config["mean"], preprocessing["mean"])
        or not np.allclose(data_config["std"], preprocessing["std"])
    ):
        raise ValueError(f"preprocessing does not match the timm data config: {data_config}")
    classifier.eval().cuda()
    return RawMarginModel(
        classifier,
        preprocessing["output_size"],
        preprocessing["interpolation"],
        preprocessing["mean"],
        preprocessing["std"],
    ).eval().cuda()


def cell_bounds(height: int, width: int, rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    """Return ``(row_start, row_end, col_start, col_end)`` per cell, row-major, with integer boundaries."""

    return [
        (row * height // rows, (row + 1) * height // rows, col * width // cols, (col + 1) * width // cols)
        for row in range(rows)
        for col in range(cols)
    ]


def grid_feature_mask(height: int, width: int, rows: int, cols: int) -> torch.Tensor:
    """Return a channel-broadcastable integer mask with one ID per grid cell."""

    mask = torch.empty((1, height, width, 1), dtype=torch.long)
    for index, (row_start, row_end, col_start, col_end) in enumerate(cell_bounds(height, width, rows, cols)):
        mask[:, row_start:row_end, col_start:col_end] = index
    return mask


def region_key(row: int, col: int) -> str:
    return f"grid::grid/r{row}/c{col}"


def model_space_extent(start: int, end: int, source: int, output: int) -> int:
    """Number of output rows (or columns) whose nearest-neighbour source index lies in ``[start, end)``."""

    index = np.floor(np.arange(output) * (source / output)).astype(np.int64)
    return int(((index >= start) & (index < end)).sum())


def matched_controls(
    *,
    sample_id: str,
    bounds: Sequence[tuple[int, int, int, int]],
    cols: int,
    controls_per_region: int,
    height: int,
    width: int,
    global_seed: int,
) -> list[tuple[str, int, int, int, int, int]]:
    """Rigidly translate every target cell to deterministic random positions.

    Returns:
        ``(target_key, control_index, row_offset, col_offset, cell_h, cell_w)``
        for every target cell and control index.
    """

    controls = []
    for index, (row_start, row_end, col_start, col_end) in enumerate(bounds):
        target = region_key(*divmod(index, cols))
        cell_h, cell_w = row_end - row_start, col_end - col_start
        for control_index in range(controls_per_region):
            rng = np.random.default_rng(
                stable_seed(global_seed, sample_id, target, control_index, "matched-control")
            )
            row_offset = int(rng.integers(0, height - cell_h + 1))
            col_offset = int(rng.integers(0, width - cell_w + 1))
            controls.append((target, control_index, row_offset, col_offset, cell_h, cell_w))
    return controls


def _target_values(
    ablator: FeatureAblation,
    image: torch.Tensor,
    labels: torch.Tensor,
    baseline: torch.Tensor,
    feature_mask: torch.Tensor,
    bounds: Sequence[tuple[int, int, int, int]],
    perturbations_per_eval: int,
) -> np.ndarray:
    """Return one ablation value per grid cell, read at each cell's first pixel."""

    attribution = ablator.attribute(
        image,
        baselines=baseline,
        additional_forward_args=(labels,),
        feature_mask=feature_mask,
        perturbations_per_eval=perturbations_per_eval,
    )
    rows = torch.tensor([bound[0] for bound in bounds], device=image.device)
    cols = torch.tensor([bound[2] for bound in bounds], device=image.device)
    return attribution[0, rows, cols, 0].double().cpu().numpy()


def _control_values(
    ablator: FeatureAblation,
    image: torch.Tensor,
    labels: torch.Tensor,
    baseline: torch.Tensor,
    controls: Sequence[tuple[str, int, int, int, int, int]],
    pixel_budget: int,
) -> np.ndarray:
    """Ablate each control rectangle against the full candidate frame.

    One sample's controls are stacked along the batch axis with a per-example
    two-group mask (control = 0, rest = 1). FeatureAblation ablates both
    groups; only group 0 is read.
    """

    height, width = image.shape[1:3]
    chunk = max(1, min(len(controls), pixel_budget // (height * width)))
    values = np.empty(len(controls), dtype=np.float64)
    for start in range(0, len(controls), chunk):
        part = controls[start : start + chunk]
        mask = torch.ones((len(part), height, width, 1), dtype=torch.long, device=image.device)
        for index, (_, _, row_offset, col_offset, cell_h, cell_w) in enumerate(part):
            mask[index, row_offset : row_offset + cell_h, col_offset : col_offset + cell_w] = 0
        attribution = ablator.attribute(
            image.expand(len(part), -1, -1, -1),
            baselines=baseline,
            additional_forward_args=(labels.expand(len(part)),),
            feature_mask=mask,
            perturbations_per_eval=2,
        )
        rows = torch.tensor([control[2] for control in part], device=image.device)
        cols = torch.tensor([control[3] for control in part], device=image.device)
        batch = torch.arange(len(part), device=image.device)
        values[start : start + len(part)] = attribution[batch, rows, cols, 0].double().cpu().numpy()
    return values


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
        if len(self.pending) >= 16384:
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
    artifacts = {"annotation": sha256_file(Path(config["data"]["annotation"]))}
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
    clean_correct: bool,
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
        "clean_correct": clean_correct,
        "perturbed_margin": clean_margin - degradation,
        "degradation": degradation,
        "source_area": source_area,
        "model_area": model_area,
        "status": "complete",
    }


def _expected_keys(
    model: str,
    dataset: str,
    sample_id: str,
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
        for region, target, is_control, control_index in regions
    }


def control_key(target: str, control_index: int, row_offset: int, col_offset: int) -> str:
    return f"control:{target}:{control_index}@{row_offset},{col_offset}"


def run_audit(
    config: Mapping[str, Any], output_dir: Path, stop_after_items: int | None = None
) -> dict[str, Any]:
    """Execute or resume the configured Captum region audit."""

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
    model_name, dataset_name = config["model"]["timm_name"], config["data"]["dataset_name"]
    rows_count, cols_count = int(config["regions"]["rows"]), int(config["regions"]["cols"])
    model_height, model_width = (int(value) for value in config["preprocessing"]["output_size"])
    controls_per_region = int(config["regions"]["controls_per_region"])
    global_seed = int(config["global_seed"])
    variants = perturbation_variants(config)
    wrapped = load_model(config)
    ablator = FeatureAblation(wrapped)
    loader = DataLoader(
        AnnotationDataset(config),
        batch_size=int(config["loader_batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        collate_fn=_collate_list,
    )
    total_forward = 0
    stopped = False

    with torch.no_grad():
        for batch in loader:
            for array, label, sample_id, candidates in batch:
                height, width = array.shape[:2]
                bounds = cell_bounds(height, width, rows_count, cols_count)
                controls = matched_controls(
                    sample_id=sample_id,
                    bounds=bounds,
                    cols=cols_count,
                    controls_per_region=controls_per_region,
                    height=height,
                    width=width,
                    global_seed=global_seed,
                )
                target_regions = [
                    (region_key(*divmod(index, cols_count)),) * 2 + (False, None) for index in range(len(bounds))
                ]
                control_regions = [
                    (control_key(target, index, row, col), target, True, index)
                    for target, index, row, col, _, _ in controls
                ]
                pending = [
                    (operator, seed_salt, kind)
                    for operator, seed_salt in variants
                    for kind, regions in (("target", target_regions), ("control", control_regions))
                    if not _expected_keys(
                        model_name, dataset_name, sample_id, regions, operator, seed_salt
                    ).issubset(store.keys)
                ]
                if not pending:
                    continue
                image = torch.from_numpy(array).cuda().float().unsqueeze(0)
                labels = torch.tensor([label], device="cuda")
                before = wrapped.forward_evaluations
                logits = wrapped.logits(image)
                clean_margin = float(margins(logits, labels)[0])
                clean_correct = bool(int(logits[0].argmax()) == label)
                target_area = [
                    (
                        (row_end - row_start) * (col_end - col_start),
                        model_space_extent(row_start, row_end, height, model_height)
                        * model_space_extent(col_start, col_end, width, model_width),
                    )
                    for row_start, row_end, col_start, col_end in bounds
                ]
                control_area = [
                    (
                        cell_h * cell_w,
                        model_space_extent(row, row + cell_h, height, model_height)
                        * model_space_extent(col, col + cell_w, width, model_width),
                    )
                    for _, _, row, col, cell_h, cell_w in controls
                ]
                feature_mask = grid_feature_mask(height, width, rows_count, cols_count).cuda()
                for operator, seed_salt, kind in pending:
                    baseline = torch.from_numpy(candidates[(operator, seed_salt)]).cuda().float().unsqueeze(0)
                    if kind == "target":
                        values = _target_values(
                            ablator, image, labels, baseline, feature_mask, bounds,
                            int(config["perturbations_per_eval"]),
                        )
                        regions, areas = target_regions, target_area
                    else:
                        values = _control_values(
                            ablator, image, labels, baseline, controls, int(config["control_pixel_budget"])
                        )
                        regions, areas = control_regions, control_area
                    rows = [
                        _row(
                            model=model_name,
                            dataset=dataset_name,
                            sample_id=sample_id,
                            gt_label=int(label),
                            region=region,
                            target_region=target,
                            is_control=is_control,
                            control_index=control_index,
                            operator=operator,
                            seed_salt=seed_salt,
                            clean_margin=clean_margin,
                            clean_correct=clean_correct,
                            degradation=float(value),
                            source_area=source_area,
                            model_area=model_area,
                        )
                        for (region, target, is_control, control_index), value, (source_area, model_area) in zip(
                            regions, values, areas, strict=True
                        )
                    ]
                    stopped = _append_with_limit(store, rows, start_rows, stop_after_items)
                    if stopped:
                        break
                total_forward += wrapped.forward_evaluations - before
                if stopped:
                    break
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
    store.flush()
    _write_accuracy(store.frame, output_dir)
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


def _write_accuracy(raw: pd.DataFrame, output_dir: Path) -> None:
    """Clean top-1 accuracy over the audited samples."""

    clean = raw.drop_duplicates("sample_id")
    document = {
        "samples": int(len(clean)),
        "correct": int(clean["clean_correct"].sum()),
        "top1": float(clean["clean_correct"].mean()),
    }
    (output_dir / "accuracy.json").write_text(json.dumps(document, indent=2), encoding="utf-8")


def _git_sha(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
            cwd=repo_root, text=True, stderr=subprocess.DEVNULL,
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
        "timm": timm.__version__,
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
