#!/usr/bin/env python3
"""A4 step 0 (implementation plan section 7.4): time the stages of the ``ssat run`` audit loop separately.

On the N=200 reference setting (4x4, V=5, K=3), using the same objects
``ssat run`` builds (``_session_service.build_context``):

* (a) source decode: ``SampleSource.load`` for every sample, in one process;
* (b) chunk preparation (decode, mask resolve, perturbation in the worker
  pool, plus the main-process effective-area mask transform):
  ``iter_prepared_work_chunks`` over every chunk with the configured
  ``num_workers``, no inference; and with ``num_workers=0`` on the first
  ``--serial-chunks`` chunks for the per-item CPU cost;
* (c) inference, per prepared chunk: adapter preprocessing
  (``transform_batch``, CPU, main process) and model forward (GPU,
  synchronized) timed separately, on the first ``--inference-chunks`` chunks.

In ``ssat run`` the workers prepare chunks while the main process runs (c)
and writes the dump, so the loop throughput is bounded by the slower of the
worker pool and the main process. Dump writing (d) is not timed here; it is
part of the gap between the main-process stages and the measured loop
(``deviations.md`` D-008). Writes ``summary/components.json``.

Example:
    python experiments/revision_1/a4_scaling/profile_components.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from experiments.revision_1.a4_scaling.sweep import A4_DIR, A4_RESULTS_DIR, PROFILE, write_config
from experiments.revision_1.common.provenance import write_json_shared, write_provenance

SUMMARY_DIR = A4_DIR / "summary"


def _rate(items: int, seconds: float) -> dict[str, float]:
    return {"items": items, "seconds": seconds, "items_per_s": items / seconds if seconds > 0 else float("nan"),
            "ms_per_item": 1000 * seconds / items if items else float("nan")}


def time_decode(context) -> dict[str, float]:
    """(a) Load every sample once in this process."""

    samples = context.source.list_samples()
    started = time.perf_counter()
    for sample in samples:
        context.source.load(sample.sample_id)
    return _rate(len(samples), time.perf_counter() - started)


def time_preparation(context, chunks, num_workers: int) -> dict[str, float]:
    """(b) Prepare ``chunks`` through the runtime pipeline without inference."""

    from ssat.core.runtime.pipeline import iter_prepared_work_chunks

    items = 0
    started = time.perf_counter()
    for prepared in iter_prepared_work_chunks(chunks, context.builder, context.source, context.adapter,
                                              global_seed=context.resolved.runtime.global_seed, num_workers=num_workers,
                                              fail_fast=False, region_resolver=context.region_resolver):
        items += len(prepared.items) + len(prepared.failures)
    return {**_rate(items, time.perf_counter() - started), "num_workers": num_workers, "chunks": len(chunks)}


def time_inference(context, chunks) -> dict[str, object]:
    """(c) Per prepared chunk: adapter preprocessing and model forward, timed separately."""

    import torch

    from ssat.core.runtime.pipeline import iter_prepared_work_chunks

    adapter = context.adapter
    preprocessor, model, device = adapter._preprocessor, adapter._model, adapter._device
    pre_s = forward_s = mask_s = 0.0
    items = 0
    warmed = False
    for prepared in iter_prepared_work_chunks(chunks, context.builder, context.source, adapter,
                                              global_seed=context.resolved.runtime.global_seed,
                                              num_workers=context.resolved.runtime.num_workers,
                                              fail_fast=False, region_resolver=context.region_resolver):
        if not prepared.items:
            continue
        batch = np.stack([item.array for item in prepared.items])
        if not warmed:
            with torch.inference_mode():
                model(preprocessor.transform_batch(batch).to(device))
            torch.cuda.synchronize()
            warmed = True
        started = time.perf_counter()
        tensor = preprocessor.transform_batch(batch)
        pre_s += time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(tensor.to(device))
        logits.cpu()
        torch.cuda.synchronize()
        forward_s += time.perf_counter() - started
        started = time.perf_counter()
        for item in prepared.items:
            adapter.transform_mask(item.mask)
        mask_s += time.perf_counter() - started
        items += len(prepared.items)
    return {"items": items, "chunks": len(chunks), "preprocess": _rate(items, pre_s), "forward": _rate(items, forward_s),
            "effective_area_mask_transform": _rate(items, mask_s)}


def main(argv: list[str] | None = None) -> int:
    from ssat.application import _session_service
    from ssat.application.application import AuditApplication

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial-chunks", type=int, default=60)
    parser.add_argument("--inference-chunks", type=int, default=150)
    parser.add_argument("--output", type=Path, default=SUMMARY_DIR / "components.json")
    args = parser.parse_args(argv)

    config = write_config(PROFILE, A4_RESULTS_DIR / "configs")
    context = _session_service.build_context(AuditApplication(), config, None)
    chunks = tuple(context.builder.enumerate())
    runtime = context.resolved.runtime
    print(f"{PROFILE.key}: {len(chunks)} chunks, {sum(c.n_items for c in chunks)} items", flush=True)
    result: dict[str, object] = {
        "setting": PROFILE.key, "planned_items": PROFILE.planned_items, "num_workers": runtime.num_workers,
        "variants_per_chunk": runtime.variants_per_chunk, "n_chunks": len(chunks),
    }
    result["decode"] = time_decode(context)
    print(f"decode: {result['decode']}", flush=True)
    result["preparation_pool"] = time_preparation(context, chunks, runtime.num_workers)
    print(f"preparation pool: {result['preparation_pool']}", flush=True)
    result["preparation_serial"] = time_preparation(context, chunks[: args.serial_chunks], 0)
    print(f"preparation serial: {result['preparation_serial']}", flush=True)
    result["inference"] = time_inference(context, chunks[: args.inference_chunks])
    print(f"inference: {result['inference']}", flush=True)
    write_json_shared(args.output, result)
    write_provenance(args.output.parent, inputs={"config": config}, filename=f"{args.output.stem}.provenance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
