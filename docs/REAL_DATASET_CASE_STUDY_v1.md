# Phase 3 Real-Dataset Case Study

This document fixes the reproducible protocol for the SoftwareX Phase-3 case study. Raw datasets, checkpoints, and dumps remain local; configs, hashes, scripts, and aggregated summaries are tracked.

## Fixed matrix

| Dataset | Model | Preprocessing | Samples | Regions | Controls |
| --- | --- | --- | ---: | --- | ---: |
| ImageNet-1k validation | `mobilenetv2_050.lamb_in1k` | official / crop-free | 10,000 | 4×4 grid | 1 per target |
| ImageNet-1k validation | `mobilenetv2_100.ra_in1k` | official / crop-free | 10,000 | 4×4 grid | 1 per target |
| NTU60 XSub test | TSM-ResNet50 | MMAction2 val / crop-free | 1,200 | 10 skeleton parts | none |

Every run uses mean fill, Gaussian blur (`sigma=3.0`), and Gaussian noise (`sigma=12.5`, seeds 0/1/2), with global seed `20260820`. The primary metric is `margin_drop`; probability drop, flip rate, region/class/sample aggregates, and stability outputs remain available in the standard stores.

## TSM checkpoint provenance

- Source: `best_acc_top1_epoch_45.pth`
- Source SHA-256: `4391c28945702f48c366168521700a80d3280c03f7e97af8ca21d9b18d80dddb`
- Training config SHA-256: `61f657c82b473029be1359d1192ef796ef39b7b6809470029ca2da1d6a83d12c`
- MMAction2 commit: `a5a167df`
- Epoch/iteration: 45 / 58,725
- Recorded historical top-1/top-5: `0.8712448559670782` / `0.9795781893004115`
- Native tensor-only checkpoint SHA-256: `b7b53db426b6f81cd3ff3f1729dd5a837c3e59018bb76aa34a2e160a7a72371a`

The historical accuracy is provenance, not an equality assertion for the new hash-selected subset. The native implementation uses no MMAction2 runtime dependency. It is based on the TSM architecture and is compatible with the supplied MMAction2 state layout; no MMAction source is copied. MMAction2 is Apache-2.0: [repository](https://github.com/open-mmlab/mmaction2), [TSM model-zoo configuration](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tsm/README.md).

Convert or reproduce the local checkpoint:

```bash
python scripts/model_tools/convert_mmaction_tsm.py \
  data/nturgbd_trained_checkpoint/best_acc_top1_epoch_45.pth \
  data/nturgbd_trained_checkpoint/native_tsm_r50_xsub_epoch45.pt \
  --mmaction-commit a5a167df \
  --training-config data/nturgbd_trained_checkpoint/ntu60_tsm_xsub_train.py
```

The converter first inspects pickle globals, rejects anything outside its narrow allowlist, loads with `weights_only=True`, maps all 320 tensors, verifies the 60×2048 head, and strict-loads the native model before writing output.

For the independent MMAction2 parity gate, export raw (pre-softmax) logits for the same three manifest IDs from commit `a5a167df`, then export the native side and compare them:

```bash
python experiments/real_dataset_case_study/parity/export_native_logits.py \
  experiments/real_dataset_case_study/configs/ntu60_tsm_exact.yaml \
  outputs/parity/native_logits.npz --count 3
python experiments/real_dataset_case_study/parity/compare_logits.py \
  outputs/parity/mmaction_raw_logits.npz outputs/parity/native_logits.npz
```

Both NPZ files contain `sample_ids` and a `(3, 60)` `logits` array. The comparison requires all top-1 labels to match and NumPy `allclose` with absolute and relative tolerance `1e-4`. MMAction's probability-valued `pred_score` is not a valid reference; export the head's raw tensor output before softmax.

The native `(3, 60)` export has been smoke-tested and contains only finite values. The MMAction2-side raw-logit export remains pending because MMAction2 is intentionally absent from the SSAT runtime environment; do not mark numerical parity complete until that independent reference file passes the comparison command.

## Dataset preparation

The NTU subset has been prepared locally as 60 classes × 20 clips. It uses the supplied `data/annotations/ntu60_xsub_test.txt`, SHA-256 ranking within each class, and the exact eight segment centers for both RGB decode and skeleton boxes:

- XSub annotation SHA-256: `254785af14ee863472ab0405ce3487ef4e5e2472578fcd9519cc6b86f2e6fd40`
- Selected manifest SHA-256: `fb087cbfeb26c9c203b224cd845f988956532ec0f1c227ca2f274e834fb830b0`
- Skeleton bbox SHA-256: `bd81e601de83d74e1ab5658a600711b63a5d6497b7575c6ae65de5d564e8c83f`

```bash
python scripts/dataset_prep/ntu_rgb_d.py \
  --rgb-root data/nturgb+d_rgb \
  --skeleton-root data/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons \
  --annotation-file data/annotations/ntu60_xsub_test.txt \
  --samples-per-class 20 --seed 20260820 \
  --num-frames 8 --sampling segment_center \
  --out data/phase3/ntu60_xsub_test_20
```

Prepare ImageNet after the Kaggle ILSVRC download completes:

```bash
python scripts/dataset_prep/imagenet_val.py \
  --val-root data/imagenet/ILSVRC/Data/CLS-LOC/val \
  --solution-csv data/imagenet/LOC_val_solution.csv \
  --synset-mapping data/imagenet/LOC_synset_mapping.txt \
  --samples-per-class 10 --seed 20260820 \
  --output-annotation data/phase3/imagenet/val_10_per_class.txt
```

If the archive expands under an extra directory, change the three ImageNet paths and the `source.root` path in the four ImageNet configs. The preparation script validates 1,000 unique synsets, every referenced image, and exactly 10 selected samples per class.

## Execution and acceptance

```bash
python experiments/real_dataset_case_study/run_matrix.py --dry-run
python experiments/real_dataset_case_study/run_matrix.py
python experiments/real_dataset_case_study/summarize.py
```

Official preprocessing is the primary result. Its center crop can legitimately fail the 5% effective-area advisory. ImageNet crop-free runs must pass that advisory and serve as the preprocessing-confound comparison. On NTU, crop-free preprocessing removes the systematic crop effect but the preflight still found 4/30 failures (maximum 14.67%) because very small hand masks quantize to only a few pixels when reduced from 1080p to 224p. This residual advisory is retained rather than tuning part boxes to the three preflight samples. Exact clean-accuracy gates are 50% (`mobilenetv2_050`), 60% (`mobilenetv2_100`), and 80% (NTU TSM); crop-free accuracy is recorded without a hard gate.

A complete run requires finite full logits and zero clean/perturbed item failures. Commit generated files under `experiments/real_dataset_case_study/summary/`; do not commit `data/`, checkpoints, or `outputs/`. ImageNet result tables remain intentionally absent while the dataset download and six full audits are incomplete.
