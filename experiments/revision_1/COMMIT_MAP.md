# Commit map: provenance hashes to commit messages

Reports, `deviations.md`, and the README name commits by their message
(subject line) only, because rebases and amends on `revision-1` change
commit hashes. The `*provenance*.json` files are machine records: their
`git.sha` field holds the hash of HEAD when the file was written. This table
gives the commit message for each hash recorded in a tracked provenance file,
so a record can be traced after its hash has left the branch. Find the
current commit with `git log --format='%h %s' --grep='<subject>' --fixed-strings`.
Provenance written from now on also stores the message in `git.subject`.

| Recorded `git.sha` | Author date | Commit message | Provenance files (under `experiments/revision_1/`) |
|---|---|---|---|
| `ac4b3bf` | 2026-09-23 | feat(revision-1): Add Phase 0 baseline freeze tooling and common helpers | `phase0/summary/baseline_manifest.json`, `phase0/summary/environment.json`, `phase0/summary/paper_numbers.provenance.json`, `phase0/summary/recomputed_baselines.provenance.json`, `phase0/summary/target_parity.provenance.json` |
| `6ef5dec` | 2026-09-24 | feat(revision-1): Add A1 grade engine, feature builder, and threshold sweep | `a1_threshold/summary/run_sweep.provenance.json`, `a1_threshold/summary/summarize.provenance.json` |
| `7bdd739` | 2026-09-24 | fix(revision-1): Keep the B1 example figure out of git | `b1_preprocessing/summary/area_tables.provenance.json`, `b1_preprocessing/summary/compare_protocols.provenance.json`, `b1_preprocessing/summary/plot_figures.provenance.json` |
| `be0b318` | 2026-09-24 | feat(revision-1): Add A2 nested parity, control tensor, ablation, and B1-4 scripts | `a2_control_count/summary/nested_parity.provenance.json`, `a2_control_count/summary/run_ablation.provenance.json`, `b1_preprocessing/summary/eff_area_controls.provenance.json`, `phase0/summary/imagenet_images.provenance.json` |
| `0b285e1` | 2026-09-28 | fix(revision-1): Hide minor tick labels on the A2 log-K axes | `a2_control_count/summary/summarize.provenance.json` |
| `d908bbd` | 2026-09-28 | docs(revision-1): Pre-register A3 multi-architecture protocol, configs, and model check | `a3_multi_arch/summary/model_selection.provenance.json` |
| `ed63a16` | 2026-09-28 | feat(revision-1): Add A3 run checks, cross-model comparison, and figures | `a3_multi_arch/summary/compare_models.provenance.json`, `a3_multi_arch/summary/run_checks.provenance.json`, `a3_multi_arch/summary/summarize.provenance.json` |
| `688d19a` | 2026-09-29 | feat(revision-1): Add post-hoc A3 per-operator profile table | `a3_multi_arch/summary/operator_profile.provenance.json` |
| `43edc7d` | 2026-09-29 | docs(revision-1): Pre-register A4 scaling protocol and add sweep, profiling, and fit scripts | `a4_scaling/summary/components.provenance.json` |
| `82eba9f` | 2026-09-30 | fix(revision-1): Label the A4 log-N axes with the measured sample counts | `a4_scaling/summary/fit_scaling.provenance.json`, `a4_scaling/summary/summarize.provenance.json` |
| `f2d3c7b` | 2026-09-30 | feat(revision-1): Add C1 resource summaries and verification against stored runs | `c1_captum/summary/provenance.json` |
| `3076550` | 2026-09-30 | feat(revision-1): Add manual annotation for NTU-RGB+D based B2 experiments | `b2_ntu_semantic/summary/evaluate_alignment.provenance.json`, `b2_ntu_semantic/summary/plot_figures.provenance.json`, `b2_ntu_semantic/summary/ssat_part_scores.provenance.json` |
| `960269b` | 2026-09-30 | feat(revision-1): Add post hoc B2 per-group agreement breakdown | `b2_ntu_semantic/summary/group_breakdown.provenance.json` |
| `8ef7b49` | 2026-10-01 | docs(revision-1): Record the post hoc C1 planner fix and re-measurement plan (D-012) | `c1_captum/summary_plan_cache/provenance.json` |
| `1aa063c` | 2026-10-03 | docs(revision-1): Record the post hoc A4 re-measurement (D-013) and a commit map | `a4_scaling/summary_plan_cache/compare_plan_cache.provenance.json` |
| `78da294` | 2026-10-05 | docs(revision-1): Pre-register C1-std ImageNet comparison and add the Captum ImageNet workflow | `c1_captum/summary_std/provenance.json` |

Several of these hashes are no longer on the branch. On 2026-10-03, while
those commits were still in the local object store, each was checked against
the branch commit with the same message: `git diff --shortstat` between them
was empty (identical trees) and `git patch-id --stable` gave the same id.
Author dates, which record when a step such as a pre-registration was first
committed, survive rewrites; committer dates and hashes do not.

`phase0/summary/baseline_manifest.json` also records `486007a^` as the config
revision of the K=1 parity runs (`K1_CONFIG_REVISION` in `common/runs.py`):
the parent of "chore: Update real data case study configuration file and
report" on `master`, the last commit whose ImageNet configs had
`n_samples: 1`.
