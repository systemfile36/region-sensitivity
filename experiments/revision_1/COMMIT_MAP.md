# Commit map: hashes recorded before history rewrites

The `revision-1` branch was rebased or amended several times while the
revision ran (2026-09-23 to 2026-10-01). Reports, `deviations.md`, and the
`*provenance*.json` files record the commit hash at the time of each step,
so some of them name commits that are no longer on the branch. Every one of
them has a commit on the branch with the same message, the same patch, and
an identical tree; only the hash and the committer date changed. Author
dates, which record when a step (for example a pre-registration) was first
committed, are unchanged.

| Recorded hash | Commit on the branch | Subject |
|---|---|---|
| `ac4b3bf` | `d1ce815` | feat(revision-1): Add Phase 0 baseline freeze tooling and common helpers |
| `cb7ef30` | `1c8a322` | docs(revision-1): Pre-register A1 threshold sensitivity protocol |
| `6ef5dec` | `571738b` | feat(revision-1): Add A1 grade engine, feature builder, and threshold sweep |
| `fab776b` | `fdde68b` | docs(revision-1): Pre-register B1 preprocessing confound protocol |
| `7bdd739` | `28c1254` | fix(revision-1): Keep the B1 example figure out of git |
| `0b285e1` | `dd77385` | fix(revision-1): Hide minor tick labels on the A2 log-K axes |
| `d908bbd` | `7cc3631` | docs(revision-1): Pre-register A3 multi-architecture protocol, configs, and model check |
| `4790ec9` | `6c79715` | docs(revision-1): Record A3 model selection check |
| `ed63a16` | `a47ef0e` | feat(revision-1): Add A3 run checks, cross-model comparison, and figures |
| `688d19a` | `abee116` | feat(revision-1): Add post-hoc A3 per-operator profile table |
| `43edc7d` | `2792c33` | docs(revision-1): Pre-register A4 scaling protocol and add sweep, profiling, and fit scripts |
| `82eba9f` | `a310768` | fix(revision-1): Label the A4 log-N axes with the measured sample counts |
| `77d437c` | `5c5e334` | docs(revision-1): Pre-register C1 resource comparison and add measurement script |
| `f2d3c7b` | `456ccae` | feat(revision-1): Add C1 resource summaries and verification against stored runs |
| `8c29dd7` | `cdacf0d` | docs(revision-1): Pre-register B2 NTU semantic validation and add annotation sheets |
| `3076550` | `c5884cc` | feat(revision-1): Add manual annotation for NTU-RGB+D based B2 experiments |
| `960269b` | `33d79b2` | feat(revision-1): Add post hoc B2 per-group agreement breakdown |
| `fb7168b` | `d0fcca8` | fix(plan): Recompute a sample's work items once per sample in materialize |
| `8ef7b49` | `1d0b19a` | docs(revision-1): Record the post hoc C1 planner fix and re-measurement plan (D-012) |

Checked on 2026-10-03 while the recorded commits were still in the local
object store. For each pair, `git diff --shortstat <recorded> <branch>` was
empty and `git show <commit> | git patch-id --stable` gave the same id.

Recorded hashes not listed here are on the branch. A rewrite of commits
that are already recorded somewhere adds rows to this table.
