#!/usr/bin/env python3
"""B2 annotation files: NTU RGB+D 60 class list, blank annotator sheets, and their validation.

``write`` creates ``annotation/classes.csv`` and a blank
``annotation/annotator_<ID>.csv`` per annotator id. An existing annotator
sheet is never overwritten. ``check`` validates the filled sheets and
reports progress; ``--require-committed`` additionally requires every sheet
to be committed unchanged at HEAD (the blind condition, protocol.md).

Class names are the official NTU RGB+D 60 action names (Shahroudy et al.,
CVPR 2016; https://github.com/shahroudy/NTURGB-D, A1-A60), label id =
action number - 1.

Example:
    python experiments/revision_1/b2_ntu_semantic/annotations.py write
    python experiments/revision_1/b2_ntu_semantic/annotations.py check --annotators A B
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from experiments.revision_1.common.runs import REPO_ROOT

B2_DIR = Path(__file__).resolve().parent
ANNOTATION_DIR = B2_DIR / "annotation"
CLASSES_CSV = ANNOTATION_DIR / "classes.csv"
GROUPS = ("head", "torso", "arms", "hands", "legs")
RATINGS = (0, 1, 2)
TWO_PERSON_FIRST_ACTION = 50
SOURCE = "NTU RGB+D 60 official action list (Shahroudy et al., CVPR 2016; github.com/shahroudy/NTURGB-D)"
NTU60_ACTIONS = (
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop", "pickup", "throw",
    "sitting down", "standing up (from sitting position)", "clapping", "reading", "writing", "tear up paper",
    "wear jacket", "take off jacket", "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses",
    "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving", "kicking something",
    "reach into pocket", "hopping (one foot jumping)", "jump up", "make a phone call/answer phone",
    "playing with phone/tablet", "typing on a keyboard", "pointing to something with finger",
    "taking a selfie", "check time (from watch)", "rub two hands together", "nod head/bow", "shake head",
    "wipe face", "salute", "put the palms together", "cross hands in front (say stop)", "sneeze/cough",
    "staggering", "falling", "touch head (headache)", "touch chest (stomachache/heart pain)",
    "touch back (backache)", "touch neck (neckache)", "nausea or vomiting condition",
    "use a fan (with hand or paper)/feeling warm", "punching/slapping other person",
    "kicking other person", "pushing other person", "pat on back of other person",
    "point finger at the other person", "hugging other person", "giving something to other person",
    "touch other person's pocket", "handshaking", "walking towards each other",
    "walking apart from each other",
)
SHEET_COLUMNS = ("label_id", "action_id", "action_name", "two_person", *GROUPS, "note")


def class_rows() -> list[dict[str, object]]:
    """One row per class: label id, action id (A001), official name, two-person flag."""

    return [{"label_id": index, "action_id": f"A{index + 1:03d}", "action_name": name,
             "two_person": int(index + 1 >= TWO_PERSON_FIRST_ACTION)} for index, name in enumerate(NTU60_ACTIONS)]


def sheet_path(annotator: str, directory: Path = ANNOTATION_DIR) -> Path:
    return directory / f"annotator_{annotator}.csv"


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    # utf-8-sig so spreadsheet programs open notes in any language correctly.
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_files(annotators: Sequence[str], directory: Path = ANNOTATION_DIR) -> list[Path]:
    """Write ``classes.csv`` and blank sheets for ``annotators`` that do not exist yet; return written paths."""

    directory.mkdir(parents=True, exist_ok=True)
    rows = class_rows()
    _write_csv(directory / CLASSES_CSV.name, ("label_id", "action_id", "action_name", "two_person"), rows)
    written = [directory / CLASSES_CSV.name]
    for annotator in annotators:
        path = sheet_path(annotator, directory)
        if path.exists():
            continue
        _write_csv(path, SHEET_COLUMNS, [{**row, **{group: "" for group in GROUPS}, "note": ""} for row in rows])
        written.append(path)
    return written


def read_sheet(path: Path) -> pd.DataFrame:
    """Read an annotator sheet as strings (blank cells stay empty)."""

    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def validate_sheet(frame: pd.DataFrame) -> tuple[list[str], int]:
    """Return ``(errors, n_complete_rows)``; errors cover structure and non-blank invalid ratings."""

    errors: list[str] = []
    missing = [column for column in SHEET_COLUMNS if column not in frame.columns]
    if missing:
        return [f"missing columns: {missing}"], 0
    expected = pd.DataFrame(class_rows()).astype(str)
    if len(frame) != len(expected) or not (frame[["label_id", "action_id", "action_name"]].to_numpy()
                                           == expected[["label_id", "action_id", "action_name"]].to_numpy()).all():
        errors.append("label_id / action_id / action_name rows differ from classes.csv (do not reorder or edit them)")
    complete = 0
    for _, row in frame.iterrows():
        values = [row[group].strip() for group in GROUPS]
        bad = [f"{group}={value!r}" for group, value in zip(GROUPS, values) if value and value not in {"0", "1", "2"}]
        if bad:
            errors.append(f"label {row['label_id']}: invalid rating(s) {', '.join(bad)} (use 0, 1, or 2)")
        complete += all(values) and not bad
    return errors, complete


def load_ratings(path: Path) -> pd.DataFrame:
    """Validated, complete ratings as an integer frame indexed by label id with the ``GROUPS`` columns."""

    frame = read_sheet(path)
    errors, complete = validate_sheet(frame)
    if errors or complete != len(NTU60_ACTIONS):
        raise ValueError(f"{path}: incomplete or invalid ({complete}/{len(NTU60_ACTIONS)} complete rows; {errors[:3]})")
    ratings = frame[["label_id", *GROUPS]].astype(int).set_index("label_id")
    return ratings


def _normalized(content: bytes) -> bytes:
    return content.replace(b"\r\n", b"\n")


def committed_unchanged(path: Path) -> bool:
    """True when ``path`` is committed at HEAD and the working file equals that blob.

    Line endings are normalized on both sides, since git may store a sheet
    saved with CRLF (for example by a spreadsheet program) as LF.
    """

    try:
        relative = path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:  # outside the repository
        return False
    try:
        blob = subprocess.run(["git", "-c", f"safe.directory={REPO_ROOT}", "-C", str(REPO_ROOT), "show", f"HEAD:{relative}"],
                              check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return False
    return _normalized(blob) == _normalized(path.read_bytes())


def require_blind_annotations(annotators: Sequence[str], directory: Path = ANNOTATION_DIR) -> dict[str, pd.DataFrame]:
    """Load every sheet, refusing unless each is complete and committed unchanged at HEAD."""

    ratings = {}
    for annotator in annotators:
        path = sheet_path(annotator, directory)
        ratings[annotator] = load_ratings(path)
        if not committed_unchanged(path):
            raise PermissionError(f"{path} must be committed unchanged before any SSAT score is computed (protocol.md)")
    return ratings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("write", "check"))
    parser.add_argument("--annotators", nargs="+", default=["A", "B"])
    parser.add_argument("--require-committed", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "write":
        for path in write_files(args.annotators):
            print(f"wrote {path}")
        return 0
    status = 0
    for annotator in args.annotators:
        path = sheet_path(annotator)
        if not path.is_file():
            print(f"{annotator}: {path} not found")
            status = 1
            continue
        errors, complete = validate_sheet(read_sheet(path))
        committed = committed_unchanged(path)
        print(f"{annotator}: {complete}/{len(NTU60_ACTIONS)} rows complete, committed unchanged: {committed}")
        for error in errors:
            print(f"  - {error}")
        if errors or complete != len(NTU60_ACTIONS) or (args.require_committed and not committed):
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
