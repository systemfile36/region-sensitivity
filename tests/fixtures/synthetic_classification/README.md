# Synthetic classification fixture

This directory is the default output of
`scripts/generate_synthetic_classification_fixture.py`.

The generator creates 18 deterministic 64×64 RGB PNG files, two intentionally
corrupt image files, and `manifest.json`. The generated binary artifacts are
committed only after review; CI consumes the committed files and does not run
the generator.

From the repository root:

```bash
python scripts/generate_synthetic_classification_fixture.py
```

The command refuses to replace generated targets by default. To deliberately
regenerate them after reviewing generator changes:

```bash
python scripts/generate_synthetic_classification_fixture.py --force
```

No external images, fonts, or network access are used. The manifest records
both encoded-file SHA-256 and decoded-pixel SHA-256 values for valid images.
