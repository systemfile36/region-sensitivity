# Contributing to SSAT

Thanks for your interest in contributing. This document covers the basics:
how to file issues, how to submit changes, and the conventions the codebase
follows.

## Issues

Before opening an issue, please check whether one already covers it. When
filing a new issue, include:

- What you expected to happen and what actually happened.
- A minimal config/command that reproduces the problem, if applicable.
- Your environment (Python version, OS, GPU/CPU).

## Pull requests

1. Fork the repository and create a branch off `master`.
2. Make your change, following the code style below.
3. Add or update tests for the behavior you changed.
4. Run the full test suite (see "Running tests" below) and make sure it
   passes.
5. Open a PR against `master` with a clear description of what changed and
   why.

## Code style

- **Docstrings**: Google-style, written in English, and kept concise rather
  than exhaustive — document parameters, return values, and raised errors
  when they aren't obvious from the signature, but don't restate what the
  code already says.
- **Comments**: terse, English-only, explaining *why* a piece of code exists
  or a particular approach was chosen, not restating *what* the next line
  does. Comments should stand on their own — avoid references to external
  design notes, issue numbers, or planning documents that a reader of the
  shipped code won't have access to.
- Match the surrounding module's existing style (naming, error handling,
  logging) rather than introducing a new convention.

## Running tests

Tests must be run inside the project's Docker Compose workspace, not
directly on the host — the workspace container has the CUDA-enabled PyTorch
base image and system dependencies (e.g. `libgl1-mesa-glx`) the test suite
and its fixtures expect.

```bash
docker compose up -d
docker compose exec region-sensitivity-workspace bash -lc 'pytest -q'
```

Equivalently, open the repository in VS Code and reopen it in the dev
container (`.devcontainer/devcontainer.json`), then run `pytest -q` in the
integrated terminal.

CI (`.github/workflows/ci.yml`) does not use this container — it installs a
CPU-only `torch`/`torchvision` directly on the runner, so contributors
without local GPU/Docker access can still rely on CI to validate a PR. The
Docker Compose workspace remains the supported way to run tests locally.

## Reporting security issues

If you find a security issue, please report it privately rather than
opening a public issue — see the repository's contact information on
GitHub.
