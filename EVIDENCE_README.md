# Evidence-Ready Repository Guide

This repository is intentionally organized so a reviewer can separate four questions:

1. **What source code exists?** — fixed by source hash and file inventory.
2. **What does the source implement?** — documented by architecture, function inventory, and code review.
3. **What actually ran in a specific environment?** — must be shown by a completed run manifest, raw outputs, screenshots, and environment freeze.
4. **What impact occurred outside this repository?** — must be established by independent or first-hand external evidence, not by GitHub alone.

For a formal review, cite the exact public release tag and commit SHA rather than a moving `main` branch.
