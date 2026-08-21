# Independent Review Protocol

A strong independent review should be reproducible and tied to the exact source version evaluated.

## Reviewer should record

- name, title, employer/affiliation, and relevant credentials;
- whether the reviewer had any prior relationship with the repository author;
- repository URL;
- release tag and Git commit SHA;
- `app.py` SHA-256;
- review date;
- hardware/OS/Python environment;
- exact installation and run commands;
- dataset used and whether it is synthetic/public/authorized;
- model/configuration selections;
- which tests and application functions were personally executed;
- raw results reproduced;
- any errors or limitations encountered;
- independent technical conclusions limited to what the review supports.

## Recommended reviewer sequence

1. Clone a fresh copy rather than accepting a pre-run screenshot alone.
2. Verify source hash.
3. Install dependencies.
4. Run syntax verification.
5. Launch Streamlit.
6. Execute a documented synthetic/public-data run.
7. Export outputs.
8. Compare exported values with displayed values.
9. Inspect recursive forecasting logic and chronological holdout logic in source.
10. State separately: (a) personally reproduced findings, (b) code-review findings, and (c) claims that could not be independently verified.

## What an independent review should not do

It should not convert synthetic performance into claimed employer savings, claim production deployment without records, or characterize algorithms such as XGBoost/LSTM/Prophet as the author's inventions.
