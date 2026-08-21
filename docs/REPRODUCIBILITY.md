# Reproducibility Protocol

## 1. Preserve the exact release
Record the public repository URL, release tag, and Git commit SHA before testing.

## 2. Create an isolated environment

Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Preserve exact environment versions
After installation succeeds:

```bash
python -m pip freeze > evidence/environment_freeze.txt
python --version > evidence/python_version.txt
```

Do not invent package versions in advance. Preserve the versions actually used.

## 4. Verify source integrity

```bash
python -m py_compile app.py
```

Compare `app.py` against `SOURCE_INTEGRITY.sha256`.

Linux/macOS:
```bash
sha256sum -c SOURCE_INTEGRITY.sha256
```

PowerShell:
```powershell
Get-FileHash .\app.py -Algorithm SHA256
```

## 5. Run the application

```bash
python -m streamlit run app.py
```

## 6. Reproduce a documented run
Use `sample_data/synthetic_telecom_equipment_demand.csv` or another authorized dataset. Record every selected configuration field in a copy of `evidence/run_manifest_template.json`.

## 7. Preserve raw outputs
Save the actual leaderboard, validation predictions, future forecast, exported workbook, screenshots, and console/environment details. Do not replace raw outputs with narrative-only summaries.
