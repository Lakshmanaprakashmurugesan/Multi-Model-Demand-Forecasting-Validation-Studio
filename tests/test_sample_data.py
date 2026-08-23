import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_sample_dataset_is_nonempty_and_documented():
 p=ROOT/'sample_data'/'synthetic_telecom_equipment_demand.csv'
 with p.open(newline='',encoding='utf-8') as f:
  r=csv.DictReader(f); rows=list(r)
 assert len(rows)>0
 assert r.fieldnames
 assert (ROOT/'sample_data'/'DATA_DICTIONARY.md').exists()
