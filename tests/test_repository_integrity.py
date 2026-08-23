import ast, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'app.py'
EXPECTED='28dbcd30bd48df4f659ea4b44c87cb1de263989182d4316df7262a0b7c23a87f'
def test_app_source_unchanged():
 assert hashlib.sha256(APP.read_bytes()).hexdigest()==EXPECTED
def test_app_parses():
 ast.parse(APP.read_text(encoding='utf-8'))
def test_required_repository_files_exist():
 for rel in ['README.md','requirements.txt','SOURCE_INTEGRITY.sha256','docs/VALIDATION_PROTOCOL.md','sample_data/synthetic_telecom_equipment_demand.csv','tools/model_execution_runner.py']:
  assert (ROOT/rel).exists(), rel
