#!/usr/bin/env python3
"""Repository evidence validator. Does not modify app.py."""
from __future__ import annotations
import ast, csv, hashlib, json, os, platform, py_compile, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
APP=ROOT/'app.py'
OUT=ROOT/'evidence'/'execution'
OUT.mkdir(parents=True, exist_ok=True)
EXPECTED_SHA='28dbcd30bd48df4f659ea4b44c87cb1de263989182d4316df7262a0b7c23a87f'
EXPECTED_FUNCS={
'format_number','safe_mape','safe_wmape','calculate_metrics','make_demo_data','prepare_time_series',
'generate_future_dates','aligned_prophet_holidays','future_driver_frame','time_features','exogenous_features',
'run_prophet_model','run_xgboost_model','run_lstm_model','run_holt_winters_model','result_leaderboard','export_excel'
}

def sha(path):
 h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def write(name,text): (OUT/name).write_text(text,encoding='utf-8')

def main():
 started=datetime.now(timezone.utc)
 checks=[]
 current_sha=sha(APP)
 checks.append(('app_source_integrity',current_sha==EXPECTED_SHA,f'expected={EXPECTED_SHA}; actual={current_sha}'))
 try:
  py_compile.compile(str(APP),doraise=True); checks.append(('python_compile',True,'py_compile PASS'))
 except Exception as e: checks.append(('python_compile',False,str(e)))
 try:
  tree=ast.parse(APP.read_text(encoding='utf-8'))
  funcs={n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
  missing=sorted(EXPECTED_FUNCS-funcs)
  checks.append(('expected_function_inventory',not missing,'missing='+','.join(missing) if missing else 'all expected functions present'))
 except Exception as e: checks.append(('expected_function_inventory',False,str(e)))
 req=ROOT/'requirements.txt'
 checks.append(('requirements_present',req.exists(),str(req.relative_to(ROOT)) if req.exists() else 'missing'))
 sample=ROOT/'sample_data'/'synthetic_telecom_equipment_demand.csv'
 checks.append(('synthetic_sample_present',sample.exists(),str(sample.relative_to(ROOT)) if sample.exists() else 'missing'))
 if sample.exists():
  try:
   with sample.open(newline='',encoding='utf-8') as f:
    r=csv.DictReader(f); rows=list(r); cols=r.fieldnames or []
   checks.append(('synthetic_sample_readable',len(rows)>0,f'rows={len(rows)}; columns={len(cols)}'))
  except Exception as e: checks.append(('synthetic_sample_readable',False,str(e)))
 dirs=['docs','evidence','results','screenshots','sample_data','tools','tests','.github/workflows']
 for d in dirs: checks.append((f'directory_{d.replace("/","_")}',(ROOT/d).exists(),d))

 ended=datetime.now(timezone.utc)
 manifest={
  'run_type':'repository static/integrity validation',
  'started_utc':started.isoformat(),'ended_utc':ended.isoformat(),
  'python':sys.version,'platform':platform.platform(),
  'app_sha256':current_sha,'expected_app_sha256':EXPECTED_SHA,
  'app_unchanged':current_sha==EXPECTED_SHA,
  'checks':[{'check':a,'status':'PASS' if b else 'FAIL','detail':c} for a,b,c in checks]
 }
 (OUT/'static_validation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 with (OUT/'static_validation_results.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f); w.writerow(['check','status','detail']);
  for a,b,c in checks:w.writerow([a,'PASS' if b else 'FAIL',c])
 report=['# Repository Validation Report','',f'- app.py SHA-256: `{current_sha}`',f'- Original-source integrity: **{"PASS" if current_sha==EXPECTED_SHA else "FAIL"}**','', '## Checks']
 report += [f'- {a}: **{"PASS" if b else "FAIL"}** — {c}' for a,b,c in checks]
 report += ['', '## Meaning of this report','This report proves repository integrity, syntax compilation, expected source structure, and evidence-package presence. It is not a substitute for full model execution or independent third-party validation.']
 write('STATIC_VALIDATION_REPORT.md','\n'.join(report)+'\n')
 print('\n'.join(f'{a}: {"PASS" if b else "FAIL"}' for a,b,c in checks))
 return 0 if all(b for _,b,_ in checks) else 1
if __name__=='__main__': raise SystemExit(main())
