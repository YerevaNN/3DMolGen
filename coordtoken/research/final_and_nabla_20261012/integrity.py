import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'logs/final_and_nabla_20261012'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify_sources():
 manifest=json.loads((HERE/'freeze.json').read_text())
 for name,digest in manifest['files'].items():
  if sha(HERE/name)!=digest:raise ValueError('Frozen source changed: '+name)
def verify_case(case):
 m=json.loads((Path(case['indices'])/'manifest.json').read_text())
 if m['status']!='complete' or m['data_manifest_sha256']!=sha(Path(case['data'])/'manifest.json'):raise ValueError('Data/indices mismatch')
 for name in ['order.npy','validation.npy']:
  if sha(Path(case['indices'])/name)!=m[name+'_sha256']:raise ValueError('Index changed: '+name)
 return m
