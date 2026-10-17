import csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
OWN=Path('/mnt/weka/fgeikyan/3D_big_data_owned');OLD=Path('/mnt/weka/fgeikyan/3D_big_data')
if (OUT/'repair_plan.json').exists():raise RuntimeError('A pinned repair plan already exists; do not regenerate it over an active or completed run')
prior=json.loads((ROOT/'data_collection/runs/owned_copy_20261004/current_files.json').read_text());files={x['relative_path']:x for x in prior['files']}
old_inventory=json.loads((ROOT/'data_collection/reviews/source_provenance_20260930/input_inventory.json').read_text())
# Existing independently counted input snapshots, checked by full copy hashes.
counts={str(Path(x['path']).relative_to(OLD)):x['physical_data_lines'] for x in old_inventory['files']}
counts.update({'grp_a/train/backup_un/chembl3d.csv':1442219,'grp_a/val/backup_un/chembl3d.csv':182420,'grp_a/test/backup_un/chembl3d.csv':37065})
for rel,row in files.items():
 s=Path(row['path']).stat()
 if (s.st_size,s.st_mtime_ns)!=(row['bytes'],row['mtime_ns']):raise ValueError('Working file changed: '+rel)
parser=json.loads((ROOT/'data_collection/reviews/remaining_verdict_20261004/parser_quarantine.json').read_text())['rows']
fixes={}
for row in parser:
 rel=str(Path(row['source']).relative_to(OLD));fixes.setdefault(rel,[]).append({'name':row['name'],'byte_offset':row['byte_offset'],'row_sha256':row['row_sha256'],'reason':'sanitized_graph_parse_failure'})
rel='grp_b/zinc.csv';offset=41869319949
with (OWN/rel).open('rb') as f:f.seek(offset);raw=f.readline()
name,text=next(csv.reader([raw.decode()]));assert name=='ZINC000573456341'
sys.path.insert(0,str(ROOT/'data_collection'));from inchikey_filter import standard_inchikey
try:standard_inchikey(text)
except ValueError:pass
else:raise ValueError('Previously failed key now works: investigate instead of quarantining')
fixes[rel].append({'name':name,'byte_offset':offset,'row_sha256':hashlib.sha256(raw).hexdigest(),'reason':'standard_inchikey_unavailable'})
review=json.loads((ROOT/'data_collection/reviews/remaining_verdict_20261004/strict_stereo_review.json').read_text())['rows']
for split in ['train','val']:
 rel=f'grp_a/{split}/backup_un/chembl3d.csv';targets={x['name'] for x in review if x['split']==split};found=set()
 with (OWN/rel).open('rb') as f:
  f.readline()
  while raw:=f.readline():
   name=next(csv.reader([raw.decode()]))[0]
   if name not in targets:continue
   found.add(name);fixes.setdefault(rel,[]).append({'name':name,'byte_offset':f.tell()-len(raw),'row_sha256':hashlib.sha256(raw).hexdigest(),'reason':'source_stereo_normalization_unresolved'})
 assert found==targets
specs=[]
for rel,removals in fixes.items():
 original=str(OLD/rel)
 if 'chembl' in rel:original=str(ROOT/'data_collection/reviews/chembl_replacement_20261004/chembl3d.checked.csv') if '/train/' in rel else '/mnt/weka/fgeikyan/fsq/val/chembl3d.csv'
 specs.append({'path':str(OWN/rel),'relative_path':rel,'sha256':files[rel]['sha256'],'rows':counts[rel],'removals':removals,'original_preserved_at':original})
(OUT/'repair_plan.json').write_text(json.dumps({'files':specs,'policy':'Quarantine 12 parse failures, one key failure, and six unresolved source-stereo cases. Exact bytes preserved. Only owned working files change; source originals remain intact.'},indent=2)+'\n')
print('Planned corrections',[(x['relative_path'],len(x['removals'])) for x in specs])
