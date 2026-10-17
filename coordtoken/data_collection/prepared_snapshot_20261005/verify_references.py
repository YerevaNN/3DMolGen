"""Full structural/key preflight of fixed Group A and separated nabla test CSVs."""
import json,hashlib,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent;BASE=ROOT/'data_collection/rebuild_20260930/overnight_review'
sys.path.insert(0,str(BASE));from audit_existing import audit;from finalize import finalize

def task(i):audit(OUT/'references/manifest.json',i,OUT/'references/chunks')
def main():
 owned=Path('/mnt/weka/fgeikyan/3D_big_data_owned');old=Path('/mnt/weka/fgeikyan/3D_big_data')
 copied=json.loads((ROOT/'data_collection/runs/owned_copy_20261004/current_files.json').read_text())['files'];byrel={x['relative_path']:x for x in copied}
 inventory=json.loads((ROOT/'data_collection/reviews/source_provenance_20260930/input_inventory.json').read_text())['files']
 counts={str(Path(x['path']).relative_to(old)):x['physical_data_lines'] for x in inventory};counts.update({'grp_a/val/backup_un/chembl3d.csv':182420,'grp_a/test/backup_un/chembl3d.csv':37065})
 # The only reference mutation in the preparatory fix is one ChEMBL validation quarantine.
 repair=json.loads((OUT/'repair_3.json').read_text());assert repair['status']=='installed_verified'
 byrel['grp_a/val/backup_un/chembl3d.csv']={**byrel['grp_a/val/backup_un/chembl3d.csv'],**repair};counts['grp_a/val/backup_un/chembl3d.csv']=repair['rows']
 manifest={'scope':'All 28 predefined Group A holdouts, three distinct nabla test sets, and GEOM-revisited; current corrected working inputs. Read-only; no model evaluation.','files':[],'tasks':[]}
 old_manifest=json.loads((BASE/'manifest.json').read_text());manifest['vocab']=old_manifest['vocab'];manifest['vocab_source_sha256']=old_manifest['vocab_source_sha256'];manifest['script_sha256']=hashlib.sha256((BASE/'audit_existing.py').read_bytes()).hexdigest()
 for rel,r in byrel.items():
  if not (rel.startswith('grp_a/val/') or rel.startswith('grp_a/test/') or rel.startswith('grp_c/test/')):continue
  p=Path(r['path']);s=p.stat()
  if s.st_size!=r['bytes'] or s.st_mtime_ns!=r['mtime_ns']:raise ValueError('Reference changed: '+str(p))
  manifest['files'].append({'dataset':rel.removesuffix('.csv'),'path':str(p),'size':s.st_size,'mtime_ns':s.st_mtime_ns,'expected_sha256':r['sha256'],'expected_data_lines':counts[rel],'role':'test' if '/test/' in rel else 'validation','molecule_exclusion':'conformations' not in rel})
 p=Path('/mnt/weka/fgeikyan/fsq/test/geom_revisited.csv');s=p.stat()
 manifest['files'].append({'dataset':'geom_revisited','path':str(p),'size':s.st_size,'mtime_ns':s.st_mtime_ns,'expected_sha256':'3ad8e05da89f1144bc9ef18620990ca596a4a61742ac3281bef4024c33f245f1','expected_data_lines':23404,'role':'test','molecule_exclusion':True})
 assert len(manifest['files'])==32
 for i,r in enumerate(manifest['files']):
  for start in range(0,r['size'],96*1024*1024):manifest['tasks'].append({'file_index':i,'start':start,'end':min(r['size'],start+96*1024*1024)})
 dest=OUT/'references';dest.mkdir(exist_ok=False);(dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('References',len(manifest['files']),'chunks',len(manifest['tasks']),'rows',sum(x['expected_data_lines'] for x in manifest['files']),flush=True)
 with ProcessPoolExecutor(16) as pool:list(pool.map(task,range(len(manifest['tasks']))))
 result=finalize(dest);result['scope']=manifest['scope'];(dest/'final_report.json').write_text(json.dumps(result,indent=2)+'\n')
 print(result['status'],[(x['dataset'],x['counts'].get('format_failure_rows',0),x['counts'].get('standard_inchikey_unavailable_review',0)) for x in result['files']],flush=True)
if __name__=='__main__':main()
