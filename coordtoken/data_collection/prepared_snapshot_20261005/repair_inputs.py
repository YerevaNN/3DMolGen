"""Remove only pinned, reviewed records from owned inputs; preserve exact removed rows."""
import argparse,csv,hashlib,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent

def sha(path):
 h=hashlib.sha256();n=0
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b);n+=b.count(b'\n')
 return h.hexdigest(),n-1

def rewrite(item,dest):
 src=Path(item['path']);before=src.stat();assert before.st_uid==os.getuid()
 tmp=dest;cuts=sorted(item['removals'],key=lambda x:x['byte_offset']);removed=[];read_hash=hashlib.sha256();write_hash=hashlib.sha256();position=0
 with src.open('rb') as f,tmp.open('xb') as out:
  for cut in cuts+[{'byte_offset':before.st_size,'sentinel':True}]:
   left=cut['byte_offset']-position
   if left<0:raise ValueError('Overlapping/out-of-order cuts')
   while left:
    block=f.read(min(left,8*1024*1024))
    if not block:raise ValueError('Unexpected EOF')
    read_hash.update(block);write_hash.update(block);out.write(block);left-=len(block);position+=len(block)
   if cut.get('sentinel'):break
   raw=f.readline();position+=len(raw);read_hash.update(raw)
   if hashlib.sha256(raw).hexdigest()!=cut['row_sha256']:raise ValueError('Quarantine row changed')
   name,text=next(csv.reader([raw.decode()]))
   if name!=cut['name']:raise ValueError('Quarantine name mismatch')
   removed.append({**cut,'source':str(src),'raw_csv_row':raw.decode()})
  out.flush();os.fsync(out.fileno())
 if read_hash.hexdigest()!=item['sha256']:raise ValueError('Input hash changed; output not installed')
 after=src.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Input changed during rewrite')
 digest,rows=sha(tmp)
 if digest!=write_hash.hexdigest() or rows!=item['rows']-len(cuts):raise ValueError('Independent output check failed')
 return {'path':str(src),'input_sha256':item['sha256'],'input_rows':item['rows'],'sha256':digest,'rows':rows,'removed':removed,'bytes':tmp.stat().st_size}

def main():
 p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
 manifest=json.loads((OUT/'repair_plan.json').read_text());item=manifest['files'][a.index];target=Path(item['path']);report=OUT/f'repair_{a.index}.json'
 if report.exists():
  r=json.loads(report.read_text());assert r['status']=='installed_verified';assert sha(target)==(r['sha256'],r['rows']);print('Already complete');return
 tmp=target.with_name(target.name+'.quality_fix_20261005.tmp')
 started=time.monotonic();result=rewrite(item,tmp)
 # Preserve exact excluded records before installing; full pre-repair input also survives at recorded original_path.
 quarantine=OUT/f'removed_{a.index}.jsonl'
 with quarantine.open('x') as f:
  for row in result['removed']:f.write(json.dumps(row)+'\n')
 result['original_preserved_at']=item['original_preserved_at'];result['quarantine']=str(quarantine)
 os.replace(tmp,target);result['mtime_ns']=target.stat().st_mtime_ns;result['owner_uid']=target.stat().st_uid
 result.update(status='installed_verified',elapsed_seconds=time.monotonic()-started)
 report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='removed'},indent=2))
if __name__=='__main__':main()
