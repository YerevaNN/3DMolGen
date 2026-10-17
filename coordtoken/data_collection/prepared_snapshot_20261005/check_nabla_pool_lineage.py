"""Check whether root nabla CSV is merely a later subsequence of the grouped pool."""
import hashlib,json,time
from pathlib import Path
OUT=Path(__file__).parent;ROOT=Path('/mnt/weka/fgeikyan/3D_big_data_owned')
def main():
 a=ROOT/'grp_c/train/nablaDFT.csv';b=ROOT/'nablaDFT.csv';before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in [a,b]};ha=hashlib.sha256();hb=hashlib.sha256();na=nb=0;extra=[];start=time.monotonic()
 with a.open('rb',buffering=8*1024*1024) as f,b.open('rb',buffering=8*1024*1024) as g:
  h=f.readline();h2=g.readline();assert h==h2;ha.update(h);hb.update(h2);row=g.readline()
  for raw in f:
   ha.update(raw);na+=1
   if raw==row:hb.update(row);nb+=1;row=g.readline()
   elif len(extra)<5:extra.append(raw.decode().split(',',1)[0])
  remaining=bool(row)
 after={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in [a,b]};assert before==after
 assert ha.hexdigest()=='2c5812234ce8daa4fd9773244035946dc40ad5e03861c249d39491a691596c79'
 if not remaining:assert hb.hexdigest()=='691ea95877836cc1dbc4c4695e8d5ed486b9fefdfcef1053340f8382ba60b4f9'
 result={'grouped_pool_rows':na,'root_rows_matched_in_order':nb,'root_is_exact_byte_subsequence':not remaining,'grouped_rows_not_in_root':na-nb if not remaining else None,'examples_of_grouped_rows_absent_from_root':extra,'grouped_sha256':ha.hexdigest(),'root_sha256':hb.hexdigest() if not remaining else None,'elapsed_seconds':time.monotonic()-start,'decision':'Use grouped train as the less-filtered legacy pool; root file adds no rows' if not remaining else 'Subset comparison inconclusive; no input selection change'}
 (OUT/'nabla_pool_lineage.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
