"""Append source-verified repairs once, retaining an exact restoration receipt."""
import csv,hashlib,io,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).parent;P=OUT.parent

def main():
 report=OUT/'restoration.json'
 if report.exists():raise RuntimeError('Restoration already attempted; inspect receipt before any retry')
 audit=json.loads((OUT/'pubchem_repairs.json').read_text());rows=[r for r in audit['rows'] if r['status']=='source_backed_repair_verified'];assert len(rows)==10
 prior=json.loads((P/'repair_0.json').read_text());target=Path(prior['path']);s=target.stat();assert (s.st_size,s.st_mtime_ns)==(prior['bytes'],prior['mtime_ns'])
 buf=io.StringIO(newline='');w=csv.writer(buf,lineterminator='\n')
 for r in rows:w.writerow([r['name'],r['replacement_enriched_text']])
 payload=buf.getvalue().encode();(OUT/'restored_pubchem_rows.csv').write_bytes(b'name,enriched_text\n'+payload)
 h=hashlib.sha256();n=0;started=time.monotonic()
 with target.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block);n+=block.count(b'\n')
 assert h.hexdigest()==prior['sha256'] and n-1==prior['rows'];assert (target.stat().st_size,target.stat().st_mtime_ns)==(s.st_size,s.st_mtime_ns)
 result={'status':'append_started','path':str(target),'before_bytes':s.st_size,'before_sha256':prior['sha256'],'before_rows':prior['rows'],'restored_rows':len(rows),'restored_names':[r['name'] for r in rows],'placement':'Appended in repair-manifest order; original retained rows remain byte-identical','source_evidence':str(OUT/'pubchem_repairs.json'),'payload_sha256':hashlib.sha256(payload).hexdigest()};report.write_text(json.dumps(result,indent=2)+'\n')
 with target.open('ab') as f:f.write(payload);f.flush();os.fsync(f.fileno())
 with target.open('rb') as f:f.seek(s.st_size);actual=f.read()
 assert actual==payload;h.update(actual)
 after=target.stat();result.update(status='source_backed_restoration_verified',sha256=h.hexdigest(),rows=prior['rows']+len(rows),bytes=after.st_size,mtime_ns=after.st_mtime_ns,owner_uid=after.st_uid,remaining_quarantined_total=9,elapsed_seconds=time.monotonic()-started,verification='Entire existing prefix read and checked against its pinned hash; appended records fsynced and independently read back byte-for-byte; combined full-file hash calculated from verified bytes')
 report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
