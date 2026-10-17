import csv
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import audit


class ExactAuditTests(unittest.TestCase):
 def fixture(self,root):
  schema=['name','enriched_text','dataset','source_name','source_file_id','source_row_index','inchikey','test_subset']
  files=[];regions=[];categories=[]
  data=[('train',[('A','same'),('A','same'),('B','same'),('A','different conformation')]),
        ('val',[('A','same')]),('test',[('nablaDFT','same'),('nablaDFT','another conformation')])]
  for fi,(role,rows) in enumerate(data):
   p=root/(role+'.csv');s=io.StringIO();w=csv.writer(s,lineterminator='\n');w.writerow(schema)
   header=s.getvalue().encode();s.seek(0);s.truncate()
   for i,(dataset,text) in enumerate(rows):
    subset='conformations' if dataset=='nablaDFT' else ''
    w.writerow([f'{role}:{i}',text,dataset,'original','source',i,'VNWKTOKETHGBQD-UHFFFAOYSA-N',subset])
    if not any(c['split']==role and c['dataset']==dataset for c in categories):
     categories.append({'id':len(categories),'dataset':dataset,'split':role,'test_subset':subset,'file_index':fi})
   body=s.getvalue().encode();p.write_bytes(header+body);st=p.stat()
   files.append({'path':str(p),'role':role,'merged':True,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns})
   regions.append({'file_index':fi,'start':len(header),'end':st.st_size,'rows':len(rows),'sha256':hashlib.sha256(body).hexdigest()})
  return {'files':files,'categories':categories},regions

 def test_exact_text_not_name_or_molecule_and_cross_split_counts(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);plan,regions=self.fixture(root)
   scan=audit.scan_worker((0,regions,plan,str(root/'out')))
   b=hashlib.sha256(b'same').digest()[0]
   result=audit.analyze_bucket((b,[scan['parts'][b]],plan,str(root/'out')))
   self.assertEqual(result['duplicate_text_groups'],1)
   self.assertEqual(result['rows_in_duplicate_groups'],5)
   self.assertEqual(result['redundant_rows_global'],4)
   self.assertEqual(result['within_split_redundant_rows']['train'],2)
   self.assertEqual(result['within_category_redundant_rows'][0],1)
   self.assertEqual(result['split_pair_shared_texts'],{'test|train':1,'test|val':1,'train|val':1})
   self.assertEqual(result['hash_collision_groups'],0)

 def test_forced_hash_collision_uses_full_string_equality(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);plan,regions=self.fixture(root)
   scan=audit.scan_worker((0,regions,plan,str(root/'out')))
   records=bytearray()
   for part in scan['parts'].values():
    raw=Path(part['path']).read_bytes()
    for off in range(0,len(raw),audit.RECORD.size):
     digest,cat,pos=audit.RECORD.unpack(raw[off:off+audit.RECORD.size])
     records.extend(audit.RECORD.pack(b'\0'*32,cat,pos))
   p=root/'collision.bin';p.write_bytes(records)
   result=audit.analyze_bucket((0,[{'path':str(p),'bytes':len(records),'sha256':audit.sha(p)}],plan,str(root/'out')))
   self.assertEqual(result['hash_collision_groups'],1)
   self.assertEqual(result['duplicate_text_groups'],1)
   self.assertEqual(result['redundant_rows_global'],4)

 def test_changed_input_region_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);plan,regions=self.fixture(root);regions[0]['sha256']='wrong'
   with self.assertRaisesRegex(ValueError,'region changed'):
    audit.scan_worker((0,regions,plan,str(root/'out')))

if __name__=='__main__':unittest.main()
