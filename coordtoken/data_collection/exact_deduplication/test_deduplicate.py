import csv
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
import deduplicate as d
import summarize_aliases


class DeduplicationTests(unittest.TestCase):
 def fixture(self,root):
  records=[['s1:0','same','A','original-a','s1','0','VNWKTOKETHGBQD-UHFFFAOYSA-N',''],
           ['s2:0','same','B','original-b','s2','0','VNWKTOKETHGBQD-UHFFFAOYSA-N',''],
           ['s2:1','different conformation','B','original-c','s2','1','VNWKTOKETHGBQD-UHFFFAOYSA-N','']]
  encoded=[]
  for row in records:
   s=io.StringIO();csv.writer(s,lineterminator='\n').writerow(row);encoded.append(s.getvalue().encode())
  source=root/'input.csv';source.write_bytes(d.merge.HEADER_BYTES+b''.join(encoded));st=source.stat()
  start=len(d.merge.HEADER_BYTES)+len(encoded[0]);end=st.st_size
  drops=np.array([(start,len(d.merge.HEADER_BYTES),hashlib.sha256(b'same').digest())],dtype=d.DTYPE)
  path=root/'drops.npy';np.save(path,drops)
  task={'id':0,'file':{'path':str(source),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'role':'train'},
        'start':start,'end':end,'rows':2,'sha256':hashlib.sha256(b''.join(encoded[1:])).hexdigest(),
        'drops':str(path),'drop_start':0,'drop_end':1,'source_file_id':'s2','dataset':'B','test_subset':'','output':str(root/'out')}
  return task,encoded

 def test_cross_source_duplicate_preserves_distinct_conformer_and_alias(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);task,encoded=self.fixture(root);result=d.filter_region(task)
   self.assertEqual(result['removed_rows'],1)
   self.assertEqual(Path(result['outputs']['train']['path']).read_bytes(),encoded[2])
   with open(result['aliases']['path']) as f:alias=next(csv.reader(f))
   self.assertEqual(alias,['s2:0','s1:0','B','original-b','s2','0','VNWKTOKETHGBQD-UHFFFAOYSA-N','','train'])

 def test_unequal_text_refuses_deletion(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);task,encoded=self.fixture(root);a=np.load(task['drops']);a['keeper']=task['start']+len(encoded[1]);np.save(task['drops'],a)
   with self.assertRaisesRegex(ValueError,'Full-string'):d.filter_region(task)

 def test_wrong_source_hash_refuses_completion(self):
  with tempfile.TemporaryDirectory() as tmp:
   task,_=self.fixture(Path(tmp));task['sha256']='wrong'
   with self.assertRaisesRegex(ValueError,'Source region'):d.filter_region(task)

 def test_empty_drop_list_preserves_all_bytes(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);task,encoded=self.fixture(root);task['drop_end']=0
   result=d.filter_region(task)
   self.assertEqual(result['removed_rows'],0)
   self.assertEqual(Path(result['outputs']['train']['path']).read_bytes(),b''.join(encoded[1:]))

 def test_source_relationship_statistics(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p=root/'aliases.csv'
   with p.open('w',newline='') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(d.ALIAS_HEADER)
    for sid,dataset in [('s2','B'),('s3','A'),('s1','A')]:
     w.writerow([sid+':1','s1:0',dataset,'original',sid,'1','VNWKTOKETHGBQD-UHFFFAOYSA-N','','train'])
   sources=[{'source_file_id':sid,'output_dataset':dataset,'test_subset':'','path':sid+'.csv'}
            for sid,dataset in [('s1','A'),('s2','B'),('s3','A')]]
   d.atomic(root/'sources.json',{'files':sources})
   d.atomic(root/'manifest.json',{'duplicate_aliases':{'path':str(p),'rows':3,'sha256':d.sha(p)},
            'removed_rows':{'train':3},'deduplication_policy':'first occurrence'})
   result=summarize_aliases.summarize(root)
   self.assertEqual(result['by_split_relationship']['train'],{
     'different_dataset':1,'different_source_file_same_dataset':1,'same_prepared_source_file':1})

if __name__=='__main__':unittest.main()
