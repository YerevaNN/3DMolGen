import csv
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
import merge


class ProvenanceTests(unittest.TestCase):
    def setup_task(self, root):
        def line(name, text):
            s=io.StringIO();csv.writer(s,lineterminator='\n').writerow([name,text]);return s.getvalue().encode()
        rows=[line('same','excluded'),line('same','kept,with,commas'),
              line('same','kept,with,commas'),line('quoted,"name"','validation')]
        source=root/'source.csv';source.write_bytes(merge.ORIGINAL_HEADER+b''.join(rows))
        key=b'VNWKTOKETHGBQD-UHFFFAOYSA-N\n';keys=root/'keys.txt';keys.write_bytes(key*4)
        parts={}
        for role,records in [('train',rows[1:3]),('val',rows[3:]),('test',[]),('excluded',rows[:1])]:
            p=root/(role+'.csv');p.write_bytes(b''.join(records));parts[role]={'path':str(p),'sha256':merge.sha(p)}
        st=source.stat()
        return {'id':0,'source':{'path':str(source),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,
                'source_file_id':'s0001','output_dataset':'example','test_subset':''},
                'start':0,'end':st.st_size,'row_start':0,'rows':4,'keys':str(keys),
                'keys_sha256':merge.sha(keys),'source_record_sha256':hashlib.sha256(b''.join(rows)).hexdigest(),
                'parts':parts,'expected_counts':{'train':2,'val':1,'test':0,'excluded':1},'output':str(root/'output')}

    def test_original_indices_duplicate_names_and_assembly(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=self.setup_task(root);result=merge.run_chunk(task)
            with open(root/'output/train.csv') as f: rows=list(csv.reader(f))
            self.assertEqual([r[0] for r in rows],['s0001:1','s0001:2'])
            self.assertEqual([r[5] for r in rows],['1','2'])
            self.assertEqual(rows[0][1:4],['kept,with,commas','example','same'])
            target=root/'merged.csv';part=result['outputs']['train']
            target.write_bytes(merge.HEADER_BYTES+b'\0'*part['bytes'])
            merge.copy_region((part['path'],str(target),len(merge.HEADER_BYTES),part['bytes'],part['sha256']))
            checked=merge.verify_output((target,[part],2));self.assertEqual(checked['rows'],2)
            self.assertEqual(checked['sha256'],merge.sha(target))
            with target.open('r+b') as f:f.seek(-2,2);f.write(b'X')
            with self.assertRaisesRegex(ValueError,'readback'):merge.verify_output((target,[part],2))

    def test_wrong_key_sidecar_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            task=self.setup_task(Path(temp));Path(task['keys']).write_bytes(b'X'*27+b'\n')
            with self.assertRaises(ValueError):merge.run_chunk(task)

    def test_changed_partition_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            task=self.setup_task(Path(temp));Path(task['parts']['train']['path']).write_bytes(b'wrong,record\n')
            with self.assertRaisesRegex(ValueError,'alignment'):merge.run_chunk(task)

    def test_nonzero_chunk_origin_and_fixed_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=self.setup_task(root)
            task['parts']={};task['fixed_role']='test';task['row_start']=100
            task['expected_counts']={'test':4};result=merge.run_chunk(task)
            with open(result['outputs']['test']['path']) as f:rows=list(csv.reader(f))
            self.assertEqual([r[5] for r in rows],['100','101','102','103'])

    def test_crlf_source_header(self):
        with tempfile.TemporaryDirectory() as temp:
            task=self.setup_task(Path(temp));p=Path(task['source']['path'])
            p.write_bytes(p.read_bytes().replace(merge.ORIGINAL_HEADER,b'name,enriched_text\r\n',1))
            st=p.stat();task['source'].update(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
            task['end']=st.st_size
            result=merge.run_chunk(task)
            self.assertEqual(result['outputs']['train']['rows'],2)


if __name__=='__main__':unittest.main()
