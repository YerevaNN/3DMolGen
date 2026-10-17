import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from rdkit import rdBase
import index_inputs
import split_pools as b

shared = b.shared
PARSER = Path(__file__).resolve().parents[1] / 'group_a_split/audit_parser.py'


def encoded(chain, x=0):
    return ''.join(f'[{atom}]<{x+i},0,0>' for i,atom in enumerate(chain))


def add_file(root, files, tasks, name, role, rows, kind=None, group_a=False):
    path=root/f'{len(files)}.csv'
    with path.open('w') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['name','enriched_text']);w.writerows(rows)
    s=path.stat();fi=len(files)
    files.append({'path':str(path),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':shared.sha(path),
                  'rows':len(rows),'dataset':name,'role':role,'kind':kind,'group_a':group_a,'label':f'{name}/{role}'})
    for start in range(0,s.st_size,83):
        tasks.append({'id':len(tasks),'file_index':fi,'start':start,'end':min(start+83,s.st_size)})


def base_manifest(files,tasks):
    return {'files':files,'tasks':tasks,'rdkit_version':rdBase.rdkitVersion,'parser_path':str(PARSER),
            'parser_sha256':shared.sha(PARSER),'vocab':[],'code_hashes':{}}


class GroupBTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_never_promotes_existing_train(self):
        outcomes=[]
        for reverse in (False,True):
            db=sqlite3.connect(':memory:')
            db.execute('CREATE TABLE counts(key TEXT PRIMARY KEY,n INTEGER,owner INTEGER,action TEXT)')
            records=[('test_a',3,3,None),('test_b',3,3,None),('val',1,2,None),
                     ('old_train',9,1,None),('new_a',2,0,None),('new_b',2,0,None),('new_c',2,0,None)]
            db.executemany('INSERT INTO counts VALUES (?,?,?,?)',records[::-1] if reverse else records)
            selection=b.choose(db,3,'fixture',42)
            routes=dict(db.execute('SELECT key,action FROM counts'))
            self.assertEqual(routes['old_train'],'train')
            self.assertEqual(sum(routes[k]=='test' for k in ('test_a','test_b')),1)
            self.assertEqual(sum(routes[k]=='excluded_test_quota' for k in ('test_a','test_b')),1)
            self.assertEqual(selection['val']['selected_rows'],3)
            outcomes.append(routes);db.close()
        self.assertEqual(*outcomes)
        db=sqlite3.connect(':memory:');db.execute('CREATE TABLE counts(key TEXT PRIMARY KEY,n INTEGER,owner INTEGER,action TEXT)')
        db.execute("INSERT INTO counts VALUES ('protected_train',100,1,NULL)")
        result=b.choose(db,5,'shortfall',42)
        self.assertEqual(result['test']['target_shortfall'],5)
        self.assertEqual(result['val']['target_shortfall'],5)
        self.assertEqual(db.execute('SELECT action FROM counts').fetchone()[0],'train')

    def test_complete_index_and_iterative_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a_files=[];a_tasks=[]
            add_file(root,a_files,a_tasks,'A','train',[('a_train',encoded(['O']))],group_a=True)
            add_file(root,a_files,a_tasks,'A','val',[('a_val',encoded(['N'])),('val_test_overlap',encoded(['C']))],group_a=True)
            add_file(root,a_files,a_tasks,'A','test',[('a_test',encoded(['C'])),('a_test2',encoded(['Br']))],group_a=True)
            am=base_manifest(a_files,a_tasks);am.update(reserved_validation_keys=[],quarantined_group_a_records=[],policy={})
            arun=root/'a_run';shared.run_pipeline(am,arun,2)
            pool_root=root/'pool_inputs';pool_root.mkdir();files=[];tasks=[]
            rows=[]
            for label,chain,n in [('test',['C'],3),('test2',['Br'],3),('val',['N'],1),('a_train',['O'],2),
                                 ('nabla_train',['F'],2),('new1',['C','C'],2),('new2',['C','C','C'],2),('new3',['C','O'],2)]:
                rows.extend((f'{label}_{i}',encoded(chain,i)) for i in range(n))
            rows.append(('bad','[C]<nan,0,0>'))
            add_file(pool_root,files,tasks,'B1','train',rows,kind='group_b_pool')
            add_file(pool_root,files,tasks,'B2','train',rows+[('new4',encoded(['C','C','O'])),('new5',encoded(['C','N']))],kind='group_b_pool')
            add_file(pool_root,files,tasks,'nablaDFT','train',[('nabla',encoded(['F']))],kind='nabla_training_reference')
            m=base_manifest(files,tasks)
            m.update(group_a_run=str(arun),quarantine={'rows':[]},policy={'group_b':{
                'datasets_in_order':['B1','B2'],'seed':42,'test_target_fraction_of_input_rows':.2,
                'validation_target_fraction_of_input_rows':.2}})
            for name,field in [('report.json','group_a_report_sha256'),('manifest.json','group_a_manifest_sha256'),
                               ('test_keys.json','group_a_test_keys_sha256'),('validation_keys.json','group_a_validation_keys_sha256')]:
                m[field]=shared.sha(arun/name)
            ix=root/'index_run';out=root/'split_run'
            index_inputs.run(m,ix,2);b.run(ix,out,2)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['status'],'group_b_iterative_split_complete')
            self.assertEqual(len(report['files']),6)
            self.assertTrue(all(x['overlapping_keys']==0 for x in report['final_overlap_checks']))
            for dataset in report['datasets']:
                self.assertEqual(dataset['counts']['quarantine_identity_failure'],1)
                self.assertEqual(sum(dataset['counts'].values()),dataset['input_rows'])
                train=next(f for f in dataset['outputs'] if f['role']=='train')
                with open(train['path']) as f:names={r['name'] for r in csv.DictReader(f)}
                self.assertIn('a_train_0',names);self.assertIn('nabla_train_0',names)
                db=b.database(out/f"{dataset['dataset']}.sqlite",readonly=True)
                # Multiple conformers receive one assignment per key.
                self.assertEqual(db.execute("SELECT sum(n) FROM counts WHERE owner=1 AND action!='train'").fetchone()[0],None)
                db.close()
            for f in files+a_files:self.assertEqual(shared.sha(f['path']),f['sha256'])


if __name__=='__main__':
    unittest.main()
