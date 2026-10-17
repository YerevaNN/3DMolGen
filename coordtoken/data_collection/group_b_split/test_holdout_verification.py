import argparse
import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import verify_holdout_precedence as audit


class VerificationTests(unittest.TestCase):
    def fixture(self, root, contaminate):
        arun=root/'a';brun=root/'b';arun.mkdir();brun.mkdir()
        keys={atom:audit.direct_key(atom) for atom in ('C','N','O')}
        a_files=[];b_files=[]
        for directory,files in [(arun,a_files),(brun,b_files)]:
            for role,atom in [('train','O'),('val','N'),('test','C')]:
                if contaminate and directory==brun and role=='val':atom='C'
                path=directory/f'{role}.csv'
                with path.open('w') as f:
                    w=csv.writer(f,lineterminator='\n');w.writerow(['name','enriched_text']);w.writerow(['record',f'[{atom}]<0,0,0>'])
                files.append({'dataset':'A' if directory==arun else 'pubchem3d','role':role,'rows':1,
                              'path':str(path),'sha256':audit.sha(path),'group_a':True,'label':role})
        (arun/'report.json').write_text(json.dumps({'files':a_files}))
        (brun/'report.json').write_text(json.dumps({'files':b_files,'datasets':[{'dataset':'pubchem3d'}]}))
        (arun/'manifest.json').write_text(json.dumps({'files':a_files,'tasks':[{'id':0,'file_index':0}]}))
        (arun/'test_keys.json').write_text(json.dumps([keys['C']]))
        (arun/'validation_keys.json').write_text(json.dumps([keys['C'],keys['N']]))
        index=arun/'index/0000';index.mkdir(parents=True)
        (index/'counts.json').write_text(json.dumps({keys['O']:1}))
        db=sqlite3.connect(brun/'pubchem3d.sqlite');db.execute('CREATE TABLE counts(key TEXT PRIMARY KEY,n INTEGER,action TEXT)')
        db.executemany('INSERT INTO counts VALUES (?,1,?)',[(keys['C'],'test'),(keys['N'],'val'),(keys['O'],'train')]);db.commit();db.close()
        return argparse.Namespace(group_a=arun,group_b=brun,output=root/'verification',workers=2)

    def test_clean_outputs_and_materialized_validation_contamination(self):
        for contaminated in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                args=self.fixture(Path(tmp),contaminated)
                if contaminated:
                    with self.assertRaisesRegex(ValueError,'Independent overlap verification failed'):
                        audit.main(args)
                else:
                    audit.main(args)
                result=json.loads((args.output/'report.json').read_text())
                self.assertEqual(result['actual_test_validation_shared_keys'],int(contaminated))
                self.assertEqual(result['pubchem_original_conflicts_by_destination'],[{'destination':'test','keys':1,'rows':1}])


if __name__=='__main__':unittest.main()
