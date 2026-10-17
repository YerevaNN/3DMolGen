import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from run_nabla import run, b, verify, shared
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'group_b_split'))
from test_group_b import add_file, base_manifest, encoded
import index_inputs


class EndToEndTest(unittest.TestCase):
    def test_nabla_writer_and_global_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);af=[];at=[]
            for role,atom in [('train','O'),('val','N'),('test','C')]:
                add_file(root,af,at,'A',role,[(role,encoded([atom]))],group_a=True)
            for name in ['nablaDFT_structures','nablaDFT_scaffolds','geom_revisited']:
                add_file(root,af,at,name,'test',[(name,encoded(['C']))],group_a=False)
            am=base_manifest(af,at);am.update(reserved_validation_keys=[],quarantined_group_a_records=[],policy={})
            arun=root/'a_run';shared.run_pipeline(am,arun,2)
            pool=root/'pools';pool.mkdir();bf=[];bt=[]
            add_file(pool,bf,bt,'pubchem3d','train',[(s,encoded(list(s))) for s in ['C','N','O','F','CO']],kind='group_b_pool')
            rows=[('test',encoded(['C'])),('val',encoded(['N']))]
            for atom,n in [('O',3),('F',3),('Cl',300),('Br',300)]:
                rows.extend((f'{atom}_{i}',encoded([atom],i)) for i in range(n))
            add_file(pool,bf,bt,'nablaDFT','train',rows,kind='nabla_training_reference')
            m=base_manifest(bf,bt);m.update(group_a_run=str(arun),quarantine={'rows':[]},policy={'group_b':{
                'datasets_in_order':['pubchem3d'],'seed':42,'test_target_fraction_of_input_rows':.005,'validation_target_fraction_of_input_rows':.005}})
            for name,field in [('report.json','group_a_report_sha256'),('manifest.json','group_a_manifest_sha256'),
                               ('test_keys.json','group_a_test_keys_sha256'),('validation_keys.json','group_a_validation_keys_sha256')]:
                m[field]=shared.sha(arun/name)
            ix=root/'index';brun=root/'b_run';index_inputs.run(m,ix,2);b.run(ix,brun,2)
            audit=root/'audit';verify.main(argparse.Namespace(group_a=arun,group_b=brun,output=audit,workers=2))
            conformer=root/'nablaDFT_conformations.csv'
            with conformer.open('w') as f:
                w=csv.writer(f,lineterminator='\n');w.writerow(['name','enriched_text']);w.writerow(['conformer',encoded(['F'])])
            st=conformer.stat();ct={'path':str(conformer),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':shared.sha(conformer),
                                  'rows':1,'dataset':'nablaDFT_conformations'}
            tests=[af[3],af[4],ct]
            out=root/'c_run';contract={'code_hashes':{},'evidence_hashes':{},'nabla_tests':tests,'policy':{}}
            run(argparse.Namespace(group_a=arun,group_b=brun,index_run=ix,audit=audit,output=out,workers=2,contract=contract))
            r=json.loads((out/'report.json').read_text())
            self.assertEqual(r['status'],'group_c_and_global_molecular_split_complete')
            self.assertEqual(r['selection']['target_rows'],3)
            self.assertEqual(r['selection']['selected_validation_rows'],301)
            self.assertEqual(r['selection']['target_shortfall'],0)
            self.assertEqual(r['counts']['excluded_test_overlap'],1)
            self.assertEqual(r['other_training_rows_removed'],0)
            with (out/'candidate/train/nablaDFT.csv').open() as f:names={x['name'] for x in csv.DictReader(f)}
            self.assertIn('O_0',names);self.assertIn('F_0',names)
            self.assertTrue((out/'combined_split_manifest.json').exists())


if __name__=='__main__':unittest.main()
