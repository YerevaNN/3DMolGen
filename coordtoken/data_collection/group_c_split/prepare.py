"""Pin completed A/B evidence and prepared nabla sources for the final step."""
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'group_a_split'))
from process_group_a import atomic,sha,check_stat

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('preflight','group-a','group-b','index-run','audit','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    inputs=json.loads((a.preflight/'prepared_input_manifest.json').read_text())
    tests=[]
    for f in inputs['files']:
        if f['relative_path'].startswith('grp_c/test/'):
            check_stat(f);tests.append({**f,'dataset':Path(f['path']).stem})
    if len(tests)!=3:raise ValueError('Expected three distinct nabla test files')
    if json.loads((a.audit/'report.json').read_text())['status']!='passed':raise ValueError('Independent A/B verification incomplete')
    evidence=[a.group_a/'manifest.json',a.group_a/'report.json',a.group_b/'report.json',
              a.index_run/'report.json',a.audit/'report.json',a.audit/'actual_holdouts.sqlite',a.group_b/'ownership.sqlite']
    code=[HERE/'run_nabla.py',HERE.parent/'group_b_split/split_pools.py',HERE.parent/'group_b_split/verify_holdout_precedence.py',
          HERE.parent/'group_a_split/process_group_a.py']
    contract={k:str(getattr(a,k).resolve()) for k in ('group_a','group_b','index_run','audit')}
    contract.update(nabla_tests=tests,code_hashes={str(f):sha(f) for f in code},evidence_hashes={str(f):sha(f) for f in evidence},
                    policy={'identity':'full Standard InChIKey','test_precedence':True,'validation_fraction_of_full_nabla_snapshot':.005,
                    'candidate_order':'test exclusions, existing-validation matches, fill from remaining nabla training keys absent from every Group A/B training set',
                    'whole_key_groups':True,'seed':42,'other_training_removals_allowed':False,'conformation_test_molecular_overlap_exception':True,
                    'scaffold_wide_exclusion':False,'unselected_existing_validation_candidates':'exclude; never return to training'})
    a.output.parent.mkdir(parents=True,exist_ok=True);atomic(a.output,contract)
    print('Prepared final nabla contract:',a.output)
