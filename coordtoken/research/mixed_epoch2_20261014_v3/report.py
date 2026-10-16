import json
from integrity import OUT,sha
from pathlib import Path
from prepare import atomic

def main():
 case=json.loads((OUT/'case.json').read_text());old=json.loads((Path(case['parent_run'])/'result.json').read_text());new=json.loads((OUT/'run/result.json').read_text())
 assert new['status']=='complete' and new['examples_seen']==189792028 and new['cumulative_examples_seen']==379584056 and new['final_lr']==0
 assert not new['test_evaluated'] and new['validation']['failures']==0
 result={'status':'complete','test_used':False,'epoch1':old['validation'],'epoch2':new['validation'],'checkpoint_sha256':sha(OUT/'run/model.pt'),'caveat':'One exploratory continuation, same fixed development cohort; no statistical-significance claim. Epoch1 test results were already reported before this extension was requested.'}
 atomic(OUT/'comparison.json',result)
 lines=['# CoordToken second epoch','','| Completed epochs | Validation macro RMSD | Validation micro RMSD |','|---|---:|---:|']
 for epoch,r in [(1,old),(2,new)]:lines.append(f"| {epoch} | {r['validation']['source_macro_rmsd']:.5f} | {r['validation']['mean_rmsd']:.5f} |")
 lines+=['','Eight GPUs; global batch512; fresh seed124 training permutation. Restore epoch1 optimizer/model/RNG; LR immediately returns to5e-5, stays constant through80% of epoch2, then cosine-decays to zero over the final20%.','',result['caveat']]
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
