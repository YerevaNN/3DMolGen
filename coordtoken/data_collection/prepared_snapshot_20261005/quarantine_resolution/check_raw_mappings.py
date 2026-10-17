import sys,json,csv,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'data_collection/rebuild_20260930';OUT=Path(__file__).parent
sys.path[:0]=[str(ROOT),str(BASE/'source_recovery'),str(BASE/'sampled_original_audit')]
from rdkit import Chem,RDLogger
from loqi_aligned_pilot import RestrictedReader
from compare_v2 import decode_serialized_text,atom_features,proper_rmsd,exact_mapping
RDLogger.DisableLog('rdApp.*');results=[]
for split,path,fix in [('val','loqi_references/val_h.pt',3),('train','loqi/train_h.pt',2)]:
 with zipfile.ZipFile(BASE/'downloads'/path) as z:
  with z.open(next(n for n in z.namelist() if n.endswith('/data.pkl'))) as f:raw=RestrictedReader(f).load()
 for row in json.loads((OUT.parent/f'repair_{fix}.json').read_text())['removed']:
  name,text=next(csv.reader([row['raw_csv_row']]))
  a=Chem.RemoveHs(Chem.Mol(raw[0]._store._mapping['mol'][int(name.rsplit('_',1)[1])].binary));b=decode_serialized_text(text);report={'name':name,'maps':{}}
  for stereo in [True,False]:
   candidates=[]
   for mp in b.GetSubstructMatches(a,uniquify=False,useChirality=stereo,maxMatches=10000):
    if any(atom_features(a.GetAtomWithIdx(i))!=atom_features(b.GetAtomWithIdx(j)) for i,j in enumerate(mp)):continue
    rmsd=proper_rmsd(a.GetConformer().GetPositions(),b.GetConformer().GetPositions()[list(mp)]);candidates.append((rmsd,mp))
   if candidates:
    rmsd,mp=min(candidates);report['maps'][str(stereo)]={'count':len(candidates),'best_rmsd_A':rmsd,'mapping':mp,'isotope_constrained_comparator_pass':exact_mapping(a,b,mp)}
  results.append(report);print(name,report['maps'],flush=True)
 del raw
(OUT/'chembl_mapping_recheck.json').write_text(json.dumps(results,indent=2)+'\n')
