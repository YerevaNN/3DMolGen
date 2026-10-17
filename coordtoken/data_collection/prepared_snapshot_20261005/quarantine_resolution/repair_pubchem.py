import sys,json,csv,re,ast,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'data_collection/rebuild_20260930';OUT=Path(__file__).parent
sys.path[:0]=[str(ROOT),str(BASE/'sampled_original_audit')]
from rdkit import Chem,RDLogger
from compare_v2 import legacy_module,compare,proper_rmsd,decode_serialized_text
RDLogger.DisableLog('rdApp.*');COORD=re.compile(r'<([^>]+)>')
def topology(m):
 r=Chem.RWMol()
 for atom in m.GetAtoms():
  a=Chem.Atom(atom.GetAtomicNum());a.SetFormalCharge(atom.GetFormalCharge());a.SetIsotope(atom.GetIsotope());a.SetNoImplicit(True);r.AddAtom(a)
 for bond in m.GetBonds():r.AddBond(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(),Chem.BondType.SINGLE)
 return r.GetMol()
def kekule_encode(m):
 u=legacy_module();m=Chem.RemoveHs(Chem.Mol(m));s=Chem.MolToSmiles(m,canonical=True,isomericSmiles=True,kekuleSmiles=True);order=ast.literal_eval(m.GetProp('_smilesAtomOutputOrder'));expected=[u._expected_plain_token(m.GetAtomWithIdx(i)) for i in order];tokens=u.tokenize_smiles(s,expected_atom_tokens=expected);parts=[];n=0
 for t in tokens:
  if t['type']=='atom':
   desc=t['text'];desc=desc if desc.startswith('[') else '['+desc+']';v=m.GetConformer().GetAtomPosition(order[n]);n+=1;parts.append(desc+'<'+','.join(u.truncate(x,4) for x in [v.x,v.y,v.z])+'>')
  else:parts.append(t['text'])
 assert n==m.GetNumAtoms();return ''.join(parts)
def main():
 rows=json.loads((OUT.parent/'repair_0.json').read_text())['removed'];manifest=json.loads((OUT/'pubchem_download_manifest.json').read_text());sources=[]
 for item in manifest:
  if 'file' not in item:continue
  p=Path(item['file']);assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'];m=Chem.MolFromMolBlock(p.read_text(),sanitize=True,removeHs=False);sources.append((item,Chem.RemoveHs(m)))
 result=[]
 for row in rows:
  name,text=next(csv.reader([row['raw_csv_row']]));cid=int(name.split('_')[0]);u=legacy_module();s=''.join((t['atom_desc'][1:-1] if t['atom_desc'][1:-1] in u._ORGANIC_SUBSET else t['atom_desc']) if t['type']=='atom_with_coords' else t['text'] for t in u.tokenize_enriched(text));old=Chem.MolFromSmiles(s,sanitize=False);a=np.array([[float(v) for v in v.split(',')] for v in COORD.findall(text)]);assert old.GetNumAtoms()==len(a);matches=[]
  for item,m in sources:
   if item['cid']!=cid:continue
   mappings=topology(m).GetSubstructMatches(topology(old),uniquify=False,useChirality=False,maxMatches=1001)
   if len(mappings)>1000:raise ValueError('Mapping bound reached')
   for mapping in mappings:
    if len(mapping)!=m.GetNumAtoms() or old.GetNumBonds()!=m.GetNumBonds():continue
    rmsd=proper_rmsd(a,m.GetConformer().GetPositions()[list(mapping)])
    if rmsd<.001:matches.append((rmsd,item,m,mapping))
  r={'name':name,'original_row_sha256':row['row_sha256'],'source_matches':len(matches),'status':'unresolved'}
  if matches:
   rmsd,item,m,mapping=min(matches,key=lambda x:x[0]);r.update(source=item,rmsd_A=rmsd,atom_mapping=list(mapping))
   for label,encoder in [('canonical_aromatic',lambda x:u.encode_cartesian_v2(x,precision=4)),('canonical_kekule',kekule_encode)]:
    try:
     proposed=encoder(m);c=compare(proposed,[m],tolerance=.001,max_matches=10000)
     r[label]={'graph_match':c['graph_match'],'conformer_match':c['conformer_match'],'status':c['status'],'error':c.get('error')}
     if c['graph_match'] and c['conformer_match']:
      r.update(status='source_backed_repair_verified',replacement_enriched_text=proposed,encoding=label,replacement_key=c['decoded_identity']['key']);break
    except Exception as e:r[label]={'error':str(e)}
  result.append(r);print(name,r['status'],r.get('encoding'),r.get('rmsd_A'),flush=True)
 (OUT/'pubchem_repairs.json').write_text(json.dumps({'scope':'Match same-CID publisher conformers to old element/charge/isotope connectivity and coordinates (proper rotation RMSD <0.001 A), then verify regenerated text against full publisher graph/stereo and geometry. Bond/aromatic annotation changes are source-backed repairs, not asserted old/new graph identity.','rows':result},indent=2)+'\n')
if __name__=='__main__':main()
