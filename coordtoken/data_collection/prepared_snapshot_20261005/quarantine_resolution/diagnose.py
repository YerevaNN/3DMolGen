import sys,json,csv,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'data_collection/rebuild_20260930';OUT=Path(__file__).parent;P=OUT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'data_collection'),str(BASE/'source_recovery'),str(BASE/'sampled_original_audit')]
from rdkit import Chem,RDLogger,rdBase
from rdkit.Chem import rdinchi,rdCIPLabeler
from inchikey_filter import stripped_text,BRACKETS,ORGANIC_ATOMS
from compare_v2 import decode_serialized_text,compare,atom_features
from loqi_aligned_pilot import RestrictedReader
RDLogger.DisableLog('rdApp.*')

def features(m):
 try:rdCIPLabeler.AssignCIPLabels(m)
 except Exception:pass
 return [{'i':b.GetIdx(),'ends':[b.GetBeginAtomIdx(),b.GetEndAtomIdx()],'stereo':str(b.GetStereo()),'refs':list(b.GetStereoAtoms()),'dir':str(b.GetBondDir()),'cip':b.GetProp('_CIPCode') if b.HasProp('_CIPCode') else None} for b in m.GetBonds() if b.GetStereo()!=Chem.BondStereo.STEREONONE or b.GetBondDir()!=Chem.BondDir.NONE]

def diagnose_small():
 out=[]
 for i in [0,1]:
  r=json.loads((P/f'repair_{i}.json').read_text())
  for row in r['removed']:
   name,text=next(csv.reader([row['raw_csv_row']]))
   smi=BRACKETS.sub(lambda m:m[1] if m[1] in ORGANIC_ATOMS else m[0],stripped_text(text));raw=Chem.MolFromSmiles(smi,sanitize=False);item={'name':name,'smiles':smi,'problems':[{'type':x.GetType(),'message':x.Message()} for x in Chem.DetectChemistryProblems(raw)]}
   m=Chem.MolFromSmiles(smi)
   if m:
    inc,code,msg,log,aux=rdinchi.MolToInchi(m);item.update(inchi=inc,return_code=code,message=msg,log=log,atoms=m.GetNumAtoms())
   else:
    m=Chem.Mol(raw)
    try:
     Chem.SanitizeMol(m,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
     item['no_kekulize_smiles']=Chem.MolToSmiles(m)
     inc,code,msg,log,aux=rdinchi.MolToInchi(m);item.update(no_kekulize_inchi=inc,no_kekulize_code=code,no_kekulize_message=msg)
    except Exception as e:item['no_kekulize_error']=str(e)
   out.append(item)
 (OUT/'pubchem_zinc_diagnosis.json').write_text(json.dumps(out,indent=2)+'\n');print('Small diagnosis',[(x['name'],x.get('return_code',x.get('no_kekulize_code')),x.get('message',x.get('no_kekulize_message'))) for x in out],flush=True)

def diagnose_chembl():
 allresults=[]
 for split,filename,fix in [('val','loqi_references/val_h.pt',3),('train','loqi/train_h.pt',2)]:
  with zipfile.ZipFile(BASE/'downloads'/filename) as z:
   with z.open(next(n for n in z.namelist() if n.endswith('/data.pkl'))) as f:root=RestrictedReader(f).load()
  mols=root[0]._store._mapping['mol']
  for row in json.loads((P/f'repair_{fix}.json').read_text())['removed']:
   name,text=next(csv.reader([row['raw_csv_row']]))
   source=Chem.RemoveHs(Chem.Mol(mols[int(name.rsplit('_',1)[1])].binary));decoded=decode_serialized_text(text)
   result={'name':name,'split':split,'source_smiles':Chem.MolToSmiles(source),'decoded_smiles':Chem.MolToSmiles(decoded),'source_stereo':features(Chem.Mol(source)),'decoded_stereo':features(Chem.Mol(decoded)),'variants':{}}
   for mode in ['assigned','directions','new_perception','cip']:
    m=Chem.Mol(source);m.RemoveAllConformers()
    try:
     if mode=='directions':
      for b in m.GetBonds():b.SetStereo(Chem.BondStereo.STEREONONE)
      Chem.SetBondStereoFromDirections(m)
     if mode=='new_perception':Chem.SetUseLegacyStereoPerception(False)
     Chem.AssignStereochemistry(m,cleanIt=True,force=True)
     if mode=='cip':rdCIPLabeler.AssignCIPLabels(m)
     # Attach unchanged source coordinates only for the independent comparison.
     m.AddConformer(Chem.Conformer(source.GetConformer()),assignId=True)
     c=compare(text,[m],tolerance=.001,max_matches=10000)
     result['variants'][mode]={'stereo':features(m),'smiles':Chem.MolToSmiles(m),'graph_match':c['graph_match'],'conformer_match':c['conformer_match'],'rmsd':c['best_heavy_atom_rmsd_angstrom']}
    except Exception as e:result['variants'][mode]={'error':str(e)}
    finally:Chem.SetUseLegacyStereoPerception(True)
   allresults.append(result);print(name,{k:(v.get('graph_match'),v.get('conformer_match')) for k,v in result['variants'].items()},flush=True)
  del mols,root
 (OUT/'chembl_diagnosis.json').write_text(json.dumps(allresults,indent=2)+'\n')
if __name__=='__main__':diagnose_small();diagnose_chembl()
