"""Promote only a freshly verified deduplicated release; publish its statistics."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import time
import deduplicate as d
import summarize_aliases


def git(repo,*args):
 return subprocess.run(['git',*args],cwd=repo,check=True,text=True,capture_output=True).stdout.strip()


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--release',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True)
 ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--paper',type=Path,required=True);args=ap.parse_args()
 out=args.release;m=d.read(out/'manifest.json');a=d.read(args.audit/'report.json');aplan=d.read(args.audit/'plan.json')
 if m['status']!='complete_verified_merge' or not m['exact_sample_deduplication_performed']:raise ValueError('Deduplication incomplete')
 if a['status']!='complete_verified_exact_duplicate_audit':raise ValueError('Final audit incomplete')
 if d.sha(args.audit/'plan.json')!=a['plan_sha256'] or d.sha(out/'manifest.json')!=aplan['release_manifest_sha256']:raise ValueError('Final audit provenance mismatch')
 if a['split_pair_shared_texts']:raise ValueError('Exact cross-split leakage')
 if any(a['within_category_redundant_rows'].values()):raise ValueError('Remaining within-dataset duplicates')
 if any(a['within_split_redundant_rows'].get(r,0) for r in ('train','val')):raise ValueError('Remaining training/validation duplicates')
 ref=next(c['id'] for c in a['categories'] if c['dataset']=='GEOM-Revisited')
 if any(ref not in map(int,key.split('|')) for key in a['category_pair_shared_texts']):raise ValueError('Remaining duplicates within merged test')
 # Only intended intersections between the separate GEOM benchmark and test
 # samples may remain in the overall benchmark inventory.
 benchmark_overlap=sum(a['category_pair_shared_texts'].values())
 if a['redundant_rows_global']!=benchmark_overlap or a['duplicate_text_groups']!=benchmark_overlap:raise ValueError('Unexpected residual duplicate groups')
 stats=summarize_aliases.summarize(out)
 status={'status':'ready_deduplicated_release','created_unix':time.time(),'manifest':str(out/'manifest.json'),
   'manifest_sha256':d.sha(out/'manifest.json'),'post_deduplication_exact_audit':str(args.audit/'report.json'),
   'post_deduplication_exact_audit_sha256':d.sha(args.audit/'report.json'),
   'source_statistics':str(out/'duplicate_source_statistics.json'),
   'source_statistics_sha256':d.sha(out/'duplicate_source_statistics.json'),
   'duplicates_within_merged_train':0,'duplicates_within_merged_val':0,'duplicates_within_merged_test':0,
   'exact_cross_split_overlap':0,'separate_geom_revisited_test_overlap':benchmark_overlap,
   'removed_rows':m['removed_rows'],'files':m['files'],'duplicate_aliases':m['duplicate_aliases'],
   'scope':'Deduplicated CSV data release; trainer-specific packing and smoke training are separate.'}
 d.atomic(out/'release_status.json',status)
 checks=[f"{f['sha256']}  {Path(f['path']).name}" for f in [*m['files'],m['duplicate_aliases']]]
 for name in ['sources.json','manifest.json','release_status.json','duplicate_source_statistics.json']:
  checks.append(f'{d.sha(out/name)}  {name}')
 (out/'SHA256SUMS').write_text('\n'.join(checks)+'\n')
 summary=['# Deduplicated release','',f"Status: {status['status']}",'',
  '| Split | Retained rows | Removed exact repeats |','| --- | ---: | ---: |']
 for f in m['files']:
  role=Path(f['path']).stem.removeprefix('merged_')
  summary.append(f"| {role} | {f['rows']:,} | {m['removed_rows'][role]:,} |")
 summary+=['','All three merged splits have zero exact duplicate strings internally and zero exact cross-split overlap. GEOM-Revisited remains a separate benchmark; '+f'{benchmark_overlap:,} of its samples also occur in the GEOM test set.','',
 '| Split | Removed row matched same source file | Different file, same dataset | Different dataset |',
 '| --- | ---: | ---: | ---: |']
 for role,c in stats['by_split_relationship'].items():
  summary.append('| '+role+' | '+' | '.join(f"{c.get(k,0):,}" for k in ['same_prepared_source_file','different_source_file_same_dataset','different_dataset'])+' |')
 summary+=['','Each removed row is classified relative to the retained first occurrence. These counts include every removed row, not just distinct shared molecular strings.','',
 '| Split | Removed dataset | Retained dataset | Removed rows |','| --- | --- | --- | ---: |']
 for p in stats['removed_to_retained_dataset_pairs']:
  if p['removed_dataset']!=p['retained_dataset']:
   summary.append(f"| {p['split']} | {p['removed_dataset']} | {p['retained_dataset']} | {p['removed_rows']:,} |")
 summary+=['','`duplicate_aliases.csv` maps every removed record to its retained record while preserving its original source metadata. `sources.json` maps source IDs to prepared input files. These indices are not raw upstream SDF/pickle indices.','']
 (out/'DEDUPLICATION_SUMMARY.md').write_text('\n'.join(summary))
 latest=out.parent/'merged_final';tmp=out.parent/'merged_final.tmp'
 if latest.exists() and not latest.is_symlink():raise ValueError('Refusing to replace non-symlink release pointer')
 if tmp.exists() or tmp.is_symlink():raise ValueError('Temporary release pointer already exists')
 tmp.symlink_to(out,target_is_directory=True);tmp.replace(latest)
 submission=out.parent/'merged_deduplicated_20261006_submission.json'
 if submission.exists():
  s=d.read(submission);s.update(status=status['status'],release_status=str(out/'release_status.json'));d.atomic(submission,s)
 # Update the methods and dataset table only after verification passes.
 categories=d.read(out/'category_manifest.json');counts={};totals=Counter()
 for f in categories['files']:
  counts.setdefault(f['dataset'],{})[f['role']]=f['rows'];totals[f['role']]+=f['rows']
 reference=categories['geom_revisited_reference'];totals['test']+=reference['rows']
 labels={'BindingMoad':'BindingMOAD','OMol25_bio_mols':'OMol25-bio-mols','OMol25_small_mols':'OMol25-small-mols',
  'chembl3d':'ChEMBL3D','geom':'GEOM','pubchem3d':'PubChem3D','zinc':'ZINC','nablaDFT':'∇²DFT',
  'nablaDFT_conformations':'∇²DFT — test conformations','nablaDFT_scaffolds':'∇²DFT — test scaffolds','nablaDFT_structures':'∇²DFT — test structures'}
 table=['| Dataset | Train | Validation | Test |','| --- | ---: | ---: | ---: |']
 for dataset,c in counts.items():table.append('| '+labels.get(dataset,dataset)+' | '+' | '.join(f'{c[r]:,}' if r in c else '—' for r in ['train','val','test'])+' |')
 table+=[f"| GEOM-Revisited | — | — | {reference['rows']:,} |",'| **Total (including GEOM-Revisited)** | '+' | '.join(f'**{totals[r]:,}**' for r in ['train','val','test'])+' |']
 paragraph='Finally, we removed exact duplicate enriched molecular strings within each merged split, retaining the first occurrence and preserving all source identities in a provenance mapping. GEOM-Revisited was retained as a separate benchmark.'
 md=args.repo/'data_collection/paper_dataset_split_revision.md';text=md.read_text()
 start=text.index('| Dataset | Train | Validation | Test |');end=text.find('\n\n',start)
 if end<0:end=len(text)
 text=text[:start]+'\n'.join(table)+text[end:]
 text=text.replace('Counts are retained samples/conformers before exact-sample deduplication.',
                   'Counts are retained samples/conformers after exact-sample deduplication within the merged splits.')
 if paragraph not in text:text=text.replace('## Filtered dataset sizes',paragraph+'\n\n## Filtered dataset sizes')
 md.write_text(text)
 tex=args.repo/'data_collection/paper_dataset_split_revision.tex'
 for p in [tex,args.paper]:
  text=p.read_text();anchor='All random selections during split construction used a fixed seed of 42.'
  if paragraph not in text:
   if text.count(anchor)!=1:raise ValueError('Manuscript insertion anchor changed')
   p.write_text(text.replace(anchor,anchor+'\n\n'+paragraph))
 evidence=args.repo/'data_collection/exact_deduplication/evidence';evidence.mkdir(exist_ok=True)
 d.atomic(evidence/'deduplicated_20261006.json',status)
 d.atomic(evidence/'source_statistics_20261006.json',stats)
 readme=args.repo/'data_collection/exact_deduplication/README.md'
 extra='\n## Completed release: 2026-10-06\n\n'+ '\n'.join(summary[2:])+ '\nThe materialization manifest records the initial audit-pending state.\n`release_status.json` is the final readiness record after the independent audit.\n\nRelease: `'+str(out)+'`\n\n[Completion evidence](evidence/deduplicated_20261006.json) and\n[source statistics](evidence/source_statistics_20261006.json).\n'
 if '## Completed release: 2026-10-06' not in readme.read_text():readme.write_text(readme.read_text()+extra)
 paths=['data_collection/paper_dataset_split_revision.md','data_collection/paper_dataset_split_revision.tex',
        'data_collection/exact_deduplication/README.md','data_collection/exact_deduplication/evidence/deduplicated_20261006.json',
        'data_collection/exact_deduplication/evidence/source_statistics_20261006.json']
 if git(args.repo,'branch','--show-current')!='data-collection-190m':raise ValueError('Unexpected publication branch')
 git(args.repo,'diff','--check');git(args.repo,'add',*paths)
 git(args.repo,'commit','--only',*paths,'-m','Publish deduplicated release, source statistics, and updated paper counts')
 commit=git(args.repo,'rev-parse','HEAD')
 try:
  git(args.repo,'-c','core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=20','push','git@github.com:YerevaNN/CoordToken.git','HEAD:data-collection-190m')
  publication={'status':'published','commit':commit}
 except Exception as e:
  publication={'status':'data_ready_publication_retry_needed','commit':commit,'error':repr(e)}
 d.atomic(out/'publication_status.json',publication)
 print(json.dumps({'release_status':status,'publication':publication}),flush=True)


if __name__=='__main__':main()
