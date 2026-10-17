"""Summarize removed-row source relationships from the verified alias ledger."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import deduplicate as d


def summarize(release):
 manifest=d.read(release/'manifest.json')
 sources={s['source_file_id']:s for s in d.read(release/'sources.json')['files']}
 path=Path(manifest['duplicate_aliases']['path']);h=hashlib.sha256()
 relationships=defaultdict(Counter);datasets=Counter();pairs=Counter();examples={};n=0
 with path.open('rb') as f:
  header=f.readline();h.update(header)
  if next(csv.reader([header.decode()]))!=d.ALIAS_HEADER:raise ValueError('Unexpected alias schema')
  for raw in f:
   h.update(raw);row=dict(zip(d.ALIAS_HEADER,next(csv.reader([raw.decode()],strict=True))));n+=1
   kept_source=row['kept_name'].rsplit(':',1)[0]
   owner=sources[kept_source];removed=sources[row['source_file_id']]
   if row['source_file_id']==kept_source:relationship='same_prepared_source_file'
   elif removed['output_dataset']==owner['output_dataset']:relationship='different_source_file_same_dataset'
   else:relationship='different_dataset'
   split=row['split'];relationships[split][relationship]+=1
   datasets[(split,row['dataset'],row['test_subset'])]+=1
   pair=(split,row['dataset'],row['test_subset'],owner['output_dataset'],owner['test_subset'])
   pairs[pair]+=1
   if pair not in examples:
    examples[pair]={'removed':row,'retained_source_file':owner['path'],
                    'retained_source_row_index':int(row['kept_name'].rsplit(':',1)[1])}
 if n!=manifest['duplicate_aliases']['rows'] or h.hexdigest()!=manifest['duplicate_aliases']['sha256']:raise ValueError('Alias integrity mismatch')
 for split,counts in relationships.items():
  if sum(counts.values())!=manifest['removed_rows'][split]:raise ValueError('Removal statistics mismatch')
 result={'status':'complete_verified_source_statistics','removed_rows':n,
  'by_split_relationship':{k:dict(v) for k,v in relationships.items()},
  'by_removed_dataset':[{'split':k[0],'dataset':k[1],'test_subset':k[2],'removed_rows':v}
                        for k,v in sorted(datasets.items(),key=lambda kv:-kv[1])],
  'removed_to_retained_dataset_pairs':[{'split':k[0],'removed_dataset':k[1],'removed_test_subset':k[2],
    'retained_dataset':k[3],'retained_test_subset':k[4],'removed_rows':v,'example':examples[k]}
    for k,v in sorted(pairs.items(),key=lambda kv:-kv[1])],
  'interpretation':'Each removed row is classified relative to its retained first occurrence. Different-dataset row counts can include repeated rows within the removed dataset; they are not counts of distinct shared texts.',
  'ownership_policy':manifest['deduplication_policy'],'alias_sha256':h.hexdigest()}
 d.atomic(release/'duplicate_source_statistics.json',result);return result


if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('release',type=Path);args=ap.parse_args()
 r=summarize(args.release);print(json.dumps({k:r[k] for k in ['status','removed_rows','by_split_relationship']},indent=2))
