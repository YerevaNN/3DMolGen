import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import deduplicate as d
import finalize


class FinalizationTests(unittest.TestCase):
 def test_promotion_paper_table_and_publication_after_verified_audit(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);release=root/'release';audit=root/'audit';repo=root/'repo'
   release.mkdir();audit.mkdir();(repo/'data_collection/exact_deduplication').mkdir(parents=True)
   files=[]
   for role in ['train','val','test']:
    p=release/f'merged_{role}.csv';p.write_bytes(d.merge.HEADER_BYTES)
    files.append({'path':str(p),'rows':0,'bytes':p.stat().st_size,'sha256':d.sha(p)})
   aliases=release/'duplicate_aliases.csv';aliases.write_bytes(d.ALIAS_HEADER_BYTES)
   d.atomic(release/'sources.json',{'files':[]})
   manifest={'status':'complete_verified_merge','exact_sample_deduplication_performed':True,
    'files':files,'removed_rows':{'train':0,'val':0,'test':0},'deduplication_policy':'first occurrence',
    'duplicate_aliases':{'path':str(aliases),'rows':0,'bytes':aliases.stat().st_size,'sha256':d.sha(aliases)}}
   d.atomic(release/'manifest.json',manifest)
   d.atomic(release/'category_manifest.json',{'files':[{'dataset':'example','role':'train','rows':0}],
    'geom_revisited_reference':{'rows':0}})
   d.atomic(audit/'plan.json',{'release_manifest_sha256':d.sha(release/'manifest.json')})
   d.atomic(audit/'report.json',{'status':'complete_verified_exact_duplicate_audit','plan_sha256':d.sha(audit/'plan.json'),
    'split_pair_shared_texts':{},'within_category_redundant_rows':{},'within_split_redundant_rows':{},
    'categories':[{'id':1,'dataset':'GEOM-Revisited'}],'category_pair_shared_texts':{},
    'redundant_rows_global':0,'duplicate_text_groups':0})
   anchor='All random selections during split construction used a fixed seed of 42.'
   md=repo/'data_collection/paper_dataset_split_revision.md'
   md.write_text(anchor+'\n\n## Filtered dataset sizes\n\nCounts are retained samples/conformers before exact-sample deduplication.\n\n| Dataset | Train | Validation | Test |\n| --- | ---: | ---: | ---: |\n| old | 1 | 2 | 3 |\n\n[^split-targets]: Keep this footnote.\n')
   (repo/'data_collection/paper_dataset_split_revision.tex').write_text(anchor+'\n')
   paper=root/'paper.tex';paper.write_text(anchor+'\n')
   (repo/'data_collection/exact_deduplication/README.md').write_text('# Deduplication\n')
   def fake_git(repo,*args):
    if args[:2]==('branch','--show-current'):return 'data-collection-190m'
    if args[:2]==('rev-parse','HEAD'):return 'testcommit'
    return ''
   argv=['finalize','--release',str(release),'--audit',str(audit),'--repo',str(repo),'--paper',str(paper)]
   with patch.object(sys,'argv',argv),patch.object(finalize,'git',fake_git),patch('sys.stdout',new_callable=io.StringIO):
    finalize.main()
   self.assertEqual(d.read(release/'release_status.json')['status'],'ready_deduplicated_release')
   self.assertEqual(d.read(release/'publication_status.json')['status'],'published')
   self.assertEqual((root/'merged_final').resolve(),release)
   self.assertIn('| example | 0 | — | — |',md.read_text())
   self.assertIn('Keep this footnote.',md.read_text())
   self.assertIn('removed exact duplicate',paper.read_text())
   self.assertEqual(d.sha(release/'manifest.json'),d.read(audit/'plan.json')['release_manifest_sha256'])

if __name__=='__main__':unittest.main()
