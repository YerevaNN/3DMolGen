"""Audit the paired full-test outputs and report the fixed-checkpoint comparison."""
import csv, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/mnt/weka/fgeikyan/CoordToken')
OUT = ROOT/'logs/paper_runs_20261013'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    folders = [OUT/'tokenizer_test', ROOT/'logs/mixed_epoch2_test_20261015']
    results = [json.loads((p/'result.json').read_text()) for p in folders]
    old, new = results
    assert all(r['status']=='complete' and r['rows']==5553421 and not r['used_for_selection'] and r['parse_or_prediction_failures']==0 for r in results)
    assert old['source_sha256']==new['source_sha256']
    assert old['checkpoint_sha256']=='c3ee867ab22d5893a8a3fbba894657279301ce2add16e1e862966b2046118a41'
    assert new['checkpoint_sha256']=='29d487cf3783978c801c5a237e3c3c86cc23ab058aa7edea7eff0aa707c4bc4f'
    freeze=json.loads(Path(__file__).with_name('new_test_freeze.json').read_text())
    for name,digest in freeze['files'].items():
        assert sha(Path(__file__).with_name(name))==digest, name
    previous_audit=json.loads((OUT/'tests_audit.json').read_text())['evidence']['tokenizer_test']
    assert sha(folders[0]/'result.json')==previous_audit['result_sha256']
    assert sha(folders[0]/'per_conformer.csv')==previous_audit['per_conformer_sha256']
    counts=Counter();sums=[defaultdict(float),defaultdict(float)];n=0
    with (folders[0]/'per_conformer.csv').open(newline='') as a,(folders[1]/'per_conformer.csv').open(newline='') as b:
        for x,y in zip(csv.DictReader(a),csv.DictReader(b),strict=True):
            assert all(x[k]==y[k] for k in ['row','name','dataset_regime'])
            assert int(x['row'])==n
            group=x['dataset_regime'];counts[group]+=1;n+=1
            for i,row in enumerate([x,y]):
                value=float(row['rmsd_angstrom']);assert math.isfinite(value) and value>=0
                sums[i][group]+=value
    assert n==5553421
    for i,r in enumerate(results):
        assert set(counts)==set(r['per_dataset_regime'])
        for group,count in counts.items():
            m=r['per_dataset_regime'][group]
            assert m['rows']==count and math.isclose(sums[i][group]/count,m['mean_rmsd'],rel_tol=1e-10,abs_tol=1e-12)
        assert math.isclose(sum(sums[i].values())/n,r['pooled']['mean_rmsd'],rel_tol=1e-10,abs_tol=1e-12)
    assert all('nablaDFT / '+regime in counts for regime in ['conformations','structures','scaffolds'])
    rows=[{'dataset_regime':g,'rows':counts[g],'old_rmsd':sums[0][g]/counts[g],'new_rmsd':sums[1][g]/counts[g]} for g in sorted(counts)]
    rows.append({'dataset_regime':'Micro average','rows':n,'old_rmsd':old['pooled']['mean_rmsd'],'new_rmsd':new['pooled']['mean_rmsd']})
    macro=[]
    for i in range(2):
        source_sums=defaultdict(float);source_counts=Counter()
        for g,nrows in counts.items():
            source='nablaDFT' if g.startswith('nablaDFT / ') else g
            source_sums[source]+=sums[i][g];source_counts[source]+=nrows
        assert len(source_sums)==19
        macro.append(sum(source_sums[g]/source_counts[g] for g in source_sums)/19)
    rows.append({'dataset_regime':'Macro average','rows':0,'old_rmsd':macro[0],'new_rmsd':macro[1]})
    for row in rows:row['new_minus_old']=row['new_rmsd']-row['old_rmsd']
    dest=folders[1]
    with (dest/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    receipt={'status':'passed','rows':n,'same_test_rows_verified':True,'rmsd_aggregates_recomputed':True,'old_checkpoint_sha256':old['checkpoint_sha256'],'new_checkpoint_sha256':new['checkpoint_sha256'],'source_sha256':new['source_sha256'],'new_per_conformer_sha256':sha(dest/'per_conformer.csv'),'new_result_sha256':sha(dest/'result.json'),'test_used_for_selection':False,'per_dataset':rows}
    (dest/'comparison_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    lines=['# Full-test comparison','','Same 5,553,421 test conformations; fixed final checkpoints, no test-based tuning. Unaligned per-conformer RMSD in Å; lower is better.','','| Dataset / regime | Test conformers | Epoch1 | Epoch2 |','|---|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['dataset_regime']} | {row['rows']:,} | {row['old_rmsd']:.5f} | {row['new_rmsd']:.5f} |")
    lines+=['','Both models use the same 189,792,028 training rows: one versus two full passes. Epoch2 restarts LR5e-5 and decays over its final20%, with a new shuffle. The conformation regime intentionally permits molecular overlap with training. GEOM-Revisited is excluded.','', 'Every paired row matched; RMSD aggregates recomputed. Both evaluations have zero parsing/prediction failures.']
    (dest/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print('Full-test comparison audit passed',flush=True)

if __name__=='__main__':main()
