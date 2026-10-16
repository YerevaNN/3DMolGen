#!/usr/bin/env python3
"""Matched pilot or one-pass DDP training; only the registered train/val cache is read."""
import argparse,hashlib,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from data import Packed,Batches,bucket_order
from model import Config,Tokenizer
from metrics import evaluate
from prepare import atomic
from nabla_data import OriginalNabla
from integrity import verify_sources, sha
from health import LossGuard
from schedule import lr_factor
verify_sources()

def save_torch(path,obj):
    temp=path.with_suffix('.tmp');torch.save(obj,temp);temp.replace(path)

def optimizer(model,lr,fused=True,weight_decay=.01,beta2=.999):
    decay=[];other=[]
    for name,p in model.named_parameters():
        (other if p.ndim==1 or name.endswith('.bias') or 'norm' in name.lower() else decay).append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':weight_decay},{'params':other,'weight_decay':0.}],lr=lr,fused=fused,betas=(.9,beta2))

def rng_state():return {'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int,default=0)
    a=p.parse_args();case=json.loads(a.case.read_text());protocol=json.loads(Path(__file__).with_name('protocol.json').read_text())
    a.data=Path(case['data']);a.indices=Path(case['indices']);a.seed=123;a.full=case['kind']=='mixed_full';a.arm=case['name'];a.global_batch=case['global_batch'];a.target_steps=0
    a.architecture=f"atom{case['bits']}_reg{case['registers']}";a.architecture_file=None
    arm={k:case[k] for k in ['width','layers','lr','global_batch','dropout','bits','registers']}
    arm.update(name=a.arm,mode='atom_only',workspace=0,weight_decay=.01,beta2=.999,clip=.1)
    world=int(os.getenv('WORLD_SIZE','1'));rank=int(os.getenv('RANK','0'));local=int(os.getenv('LOCAL_RANK','0'))
    if a.full and world!=8:raise ValueError('Full training requires the authorized eight GPUs')
    torch.cuda.set_device(local);torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    if world>1:dist.init_process_group('nccl')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    Data=Packed if a.full else OriginalNabla
    train=Data(a.data,'train');val=Data(a.data,'val');index_manifest=json.loads((a.indices/'manifest.json').read_text())
    if index_manifest['status']!='complete' or index_manifest['data_manifest_sha256']!=hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest():raise ValueError('Index/data mismatch')
    cfg=Config(vocab_size=len(train.vocab),width=arm['width'],layers=arm['layers'],workspace=0,dropout=arm['dropout'],bits=arm['bits'],mode=arm['mode'],registers=arm['registers'])
    batch_size=64 if a.full else 128
    assert a.global_batch in [256,512,1024,2048] and a.global_batch%(batch_size*world)==0
    model=Tokenizer(cfg).cuda();opt=optimizer(model,arm['lr'],weight_decay=arm['weight_decay'],beta2=arm['beta2']);accumulation=a.global_batch//(batch_size*world);global_batch=batch_size*world*accumulation
    selected=np.load(a.indices/'order.npy',mmap_mode='r')
    order_path=a.output/'order.npy';a.output.mkdir(parents=True,exist_ok=True)
    manifest={'arm':arm,'config':model.architecture(),'seed':a.seed,'world':world,'batch_size':batch_size,'accumulation':accumulation,'global_batch':global_batch,'full':a.full,'examples':len(selected),
              'data_manifest_sha256':index_manifest['data_manifest_sha256'],'index_manifest_sha256':hashlib.sha256((a.indices/'manifest.json').read_bytes()).hexdigest(),
              'code_hashes':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['train_run.py','data.py','nabla_data.py','model.py','quantizer.py','metrics.py','protocol.json']},
              'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'test_evaluated':False,'optimizer':'selective-decay AdamW fused','precision':'bf16 + FP32 XYZ/quantization; TF32 allowed','training_fraction':1.0,'validation_selection':'fixed development cohort; second full mixed epoch; no test access', 'parent_checkpoint_sha256':index_manifest['parent_checkpoint_sha256'],'shuffle_seed':124,'epoch_number':2,'architecture':a.architecture, 'architecture_file_sha256':hashlib.sha256(a.architecture_file.read_bytes()).hexdigest() if a.architecture_file else None}
    if (a.output/'manifest.json').exists() and json.loads((a.output/'manifest.json').read_text())!=manifest:raise ValueError('Existing run configuration differs')
    if rank==0:
        atomic(a.output/'manifest.json',manifest)
        if not order_path.exists():
            order_path.symlink_to((a.indices/'order.npy').resolve())
    if world>1:dist.barrier()
    order=np.load(order_path,mmap_mode='r')
    if rank==0:atomic(a.output/'order_receipt.json',{'sha256':hashlib.sha256(order_path.read_bytes()).hexdigest(),'rows':len(order)})
    steps=math.ceil(len(order)/global_batch);target_steps=steps
    if not 0<target_steps<=steps:raise ValueError('Invalid target budget')
    start=0;previous_seconds=0.;guard=LossGuard(min_step=0)
    if a.resume and (a.output/'last.pt').exists():
        ckpt=torch.load(a.output/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(ckpt['model']);opt.load_state_dict(ckpt['optimizer']);start=ckpt['step'];previous_seconds=ckpt['training_seconds']
        state=ckpt['rng'][rank];torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda']);guard=LossGuard.from_state_dict(ckpt['health_guard']);del ckpt
    elif (a.output/'last.pt').exists():raise ValueError('Existing checkpoint requires --resume')
    else:
        parent=Path(case['parent_run'])/'last.pt'
        if sha(parent)!=index_manifest['parent_checkpoint_sha256']:raise ValueError('Parent checkpoint changed')
        ckpt=torch.load(parent,map_location='cpu',weights_only=False)
        assert ckpt['step']==370688 and len(ckpt['rng'])==8
        assert ckpt['config']==manifest['config'] and ckpt['vocab']==train.vocab
        model.load_state_dict(ckpt['model'],strict=True);opt.load_state_dict(ckpt['optimizer'])
        state=ckpt['rng'][rank];torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda'])
        guard.recent=index_manifest['parent_last_losses'][-10:];guard.best_median=float(np.median(guard.recent))
        del ckpt

    wrapped=DDP(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True,bucket_cap_mb=64) if world>1 else model
    loader=DataLoader(Batches(train,order,batch_size,rank,world,start*accumulation,accumulation),batch_size=None,num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=4,generator=torch.Generator().manual_seed(a.seed+17))
    validation=np.load(a.indices/'validation.npy');metrics=(a.output/f'metrics_rank{rank}.jsonl').open('a',buffering=1)
    started=time.monotonic();excluded=0.;last_step=start;latest_eval=None
    def checkpoint(step,seconds):
        states=[None]*world
        if world>1:dist.all_gather_object(states,rng_state())
        else:states=[rng_state()]
        if rank==0:save_torch(a.output/'last.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'config':manifest['config'],'vocab':train.vocab,'step':step,'rng':states,'training_seconds':seconds,'manifest':manifest,'health_guard':guard.state_dict()})
        if rank==0:
            retained=a.output/f'recovery_{step:09d}.pt'
            if not retained.exists():os.link(a.output/'last.pt',retained)
            for old in sorted(a.output.glob('recovery_*.pt'))[:-3]:old.unlink()
            if step==math.floor(.8*len(order)/global_batch):
                pinned=a.output/'pre_decay_epoch2.pt'
                if not pinned.exists():os.link(a.output/'last.pt',pinned)
                atomic(a.output/'pre_decay_epoch2.json',{'step':step,'epoch_number':2,'examples_seen':step*global_batch,'cumulative_examples_seen':189792028+step*global_batch,'sha256':sha(pinned),'lr':opt.param_groups[0]['lr']})
        if world>1:dist.barrier()
    try:
        for batch in loader:
            microstep=int(batch.pop('step'));step=microstep//accumulation;global_atoms=int(batch.pop('global_atoms'));examples=int(batch.pop('global_examples'))
            batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
            factor=lr_factor(min((step+1)*global_batch,len(order)),len(order))
            for group in opt.param_groups:group['lr']=arm['lr']*factor
            wrapped.train()
            if microstep%accumulation==0:
                opt.zero_grad(set_to_none=True);step_loss=0
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred,_=wrapped(**batch)
                # DDP averages gradients. CPU metadata supplies the global atom
                # denominator exactly, avoiding a per-step count collective.
                loss=((pred-batch['coords']).square()*batch['atoms'].unsqueeze(-1)).sum()*world/(3*max(1,global_atoms))
            loss.backward();step_loss=step_loss+loss.detach()
            if (microstep+1)%accumulation and microstep+1<math.ceil(len(order)/(batch_size*world)):continue
            norm=torch.nn.utils.clip_grad_norm_(wrapped.parameters(),arm['clip'],error_if_nonfinite=True);opt.step();last_step=step+1
            if last_step%100==0 or last_step==steps:
                if not bool(torch.isfinite(step_loss)&torch.isfinite(norm)):raise FloatingPointError('Nonfinite loss/gradient')
                logged=step_loss.clone()
                if world>1:dist.all_reduce(logged);logged/=world
                event={'step':last_step,'examples_seen':min(last_step*global_batch,len(order)),'loss':float(logged),'gradient_norm':float(norm),'lr':opt.param_groups[0]['lr'],'epoch_number':2,'cumulative_examples_seen':189792028+min(last_step*global_batch,len(order)),'training_seconds':previous_seconds+time.monotonic()-started-excluded,'mean_tokens':batch['tokens'].shape[1]}
                event['health']=guard.observe(last_step,float(logged),float(norm))
                if not event['health']['healthy']:raise FloatingPointError('Sustained training loss deterioration')
                metrics.write(json.dumps(event)+'\n')
                if rank==0:atomic(a.output/'progress.json',{'status':'training','total_steps':steps,**event});print(json.dumps(event),flush=True)
            # Verify cross-rank parameter identity early; fail before a long run.
            if a.full and last_step in {100,200}:
                checks=torch.stack([sum(p.detach().float().sum() for p in model.parameters()),sum(p.detach().float().square().sum() for p in model.parameters())]);gathered=[torch.empty_like(checks) for _ in range(world)];dist.all_gather(gathered,checks)
                if not all(torch.equal(gathered[0],x) for x in gathered):raise RuntimeError('DDP parameter checksums diverged')
                if rank==0:atomic(a.output/f'ddp_gate_{last_step}.json',{'status':'passed','world':world,'step':last_step,'resumed_at_step':start,'seconds':time.monotonic()-started,'checks':checks.tolist()})
            checkpoints={math.ceil(f*len(order)/global_batch) for f in [.25,.5,.75,.9,1.]}
            evaluate_now=last_step in checkpoints or last_step==target_steps
            checkpoint_now=last_step==math.floor(.8*len(order)/global_batch) or last_step%(5000 if a.full else 1000)==0 or last_step==target_steps or (a.stop_after and last_step>=a.stop_after)
            if evaluate_now:
                torch.cuda.synchronize();t=time.monotonic()
                # A single fixed validation cohort; other ranks wait only here.
                if rank==0:
                    latest_eval,errors=evaluate(model,val,validation)
                    if latest_eval['failures']:raise FloatingPointError('Nonfinite validation outputs')
                    metrics.write(json.dumps({'step':last_step,'validation':latest_eval})+'\n');np.save(a.output/'validation_rmsd.npy',errors);np.save(a.output/f'validation_rmsd_{last_step}.npy',errors)
                    atomic(a.output/'validation.json',{'step':last_step,'examples_seen':min(last_step*global_batch,len(order)),**latest_eval});atomic(a.output/f'validation_{last_step}.json',{'step':last_step,'examples_seen':min(last_step*global_batch,len(order)),**latest_eval})
                if world>1:dist.barrier()
                excluded+=time.monotonic()-t
            if checkpoint_now:
                torch.cuda.synchronize();t=time.monotonic();seconds=previous_seconds+t-started-excluded;checkpoint(last_step,seconds);excluded+=time.monotonic()-t
            if last_step>=target_steps or (a.stop_after and last_step>=a.stop_after):break
        torch.cuda.synchronize();elapsed=previous_seconds+time.monotonic()-started-excluded
        if rank==0:
            result={'status':'complete' if last_step==target_steps else 'stopped','arm':a.arm,'seed':a.seed,'steps':last_step,'examples_seen':min(last_step*global_batch,len(order)),'total_steps':steps,'training_seconds':elapsed,'ms_per_step':1000*elapsed/max(1,last_step),'validation':latest_eval,'test_evaluated':False,'manifest_sha256':hashlib.sha256((a.output/'manifest.json').read_bytes()).hexdigest(),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'epoch_number':2,'cumulative_examples_seen':189792028+min(last_step*global_batch,len(order)),'final_lr':opt.param_groups[0]['lr']}
            atomic(a.output/'result.json',result);atomic(a.output/f'result_{last_step}.json',result);atomic(a.output/'progress.json',result)
            save_torch(a.output/'model.pt',{'model':model.state_dict(),'config':manifest['config'],'vocab':train.vocab,'manifest':manifest})
    except (FloatingPointError, RuntimeError) as exc:
        if rank==0 and (isinstance(exc,FloatingPointError) or 'non-finite' in str(exc).lower() or 'nonfinite' in str(exc).lower()):
            atomic(a.output/'unstable.json',{'status':'stopped_unstable','step':last_step,'reason':str(exc),'test_used':False})
        raise
    finally:
        metrics.close()
        if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
