"""Bounded input/memory/forward/backward readiness checks; never optimize.

CUDA is required unless --allow-cpu is explicitly supplied for local checks.
No optimizer is constructed, no model is saved, and no test split is scored.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
from pathlib import Path
import random
import threading
import time

import numpy as np
import psutil
import torch

from unisar.batch import collate_requests, to_device
from unisar.common import digest, object_hash, read_json, write_json
from unisar.data.dataset import PreparedDataset
from unisar.data.temporal import FEATURE_POLICY
from unisar.losses import joint_loss
from unisar.model import ModelConfig, UniSAR
from unisar.training import load_config, microbatches, experiment_hash, software_provenance

GIB=1024**3


def check_dataset_contract(dataset):
    manifest=dataset.manifest
    without_hash={key:value for key,value in manifest.items() if key!='data_hash'}
    if object_hash(without_hash) != dataset.data_hash:
        raise ValueError('Prepared manifest identity mismatch')
    if manifest['vocabulary_hash'] != object_hash(dataset.vocab):
        raise ValueError('Vocabulary identity mismatch')
    policy=read_json(dataset.root/'feature-policy.json')
    if object_hash(policy) != manifest['feature_policy_hash'] or object_hash(policy) != object_hash(FEATURE_POLICY):
        raise ValueError('Feature policy differs from the supported temporal contract')
    if (dataset.item_count != len(dataset.vocab['items'])+2 or
        dataset.token_count != len(dataset.vocab['tokens'])+2 or
        dataset.vocab.get('pad') != 0 or dataset.vocab.get('oov') != 1):
        raise ValueError('Vocabulary shape or PAD/OOV contract mismatch')
    for name in ('items','tokens'):
        values=dataset.vocab[name]
        if any(type(value) is not int or value < 0 for value in values) or any(a>=b for a,b in zip(values,values[1:])):
            raise ValueError(f'{name} vocabulary must contain strictly ordered unique nonnegative IDs')


def _representative_batch(dataset, task, size, seed):
    view=dataset.requests('train',task)
    if not len(view):
        raise ValueError('Preflight requires both nonempty training tasks')
    rng=random.Random(seed)
    chosen=rng.sample(range(len(view)),min(size,len(view)))
    longest_candidates=max(range(len(view)),key=lambda i:len(dataset.raw_requests[view.indexes[i]]['item_raw']))
    longest_history=max(range(len(view)),key=lambda i:len(dataset.raw_requests[view.indexes[i]]['history_ids']))
    chosen[0]=longest_candidates
    if len(chosen)>1:
        chosen[1]=longest_history
    examples=[]
    for index in chosen:
        example=view[index]
        if task=='recommendation':
            column=rng.randrange(len(example['item_ids']))
            example=dict(example)
            for key in ('item_ids','item_raw','caption_ids','metadata_present','labels'):
                if key in example:
                    example[key]=[example[key][column]]
        examples.append(example)
    sizes=np.fromiter((len(dataset.raw_requests[i]['item_raw']) for i in view.indexes),dtype=np.int64)
    return examples,{
        'requests':len(view),'recorded_exposures':int(sizes.sum()),
        'largest_recorded_group':int(sizes.max()),'candidate_group_p95':float(np.quantile(sizes,.95)),
        'probe_requests':len(examples),'probe_candidates':sum(len(row['item_ids']) for row in examples),
        'sampling':'fixed-seed request sample plus largest candidate group and longest-history request; rec reduced to one observed exposure per request',
    }


def _check_encoded_ids(batch,architecture):
    for key,count in (
        ('item_ids',architecture.item_count),('history_rec_ids',architecture.item_count),
        ('caption_ids',architecture.token_count),('query_ids',architecture.token_count),
        ('history_query_ids',architecture.token_count),('user_ids',architecture.user_count)):
        values=batch[key]
        if values.numel() and (int(values.min())<0 or int(values.max())>=count):
            raise ValueError(f'{key} falls outside model vocabulary')
    valid=batch['history_mask']
    if valid.any() and torch.any(batch['history_timestamp'][valid] >= batch['request_timestamp'][:,None].expand_as(valid)[valid]):
        raise ValueError('Noncausal history reached the model')


def run_preflight(data:Path|str,config_path:Path|str,allow_cpu=False):
    start=time.perf_counter()
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError('CUDA unavailable. In Colab select Runtime > Change runtime type > GPU, then rerun this cell. Local CPU checks require --allow-cpu.')
    overrides={'device':'cpu','amp':False} if not torch.cuda.is_available() else {}
    config=load_config(config_path,**overrides)
    if config.device!='cuda' and not allow_cpu:
        raise ValueError('Cloud preflight requires device: cuda; --allow-cpu permits explicit local checks')
    config.check()
    torch.set_num_threads(config.threads)
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    process=psutil.Process()
    peak=[process.memory_info().rss]
    stop=threading.Event()
    def monitor():
        while not stop.wait(.05):
            peak[0]=max(peak[0],process.memory_info().rss)
    sampler=threading.Thread(target=monitor,daemon=True);sampler.start()
    try:
        available_before=psutil.virtual_memory().available
        dataset=PreparedDataset(data)
        check_dataset_contract(dataset)
        gc.collect()
        available_after=psutil.virtual_memory().available
        architecture=ModelConfig(item_count=dataset.item_count,token_count=dataset.token_count,**config.model)
        # Build shapes on the meta device before allocating large tables.
        with torch.device('meta'):
            blueprint=UniSAR(architecture)
        parameters=sum(value.numel() for value in blueprint.parameters())
        weight_bytes=sum(value.numel()*value.element_size() for value in blueprint.parameters())
        del blueprint
        if config.device=='cpu' and available_after < 2*weight_bytes+GIB/2:
            raise MemoryError('Insufficient free host memory for weights, gradients and a safety margin. Use a larger-memory runtime or the separately prepared pilot configuration; do not silently truncate data.')
        gpu=None
        if config.device=='cuda':
            free,total=torch.cuda.mem_get_info()
            gpu={'name':torch.cuda.get_device_name(),'total_bytes':total,'free_before_bytes':free}
            if free < 4*weight_bytes+GIB/2:
                raise MemoryError('Insufficient free GPU memory for FP32 weights/gradients/Adam state plus margin. Reduce the documented architecture or batch budget and rerun preflight.')
            torch.cuda.reset_peak_memory_stats()
        model=UniSAR(architecture).to(config.device)
        model.train()
        rss_loaded=process.memory_info().rss
        probes={}
        for task_code,task in enumerate(('search','recommendation')):
            examples,stats=_representative_batch(dataset,task,config.batch_size,config.seed+task_code)
            results=[]
            task_start=time.perf_counter()
            model.zero_grad(set_to_none=True)
            for part in microbatches(examples,config.max_padded_candidates):
                batch=collate_requests(part,model.max_history)
                _check_encoded_ids(batch,architecture)
                tensor_bytes=sum(value.numel()*value.element_size() for value in batch.values() if isinstance(value,torch.Tensor))
                batch=to_device(batch,config.device)
                if config.device=='cuda':torch.cuda.synchronize()
                probe_start=time.perf_counter()
                with torch.autocast(device_type=config.device,dtype=torch.float16,enabled=config.amp):
                    output=model.forward_with_aux(batch,candidate_chunk_size=config.candidate_chunk_size)
                    losses=joint_loss(output,batch,config.alignment_weight,config.transition_weight,config.temperature)
                if output['logits'].shape != batch['candidate_mask'].shape or not torch.isfinite(output['logits']).all():
                    raise RuntimeError('Nonfinite or candidate-misaligned model output')
                if any(not torch.isfinite(value).all() for value in losses.values()):
                    raise RuntimeError('Nonfinite supervised or contrastive loss')
                (losses['total']*(len(part)/len(examples))).backward()
                if any(value.grad is not None and not torch.isfinite(value.grad).all() for value in model.parameters()):
                    raise RuntimeError('Nonfinite gradient in readiness probe')
                if config.device=='cuda':torch.cuda.synchronize()
                results.append({
                    'requests':len(part),'valid_candidates':int(batch['candidate_mask'].sum()),
                    'padded_candidates':batch['item_ids'].numel(),
                    'batch_tensor_bytes':tensor_bytes,'forward_backward_seconds':time.perf_counter()-probe_start,
                    'losses':{key:float(value.detach()) for key,value in losses.items()},
                    'finite_logits_losses_gradients':True,
                })
                del batch,output,losses
            probes[task]={**stats,'microbatches':results,'elapsed_seconds':time.perf_counter()-task_start,
                'largest_group_exceeds_padding_budget':stats['largest_recorded_group']>config.max_padded_candidates,
                'training_padding_budget':config.max_padded_candidates,'candidate_chunk_size':config.candidate_chunk_size}
            model.zero_grad(set_to_none=True)
        if gpu is not None:
            gpu.update({'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        result={
            'status':'passed','optimizer_updates':0,'test_predictions_or_metrics':False,
            'device':config.device,'cuda_training_preflight_verified':config.device=='cuda',
            'data_hash':dataset.data_hash,'vocabulary_hash':dataset.manifest['vocabulary_hash'],
            'feature_policy_hash':dataset.manifest['feature_policy_hash'],
            'config_sha256':digest(Path(config_path)),'effective_config_hash':experiment_hash(config),
            'software':software_provenance(),
            'architecture':asdict(architecture),'parameters':parameters,'fp32_weight_bytes':weight_bytes,
            'fp32_weights_gradients_adam_states_minimum_bytes':4*weight_bytes,
            'host_memory':{'available_before_data_bytes':available_before,'available_after_data_bytes':available_after,
                           'rss_with_data_model_bytes':rss_loaded,'sampled_peak_rss_bytes':max(peak[0],process.memory_info().rss)},
            'gpu':gpu,'probes':probes,'elapsed_seconds':time.perf_counter()-start,
            'limits':[
                'Forward/backward probes never perform optimizer updates; they do not establish learning or convergence.',
                'Largest training search group and long histories are included, but this bounded probe does not cover every batch combination.',
                'Adam memory is a lower bound on FP32 training allocation, not a guarantee; optimizer kernels, activation/workspace and caching add cost.',
                'Probe wall time excludes optimizer updates, periodic validation and checkpoint I/O; no full-run duration or quota is inferred.',
                'CPU success leaves the actual cloud GPU/device quota check pending.' if config.device=='cpu' else 'GPU success concerns this runtime and configuration; a changed device/configuration requires another probe.',
            ],
        }
        if gpu is not None:
            activation=max(0,gpu['peak_allocated_bytes']-2*weight_bytes)
            result['gpu']['observed_activation_workspace_extra_bytes']=activation
            result['gpu']['planning_bytes_including_adam_and_margin']=4*weight_bytes+activation+GIB/2
            if result['gpu']['planning_bytes_including_adam_and_margin'] > gpu['free_before_bytes']:
                raise MemoryError('Measured activation/workspace plus the FP32 Adam plan exceeds available GPU headroom. Reduce the declared batch/model budget and rerun preflight.')
        return result
    finally:
        stop.set();sampler.join()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-cpu',action='store_true')
    args=parser.parse_args()
    try:
        result=run_preflight(args.data,args.config,args.allow_cpu)
    except Exception as error:
        write_json(args.output,{'status':'failed','error_type':type(error).__name__,'message':str(error),
                               'optimizer_updates':0,'next_action':'Correct the input/runtime/configuration and rerun preflight before training.'})
        raise
    write_json(args.output,result)
    print(f"Preflight passed on {result['device']}: {result['parameters']:,} parameters; no optimizer updates.")
    print(f"Weights: {result['fp32_weight_bytes']/GIB:.3f} GiB; FP32 weights/gradients/Adam-state lower bound: {result['fp32_weights_gradients_adam_states_minimum_bytes']/GIB:.3f} GiB.")
    print(f"Read full memory/limits record at {args.output}")


if __name__=='__main__':
    main()
