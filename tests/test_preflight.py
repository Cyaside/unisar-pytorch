"""GPU guard and genuine fixture forward/backward with no optimizer steps."""
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from unisar.common import digest,write_json
from unisar.data.audit import CUTOFFS
from unisar.data.dataset import PreparedDataset
from unisar.data.prepare import build
from unisar.preflight import check_dataset_contract,run_preflight


def prepared_fixture(tmp_path,monkeypatch):
    import unisar.data.prepare as preparation
    monkeypatch.setattr(preparation,'ROOT',tmp_path)
    canonical=tmp_path/'canonical';canonical.mkdir()
    t=CUTOFFS[0]
    search=[{'user_raw':1,'session_raw':1,'timestamp_ms':t-3000,'query_signature':'[3]',
             'item_raw':[7,8],'labels':[1,0],'raw_count':2,'retained_raw_count':2},
            {'user_raw':2,'session_raw':2,'timestamp_ms':t-3000,'query_signature':'[5]',
             'item_raw':[7,8],'labels':[0,1],'raw_count':2,'retained_raw_count':2}]
    rec=[{'user_raw':user,'item_raw':item,'timestamp_ms':t-4000,'label':label,'raw_row':i}
         for i,(user,item,label) in enumerate([(1,7,1),(1,8,0),(2,7,0),(2,8,1)])]
    metadata=[{'item_raw':item,'caption':f'[{token}]','author_raw':1,'category_raw':1,'category_name':'fixture'}
              for item,token in [(7,3),(8,5)]]
    for name,rows in [('search',search),('recommendation',rec),('metadata',metadata)]:
        pq.write_table(pa.Table.from_pylist(rows),canonical/f'{name}.parquet')
    write_json(canonical/'users.json',[1,2])
    files={name:digest(canonical/name) for name in ('search.parquet','recommendation.parquet','metadata.parquet','users.json')}
    write_json(canonical/'manifest.json',{'data_hash':'fixture','cohort_hash':'fixture','cutoffs_utc8_ms':CUTOFFS,'files':files})
    output=tmp_path/'prepared';build(canonical,output)
    config=tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'device':'cpu','threads':1,'batch_size':2,'model':{
        'embedding_dim':4,'hidden_dim':8,'heads':2,'ff_dim':16,'expert_hidden':8,
        'max_history':4,'dropout':0,'use_candidate_text':True,'contrastive_mode':'infonce'}}),encoding='utf-8')
    return output,config


def test_default_cloud_guard_fails_before_data_read(monkeypatch,tmp_path):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='Select|select'):
        run_preflight(tmp_path/'absent-data',tmp_path/'absent-config')


def test_cpu_fixture_preflight_backward_never_constructs_optimizer(tmp_path,monkeypatch):
    prepared,config=prepared_fixture(tmp_path,monkeypatch)
    def fail_optimizer(*args,**kwargs):
        raise AssertionError('Preflight must not construct an optimizer')
    monkeypatch.setattr(torch.optim,'AdamW',fail_optimizer)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    result=run_preflight(prepared,config,allow_cpu=True)
    assert result['status']=='passed'
    assert result['optimizer_updates']==0
    assert not result['cuda_training_preflight_verified']
    assert not result['test_predictions_or_metrics']
    assert result['fp32_weights_gradients_adam_states_minimum_bytes']==4*result['fp32_weight_bytes']
    assert set(result['probes'])=={'search','recommendation'}
    assert all(part['finite_logits_losses_gradients'] for task in result['probes'].values() for part in task['microbatches'])
    assert result['probes']['recommendation']['probe_candidates']==2


def test_preflight_rejects_model_vocabulary_shape_mismatch(tmp_path,monkeypatch):
    prepared,_=prepared_fixture(tmp_path,monkeypatch)
    dataset=PreparedDataset(prepared)
    dataset.item_count+=1
    with pytest.raises(ValueError,match='Vocabulary shape'):
        check_dataset_contract(dataset)


def test_preflight_rejects_changed_manifest_identity(tmp_path,monkeypatch):
    prepared,_=prepared_fixture(tmp_path,monkeypatch)
    dataset=PreparedDataset(prepared)
    dataset.manifest['vocabulary_hash']='changed'
    with pytest.raises(ValueError,match='manifest identity'):
        check_dataset_contract(dataset)
