"""Synthetic-only final protocol and geometry tests; never reads reserved data."""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import evaluate_final as final
import evaluate
import predict
from train import SphericalKMeans
from experiment import file_hash, write_json


@pytest.fixture
def geometry():
    x=np.array([[1.,0.],[.8,.6],[-1.,0.],[-.8,-.6]],dtype=np.float32)
    c=np.array([[1.,0.],[-1.,0.],[0.,1.]],dtype=np.float32)
    y=(x@c.T).argmax(axis=1)
    return x,y,c


CONFIG={'sample_size':0,'sample_seed':42,'working_memory_mb':32}


def test_exact_cosine_and_frozen_dispersion(geometry):
    x,y,c=geometry
    result,assigned,distances,margins=final.geometry_metrics(x,y,c,CONFIG)
    assert result['silhouette']['silhouette_cosine']==pytest.approx(silhouette_score(x,y,metric='cosine'))
    assert np.allclose(assigned,[1,.8,1,.8])
    assert np.allclose(distances,[0,.2,0,.2])
    assert np.allclose(margins,[1,.2,1,1.4])
    assert result['original_centroid_cosine_distance_matrix'][0][1]==2
    assert result['empty_cluster_ids']==[2]
    assert result['clusters'][2]['assigned_similarity']['status']=='unavailable'
    assert np.array_equal(c,geometry[2])


def test_deterministic(geometry):
    a=final.geometry_metrics(*geometry,CONFIG)
    b=final.geometry_metrics(*geometry,CONFIG)
    assert a[0]==b[0]
    assert all(np.array_equal(x,y) for x,y in zip(a[1:],b[1:]))


def test_singleton_and_undefined():
    x=np.array([[1.,0.],[.8,.6],[-1.,0.]],dtype=np.float32)
    c=np.array([[1.,0.],[-1.,0.]],dtype=np.float32)
    y=np.array([0,0,1])
    result,*_=final.geometry_metrics(x,y,c,CONFIG)
    assert result['clusters'][1]['count']==1
    assert result['silhouette']['silhouette_cosine'] is not None
    result,*_=final.geometry_metrics(x[:2],np.array([0,0]),c,CONFIG)
    assert result['silhouette']['silhouette_cosine'] is None
    assert result['silhouette']['undefined_reason']


@pytest.mark.parametrize('kind',['nonunit','zero','nan','bad_center','wrong_labels','float_labels','out_of_range','shape'])
def test_bad_geometry(geometry,kind):
    x,y,c=map(np.copy,geometry)
    if kind=='nonunit':x*=2
    elif kind=='zero':x[0]=0
    elif kind=='nan':x[0,0]=np.nan
    elif kind=='bad_center':c[0]*=2
    elif kind=='wrong_labels':y[0]=1
    elif kind=='float_labels':y=y.astype(float)
    elif kind=='out_of_range':y[0]=3
    else:y=y[:,None]
    with pytest.raises(ValueError):final.geometry_metrics(x,y,c,CONFIG)


def csv_file(path,ids):
    with path.open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['','commentaire']);w.writerows((i,'synthetic only') for i in ids)
    return path


@pytest.fixture
def protocol(tmp_path,monkeypatch):
    inp=csv_file(tmp_path/'final.csv',['t0','t1','t2','t3'])
    cohort=csv_file(tmp_path/'train.csv',['r0','r1'])
    monkeypatch.setattr(final,'COHORTS',{'selection':str(cohort),'training':str(cohort)})
    selection=Path('config/selection.docker.json').resolve()
    prior=tmp_path/'selection.json';d=json.loads(selection.read_text());d['selection_rule']='synthetic test';prior.write_text(json.dumps(d))
    args=final.parser().parse_args(['--audit-only','--input',str(inp),'--expected-rows','4','--selection',str(selection),
                                  '--selection-record',str(prior),'--output',str(tmp_path/'audit')])
    final.audit(args)
    return args,tmp_path


@pytest.mark.parametrize('kind',['duplicate','missing','overlap','wrong_count'])
def test_input_validation(protocol,kind):
    args,tmp=protocol;args.output=tmp/'newaudit'
    ids={'duplicate':['a','a','b','c'],'missing':['','a','b','c'],'overlap':['r0','a','b','c'],'wrong_count':['a']}[kind]
    csv_file(args.input,ids)
    with pytest.raises(Exception):final.audit(args)


def execution_args(protocol):
    audit,tmp=protocol
    return final.parser().parse_args(['--execute','--protocol',str(audit.output/'protocol.json'),'--output',str(tmp/'execution')])


def fake_loader(protocol,geometry):
    _,tmp=protocol
    x,y,c=geometry
    selection=json.loads(Path('config/selection.docker.json').read_text())
    # Use a synthetic binary and reseal the synthetic protocol/selection for tests.
    binary=tmp/'model.bin';binary.write_bytes(b'synthetic model bytes')
    selection['model_sha256']=file_hash(binary)
    selected=tmp/'selected.json';write_json(selected,selection)
    audit,_=protocol;path=audit.output/'protocol.json';p=json.loads(path.read_text())
    p['selection_path']=str(selected);p['selection_sha256']=file_hash(selected)
    p['model']['model_sha256']=file_hash(binary)
    p['model']['n_clusters']=10
    write_json(path,p);write_json(path.with_name('seal.json'),{'protocol_sha256':file_hash(path)})
    model=SphericalKMeans(**selection['configuration']['hyperparameters'])
    # Distinct original unit centroids that do not beat the fixture's assigned ones.
    model.cluster_centers_=np.vstack([c,np.tile([0.,-1.],(7,1))]).astype(np.float32)
    labels=(x@model.cluster_centers_.T).argmax(axis=1)
    model.fit=Mock(side_effect=AssertionError('must not fit'))
    predictor=SimpleNamespace(manifest={**selection,'model_path':str(binary)},bundle={'model':model},
                              predict_texts=Mock(return_value=(labels,x)))
    return predictor


def test_execute_exports_and_never_refits(protocol,geometry,monkeypatch):
    predictor=fake_loader(protocol,geometry)
    monkeypatch.setattr(evaluate,'fit_clustering',Mock(side_effect=AssertionError('no refit')))
    args=execution_args(protocol)
    result=final.execute(args,lambda *a:predictor)
    assert result['rows']==4
    predictor.bundle['model'].fit.assert_not_called()
    assert len(list((args.output/'figures').glob('*.png')))==3
    assert len(list((args.output/'figures').glob('*.svg')))==3
    metadata=json.loads((args.output/'metadata.json').read_text())
    assert metadata['coverage']['successful_rows']==4
    assert json.loads((args.output/'execution_state.json').read_text())['status']=='complete'
    assert 'synthetic only' not in (args.output/'predictions.csv').read_text()
    args.output=args.output.with_name('second-output')
    with pytest.raises(FileExistsError):final.execute(args,lambda *a:pytest.fail('second inference forbidden'))


def test_protocol_changed_rejected(protocol):
    args=execution_args(protocol);args.protocol.write_text('{}')
    with pytest.raises(ValueError,match='seal'):final.execute(args,lambda *a:pytest.fail('no load'))


def test_dataset_changed_rejected(protocol):
    original,_=protocol;csv_file(original.input,['a','b','c','d'])
    with pytest.raises(ValueError,match='dataset changed'):final.execute(execution_args(protocol),lambda *a:pytest.fail('no load'))


def test_loaded_model_mismatch(protocol,geometry):
    predictor=fake_loader(protocol,geometry);predictor.manifest['logical_id']='wrong'
    with pytest.raises(ValueError,match='loaded model'):final.execute(execution_args(protocol),lambda *a:predictor)


def test_mutated_centers_rejected(protocol,geometry):
    predictor=fake_loader(protocol,geometry)
    def mutation(*a,**kw):
        predictor.bundle['model'].cluster_centers_[0]=[0,1]
        return geometry[1],geometry[0]
    predictor.predict_texts=mutation
    with pytest.raises(ValueError,match='mutated'):final.execute(execution_args(protocol),lambda *a:predictor)


def test_no_previous_comparison():
    assert final.comparison(None,{},.1)['status']=='unavailable'


def test_shared_predictor_optional_embeddings(monkeypatch,geometry):
    x,y,c=geometry
    encoder=SimpleNamespace(encode=Mock(side_effect=lambda texts,**kw:x[:len(texts)]))
    monkeypatch.setitem(sys.modules,'sentence_transformers',SimpleNamespace(SentenceTransformer=Mock()))
    p=predict.ExperimentPredictor.__new__(predict.ExperimentPredictor)
    from experiment import PREPROCESSING
    p.manifest={'configuration':{'preprocessing':PREPROCESSING}};p.encoder=encoder
    p.predict_cached=lambda block:(block@c.T).argmax(axis=1)
    labels,blocks=p.predict_texts(['a','b'],return_embeddings=True)
    assert np.array_equal(blocks,x[:2]) and np.array_equal(labels,y[:2])
    assert isinstance(p.predict_texts(['a','b']),np.ndarray)
    assert encoder.encode.call_args.kwargs['normalize_embeddings'] is True


def test_all_singletons_undefined():
    x=np.array([[1.,0.],[0.,1.],[-1.,0.]],dtype=np.float32)
    result,*_=final.geometry_metrics(x,np.arange(3),x.copy(),CONFIG)
    assert result['silhouette']['silhouette_cosine'] is None
    assert result['represented_clusters']==3


def test_incompatible_prior_result(tmp_path):
    path=tmp_path/'previous.json';write_json(path,{'logical_id':'wrong'})
    protocol={'model':{'logical_id':'expected'}}
    result=final.comparison({'path':str(path),'sha256':file_hash(path)},protocol,.1)
    assert result['status']=='incompatible'


def test_compatible_prior_result(tmp_path):
    model={'logical_id':'id','configuration':{'preprocessing':'l2'},'model_sha256':'sha'}
    protocol={'model':model,'silhouette':CONFIG,'cohorts':{'selection':{'sha256':'data','rows':4}}}
    path=tmp_path/'previous.json'
    write_json(path,{'logical_id':'id','configuration':model['configuration'],'evaluation_config':CONFIG,
                    'train_manifest':{'model_sha256':'sha'},'eval_input_sha256':'data',
                    'metrics':{'silhouette_cosine':.2,'silhouette_n_samples':4}})
    result=final.comparison({'path':str(path),'sha256':file_hash(path)},protocol,.3)
    assert result['status']=='available'
    assert result['difference_final_minus_previous']==pytest.approx(.1)


def test_missing_execute_guard(protocol):
    args=execution_args(protocol);args.execute=False
    with pytest.raises(ValueError,match='explicit'):final.execute(args)


@pytest.mark.parametrize('kind',['implementation','versions'])
def test_runtime_protocol_invariants(protocol,kind):
    args=execution_args(protocol);p=json.loads(args.protocol.read_text())
    if kind=='implementation':p['implementation_sha256']['src/predict.py']='changed'
    else:p['versions']['numpy']='0.0.invalid'
    write_json(args.protocol,p);write_json(args.protocol.with_name('seal.json'),{'protocol_sha256':file_hash(args.protocol)})
    with pytest.raises(ValueError,match='implementation|software versions'):
        final.execute(args,lambda *a:pytest.fail('must reject before loading'))


def test_binary_checksum_rejected_before_prediction(protocol,geometry):
    predictor=fake_loader(protocol,geometry)
    Path(predictor.manifest['model_path']).write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum'):
        final.execute(execution_args(protocol),lambda *a:predictor)
    predictor.predict_texts.assert_not_called()


def test_incomplete_predictions_rejected(protocol,geometry):
    predictor=fake_loader(protocol,geometry)
    predictor.predict_texts.return_value=(geometry[1][:-1],geometry[0][:-1])
    with pytest.raises(ValueError,match='coverage'):
        final.execute(execution_args(protocol),lambda *a:predictor)
