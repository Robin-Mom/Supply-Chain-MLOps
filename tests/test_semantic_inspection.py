"""Synthetic-only sampling tests; no model, embeddings or final evaluation calls."""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import sample_semantic_inspection as sample


def candidates(count=20, clusters=10):
    return [{'source_row_id': f'{k}-{i:02d}', 'cluster_id': k,
             'assigned_centroid_cosine_similarity': .1 + .04*i,
             'assignment_margin': .01 + .005*i,
             'review_text': f'synthetic private {k}-{i}\nsecond line'}
            for k in range(clusters) for i in range(count)]


def test_category_rules_and_quota():
    rows=candidates()
    chosen,details=sample.select_sample(rows,10)
    assert len(chosen)==len({r['source_row_id'] for r in chosen})==100
    for k in range(10):
        cluster=[r for r in chosen if r['cluster_id']==k]
        high=[r for r in cluster if r['sampling_category']=='high_similarity']
        middle=[r for r in cluster if r['sampling_category']=='intermediate_similarity']
        low=[r for r in cluster if r['sampling_category']=='low_margin']
        assert {r['source_row_id'] for r in high}=={f'{k}-{i:02d}' for i in range(16,20)}
        assert len(middle)==3
        assert sorted(abs(r['assigned_centroid_cosine_similarity']-.48) for r in middle)==pytest.approx([.02,.02,.06])
        assert {r['source_row_id'] for r in low}=={f'{k}-{i:02d}' for i in range(3)}
        assert all(r[c]=='' for r in cluster for c in sample.ANNOTATIONS)
    assert len(details)==10


def test_ties_deterministic_and_order_independent():
    rows=candidates()
    for row in rows:row.update(assigned_centroid_cosine_similarity=.5,assignment_margin=.1)
    a,details=sample.select_sample(rows,10,42)
    b,_=sample.select_sample(list(reversed(rows)),10,42)
    c,_=sample.select_sample(rows,10,43)
    assert a==b and a!=c
    assert all(d['skipped_candidates']['intermediate_similarity']['already_selected_id']==4 for d in details)
    assert all(d['skipped_candidates']['low_margin']['already_selected_id']==7 for d in details)


def test_duplicate_text_uses_next_candidate():
    rows=candidates(clusters=1)
    rows[-1]['review_text']='same';rows[-2]['review_text']=' same '
    chosen,details=sample.select_sample(rows,1)
    assert len({r['review_text'].strip() for r in chosen})==10
    assert details[0]['skipped_candidates']['high_similarity']['duplicate_stripped_text']==1


@pytest.mark.parametrize('kind',['insufficient','empty','duplicate_id','duplicate_text'])
def test_unavailable_quota_rejected(kind):
    rows=candidates(clusters=1)
    if kind=='insufficient':rows=rows[:9]
    elif kind=='empty':rows=[]
    elif kind=='duplicate_id':rows[1]['source_row_id']=rows[0]['source_row_id']
    else:
        for row in rows:row['review_text']='same'
    with pytest.raises(ValueError):sample.select_sample(rows,1)


@pytest.fixture
def inputs(tmp_path):
    evaluation=tmp_path/'evaluation';evaluation.mkdir()
    source=tmp_path/'source.csv';rows=candidates()
    with source.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['','commentaire'])
        writer.writerows((r['source_row_id'],r['review_text']) for r in rows)
    predictions=[]
    for i,row in enumerate(rows):
        predictions.append({k:v for k,v in row.items() if k!='review_text'} | {'source_row_index':i,'model_id':'synthetic-model','success':True})
    with (evaluation/'predictions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=predictions[0]);writer.writeheader();writer.writerows(predictions)
    (evaluation/'metrics.json').write_text(json.dumps({'clusters':[{'cluster_id':k,'count':20} for k in range(10)]}))
    (evaluation/'execution_state.json').write_text(json.dumps({'status':'complete'}))
    metadata={'dataset':{'sha256':sample.sha(source),'rows':200,'id_column':'','text_column':'commentaire'},
              'model':{'logical_id':'synthetic-model','n_clusters':10},
              'artifacts':{name:sample.sha(evaluation/name) for name in ['metrics.json','predictions.csv']}}
    (evaluation/'metadata.json').write_text(json.dumps(metadata))
    manifest={'status':'complete','evaluation_invocations':1,'artifacts':{name:sample.sha(evaluation/name) for name in ['metadata.json','metrics.json','predictions.csv']}}
    (evaluation/'report-manifest.json').write_text(json.dumps(manifest))
    return SimpleNamespace(evaluation=evaluation,source=source,seed=42,output=tmp_path/'sample')


def test_end_to_end_private_csv_without_inference(inputs,monkeypatch,capsys):
    for name in ['predict','evaluate_final','sentence_transformers','mlflow']:
        monkeypatch.setitem(sys.modules,name,None)
    before={p.name:sample.sha(p) for p in inputs.evaluation.iterdir()}
    result=sample.generate(inputs)
    assert result['sample_rows']==100
    fields,rows=sample.csv_rows(inputs.output/'annotation_sample.csv')
    assert fields==list(sample.COLUMNS) and len(rows)==100
    assert all('\nsecond line' in r['review_text'] for r in rows)
    assert all(r[k]=='' for r in rows for k in sample.ANNOTATIONS)
    assert (inputs.output.stat().st_mode&0o777)==0o700
    assert ((inputs.output/'annotation_sample.csv').stat().st_mode&0o777)==0o600
    assert before=={p.name:sample.sha(p) for p in inputs.evaluation.iterdir()}
    assert 'synthetic private' not in capsys.readouterr().out
    with pytest.raises(ValueError,match='exists'):sample.generate(inputs)


@pytest.mark.parametrize('kind',['source_hash','artifact_hash','model','index','duplicate','range','nan','metric_count'])
def test_invalid_provenance_or_alignment(inputs,kind):
    if kind=='source_hash':inputs.source.write_text('changed')
    elif kind=='artifact_hash':(inputs.evaluation/'predictions.csv').write_text('changed')
    else:
        path=inputs.evaluation/'predictions.csv';fields,rows=sample.csv_rows(path)
        if kind=='model':rows[0]['model_id']='wrong'
        elif kind=='index':rows[0]['source_row_index']='199'
        elif kind=='duplicate':rows[0]['source_row_id']=rows[1]['source_row_id']
        elif kind=='range':rows[0]['cluster_id']='10'
        elif kind=='nan':rows[0]['assignment_margin']='nan'
        else:(inputs.evaluation/'metrics.json').write_text(json.dumps({'clusters':[]}))
        with path.open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)
        metadata=json.loads((inputs.evaluation/'metadata.json').read_text())
        metadata['artifacts']={n:sample.sha(inputs.evaluation/n) for n in ['metrics.json','predictions.csv']}
        (inputs.evaluation/'metadata.json').write_text(json.dumps(metadata))
        manifest=json.loads((inputs.evaluation/'report-manifest.json').read_text())
        manifest['artifacts']={n:sample.sha(inputs.evaluation/n) for n in ['metadata.json','metrics.json','predictions.csv']}
        (inputs.evaluation/'report-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):sample.generate(inputs)
    assert not inputs.output.exists()
