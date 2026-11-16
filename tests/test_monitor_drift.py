"""Offline monitoring contracts; all reviews and predictions here are synthetic."""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import monitor_drift as m
import simulate_production as sim


def write_csv(path, ids, texts=None):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['', 'commentaire'])
        writer.writerows(zip(ids, texts or ['private text'] * len(ids)))
    return path


@pytest.fixture
def setup(tmp_path):
    ref = write_csv(tmp_path / 'reference.csv', ['r0', 'r1', 'r2', 'r3'])
    prod = write_csv(tmp_path / 'production.csv', ['p0', 'p1', 'p2', 'p3'])
    args = m.parser().parse_args(['report', '--reference', str(ref), '--production', str(prod),
                                 '--output', str(tmp_path / 'report'), '--resamples', '99'])
    return args


def artifacts(args, tmp_path, limit=None):
    identity = m.selection_identity(args.selection)
    rows, info = m.dataset(args.reference, 'commentaire', '')
    labels = np.array([0, 1, 1, 2])
    predictor = SimpleNamespace(manifest={'logical_id': identity['logical_id'],
                               'model_sha256': identity['model_sha256'], 'configuration': identity['configuration']},
                               predict_texts=lambda *a, **kw: labels)
    refpath = tmp_path / 'reference.json'
    opts = SimpleNamespace(selection=args.selection, reference=args.reference, text_column='commentaire',
                           id_column='', output=refpath, batch_size=2)
    m.generate_reference(opts, lambda *a: predictor)
    opts = sim.parser().parse_args(['--input', str(args.production), '--output-root', str(tmp_path / 'simulation'),
                                   '--run-id', 'synthetic', '--batch-size', '2', '--delay', '0']
                                  + ([] if limit is None else ['--limit', str(limit)]))
    def transport(method, url, payload, timeout):
        return (200, {'model_ready': True}) if method == 'GET' else (200, {
            'model_id': identity['logical_id'], 'predictions': [{'cluster_id': 1} for _ in payload['texts']]})
    run = sim.simulate(opts, transport=transport, sleep=lambda _: None)
    args.reference_predictions = refpath
    args.production_run = Path(run['output_directory'])
    return identity, rows, info


@pytest.mark.parametrize('ids,texts', [(['', 'a'], None), (['a', 'a'], None), (['a'], [' ']), ([], None)])
def test_bad_csv(tmp_path, ids, texts):
    with pytest.raises(sim.SimulationError):
        m.dataset(write_csv(tmp_path / 'bad.csv', ids, texts), 'commentaire', '')


def test_missing_columns(tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_text('id,text\na,review\n')
    with pytest.raises(sim.SimulationError):
        m.dataset(path, 'commentaire', '')


def test_features_clean_and_private():
    frame = m.features([{'text': '  un deux\n trois  '}])
    assert frame.to_dict('records') == [{'character_length': 14, 'word_count': 3}]


def test_identical_and_shifted():
    a = pd.DataFrame({'character_length': [5] * 30, 'word_count': [1] * 30})
    b = pd.DataFrame({'character_length': [500] * 30, 'word_count': [100] * 30})
    identical = m.input_analysis(a, a, 199, 42, .1)
    shifted = m.input_analysis(a, b, 199, 42, .1)
    m.holm(list(shifted.values()), .05)
    assert all(v['p_value'] == 1 and v['effect_size'] == 0 for v in identical.values())
    assert all(v['statistically_significant'] and v['practically_important'] for v in shifted.values())
    again = m.input_analysis(a, b, 199, 42, .1)
    m.holm(list(again.values()), .05)
    assert shifted == again


def test_cluster_frequencies_and_sparse():
    result = m.cluster_analysis([0, 0, 1], [1, 1, 1, 1], 10, 99, 42, .1)
    assert result['clusters'][0]['reference_count'] == 2
    assert result['clusters'][0]['production_count'] == 0
    assert result['effect_size'] == pytest.approx(2 / 3)
    assert sum(c['production_proportion'] for c in result['clusters']) == 1
    assert m.cluster_analysis([1] * 4, [1] * 6, 10, 99, 42, .1)['p_value'] == 1


def test_holm():
    tests = [{'p_value': p, 'effect_size': .01, 'practical_threshold': .1} for p in [.03, .01, .2]]
    m.holm(tests, .05)
    assert [t['adjusted_p_value'] for t in tests] == [.06, .03, .2]
    assert tests[1]['statistically_significant'] and not tests[1]['practically_important']


def test_input_report(setup):
    m.monitor(setup)
    result = m.read_json(setup.output / 'results.json')
    assert result['predictions']['status'] == 'unavailable'
    assert result['embeddings']['status'] == 'deferred'
    assert result['versions']['evidently'] == '0.7.23'
    assert (setup.output / 'input.html').stat().st_size > 1000
    assert 'private text' not in (setup.output / 'input.html').read_text()
    assert (setup.output / 'input.json').is_file()
    with pytest.raises(FileExistsError):
        m.monitor(setup)


def test_prediction_report(setup, tmp_path):
    artifacts(setup, tmp_path)
    m.monitor(setup)
    result = m.read_json(setup.output / 'results.json')
    assert result['predictions']['status'] == 'available'
    assert result['predictions']['coverage']['production']['successful_rows'] == 4
    assert (setup.output / 'predictions.html').is_file()


def test_subset_must_be_explicit(setup, tmp_path):
    artifacts(setup, tmp_path, limit=2)
    with pytest.raises(ValueError, match='incomplete'):
        m.monitor(setup)
    setup.allow_subset = True
    m.monitor(setup)
    coverage = m.read_json(setup.output / 'results.json')['predictions']['coverage']['production']
    assert coverage['subset'] and coverage['missing_rows'] == 2


@pytest.mark.parametrize('change', ['model', 'checksum', 'duplicate', 'unknown', 'index', 'range', 'bool', 'empty', 'missing'])
def test_invalid_reference(setup, tmp_path, change):
    identity, rows, info = artifacts(setup, tmp_path)
    data = m.read_json(setup.reference_predictions)
    if change == 'model':
        data['model']['logical_id'] = 'wrong'
    elif change == 'checksum':
        data['model']['model_sha256'] = 'wrong'
    elif change == 'duplicate':
        data['records'].append(data['records'][0])
    elif change == 'unknown':
        data['records'][0]['source_row_id'] = 'unknown'
    elif change == 'index':
        data['records'][0]['source_row_index'] = 1
    elif change == 'range':
        data['records'][0]['cluster_id'] = 10
    elif change == 'bool':
        data['records'][0]['cluster_id'] = True
    elif change == 'empty':
        data['records'] = []
    else:
        data['records'].pop()
    data['records_sha256'] = m.canonical_hash(data['records'])
    setup.reference_predictions.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        m.reference_labels(setup.reference_predictions, rows, info, identity, False)


@pytest.mark.parametrize('change', ['model', 'dataset', 'plan', 'pending', 'missing', 'batch', 'failed', 'index'])
def test_invalid_production(setup, tmp_path, change):
    identity, _, _ = artifacts(setup, tmp_path)
    manifest = m.read_json(setup.production_run / 'run.json')
    path = setup.production_run / 'requests' / 'batch-000000.json'
    batch = m.read_json(path)
    if change == 'model':
        manifest['configuration']['model_id'] = 'wrong'
    elif change == 'dataset':
        manifest['configuration']['input_sha256'] = 'wrong'
    elif change == 'plan':
        manifest['order_sha256'] = 'wrong'
    elif change == 'pending':
        batch['state'] = 'pending'
        batch['records'] = []
    elif change == 'missing':
        path.unlink()
    elif change == 'batch':
        batch['source_row_ids'].reverse()
    elif change == 'failed':
        batch['records'][0].update(success=False, cluster_id=None)
    else:
        batch['records'][0]['source_row_index'] = 3
    if change != 'missing':
        path.write_text(json.dumps(batch))
    (setup.production_run / 'run.json').write_text(json.dumps(manifest))
    rows, info = m.dataset(setup.production, 'commentaire', '')
    with pytest.raises(ValueError):
        m.production_labels(setup.production_run, rows, info, identity, False)


def test_empty_json_rejected(setup, tmp_path):
    artifacts(setup, tmp_path)
    setup.reference_predictions.write_text('')
    with pytest.raises(ValueError):
        m.monitor(setup)


def test_no_unpaired_predictions(setup):
    setup.reference_predictions = Path('missing.json')
    with pytest.raises(ValueError, match='both'):
        m.monitor(setup)


def test_reference_loader_identity_rejected(setup, tmp_path):
    args = SimpleNamespace(selection=setup.selection, reference=setup.reference, text_column='commentaire',
                           id_column='', output=tmp_path / 'ref.json', batch_size=2)
    with pytest.raises(ValueError, match='loaded model'):
        m.generate_reference(args, lambda *a: SimpleNamespace(manifest={'logical_id': 'wrong'}))
    assert not args.output.exists()


def test_overlap_rejected(setup):
    write_csv(setup.production, ['r0', 'p1'])
    with pytest.raises(ValueError, match='overlap'):
        m.monitor(setup)


def test_reference_hash_rejected(setup, tmp_path):
    identity, rows, info = artifacts(setup, tmp_path)
    artifact = m.read_json(setup.reference_predictions)
    artifact['records'][0]['cluster_id'] = 4
    setup.reference_predictions.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match='checksum'):
        m.reference_labels(setup.reference_predictions, rows, info, identity, False)


def test_empty_production_rejected_even_subset(setup, tmp_path):
    identity, _, _ = artifacts(setup, tmp_path)
    for path in (setup.production_run / 'requests').glob('batch-*.json'):
        path.unlink()
    rows, info = m.dataset(setup.production, 'commentaire', '')
    with pytest.raises(ValueError, match='empty'):
        m.production_labels(setup.production_run, rows, info, identity, True)


def test_reference_dataset_mismatch(setup, tmp_path):
    identity, rows, info = artifacts(setup, tmp_path)
    info['sha256'] = 'different'
    with pytest.raises(ValueError, match='dataset'):
        m.reference_labels(setup.reference_predictions, rows, info, identity, True)


def test_subset_retains_failures_in_coverage(setup, tmp_path):
    identity, _, _ = artifacts(setup, tmp_path)
    path = setup.production_run / 'requests' / 'batch-000000.json'
    batch = m.read_json(path)
    batch['records'][0].update(success=False, cluster_id=None, http_status=422, error_category='http_error')
    path.write_text(json.dumps(batch))
    rows, info = m.dataset(setup.production, 'commentaire', '')
    labels, coverage = m.production_labels(setup.production_run, rows, info, identity, True)
    assert len(labels) == 3 and coverage['failed_rows'] == 1 and coverage['missing_rows'] == 0


def test_reference_output_not_overwritten(setup, tmp_path):
    artifacts(setup, tmp_path)
    opts = SimpleNamespace(selection=setup.selection, reference=setup.reference, text_column='commentaire',
                           id_column='', output=setup.reference_predictions, batch_size=2)
    with pytest.raises(ValueError, match='exists'):
        m.generate_reference(opts, lambda *a: pytest.fail('must not reload model'))


def test_failed_cli_no_false_success(setup, capsys):
    status = m.main(['report', '--output', str(setup.output), '--alpha', 'nan'])
    assert status == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'failed'
