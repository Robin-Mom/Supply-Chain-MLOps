"""Sample existing final predictions for local human annotation; no inference."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics


QUOTAS = (("high_similarity", 4), ("intermediate_similarity", 3), ("low_margin", 3))
ANNOTATIONS = ("proposed_theme", "thematic_consistency", "ambiguity", "comments")
COLUMNS = ("source_row_id", "cluster_id", "assigned_centroid_cosine_similarity",
           "assignment_margin", "sampling_category", "review_text", *ANNOTATIONS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), 'invalid CSV headers')
        rows = list(reader)
        require(rows and all(None not in row and all(v is not None for v in row.values()) for row in rows), 'empty or malformed CSV')
        return reader.fieldnames, rows


def load_inputs(evaluation, source):
    evaluation, source = Path(evaluation), Path(source)
    manifest = read_json(evaluation / 'report-manifest.json')
    metadata = read_json(evaluation / 'metadata.json')
    require(read_json(evaluation / 'execution_state.json')['status'] == 'complete', 'final evaluation incomplete')
    require(manifest['status'] == 'complete' and manifest['evaluation_invocations'] == 1, 'invalid final provenance')
    files = ['metadata.json', 'metrics.json', 'predictions.csv']
    fingerprints = {name: sha(evaluation / name) for name in files}
    require(all(manifest['artifacts'][name] == value for name, value in fingerprints.items()), 'final artifact checksum mismatch')
    require(all(metadata['artifacts'][name] == fingerprints[name] for name in ['metrics.json', 'predictions.csv']), 'metadata artifact mismatch')
    source_sha = sha(source)
    require(source_sha == metadata['dataset']['sha256'], 'source CSV checksum mismatch')
    fields, source_rows = csv_rows(source)
    id_column, text_column = metadata['dataset']['id_column'], metadata['dataset']['text_column']
    require(id_column in fields and text_column in fields, 'missing source columns')
    lookup = {}
    for index, row in enumerate(source_rows):
        source_id = row[id_column]
        require(source_id.strip() and source_id not in lookup and row[text_column].strip(), 'missing/duplicate IDs or empty text')
        lookup[source_id] = (index, row[text_column])
    require(len(lookup) == metadata['dataset']['rows'], 'source coverage mismatch')
    _, predictions = csv_rows(evaluation / 'predictions.csv')
    require(len(predictions) == len(lookup), 'prediction coverage mismatch')
    n_clusters = metadata['model']['n_clusters']
    require(type(n_clusters) is int and n_clusters > 0, 'invalid frozen cluster count')
    seen, records = set(), []
    for row in predictions:
        source_id = row['source_row_id']
        require(source_id in lookup and source_id not in seen, 'unknown or duplicate prediction ID')
        seen.add(source_id)
        index, text = lookup[source_id]
        require(row['source_row_index'] == str(index), 'prediction/source alignment mismatch')
        require(row['model_id'] == metadata['model']['logical_id'] and row['success'] == 'True', 'prediction identity or success mismatch')
        cluster = int(row['cluster_id'])
        similarity = float(row['assigned_centroid_cosine_similarity'])
        margin = float(row['assignment_margin'])
        require(0 <= cluster < n_clusters and math.isfinite(similarity) and -1 <= similarity <= 1
                and math.isfinite(margin) and 0 <= margin <= 2, 'invalid geometry/cluster')
        records.append({'source_row_id': source_id, 'cluster_id': cluster,
                        'assigned_centroid_cosine_similarity': similarity,
                        'assignment_margin': margin, 'review_text': text})
    metrics = read_json(evaluation / 'metrics.json')
    counts = Counter(r['cluster_id'] for r in records)
    require({c['cluster_id']: c['count'] for c in metrics['clusters']} ==
            {k: counts[k] for k in range(n_clusters)}, 'frozen metric counts mismatch')
    require(sha(source) == source_sha and all(sha(evaluation / name) == value for name,value in fingerprints.items()), 'input changed while reading')
    return records, n_clusters, {'source_csv_sha256': source_sha, 'final_artifacts_sha256': fingerprints,
                                 'model': metadata['model'], 'final_rows': len(records)}


def select_sample(records, n_clusters, seed=42):
    ids = [r['source_row_id'] for r in records]
    require(len(ids) == len(set(ids)), 'duplicate sample candidate IDs')
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    tie_rank = {source_id: rank for rank, source_id in enumerate(order)}
    selected, details = [], []
    for cluster in range(n_clusters):
        candidates = [r for r in records if r['cluster_id'] == cluster]
        require(candidates, 'empty cluster cannot supply inspection quota')
        median = statistics.median(r['assigned_centroid_cosine_similarity'] for r in candidates)
        used_ids, used_texts, skipped = set(), set(), {}
        rankings = {
            'high_similarity': sorted(candidates, key=lambda r: (-r['assigned_centroid_cosine_similarity'], tie_rank[r['source_row_id']])),
            'intermediate_similarity': sorted(candidates, key=lambda r: (abs(r['assigned_centroid_cosine_similarity'] - median), tie_rank[r['source_row_id']])),
            'low_margin': sorted(candidates, key=lambda r: (r['assignment_margin'], tie_rank[r['source_row_id']]))}
        for category, count in QUOTAS:
            taken = 0
            skipped[category] = {'already_selected_id': 0, 'duplicate_stripped_text': 0}
            for row in rankings[category]:
                if row['source_row_id'] in used_ids:
                    skipped[category]['already_selected_id'] += 1
                    continue
                text_key = row['review_text'].strip()
                if text_key in used_texts:
                    skipped[category]['duplicate_stripped_text'] += 1
                    continue
                selected.append({**row, 'sampling_category': category, **dict.fromkeys(ANNOTATIONS, '')})
                used_ids.add(row['source_row_id']); used_texts.add(text_key)
                taken += 1
                if taken == count:
                    break
            require(taken == count, 'cluster cannot supply ten distinct eligible reviews; no output written')
        details.append({'cluster_id': cluster, 'candidate_rows': len(candidates), 'cluster_median_similarity': median,
                        'category_counts': dict(QUOTAS), 'skipped_candidates': skipped})
    require(len(selected) == n_clusters * 10 and len({r['source_row_id'] for r in selected}) == len(selected), 'sample quota or distinctness failure')
    return selected, details


GUIDE = '''# Guide d'annotation exploratoire

Le CSV local contient des textes originaux potentiellement sensibles. Ne pas le
publier, le commiter ou le transmettre à un service externe. Importer les colonnes
comme texte dans le tableur afin de préserver identifiants et retours à la ligne.
Les colonnes d'annotation sont volontairement vides : aucun thème n'est présumé.

## Lecture et annotation

Lire d'abord les dix avis d'un cluster sans supposer que leur numéro porte un sens
thématique. Si possible, masquer les colonnes de géométrie et de catégorie lors de
la première lecture pour limiter l'ancrage. Proposer un thème court et descriptif
uniquement après lecture, ou laisser « indéterminé » si les avis ne le permettent pas.
Un cluster peut couvrir plusieurs thèmes ; ne pas forcer une étiquette unique.

- `proposed_theme` : thème humain proposé, sans données personnelles.
- `thematic_consistency` : `coherent`, `mixed`, `clear_mismatch` ou `insufficient_context`.
  Juger le lien au thème provisoire du groupe, pas la seule présence de mots communs.
- `ambiguity` : `yes`, `no` ou `uncertain`. Expliquer les hésitations dans `comments`.
- `comments` : justification courte, éléments contradictoires ou limites de contexte,
  sans recopier inutilement les textes ou des informations personnelles.

Un avis ambigu peut traiter plusieurs sujets, être trop bref ou convenir à plusieurs
clusters. Un décalage clair est un avis interprétable dont le thème diffère nettement
du thème provisoire du groupe. Un avis bref n'est pas automatiquement mal classé.
Une faible marge géométrique désigne un candidat à examiner, pas une ambiguïté
sémantique établie ; une forte similarité ne garantit pas la cohérence thématique.
Documenter les cas incertains et, si plusieurs annotateurs participent, conserver
leurs fichiers séparés avant discussion. Ne pas modifier les prédictions ou métriques.

## Échantillonnage figé

Seed 42 par défaut. Les IDs triés lexicalement sont mélangés par `random.Random(seed)` ;
leur rang sert uniquement à départager les égalités. Par cluster, dans cet ordre :
4 plus fortes similarités ; 3 plus proches de la médiane de similarité de TOUS les
avis du cluster ; puis 3 plus faibles marges parmi les avis encore admissibles.
La médiane peut être la moyenne de deux valeurs. Les IDs sont uniques sur les
100 lignes ; les textes après suppression des seuls espaces de bord sont distincts
au sein de chaque cluster. Le CSV conserve toujours le texte ORIGINAL. Une collision
fait passer au candidat suivant dans le même classement. Les quotas ne sont pas
assouplis : si impossibles, le programme échoue explicitement. Les candidats sautés
et les médianes sont documentés dans `sampling_metadata.json`.

## Limites d'interprétation

Il s'agit de 10 avis ciblés par cluster, pas d'un échantillon représentatif des
2 261 avis. Les petits clusters sont surreprésentés, les extrêmes et faibles marges
sont sélectionnés intentionnellement, et les répétitions de texte sont évitées.
Les probabilités d'inclusion sont inégales et certains avis ne peuvent être retenus.
La seed rend le choix reproductible ; elle ne le rend pas aléatoire représentatif.
Aucun taux global de justesse non biaisé, intervalle de confiance d'accuracy ou
classement de modèles ne peut être déduit de ce plan. Les thèmes et la cohérence
ne seront connus qu'après annotation humaine ; les conclusions resteront exploratoires.
Les 127 correspondances textuelles avec l'entraînement et les 76 avec la sélection
limitent en outre la nouveauté du jeu final (ensembles pouvant se recouper).
'''


def generate(args):
    records, n_clusters, provenance = load_inputs(args.evaluation, args.source)
    require(n_clusters == 10, 'expected the frozen ten-cluster evaluation')
    sample, details = select_sample(records, n_clusters, args.seed)
    require(not args.output.exists(), 'output directory exists; refuse to overwrite annotations')
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    csv_path = args.output / 'annotation_sample.csv'
    with csv_path.open('x', encoding='utf-8', newline='') as stream:
        os.chmod(csv_path, 0o600)
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader(); writer.writerows(sample)
    (args.output / 'annotation_guide.md').write_text(GUIDE, encoding='utf-8')
    metadata = {'schema_version': 1, 'seed': args.seed, 'sample_rows': len(sample), 'clusters': n_clusters,
                'rules': {'category_order_and_quotas': list(QUOTAS), 'intermediate_target': 'median of all cluster similarities',
                          'ties': 'shuffle lexically sorted source IDs with random.Random(seed)',
                          'distinctness': 'globally unique IDs; distinct stripped texts within each cluster',
                          'raw_text': 'original source CSV value, unchanged'},
                'cluster_details': details, 'deviations': [], 'semantic_annotation_status': 'not_started',
                'provenance': provenance, 'sample_csv_sha256': sha(csv_path)}
    (args.output / 'sampling_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'status': 'complete', 'sample_rows': len(sample), 'clusters': n_clusters,
            'output_directory': str(args.output), 'deviations': []}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, default=Path('reports/final_evaluation/final_20261116'))
    p.add_argument('--source', type=Path, default=Path('data/processed/avis_3k_test.csv'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, required=True)
    return p


def main(argv=None):
    try:
        print(json.dumps(generate(parser().parse_args(argv)), indent=2))
    except (ValueError, KeyError, OSError, TypeError):
        print(json.dumps({'status': 'failed', 'reason': 'Invalid inputs or unavailable distinct sampling quotas; no annotation success claimed.'}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
