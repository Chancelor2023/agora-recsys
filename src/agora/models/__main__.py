"""CLI for Ronald's component. Run python -m agora.models --help."""
import argparse
import hashlib
import random
import json
from pathlib import Path
import numpy as np
import pandas as pd
from agora.profiles import ProfileBuilder, TemporalProtocol
from .config import load_model_config
from .embeddings import encode_articles, load_embeddings
from .io import (file_hash, load_articles, load_feeds, load_reads, read_records,
                 require_ids, write_json)
from .relevance import FeedPredictor, RelevanceModel, rank_candidates


def load_context(args):
    config = load_model_config(args.config)
    articles = load_articles(args.articles)
    vectors, encoder = load_embeddings(Path(args.embeddings), articles)
    if not config.get('synthetic', False) and encoder['backend'] == 'hashing':
        raise ValueError('Hashing is restricted to synthetic smoke runs in this implementation')
    protocol = TemporalProtocol.from_config(config)
    builder = ProfileBuilder(load_reads(args.reads), articles, vectors, protocol)
    binding = {'config_sha256': file_hash(args.config), 'reads_sha256': file_hash(args.reads),
               'embeddings_sha256': file_hash(args.embeddings),
               'embedding_metadata_sha256': file_hash(str(args.embeddings) + '.json')}
    return config, builder, binding, encoder


def train(args):
    config, builder, binding, encoder = load_context(args)
    feeds = load_feeds(args.feeds)
    labels = require_ids(pd.DataFrame(read_records(args.labels, 'labels')), ['feed_id', 'article_id'])
    if 'label' not in labels or not labels.label.map(lambda x: type(x) is int and x in (0, 1)).all():
        raise ValueError('Training labels must be JSON integers 0 or 1')
    if labels.duplicated(['feed_id', 'article_id']).any():
        raise ValueError('Duplicate training labels')
    expected = {(f['feed_id'], a) for f in feeds for a in f['candidates']}
    actual = set(zip(labels.feed_id, labels.article_id))
    if expected != actual:
        raise ValueError('Training feeds and labels must have identical candidate keys')
    target = labels.set_index(['feed_id', 'article_id']).label
    matrices, targets = [], []
    skipped = 0
    for feed in feeds:
        cutoff = builder.protocol.cutoff(feed['generated_at'], 'train')
        user = feed['user_id']
        if user not in builder.eligible:
            raise ValueError(f'Feed contains an ineligible user: {user}')
        if not ((builder.reads.user_id == user) & (builder.reads.ts < cutoff)).any():
            skipped += 1
            continue
        features, _ = builder.features(user, feed['candidates'], cutoff)
        matrices.append(features)
        targets.extend(int(target.loc[(feed['feed_id'], a)]) for a in feed['candidates'])
    if not matrices:
        raise ValueError('No training feeds with nonempty prior history')
    model, report = RelevanceModel.fit(np.concatenate(matrices), targets, args.C, config["seed"])
    report.update({'synthetic': config.get('synthetic', False), 'feeds_received': len(feeds),
                   'feeds_trained': len(matrices), 'feeds_skipped_empty_history': skipped,
                   'eligible_users': len(builder.eligible), 'encoder': encoder,
                   'training_labels_sha256': file_hash(args.labels),
                   'training_feeds_sha256': file_hash(args.feeds)})
    output = Path(args.model)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output, {'binding': binding, 'training_report': report})
    write_json(args.report, report)
    print(json.dumps(report, indent=2))


def predict(args):
    config, builder, binding, _ = load_context(args)
    model, payload = RelevanceModel.load(args.model)
    if payload['metadata']['binding'] != binding:
        raise ValueError('Model inputs changed; use the same config, histories and embedding cache')
    if args.split == 'test':
        freeze_path = Path(str(args.model) + '.frozen.json')
        freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
        if freeze['model_sha256'] != file_hash(args.model):
            raise ValueError('Frozen model does not match the current model')
    predictor = FeedPredictor(builder, model)
    feeds = load_feeds(args.feeds)
    feeds.sort(key=lambda f: (pd.Timestamp(f['generated_at']), f['user_id'], f['feed_id']))
    predictions = []
    for feed in feeds:
        if args.ranking == 'model':
            rows = predictor.predict_feed(feed['user_id'], feed['feed_id'],
                       feed['generated_at'], feed['candidates'], args.split)
        else:
            cutoff = builder.protocol.cutoff(feed['generated_at'], args.split)
            features, _ = builder.features(feed['user_id'], feed['candidates'], cutoff)
            if args.ranking == 'popularity':
                scores = features[:, 2]
            elif args.ranking == 'content':
                scores = features[:, 0]
            else:
                seed = int(hashlib.sha256(f"{args.random_seed}:{feed['feed_id']}".encode()).hexdigest(), 16)
                generator = random.Random(seed)
                by_id = {a: generator.random() for a in sorted(feed['candidates'])}
                scores = [by_id[a] for a in feed['candidates']]
            rows = rank_candidates(feed['user_id'], feed['feed_id'], feed['candidates'], scores)
        predictions.extend(rows)
    write_json(args.output, {'is_synthetic': config.get('synthetic', False), 'predictions': predictions})
    report = {'synthetic': config.get('synthetic', False), 'split': args.split,
              'feeds': len(feeds), 'predictions': len(predictions),
              'ranking': args.ranking, 'random_seed': args.random_seed,
              'model_version': payload['model_version'], 'output_sha256': file_hash(args.output)}
    write_json(str(args.output) + '.report.json', report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    encode = commands.add_parser('encode')
    encode.add_argument('--articles', required=True)
    encode.add_argument('--output', required=True)
    encode.add_argument('--backend', choices=['sentence-transformer', 'hashing'], default='sentence-transformer')
    encode.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2')
    encode.add_argument('--revision', help='Pin the encoder commit for the final experiment')
    for name in ('train', 'predict'):
        sub = commands.add_parser(name)
        for option in ('config', 'articles', 'reads', 'embeddings', 'feeds', 'model'):
            sub.add_argument('--' + option, required=True)
        if name == 'train':
            sub.add_argument('--labels', required=True)
            sub.add_argument('--report', required=True)
            sub.add_argument('--C', type=float, default=1.0)
        else:
            sub.add_argument('--split', choices=['validation', 'test'], required=True)
            sub.add_argument('--output', required=True)
            sub.add_argument('--ranking', choices=['model', 'popularity', 'random', 'content'], default='model')
            sub.add_argument('--random-seed', type=int, default=42)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--model', required=True)
    freeze.add_argument('--validation-note', required=True, help='Reference to selected validation settings/results')
    args = parser.parse_args()
    if args.command == 'encode':
        result = encode_articles(load_articles(args.articles), Path(args.output),
                                 args.backend, args.encoder, args.revision)
        print(json.dumps(result, indent=2))
    elif args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict(args)
    else:
        if not args.validation_note.strip():
            raise ValueError('Supply a validation selection note')
        RelevanceModel.load(args.model)
        write_json(str(args.model) + '.frozen.json', {'model_sha256': file_hash(args.model),
                   'validation_note': args.validation_note})
        print('Model frozen for final test prediction')


if __name__ == '__main__':
    main()
