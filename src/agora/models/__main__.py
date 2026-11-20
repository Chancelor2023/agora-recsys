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
from .curves import plot_loss_curves
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


def _load_labels(path):
    labels = require_ids(pd.DataFrame(read_records(path, 'labels')), ['feed_id', 'article_id'])
    if 'label' not in labels or not labels.label.map(lambda x: type(x) is int and x in (0, 1)).all():
        raise ValueError(f'Labels in {path} must be JSON integers 0 or 1')
    if labels.duplicated(['feed_id', 'article_id']).any():
        raise ValueError(f'Duplicate labels in {path}')
    return labels


def _matrices(builder, feeds, labels, split, strict):
    """Feature matrix + targets for one split; feeds without prior history are skipped."""
    expected = {(f['feed_id'], a) for f in feeds for a in f['candidates']}
    if expected != set(zip(labels.feed_id, labels.article_id)):
        raise ValueError(f'{split} feeds and labels must have identical candidate keys')
    target = labels.set_index(['feed_id', 'article_id']).label
    matrices, targets = [], []
    skipped = {'empty_history': 0, 'ineligible_user': 0}
    for feed in feeds:
        cutoff = builder.protocol.cutoff(feed['generated_at'], split)
        user = feed['user_id']
        if user not in builder.eligible:
            if strict:
                raise ValueError(f'Feed contains an ineligible user: {user}')
            skipped['ineligible_user'] += 1
            continue
        if not ((builder.reads.user_id == user) & (builder.reads.ts < cutoff)).any():
            skipped['empty_history'] += 1
            continue
        features, _ = builder.features(user, feed['candidates'], cutoff)
        matrices.append(features)
        targets.extend(int(target.loc[(feed['feed_id'], a)]) for a in feed['candidates'])
    return matrices, targets, skipped


def train(args):
    config, builder, binding, encoder = load_context(args)
    feeds = load_feeds(args.feeds)
    matrices, targets, skipped = _matrices(builder, feeds, _load_labels(args.labels), 'train', True)
    if not matrices:
        raise ValueError('No training feeds with nonempty prior history')
    validation, val_info = None, {}
    if bool(args.validation_feeds) != bool(args.validation_labels):
        raise ValueError('Give both --validation-feeds and --validation-labels, or neither')
    if args.validation_feeds:
        val_feeds = load_feeds(args.validation_feeds)
        val_m, val_t, val_skipped = _matrices(builder, val_feeds,
                                              _load_labels(args.validation_labels), 'validation', False)
        if not val_m:
            raise ValueError('No validation feeds with nonempty prior history')
        validation = (np.concatenate(val_m), val_t)
        val_info = {'validation_feeds_received': len(val_feeds), 'validation_feeds_used': len(val_m),
                    'validation_feeds_skipped': val_skipped,
                    'validation_labels_sha256': file_hash(args.validation_labels),
                    'validation_feeds_sha256': file_hash(args.validation_feeds)}
    model, report = RelevanceModel.fit(np.concatenate(matrices), targets, args.C, config["seed"],
                                       validation=validation, epochs=args.epochs,
                                       learning_rate=args.learning_rate, patience=args.patience)
    report.update({'synthetic': config.get('synthetic', False), 'feeds_received': len(feeds),
                   'feeds_trained': len(matrices), 'feeds_skipped_empty_history': skipped['empty_history'],
                   'eligible_users': len(builder.eligible), 'encoder': encoder,
                   'training_labels_sha256': file_hash(args.labels),
                   'training_feeds_sha256': file_hash(args.feeds), **val_info})
    output = Path(args.model)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output, {'binding': binding, 'training_report': report})
    write_json(args.report, report)
    if args.curves:
        print(plot_loss_curves(report, args.curves))
    summary = {k: v for k, v in report.items() if k != 'history'}
    print(json.dumps(summary, indent=2))


def train_neural(args):
    from .neural import NeuralConfig, NeuralRecommender, build_examples
    config, builder, binding, encoder = load_context(args)
    feeds = load_feeds(args.feeds)
    train, skipped = build_examples(builder, feeds, 'train', _load_labels(args.labels),
                                    max_history=args.max_history, strict=True)
    if not train:
        raise ValueError('No training feeds with nonempty prior history')
    validation = None
    if args.validation_feeds:
        validation, _ = build_examples(builder, load_feeds(args.validation_feeds), 'validation',
                                       _load_labels(args.validation_labels),
                                       max_history=args.max_history)
    cfg = NeuralConfig(epochs=args.epochs, patience=args.patience, max_history=args.max_history,
                       hidden=args.hidden, learning_rate=args.learning_rate, seed=config['seed'],
                       dropout=args.dropout, weight_decay=args.weight_decay, loss=args.loss,
                       position_debias=not args.no_position_debias,
                       personal_signals_only=not args.with_global_signals)
    model = NeuralRecommender(builder, cfg)
    report = model.fit(train, validation)
    report.update({'synthetic': config.get('synthetic', False), 'feeds_received': len(feeds),
                   'feeds_skipped_empty_history': skipped['empty_history'], 'encoder': encoder,
                   'training_labels_sha256': file_hash(args.labels)})
    if args.validation_labels:
        report['validation_labels_sha256'] = file_hash(args.validation_labels)
    output = Path(args.model)
    output.parent.mkdir(parents=True, exist_ok=True)
    report['model_version'] = model.save(output, {'binding': binding})
    write_json(args.report, report)
    if args.curves:
        report['title'] = 'NRMS-style neural recommender: training and validation loss'
        print(plot_loss_curves(report, args.curves))
    print(json.dumps({k: v for k, v in report.items() if k != 'history'}, indent=2))


def _load_predictor(args, builder, binding):
    """Return (predictor, payload) for a logistic (.json) or neural (.pt) model."""
    if str(args.model).endswith('.pt'):
        from .neural import NeuralFeedPredictor, NeuralRecommender
        model, payload = NeuralRecommender.load(args.model, builder)
        predictor = NeuralFeedPredictor(builder, model)
    else:
        model, payload = RelevanceModel.load(args.model)
        predictor = FeedPredictor(builder, model)
    if payload['metadata']['binding'] != binding:
        raise ValueError('Model inputs changed; use the same config, histories and embedding cache')
    return predictor, payload


def predict(args):
    config, builder, binding, _ = load_context(args)
    predictor, payload = _load_predictor(args, builder, binding)
    if args.split == 'test':
        freeze_path = Path(str(args.model) + '.frozen.json')
        freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
        if freeze['model_sha256'] != file_hash(args.model):
            raise ValueError('Frozen model does not match the current model')
    feeds = load_feeds(args.feeds)
    feeds.sort(key=lambda f: (pd.Timestamp(f['generated_at']), f['user_id'], f['feed_id']))
    predictions = []
    neural_scores = {}
    if args.ranking == 'model' and hasattr(predictor.model, 'score'):
        from .neural import build_examples  # batch scoring is much faster for the network
        examples, _ = build_examples(builder, feeds, args.split, max_history=predictor.model.cfg.max_history)
        neural_scores = {e['feed_id']: s for e, s in zip(examples, predictor.model.score(examples))}
    for feed in feeds:
        if args.ranking == 'model' and feed['feed_id'] in neural_scores:
            rows = rank_candidates(feed['user_id'], feed['feed_id'], feed['candidates'],
                                   neural_scores[feed['feed_id']])
        elif args.ranking == 'model':
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
    encode.add_argument('--backend', choices=['tfidf', 'sentence-transformer', 'hashing'], default='tfidf')
    encode.add_argument('--encoder', default='sentence-transformers/all-MiniLM-L6-v2')
    encode.add_argument('--revision', help='Pin the encoder commit for the final experiment')
    encode.add_argument('--config', help='Fit TF-IDF/SVD only on articles published before the end of '
                        'training (from this experiment config); recommended')
    for name in ('train', 'train-neural', 'predict'):
        sub = commands.add_parser(name)
        for option in ('config', 'articles', 'reads', 'embeddings', 'feeds', 'model'):
            sub.add_argument('--' + option, required=True)
        if name == 'train-neural':
            sub.add_argument('--labels', required=True)
            sub.add_argument('--report', required=True)
            sub.add_argument('--validation-feeds')
            sub.add_argument('--validation-labels')
            sub.add_argument('--epochs', type=int, default=60)
            sub.add_argument('--patience', type=int, default=8)
            sub.add_argument('--max-history', type=int, default=50)
            sub.add_argument('--hidden', type=int, default=64)
            sub.add_argument('--learning-rate', type=float, default=1e-4)
            sub.add_argument('--dropout', type=float, default=0.3)
            sub.add_argument('--weight-decay', type=float, default=1e-3)
            sub.add_argument('--loss', choices=['listwise', 'pointwise'], default='listwise')
            sub.add_argument('--with-global-signals', action='store_true',
                             help='Also feed freshness, topic and outlet popularity (same for every reader)')
            sub.add_argument('--no-position-debias', action='store_true',
                             help='Ablation: train without the position-bias tower')
            sub.add_argument('--curves')
        elif name == 'train':
            sub.add_argument('--labels', required=True)
            sub.add_argument('--report', required=True)
            sub.add_argument('--C', type=float, default=1.0)
            sub.add_argument('--validation-feeds', help='Validation feeds for the loss curve and early stopping')
            sub.add_argument('--validation-labels', help='Validation labels (never test labels)')
            sub.add_argument('--epochs', type=int, default=1500)
            sub.add_argument('--learning-rate', type=float, default=0.05)
            sub.add_argument('--patience', type=int, default=50)
            sub.add_argument('--curves', help='Output PNG of training/validation log loss per epoch')
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
        fit_before = load_model_config(args.config)['train_end'] if args.config else None
        result = encode_articles(load_articles(args.articles), Path(args.output),
                                 args.backend, args.encoder, args.revision, fit_before)
        print(json.dumps(result, indent=2))
    elif args.command == 'train':
        train(args)
    elif args.command == 'train-neural':
        train_neural(args)
    elif args.command == 'predict':
        predict(args)
    else:
        if not args.validation_note.strip():
            raise ValueError('Supply a validation selection note')
        if not str(args.model).endswith('.pt'):
            RelevanceModel.load(args.model)
        elif not Path(args.model).exists():
            raise ValueError(f'Model not found: {args.model}')
        write_json(str(args.model) + '.frozen.json', {'model_sha256': file_hash(args.model),
                   'validation_note': args.validation_note})
        print('Model frozen for final test prediction')


if __name__ == '__main__':
    main()
