"""Synthetic end-to-end run. Does not measure IDEA recommendation performance."""
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
from agora.models.io import write_json


def main():
    root = Path('artifacts/ronald_smoke')
    root.mkdir(parents=True, exist_ok=True)
    articles = []
    for topic in ('science', 'sport'):
        for index in range(16):
            articles.append({'article_id': f'{topic}-{index:02}', 'title': f'{topic} news item {index}',
                             'lead': f'{topic} research discoveries teams {topic}', 'topic': topic})
    write_json(root / 'articles.json', articles)
    reads = []
    for user, topic in [('reader-a', 'science'), ('reader-b', 'sport')]:
        for day in range(5):
            reads.append({'user_id': user, 'article_id': f'{topic}-{day:02}',
                          'ts': f'2023-11-{23 + day}T08:00:00+00:00'})
    pd.DataFrame(reads).to_csv(root / 'reads_train.csv', index=False)
    candidates = [a['article_id'] for a in articles if int(a['article_id'][-2:]) >= 3]
    for split, date in [('train', '2023-11-28'), ('validation', '2023-11-30'), ('test', '2023-12-03')]:
        feeds, labels = [], []
        for user, topic in [('reader-a', 'science'), ('reader-b', 'sport')]:
            feed_id = f'{split}-{user}'
            feeds.append({'feed_id': feed_id, 'user_id': user,
                          'generated_at': date + 'T12:00:00+00:00', 'candidates': candidates})
            for article in candidates:
                labels.append({'feed_id': feed_id, 'article_id': article,
                               'label': int(article == f'{topic}-07')})
        write_json(root / f'feeds_{split}.json', feeds)
        write_json(root / f'{split}_labels.json', labels)
    config = Path('configs/experiment.toml').read_text(encoding='utf-8')
    config = config.replace('source_timezone = "UNVERIFIED"', 'source_timezone = "UTC"')
    config += '\n[run]\nsynthetic = true\n'
    (root / 'config.toml').write_text(config, encoding='utf-8')

    def run(*arguments):
        subprocess.run([sys.executable, '-m', 'agora.models', *map(str, arguments)], check=True)

    run('encode', '--articles', root / 'articles.json', '--output', root / 'embeddings.npz', '--backend', 'hashing')
    shared = ['--config', root / 'config.toml', '--articles', root / 'articles.json',
              '--reads', root / 'reads_train.csv', '--embeddings', root / 'embeddings.npz',
              '--model', root / 'model.json']
    run('train', *shared, '--feeds', root / 'feeds_train.json', '--labels', root / 'train_labels.json',
        '--validation-feeds', root / 'feeds_validation.json',
        '--validation-labels', root / 'validation_labels.json',
        '--report', root / 'training_report.json', '--curves', root / 'loss_curves.png')
    report = json.loads((root / 'training_report.json').read_text())
    assert report['history'] and 'validation_log_loss' in report['history'][-1]
    run('predict', *shared, '--feeds', root / 'feeds_validation.json', '--split', 'validation',
        '--output', root / 'validation_predictions.json')
    refused = subprocess.run([sys.executable, '-m', 'agora.models', 'predict', *map(str, shared),
                              '--feeds', str(root / 'feeds_test.json'), '--split', 'test',
                              '--output', str(root / 'test_predictions.json')], capture_output=True)
    assert refused.returncode != 0 and not (root / 'test_predictions.json').exists()
    predictions = json.loads((root / 'validation_predictions.json').read_text())['predictions']
    assert len(predictions) == 52
    for user in ('reader-a', 'reader-b'):
        rows = [r for r in predictions if r['user_id'] == user]
        assert [r['rank'] for r in rows] == list(range(1, 27))
        assert set(r['article_id'] for r in rows) == set(candidates)
    subprocess.run(
        [
            sys.executable,
            "-m", "agora.evaluation",
            "--config", str(root / "config.toml"),
            "--predictions", str(root / "validation_predictions.json"),
            "--labels", str(root / "validation_labels.json"),
            "--output", str(root / "validation_metrics.json"),
        ],
        check=True,
    )
    from agora.evaluation import evaluate
    truth = json.loads((root / 'validation_labels.json').read_text())
    for method in ('popularity', 'random', 'content'):
        output = root / f'validation_{method}.json'
        run('predict', *shared, '--feeds', root / 'feeds_validation.json',
            '--split', 'validation', '--ranking', method, '--output', output)
        result = evaluate(json.loads(output.read_text())['predictions'], truth)
        assert result['total_feeds'] == 2
    print('PASS: popularity, random and content baselines accepted by evaluation.')
    print('PASS: synthetic training, reload, validation export; test predictions refused outside the declared test run.')


if __name__ == '__main__':
    main()
