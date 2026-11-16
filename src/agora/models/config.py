"""Read the team's shared experiment.toml; preserve local calendar boundaries."""
from pathlib import Path
import tomllib
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import pandas as pd


def load_model_config(path):
    with Path(path).open('rb') as stream:
        config = tomllib.load(stream)
    protocol = config['protocol']
    timezone = protocol.get('source_timezone')
    if not timezone or timezone == 'UNVERIFIED':
        raise ValueError('protocol.source_timezone is UNVERIFIED. Brett must confirm it. '
                         'Use examples/ronald/smoke.py for synthetic checks.')
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError(f'Invalid IANA source timezone: {timezone}') from error
    profiles = config['profiles']
    expected = {'history_source': 'training_reads_only',
                'freeze_for_validation_and_test': True,
                'training_feed_history': 'strictly_before_feed'}
    for key, value in expected.items():
        if profiles.get(key) != value:
            raise ValueError(f'Unsupported profiles.{key}: {profiles.get(key)}')
    split = config['split']

    def boundary(key):
        value = pd.Timestamp(split[key])
        if value.tzinfo is not None:
            raise ValueError('Shared split boundaries must be local dates without offset')
        return value.tz_localize(timezone, ambiguous='raise', nonexistent='raise').tz_convert('UTC').isoformat()

    if boundary('train_end_exclusive') != boundary('validation_start'):
        raise ValueError('Training and validation must be adjacent')
    if boundary('validation_end_exclusive') != boundary('test_start'):
        raise ValueError('Validation and test must be adjacent')
    synthetic = config.get('run', {}).get('synthetic', False)
    if type(synthetic) is not bool:
        raise ValueError('run.synthetic must be boolean')
    return {'train_start': boundary('train_start'),
            'train_end': boundary('train_end_exclusive'),
            'validation_end': boundary('validation_end_exclusive'),
            'test_end': boundary('test_end_exclusive'),
            'min_train_articles': profiles['min_distinct_training_articles'],
            'synthetic': synthetic, 'source_timezone': timezone,
            'seed': config.get('reproducibility', {}).get('seed', 42)}
