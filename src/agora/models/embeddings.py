"""Cached frozen sentence embeddings; hashing is an explicit offline smoke mode."""
import numpy as np
from .io import records_hash, write_json


def encode_articles(articles, output, backend='sentence-transformer', model_name=None, revision=None):
    texts = (articles.title + '\n' + articles.lead).tolist()
    if backend == 'sentence-transformer':
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model_name, revision=revision, device='cpu')
        vectors = encoder.encode(texts, batch_size=32, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=True)
    elif backend == 'hashing':
        from sklearn.feature_extraction.text import HashingVectorizer
        vectors = HashingVectorizer(n_features=512, alternate_sign=False, norm='l2').transform(texts).toarray()
    else:
        raise ValueError(f'Unknown encoder backend: {backend}')
    metadata = {'backend': backend,
                'model_name': model_name if backend == 'sentence-transformer' else 'HashingVectorizer-512',
                'revision': revision if backend == 'sentence-transformer' else None,
                'catalogue_hash': records_hash(articles[['article_id', 'title', 'lead', 'topic']].to_dict('records'))}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as handle:
        np.savez_compressed(handle, ids=articles.article_id.to_numpy(dtype=str), vectors=vectors)
    write_json(str(output) + '.json', metadata)
    return metadata


def load_embeddings(path, articles):
    import json
    from pathlib import Path
    metadata = json.loads(Path(str(path) + '.json').read_text(encoding='utf-8'))
    expected = records_hash(articles[['article_id', 'title', 'lead', 'topic']].to_dict('records'))
    if metadata['catalogue_hash'] != expected:
        raise ValueError('Embedding cache does not match this article catalogue')
    with np.load(path, allow_pickle=False) as data:
        if data['ids'].tolist() != articles.article_id.tolist():
            raise ValueError('Embedding IDs/order mismatch')
        vectors = data['vectors'].copy()
    return vectors, metadata
