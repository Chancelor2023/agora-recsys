"""Article text vectors (TF-IDF+SVD by default, or sentence embeddings); hashing is a smoke-test mode.

The fitted encoder is saved next to the vectors so that new articles can be encoded later."""
import numpy as np
from .io import records_hash, write_json


def encode_articles(articles, output, backend='sentence-transformer', model_name=None, revision=None,
                    fit_before=None):
    """Encode every catalogue article; a fitted encoder only learns from the past.

    ``fit_before`` (UTC timestamp, normally the end of training): TF-IDF and SVD are
    fitted on articles published strictly before it, then applied to every article,
    so that no vocabulary or topic direction comes from validation or test text.
    """
    import pandas as pd
    texts = (articles.title + '\n' + articles.lead).tolist()
    fitted = None
    fit_mask = np.ones(len(articles), dtype=bool)
    if fit_before is not None and backend == 'tfidf':
        if 'published_at' not in articles:
            raise ValueError('fit_before needs a published_at column in the article catalogue')
        published = pd.to_datetime(articles.published_at, utc=True, errors='coerce')
        fit_mask = (published < pd.Timestamp(fit_before)).to_numpy()
        if fit_mask.sum() < 100:
            raise ValueError(f'Only {int(fit_mask.sum())} articles published before {fit_before}')
    fit_texts = [t for t, keep in zip(texts, fit_mask) if keep]
    if backend == 'sentence-transformer':
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model_name, revision=revision, device='cpu')
        vectors = encoder.encode(texts, batch_size=32, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=True)
    elif backend == 'tfidf':
        # Offline, CPU-only content representation: TF-IDF (1-2 grams) reduced by
        # truncated SVD. Uses article text only, never interactions or labels.
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                                sublinear_tf=True, stop_words='english')
        fit_matrix = tfidf.fit_transform(fit_texts)
        components = min(256, fit_matrix.shape[1] - 1, fit_matrix.shape[0] - 1)
        svd = TruncatedSVD(components, random_state=42).fit(fit_matrix)
        vectors = normalize(svd.transform(tfidf.transform(texts)))
        fitted = (tfidf, svd)  # kept so that new articles can be encoded identically
    elif backend == 'hashing':
        from sklearn.feature_extraction.text import HashingVectorizer
        vectors = HashingVectorizer(n_features=512, alternate_sign=False, norm='l2').transform(texts).toarray()
    else:
        raise ValueError(f'Unknown encoder backend: {backend}')
    metadata = {'backend': backend,
                'model_name': (model_name if backend == 'sentence-transformer'
                               else 'TfidfVectorizer-1-2gram+TruncatedSVD-256' if backend == 'tfidf'
                               else 'HashingVectorizer-512'),
                'revision': revision if backend == 'sentence-transformer' else None,
                'catalogue_hash': records_hash(articles[['article_id', 'title', 'lead', 'topic']].to_dict('records')),
                'fit_before': str(fit_before) if fit_before is not None and backend == 'tfidf' else None,
                'fit_articles': int(fit_mask.sum()) if backend == 'tfidf' else None}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as handle:
        np.savez_compressed(handle, ids=articles.article_id.to_numpy(dtype=str), vectors=vectors)
    write_json(str(output) + '.json', metadata)
    if fitted is not None:
        import joblib
        joblib.dump(fitted, str(output) + '.encoder.joblib')
    return metadata


def load_text_encoder(embeddings_path):
    """Return a function texts -> unit vectors in the same space as the catalogue.

    Used to score articles that are not in the training catalogue (live RSS items).
    """
    import json
    from pathlib import Path
    metadata = json.loads(Path(str(embeddings_path) + '.json').read_text(encoding='utf-8'))
    backend = metadata['backend']
    if backend == 'tfidf':
        import joblib
        from sklearn.preprocessing import normalize
        tfidf, svd = joblib.load(str(embeddings_path) + '.encoder.joblib')
        return lambda texts: normalize(svd.transform(tfidf.transform(list(texts))))
    if backend == 'hashing':
        from sklearn.feature_extraction.text import HashingVectorizer
        vectorizer = HashingVectorizer(n_features=512, alternate_sign=False, norm='l2')
        return lambda texts: vectorizer.transform(list(texts)).toarray()
    if backend == 'sentence-transformer':
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(metadata['model_name'], revision=metadata.get('revision'), device='cpu')
        return lambda texts: encoder.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
    raise ValueError(f'Unknown encoder backend: {backend}')


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
