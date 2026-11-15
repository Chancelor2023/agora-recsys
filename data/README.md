# IDEA data
Brett owns acquisition, cleaning and storage.

Official source: https://github.com/Informfully/Datasets/tree/main/IDEA

1. Review the dataset documentation and applicable conditions.
2. Record the downloaded revision, file checksums and retrieval date.
3. Put original articles, interactions and recommendations under `data/raw/`.
4. Implement the rerunnable pipeline in `src/agora/data/`.
5. Write processed outputs under `data/processed/` and document schemas.

Both local data directories are ignored by Git. Do not upload participant-level records.
Report actual counts, unresolved joins, duplicate IDs/URLs, article variants and experimental flags. Document timestamp interpretation and the first-read attribution window, including the final feed boundary. Processing the entire initial dataset is a Week 2 requirement; this scaffold does not satisfy it.
