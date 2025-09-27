# Neuranki — Local Semantic Search for Anki Notes

Semantic search over your Anki notes (all fields), with deck-filtered indexing, robust cloze unfolding, and a simple web UI. Incremental updates are supported without full rebuilds.

## Quick start
- Requirements: Python 3.10+, Anki running locally with AnkiConnect (http://127.0.0.1:8765).
- Install:
  ```bash
  git clone https://github.com/CyberSyntax/neuranki.git
  cd neuranki
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Configure `config.json` (copy from `config.example.json` and edit paths/filters):
  ```json
  {
    "anki_db_path": "/Users/<you>/Library/Application Support/Anki2/User 1/collection.anki2",
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "hnsw": { "M": 16, "ef_construction": 400, "ef_search": 512 },
    "cleaner_version": 2,
    "filter": { "allow_deck_ids": [], "deny_deck_ids": [] }
  }
  ```

## Build or update the index
- First build or incremental update:
  ```bash
  python build_index.py
  ```
  This creates/updates:
  - `anki_notes_hnsw.bin` (HNSW index)
  - `anki_notes_meta.jsonl` (note metadata; includes a per-note `label`)
  - `index-info.json` (index metadata)

- Full rebuild (only needed if the embedding model or dimension changes, or the index is missing):
  ```bash
  python build_index.py --rebuild
  ```

## Run the server
```bash
uvicorn server:app --reload --port 8000
# UI: http://127.0.0.1:8000/ui
```

Quick test:
```bash
curl -s http://127.0.0.1:8000/ | jq
curl --get 'http://127.0.0.1:8000/search' --data-urlencode 'q=your query' --data-urlencode 'k=20' | jq
```

## Incremental updates (no full rebuild needed)
- The index uses label-versioning:
  - When a note changes, a new unique `label` is added to the index and the old label is marked deleted.
  - When a note is removed or filtered out by deck rules, its current label is marked deleted.
  - This avoids the hnswlib replace-deleted constraint; `allow_replace_deleted` is not required for incremental updates.
- Deck filter changes:
  - Updates are applied incrementally: notes that newly match are added; notes that no longer match are marked deleted.
  - No rebuild is required solely due to filter changes.

## Deck filters and IDs
- Use `filter.allow_deck_ids` and/or `filter.deny_deck_ids` in `config.json`.
- You can fetch deck IDs from Anki via the server:
  ```bash
  curl -s http://127.0.0.1:8000/anki/decks | jq
  ```
- After editing `config.json`, you can hot-reload the server (no rebuild needed):
  ```bash
  curl -X POST http://127.0.0.1:8000/reload
  ```

## Notes
- Cloze like `{{c1::...}}` and `{{c2::...::hint}}` (even multiline) are unfolded to the answer text for embedding.
- Device selection is automatic (MPS on Apple Silicon, CUDA if available, else CPU).
- For better recall/accuracy at query time, increase `hnsw.ef_search` in `config.json` and POST `/reload`.

## Troubleshooting
- If you ever want a clean slate:
  ```bash
  rm -f anki_notes_hnsw.bin index-info.json anki_notes_meta.jsonl
  python build_index.py --rebuild
  ```
- If the server warns about a model/filter mismatch at `/`, rebuild the index to align with your current config and model.