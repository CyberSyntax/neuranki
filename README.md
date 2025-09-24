# Neuranki — Local Semantic Search for Anki Notes

Semantic search over your Anki notes (all fields), with deck-filtered indexing, robust cloze unfolding, and a simple web UI.

## Features
- Note-level embeddings (all fields are included; cloze answers are unfolded, multiline-safe)
- Deck-filtered index (only allowed decks are indexed)
- HNSW ANN index (configurable M, ef_construction, ef_search)
- Simple cards-first UI (front-only display), AnkiConnect integration

## Requirements
- Python 3.10+
- Anki running locally with AnkiConnect enabled (default: http://127.0.0.1:8765)

## Install
```bash
git clone https://github.com/CyberSyntax/neuranki.git
cd neuranki
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configure
Create `config.json` (or copy from `config.example.json`) and edit paths/filters:
```json
{
  "anki_db_path": "/Users/<you>/Library/Application Support/Anki2/User 1/collection.anki2",
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "hnsw": { "M": 16, "ef_construction": 400, "ef_search": 512 },
  "cleaner_version": 2,
  "filter": {
    "allow_deck_ids": [1234567890],
    "deny_deck_ids": []
  }
}
```

Tip: you can fetch deck IDs via AnkiConnect or the `/anki/decks` endpoint once the server runs.

## Build the index
```bash
python build_index.py --rebuild --log-level INFO
```

## Run the server
```bash
uvicorn server:app --reload --port 8000
# Open http://127.0.0.1:8000/ui
```

## Quick test
```bash
curl -s http://127.0.0.1:8000/ | jq
curl --get 'http://127.0.0.1:8000/search' \
  --data-urlencode 'q=your query here' \
  --data-urlencode 'k=20' | jq
```

## Tips
- Changing `filter.allow_deck_ids` requires a rebuild:
  ```bash
  python build_index.py --rebuild
  ```
- If you see a `warning` at `/`, rebuild to align model/filter with the index.
- For better recall: increase `ef_search` in `config.json` and POST `/reload`.
- Cloze like `{{c1::...}}`, `{{c2::...::hint}}` (with newlines) are unfolded safely.
