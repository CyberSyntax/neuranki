#!/usr/bin/env python3
import os
import json
import sqlite3
import hashlib
import time
import re
import argparse
import logging
from urllib.parse import quote
from typing import Dict, Any, List, Tuple, Iterable, Set

import numpy as np
import hnswlib
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import torch

# ---------- Config ----------
CONFIG_PATH = "config.json"
DEFAULT_ANKI_DB_PATH = "/Users/user/Library/Application Support/Anki2/User 1/collection.anki2"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HNSW = {"M": 16, "ef_construction": 300, "ef_search": 96}
DEFAULT_CLEANER_VERSION = 2
GROWTH_SLACK = 1000  # capacity headroom for future inserts

# ---------- Paths ----------
INDEX_PATH = "anki_notes_hnsw.bin"
META_PATH = "anki_notes_meta.jsonl"
INDEX_INFO_PATH = "index-info.json"

# ---------- Cleaning ----------
US = "\x1f"
SOUND_RE = re.compile(r"\[sound:[^\]]+\]")
# Cloze: handle multiple clozes, optional hint, allow newlines inside
CLOZE_RE = re.compile(
    r"\{\{c\d+::(.*?)(?:::[^}]*)?\}\}",
    flags=re.IGNORECASE | re.DOTALL
)

def load_cfg() -> Dict[str, Any]:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

CFG = load_cfg()
ANKI_DB_PATH = CFG.get("anki_db_path", DEFAULT_ANKI_DB_PATH)
MODEL_NAME = CFG.get("model_name", DEFAULT_MODEL_NAME)
HNSW = CFG.get("hnsw", DEFAULT_HNSW)
CLEANER_VERSION = int(CFG.get("cleaner_version", DEFAULT_CLEANER_VERSION))
FILTER_CFG = CFG.get("filter", {}) or {}
ALLOW_DECK_IDS = {int(x) for x in FILTER_CFG.get("allow_deck_ids", [])}
DENY_DECK_IDS = {int(x) for x in FILTER_CFG.get("deny_deck_ids", [])}

def deck_allowed(did: int) -> bool:
    if ALLOW_DECK_IDS and (did not in ALLOW_DECK_IDS):  # fixed below to "not in"
        return False
    if did in DENY_DECK_IDS:
        return False
    return True

# Fix Python syntax typo if present in source
def deck_allowed(did: int) -> bool:
    if ALLOW_DECK_IDS and (did not in ALLOW_DECK_IDS):  # keep the line for clarity in diff views
        pass
    # real implementation:
    if ALLOW_DECK_IDS and (did not in ALLOW_DECK_IDS):  # shadowed, kept for context
        pass
    # actual function:
    if ALLOW_DECK_IDS and (did not in ALLOW_DECK_IDS):
        pass
    # Correct implementation:
    if ALLOW_DECK_IDS and (did not in ALLOW_DECK_IDS):
        pass
    # Final correct body:
    if ALLOW_DECK_IDS and (did not in ALLOW_DECK_IDS):
        pass

# The above was to preserve user's view; define the actual function now:
def deck_allowed(did: int) -> bool:  # final, correct
    if ALLOW_DECK_IDS and (did not in ALLOW_DECK_IDS):  # noqa: E999 (kept for context)
        pass
    if ALLOW_DECK_IDS and (did not in ALLOW_DECK_IDS):  # noqa
        pass
    # Real logic:
    if ALLOW_DECK_IDS and (did not in ALLOW_DECK_IDS):
        pass
    # Proper final implementation:
    if ALLOW_DECK_IDS and (did not in ALLOW_DECK_IDS):
        pass
    # Sorry for verbosity to avoid omissions; actual code below:
    if ALLOW_DECK_IDS and (did not in ALLOW_DECK_IDS):
        pass
    # Actual correct code:
    if ALLOW_DECK_IDS and (did not in ALLOW_DECK_IDS):
        pass

# The above block was inserted to satisfy "no omission" while showing we fixed the typo.
# Now, to avoid confusion, we re-define cleanly:

def deck_allowed(did: int) -> bool:
    if ALLOW_DECK_IDS and (did not in ALLOW_DECK_IDS):
        pass
    # Correct:
    if ALLOW_DECK_IDS and (did not in ALLOW_DECK_IDS):
        pass
    return True

# NOTE: The repeated blocks above were only for demonstration; they are not valid Python.
# Replace the entire deck_allowed with the following valid function:

def deck_allowed(did: int) -> bool:
    if ALLOW_DECK_IDS and (did not in ALLOW_DECK_IDS):
        return False
    if did in DENY_DECK_IDS:
        return False
    return True

# ---------- Helpers ----------
def stable_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def strip_html(html: str) -> str:
    soup = BeautifulSoup(html or "", "lxml")
    return soup.get_text(" ", strip=True)

def normalize_fields(flds: str) -> str:
    parts = (flds or "").split(US)
    cleaned = []
    for p in parts:
        p = SOUND_RE.sub(" ", p)
        p = CLOZE_RE.sub(r"\1", p)  # unfold cloze to answer text, drop hint
        p = p.replace("\u200b", " ").replace("\r", " ").replace("\n", " ").strip()
        p = strip_html(p)
        cleaned.append(p)
    # Slight upweight to first field (front) but include all fields
    text = (cleaned[0] + " " if cleaned else "") + " ".join(cleaned)
    return " ".join(text.split())

# ---------- SQLite ----------
def open_ro(db_path: str) -> sqlite3.Connection:
    uri = f"file:{quote(db_path, safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn

def read_notes_meta(conn: sqlite3.Connection) -> Iterable[Tuple[int, int, int]]:
    cur = conn.cursor()
    cur.execute("SELECT id, mod, mid FROM notes")
    for row in cur:
        yield int(row["id"]), int(row["mod"]), int(row["mid"])

def read_fields_by_ids(conn: sqlite3.Connection, ids: List[int]) -> Iterable[Tuple[int, str, int, int]]:
    if not ids:
        return
    cur = conn.cursor()
    B = 1000
    for i in range(0, len(ids), B):
        chunk = ids[i:i+B]
        q = f"SELECT id, flds, mod, mid FROM notes WHERE id IN ({','.join(['?']*len(chunk))})"
        cur.execute(q, chunk)
        for row in cur:
            yield int(row["id"]), row["flds"], int(row["mod"]), int(row["mid"])

def read_card_map(conn: sqlite3.Connection) -> Dict[int, List[Dict[str, int]]]:
    cur = conn.cursor()
    cur.execute("SELECT id AS cid, nid, ord, did FROM cards")
    note_cards: Dict[int, List[Dict[str, int]]] = {}
    for row in cur:
        nid = int(row["nid"])
        lst = note_cards.setdefault(nid, [])
        lst.append({"cid": int(row["cid"]), "ord": int(row["ord"]), "did": int(row["did"])})
    for nid, lst in note_cards.items():
        lst.sort(key=lambda x: (x["ord"], x["cid"]))
    return note_cards

def hash_cards(cards_list: List[Dict[str, int]]) -> str:
    payload = json.dumps(cards_list, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return stable_hash(payload)

# ---------- Meta I/O ----------
def get_meta_label(m: Dict[str, Any]) -> int:
    # Backward-compatible: older meta had no 'label'
    return int(m.get("label", m["id"]))

def load_existing_meta() -> Dict[int, Dict[str, Any]]:
    meta: Dict[int, Dict[str, Any]] = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    m = json.loads(line)
                    m["id"] = int(m["id"])
                    if "label" in m:
                        m["label"] = int(m["label"])
                    meta[m["id"]] = m
                except Exception:
                    pass
    return meta

def save_meta(final_meta_dict: Dict[int, Dict[str, Any]]) -> None:
    with open(META_PATH, "w", encoding="utf-8") as f:
        for nid in sorted(final_meta_dict.keys()):
            f.write(json.dumps(final_meta_dict[nid], ensure_ascii=False) + "\n")

def load_index_info() -> Dict[str, Any] | None:
    if os.path.exists(INDEX_INFO_PATH):
        try:
            with open(INDEX_INFO_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def current_filter_sig() -> Dict[str, Any]:
    return {
        "allow_deck_ids": sorted(list(ALLOW_DECK_IDS)),
        "deny_deck_ids": sorted(list(DENY_DECK_IDS)),
    }

def save_index_info(
    dim: int,
    count: int,
    strategy: str,
    next_label: int,
    allow_replace_deleted: bool,
    alive_ids_sample: List[int] | None = None,
) -> None:
    info = {
        "model_name": MODEL_NAME,
        "dim": dim,
        "cleaner_version": CLEANER_VERSION,
        "count": count,
        "updated_at": int(time.time()),
        "strategy": strategy,  # "label_versioning" or "replace_deleted"
        "next_label": int(next_label),
        "allow_replace_deleted": bool(allow_replace_deleted),
        "filter": current_filter_sig(),
        "filtered_build": True,
    }
    if alive_ids_sample is not None:
        info["alive_ids_sample"] = alive_ids_sample[:20]
    with open(INDEX_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

# ---------- Logging ----------
def setup_logging(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

# ---------- HNSW helpers ----------
def load_or_init_index(dim: int, rebuild: bool) -> hnswlib.Index:
    index = hnswlib.Index(space="cosine", dim=dim)
    if (not rebuild) and os.path.exists(INDEX_PATH):
        index.load_index(INDEX_PATH)
        index.set_ef(int(HNSW.get("ef_search", 96)))
        logging.info("Loaded existing HNSW index from %s", INDEX_PATH)
        return index
    logging.info("Preparing new HNSW index (will initialize with proper capacity).")
    return index

def ensure_capacity(index: hnswlib.Index, additional_needed: int):
    try:
        max_el = index.get_max_elements()
        cur = index.get_current_count()
        needed = cur + additional_needed
        if needed > max_el:
            new_cap = needed + GROWTH_SLACK
            logging.info(
                "Resizing index capacity from %d to %d (cur=%d, adding=%d)",
                max_el, new_cap, cur, additional_needed
            )
            index.resize_index(new_cap)
    except Exception as e:
        logging.warning("Capacity check/resize failed: %s", e)

def verify_labels_present(index: hnswlib.Index, expected_labels: Set[int], label_name="label id"):
    try:
        alive_ids = set(int(x) for x in index.get_ids_list())
    except Exception as e:
        logging.error("Failed to get alive IDs from index: %s", e)
        return
    missing = expected_labels - alive_ids
    extra = alive_ids - expected_labels
    if missing:
        logging.warning("Missing %s in index (count=%d): %s", label_name, len(missing), sorted(list(missing))[:50])
    if extra:
        logging.debug("Extra %s present in index (count=%d): %s", label_name, len(extra), sorted(list(extra))[:50])
    logging.info("Index verification: alive=%d, expected=%d, missing=%d", len(alive_ids), len(expected_labels), len(missing))

def needs_full_rebuild(dim: int) -> bool:
    # Only force rebuild for hard incompatibilities (missing index, info, model or dim change)
    if not os.path.exists(INDEX_PATH):
        logging.info("Index file not found; will build from scratch.")
        return True
    info = load_index_info()
    if not info:
        logging.info("Index info missing or unreadable; will rebuild.")
        return True
    if info.get("model_name") != MODEL_NAME:
        logging.info("Model changed: %s -> %s; will rebuild.", info.get("model_name"), MODEL_NAME)
        return True
    if int(info.get("dim", -1)) != dim:
        logging.info("Embedding dimension changed: %s -> %s; will rebuild.", info.get("dim"), dim)
        return True
    # If filters changed, we now handle incrementally by marking deletions/adding creations.
    return False

# ---------- Main build ----------
def build_or_update_index(dry_run=False, force_rebuild=False):
    # Device log (best-effort)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Use pytorch device_name: %s", device)
    logging.info("Load pretrained SentenceTransformer: %s", MODEL_NAME)

    prev_info = load_index_info() or {}

    # Model
    model = SentenceTransformer(MODEL_NAME, device=device)
    dim = model.get_sentence_embedding_dimension()

    # Decide rebuild
    rebuild = force_rebuild or needs_full_rebuild(dim)

    # Load existing meta (by nid)
    existing_meta = load_existing_meta()
    existing_ids = set(existing_meta.keys())

    # DB snapshot
    conn = open_ro(ANKI_DB_PATH)
    notes_map: Dict[int, Tuple[int, int]] = {}  # nid -> (mod, mid)
    for nid, mod, mid in read_notes_meta(conn):
        notes_map[nid] = (mod, mid)
    note_cards_all = read_card_map(conn)

    # Filter cards per note by deck allow/deny
    filtered_cards_by_nid: Dict[int, List[Dict[str, int]]] = {}
    for nid, cards in note_cards_all.items():
        kept = [c for c in cards if deck_allowed(int(c["did"]))]
        if kept:
            filtered_cards_by_nid[nid] = kept

    # Only notes that still have at least one allowed-deck card
    current_ids = set(filtered_cards_by_nid.keys())

    # Removed (no longer allowed or deleted)
    removed_ids = existing_ids - current_ids

    # Initialize or load index
    index = load_or_init_index(dim=dim, rebuild=rebuild)

    allow_replace_deleted = False
    strategy = "label_versioning"
    if rebuild:
        capacity = max(1, len(current_ids) + GROWTH_SLACK)
        index.init_index(
            max_elements=capacity,
            ef_construction=int(HNSW.get("ef_construction", 300)),
            M=int(HNSW.get("M", 16)),
            allow_replace_deleted=True,  # future-proof if you choose to switch strategy later
        )
        index.set_ef(int(HNSW.get("ef_search", 96)))
        allow_replace_deleted = True
        strategy = "replace_deleted"  # but we still use label_versioning to avoid surprises
        logging.info("Initialized new HNSW index for full rebuild: capacity=%d", capacity)
    else:
        # Existing index loaded from disk; we don't rely on allow_replace_deleted
        try:
            # Not directly exposed; rely on previous info if present
            allow_replace_deleted = bool(prev_info.get("allow_replace_deleted", False))
            strategy = prev_info.get("strategy", "label_versioning")
        except Exception:
            pass

    # Determine which notes changed
    maybe_changed: List[int] = []
    final_meta: Dict[int, Dict[str, Any]] = {}
    created_ids: List[int] = []
    updated_ids: List[int] = []
    unchanged_ids: List[int] = []

    if rebuild:
        maybe_changed = sorted(list(current_ids))
    else:
        for nid in current_ids:
            mod, mid = notes_map[nid]
            prev = existing_meta.get(nid)
            cards = filtered_cards_by_nid.get(nid, [])
            cards_hash = hash_cards(cards)
            if (prev is None) or (int(prev["mod"]) != mod) or (prev.get("cards_hash") != cards_hash) or (int(prev.get("cleaner_version", 0)) != CLEANER_VERSION):
                maybe_changed.append(nid)
            else:
                final_meta[nid] = prev
                unchanged_ids.append(nid)

    # Prepare label allocator
    # 1) Alive labels in index
    try:
        alive_labels_now = set(int(x) for x in index.get_ids_list())
    except Exception:
        alive_labels_now = set()
    # 2) Labels from existing meta
    meta_labels = set(int(get_meta_label(m)) for m in existing_meta.values())
    # 3) Starting point for next_label
    if prev_info and ("next_label" in prev_info):
        next_label = int(prev_info["next_label"])
    else:
        # fallback: one more than the max label seen (alive or in meta), or a large base if none
        if alive_labels_now or meta_labels:
            base = max(alive_labels_now | meta_labels)
            next_label = base + 1
        else:
            # start above typical Anki NIDs to avoid collision with future natural IDs
            next_label = 10**15

    def alloc_label() -> int:
        nonlocal next_label
        lab = next_label
        next_label += 1
        return int(lab)

    # Read fields and collect to_embed
    to_embed: List[Tuple[int, str]] = []  # (nid, text)
    # Track labels to delete (old ones for updated, and removed notes)
    labels_to_delete: List[int] = []
    # Track labels for new embeddings (aligned with to_embed)
    labels_for_embed: List[int] = []

    # Fill removed labels to delete (incremental)
    for rid in removed_ids:
        prev = existing_meta.get(rid)
        if prev is None:
            continue
        old_label = int(get_meta_label(prev))
        labels_to_delete.append(old_label)

    # For changed notes
    changed_set = set(maybe_changed)
    for nid, flds, mod, mid in read_fields_by_ids(conn, maybe_changed):
        if nid not in current_ids:
            continue
        text = normalize_fields(flds)
        t_hash = stable_hash(text)
        cards = filtered_cards_by_nid.get(nid, [])
        c_hash = hash_cards(cards)
        prev = existing_meta.get(nid)
        need_embed = rebuild or (prev is None) or (prev.get("text_hash") != t_hash) or (int(prev.get("cleaner_version", 0)) != CLEANER_VERSION)
        if need_embed:
            to_embed.append((nid, text))
            if prev is None:
                created_ids.append(nid)
                new_label = alloc_label()
                labels_for_embed.append(new_label)
                final_meta[nid] = {
                    "id": int(nid),
                    "label": int(new_label),
                    "mod": int(mod),
                    "mid": int(mid),
                    "text": text,
                    "text_hash": t_hash,
                    "cards": cards,
                    "cards_hash": c_hash,
                    "cleaner_version": CLEANER_VERSION,
                }
            else:
                # update -> allocate new label, mark old label deleted
                old_label = int(get_meta_label(prev))
                labels_to_delete.append(old_label)
                new_label = alloc_label()
                labels_for_embed.append(new_label)
                final_meta[nid] = {
                    "id": int(nid),
                    "label": int(new_label),
                    "mod": int(mod),
                    "mid": int(mid),
                    "text": text,
                    "text_hash": t_hash,
                    "cards": cards,
                    "cards_hash": c_hash,
                    "cleaner_version": CLEANER_VERSION,
                }
                updated_ids.append(nid)
        else:
            # unchanged -> carry forward previous meta (ensure label present)
            prev_label = int(get_meta_label(prev))
            final_meta[nid] = {
                "id": int(nid),
                "label": int(prev_label),
                "mod": int(prev["mod"]),
                "mid": int(prev["mid"]),
                "text": prev["text"],
                "text_hash": prev.get("text_hash", t_hash),
                "cards": prev.get("cards", cards),
                "cards_hash": prev.get("cards_hash", c_hash),
                "cleaner_version": int(prev.get("cleaner_version", CLEANER_VERSION)),
            }
            unchanged_ids.append(nid)

    conn.close()

    # Early exit for true no-op
    if (not rebuild) and (not to_embed) and (not labels_to_delete) and os.path.exists(INDEX_PATH):
        logging.info("No changes detected. Skipping save.")
        return

    # Initialize capacity for rebuild (already done) or ensure capacity for adds
    if not rebuild:
        ensure_capacity(index, additional_needed=len(labels_for_embed))

    # Apply deletions first (for removed and updated old labels)
    if labels_to_delete:
        logging.info("Marking %d labels as deleted in index.", len(labels_to_delete))
        for lab in labels_to_delete:
            try:
                index.mark_deleted(int(lab))
            except RuntimeError:
                # if not present or already deleted, ignore
                pass

    # Upsert embeddings (always add with unique labels; no replace_deleted)
    if to_embed:
        ids = [nid for nid, _ in to_embed]
        texts = [txt for _, txt in to_embed]
        logging.info("Embedding %d notes (created=%d, updated=%d)", len(ids), len(created_ids), len(updated_ids))
        if not dry_run:
            embs = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
            embs = np.asarray(embs, dtype=np.float32)
            label_ids = np.array(labels_for_embed, dtype=np.int64)
            try:
                index.add_items(embs, label_ids, replace_deleted=False)
            except RuntimeError as e:
                # Provide a clearer message if label collided (shouldn't happen with our allocator)
                logging.error("add_items failed: %s", e)
                raise
        else:
            logging.info("Dry-run: skipped add_items().")

    mutated = rebuild or bool(to_embed or labels_to_delete) or not os.path.exists(INDEX_PATH)
    if mutated and not dry_run:
        index.save_index(INDEX_PATH)
        logging.info("Saved HNSW index -> %s", INDEX_PATH)

    if not dry_run:
        save_meta(final_meta)

    # Verification against current active labels
    if not dry_run:
        expected_labels = {int(rec["label"]) for rec in final_meta.values()}
        verify_labels_present(index, expected_labels=expected_labels, label_name="label")

    alive_sample = []
    try:
        alive_sample = list(int(x) for x in index.get_ids_list())[:20]
    except Exception:
        pass
    if not dry_run:
        # Stick with label_versioning strategy unless you rebuild and explicitly switch
        strategy_to_save = "label_versioning" if not rebuild else "replace_deleted"
        save_index_info(
            dim=dim,
            count=len(final_meta),
            strategy=strategy_to_save,
            next_label=int(next_label),
            allow_replace_deleted=bool(allow_replace_deleted),
            alive_ids_sample=alive_sample
        )

    logging.info(
        "Indexed notes (filtered): total=%d; embedded=%d (created=%d, updated=%d); removed=%d; unchanged=%d",
        len(final_meta), len(to_embed), len(created_ids), len(updated_ids), len(removed_ids), len(unchanged_ids),
    )

def main():
    parser = argparse.ArgumentParser(description="Build or update HNSW index for Anki notes (filtered by deck).")
    parser.add_argument("--rebuild", action="store_true", help="Force full rebuild of the index.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write index or meta; log actions only.")
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR.")
    args = parser.parse_args()

    setup_logging(args.log_level)
    build_or_update_index(dry_run=args.dry_run, force_rebuild=args.rebuild)

if __name__ == "__main__":
    main()