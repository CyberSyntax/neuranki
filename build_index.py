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
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import hnswlib
import numpy as np

# ---------- Config ----------
CONFIG_PATH = "config.json"
DEFAULT_ANKI_DB_PATH = "/Users/user/Library/Application Support/Anki2/User 1/collection.anki2"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HNSW = {"M": 16, "ef_construction": 300, "ef_search": 96}
DEFAULT_CLEANER_VERSION = 2
GROWTH_SLACK = 1000  # capacity headroom for future inserts

def load_cfg():
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
    if ALLOW_DECK_IDS and did not in ALLOW_DECK_IDS:
        return False
    if did in DENY_DECK_IDS:
        return False
    return True

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

# ---------- SQLite helpers ----------
def open_ro(db_path):
    uri = f"file:{quote(db_path, safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn

def read_notes_meta(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, mod, mid FROM notes")
    for row in cur:
        yield int(row["id"]), int(row["mod"]), int(row["mid"])

def read_fields_by_ids(conn, ids):
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

def read_card_map(conn):
    # Returns dict: nid -> list of {"cid": int, "ord": int, "did": int}
    cur = conn.cursor()
    cur.execute("SELECT id AS cid, nid, ord, did FROM cards")
    note_cards = {}
    for row in cur:
        nid = int(row["nid"])
        lst = note_cards.setdefault(nid, [])
        lst.append({"cid": int(row["cid"]), "ord": int(row["ord"]), "did": int(row["did"])})
    for nid, lst in note_cards.items():
        lst.sort(key=lambda x: (x["ord"], x["cid"]))
    return note_cards

def hash_cards(cards_list):
    payload = json.dumps(cards_list, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return stable_hash(payload)

# ---------- Meta I/O ----------
def load_existing_meta():
    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    m = json.loads(line)
                    meta[int(m["id"])] = m
                except Exception:
                    pass
    return meta

def save_meta(final_meta_dict):
    with open(META_PATH, "w", encoding="utf-8") as f:
        for nid in sorted(final_meta_dict.keys()):
            f.write(json.dumps(final_meta_dict[nid], ensure_ascii=False) + "\n")

def load_index_info():
    if os.path.exists(INDEX_INFO_PATH):
        try:
            with open(INDEX_INFO_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def current_filter_sig():
    return {
        "allow_deck_ids": sorted(list(ALLOW_DECK_IDS)),
        "deny_deck_ids": sorted(list(DENY_DECK_IDS)),
    }

def save_index_info(dim: int, count: int, alive_ids_sample=None):
    info = {
        "model_name": MODEL_NAME,
        "dim": dim,
        "cleaner_version": CLEANER_VERSION,
        "count": count,
        "updated_at": int(time.time()),
        "allow_replace_deleted": True,
        "filter": current_filter_sig(),     # record deck filter signature used for this build
        "filtered_build": True,             # index contains only allowed-deck notes
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

def verify_ids_present(index: hnswlib.Index, expected_ids: set, label_name="note id"):
    try:
        alive_ids = set(index.get_ids_list())
    except Exception as e:
        logging.error("Failed to get alive IDs from index: %s", e)
        return
    missing = expected_ids - alive_ids
    extra = alive_ids - expected_ids
    if missing:
        logging.warning("Missing %s in index (count=%d): %s", label_name, len(missing), sorted(list(missing))[:50])
    if extra:
        logging.debug("Extra %s present in index (count=%d): %s", label_name, len(extra), sorted(list(extra))[:50])
    logging.info("Index verification: alive=%d, expected=%d, missing=%d", len(alive_ids), len(expected_ids), len(missing))

def needs_full_rebuild(dim: int) -> bool:
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
    if not info.get("allow_replace_deleted", False):
        logging.info("Index was created without allow_replace_deleted; will rebuild to enable replacements.")
        return True
    # Filter signature change -> force rebuild to ensure index strictly contains allowed subset
    prev_filter = (info.get("filter") or {})
    if prev_filter != current_filter_sig():
        logging.info("Deck filter changed; will rebuild. prev=%s new=%s", prev_filter, current_filter_sig())
        return True
    return False

# ---------- Main build ----------
def build_or_update_index(dry_run=False, force_rebuild=False):
    existing_meta = load_existing_meta()
    existing_ids = set(existing_meta.keys())

    # Model
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()

    # DB snapshot
    conn = open_ro(ANKI_DB_PATH)
    notes_map = {}  # nid -> (mod, mid)
    for nid, mod, mid in read_notes_meta(conn):
        notes_map[nid] = (mod, mid)

    note_cards_all = read_card_map(conn)

    # Filter cards per note by deck allow/deny
    filtered_cards_by_nid = {}
    for nid, cards in note_cards_all.items():
        kept = [c for c in cards if deck_allowed(int(c["did"]))]
        if kept:
            filtered_cards_by_nid[nid] = kept
    # Only notes that still have at least one allowed-deck card
    current_ids = set(filtered_cards_by_nid.keys())

    removed_ids = existing_ids - current_ids

    # Decide rebuild
    rebuild = force_rebuild or needs_full_rebuild(dim)

    # Determine which notes changed
    maybe_changed = []
    final_meta = {}
    created_ids = []
    updated_ids = []
    unchanged_ids = []

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

    # Read fields and collect to_embed
    to_embed = []  # (nid, text)
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
            else:
                updated_ids.append(nid)
        else:
            unchanged_ids.append(nid)
        final_meta[nid] = {
            "id": int(nid),
            "mod": int(mod),
            "mid": int(mid),
            "text": text,
            "text_hash": t_hash,
            "cards": cards,          # only allowed-deck cards
            "cards_hash": c_hash,
            "cleaner_version": CLEANER_VERSION,
        }
    conn.close()

    # Early exit for true no-op
    if (not rebuild) and (not to_embed) and (not removed_ids) and os.path.exists(INDEX_PATH):
        logging.info("No changes detected. Skipping save.")
        return

    # Create/load index
    index = load_or_init_index(dim=dim, rebuild=rebuild)

    # Initialize or resize index
    if rebuild:
        capacity = max(1, len(current_ids) + GROWTH_SLACK)
        index.init_index(
            max_elements=capacity,
            ef_construction=int(HNSW.get("ef_construction", 300)),
            M=int(HNSW.get("M", 16)),
            allow_replace_deleted=True,
        )
        index.set_ef(int(HNSW.get("ef_search", 96)))
        logging.info("Initialized new HNSW index for full rebuild: capacity=%d", capacity)

    # Apply deletions (incremental)
    if removed_ids and not rebuild:
        logging.info("Marking %d notes as deleted in index (filtered set).", len(removed_ids))
        for rid in removed_ids:
            try:
                index.mark_deleted(int(rid))
            except RuntimeError:
                pass

    # Upsert embeddings
    if to_embed:
        ids = [nid for nid, _ in to_embed]
        texts = [txt for _, txt in to_embed]
        logging.info("Embedding %d notes (created=%d, updated=%d)", len(ids), len(created_ids), len(updated_ids))
        if not dry_run:
            embs = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
            embs = np.asarray(embs, dtype=np.float32)
            ensure_capacity(index, additional_needed=len(ids))
            for nid in ids:
                try:
                    index.mark_deleted(int(nid))
                except RuntimeError:
                    pass
            label_ids = np.array(ids, dtype=np.int64)
            index.add_items(embs, label_ids, replace_deleted=True)
        else:
            logging.info("Dry-run: skipped add_items().")

    mutated = rebuild or bool(to_embed or removed_ids) or not os.path.exists(INDEX_PATH)
    if mutated and not dry_run:
        index.save_index(INDEX_PATH)
        logging.info("Saved HNSW index -> %s", INDEX_PATH)

    if not dry_run:
        save_meta(final_meta)

    # Verification
    if not dry_run:
        verify_ids_present(index, expected_ids=set(final_meta.keys()), label_name="note id")

    alive_sample = []
    try:
        alive_sample = list(index.get_ids_list())[:20]
    except Exception:
        pass
    if not dry_run:
        save_index_info(dim=dim, count=len(final_meta), alive_ids_sample=alive_sample)

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
