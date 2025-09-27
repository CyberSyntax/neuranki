import os
import json
import numpy as np
import hnswlib
import requests
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import torch
from bs4 import BeautifulSoup
from typing import Dict, Any, Tuple, List

# ---------- Config ----------
CONFIG_PATH = "config.json"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HNSW_EF = 96
ANKICONNECT_URL = "http://127.0.0.1:8765"

def load_cfg():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

CFG = load_cfg()
MODEL_NAME = CFG.get("model_name", DEFAULT_MODEL_NAME)
EF_SEARCH = int(CFG.get("hnsw", {}).get("ef_search", DEFAULT_HNSW_EF))

# Deck filter (allow/deny)
FILTER_CFG = CFG.get("filter", {}) or {}
ALLOW_DECK_IDS = {int(x) for x in FILTER_CFG.get("allow_deck_ids", [])}
DENY_DECK_IDS = {int(x) for x in FILTER_CFG.get("deny_deck_ids", [])}

def deck_allowed(did: int) -> bool:
    if ALLOW_DECK_IDS and did not in ALLOW_DECK_IDS:  # corrected below
        return False
    if did in DENY_DECK_IDS:
        return False
    return True

def deck_allowed(did: int) -> bool:  # final, correct
    if ALLOW_DECK_IDS and (did not in ALLOW_DECK_IDS):
        return False
    if did in DENY_DECK_IDS:
        return False
    return True

def current_filter_sig():
    return {
        "allow_deck_ids": sorted(list(ALLOW_DECK_IDS)),
        "deny_deck_ids": sorted(list(DENY_DECK_IDS)),
    }

# ---------- Paths ----------
INDEX_PATH = "anki_notes_hnsw.bin"
META_PATH = "anki_notes_meta.jsonl"
INDEX_INFO_PATH = "index-info.json"

if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
    raise SystemExit("Build the index first: `python build_index.py`")

# ---------- Model/Index ----------
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(MODEL_NAME, device=device)
dim = model.get_sentence_embedding_dimension()

index = hnswlib.Index(space="cosine", dim=dim)
index.load_index(INDEX_PATH)
index.set_ef(EF_SEARCH)

# ---------- Meta ----------
def load_meta() -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    by_nid: Dict[int, Dict[str, Any]] = {}
    by_label: Dict[int, Dict[str, Any]] = {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            nid = int(m["id"])
            label = int(m.get("label", nid))
            m["id"] = nid
            m["label"] = label
            by_nid[nid] = m
            by_label[label] = m
    return by_nid, by_label

def load_index_info():
    if os.path.exists(INDEX_INFO_PATH):
        with open(INDEX_INFO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

META_BY_NID, META_BY_LABEL = load_meta()
INDEX_INFO = load_index_info()

# Validate filtered build vs current filter config
FILTER_WARNING = None
if INDEX_INFO.get("filtered_build") is not True:
    FILTER_WARNING = "Index was not built as filtered subset. Rebuild with current filter for best results."
else:
    built_filter = INDEX_INFO.get("filter") or {}
    if built_filter != current_filter_sig():
        FILTER_WARNING = f"Index filter {built_filter} differs from server config {current_filter_sig()}. Rebuild recommended."

# ---------- App ----------
app = FastAPI(title="Neuranki — Deck-filtered Semantic Search for Anki")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def anki_request(action: str, **params):
    payload = {"action": action, "version": 6, "params": params}
    r = requests.post(ANKICONNECT_URL, json=payload, timeout=5.0)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError(data["error"])
    return data.get("result")

def try_cards_info(cids):
    if not cids:
        return {}, False
    try:
        info = anki_request("cardsInfo", cards=cids) or []
        out = {}
        for ci in info:
            cid = int(ci.get("cardId") or ci.get("id"))
            out[cid] = {
                "question": ci.get("question", ""),
                "flags": int(ci.get("flags", 0)),
                "noteId": int(ci.get("noteId") or ci.get("nid") or 0),
                "deckName": ci.get("deckName", ""),
            }
        return out, True
    except Exception:
        return {}, False

def sanitize_front(html: str) -> str:
    try:
        soup = BeautifulSoup(html or "", "lxml")
        for tag in soup(["script", "style", "link", "img", "audio", "video", "source", "svg", "iframe", "object", "embed"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        return " ".join(text.split())
    except Exception:
        return (html or "").replace("<", " ").replace(">", " ").strip()

def snippet(text, max_len=180):
    t = text or ""
    return t if len(t) <= max_len else t[:max_len-1] + "…"

@app.get("/")
def root():
    warn = None
    # model mismatch warning
    if INDEX_INFO.get("model_name") and INDEX_INFO["model_name"] != MODEL_NAME:
        warn = f"Index built with {INDEX_INFO['model_name']} but server uses {MODEL_NAME}. Rebuild recommended."
    # filter mismatch warning
    if FILTER_WARNING:
        if warn:
            warn = warn + " | " + FILTER_WARNING
        else:
            warn = FILTER_WARNING
    return {"ok": True, "message": "Use /ui for the GUI, or /search?q=...&k=20", "warning": warn}

@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    k: int = 20,
    kn: int | None = None,
    ef: int | None = None,  # optional per-request ef override
):
    q = q.strip()
    if not q:
        return JSONResponse({"results": [], "anki_ok": None})

    if ef is not None:
        try:
            index.set_ef(int(ef))
        except Exception:
            pass

    q_emb = model.encode([q], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)

    # Since index is already filtered, collect generously then trim
    index_count = int(INDEX_INFO.get("count") or len(META_BY_LABEL) or 0)
    default_top = min(index_count, max(1000, int(k) * 50))
    top_notes = int(kn) if kn is not None else default_top

    labels, distances = index.knn_query(q_emb, k=top_notes)
    labs = [int(x) for x in labels[0]]
    dists = [float(x) for x in distances[0]]

    candidates = []
    for lab, dist in zip(labs, dists):
        if lab == -1 or lab not in META_BY_LABEL:
            continue
        rec = META_BY_LABEL[lab]
        nid = int(rec["id"])
        sim = 1.0 - dist / 2.0
        for c in rec.get("cards", []):  # already filtered at build time
            did = int(c.get("did", 0))
            if not deck_allowed(did):
                continue
            candidates.append({
                "cid": int(c["cid"]),
                "nid": nid,
                "ord": int(c.get("ord", 0)),
                "did": did,
                "similarity": float(sim),
            })

    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    candidates = candidates[:k]

    cids = [c["cid"] for c in candidates]
    cards_info, ok = try_cards_info(cids)

    results = []
    for c in candidates:
        info = cards_info.get(c["cid"], {})
        if ok:
            front = sanitize_front(info.get("question", ""))
            flag = int(info.get("flags", 0))
            deckName = info.get("deckName", "")
        else:
            rec = META_BY_NID.get(c["nid"], {})
            front = snippet(rec.get("text", ""), 180)
            flag = 0
            deckName = ""
        results.append({
            "cid": c["cid"],
            "nid": c["nid"],
            "ord": c["ord"],
            "did": c["did"],
            "deckName": deckName,
            "similarity": c["similarity"],
            "flag": flag,
            "front": front,
        })

    return JSONResponse({"results": results, "anki_ok": ok})

@app.get("/anki/status")
def anki_status():
    try:
        v = anki_request("version")
        return {"ok": True, "version": v}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=503)

@app.get("/anki/decks")
def anki_decks():
    try:
        mapping = anki_request("deckNamesAndIds") or {}
        out = {name: int(deck_id) for name, deck_id in mapping.items()}
        return {"ok": True, "decks": out}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=503)

def set_card_flag(cid: int, flag: int) -> int:
    try:
        anki_request(
            "setSpecificValueOfCard",
            card=cid,
            keys=["flags"],
            newValues=[int(flag)],
            warning_check=True
        )
        try:
            anki_request("reloadCollection")
        except Exception:
            pass
        info = anki_request("cardsInfo", cards=[cid]) or []
        eff = int(info[0].get("flags", flag)) if info else int(flag)
        return eff
    except Exception as e:
        raise e

@app.get("/anki/set_flag_card")
def anki_set_flag_card(cid: int, flag: int):
    if flag < 0 or flag > 7:
        return JSONResponse({"ok": False, "error": "flag must be 0..7"}, status_code=400)
    try:
        eff = set_card_flag(cid, flag)
        return {"ok": True, "flag": int(eff), "requested": int(flag)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/anki/open_card")
def anki_open_card(cid: int):
    try:
        anki_request("guiBrowse", query=f"cid:{cid}")
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/anki/open_cards")
def anki_open_cards(cids: str):
    try:
        cid_list = [int(x) for x in cids.split(",") if x.strip()]
        if not cid_list:
            return JSONResponse({"ok": False, "error": "no cids"}, status_code=400)
        q = " OR ".join([f"cid:{c}" for c in cid_list])
        anki_request("guiBrowse", query=q)
        return {"ok": True, "count": len(cid_list)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/anki/add_tag_note")
def anki_add_tag_note(nid: int, tag: str):
    tag = (tag or "").strip()
    if not tag:
        return JSONResponse({"ok": False, "error": "empty tag"}, status_code=400)
    try:
        anki_request("addTags", notes=[nid], tags=tag)
        info = anki_request("notesInfo", notes=[nid]) or []
        tags = info[0].get("tags", []) if info else []
        return {"ok": True, "tags": tags}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/reload")
def reload_index():
    global index, META_BY_NID, META_BY_LABEL, INDEX_INFO, CFG, EF_SEARCH, FILTER_CFG, ALLOW_DECK_IDS, DENY_DECK_IDS
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        return JSONResponse({"ok": False, "error": "Index/meta missing, run build_index.py"}, status_code=400)

    # Reload config and apply ef_search + filters
    CFG = load_cfg()
    EF_SEARCH = int(CFG.get("hnsw", {}).get("ef_search", DEFAULT_HNSW_EF))
    FILTER_CFG = CFG.get("filter", {}) or {}
    ALLOW_DECK_IDS = {int(x) for x in FILTER_CFG.get("allow_deck_ids", [])}
    DENY_DECK_IDS = {int(x) for x in FILTER_CFG.get("deny_deck_ids", [])}

    # Reload the on-disk index
    index.load_index(INDEX_PATH)
    index.set_ef(EF_SEARCH)

    # Reload meta and index info
    META_BY_NID, META_BY_LABEL = load_meta()
    INDEX_INFO = load_index_info()

    # Recompute filter warning
    global FILTER_WARNING
    FILTER_WARNING = None
    if INDEX_INFO.get("filtered_build") is not True:
        FILTER_WARNING = "Index was not built as filtered subset. Rebuild with current filter for best results."
    else:
        built_filter = INDEX_INFO.get("filter") or {}
        if built_filter != current_filter_sig():
            FILTER_WARNING = f"Index filter {built_filter} differs from server config {current_filter_sig()}. Rebuild recommended."

    return {
        "ok": True,
        "count": len(META_BY_LABEL),
        "ef_search": EF_SEARCH,
        "allow_deck_ids": sorted(list(ALLOW_DECK_IDS)),
        "deny_deck_ids": sorted(list(DENY_DECK_IDS)),
        "filtered_build": INDEX_INFO.get("filtered_build", False),
    }

# ---------- UI ----------
@app.get("/ui", response_class=HTMLResponse)
def ui():
    html = r"""
<!doctype html>
<meta charset="utf-8" />
<title>Neuranki — Semantic Search (Filtered)</title>
<style>
  html { overflow-y: scroll; }
  :root {
    --muted:#666; --border:#eee; --bg:#fff; --accent:#1e90ff;
    --f0:#bbb; --f1:#ff6b6b; --f2:#ffa94d; --f3:#51cf66; --f4:#339af0; --f5:#f783ac; --f6:#20c997; --f7:#845ef7;
  }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: var(--bg); }
  .container { max-width: 980px; margin: 0 auto; padding: 16px; }
  .toolbar { position: sticky; top: 0; background: var(--bg); z-index: 10; border-bottom: 1px solid var(--border); padding: 12px 0; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; width: 100%; }
  #q { flex: 1 1 600px; min-width: 480px; padding: 10px; font-size: 16px; }
  input[type="number"] { width: 90px; padding: 8px; }
  button { padding: 8px 12px; cursor: pointer; }
  .result { padding: 12px 0; border-bottom: 1px solid var(--border); }
  .meta { color: var(--muted); font-size: 12px; margin: 4px 0; }
  .front { padding: 6px 8px; background: #fafafa; border: 1px solid var(--border); border-radius: 6px; word-break: break-word; overflow-wrap: anywhere; }
  .actions { margin-top: 6px; display:flex; gap:8px; align-items:center; flex-wrap: wrap; }
  .flag-dot { display:inline-block; width:10px; height:10px; border-radius:50%; border:1px solid #999; vertical-align:middle; margin-right:4px; }
  .flag-btn { display:inline-block; width:16px; height:16px; border-radius:50%; border:1px solid #888; cursor:pointer; }
  .f0 { background: var(--f0); } .f1 { background: var(--f1); } .f2 { background: var(--f2); } .f3 { background: var(--f3); }
  .f4 { background: var(--f4); } .f5 { background: var(--f5); } .f6 { background: var(--f6); } .f7 { background: var(--f7); }
  .status-ok { color: #0a0; } .status-bad { color: #b00; }
</style>
<div class="container">
  <div class="toolbar">
    <div class="row">
      <input id="q" type="text" placeholder="Type to search your Anki notes…" autofocus />
      <label>Top-k</label><input id="k" type="number" min="1" max="200" value="20" />
      <label>Tag</label><input id="tag" type="text" value="semantic-match" style="width:200px;" />
      <button id="reload">Reload index</button>
      <button id="open-all">Open all</button>
    </div>
    <div class="row">
      <div id="ac" class="meta">AnkiConnect: checking…</div>
      <div id="status" class="meta"></div>
    </div>
  </div>
  <div id="out"></div>
</div>
<script>
const out = document.getElementById('out');
const qEl = document.getElementById('q');
const kEl = document.getElementById('k');
const tagEl = document.getElementById('tag');
const statusEl = document.getElementById('status');
const acEl = document.getElementById('ac');
const reloadBtn = document.getElementById('reload');
const openAllBtn = document.getElementById('open-all');
let t;
let lastResults = [];

function escapeRegExp(s){return s.replace(/[.*+?^${}()|[\]\\]/g,"\\$&");}
function highlight(text, q){
  const tokens = q.toLowerCase().split(/\s+/).filter(x => x.length > 2);
  let html = text;
  for (const t of tokens){
    const re = new RegExp("(" + escapeRegExp(t) + ")", "ig");
    html = html.replace(re, "<mark>$1</mark>");
  }
  return html;
}
function setStatus(msg, ok=true){
  statusEl.textContent = msg;
  statusEl.style.color = ok ? "#0a0" : "#b00";
  if (ok) setTimeout(()=>{ if (statusEl.textContent === msg) statusEl.textContent=""; }, 1500);
}

async function pingAC(){
  try{
    const res = await fetch("/anki/status");
    const data = await res.json();
    if (data.ok){ acEl.textContent = "AnkiConnect OK (v" + data.version + ")"; acEl.className = "meta status-ok"; }
    else { acEl.textContent = "AnkiConnect not available"; acEl.className = "meta status-bad"; }
  }catch(e){ acEl.textContent = "AnkiConnect not available"; acEl.className = "meta status-bad"; }
}

async function reloadIndex(){
  setStatus("Reloading index…", true);
  try{
    const res = await fetch("/reload", {method: "POST"});
    const data = await res.json();
    if (data.ok){ setStatus("Index reloaded (" + data.count + " notes)", true); if (qEl.value.trim()) search(); }
    else { setStatus("Reload failed: " + (data.error || "?"), false); }
  }catch(e){ setStatus("Reload error: " + e, false); }
}

async function openCard(cid){
  try{
    const r = await fetch("/anki/open_card?cid=" + cid);
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || "open failed");
  }catch(e){ setStatus("Open failed: " + e, false); }
}

async function openAll(){
  if (!lastResults.length) return;
  const cids = lastResults.map(r => r.cid).join(",");
  try{
    const r = await fetch("/anki/open_cards?cids=" + cids);
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || "open failed");
  }catch(e){ setStatus("Open all failed: " + e, false); }
}

async function setFlag(cid, flag){
  try{
    const r = await fetch("/anki/set_flag_card?cid=" + cid + "&flag=" + flag);
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || "set flag failed");
    const dot = document.getElementById("flag-dot-" + cid);
    if (dot){
      for (let i=0;i<=7;i++) dot.classList.remove("f"+i);
      dot.classList.add("f"+flag);
    }
    const shown = (typeof d.flag === "number") ? d.flag : flag;
    setStatus("Flag set to " + shown, true);
  }catch(e){
    setStatus("Set flag failed: " + e, false);
  }
}

async function addTagNote(nid){
  const tag = (tagEl.value || "").trim();
  if (!tag){ setStatus("Enter a tag in the Tag box", false); return; }
  try{
    const r = await fetch("/anki/add_tag_note?nid=" + nid + "&tag=" + encodeURIComponent(tag));
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || "add tag failed");
    setStatus("Tag added", true);
  }catch(e){ setStatus("Add tag failed: " + e, false); }
}

function flagButtonsHTML(cid){
  return `
    <span class="flag-btn f0" title="0" onclick="setFlag(${cid},0)"></span>
    <span class="flag-btn f1" title="1" onclick="setFlag(${cid},1)"></span>
    <span class="flag-btn f2" title="2" onclick="setFlag(${cid},2)"></span>
    <span class="flag-btn f3" title="3" onclick="setFlag(${cid},3)"></span>
    <span class="flag-btn f4" title="4" onclick="setFlag(${cid},4)"></span>
    <span class="flag-btn f5" title="5" onclick="setFlag(${cid},5)"></span>
    <span class="flag-btn f6" title="6" onclick="setFlag(${cid},6)"></span>
    <span class="flag-btn f7" title="7" onclick="setFlag(${cid},7)"></span>
  `;
}

function render(results, ankiOk){
  lastResults = results;
  const q = qEl.value;
  out.innerHTML = results.map(r => `
    <div class="result">
      <div class="meta">
        <span id="flag-dot-${r.cid}" class="flag-dot f${r.flag}"></span>
        cid ${r.cid} • nid ${r.nid} • sim ${r.similarity.toFixed(3)} ${r.deckName ? "• " + r.deckName : ""}
      </div>
      <div class="front">${ankiOk ? r.front : highlight(r.front, q)}</div>
      <div class="actions">
        <span class="meta">Flags:</span> ${flagButtonsHTML(r.cid)}
        <button onclick="openCard(${r.cid})">Open</button>
        <button onclick="addTagNote(${r.nid})">Add tag to note</button>
      </div>
    </div>
  `).join("");
}

async function search(){
  const q = qEl.value.trim();
  const k = parseInt(kEl.value || "20", 10);
  if (!q){ out.innerHTML = ""; setStatus(""); lastResults = []; return; }
  setStatus("Searching…", true);
  try{
    const res = await fetch("/search?q=" + encodeURIComponent(q) + "&k=" + k);
    const data = await res.json();
    render(data.results || [], !!data.anki_ok);
    setStatus((data.results || []).length + " cards", true);
  }catch(e){ setStatus("Error: " + e, false); }
}

qEl.addEventListener('input', () => { clearTimeout(t); t = setTimeout(search, 200); });
kEl.addEventListener('change', () => { clearTimeout(t); t = setTimeout(search, 10); });
reloadBtn.addEventListener('click', reloadIndex);
openAllBtn.addEventListener('click', openAll);

pingAC();
</script>
"""
    return HTMLResponse(content=html)