from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import operator
import os
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

# --------------------------------------------------------------------------- config
def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


MIN_DUR = float(os.getenv("MIN_PROMO_DURATION", "20"))
MAX_DUR = float(os.getenv("MAX_PROMO_DURATION", "60"))

MAX_CHARS = int(os.getenv("WINDOW_CHARS", "20000"))
MIN_GAP = float(os.getenv("MIN_SECONDS_BETWEEN_CALLS", "2.5"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "2"))

ENABLE_CROSS = _bool("ENABLE_CROSS_EPISODE", "true")
TWO_STAGE = _bool("TWO_STAGE_DISCOVERY", "true")        # discovery -> construction (2 LLM calls per episode window)
INCLUDE_CAMERA = _bool("INCLUDE_CAMERA", "false")

# continuity rules for multi-segment promos
MAX_SEG_GAP = float(os.getenv("MAX_SEGMENT_GAP", "15"))        # max skipped seconds between 2 segments of one episode
EDGE_SECONDS = float(os.getenv("EPISODE_EDGE_SECONDS", "15"))  # ep N segment must end within this of ep N's end; ep N+1 segment must start within this of its start
STITCH_TAIL = float(os.getenv("STITCH_TAIL_SECONDS", "90"))    # how much of the end of ep N the stitcher sees
STITCH_HEAD = float(os.getenv("STITCH_HEAD_SECONDS", "60"))    # how much of the start of ep N+1 it sees

# judge
JUDGE_BATCH = int(os.getenv("JUDGE_BATCH", "4"))                # max clips per judge call
JUDGE_MAX_CHARS = int(os.getenv("JUDGE_MAX_CHARS", "9000"))     # max transcript chars per judge call (free-tier TPM)
KEEP_UNSCORED = _bool("KEEP_UNSCORED", "false")                 # keep promos the judge could not score?
MIN_HOOK = int(os.getenv("MIN_HOOK_SCORE", "7"))
MIN_CLIFF = int(os.getenv("MIN_CLIFF_SCORE", "7"))
MIN_STANDALONE = int(os.getenv("MIN_STANDALONE_SCORE", "6"))

LLM_RETRIES = int(os.getenv("LLM_RETRIES", "4"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "3"))


# --------------------------------------------------------------------------- LLM (interchangeable)
@lru_cache(maxsize=1)
def get_llm():
    """Any LangChain chat model. Set LLM_PROVIDER + LLM_MODEL (+ LLM_BASE_URL for OpenAI-compatible hosts)."""
    provider, model = os.environ["LLM_PROVIDER"], os.environ["LLM_MODEL"]
    kwargs: dict[str, Any] = {"temperature": 0.2}
    if os.getenv("LLM_BASE_URL"):
        kwargs["base_url"] = os.environ["LLM_BASE_URL"]
    return init_chat_model(model, model_provider=provider, **kwargs)


_lock, _last = threading.Lock(), [0.0]


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


def _extract_json(s: str):
    s = re.sub(r"```(?:json)?", "", s)
    a, b = s.find("{"), s.rfind("}")
    return json.loads(s[a : b + 1])


def call_json(system: str, user: str, retries: int = LLM_RETRIES) -> dict | None:
    """Plain-text JSON prompting works on every provider (no tool/structured-output support needed)."""
    llm = get_llm()
    for attempt in range(retries):
        with _lock:
            wait = MIN_GAP - (time.time() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            return _extract_json(_text(resp.content))
        except Exception as e:  # rate limit, bad/truncated JSON, transient error
            msg = str(e)
            delay = RETRY_BASE_DELAY * 2**attempt
            if any(k in msg.lower() for k in ("429", "rate", "quota", "resource_exhausted")):
                delay = max(delay, 20)  # rate limits need a real cool-down
            print(f"  [llm retry {attempt + 1}/{retries}] {type(e).__name__}: {msg[:120]} (sleep {delay:.0f}s)")
            time.sleep(delay)
    return None


# --------------------------------------------------------------------------- screenplay parsing
# Format handled (per episode):
#   EPISODE NUMBER: Episode 1 / EPISODE LOGLINE: ... / CONTINUITY NOTES: ...
#   --- SCRIPT START ---
#   SCENE 01 / Scene heading: ... / Scene start: 00:00.00 / Scene end: 02:18.00
#   **CAMERA:** | **ACTION:** | **SOUND FX:** | **MATRIARCH (Cold, Commanding):**
#   Start: (00:11.30) | End: (00:13.10)
#   text / "dialogue"
# Timestamps are MM:SS.ss from the start of the episode. One file may hold one or many episodes.
@dataclass
class Event:
    start: float
    end: float
    kind: str  # dialogue | action | sound | camera | scene
    text: str  # already labelled, e.g. 'MATRIARCH (Cold): "Not anymore."' or '(action) ...'


BLOCK = re.compile(r"^\*\*(?P<who>.+?):\*\*\s*$")
TIMES = re.compile(r"Start:\s*\(?(?P<s>[\d:.]+)\)?\s*\|\s*End:\s*\(?(?P<e>[\d:.]+)\)?")
EP_HDR = re.compile(r"^EPISODE NUMBER:\s*(?:Episode\s*)?(?P<n>\d+)", re.I | re.M)
SCENE_LINE = re.compile(r"^(SCENE\s+\d+|Scene (heading|start|end):|Actions and dialogue:|--- SCRIPT)", re.I)


def ts_to_sec(t) -> float:
    if isinstance(t, (int, float)):
        return float(t)
    sec = 0.0
    for p in str(t).strip().strip("[]()").replace(",", ".").split(":"):
        sec = sec * 60 + float(p)
    return sec


def sec_to_ts(x: float) -> str:
    m, s = divmod(x, 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"


def parse_screenplay(raw: str) -> tuple[list[Event], dict]:
    meta = {}
    for key in ("EPISODE LOGLINE", "CONTINUITY NOTES"):
        m = re.search(rf"^{key}:\s*(.+)$", raw, re.M)
        meta[key] = m.group(1).strip() if m else ""
    body = raw.split("--- SCRIPT START ---", 1)[-1]

    blocks, scenes, cur, heading = [], [], None, ""
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if SCENE_LINE.match(line):
            cur = None
            if line.lower().startswith("scene heading:"):
                heading = line.split(":", 1)[1].strip()
            elif line.lower().startswith("scene start:") and heading:
                t = ts_to_sec(line.split(":", 1)[1])
                scenes.append(Event(t, t + 0.1, "scene", f"(scene) {heading}"))
            continue
        m = BLOCK.match(line)
        if m:
            cur = {"who": m["who"].strip(), "s": None, "e": None, "txt": []}
            blocks.append(cur)
            continue
        if cur is None:
            continue
        t = TIMES.search(line)
        if t and cur["s"] is None:
            cur["s"], cur["e"] = ts_to_sec(t["s"]), ts_to_sec(t["e"])
        else:
            cur["txt"].append(line)

    evs = list(scenes)
    for b in blocks:
        if b["s"] is None or not b["txt"]:
            continue
        txt = " ".join(b["txt"])
        base = b["who"].split("(")[0].strip().upper()
        if base == "CAMERA":
            if not INCLUDE_CAMERA:
                continue
            kind, label = "camera", f"(camera) {txt}"
        elif base == "ACTION":
            kind, label = "action", f"(action) {txt}"
        elif base in ("SOUND FX", "SFX", "SOUND", "MUSIC"):
            kind, label = "sound", f"(sound) {txt}"
        else:
            kind, label = "dialogue", f"{b['who']}: {txt}"
        evs.append(Event(b["s"], b["e"], kind, label))
    evs.sort(key=lambda e: (e.start, e.end))
    return evs, meta


def fmt(e: Event) -> str:
    return f"[{sec_to_ts(e.start)}-{sec_to_ts(e.end)}] {e.text}"


def windows(events: list[Event], max_chars: int) -> list[list[Event]]:
    out, cur, size = [], [], 0
    for e in events:
        n = len(fmt(e)) + 1
        if cur and size + n > max_chars:
            out.append(cur)
            cur = cur[-10:]  # overlap so cliffhangers on a boundary aren't lost
            size = sum(len(fmt(x)) + 1 for x in cur)
        cur.append(e)
        size += n
    if cur:
        out.append(cur)
    return out


def _bounds(ev: list[Event]) -> tuple[float, float]:
    return min(e.start for e in ev), max(e.end for e in ev)


# --------------------------------------------------------------------------- state
class State(TypedDict, total=False):
    input_dir: str
    out_dir: str
    episodes: dict[int, list[Event]]
    meta: dict[int, dict]
    raw: Annotated[list, operator.add]              # LLM single-episode candidates (parallel analyze nodes)
    moments: Annotated[list, operator.add]          # stage-1 discovered moments
    stitched: Annotated[list, operator.add]         # validated cross-episode promos (parallel stitch nodes)
    stitch_rejected: Annotated[list, operator.add]
    valid: list
    rejected: list
    final: list


# --------------------------------------------------------------------------- prompts
PROMO_DEF = """You are an expert short-drama (ReelShort / DramaBox style) promo editor.
A PROMO is a 20-60 second clip taken from the show to pull in NEW viewers on TikTok/Instagram/YouTube.
A good promo:
 1. HOOK: opens on something that stops a scroller in the first 3 seconds (slap, reveal, shocking line, betrayal, threat).
 2. CLIFFHANGER: cuts at an unresolved moment so the viewer must know what happens next. Never end on a resolution.
 3. SELF-CONTAINED: makes sense to someone who has never seen the show (who is who, what is at stake is clear from the clip).
Total length of all segments must be 20-60 seconds. Usually one continuous segment.
A promo does NOT have to start at the beginning of an episode or scene; start at the moment of impact."""

ANALYZE_SYS = PROMO_DEF + """

You get one episode: its logline, continuity notes, and screenplay lines formatted "[START-END] text"
(MM:SS.ss from episode start). Lines are dialogue (NAME (tone): "..."), (action), (sound), (camera) or (scene).
Timestamps you output MUST be copied from the lines given:
 - "start" = START time of the FIRST line included (ideally the hook line).
 - "end"   = END time of the LAST line included (ideally the cliffhanger line).
Return ONLY JSON:
{"promos": [ {"segments":[{"start":"MM:SS.ss","end":"MM:SS.ss"}], "hook":"what happens in first 3s", "cliffhanger":"the unresolved moment it ends on", "why":"one line on why it works"} ]}
Give 0-4 promos (quality over quantity; 0 is fine if nothing is strong). Prefer ONE continuous segment.""" + (
    f"\nUse a 2nd/3rd segment only to skip short filler (under {MAX_SEG_GAP:.0f}s) inside the same scene; "
    "the story must flow straight through the cut. Never jump around the episode."
)

MOMENT_TYPES = ("CONFLICT", "REVELATION", "TURNING_POINT", "THREAT", "UNANSWERED_QUESTION", "SUSPENSE")

DISCOVER_SYS = """You are a story analyst for a short-drama app. Read one episode and list its most promo-worthy MOMENTS.
This is DISCOVERY only: do NOT think about promo length, hooks or formatting yet, and do not worry about whether a
moment is self-contained. Cast a wide net, then rate each moment.
Look for:
 CONFLICT (confrontations, public humiliations, betrayals), REVELATION (unexpected reveals, secrets, hidden identities),
 TURNING_POINT (emotional shifts, a character's resolve changing), THREAT (threats, ultimatums, deadlines),
 UNANSWERED_QUESTION (lines or actions that raise a question the episode does not answer), SUSPENSE (tense, loaded dialogue).
Input lines are "[START-END] text" (MM:SS.ss from episode start). Timestamps MUST be copied from the lines:
 "start" = START of the first line of the moment, "end" = END of the last line of the moment (typically 5-40s).
Return ONLY JSON:
{"moments":[{"start":"MM:SS.ss","end":"MM:SS.ss","type":"CONFLICT","intensity":8,"summary":"what happens","open_question":"what a viewer would want answered, or empty"}]}
List 4-8 moments (fewer if the episode is quiet), in time order. type must be one of the types above."""

CONSTRUCT_SYS = PROMO_DEF + f"""

You get one episode (logline, continuity notes, screenplay lines "[START-END] text") plus a list of DISCOVERED MOMENTS
with ids. Turn the best moments into valid promos. For each promo:
 - Open on the hook: start at the line of impact (or the shortest set-up line that makes the conflict understandable);
   the first 3 seconds must grab a stranger.
 - Include only the context a newcomer needs (who is who, what is at stake). Cut filler and repeated pleading.
 - End on an unresolved narrative point: the line that raises the open question, BEFORE any answer or resolution.
 - Total 20-60s. Prefer ONE continuous segment; use more only to skip filler of under {MAX_SEG_GAP:.0f}s inside the same
   scene, in chronological order, with the story flowing straight through each cut.
 - Timestamps must be copied from the lines: "start" = START of the first included line, "end" = END of the last included line.
 - List the moment ids the promo is built on in "moment_ids". You may skip moments that cannot make a good promo,
   and you may build promos that span or combine neighbouring moments.
Return ONLY JSON:
{{"promos":[{{"moment_ids":["E01-m2"],"segments":[{{"start":"MM:SS.ss","end":"MM:SS.ss"}}],"hook":"what happens in first 3s","cliffhanger":"the unresolved moment it ends on","why":"one line on why it works"}}]}}
Give 0-6 promos; quality over quantity."""

STITCH_SYS = PROMO_DEF + f"""

You are shown the END of episode N and the START of episode N+1. A viewer watches them back-to-back, so the story
flows continuously across the boundary. Propose 0-2 promos that play CONTINUOUSLY across that boundary.
Rules:
 - Segment order: episode N segment(s) first, then episode N+1 segment(s). Max 3 segments in total.
 - The last segment in episode N must run to within {EDGE_SECONDS:.0f}s of that episode's end.
   The first segment in episode N+1 must start within {EDGE_SECONDS:.0f}s of that episode's start.
 - Two segments inside the same episode are allowed only if the skipped filler between them is under {MAX_SEG_GAP:.0f}s.
 - Total duration of all segments: 20-60s. Open on a hook; end on an unresolved cliffhanger.
 - Timestamps ("[START-END]") must be copied from the lines given. "start" = START of the first included line,
   "end" = END of the last included line of that segment.
Return ONLY JSON:
{{"stitched":[{{"segments":[{{"episode":N,"start":"MM:SS.ss","end":"MM:SS.ss"}}],"hook":"...","cliffhanger":"...","why":"one line"}}]}}
Return {{"stitched":[]}} if nothing flows continuously."""

JUDGE_SYS = PROMO_DEF + """

You are a strict reviewer. Several clips are given, numbered CLIP 1, CLIP 2, ... Score EACH clip 1-10 (integers):
 hook (does the first 3 seconds stop a scroller?), cliffhanger (does it end unresolved, demanding the next clip?),
 standalone (can a new viewer follow it with no prior knowledge?). Be harsh: 7+ means you would really post it.
Return ONLY JSON with exactly one entry per clip, using the clip NUMBER:
{"scores":[{"clip":1,"hook":8,"cliffhanger":7,"standalone":6,"problem":"main weakness, or empty"}]}"""


# --------------------------------------------------------------------------- nodes
def load_episodes(state: State):
    eps, meta = {}, {}
    for p in sorted(Path(state["input_dir"]).glob("*")):
        if p.suffix.lower() not in {".txt", ".md"}:
            continue
        raw = p.read_text(encoding="utf-8", errors="ignore")
        hdrs = list(EP_HDR.finditer(raw))
        if hdrs:  # supports one-file-per-episode or all episodes in one file
            chunks = [(int(h["n"]), raw[h.start() : hdrs[i + 1].start() if i + 1 < len(hdrs) else len(raw)])
                      for i, h in enumerate(hdrs)]
        else:
            m = re.search(r"(\d+)", p.stem)
            chunks = [(int(m.group(1)) if m else len(eps) + 1, raw)]
        for ep, txt in chunks:
            evs, mt = parse_screenplay(txt)
            if evs:
                eps[ep], meta[ep] = evs, mt
            else:
                print(f"! no timed blocks parsed for episode {ep} in {p.name}")
    print(f"loaded {len(eps)} episodes")
    return {"episodes": eps, "meta": meta}


def fan_out(state: State):
    return [Send("analyze", {"ep": ep, "events": ev, "meta": state["meta"][ep]})
            for ep, ev in sorted(state["episodes"].items())]


def _discover(ep: int, w: list[Event], ctx: str, offset: int) -> list[dict] | None:
    """Stage 1. Returns snapped moments with ids, or None if the LLM call/parse failed."""
    res = call_json(DISCOVER_SYS, ctx + "SCREENPLAY:\n" + "\n".join(fmt(e) for e in w))
    if not isinstance(res, dict) or not isinstance(res.get("moments"), list):
        return None
    out = []
    for m in res["moments"]:
        try:
            a = w[_snap_start(w, ts_to_sec(m["start"]))].start
            z = w[_snap_end(w, ts_to_sec(m["end"]))].end
        except Exception:
            continue
        if z <= a:
            continue
        mtype = str(m.get("type", "")).upper().replace(" ", "_")
        out.append({
            "id": f"E{ep:02d}-m{offset + len(out) + 1}", "episode": ep, "start_s": a, "end_s": z,
            "type": mtype if mtype in MOMENT_TYPES else "OTHER",
            "intensity": _to_score(m.get("intensity")) or 5,
            "summary": str(m.get("summary", "")), "open_question": str(m.get("open_question") or ""),
        })
    return out


def _moments_block(ms: list[dict]) -> str:
    lines = [f'{m["id"]} [{m["type"]}, intensity {m["intensity"]}] {sec_to_ts(m["start_s"])}-{sec_to_ts(m["end_s"])}: {m["summary"]}'
             + (f' | open question: {m["open_question"]}' if m["open_question"] else "") for m in ms]
    return "DISCOVERED MOMENTS (build promos from the best of these):\n" + "\n".join(lines) + "\n\n"


def analyze_episode(payload: dict):
    ep, events, meta = payload["ep"], payload["events"], payload["meta"]
    ctx = (f"EPISODE {ep}\nLOGLINE: {meta.get('EPISODE LOGLINE', '')}\n"
           f"CONTINUITY NOTES: {meta.get('CONTINUITY NOTES', '')}\n\n")
    raw, moments = [], []
    for w in windows(events, MAX_CHARS):
        text = "\n".join(fmt(e) for e in w)
        res = None
        if TWO_STAGE:
            found = _discover(ep, w, ctx, len(moments))
            if found is None:
                print(f"  ep {ep}: discovery failed, falling back to one-shot extraction")
            else:
                moments += found
                res = (call_json(CONSTRUCT_SYS, ctx + _moments_block(found) + "SCREENPLAY:\n" + text) if found else {})
        if res is None:  # one-shot mode, or a stage failed
            res = call_json(ANALYZE_SYS, ctx + "SCREENPLAY:\n" + text) or {}
        ids = {m["id"] for m in moments}
        for p in res.get("promos", []):
            if isinstance(p, dict):
                cited = p.get("moment_ids") if isinstance(p.get("moment_ids"), list) else []
                raw.append({"episode": ep, **p, "moment_ids": [i for i in cited if i in ids]})
    print(f"  ep {ep}: {len(moments)} moments -> {len(raw)} candidates")
    return {"raw": raw, "moments": moments}


def _snap_start(ev: list[Event], t: float) -> int:
    return min(range(len(ev)), key=lambda i: abs(ev[i].start - t))


def _snap_end(ev: list[Event], t: float) -> int:
    return min(range(len(ev)), key=lambda i: abs(ev[i].end - t))


def make_promo(pid: str, specs: list[tuple[int, float, float]], eps: dict[int, list[Event]], meta: dict):
    """specs: [(episode, start_sec, end_sec)] -> (promo|None, reject_reason|None).
    Enforces: real lines, chronological order, CONTINUITY between segments, total duration."""
    segs, transcript, total, prev = [], [], 0.0, None  # prev = (episode, end_sec)
    for ep, s, e in specs:
        if ep not in eps:
            return None, f"unknown episode {ep}"
        ev = eps[ep]
        a, b = ev[_snap_start(ev, s)].start, ev[_snap_end(ev, e)].end
        if b <= a:
            return None, "end before start"
        if prev:
            pe, pb = prev
            if ep == pe:
                gap = a - pb
                if gap < -0.05:
                    return None, "segments overlap or out of order"
                if gap > MAX_SEG_GAP:
                    return None, f"{gap:.0f}s gap between segments (story not continuous)"
            elif ep == pe + 1:
                if _bounds(eps[pe])[1] - pb > EDGE_SECONDS or a - _bounds(ev)[0] > EDGE_SECONDS:
                    return None, "episode jump not continuous (must run to end of ep N and resume at top of ep N+1)"
            else:
                return None, "segments not in the same or consecutive episodes"
        prev = (ep, b)
        total += b - a
        segs.append({"episode": ep, "start": sec_to_ts(a), "end": sec_to_ts(b), "start_s": a, "end_s": b})
        transcript.append(f"--- E{ep:02d} ---\n" + "\n".join(fmt(x) for x in ev if a - 0.05 <= x.start < b - 0.05))
    if not (MIN_DUR - 0.5 <= total <= MAX_DUR + 0.5):
        return None, f"duration {total:.0f}s outside {MIN_DUR:.0f}-{MAX_DUR:.0f}s"
    first = eps[segs[0]["episode"]]
    first3 = " | ".join(x.text for x in first if segs[0]["start_s"] - 0.05 <= x.start < segs[0]["start_s"] + 3)
    return {
        "id": pid, "segments": segs, "duration_s": round(total, 1),
        "hook": meta.get("hook", ""), "cliffhanger": meta.get("cliffhanger", ""), "why": meta.get("why", ""),
        "first_3s_text": first3, "transcript": "\n".join(transcript), "scores": None,
        "moment_ids": meta.get("moment_ids", []),
    }, None


def validate(state: State):
    eps, valid, rejected, counter = state["episodes"], [], [], {}
    for r in state.get("raw", []):
        ep = r["episode"]
        try:
            specs = [(ep, ts_to_sec(s["start"]), ts_to_sec(s["end"])) for s in r["segments"]]
        except Exception:
            rejected.append({"candidate": r, "reason": "unparseable segments"})
            continue
        counter[ep] = counter.get(ep, 0) + 1
        promo, why = make_promo(f"E{ep:02d}-p{counter[ep]}", specs, eps, r)
        (valid.append(promo) if promo else rejected.append({"candidate": r, "reason": why}))
    print(f"validated: {len(valid)} ok, {len(rejected)} rejected")
    return {"valid": valid, "rejected": rejected}


# ---- stitch: one LLM call per episode boundary (N -> N+1), seeing the real text on both sides
def fan_out_stitch(state: State):
    if not ENABLE_CROSS:
        return "judge"
    eps, meta = state["episodes"], state["meta"]
    sends = [
        Send("stitch", {"n": n, "episodes": {n: eps[n], n + 1: eps[n + 1]}, "meta": {n: meta[n], n + 1: meta[n + 1]}})
        for n in sorted(eps) if n + 1 in eps
    ]
    return sends or "judge"


def stitch_pair(p: dict):
    n, eps, meta = p["n"], p["episodes"], p["meta"]
    n_end = _bounds(eps[n])[1]
    n1_start = _bounds(eps[n + 1])[0]
    tail = [e for e in eps[n] if e.end >= n_end - STITCH_TAIL]
    head = [e for e in eps[n + 1] if e.start <= n1_start + STITCH_HEAD]
    text = (
        f"EPISODE N = {n}. Episode {n} ends at {sec_to_ts(n_end)}.\n"
        f"Episode {n} logline: {meta[n].get('EPISODE LOGLINE', '')}\n"
        f"Episode {n + 1} logline: {meta[n + 1].get('EPISODE LOGLINE', '')}\n\n"
        f"=== END OF EPISODE {n} ===\n" + "\n".join(fmt(e) for e in tail) +
        f"\n\n=== START OF EPISODE {n + 1} ===\n" + "\n".join(fmt(e) for e in head)
    )
    res = call_json(STITCH_SYS, text) or {}
    out, rej = [], []
    for k, s in enumerate(res.get("stitched", []), 1):
        if not isinstance(s, dict):
            continue
        try:
            specs = [(int(x["episode"]), ts_to_sec(x["start"]), ts_to_sec(x["end"])) for x in s["segments"]]
        except Exception:
            rej.append({"candidate": s, "reason": "stitch: unparseable segments"})
            continue
        if len({sp[0] for sp in specs}) < 2:
            rej.append({"candidate": s, "reason": "stitch: does not cross the episode boundary"})
            continue
        promo, why = make_promo(f"X{n:02d}-{k}", specs, eps, s)
        if promo:
            out.append(promo)
        else:
            rej.append({"candidate": s, "reason": f"stitch: {why}"})
    print(f"  stitch ep {n}->{n + 1}: {len(out)} ok, {len(rej)} rejected")
    return {"stitched": out, "stitch_rejected": rej}


# ---- judge: numbered clips, tolerant parsing, per-clip retry for anything left unscored
def _to_score(v) -> int | None:
    try:
        if isinstance(v, str):
            v = re.search(r"\d+(?:\.\d+)?", v).group(0)
        return max(1, min(10, int(round(float(v)))))
    except Exception:
        return None


def _parse_scores(res, batch: list[dict]) -> dict[int, dict]:
    """Return {index_in_batch: scores}. Maps by clip number; falls back to position if counts match."""
    items = res.get("scores") if isinstance(res, dict) else None
    if not isinstance(items, list):
        return {}
    out = {}
    for pos, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        c, idx = it.get("clip"), None
        if isinstance(c, (int, float)) or (isinstance(c, str) and c.strip().isdigit()):
            idx = int(c) - 1
        elif len(items) == len(batch):
            idx = pos
        if idx is None or not 0 <= idx < len(batch):
            continue
        h, cl, st = _to_score(it.get("hook")), _to_score(it.get("cliffhanger")), _to_score(it.get("standalone"))
        if None in (h, cl, st):
            continue
        out[idx] = {"hook": h, "cliffhanger": cl, "standalone": st, "problem": str(it.get("problem", "") or "")}
    return out


def _score(batch: list[dict]) -> dict[int, dict]:
    body = "\n\n".join(
        f'CLIP {i + 1} ({p["duration_s"]}s)\nFIRST 3 SECONDS: {p["first_3s_text"]}\n{p["transcript"]}'
        for i, p in enumerate(batch)
    )
    return _parse_scores(call_json(JUDGE_SYS, body), batch)


def _batches(promos: list[dict]):
    cur, size = [], 0
    for p in promos:
        n = len(p["transcript"]) + len(p["first_3s_text"]) + 60
        if cur and (len(cur) >= JUDGE_BATCH or size + n > JUDGE_MAX_CHARS):
            yield cur
            cur, size = [], 0
        cur.append(p)
        size += n
    if cur:
        yield cur


def judge(state: State):
    promos = list(state.get("valid", [])) + list(state.get("stitched", []))
    for batch in _batches(promos):                     # pass 1: small batches
        for i, s in _score(batch).items():
            batch[i]["scores"] = s
    for p in promos:                                   # pass 2: anything unscored is re-judged on its own
        if not p.get("scores"):
            p["scores"] = _score([p]).get(0)
    failed = [p["id"] for p in promos if not p.get("scores")]
    print(f"judged: {len(promos) - len(failed)}/{len(promos)} scored" + (f"; FAILED: {failed}" if failed else ""))
    return {"valid": promos}


def _overlap(a: dict, b: dict) -> float:
    ov = 0.0
    for x in a["segments"]:
        for y in b["segments"]:
            if x["episode"] == y["episode"]:
                ov += max(0.0, min(x["end_s"], y["end_s"]) - max(x["start_s"], y["start_s"]))
    return ov / max(1e-6, min(a["duration_s"], b["duration_s"]))


def _total(p):  # only breaks ties when de-duplicating; real ranking is Task 2
    s = p.get("scores")
    return (s["hook"] + s["cliffhanger"] + s["standalone"]) if s else 0


def finalize(state: State):
    rejected, kept = list(state.get("rejected", [])) + list(state.get("stitch_rejected", [])), []
    for p in state["valid"]:
        s = p.get("scores")
        if not s:
            if KEEP_UNSCORED:
                p["judge_failed"] = True
                kept.append(p)
            else:
                rejected.append({"candidate": p, "reason": "judge failed: no valid scores after retries"})
            continue
        if s["hook"] < MIN_HOOK or s["cliffhanger"] < MIN_CLIFF or s["standalone"] < MIN_STANDALONE:
            rejected.append({"candidate": p, "reason": f"judge scores {s}"})
        else:
            kept.append(p)
    kept.sort(key=_total, reverse=True)
    final = []
    for p in kept:
        if all(_overlap(p, q) < 0.5 for q in final):
            final.append(p)
    final.sort(key=lambda p: (p["segments"][0]["episode"], p["segments"][0]["start_s"]))

    out = Path(state.get("out_dir", "out"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "promos.json").write_text(json.dumps(
        [{k: v for k, v in p.items() if k != "transcript"} for p in final], indent=2, ensure_ascii=False))
    (out / "rejected.json").write_text(json.dumps(rejected, indent=2, ensure_ascii=False, default=str))
    cited = {i for r in state.get("raw", []) for i in r.get("moment_ids", [])}
    used = {i for p in final for i in p.get("moment_ids", [])}
    mo = [{**m, "start": sec_to_ts(m["start_s"]), "end": sec_to_ts(m["end_s"]),
           "status": "in_final" if m["id"] in used else "candidate_dropped" if m["id"] in cited else "not_used"}
          for m in state.get("moments", [])]
    (out / "moments.json").write_text(json.dumps(mo, indent=2, ensure_ascii=False))
    lines = ["# Promos\n"]
    for p in final:
        segs = "; ".join(f'Ep {s["episode"]} {s["start"]}-{s["end"]}' for s in p["segments"])
        lines += [f'## {p["id"]} - {segs} ({p["duration_s"]}s)',
                  f'- **Hook:** {p["hook"]}', f'- **Cliffhanger:** {p["cliffhanger"]}',
                  f'- **Why it works:** {p["why"]}', f'- Judge: {p.get("scores")}', ""]
    (out / "promos.md").write_text("\n".join(lines))
    if mo:
        print("stage-1 moments: " + ", ".join(f"{k}={v}" for k, v in sorted(
            {s_: sum(m["status"] == s_ for m in mo) for s_ in ("in_final", "candidate_dropped", "not_used")}.items())))
    print(f"FINAL: {len(final)} promos -> {out}/promos.json, promos.md")
    return {"final": final}


# --------------------------------------------------------------------------- graph
def build_graph():
    g = StateGraph(State)
    g.add_node("load", load_episodes)
    g.add_node("analyze", analyze_episode)
    g.add_node("validate", validate)
    g.add_node("stitch", stitch_pair)
    g.add_node("judge", judge)
    g.add_node("finalize", finalize)
    g.add_edge(START, "load")
    g.add_conditional_edges("load", fan_out, ["analyze"])
    g.add_edge("analyze", "validate")
    g.add_conditional_edges("validate", fan_out_stitch, ["stitch", "judge"])
    g.add_edge("stitch", "judge")
    g.add_edge("judge", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.getenv("INPUT_DIR", "Input"))
    ap.add_argument("--out", default=os.getenv("OUTPUT_DIR", "out"))
    ap.add_argument("--parse-only", action="store_true", help="check the parser, no LLM calls")
    a = ap.parse_args()
    if a.parse_only:
        st = load_episodes({"input_dir": a.input})
        for ep, ev in sorted(st["episodes"].items()):
            print(f"ep {ep}: {len(ev)} events, {ev[0].start:.1f}s -> {max(e.end for e in ev):.1f}s | {fmt(ev[min(1, len(ev) - 1)])[:100]}")
        raise SystemExit
    build_graph().invoke(
        {"input_dir": a.input, "out_dir": a.out, "rejected": []},
        config={"max_concurrency": MAX_CONCURRENCY, "recursion_limit": 300},
    )

# --------------------------------------------------------------------------- README
# pip install langchain langgraph python-dotenv langchain-google-genai langchain-groq langchain-openai langchain-mistralai langchain-ollama
#
# Pick ANY provider by env vars / .env (code never changes):
#   Gemini (free tier):    LLM_PROVIDER=google_genai  LLM_MODEL=<gemini flash model>   GOOGLE_API_KEY=...
#   Groq (free tier):      LLM_PROVIDER=groq          LLM_MODEL=<model>                GROQ_API_KEY=...
#   Mistral (free tier):   LLM_PROVIDER=mistralai     LLM_MODEL=<model>                MISTRAL_API_KEY=...
#   OpenRouter (:free):    LLM_PROVIDER=openai        LLM_MODEL=<model>:free  LLM_BASE_URL=https://openrouter.ai/api/v1  OPENAI_API_KEY=<openrouter key>
#   Local Ollama:          LLM_PROVIDER=ollama        LLM_MODEL=<model>
#
# 1) python promo_finder.py --input Input --parse-only      # verify parsing first (no API calls)
# 2) python promo_finder.py --input Input --out out
#
# Two-stage discovery: TWO_STAGE_DISCOVERY=false goes back to one call per episode (half the API calls).
# Tuning: MIN_SECONDS_BETWEEN_CALLS, WINDOW_CHARS, MAX_CONCURRENCY=1, ENABLE_CROSS_EPISODE=false, INCLUDE_CAMERA=true
# Continuity: MAX_SEGMENT_GAP (default 15s), EPISODE_EDGE_SECONDS (15s), STITCH_TAIL_SECONDS (90), STITCH_HEAD_SECONDS (60)
# Judge: JUDGE_BATCH (4), JUDGE_MAX_CHARS (9000), KEEP_UNSCORED=false