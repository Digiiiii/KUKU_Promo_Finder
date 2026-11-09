"""
rank_and_audit.py - Task 2 (rank the promos) and Task 3 (where does the tool pick badly?).

Run AFTER promo_finder.py (reads out/promos.json + out/rejected.json and re-reads the screenplays).
Reuses promo_finder's provider-agnostic LLM wrapper, parser and helpers, so the same .env works.

  python rank_and_audit.py --input Input --out out            # rank + audit
  python rank_and_audit.py --rank-only | --audit-only
  python rank_and_audit.py --no-llm-audit                      # audit with deterministic checks only (no API calls)
  python rank_and_audit.py --labels out/audit_labels.csv       # score your hand labels (see audit.md)

RANKING  = LLM "head of growth" pass (lead score, premise clarity, spoiler risk) blended with the judge's scores,
           small rule-based adjustments (entry-point episode, spoilers, length), then a diversity-aware greedy
           pick so the top of the list is not five versions of the same scene. Top-K get a written "why".
AUDIT    = funnel + distribution checks, per-promo deterministic flags, judge-noise check, an adversarial critic
           on a sample, a proxy recall check (high-intensity windows nobody picked), and a hand-label template.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics as st
from collections import Counter
from pathlib import Path

import promo_finder as pf

# --------------------------------------------------------------------------- config
W_HOOK, W_CLIFF, W_STAND = (float(os.getenv(k, d)) for k, d in
                            (("W_HOOK", "0.4"), ("W_CLIFF", "0.3"), ("W_STANDALONE", "0.3")))
EARLY_EP = int(os.getenv("EARLY_EPISODE", "3"))   # promos from these episodes get an entry-point bonus
TOP_K = int(os.getenv("RANK_TOP_K", "10"))
CRITIC_N = int(os.getenv("CRITIC_SAMPLE", "12"))
CRITIC_BATCH = int(os.getenv("CRITIC_BATCH", "3"))

STOP = set("the a an and or of to in on at with for his her their is are was be as by from that this it he she "
           "they him them over into then who what while after before out up down not no but".split())
TONE_WORDS = set("outraged accusing shocked gasping screaming furious crying horrified devastated trembling desperate "
                 "begging threatening gloating menacing cruel shouting sobbing enraged panicked stunned betrayed "
                 "heartbroken pleading venomous seething terrified".split())

# --------------------------------------------------------------------------- shared helpers
def promo_text(p: dict, eps: dict) -> str:
    parts = []
    for s in p["segments"]:
        ev = eps[s["episode"]]
        body = "\n".join(pf.fmt(x) for x in ev if s["start_s"] - 0.05 <= x.start < s["end_s"] - 0.05)
        parts.append(f"--- E{s['episode']:02d} ---\n{body}")
    return "\n".join(parts)


def load_inputs(inp: str, out: str):
    loaded = pf.load_episodes({"input_dir": inp})
    eps, meta = loaded["episodes"], loaded["meta"]
    promos = json.loads((Path(out) / "promos.json").read_text())
    rej_path = Path(out) / "rejected.json"
    rejected = json.loads(rej_path.read_text()) if rej_path.exists() else []
    for p in promos:
        p["transcript"] = promo_text(p, eps)
    return eps, meta, promos, rejected


def parse_numbered(res, key: str, n: int) -> dict[int, dict]:
    items = res.get(key) if isinstance(res, dict) else None
    if not isinstance(items, list):
        return {}
    out = {}
    for pos, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        c, idx = it.get("clip"), None
        if isinstance(c, (int, float)) or (isinstance(c, str) and c.strip().isdigit()):
            idx = int(c) - 1
        elif len(items) == n:
            idx = pos
        if idx is not None and 0 <= idx < n:
            out[idx] = it
    return out


def run_numbered(batches, system, key, parse_item, header="") -> dict[str, dict]:
    """LLM over numbered clips; anything missing/unparseable is retried one clip at a time."""
    result: dict[str, dict] = {}

    def attempt(batch):
        body = header + "\n\n".join(f'CLIP {i + 1} ({p["duration_s"]}s)\n{p["transcript"]}' for i, p in enumerate(batch))
        for i, it in parse_numbered(pf.call_json(system, body), key, len(batch)).items():
            parsed = parse_item(it)
            if parsed:
                result[batch[i]["id"]] = parsed

    for b in batches:
        attempt(b)
    for p in [p for b in batches for p in b]:
        if p["id"] not in result:
            attempt([p])
    return result


def chunks(lst, k):
    return [lst[i : i + k] for i in range(0, len(lst), k)]


def words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", (s or "").lower()) if w not in STOP and len(w) > 2}


def jacc(a: str, b: str) -> float:
    x, y = words(a), words(b)
    return len(x & y) / len(x | y) if x and y else 0.0


# =========================================================================== TASK 2: RANKING
RANK_SYS = """You are the head of growth at a micro-drama app (ReelShort / DramaBox style). You must choose which
promo clips from this show to post FIRST on TikTok/Instagram/YouTube to acquire new viewers.
For each numbered clip give:
 lead_score 1-10: how strongly this clip would perform as a LEAD promo for a cold audience (hook, emotional spike, curiosity gap).
 premise_clarity 1-10: after watching, does a stranger understand the core conflict and want to start Episode 1?
 spoiler_risk: low | medium | high (does it give away a late twist or resolve the central conflict, removing the reason to watch?)
 reason: one short line.
Be discriminating: use the full 1-10 range.
Return ONLY JSON: {"ranks":[{"clip":1,"lead_score":8,"premise_clarity":7,"spoiler_risk":"low","reason":"..."}]}"""

WHY_SYS = """You are the head of growth at a micro-drama app. You are given promos ALREADY in final posting order.
For each, write 1-2 concrete sentences on why it sits at that rank (reference its hook/cliffhanger and, where useful,
why it beats the next one). Then write a 2-3 sentence posting plan (what goes out first, what is held back and why).
Return ONLY JSON: {"plan":"...","items":[{"rank":1,"why":"..."}]}"""


def _parse_rank(it: dict) -> dict | None:
    lead, prem = pf._to_score(it.get("lead_score")), pf._to_score(it.get("premise_clarity"))
    if lead is None or prem is None:
        return None
    sp = str(it.get("spoiler_risk", "low")).lower()
    sp = "high" if sp.startswith("h") else "medium" if sp.startswith("m") else "low"
    return {"lead_score": lead, "premise_clarity": prem, "spoiler_risk": sp, "reason": str(it.get("reason", ""))}


def composite(p: dict, ed: dict | None) -> tuple[float, dict]:
    s = p.get("scores")
    judge = (W_HOOK * s["hook"] + W_CLIFF * s["cliffhanger"] + W_STAND * s["standalone"]) / (W_HOOK + W_CLIFF + W_STAND) if s else None
    if ed and judge is not None:
        base = 0.4 * judge + 0.4 * ed["lead_score"] + 0.2 * ed["premise_clarity"]
    elif ed:
        base = 0.7 * ed["lead_score"] + 0.3 * ed["premise_clarity"]
    else:
        base = judge if judge is not None else 5.0
    adj, notes = 0.0, []
    if min(seg["episode"] for seg in p["segments"]) <= EARLY_EP:
        adj += 0.5
        notes.append(f"+0.5 entry-point episode (<= Ep {EARLY_EP})")
    sp = ed["spoiler_risk"] if ed else "low"
    pen = {"low": 0.0, "medium": 0.4, "high": 1.0}[sp]
    if pen:
        adj -= pen
        notes.append(f"-{pen} spoiler risk {sp}")
    if p["duration_s"] > 50:
        adj -= 0.3
        notes.append("-0.3 longer than 50s")
    return round(base + adj, 2), {"judge": None if judge is None else round(judge, 2), "adjustments": notes}


def _same_moment(p: dict, q: dict) -> bool:
    return any(a["episode"] == b["episode"] and abs(a["start_s"] - b["start_s"]) < 30
               for a in p["segments"] for b in q["segments"])


def _same_episode(p: dict, q: dict) -> bool:
    return bool({s["episode"] for s in p["segments"]} & {s["episode"] for s in q["segments"]})


def diversify(items: list[dict]) -> list[dict]:
    """Greedy pick: highest composite first, penalising near-duplicates of what is already chosen."""
    pool, chosen = items[:], []
    while pool:
        def eff(p):
            pen = 0.0
            for q in chosen:
                if _same_moment(p, q):
                    pen += 1.5
                elif _same_episode(p, q):
                    pen += 0.25
                if jacc(p["cliffhanger"], q["cliffhanger"]) >= 0.5:
                    pen += 0.5
            return p["composite"] - pen
        best = max(pool, key=eff)
        best["picked_score"] = round(eff(best), 2)
        chosen.append(best)
        pool.remove(best)
    return chosen


def rank(promos: list[dict], meta: dict, out: Path) -> list[dict]:
    first = min(meta) if meta else None
    header = ""
    if first is not None:
        header = (f"SHOW CONTEXT (Episode {first}): {meta[first].get('EPISODE LOGLINE', '')} "
                  f"{meta[first].get('CONTINUITY NOTES', '')}\n\n")
    editor = run_numbered(list(pf._batches(promos)), RANK_SYS, "ranks", _parse_rank, header)
    print(f"editor pass: {len(editor)}/{len(promos)} scored")
    for p in promos:
        ed = editor.get(p["id"])
        p["editor"] = ed
        p["composite"], p["breakdown"] = composite(p, ed)
    ordered = diversify(sorted(promos, key=lambda p: p["composite"], reverse=True))
    for i, p in enumerate(ordered, 1):
        p["rank"] = i

    # written reasons for the top K
    top = ordered[:TOP_K]
    def line(p):
        where = "; ".join("Ep%d %s-%s" % (x["episode"], x["start"], x["end"]) for x in p["segments"])
        sc, ed = p.get("scores") or {}, p["editor"] or {}
        return (f'#{p["rank"]} {p["id"]} | {where} | {p["duration_s"]}s | hook: {p["hook"]} | cliffhanger: {p["cliffhanger"]} | '
                f'judge H/C/S: {sc.get("hook")}/{sc.get("cliffhanger")}/{sc.get("standalone")} | '
                f'editor lead/premise/spoiler: {ed.get("lead_score")}/{ed.get("premise_clarity")}/{ed.get("spoiler_risk")} | '
                f'notes: {", ".join(p["breakdown"]["adjustments"]) or "none"}')

    listing = "\n".join(line(p) for p in top)
    res = pf.call_json(WHY_SYS, listing) or {}
    why = {}
    for it in res.get("items", []) if isinstance(res, dict) else []:
        try:
            why[int(it["rank"])] = str(it["why"])
        except Exception:
            continue
    plan = str(res.get("plan", "")) if isinstance(res, dict) else ""
    for p in top:
        p["why_rank"] = why.get(p["rank"]) or (
            f'Composite {p["composite"]}: ' + (p["editor"] or {}).get("reason", "") + " " + " ".join(p["breakdown"]["adjustments"])).strip()
    if not plan:
        plan = "(LLM plan unavailable - ordering is purely score-based.)"

    slim = [{k: v for k, v in p.items() if k != "transcript"} for p in ordered]
    (out / "ranking.json").write_text(json.dumps({"plan": plan, "ranked": slim}, indent=2, ensure_ascii=False))
    lines = ["# Promo ranking - what to post first\n", f"**Posting plan:** {plan}\n",
             "| # | Promo | Where | Dur | Hook | Cliffhanger | Score | Why here |", "|---|---|---|---|---|---|---|---|"]
    for p in ordered[:TOP_K]:
        where = "; ".join(f'Ep {s["episode"]} {s["start"]}-{s["end"]}' for s in p["segments"])
        lines.append(f'| {p["rank"]} | {p["id"]} | {where} | {p["duration_s"]}s | {p["hook"]} | {p["cliffhanger"]} | '
                     f'{p["composite"]} | {p["why_rank"]} |')
    lines += ["", f"Full ordered list ({len(ordered)} promos) in ranking.json. Score = 0.4*judge + 0.4*editor lead + 0.2*premise clarity, "
              f"then +0.5 entry-point episode, -0.4/-1.0 spoiler risk, -0.3 if >50s, then diversity-aware greedy ordering."]
    (out / "ranking.md").write_text("\n".join(lines))
    print(f"ranking written: {out}/ranking.md, ranking.json")
    return ordered


# =========================================================================== TASK 3: QUALITY AUDIT
CRITIC_SYS = pf.PROMO_DEF + """

You are an ADVERSARIAL QA reviewer. Your job is to find what is wrong with each clip, as a viewer who has never seen the show.
For each numbered clip return:
 verdict: good | fixable | bad
 tags: zero or more of WEAK_HOOK (first 3s would not stop a scroll), RESOLVED_ENDING (not a real cliffhanger),
       NEEDS_CONTEXT (relationships/stakes unclear without prior episodes), SLOW_SETUP, SPOILER (gives away a big twist),
       ABRUPT_CUT (starts/ends mid-thought), REPETITIVE (same beat repeated, no escalation)
 explanation: one line
 fix: concrete edit (e.g. "start at the slap line", "end 6s earlier", or "drop")
If a clip is genuinely strong, say good with no tags. Do not be polite.
Return ONLY JSON: {"reviews":[{"clip":1,"verdict":"fixable","tags":["WEAK_HOOK"],"explanation":"...","fix":"..."}]}"""

TAGS = {"WEAK_HOOK", "RESOLVED_ENDING", "NEEDS_CONTEXT", "SLOW_SETUP", "SPOILER", "ABRUPT_CUT", "REPETITIVE"}


def _parse_review(it: dict) -> dict | None:
    v = str(it.get("verdict", "")).lower()
    if v not in ("good", "fixable", "bad"):
        return None
    tags = [str(t).upper() for t in (it.get("tags") or []) if str(t).upper() in TAGS]
    return {"verdict": v, "tags": tags, "explanation": str(it.get("explanation", "")), "fix": str(it.get("fix", ""))}


def _max_gap(core):
    cur_end, mg = None, 0.0
    for e in core:
        if cur_end is not None:
            mg = max(mg, e.start - cur_end)
        cur_end = e.end if cur_end is None else max(cur_end, e.end)
    return mg


def det_flags(p: dict, eps: dict) -> list[str]:
    flags = []
    segs = p["segments"]
    a = segs[0]["start_s"]
    early = [e for e in eps[segs[0]["episode"]] if a - 0.05 <= e.start < a + 3 and e.kind in ("dialogue", "action")]
    if not early:
        flags.append("silent_open: no dialogue/action in first 3s")
    else:
        d = [e for e in early if e.kind == "dialogue"]
        if d:
            line = d[0].text.split(":", 1)[-1].strip().strip('"')
            if len(line.split()) <= 2 and "!" not in line:
                flags.append(f"reply_fragment_open: starts on '{line}' (needs the line before it)")
    last = segs[-1]
    if any(e.kind in ("dialogue", "action") and e.start < last["end_s"] - 0.05 and e.end > last["end_s"] + 0.5
           for e in eps[last["episode"]]):
        flags.append("cuts_through_overlapping_beat")
    names = set()
    for s in segs:
        core = [e for e in eps[s["episode"]] if e.kind in ("dialogue", "action")
                and s["start_s"] - 0.05 <= e.start < s["end_s"] - 0.05]
        g = _max_gap(core)
        if g > 4:
            flags.append(f"dead_air: {g:.1f}s with nothing happening")
        names |= {re.sub(r"\s*\(.*?\)", "", e.text.split(":", 1)[0]).strip() for e in core if e.kind == "dialogue"}
    if len(names) >= 4:
        flags.append(f"crowded_cast: {len(names)} speakers (hard to follow cold)")
    return flags


def bucket(reason: str) -> str:
    r = reason.lower()
    for k, v in (("judge scores", "judge: below threshold"), ("judge failed", "judge: could not score"),
                 ("duration", "validation: duration out of range"), ("gap between", "validation: not continuous (gap)"),
                 ("episode jump", "validation: not continuous (episode jump)"),
                 ("does not cross", "stitch: did not cross boundary"), ("overlap", "validation: overlap/order"),
                 ("unparseable", "validation: unparseable"), ("consecutive", "validation: non-consecutive episodes")):
        if k in r:
            return v
    return "other"


def intensity(e: pf.Event) -> float:
    if e.kind not in ("dialogue", "action"):
        return 0.0
    t, s = e.text, 0.0
    s += 2 * t.count("!") + t.count("?")
    m = re.match(r"[^:]*\(([^)]*)\)", t) if e.kind == "dialogue" else None
    if m:
        s += sum(1 for w in re.findall(r"[a-z]+", m.group(1).lower()) if w in TONE_WORDS)
    if e.kind == "dialogue" and len(t.split(":", 1)[-1].split()) <= 4:
        s += 0.5
    return s


def uncovered_peaks(eps: dict, promos: list[dict], win=30.0, step=5.0) -> list[dict]:
    """Proxy recall check: the most 'intense' 30s window per episode (punctuation + tone tags); is it covered?"""
    best = []
    for ep, ev in eps.items():
        lo, hi = pf._bounds(ev)
        top, t = (-1.0, lo), lo
        while t + win <= hi + 1e-6:
            sc = sum(intensity(e) for e in ev if t <= e.start < t + win)
            if sc > top[0]:
                top = (sc, t)
            t += step
        sc, t = top
        cov = 0.0
        for p in promos:
            for s in p["segments"]:
                if s["episode"] == ep:
                    cov = max(cov, max(0.0, min(s["end_s"], t + win) - max(s["start_s"], t)) / win)
        best.append({"episode": ep, "start": pf.sec_to_ts(t), "end": pf.sec_to_ts(t + win), "intensity": round(sc, 1), "covered": round(cov, 2)})
    scores = sorted(b["intensity"] for b in best)
    cut = scores[int(0.75 * (len(scores) - 1))] if scores else 0
    return sorted([b for b in best if b["covered"] < 0.4 and b["intensity"] >= cut], key=lambda b: -b["intensity"])


def score_labels(path: str, promos: list[dict]) -> dict:
    by_id = {p["id"]: p for p in promos}
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if (r.get("human_verdict") or "").strip().lower() in ("good", "ok", "bad")]
    if not rows:
        return {"note": "no filled human_verdict (good/ok/bad) rows found"}
    verdicts = Counter(r["human_verdict"].strip().lower() for r in rows)
    mean_by = {}
    for v in verdicts:
        tot = [(by_id[r["id"]].get("scores") or {}) for r in rows if r["human_verdict"].strip().lower() == v and r["id"] in by_id]
        vals = [(s["hook"] + s["cliffhanger"] + s["standalone"]) / 3 for s in tot if s]
        mean_by[v] = round(st.mean(vals), 2) if vals else None
    bad_high = [r["id"] for r in rows if r["human_verdict"].strip().lower() == "bad" and r["id"] in by_id
                and (by_id[r["id"]].get("scores") or {}) and (sum(by_id[r["id"]]["scores"][k] for k in ("hook", "cliffhanger", "standalone")) / 3) >= 8]
    return {"labelled": len(rows), "verdicts": dict(verdicts), "precision_good_or_ok": round((verdicts["good"] + verdicts["ok"]) / len(rows), 2),
            "mean_judge_score_by_human_verdict": mean_by, "judge_loved_but_human_said_bad": bad_high}


def audit(eps: dict, promos: list[dict], rejected: list[dict], ranked: list[dict] | None, out: Path,
          use_llm: bool, labels: str | None):
    A: dict = {}
    findings: list[str] = []
    n = len(promos)

    # 1. funnel
    reasons = Counter(bucket(r.get("reason", "")) for r in rejected)
    A["funnel"] = {"kept": n, "rejected": len(rejected), "rejected_by_reason": dict(reasons)}
    thr = {"hook": pf.MIN_HOOK, "cliffhanger": pf.MIN_CLIFF, "standalone": pf.MIN_STANDALONE}
    border = 0
    for r in rejected:
        c = r.get("candidate", {})
        s = c.get("scores") if isinstance(c, dict) else None
        if s and "judge scores" in r.get("reason", ""):
            if sum(max(0, thr[k] - s[k]) for k in thr) <= 1:
                border += 1
    A["funnel"]["borderline_judge_rejects"] = border
    if border:
        findings.append(f"{border} promos were rejected by the judge by only 1 point - the thresholds ({thr}) decide a lot; review them in rejected.json.")
    near = [r for r in rejected if bucket(r.get("reason", "")) in
            ("validation: duration out of range", "validation: not continuous (gap)", "validation: not continuous (episode jump)")]
    if near:
        findings.append(f"{len(near)} good-sounding ideas died on mechanical rules (duration/continuity), not on quality - candidates for auto-repair (trim/extend to snap into 20-60s).")

    # 2. distributions
    durs = [p["duration_s"] for p in promos]
    per_ep = Counter(s["episode"] for p in promos for s in p["segments"][:1])
    A["distribution"] = {
        "duration_buckets": dict(Counter(f"{int(d // 10) * 10}-{int(d // 10) * 10 + 10}s" for d in durs)),
        "episodes_with_no_promo": sorted(set(eps) - set(per_ep)),
        "promos_per_episode": dict(sorted(per_ep.items())),
    }
    if n:
        if sum(d >= 55 for d in durs) / n > 0.4:
            findings.append(f"{sum(d >= 55 for d in durs)}/{n} promos run >= 55s: the tool drifts to max length instead of cutting tight (short-form favours ~25-45s).")
        top3 = sum(c for _, c in per_ep.most_common(3))
        if len(eps) >= 10 and top3 / n > 0.4:
            findings.append(f"Top 3 episodes hold {top3}/{n} promos: coverage is concentrated, {len(A['distribution']['episodes_with_no_promo'])} of {len(eps)} episodes have none.")
        thirds = Counter()
        for p in promos:
            s = p["segments"][0]
            lo, hi = pf._bounds(eps[s["episode"]])
            thirds[min(2, int(3 * (s["start_s"] - lo) / max(1e-6, hi - lo)))] += 1
        A["distribution"]["start_position_in_episode_thirds"] = {["first", "middle", "last"][k]: v for k, v in sorted(thirds.items())}
        tot = [(p["scores"]["hook"] + p["scores"]["cliffhanger"] + p["scores"]["standalone"]) / 3 for p in promos if p.get("scores")]
        if len(tot) > 3 and st.pstdev(tot) < 0.6:
            findings.append("Judge scores barely vary among kept promos (stdev < 0.6): the judge is not discriminating, so scores alone cannot rank well.")

    # 3. deterministic flags
    flags = {p["id"]: det_flags(p, eps) for p in promos}
    A["deterministic_flags"] = {k: v for k, v in flags.items() if v}
    kinds = Counter(f.split(":")[0] for v in flags.values() for f in v)
    A["deterministic_flag_counts"] = dict(kinds)
    for k, c in kinds.items():
        if n and c / n >= 0.15:
            findings.append(f"{c}/{n} promos trigger '{k}'.")

    # 4. repetitiveness
    dup = [(a["id"], b["id"]) for i, a in enumerate(promos) for b in promos[i + 1:] if jacc(a["cliffhanger"], b["cliffhanger"]) >= 0.6]
    common = Counter(w for p in promos for w in words(p["hook"] + " " + p["cliffhanger"]))
    A["repetition"] = {"near_duplicate_cliffhanger_pairs": dup[:20], "most_common_hook_cliff_words": common.most_common(8)}
    if dup:
        findings.append(f"{len(dup)} pairs of promos end on essentially the same cliffhanger - the tool favours one trope; a feed of these would feel repetitive.")

    # 5. proxy recall
    peaks = uncovered_peaks(eps, promos)
    A["possible_misses_proxy"] = peaks[:10]
    if peaks:
        findings.append(f"{len(peaks)} episodes have a high-intensity 30s window (punctuation + tone tags) that no promo covers - possible misses (proxy metric, check by eye).")

    # 5b. stage-1 moments that never became a promo (needs moments.json from promo_finder two-stage mode)
    mpath = out / "moments.json"
    if mpath.exists():
        ms = json.loads(mpath.read_text())
        A["stage1_moments_by_status"] = dict(Counter(m["status"] for m in ms))
        lost = sorted([m for m in ms if m["status"] != "in_final" and m.get("intensity", 0) >= 8], key=lambda m: -m["intensity"])
        A["stage1_high_intensity_not_in_final"] = [
            {k: m[k] for k in ("id", "episode", "start", "end", "type", "intensity", "status", "summary")} for m in lost[:15]]
        if lost:
            findings.append(f"{len(lost)} discovered moments rated intensity >= 8 did not reach the final list "
                            f"({sum(m['status'] == 'not_used' for m in lost)} never turned into a promo, "
                            f"{sum(m['status'] == 'candidate_dropped' for m in lost)} dropped by validation/judge/de-dup).")

    # 6. LLM checks: judge noise + adversarial critic
    sample = []
    if use_llm and promos:
        pool = ranked or promos
        k_top = math.ceil(CRITIC_N / 2)
        rest = pool[k_top:]
        step = max(1, len(rest) // max(1, CRITIC_N - k_top))
        sample = pool[:k_top] + rest[::step][: CRITIC_N - k_top]
        again = {}
        for batch in chunks([p for p in sample if p.get("scores")], pf.JUDGE_BATCH):
            for i, s in pf._score(batch).items():
                again[batch[i]["id"]] = s
        diffs = [abs(again[p["id"]][k] - p["scores"][k]) for p in sample if p["id"] in again and p.get("scores") for k in thr]
        if diffs:
            A["judge_noise"] = {"clips_rescored": len(again), "mean_abs_score_change": round(st.mean(diffs), 2),
                                "share_changed_by_2_or_more": round(sum(d >= 2 for d in diffs) / len(diffs), 2)}
            if st.mean(diffs) >= 1.0:
                findings.append(f"Re-judging the same clips moved scores by {st.mean(diffs):.1f} points on average: the judge is noisy; treat small score gaps as ties.")
        reviews = run_numbered(chunks(sample, CRITIC_BATCH), CRITIC_SYS, "reviews", _parse_review)
        A["critic"] = {"reviewed": len(reviews), "by_verdict": dict(Counter(r["verdict"] for r in reviews.values())),
                       "failure_tags": dict(Counter(t for r in reviews.values() for t in r["tags"])),
                       "details": {pid: r for pid, r in reviews.items() if r["verdict"] != "good"}}
        by_id = {p["id"]: p for p in promos}
        blind = [pid for pid, r in reviews.items() if r["verdict"] == "bad" and (by_id[pid].get("scores") or {})
                 and sum(by_id[pid]["scores"][k] for k in thr) / 3 >= 8]
        A["critic"]["judge_loved_critic_said_bad"] = blind
        if reviews:
            bad = sum(r["verdict"] != "good" for r in reviews.values())
            findings.append(f"Critic flagged {bad}/{len(reviews)} sampled promos (even among the top-ranked); most common failure: "
                            f"{Counter(t for r in reviews.values() for t in r['tags']).most_common(2) or 'none'}.")
        if blind:
            findings.append(f"Judge/critic disagree on {len(blind)} clips the judge scored >= 8: the judge shares the generator's blind spots.")

    # 7. human-label template (+ scoring if filled)
    label_path = out / "audit_labels.csv"
    if labels and Path(labels).exists():
        A["human_labels"] = score_labels(labels, promos)
    elif not label_path.exists():
        with open(label_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "where", "hook", "cliffhanger", "judge_avg", "human_verdict", "human_note"])
            pool = ranked or promos
            for p in (pool[:10] + random.Random(0).sample(pool[10:], min(10, max(0, len(pool) - 10)))):
                s = p.get("scores") or {}
                w.writerow([p["id"], "; ".join(f'Ep{x["episode"]} {x["start"]}-{x["end"]}' for x in p["segments"]), p["hook"], p["cliffhanger"],
                            round((s.get("hook", 0) + s.get("cliffhanger", 0) + s.get("standalone", 0)) / 3, 1) if s else "", "", ""])

    (out / "audit.json").write_text(json.dumps(A, indent=2, ensure_ascii=False, default=str))

    md = ["# Quality audit - where the tool picks badly\n", "## Headline findings"]
    md += [f"- {f}" for f in findings] or ["- No automatic red flags - which still does not prove quality; hand-label a sample (below)."]
    md += ["", "## Pipeline funnel", f"- Kept {n}, rejected {len(rejected)}: " + ", ".join(f"{k}: {v}" for k, v in reasons.most_common()),
           "", "## Distribution", f"- Duration buckets: {A['distribution']['duration_buckets']}",
           f"- Episodes with no promo: {A['distribution']['episodes_with_no_promo']}",
           f"- Start position within episode: {A['distribution'].get('start_position_in_episode_thirds')}",
           "", "## Per-promo mechanical flags"]
    md += [f"- {pid}: {', '.join(v)}" for pid, v in A["deterministic_flags"].items()] or ["- none"]
    if "critic" in A:
        md += ["", "## Adversarial critic (sample)", f"- Verdicts: {A['critic']['by_verdict']}; tags: {A['critic']['failure_tags']}"]
        md += [f"- {pid} [{r['verdict']}] {r['tags']}: {r['explanation']} -> fix: {r['fix']}" for pid, r in A["critic"]["details"].items()]
    if "judge_noise" in A:
        md += ["", "## Judge stability", f"- {A['judge_noise']}"]
    if peaks:
        md += ["", "## Possible misses (proxy: most intense 30s per episode, not covered by any promo)"]
        md += [f"- Ep {b['episode']} {b['start']}-{b['end']} (intensity {b['intensity']})" for b in peaks[:10]]
    if A.get("stage1_high_intensity_not_in_final"):
        md += ["", "## Strong moments discovered but not in the final list"]
        md += [f"- {m['id']} Ep {m['episode']} {m['start']}-{m['end']} [{m['type']}, {m['intensity']}, {m['status']}]: {m['summary']}"
               for m in A["stage1_high_intensity_not_in_final"]]
    if "human_labels" in A:
        md += ["", "## Human labels", f"- {A['human_labels']}"]
    else:
        md += ["", "## Next step: ground truth", f"- Fill `human_verdict` (good/ok/bad) in {label_path.name} for the listed promos, then run `--labels {label_path}`. "
               "Until then every number above is the model grading itself, and mechanical flags and proxies are not ground truth."]
    md += ["", "## Known structural limits",
           "- Text only: acting, music, visuals and pacing of the real video are invisible, so 'hook in first 3 seconds' is judged from script lines.",
           "- The same LLM family proposes, judges and critiques, so shared blind spots survive every check.",
           "- Thresholds (7/7/6) and weights are my choices, not tuned against real post performance."]
    (out / "audit.md").write_text("\n".join(md))
    print(f"audit written: {out}/audit.md, audit.json, audit_labels.csv")
    return A


# --------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.getenv("INPUT_DIR", "Input"))
    ap.add_argument("--out", default=os.getenv("OUTPUT_DIR", "out"))
    ap.add_argument("--rank-only", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--no-llm-audit", action="store_true", help="skip critic + judge-noise checks (no API calls)")
    ap.add_argument("--labels", default=None, help="CSV with human_verdict filled in")
    a = ap.parse_args()
    out = Path(a.out)
    eps, meta, promos, rejected = load_inputs(a.input, a.out)
    if not promos:
        raise SystemExit("promos.json is empty - run promo_finder.py first (or loosen thresholds).")
    ranked = None
    if not a.audit_only:
        ranked = rank(promos, meta, out)
    elif (out / "ranking.json").exists():
        by_id = {p["id"]: p for p in promos}
        ranked = [by_id[r["id"]] for r in json.loads((out / "ranking.json").read_text())["ranked"] if r["id"] in by_id]
    if not a.rank_only:
        audit(eps, promos, rejected, ranked, out, use_llm=not a.no_llm_audit, labels=a.labels)