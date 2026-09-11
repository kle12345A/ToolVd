"""YouTube channel research & reup-scoring pipeline.

5-step pipeline (spec from promt.txt):
  1. search channels by keywords  → channels.json
  2. analyze 50 recent videos/ch  → channels_analyzed.json (+ videos_raw.json)
  3. add engagement + reup_score  → channels_scored.json
  4. AI score videos (optional)   → video_ai_scores.json
  5. build HTML report            → report.html

All HTTP uses stdlib urllib only (no extra deps).
"""

from __future__ import annotations

import json
import re
import time
import html
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import urllib.error
import urllib.parse
import urllib.request

from src.core import ai_client
from src.utils.logger import logger

# ── Defaults / thresholds (UI prefill) ──────────────────────────────────────────

DEFAULT_KEYWORDS = [
    "oddly satisfying",
    "manufacturing process",
    "factory tour",
    "soap cutting asmr",
    "restoration",
]

SUB_MIN = 10_000
SUB_MAX = 500_000
VIEW_SUB_RATIO_MIN = 10        # total_view / subscriber
PREFERRED_COUNTRIES = {"US", "GB", "AU", "CA"}
VIRAL_THRESHOLD = 1_000_000
REQUEST_DELAY = 1.0            # seconds between API requests (rate-limit safety)

_API_BASE = "https://www.googleapis.com/youtube/v3"

StatusCb = Optional[Callable[[str], None]]


class StopRequested(Exception):
    """Raised to abort the pipeline mid-run."""


# ── Low-level HTTP ───────────────────────────────────────────────────────────────

def _get_json(endpoint: str, params: dict, timeout: int = 30) -> dict:
    """GET <_API_BASE>/<endpoint>?<params> and return parsed JSON."""
    url = f"{_API_BASE}/{endpoint}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="replace")
        # Surface quota / key errors clearly
        if e.code == 403:
            raise RuntimeError(
                f"YouTube API 403 (quota hết hoặc key sai/chưa bật API).\n{msg[:400]}"
            ) from e
        raise RuntimeError(f"YouTube API HTTP {e.code}: {msg[:400]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Lỗi mạng khi gọi YouTube API: {e.reason}") from e


def _say(status_cb: StatusCb, msg: str) -> None:
    logger.info(f"[YT-research] {msg}")
    if status_cb:
        status_cb(msg)


def _check_stop(stop_flag: Optional[Callable[[], bool]]) -> None:
    if stop_flag and stop_flag():
        raise StopRequested()


def _sleep(stop_flag: Optional[Callable[[], bool]]) -> None:
    """Delay between requests, but bail quickly if stop requested."""
    end = time.time() + REQUEST_DELAY
    while time.time() < end:
        _check_stop(stop_flag)
        time.sleep(0.1)


def _to_int(val) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _chunks(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


# ── API wrappers ─────────────────────────────────────────────────────────────────

def search_channels(keyword: str, api_key: str, max_results: int = 20) -> list[str]:
    """Return up to *max_results* channel IDs matching *keyword*."""
    data = _get_json("search", {
        "key": api_key,
        "q": keyword,
        "type": "channel",
        "part": "snippet",
        "maxResults": min(max_results, 50),
    })
    ids = []
    for item in data.get("items", []):
        cid = item.get("id", {}).get("channelId") or \
            item.get("snippet", {}).get("channelId")
        if cid:
            ids.append(cid)
    return ids[:max_results]


def get_channel_stats(channel_ids: list[str], api_key: str) -> list[dict]:
    """Return channel detail dicts (batched 50/request)."""
    out = []
    for batch in _chunks(channel_ids, 50):
        data = _get_json("channels", {
            "key": api_key,
            "id": ",".join(batch),
            "part": "snippet,statistics,contentDetails",
        })
        for item in data.get("items", []):
            snip = item.get("snippet", {})
            stats = item.get("statistics", {})
            uploads = (
                item.get("contentDetails", {})
                .get("relatedPlaylists", {})
                .get("uploads", "")
            )
            out.append({
                "channel_id": item.get("id", ""),
                "channel_name": snip.get("title", ""),
                "subscriber_count": _to_int(stats.get("subscriberCount")),
                "total_view_count": _to_int(stats.get("viewCount")),
                "video_count": _to_int(stats.get("videoCount")),
                "country": snip.get("country") or None,
                "description": snip.get("description", ""),
                "published_at": snip.get("publishedAt", ""),
                "uploads_playlist": uploads,
            })
    return out


def get_playlist_videos(playlist_id: str, api_key: str, limit: int = 50) -> list[str]:
    """Return up to *limit* most-recent video IDs from an uploads playlist."""
    ids = []
    page_token = ""
    while len(ids) < limit:
        params = {
            "key": api_key,
            "playlistId": playlist_id,
            "part": "contentDetails",
            "maxResults": min(50, limit - len(ids)),
        }
        if page_token:
            params["pageToken"] = page_token
        data = _get_json("playlistItems", params)
        for item in data.get("items", []):
            vid = item.get("contentDetails", {}).get("videoId")
            if vid:
                ids.append(vid)
        page_token = data.get("nextPageToken", "")
        if not page_token:
            break
    return ids[:limit]


def _parse_duration(iso: str) -> int:
    """Parse ISO-8601 duration (PT#H#M#S) to seconds."""
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not m:
        return 0
    h, mi, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + s


def get_video_stats(video_ids: list[str], api_key: str) -> list[dict]:
    """Return video detail dicts (batched 50/request)."""
    out = []
    for batch in _chunks(video_ids, 50):
        data = _get_json("videos", {
            "key": api_key,
            "id": ",".join(batch),
            "part": "snippet,statistics,contentDetails",
        })
        for item in data.get("items", []):
            snip = item.get("snippet", {})
            stats = item.get("statistics", {})
            thumbs = snip.get("thumbnails", {})
            thumb = (
                thumbs.get("maxres") or thumbs.get("high") or
                thumbs.get("medium") or thumbs.get("default") or {}
            ).get("url", "")
            out.append({
                "video_id": item.get("id", ""),
                "title": snip.get("title", ""),
                "published_at": snip.get("publishedAt", ""),
                "view_count": _to_int(stats.get("viewCount")),
                "like_count": _to_int(stats.get("likeCount")),
                "comment_count": _to_int(stats.get("commentCount")),
                "duration": _parse_duration(item.get("contentDetails", {}).get("duration", "")),
                "tags": snip.get("tags", []),
                "thumbnail_url": thumb,
            })
    return out


# ── Pipeline steps ───────────────────────────────────────────────────────────────

def step1_find_channels(
    keywords: list[str],
    api_key: str,
    out_dir: Path,
    max_per_keyword: int = 20,
    status_cb: StatusCb = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> list[dict]:
    """Search + filter channels. Saves channels.json."""
    _say(status_cb, f"🔎 Bước 1: tìm kênh theo {len(keywords)} keyword...")
    found_ids: dict[str, str] = {}          # id -> keyword (dedup)
    for kw in keywords:
        _check_stop(stop_flag)
        _say(status_cb, f"  • Tìm: '{kw}'")
        try:
            ids = search_channels(kw, api_key, max_per_keyword)
        except RuntimeError as e:
            _say(status_cb, f"    ⚠️ {e}")
            raise
        for cid in ids:
            found_ids.setdefault(cid, kw)
        _sleep(stop_flag)

    _say(status_cb, f"  Tìm thấy {len(found_ids)} kênh (dedup). Lấy thống kê...")
    channels = []
    for batch in _chunks(list(found_ids.keys()), 50):
        _check_stop(stop_flag)
        channels.extend(get_channel_stats(batch, api_key))
        _sleep(stop_flag)

    # Filter
    passed = []
    for ch in channels:
        ch["matched_keyword"] = found_ids.get(ch["channel_id"], "")
        subs = ch["subscriber_count"]
        views = ch["total_view_count"]
        country = ch["country"]
        if not (SUB_MIN <= subs <= SUB_MAX):
            continue
        if subs <= 0 or (views / subs) <= VIEW_SUB_RATIO_MIN:
            continue
        if country is not None and country not in PREFERRED_COUNTRIES:
            continue
        passed.append(ch)

    _say(status_cb, f"✅ Bước 1: {len(passed)}/{len(channels)} kênh pass filter.")
    _save_json(out_dir / "channels.json", passed)
    return passed


def step2_analyze_videos(
    channels: list[dict],
    api_key: str,
    out_dir: Path,
    status_cb: StatusCb = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> tuple[list[dict], dict]:
    """Pull 50 recent videos/channel, compute metrics.

    Saves channels_analyzed.json + videos_raw.json.
    Returns (analyzed_channels, videos_by_channel).
    """
    _say(status_cb, f"📊 Bước 2: phân tích video của {len(channels)} kênh...")
    videos_by_channel: dict[str, list[dict]] = {}
    analyzed = []

    for i, ch in enumerate(channels, 1):
        _check_stop(stop_flag)
        name = ch["channel_name"]
        _say(status_cb, f"  ({i}/{len(channels)}) {name}")
        playlist = ch.get("uploads_playlist", "")
        if not playlist:
            continue
        try:
            vid_ids = get_playlist_videos(playlist, api_key, 50)
            _sleep(stop_flag)
            videos = get_video_stats(vid_ids, api_key) if vid_ids else []
            _sleep(stop_flag)
        except RuntimeError as e:
            _say(status_cb, f"    ⚠️ Bỏ qua kênh ({e})")
            continue

        videos_by_channel[ch["channel_id"]] = videos
        views = [v["view_count"] for v in videos] or [0]
        subs = max(1, ch["subscriber_count"])
        avg_views = sum(views) / len(views)
        best = max(videos, key=lambda v: v["view_count"], default=None)

        ch2 = dict(ch)
        ch2.update({
            "videos_analyzed": len(videos),
            "avg_views": round(avg_views, 1),
            "median_views": round(statistics.median(views), 1),
            "max_views": max(views),
            "viral_count": sum(1 for v in views if v > VIRAL_THRESHOLD),
            "view_velocity": round(avg_views / subs, 3),
            "best_video_id": best["video_id"] if best else "",
        })
        analyzed.append(ch2)

    analyzed.sort(key=lambda c: c["view_velocity"], reverse=True)
    _save_json(out_dir / "videos_raw.json", videos_by_channel)
    _save_json(out_dir / "channels_analyzed.json", analyzed)

    top = analyzed[:20]
    _say(status_cb, f"✅ Bước 2 xong. Top {len(top)} kênh theo view_velocity:")
    for c in top:
        _say(status_cb, f"    {c['view_velocity']:>6.2f}  {c['channel_name']}")
    return analyzed, videos_by_channel


def step3_score_channels(
    analyzed: list[dict],
    videos_by_channel: dict,
    out_dir: Path,
    status_cb: StatusCb = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> list[dict]:
    """Add engagement metrics + reup_score. Saves channels_scored.json."""
    _say(status_cb, "🏆 Bước 3: tính engagement + reup_score...")
    now = datetime.now(timezone.utc)
    scored = []

    for ch in analyzed:
        _check_stop(stop_flag)
        videos = videos_by_channel.get(ch["channel_id"], [])
        like_rates, comment_rates, ages, pub_dates = [], [], [], []
        for v in videos:
            vc = v["view_count"]
            if vc > 0:
                like_rates.append(v["like_count"] / vc * 100)
                comment_rates.append(v["comment_count"] / vc * 100)
            pub = _parse_dt(v["published_at"])
            if pub:
                pub_dates.append(pub)
                ages.append((now - pub).days)

        like_rate = round(sum(like_rates) / len(like_rates), 2) if like_rates else 0.0
        comment_rate = round(sum(comment_rates) / len(comment_rates), 3) if comment_rates else 0.0
        content_age_avg = round(sum(ages) / len(ages), 1) if ages else 0.0
        if len(pub_dates) >= 2:
            weeks = max(1.0, (max(pub_dates) - min(pub_dates)).days / 7.0)
            upload_frequency = round(len(videos) / weeks, 2)
        else:
            upload_frequency = 0.0

        desc_lower = (ch.get("description", "") or "").lower()
        has_watermark_risk = any(
            t in desc_lower or t in (ch.get("description", "") or "")
            for t in ["©", "all rights reserved", "do not repost"]
        )
        is_faceless = not any(
            t in desc_lower for t in ["my name", "i am", "meet me"]
        )

        score = 0
        if ch["view_velocity"] > 5:
            score += 30
        if ch["viral_count"] >= 3:
            score += 20
        if like_rate > 3:
            score += 15
        if is_faceless:
            score += 20
        if not has_watermark_risk:
            score += 15

        ch2 = dict(ch)
        ch2.update({
            "like_rate": like_rate,
            "comment_rate": comment_rate,
            "upload_frequency": upload_frequency,
            "content_age_avg": content_age_avg,
            "has_watermark_risk": has_watermark_risk,
            "is_faceless": is_faceless,
            "reup_score": score,
        })
        scored.append(ch2)

    scored.sort(key=lambda c: c["reup_score"], reverse=True)
    _save_json(out_dir / "channels_scored.json", scored)

    _say(status_cb, "✅ Bước 3 xong. Top 15 kênh theo reup_score:")
    for c in scored[:15]:
        _say(status_cb, f"    {c['reup_score']:>3}  {c['channel_name']}")
    return scored


def _build_channel_ai_prompt(ch: dict, top_titles: list[str]) -> str:
    """Build a CHANNEL-level evaluation prompt (1 call/channel = ít token)."""
    data = {
        "channel_name": ch["channel_name"],
        "description": (ch.get("description", "") or "")[:600],
        "subscriber_count": ch["subscriber_count"],
        "avg_views": ch.get("avg_views", 0),
        "view_velocity": ch.get("view_velocity", 0),
        "viral_count": ch.get("viral_count", 0),
        "like_rate_pct": ch.get("like_rate", 0),
        "is_faceless": ch.get("is_faceless", None),
        "has_watermark_risk": ch.get("has_watermark_risk", None),
        "country": ch.get("country"),
        "top_video_titles": top_titles[:8],
    }
    return (
        "Bạn là chuyên gia phân tích kênh YouTube cho mục đích reup TikTok US. "
        "Nhiệm vụ: đánh giá KÊNH sau có phù hợp để cắt clip reup lên TikTok US không.\n\n"
        "Phân tích kênh và trả về JSON DUY NHẤT, không kèm text nào khác:\n\n"
        + json.dumps(data, ensure_ascii=False, indent=2) +
        "\n\nTrả về JSON đúng format:\n"
        "{\n"
        '  "reup_potential": 1-10,\n'
        '  "reup_recommended": true/false,\n'
        '  "content_type": "loại nội dung ngắn gọn (vd: ASMR satisfying, restoration...)",\n'
        '  "reason": "2-3 câu giải thích VÌ SAO nên hoặc không nên reup kênh này",\n'
        '  "suggested_angle": "góc tiếp cận / ý tưởng caption cho TikTok US"\n'
        "}\n\n"
        "Tiêu chí chấm reup_potential cao khi:\n"
        "- Nội dung visual mạnh, KHÔNG cần hiểu lời thoại (xem là hiểu)\n"
        "- Dễ cắt thành clip ngắn 30-90s độc lập\n"
        "- Có khoảnh khắc WOW / satisfying gây tò mò trong 3 giây đầu\n"
        "- Tiềm năng viral (đã có video triệu view, view_velocity cao)\n"
        "- Ít rủi ro bản quyền (faceless, không watermark 'do not repost')"
    )


def _call_ai(prompt: str, provider: str, model: str) -> str:
    if provider == "gemini":
        key = ai_client.load_api_key("gemini")
        if not key:
            raise RuntimeError("Chưa có Gemini API key.")
        return ai_client.call_gemini(prompt, key, model or "gemini-2.5-flash-lite")
    if provider == "groq":
        key = ai_client.load_api_key("groq")
        if not key:
            raise RuntimeError("Chưa có Groq API key.")
        return ai_client.call_groq(prompt, key, model or "llama-3.3-70b-versatile")
    if provider == "openrouter":
        key = ai_client.load_api_key("openrouter")
        if not key:
            raise RuntimeError("Chưa có OpenRouter API key.")
        return ai_client.call_openrouter(prompt, key, model or "google/gemini-flash-1.5:free")
    if provider == "ollama":
        return ai_client.call_ollama(prompt, model or "llama3.1:8b")
    raise RuntimeError(f"Provider không hợp lệ: {provider}")


def step4_ai_score_channels(
    scored: list[dict],
    videos_by_channel: dict,
    provider: str,
    model: str,
    out_dir: Path,
    top_channels: int = 0,
    status_cb: StatusCb = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> list[dict]:
    """AI-score each CHANNEL (1 call/channel) and output a reason.

    Far fewer tokens than per-video scoring. Saves channels_ai.json and
    merges AI fields into the passed `scored` list in-place.
    `top_channels=0` means score all channels.
    """
    _say(status_cb, f"🤖 Bước 4: AI chấm KÊNH + lý do ({provider})...")
    targets = scored[:top_channels] if top_channels > 0 else scored
    results = []
    for i, ch in enumerate(targets, 1):
        _check_stop(stop_flag)
        _say(status_cb, f"  ({i}/{len(targets)}) {ch['channel_name']}")
        videos = videos_by_channel.get(ch["channel_id"], [])
        top_titles = [
            v["title"] for v in
            sorted(videos, key=lambda v: v["view_count"], reverse=True)[:8]
        ]
        try:
            raw = _call_ai(
                _build_channel_ai_prompt(ch, top_titles), provider, model
            )
            m = re.search(r"\{[\s\S]*\}", raw)
            ai = json.loads(m.group()) if m else {}
        except Exception as e:
            _say(status_cb, f"    ⚠️ AI lỗi: {e}")
            ai = {}

        # Merge into the channel dict so the report can show it
        ch["ai_reup_potential"] = ai.get("reup_potential", 0)
        ch["ai_recommended"] = ai.get("reup_recommended", None)
        ch["ai_content_type"] = ai.get("content_type", "")
        ch["ai_reason"] = ai.get("reason", "")
        ch["ai_suggested_angle"] = ai.get("suggested_angle", "")

        results.append({
            "channel_id": ch["channel_id"],
            "channel_name": ch["channel_name"],
            "channel_link": f"https://youtube.com/channel/{ch['channel_id']}",
            "subscriber_count": ch["subscriber_count"],
            "reup_score": ch.get("reup_score", 0),
            "ai_reup_potential": ch["ai_reup_potential"],
            "ai_recommended": ch["ai_recommended"],
            "ai_content_type": ch["ai_content_type"],
            "ai_reason": ch["ai_reason"],
            "ai_suggested_angle": ch["ai_suggested_angle"],
        })
        time.sleep(0.5)

    results.sort(key=lambda r: r.get("ai_reup_potential", 0) or 0, reverse=True)
    _save_json(out_dir / "channels_ai.json", results)
    _say(status_cb, f"✅ Bước 4 xong. Chấm {len(results)} kênh.")
    return results


def step5_build_report(
    scored: list[dict],
    ai_scores: list[dict],
    out_dir: Path,
    keywords: list[str],
    status_cb: StatusCb = None,
) -> str:
    """Build self-contained sortable HTML report. Returns file path."""
    _say(status_cb, "📄 Bước 5: tạo báo cáo HTML...")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.html"
    html_str = _render_html(scored, ai_scores, keywords)
    path.write_text(html_str, encoding="utf-8")
    _say(status_cb, f"✅ Báo cáo: {path}")
    return str(path)


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _parse_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _esc(v) -> str:
    return html.escape(str(v))


def _yesno(v) -> str:
    if v is True:
        return "✅ Nên"
    if v is False:
        return "❌ Không"
    return "—"


def _render_html(scored: list[dict], ai_scores: list[dict], keywords: list[str]) -> str:
    # Sheet 1: channels (with AI score column)
    ch_rows = []
    for i, c in enumerate(scored, 1):
        cls = ' class="hi-green"' if c.get("reup_score", 0) > 70 else ""
        link = f"https://youtube.com/channel/{c['channel_id']}"
        ch_rows.append(
            f"<tr{cls}><td>{i}</td><td>{_esc(c['channel_name'])}</td>"
            f"<td>{c['subscriber_count']:,}</td><td>{c.get('avg_views',0):,.0f}</td>"
            f"<td>{c.get('view_velocity',0)}</td><td>{c.get('viral_count',0)}</td>"
            f"<td>{c.get('reup_score',0)}</td>"
            f"<td>{c.get('ai_reup_potential','—')}</td>"
            f"<td><a href='{link}' target='_blank'>Mở</a></td></tr>"
        )

    # Sheet 2: AI channel verdicts + reasons
    ai_rows = []
    for c in ai_scores:
        pot = c.get("ai_reup_potential", 0) or 0
        cls = ' class="hi-yellow"' if pot >= 8 else ""
        ai_rows.append(
            f"<tr{cls}><td>{_esc(c['channel_name'])}</td>"
            f"<td>{c['subscriber_count']:,}</td>"
            f"<td>{pot}</td><td>{_yesno(c.get('ai_recommended'))}</td>"
            f"<td>{_esc(c.get('ai_content_type',''))}</td>"
            f"<td>{_esc(c.get('ai_reason',''))}</td>"
            f"<td>{_esc(c.get('ai_suggested_angle',''))}</td>"
            f"<td><a href='{c['channel_link']}' target='_blank'>Mở</a></td></tr>"
        )

    # Sheet 3: summary
    total_ch = len(scored)
    total_ai = len(ai_scores)
    # top niche by avg view_velocity per matched_keyword
    niche: dict[str, list[float]] = {}
    for c in scored:
        kw = c.get("matched_keyword", "?") or "?"
        niche.setdefault(kw, []).append(c.get("view_velocity", 0))
    niche_avg = sorted(
        ((k, sum(v) / len(v)) for k, v in niche.items()),
        key=lambda x: x[1], reverse=True,
    )[:3]
    niche_rows = "".join(
        f"<li>{_esc(k)} — avg velocity {a:.2f}</li>" for k, a in niche_avg
    )
    top5 = sorted(ai_scores, key=lambda c: c.get("ai_reup_potential", 0) or 0, reverse=True)[:5]
    top5_rows = "".join(
        f"<li>[{c.get('ai_reup_potential',0)}] {_esc(c['channel_name'])} — {_esc(c.get('ai_reason',''))}</li>"
        for c in top5
    )

    return f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>YouTube Reup Research Report</title>
<style>
 body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#1e1e2e;color:#cdd6f4}}
 h1{{color:#89b4fa}} h2{{color:#94e2d5;margin-top:32px}}
 table{{border-collapse:collapse;width:100%;margin-top:8px;font-size:13px}}
 th,td{{border:1px solid #45475a;padding:6px 8px;text-align:left}}
 th{{background:#313244;cursor:pointer;position:sticky;top:0}}
 th:hover{{background:#45475a}}
 tr:nth-child(even){{background:#252537}}
 .hi-green{{background:#2d4a2d!important}}
 .hi-yellow{{background:#4a4a2d!important}}
 a{{color:#89b4fa}} .meta{{color:#a6adc8;font-size:12px}}
 ul{{line-height:1.6}}
</style>
<script>
function sortTable(tbl, col){{
  var t=document.getElementById(tbl), rows=Array.from(t.tBodies[0].rows);
  var dir=t.getAttribute('data-dir-'+col)==='asc'?-1:1;
  t.setAttribute('data-dir-'+col, dir===1?'asc':'desc');
  rows.sort(function(a,b){{
    var x=a.cells[col].innerText.replace(/[,%s]/g,''), y=b.cells[col].innerText.replace(/[,%s]/g,'');
    var nx=parseFloat(x), ny=parseFloat(y);
    if(!isNaN(nx)&&!isNaN(ny)) return (nx-ny)*dir;
    return x.localeCompare(y)*dir;
  }});
  rows.forEach(function(r){{t.tBodies[0].appendChild(r)}});
}}
</script>
</head><body>
<h1>📊 YouTube Reup Research Report</h1>
<p class="meta">Tạo lúc {datetime.now().strftime('%Y-%m-%d %H:%M')} • Keywords: {_esc(', '.join(keywords))}</p>

<h2>1. Kênh tiềm năng <span class="meta">(xanh = reup_score &gt; 70 • click tiêu đề để sắp xếp)</span></h2>
<table id="t1">
<thead><tr>
 <th onclick="sortTable('t1',0)">Rank</th><th onclick="sortTable('t1',1)">Channel</th>
 <th onclick="sortTable('t1',2)">Subscribers</th><th onclick="sortTable('t1',3)">Avg Views</th>
 <th onclick="sortTable('t1',4)">View/Sub</th><th onclick="sortTable('t1',5)">Viral</th>
 <th onclick="sortTable('t1',6)">Reup Score</th><th onclick="sortTable('t1',7)">AI Score</th><th>Link</th>
</tr></thead><tbody>{''.join(ch_rows) or '<tr><td colspan=9>Không có dữ liệu</td></tr>'}</tbody></table>

<h2>2. Đánh giá AI từng kênh <span class="meta">(vàng = AI Score ≥ 8 • kèm lý do nên/không nên reup)</span></h2>
<table id="t2">
<thead><tr>
 <th onclick="sortTable('t2',0)">Channel</th><th onclick="sortTable('t2',1)">Subscribers</th>
 <th onclick="sortTable('t2',2)">AI Score</th><th onclick="sortTable('t2',3)">Nên reup?</th>
 <th onclick="sortTable('t2',4)">Loại nội dung</th><th>Lý do</th>
 <th>Góc tiếp cận / Caption</th><th>Link</th>
</tr></thead><tbody>{''.join(ai_rows) or '<tr><td colspan=8>Chưa chấm AI (bật tùy chọn AI để có dữ liệu)</td></tr>'}</tbody></table>

<h2>3. Summary</h2>
<ul>
 <li>Tổng kênh đã scan: <b>{total_ch}</b></li>
 <li>Tổng kênh AI đánh giá: <b>{total_ai}</b></li>
 <li>Top 3 niche hiệu quả nhất:<ul>{niche_rows or '<li>—</li>'}</ul></li>
 <li>Top 5 kênh AI đánh giá cao nhất:<ul>{top5_rows or '<li>—</li>'}</ul></li>
</ul>
</body></html>"""


# ── Full pipeline orchestration ─────────────────────────────────────────────────

def run_pipeline(
    keywords: list[str],
    api_key: str,
    out_dir: str,
    run_ai: bool = False,
    ai_provider: str = "gemini",
    ai_model: str = "",
    max_per_keyword: int = 20,
    status_cb: StatusCb = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> str:
    """Run steps 1→5. Returns path to report.html."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    channels = step1_find_channels(
        keywords, api_key, out, max_per_keyword, status_cb, stop_flag)
    if not channels:
        _say(status_cb, "⚠️ Không có kênh nào pass filter. Dừng.")
        return step5_build_report([], [], out, keywords, status_cb)

    analyzed, videos_by_channel = step2_analyze_videos(
        channels, api_key, out, status_cb, stop_flag)
    scored = step3_score_channels(
        analyzed, videos_by_channel, out, status_cb, stop_flag)

    ai_scores = []
    if run_ai:
        ai_scores = step4_ai_score_channels(
            scored, videos_by_channel, ai_provider, ai_model, out,
            status_cb=status_cb, stop_flag=stop_flag)
        # scored now carries merged AI fields → re-save
        _save_json(out / "channels_scored.json", scored)

    return step5_build_report(scored, ai_scores, out, keywords, status_cb)
