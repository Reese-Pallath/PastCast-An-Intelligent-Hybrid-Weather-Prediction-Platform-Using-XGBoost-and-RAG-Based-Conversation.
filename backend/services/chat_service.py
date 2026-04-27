from __future__ import annotations

"""
Chat Service — Orchestrates the full response pipeline.

Extracted from app.py to keep route handlers thin and business logic testable.
"""

import logging
import os
import re

import pandas as pd
import requests
import wikipedia

from utils.nlp_model import generate_rag_reply, translate_text

logger = logging.getLogger(__name__)

# ── Cached trends DataFrame ─────────────────────────────────────
_trends_df = None
_trends_mtime = 0.0


def _load_trends(data_path: str):
    """Load trends CSV, caching the result. Reload only if file changes."""
    global _trends_df, _trends_mtime
    if not os.path.exists(data_path):
        return None
    mtime = os.path.getmtime(data_path)
    if _trends_df is None or mtime > _trends_mtime:
        try:
            _trends_df = pd.read_csv(data_path)
            _trends_mtime = mtime
            logger.info("Trends CSV loaded/reloaded: %s (%d rows)", data_path, len(_trends_df))
        except Exception as exc:
            logger.error("Failed to load trends CSV: %s", exc)
    return _trends_df


# ── Utility functions ───────────────────────────────────────────

def extract_location(text: str):
    """Extract city name after 'in', 'at', or 'for' (handles punctuation)."""
    cleaned = text.strip()
    cleaned = re.sub(r"[?.!]+$", "", cleaned)
    m = re.search(r"\b(?:in|at|for)\s+([A-Za-z\s\.,-]+)$", cleaned, re.I)
    return m.group(1).strip(" .,-") if m else None


def get_weather(city: str, api_key: str, weather_url: str) -> str:
    """Fetch real-time weather summary for a city."""
    if not api_key:
        return "Weather unavailable (API key missing)."
    cleaned_city = re.sub(
        r"\b(today|tonight|tomorrow|yesterday|now|right|currently|outside|"
        r"this week|this weekend|at the moment|right now)\b",
        "", city, flags=re.IGNORECASE,
    ).strip(" ,.-")
    if not cleaned_city:
        cleaned_city = city.strip(" ,.-")
    try:
        r = requests.get(
            f"{weather_url}/weather",
            params={"q": cleaned_city, "appid": api_key, "units": "metric"},
            timeout=8,
        ).json()
        if "main" not in r:
            return f"Couldn't fetch weather for {cleaned_city}."
        desc = r["weather"][0]["description"].capitalize()
        temp = r["main"]["temp"]
        humidity = r["main"]["humidity"]
        return f"Weather in {cleaned_city}: {desc}, {temp} C, humidity {humidity}%."
    except requests.RequestException as exc:
        logger.error("Weather fetch error for %s: %s", cleaned_city, exc)
        return f"Couldn't fetch weather for {cleaned_city}."


_WIKI_SKIP_PREFIXES = (
    "list of", "history of", "historical", "timeline of", "spouse of",
    "records of", "centennial light", "light bulb", "lamp", "incandescent",
)

# Titles that are role/office articles rather than person articles — the full
# page content is used for these so the current holder's name is captured.
_WIKI_ROLE_WORDS = ("prime minister", "president", "chancellor", "secretary",
                    "minister of", "governor", "chief minister", "emperor")


def _is_role_article(title: str) -> bool:
    t = title.lower()
    return any(w in t for w in _WIKI_ROLE_WORDS)


def get_wiki_summary(query: str, deep: bool = False):
    """
    Smart Wikipedia fetch.

    Parameters
    ----------
    deep : bool
        When True (used for who-is queries), returns the first ~2 500 chars of
        the full page content instead of a short summary — this ensures the
        current office holder's name is included even when it isn't in the
        article's first few sentences.
    """
    try:
        results = wikipedia.search(query, results=8)
        if not results:
            return None

        for title in results:
            if any(title.lower().startswith(p) for p in _WIKI_SKIP_PREFIXES):
                continue
            try:
                if deep or _is_role_article(title):
                    page = wikipedia.page(title)
                    # 4 000 chars covers the full lead section of role articles
                    # (e.g. "Prime Minister of the UK") which includes the
                    # current incumbent's name (appears ~3 100 chars in).
                    content = page.content[:4000].strip()
                    return title, content
                else:
                    summary = wikipedia.summary(title, sentences=5)
                    return title, summary
            except wikipedia.exceptions.DisambiguationError as exc:
                opts = getattr(exc, "options", [])
                for opt in opts[:2]:
                    try:
                        summary = wikipedia.summary(opt, sentences=5)
                        return opt, summary
                    except Exception:
                        continue
            except wikipedia.exceptions.PageError:
                continue
            except Exception:
                continue

        return None
    except Exception as exc:
        logger.warning("Wikipedia lookup failed for '%s': %s", query, exc)
        return None


def duckduckgo_fallback(query: str):
    """Fallback factual search."""
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        data = requests.get(url, timeout=5).json()
        if data.get("AbstractText"):
            return data["AbstractText"]
        for item in data.get("RelatedTopics") or []:
            if isinstance(item, dict) and item.get("Text"):
                return item["Text"]
        return None
    except requests.RequestException as exc:
        logger.warning("DuckDuckGo fallback failed for '%s': %s", query, exc)
        return None


def analyze_trends(text: str, data_path: str) -> str:
    """Search the cached CSV for any related text."""
    df = _load_trends(data_path)
    if df is None:
        return ""
    try:
        mask = df.select_dtypes(include="object").apply(
            lambda col: col.str.contains(text, case=False, na=False)
        ).any(axis=1)
        matches = df[mask]
        if matches.empty:
            return ""
        col = matches.select_dtypes(include="object").columns[0]
        values = ", ".join(matches[col].head(3).values)
        return f"Historical trends related: {values}. "
    except Exception as exc:
        logger.warning("Trends analysis error: %s", exc)
        return ""


def parse_who_name(q: str):
    m = re.match(r"^\s*who\s+(is|was)\s+(.+?)\??\s*$", q, re.I)
    if not m:
        return None
    name = re.sub(r"\b(in|from|of|at)\s*$", "", m.group(2), flags=re.I).strip()
    return name


def format_direct_with_bullets(subject: str, summary: str) -> str:
    parts = [s.strip() for s in summary.split(".") if s.strip()]
    if not parts:
        return subject
    first = parts[0] + "."
    bullets = parts[1:5]
    if bullets:
        return f"{first}\n- " + "\n- ".join(bullets)
    return first


def is_translation_request(text: str) -> bool:
    return "translate" in text.lower()


_SUPPORTED_LANGS = {"hindi", "marathi", "tamil", "telugu"}


def parse_translation_query(text: str) -> tuple:
    # Strip trailing punctuation so "into Marathi." matches correctly
    clean = re.sub(r"[.!?]+$", "", text.strip())
    q = re.search(r"""[\"'\u201c\u201d\u2018\u2019](.+?)[\"'\u201c\u201d\u2018\u2019]""", clean)
    phrase = q.group(1) if q else None
    lang = re.search(r"\b(to|into)\s+([A-Za-z]+)\s*$", clean, re.I)
    target = lang.group(2) if lang else None
    return phrase, target


def _last_assistant_message(history: str) -> str | None:
    """Return the most recent non-trivial assistant line from conversation history.

    Messages are stored with role 'ai' (not 'assistant'), so lines appear as
    'AI: ...' after memory_context uppercases the role.
    """
    for line in reversed(history.splitlines()):
        upper = line.upper()
        # Match both 'AI:' (stored role) and 'ASSISTANT:' (defensive)
        if upper.startswith("AI:") or upper.startswith("ASSISTANT:"):
            colon = line.index(":")
            content = line[colon + 1:].strip()
            if len(content) > 40 and "please specify" not in content.lower():
                return content
    return None


def is_capability_query(text: str) -> bool:
    keys = ["what can you do", "capabilities", "help", "what do you do"]
    t = text.lower()
    return any(k in t for k in keys)


def assistant_capabilities() -> str:
    return (
        "I can help with:\n"
        "- General questions and explanations.\n"
        "- Wikipedia-based factual summaries.\n"
        "- English to Hindi/Marathi/Tamil/Telugu translations.\n"
        "- Accurate weather lookups.\n"
        "- RAG-powered weather and climate knowledge.\n"
        "- Persistent conversation memory (LSTM).\n"
        "- Trend lookup from CSV.\n"
        "- Clean reasoning using Qwen 1.5B."
    )


def memory_context(get_recent_fn, session_id=None) -> str:
    msgs = get_recent_fn(6, session_id=session_id)
    return "\n".join(f"{r.upper()}: {c}" for r, c in msgs)


# ── Main response pipeline ──────────────────────────────────────

def full_response(
    user_input: str,
    session_id,
    rag_engine,
    lstm_memory,
    get_recent_messages_fn,
    api_key,
    weather_url: str,
    data_path: str,
) -> str:
    """
    Enhanced response pipeline with RAG retrieval and LSTM memory.

    This is the core brain of the chatbot — extracted from app.py so
    it can be unit-tested with mocked dependencies.
    """
    text = user_input.lower().strip()
    recent_history = memory_context(get_recent_messages_fn, session_id)
    trends = analyze_trends(user_input, data_path)

    # Get RAG context
    rag_context_str = ""
    if rag_engine and rag_engine.is_ready:
        rag_results = rag_engine.retrieve(user_input, top_k=3)
        if rag_results:
            rag_context_str = rag_engine.format_context(rag_results)

    # Get LSTM memory context
    lstm_context_str = ""
    if lstm_memory and lstm_memory.is_ready and session_id:
        lstm_context_str = lstm_memory.get_context_summary(session_id)

    # Standalone language reply — user responding to a prior "please specify language" prompt
    if text.strip().lower() in _SUPPORTED_LANGS and recent_history:
        target_lang = text.strip().lower()
        phrase = _last_assistant_message(recent_history)
        if phrase:
            return trends + translate_text(phrase, target_lang)

    # Translation requests
    if is_translation_request(text):
        phrase, target = parse_translation_query(user_input)
        if not phrase:
            # "translate your last answer into X" — pull last assistant reply
            refers_to_previous = any(k in text for k in ["last", "above", "it", "that", "your", "previous"])
            if refers_to_previous and recent_history:
                phrase = _last_assistant_message(recent_history)
            if not phrase:
                # Remove translate/language words and use remainder as text
                phrase = re.sub(r"\b(translate|into|to)\b", "", text, flags=re.I).strip()
                for lang in _SUPPORTED_LANGS:
                    phrase = re.sub(rf"\b{lang}\b", "", phrase, flags=re.I).strip()
                phrase = phrase.strip("., ") or None
        if not target:
            return "Please specify a target language — I support Hindi, Marathi, Tamil, and Telugu."
        if not phrase:
            return "Please specify the text you'd like me to translate."
        return trends + translate_text(phrase, target)

    # Capabilities query
    if is_capability_query(text):
        return assistant_capabilities()

    # Weather queries
    weather_terms = ["weather", "temp", "temperature", "forecast", "humidity"]
    if any(w in text for w in weather_terms):
        city = extract_location(user_input)
        if city:
            weather_data = get_weather(city, api_key, weather_url)
            if "Couldn't fetch" not in weather_data and "unavailable" not in weather_data:
                if rag_context_str:
                    return weather_data + "\n\n" + generate_rag_reply(
                        user_input, rag_context=rag_context_str,
                        memory_context=lstm_context_str,
                        conversation_history=recent_history, max_tokens=100,
                    )
                return weather_data

        # Check for conceptual weather questions
        concept_terms = ["why", "how", "what is", "what are", "explain", "define", "reason"]
        is_concept = any(q in text for q in concept_terms)

        if len(text.split()) <= 6 and not is_concept:
            return "Please specify a city, e.g., 'weather in Pune'."

    # Who-is queries — Wikipedia with deep content fetch (gets current office
    # holders), then DuckDuckGo as backup. Retrieved text is always passed as
    # grounding context so the model can't hallucinate.
    who = parse_who_name(user_input)
    if who:
        # Strip "the current" prefix — it confuses Wikipedia search and
        # surfaces irrelevant articles instead of the exact role/person page.
        clean_who = re.sub(r"^(the\s+)?(current\s+)?", "", who, flags=re.I).strip()
        wiki_result = get_wiki_summary(clean_who, deep=True)
        if wiki_result:
            _title, summ = wiki_result
            response = generate_rag_reply(
                user_input,
                rag_context=summ,
                memory_context=lstm_context_str,
                conversation_history=recent_history,
                max_tokens=150,
            )
            return trends + response
        ddg = duckduckgo_fallback(f"who is {who}")
        if ddg and len(ddg) > 30:
            response = generate_rag_reply(
                user_input,
                rag_context=ddg,
                memory_context=lstm_context_str,
                conversation_history=recent_history,
                max_tokens=120,
            )
            return trends + response

    # RAG-enhanced response
    if rag_context_str:
        response = generate_rag_reply(
            user_input,
            rag_context=rag_context_str,
            memory_context=lstm_context_str,
            conversation_history=recent_history,
        )
        if response and len(response) > 20:
            return trends + response

    # Wikipedia fallback — pass the retrieved summary as grounding context
    wiki = get_wiki_summary(user_input)
    if wiki:
        _title, summary = wiki
        response = generate_rag_reply(
            user_input,
            rag_context=summary,
            memory_context=lstm_context_str,
            conversation_history=recent_history,
            max_tokens=150,
        )
        return trends + response

    # DuckDuckGo fallback
    ddg = duckduckgo_fallback(user_input)
    if ddg:
        response = generate_rag_reply(
            user_input,
            rag_context=ddg,
            memory_context=lstm_context_str,
            conversation_history=recent_history,
            max_tokens=150,
        )
        return trends + response

    # General fallback
    return generate_rag_reply(
        user_input,
        rag_context=rag_context_str,
        memory_context=lstm_context_str,
        conversation_history=recent_history,
    )
