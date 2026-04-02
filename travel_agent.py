# travel_agent.py — Elite Travel Concierge Agent v3.0
# ─────────────────────────────────────────────────────────────────────────────
# Features:
#   • Multi-person group travel with per-person cost breakdown
#   • Vegetarian / dietary filtering propagated to every restaurant suggestion
#   • Real-time weather fetched and woven into daily schedule
#   • Budget feasibility with intelligent alternatives
#   • Safety alerts, visa info, emergency contacts
#   • Packing checklist tailored to weather + destination + trip duration
#   • Local .ics calendar export
#   • Persistent user memory (diet, budget style, home city, nationality)
#   • Agent reflection + fallback replanning loop
#   • Full REST API with history, memory, calendar, logs endpoints
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import datetime
import json
import operator
import os
import re
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Annotated, Optional, TypedDict

from duckduckgo_search import DDGS
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from icalendar import Calendar, Event as ICSEvent, vText
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set.\n"
        "Run: export GROQ_API_KEY=your_key_here"
    )

MODEL          = os.getenv("GROQ_MODEL",     "llama-3.3-70b-versatile")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
PORT           = int(os.getenv("PORT",        "8001"))
HOST           = os.getenv("HOST",            "127.0.0.1")
VERSION        = "3.0.0"

_STARTUP_TIME  = datetime.datetime.utcnow()

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent.resolve()
ITINERARY_DIR  = BASE_DIR / "itineraries"
HISTORY_FILE   = BASE_DIR / "history.json"
MEMORY_FILE    = BASE_DIR / "user_memory.json"

ITINERARY_DIR.mkdir(exist_ok=True)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MEMORY: dict = {
    "last_destination": None,
    "diet":             "Any",
    "budget_style":     "mid-range",
    "home_city":        "Delhi, India",
    "nationality":      "Indian",
    "past_trips":       [],
}


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(
    __name__,
    static_folder="travel-agent-ai/dist",
    static_url_path="/",
)
CORS(app, origins=ALLOWED_ORIGIN)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM
# ═══════════════════════════════════════════════════════════════════════════════

llm = ChatGroq(
    model=MODEL,
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
    max_tokens=3000,
)


def llm_call(prompt: str, fallback: str = "") -> str:
    """Single-shot LLM call with graceful fallback."""
    try:
        response = llm.invoke(prompt)
        return getattr(response, "content", "") or fallback
    except Exception as exc:
        return fallback or f"LLM error: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGER
# ═══════════════════════════════════════════════════════════════════════════════

class AgentLogger:
    """Structured step logger for the planning graph."""

    def __init__(self) -> None:
        self.logs: list[dict] = []
        self.step: int = 0

    def log(
        self,
        node: str,
        event: str,
        content: str,
        duration_ms: Optional[int] = None,
    ) -> None:
        self.step += 1
        entry: dict = {
            "step":      self.step,
            "timestamp": datetime.datetime.now().isoformat(),
            "node":      node,
            "event":     event,
            "content":   str(content)[:700],
        }
        if duration_ms is not None:
            entry["duration_ms"] = duration_ms
        self.logs.append(entry)
        suffix = f" ({duration_ms}ms)" if duration_ms is not None else ""
        print(f"[{node}]{suffix} {event} → {str(content)[:80]}")

    def reset(self) -> None:
        self.logs = []
        self.step = 0

    def get_logs(self) -> list[dict]:
        return self.logs


logger = AgentLogger()


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH STATE
# ═══════════════════════════════════════════════════════════════════════════════

class TravelState(TypedDict):
    user_request:     str
    parsed_goal:      dict
    research:         str
    weather_info:     str
    budget_plan:      str
    budget_summary:   dict
    is_infeasible:    bool
    activities:       str
    packing_list:     str
    safety_info:      str
    reflect_notes:    str
    needs_replan:     bool
    final_itinerary:  str
    ics_path:         str
    calendar_events:  list
    memory:           dict
    overlap_warning:  str
    logs:             Annotated[list, operator.add]


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _atomic_write(path: Path, data: str) -> None:
    """Write data atomically via a temp file in the same directory."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_memory() -> dict:
    if not MEMORY_FILE.exists():
        _atomic_write(MEMORY_FILE, json.dumps(DEFAULT_MEMORY, indent=2))
        return DEFAULT_MEMORY.copy()
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    memory = DEFAULT_MEMORY.copy()
    if isinstance(data, dict):
        memory.update(data)
    if not isinstance(memory.get("past_trips"), list):
        memory["past_trips"] = []
    return memory


def infer_budget_style(budget_inr: int) -> str:
    if budget_inr < 20_000:
        return "budget"
    if budget_inr > 60_000:
        return "luxury"
    return "mid-range"


def save_memory(destination: str, diet: str, budget_inr: int) -> dict:
    memory = load_memory()
    memory["last_destination"] = destination or memory.get("last_destination")
    memory["diet"]             = diet or memory.get("diet", "Any")
    memory["budget_style"]     = infer_budget_style(int(budget_inr or 0))
    memory.setdefault("past_trips", [])
    memory["past_trips"].append({
        "destination": destination,
        "budget_inr":  int(budget_inr or 0),
        "date":        datetime.date.today().isoformat(),
    })
    _atomic_write(MEMORY_FILE, json.dumps(memory, indent=2))
    return memory


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_int(value, default: int) -> int:
    try:
        if value is None:
            return default
        return int(float(str(value).strip()))
    except Exception:
        return default


def valid_date_string(value) -> str:
    try:
        return str(datetime.date.fromisoformat(str(value)[:10]))
    except Exception:
        return str(datetime.date.today())


def extract_json_object(text: str) -> dict:
    if not text:
        raise ValueError("Empty response")
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group())


def request_mentions_diet(text: str) -> bool:
    return bool(re.search(
        r"\b(veg|vegetarian|vegan|halal|jain|non[-\s]?veg|kosher|gluten.free)\b",
        text, re.I,
    ))


def request_mentions_budget(text: str) -> bool:
    return bool(re.search(
        r"(₹|rs\.?|inr|budget|luxury|premium|mid-range|budget-friendly|cheap|affordable)",
        text, re.I,
    ))


def request_mentions_group(text: str) -> Optional[int]:
    """Return number of travellers if explicitly mentioned, else None."""
    match = re.search(
        r"(\d+)\s+(?:people|persons?|travell?ers?|friends?|family members?|pax)",
        text, re.I,
    )
    return int(match.group(1)) if match else None


def default_budget_for_style(style: str) -> int:
    style = (style or "").lower()
    if style == "budget":
        return 18_000
    if style == "luxury":
        return 90_000
    return 45_000


def normalize_goal(parsed: dict, request_text: str, memory: dict) -> dict:
    explicit_diet   = request_mentions_diet(request_text)
    explicit_budget = request_mentions_budget(request_text)
    group_size      = request_mentions_group(request_text)

    home_city    = memory.get("home_city", "Delhi, India")
    destination  = str(parsed.get("destination") or memory.get("last_destination") or "Goa").strip()
    origin       = str(parsed.get("origin") or home_city).strip()
    duration_days = max(1, parse_int(parsed.get("duration_days"), 3))
    budget_inr   = parse_int(parsed.get("budget_inr"), 0)
    nationality  = str(parsed.get("nationality") or memory.get("nationality", "Indian")).strip()
    diet         = str(parsed.get("diet") or memory.get("diet") or "Any").strip()
    budget_style = str(parsed.get("budget_style") or memory.get("budget_style") or "mid-range").strip()
    travellers   = parse_int(parsed.get("travellers") or group_size, 1)
    start_date   = valid_date_string(
        parsed.get("start_date")
        or str(datetime.date.today() + datetime.timedelta(days=7))
    )

    if not explicit_diet:
        diet = memory.get("diet", "Any")
    if not explicit_budget or budget_inr <= 0:
        budget_inr = default_budget_for_style(budget_style)

    return {
        "destination":   destination,
        "origin":        origin,
        "duration_days": duration_days,
        "budget_inr":    budget_inr,
        "start_date":    start_date,
        "nationality":   nationality,
        "diet":          diet,
        "budget_style":  budget_style,
        "travellers":    max(1, travellers),
    }


def regex_goal_fallback(request_text: str, memory: dict) -> dict:
    destination_match = re.search(
        r"(?:to|for|visit|plan)\s+([a-zA-Z\s,]+?)(?:\s+(?:trip|for|from|starting|next|with|$))",
        request_text, re.I,
    )
    days_match   = re.search(r"(\d+)\s*-?\s*day",              request_text, re.I)
    budget_match = re.search(r"(?:₹|rs\.?|inr)?\s*(\d{4,})",  request_text, re.I)
    group_size   = request_mentions_group(request_text) or 1

    parsed = {
        "destination":   (destination_match.group(1).strip()
                          if destination_match
                          else memory.get("last_destination") or "Goa"),
        "origin":        memory.get("home_city", "Delhi, India"),
        "duration_days": int(days_match.group(1))   if days_match   else 3,
        "budget_inr":    int(budget_match.group(1)) if budget_match else 0,
        "start_date":    str(datetime.date.today() + datetime.timedelta(days=7)),
        "nationality":   memory.get("nationality", "Indian"),
        "diet":          memory.get("diet", "Any"),
        "travellers":    group_size,
    }
    return normalize_goal(parsed, request_text, memory)


def looks_bad_output(text: str) -> bool:
    if not text or len(text.strip()) < 20:
        return True
    bad_patterns = [
        r"agent stopped due to iteration limit",
        r"\[insert [^\]]+\]",
        r"replace the placeholders",
        r"no further details available",
        r"\[your [^\]]+\]",
        r"\[specific [^\]]+\]",
    ]
    return any(re.search(p, text, re.I) for p in bad_patterns)


def output_conflicts_with_destination(text: str, destination: str) -> bool:
    if not text or not destination:
        return False
    tokens = {
        t.strip().lower()
        for t in re.split(r"[\s,]+", destination)
        if len(t.strip()) > 2
    }
    return not any(t in text.lower() for t in tokens)


def sanitize_output(text: str, destination: str, fallback: str) -> str:
    cleaned = (text or "").strip()
    if looks_bad_output(cleaned):
        return fallback
    if output_conflicts_with_destination(cleaned, destination):
        return fallback
    return cleaned





# ═══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

def tool_web_search(query: str) -> str:
    logger.log("TOOL", f"WebSearch: {query}", "")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        return "\n".join(
            f"- {row['title']}: {row['body'][:250]}" for row in results
        )
    except Exception:
        return "Search unavailable. Use general knowledge."


def tool_calculator(expression: str) -> str:
    logger.log("TOOL", f"Calculator: {expression}", "")
    expression = re.sub(r"=.*$", "", expression.replace('"', "").replace("'", "").strip())
    allowed = set("0123456789+-*/(). ")
    if not all(ch in allowed for ch in expression):
        return "Calc error: invalid characters in expression"
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return f"{expression} = {round(float(result), 2)}"
    except Exception as exc:
        return f"Calc error: {exc}"


def tool_currency_convert(expression: str) -> str:
    """
    Convert between currencies. Input format: 'USD 100 to INR'.
    Uses approximate fixed rates as a fallback when live data is unavailable.
    """
    logger.log("TOOL", f"CurrencyConvert: {expression}", "")
    rates_to_inr = {
        "USD": 83.5,  "EUR": 90.2,  "GBP": 105.5, "AUD": 54.0,
        "CAD": 61.0,  "SGD": 62.0,  "AED": 22.7,  "THB":  2.4,
        "JPY":  0.56, "MYR": 17.8,
    }
    match = re.match(
        r"([A-Z]{3})\s+([\d.]+)\s+to\s+([A-Z]{3})",
        expression.strip().upper(),
    )
    if not match:
        return "Format: 'USD 100 to INR'"
    from_cur, amount, to_cur = match.group(1), float(match.group(2)), match.group(3)
    if from_cur == "INR" and to_cur in rates_to_inr:
        result = amount / rates_to_inr[to_cur]
        return f"₹{amount:,.0f} ≈ {to_cur} {result:,.2f} (approx)"
    if to_cur == "INR" and from_cur in rates_to_inr:
        result = amount * rates_to_inr[from_cur]
        return f"{from_cur} {amount:,.2f} ≈ ₹{result:,.0f} (approx)"
    return f"Conversion for {from_cur}→{to_cur} not available offline."


def tool_ics_writer(data_json: str) -> str:
    """
    Generates a .ics calendar file.
    Input JSON keys: destination, start_date, duration_days,
    optional events: [{summary, description}]
    """
    try:
        data        = json.loads(data_json)
        destination = data.get("destination", "Trip")
        start_date  = datetime.date.fromisoformat(valid_date_string(data.get("start_date")))
        duration    = max(1, parse_int(data.get("duration_days"), 3))
        provided    = data.get("events", [])

        cal = Calendar()
        cal.add("prodid",   "-//Travel Concierge Agent//EN")
        cal.add("version",  "2.0")
        cal.add("calscale", "GREGORIAN")
        cal.add("x-wr-calname", f"Trip to {destination}")

        for idx in range(duration):
            event_date = start_date + datetime.timedelta(days=idx)
            day_num    = idx + 1
            if idx < len(provided):
                pe          = provided[idx]
                summary     = pe.get("summary")     or f"Trip to {destination} – Day {day_num}"
                description = pe.get("description") or pe.get("info") or ""
            else:
                summary     = f"Trip to {destination} – Day {day_num}"
                description = ""

            ev = ICSEvent()
            ev.add("summary", summary)
            ev.add("dtstart", event_date)
            ev.add("dtend",   event_date + datetime.timedelta(days=1))
            ev.add("uid",     f"{uuid.uuid4()}@travel-concierge")
            if description:
                ev["description"] = vText(description)
            cal.add_component(ev)

        filename = f"trip_{uuid.uuid4().hex[:8]}.ics"
        (ITINERARY_DIR / filename).write_bytes(cal.to_ical())
        return filename
    except Exception as exc:
        return f"ICS error: {exc}"





def tool_get_weather(location: str) -> str:
    """Fetch real weather from wttr.in (no API key required)."""
    import ssl
    logger.log("TOOL", f"Weather: {location}", "")
    try:
        url = (
            f"https://wttr.in/{urllib.request.quote(location)}"
            "?format=%C+%t+%h+%w&m"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "TravelConcierge/3.0"})
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=7, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace").strip()
        return raw if raw and len(raw) < 200 else "Weather data unavailable"
    except Exception:
        return "Weather lookup failed — check destination manually"


# ── Tool registry ─────────────────────────────────────────────────────────────

tools = [
    Tool(
        name="WebSearch",
        func=tool_web_search,
        description=(
            "Search travel costs, flights, hotels, attractions, visa info, "
            "restaurant reviews, and safety tips."
        ),
    ),
    Tool(
        name="Calculator",
        func=tool_calculator,
        description="Evaluate math expressions like '5000*3 + 2000'. Returns the result.",
    ),
    Tool(
        name="CurrencyConvert",
        func=tool_currency_convert,
        description=(
            "Convert currencies. Input format: 'USD 100 to INR' or 'INR 5000 to USD'."
        ),
    ),
    Tool(
        name="ICSWriter",
        func=tool_ics_writer,
        description=(
            "Generate a .ics calendar file. "
            "Input JSON: {destination, start_date, duration_days, "
            "events:[{summary, description}]}"
        ),
    ),

    Tool(
        name="GetWeather",
        func=tool_get_weather,
        description="Get current weather conditions for a city. Input: city name.",
    ),
]

# ── Agent prompt ──────────────────────────────────────────────────────────────

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise, expert travel planning sub-agent. "
        "Use tools when needed. Keep answers tied to the exact destination in the task. "
        "Never use placeholder text like [insert X] or [specific restaurant]. "
        "Always name real places, real restaurants, and real attractions. "
        "When dietary restrictions are specified, ONLY suggest compliant options.",
    ),
    ("user",        "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])


def create_agent() -> AgentExecutor:
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=AGENT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=6,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


def run_agent(task: str, destination: str, fallback: str) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            result = create_agent().invoke({"input": task})
            output = sanitize_output(result.get("output", ""), destination, fallback)
            if not looks_bad_output(output):
                for action, observation in result.get("intermediate_steps", []):
                    logger.log(
                        "AGENT",
                        f"Tool: {action.tool}",
                        str(observation)[:150],
                    )
                return output
            raise ValueError("Bad or placeholder output detected")
        except Exception as exc:
            last_error = exc
            logger.log("AGENT", f"Attempt {attempt} failed", str(exc))
            if attempt < 3:
                time.sleep(1)

    # ── Enhanced fallback: direct LLM call with web search context ────────
    logger.log("AGENT", "Tool-calling failed — trying direct LLM", str(last_error)[:80])
    try:
        search_context = tool_web_search(f"{destination} travel guide attractions restaurants budget INR")
        enriched_prompt = (
            f"{task}\n\n"
            f"Here is some research data to help you:\n{search_context}\n\n"
            f"Provide a comprehensive, specific answer. Name real places, real restaurants, "
            f"and real attractions in {destination}. Do NOT use placeholders."
        )
        direct_result = llm_call(enriched_prompt, fallback)
        direct_result = sanitize_output(direct_result, destination, fallback)
        if not looks_bad_output(direct_result):
            logger.log("AGENT", "Direct LLM succeeded", direct_result[:100])
            return direct_result
    except Exception as direct_exc:
        logger.log("AGENT", "Direct LLM also failed", str(direct_exc)[:80])

    logger.log("AGENT", "All attempts failed — using fallback", str(last_error))
    return fallback


# ═══════════════════════════════════════════════════════════════════════════════
# BUDGET PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def extract_budget_summary(text: str) -> dict:
    if not text:
        return {}
    for candidate in reversed(re.findall(r"\{[\s\S]*?\}", text)):
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        required = {"flights", "hotel", "food", "activities", "total", "verdict"}
        if not required.issubset(data.keys()):
            continue
        summary = {
            "flights":    parse_int(data.get("flights"),    0),
            "hotel":      parse_int(data.get("hotel"),      0),
            "food":       parse_int(data.get("food"),       0),
            "activities": parse_int(data.get("activities"), 0),
            "total":      parse_int(data.get("total"),      0),
            "verdict":    str(data.get("verdict", "")).strip().upper(),
        }
        if summary["verdict"] in {"PASS", "OVER BUDGET"}:
            return summary
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# CALENDAR EVENT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_calendar_events(activities_text: str, start_date_str: str) -> list[dict]:
    if not activities_text.strip():
        return []
    try:
        start_date = datetime.date.fromisoformat(valid_date_string(start_date_str))
    except Exception:
        start_date = datetime.date.today()

    events: list[dict] = []
    pattern = r"\*\*Day\s+(\d+)(.*?)\*\*"
    matches = list(re.finditer(pattern, activities_text, re.I))

    if matches:
        for i, match in enumerate(matches):
            try:
                day_num = int(match.group(1))
                theme = match.group(2).strip(" \t-—:")
                if not theme:
                    theme = f"Day {day_num}"
                
                start_idx = match.end()
                end_idx = matches[i+1].start() if i + 1 < len(matches) else len(activities_text)
                content = activities_text[start_idx:end_idx].strip()
                
                if not content:
                    continue
                
                event_date = start_date + datetime.timedelta(days=day_num - 1)
                
                events.append({
                    "date":    str(event_date),
                    "summary": f"Day {day_num}: {theme}",
                    "info":    f"**Day {day_num} — {theme}**\n{content}",
                })
            except Exception:
                continue
    else:
        events.append({
            "date":    str(start_date),
            "summary": "Trip Day 1",
            "info":    activities_text.strip(),
        })
    return events


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK ITINERARY BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_fallback_activities(goal: dict) -> str:
    destination = goal.get("destination", "your destination")
    diet        = goal.get("diet", "Any")
    duration    = max(1, parse_int(goal.get("duration_days"), 3))
    travellers  = max(1, parse_int(goal.get("travellers"), 1))
    group_note  = f" for your group of {travellers}" if travellers > 1 else ""

    days = []
    for idx in range(duration):
        day_num = idx + 1
        if idx == 0:
            days.append(
                f"**Day {day_num} — Arrival & First Impressions**\n"
                f"- 08:00 Breakfast: Light meal at the airport or hotel café{group_note}.\n"
                f"- 10:00 Morning: Arrive in {destination}, transfer to hotel, check in.\n"
                f"- 14:00 Afternoon: Orientation walk to get your bearings.\n"
                f"- 19:00 Evening: Dinner at a well-rated {diet}-friendly local restaurant.\n"
                f"- Local tip: Ask hotel staff for their favourite hidden-gem eatery.\n"
                f"- Indoor backup: Hotel lounge or nearby museum if tired from travel."
            )
        elif idx == duration - 1:
            days.append(
                f"**Day {day_num} — Departure Day**\n"
                f"- 08:00 Breakfast: Final local breakfast — try a street-food spot.\n"
                f"- 10:00 Morning: Last-minute souvenir shopping at a local market.\n"
                f"- 12:00 Afternoon: Check out, transfer to airport/station.\n"
                f"- Local tip: Pack local spices or sweets as gifts.\n"
                f"- Indoor backup: Airport lounge if time allows."
            )
        else:
            days.append(
                f"**Day {day_num} — Local Highlights**\n"
                f"- 08:00 Breakfast: {diet}-friendly café near the hotel.\n"
                f"- 10:00 Morning: Visit the most iconic landmark in {destination}.\n"
                f"- 13:00 Lunch: Authentic {diet} local cuisine at a highly rated restaurant.\n"
                f"- 15:00 Afternoon: Explore a local market or cultural site.\n"
                f"- 19:00 Evening: Sunset viewpoint, then dinner at a rooftop restaurant.\n"
                f"- Local tip: Hire a local guide for the morning landmark visit.\n"
                f"- Indoor backup: Museum, gallery, or cooking class."
            )
    return "\n\n".join(days)


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL ITINERARY FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

def format_final_itinerary(
    goal:           dict,
    activities:     str,
    budget_summary: dict,
    safety_info:    str,
    weather_info:   str,
    packing_list:   str,
) -> str:
    destination   = goal.get("destination",  "Unknown")
    origin        = goal.get("origin",       "Delhi, India")
    duration      = goal.get("duration_days", 3)
    start_date    = goal.get("start_date",   str(datetime.date.today()))
    budget        = goal.get("budget_inr",   0)
    diet          = goal.get("diet",         "Any")
    travellers    = max(1, parse_int(goal.get("travellers"), 1))

    total_cost    = budget_summary.get("total", 0)
    per_day       = round(total_cost / max(1, duration))       if total_cost else 0
    per_person    = round(total_cost / travellers)              if total_cost else 0
    verdict       = budget_summary.get("verdict", "PASS")

    group_note = f" for {travellers} people" if travellers > 1 else ""

    lines = [
        f"# {destination} — {duration}-Day Itinerary{group_note}",
        "",
        (
            f"**Origin:** {origin} &nbsp;|&nbsp; "
            f"**Start:** {start_date} &nbsp;|&nbsp; "
            f"**Budget:** ₹{budget:,} &nbsp;|&nbsp; "
            f"**Diet:** {diet} &nbsp;|&nbsp; "
            f"**Travellers:** {travellers}"
        ),
    ]

    if weather_info and "fail" not in weather_info.lower():
        lines += ["", f"**Current Weather in {destination}:** {weather_info}"]

    lines += [
        "", "---", "",
        "## Daily Schedule", "",
        activities.strip(),
        "", "---", "",
        "## Budget Breakdown", "",
        f"*Total budget for {travellers} person{'s' if travellers > 1 else ''}*", "",
        "| Category              | Total Cost      | Per Person       |",
        "|-----------------------|-----------------|------------------|",
        f"| Flights (return)   | ₹{budget_summary.get('flights', 0):,}  | ₹{round(budget_summary.get('flights', 0)/travellers):,} |",
        f"| Hotels ({duration} nights) | ₹{budget_summary.get('hotel', 0):,}  | ₹{round(budget_summary.get('hotel', 0)/travellers):,} |",
        f"| Food               | ₹{budget_summary.get('food', 0):,}  | ₹{round(budget_summary.get('food', 0)/travellers):,} |",
        f"| Activities        | ₹{budget_summary.get('activities', 0):,}  | ₹{round(budget_summary.get('activities', 0)/travellers):,} |",
        f"| **Total** | **₹{total_cost:,}** | **₹{per_person:,}** |",
        f"| **~Per Day** | **₹{per_day:,}** | **₹{round(per_day/travellers):,}** |",
        f"| **Verdict** | **{'Within Budget' if verdict == 'PASS' else 'Over Budget'}** | — |",
        "",
        "> **Savings tip:** Book flights 4–6 weeks in advance and choose mid-week dates for up to 30% off.",
        "", "---", "",
        "## Visa & Safety", "",
        safety_info.strip() or f"Verify visa requirements and safety guidance for {destination} with official sources.",
        "", "---",
    ]

    if packing_list and not looks_bad_output(packing_list):
        lines += ["", "## Packing List", "", packing_list.strip(), "", "---"]

    lines += [
        "", "## Contingency Tips",
        "- Keep all bookings flexible where possible",
        "- Download offline maps (Google Maps / Maps.me) before departure",
        f"- Carry ₹{round(per_day * 0.15):,}–₹{round(per_day * 0.20):,} per person per day as emergency cash",
        "- Save hotel confirmations, tickets, and passport copies offline",
        "- Share your itinerary with a trusted contact at home",
        f"- For groups: designate one person as trip coordinator for bookings",
        "", "---", "",
        f"*Generated by Travel Concierge Agent v{VERSION} — "
        f"{datetime.datetime.now().strftime('%d %b %Y %H:%M')}*",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def read_history_records() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def normalize_event(
    event: dict, trip_id: str, index: int, trip_start_date: str
) -> Optional[dict]:
    if not isinstance(event, dict):
        return None
    raw_date = str(event.get("date") or event.get("start") or trip_start_date)[:10]
    info     = str(event.get("info") or event.get("description") or event.get("title") or "").strip()
    summary  = str(event.get("summary") or event.get("title") or "").strip()
    if not raw_date or not info:
        return None
    try:
        valid_date = str(datetime.date.fromisoformat(raw_date))
    except Exception:
        valid_date = trip_start_date
    return {
        "id":      str(event.get("id") or f"{trip_id}-{index}"),
        "date":    valid_date,
        "summary": summary or f"Day {index + 1}",
        "info":    info[:2000],
    }


def normalize_trip_record(record: dict) -> Optional[dict]:
    if not isinstance(record, dict):
        return None
    destination = str(record.get("destination") or "").strip()
    start_date  = str(record.get("start_date")  or "").strip()
    itinerary   = str(record.get("itinerary") or record.get("final_itinerary") or "").strip()
    if not destination or not start_date or not itinerary:
        return None
    start_date = valid_date_string(start_date)
    trip_id    = str(record.get("id") or uuid.uuid4().hex[:8])
    raw_events = record.get("events") or record.get("calendar_events") or []
    events = []
    for idx, event in enumerate(raw_events if isinstance(raw_events, list) else []):
        normalized = normalize_event(event, trip_id, idx, start_date)
        if normalized:
            events.append(normalized)
    return {
        "id":          trip_id,
        "destination": destination,
        "start_date":  start_date,
        "itinerary":   itinerary,
        "ics_path":    str(record.get("ics_path") or ""),
        "events":      events,
    }


def load_normalized_history(persist: bool = False) -> list:
    raw_records = read_history_records()
    normalized  = [normalize_trip_record(r) for r in raw_records]
    normalized  = [r for r in normalized if r]
    if persist and len(normalized) != len(raw_records):
        _atomic_write(HISTORY_FILE, json.dumps(normalized, indent=2))
    return normalized


def save_history(history: list) -> None:
    _atomic_write(HISTORY_FILE, json.dumps(history, indent=2))


def build_calendar_feed(page: int = 1, per: int = 100) -> dict:
    history = load_normalized_history(persist=False)
    events: list[dict] = []
    seen:   set[str]   = set()
    for trip in history:
        trip_id = trip["id"]
        for idx, event in enumerate(trip.get("events", [])):
            key = f"{trip_id}:{event.get('date', idx)}"
            if key in seen:
                continue
            seen.add(key)
            events.append({
                "id":          str(event.get("id") or f"{trip_id}-{idx}"),
                "trip_id":     trip_id,
                "destination": trip["destination"],
                "date":        str(event.get("date", "")),
                "summary":     str(event.get("summary", f"Day {idx + 1}")),
                "info":        str(event.get("info", "")),
            })
    events.sort(key=lambda x: x.get("date", ""))
    start = (page - 1) * per
    return {
        "total":  len(events),
        "page":   page,
        "per":    per,
        "events": events[start : start + per],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# OVERLAP DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def check_date_overlap(start_date_str: str, duration_days: int) -> list[dict]:
    """Check if the requested trip dates overlap with any existing trips in history."""
    try:
        new_start = datetime.date.fromisoformat(valid_date_string(start_date_str))
    except Exception:
        return []
    new_end = new_start + datetime.timedelta(days=max(1, duration_days) - 1)

    history = load_normalized_history(persist=False)
    conflicts = []
    for trip in history:
        trip_start_str = trip.get("start_date", "")
        try:
            trip_start = datetime.date.fromisoformat(valid_date_string(trip_start_str))
        except Exception:
            continue
        # Determine trip duration from events or default to number of events
        events = trip.get("events", [])
        if events:
            event_dates = []
            for ev in events:
                try:
                    event_dates.append(datetime.date.fromisoformat(ev.get("date", "")[:10]))
                except Exception:
                    pass
            if event_dates:
                trip_end = max(event_dates)
            else:
                trip_end = trip_start
        else:
            trip_end = trip_start

        # Check overlap: two ranges overlap if start1 <= end2 AND start2 <= end1
        if new_start <= trip_end and trip_start <= new_end:
            conflicts.append({
                "destination": trip.get("destination", "Unknown"),
                "start_date": str(trip_start),
                "end_date": str(trip_end),
                "trip_id": trip.get("id", ""),
            })
    return conflicts


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH NODES
# ═══════════════════════════════════════════════════════════════════════════════

def parse_goal_node(state: TravelState) -> TravelState:
    t0           = datetime.datetime.now()
    request_text = state["user_request"]
    memory       = load_memory()
    logger.log("PLANNER", "Parsing request", request_text[:100])

    prompt = (
        "Extract travel details. Return ONLY raw JSON, no markdown fences.\n"
        f"User memory defaults: diet={memory['diet']}, budget_style={memory['budget_style']}\n"
        f"Request: \"{request_text}\"\n"
        f"Today: {datetime.date.today()}\n\n"
        "JSON template:\n"
        '{"destination":"city","origin":"Delhi, India","duration_days":3,'
        '"budget_inr":30000,"start_date":"YYYY-MM-DD","nationality":"Indian",'
        '"diet":"Any","travellers":1}'
    )

    parsed: Optional[dict] = None
    try:
        parsed = normalize_goal(
            extract_json_object(llm_call(prompt, "")), request_text, memory
        )
    except Exception as primary_err:
        logger.log("PLANNER", "Primary parse failed", str(primary_err))
        simpler = (
            "Extract ONLY these fields from the text as JSON: "
            f"destination, duration_days, budget_inr, travellers. Text: {request_text}"
        )
        try:
            simple = extract_json_object(llm_call(simpler, ""))
            parsed = normalize_goal(
                {
                    "destination":   simple.get("destination"),
                    "origin":        memory.get("home_city", "Delhi, India"),
                    "duration_days": simple.get("duration_days"),
                    "budget_inr":    simple.get("budget_inr"),
                    "travellers":    simple.get("travellers", 1),
                    "start_date":    str(datetime.date.today() + datetime.timedelta(days=7)),
                    "nationality":   memory.get("nationality", "Indian"),
                    "diet":          memory.get("diet", "Any"),
                },
                request_text,
                memory,
            )
        except Exception as secondary_err:
            logger.log("PLANNER", "Secondary parse failed", str(secondary_err))
            parsed = regex_goal_fallback(request_text, memory)

    ms = int((datetime.datetime.now() - t0).total_seconds() * 1000)
    logger.log("PLANNER", "Goal parsed", json.dumps(parsed), duration_ms=ms)
    # ── Check for date overlaps with existing trips ────────────────────────
    overlap_warning = ""
    conflicts = check_date_overlap(
        parsed.get("start_date", ""),
        parse_int(parsed.get("duration_days"), 3),
    )
    if conflicts:
        parts = []
        for c in conflicts:
            parts.append(
                f"  • {c['destination']} ({c['start_date']} to {c['end_date']})"
            )
        overlap_warning = (
            "⚠️ **Schedule Conflict Detected!** Your requested dates overlap "
            "with existing trip(s):\n" + "\n".join(parts) + "\n"
            "Consider choosing different dates to avoid conflicts."
        )
        logger.log("PLANNER", "Overlap detected", overlap_warning[:120])

    return {**state, "parsed_goal": parsed, "memory": memory, "overlap_warning": overlap_warning, "logs": [{"node": "PLANNER"}]}


def weather_node(state: TravelState) -> TravelState:
    t0          = datetime.datetime.now()
    goal        = state["parsed_goal"]
    destination = goal.get("destination", "")
    weather     = tool_get_weather(destination) if destination else "Weather unavailable"
    ms          = int((datetime.datetime.now() - t0).total_seconds() * 1000)
    logger.log("WEATHER", "Weather fetched", weather, duration_ms=ms)
    return {**state, "weather_info": weather, "logs": [{"node": "WEATHER"}]}


def researcher_node(state: TravelState) -> TravelState:
    t0   = datetime.datetime.now()
    goal = state["parsed_goal"]
    diet = goal.get("diet", "Any")

    task = (
        f"Research realistic round-trip flight costs from {goal['origin']} to "
        f"{goal['destination']}, {goal.get('budget_style', 'mid-range')} hotel prices per night, "
        f"and the top 5 attractions with approximate INR entry fees. "
        f"Also list 3 highly-rated {diet}-friendly restaurants with price range. "
        f"Note if any attraction requires advance booking. Provide concrete numbers in INR."
    )
    fallback = (
        f"Flights and hotels for {goal['destination']} should be verified manually. "
        "Use the stated budget to shortlist realistic options."
    )
    result = run_agent(task, goal.get("destination", ""), fallback)
    ms     = int((datetime.datetime.now() - t0).total_seconds() * 1000)
    logger.log("PLAN_TRIP", "Research complete", result[:180], duration_ms=ms)
    return {**state, "research": result, "logs": [{"node": "PLAN_TRIP"}]}


def budgeter_node(state: TravelState) -> TravelState:
    t0            = datetime.datetime.now()
    goal          = state["parsed_goal"]
    duration_days = parse_int(goal.get("duration_days"), 3)
    target_budget = parse_int(goal.get("budget_inr"),    30_000)
    travellers    = max(1, parse_int(goal.get("travellers"), 1))

    # Deterministic fallback budget (per-person then scaled)
    food           = duration_days * 900  * travellers
    activities     = duration_days * 500  * travellers
    hotel          = duration_days * 2_500
    flights        = 8_000 * travellers
    fallback_total = flights + hotel + food + activities
    fallback_summary = {
        "flights":    flights,
        "hotel":      hotel,
        "food":       food,
        "activities": activities,
        "total":      fallback_total,
        "verdict":    "PASS" if fallback_total <= target_budget else "OVER BUDGET",
    }

    task = f"""
Calculate a detailed travel budget for this exact trip.

Destination:    {goal['destination']}
Origin:         {goal['origin']}
Duration:       {duration_days} days
Travellers:     {travellers} people
Budget target:  INR {target_budget:,} TOTAL (all {travellers} people)
Research data:  {state.get('research', '')}

Use the Calculator tool for all arithmetic.
Show cost per person AND total.
Return a paragraph explanation followed by EXACTLY this JSON on the last line:
{{"flights": 6500, "hotel": 7500, "food": 2700, "activities": 1500, "total": 18200, "verdict": "PASS"}}
All values are TOTAL across all {travellers} travellers.
verdict must be exactly PASS or OVER BUDGET.
"""
    result  = run_agent(task, goal.get("destination", ""), json.dumps(fallback_summary))
    summary = extract_budget_summary(result) or fallback_summary
    if summary.get("total", 0) > target_budget:
        summary["verdict"] = "OVER BUDGET"

    ms = int((datetime.datetime.now() - t0).total_seconds() * 1000)
    logger.log("PLAN_TRIP", "Budget complete", json.dumps(summary), duration_ms=ms)
    return {
        **state,
        "budget_plan":    result,
        "budget_summary": summary,
        "logs":           [{"node": "PLAN_TRIP"}],
    }


def feasibility_node(state: TravelState) -> TravelState:
    t0           = datetime.datetime.now()
    summary      = state.get("budget_summary", {})
    is_infeasible = str(summary.get("verdict", "")).upper() == "OVER BUDGET"
    ms            = int((datetime.datetime.now() - t0).total_seconds() * 1000)
    logger.log(
        "PLAN_TRIP",
        "Feasibility checked",
        "OVER BUDGET" if is_infeasible else "PASS",
        duration_ms=ms,
    )
    return {**state, "is_infeasible": is_infeasible, "logs": [{"node": "PLAN_TRIP"}]}


def curator_node(state: TravelState) -> TravelState:
    t0         = datetime.datetime.now()
    goal       = state["parsed_goal"]
    weather    = state.get("weather_info", "")
    travellers = max(1, parse_int(goal.get("travellers"), 1))
    fallback   = build_fallback_activities(goal)

    diet_note = (
        f"CRITICAL: All restaurant suggestions MUST be {goal['diet']}-friendly. "
        f"No exceptions."
        if goal.get("diet", "Any").lower() not in ("any", "")
        else "Suggest popular local cuisine options."
    )

    group_note = (
        f"This is a GROUP trip for {travellers} people. "
        "Suggest group-friendly venues, shared transport, and group booking tips."
        if travellers > 1
        else ""
    )

    overlap_note = state.get("overlap_warning", "")
    overlap_instruction = ""
    if overlap_note:
        overlap_instruction = (
            f"\n\nIMPORTANT SCHEDULING NOTE: {overlap_note}\n"
            "Mention this conflict clearly at the start of the itinerary."
        )

    task = f"""
You are an award-winning luxury travel concierge. Create an EXCEPTIONAL, highly detailed,
{goal['duration_days']}-day itinerary for {goal['destination']} only.

Context:
- Traveller diet:   {goal['diet']} — {diet_note}
- Group size:       {travellers} people — {group_note}
- Budget style:     {goal.get('budget_style', 'mid-range')}
- Current weather:  {weather or 'check locally'}
- Origin:           {goal['origin']}
{overlap_instruction}

STRICT RULES:
1. Name REAL, specific restaurants that are {goal['diet']}-friendly.
2. Name REAL, specific attractions with approximate INR entry cost.
3. Specify EXACT transport method for each move (including group transport options).
4. Include OPENING HOURS for major attractions.
5. Separate breakfast, lunch, and dinner.
6. End each day with one "Local tip" and one "Indoor backup" alternative.
7. Day 1 must start with arrival and hotel check-in.
8. Day {goal['duration_days']} must end with departure logistics.
9. Include a "Group tip" for each day with advice specific to travelling in a group.

Use this EXACT format for every day:
**Day X — [Creative Evocative Theme]**
- 07:30 Breakfast: [Named café] — try [specific dish, ~₹NNN/person]
- 09:30 Morning: [Specific attraction] (open HH:MM–HH:MM, entry ₹NNN) — [vivid description]
  → Transport: [specific mode, cost, duration]
- 13:00 Lunch: [Named restaurant] — recommended: [specific dish, ~₹NNN/person]
- 15:00 Afternoon: [Specific activity/attraction] — [vivid description]
  → Transport: [specific mode, cost]
- 19:00 Sunset: [Viewpoint/spot name] — [one sentence]
- 20:00 Dinner: [Named restaurant] — must-try: [specific dish, ~₹NNN/person]
- Local tip: [Genuine insider knowledge]
- Indoor backup: [Specific named indoor venue]
- Group tip: [Advice for {travellers} people travelling together]
"""
    activities = run_agent(task, goal.get("destination", ""), fallback)
    activities = sanitize_output(activities, goal.get("destination", ""), fallback)
    ms = int((datetime.datetime.now() - t0).total_seconds() * 1000)
    logger.log("PLAN_TRIP", "Itinerary created", activities[:180], duration_ms=ms)
    return {**state, "activities": activities, "logs": [{"node": "PLAN_TRIP"}]}


def safety_node(state: TravelState) -> TravelState:
    t0   = datetime.datetime.now()
    goal = state["parsed_goal"]
    fallback = (
        f"Visa: verify official entry rules for {goal['nationality']} travelers "
        f"visiting {goal['destination']}.\n"
        "Carry local currency, keep ID copies, and use registered transport.\n"
        "Emergency: 112 (India universal) or local equivalent."
    )
    prompt = (
        f"Give a structured safety & logistics briefing for a {goal['nationality']} "
        f"traveler visiting {goal['destination']} for {goal.get('duration_days', 3)} days"
        f" (group of {goal.get('travellers', 1)}).\n"
        "Cover:\n"
        "1. Visa requirements and e-visa link if available\n"
        "2. Currency / payment tips\n"
        "3. Two specific safety tips for this destination\n"
        "4. Emergency contact numbers (police, ambulance, tourist helpline)\n"
        "5. One health/vaccination note if relevant\n"
        "6. Group travel safety tip (scam awareness, staying together)\n"
        "Be concise, factual, and use bullet points."
    )
    safety = llm_call(prompt, fallback)
    safety = sanitize_output(safety, goal.get("destination", ""), fallback)
    ms     = int((datetime.datetime.now() - t0).total_seconds() * 1000)
    logger.log("PLAN_TRIP", "Safety notes ready", safety[:180], duration_ms=ms)
    return {**state, "safety_info": safety, "logs": [{"node": "PLAN_TRIP"}]}


def packing_node(state: TravelState) -> TravelState:
    t0         = datetime.datetime.now()
    goal       = state["parsed_goal"]
    weather    = state.get("weather_info", "")
    duration   = goal.get("duration_days", 3)
    travellers = max(1, parse_int(goal.get("travellers"), 1))
    diet       = goal.get("diet", "Any")

    prompt = (
        f"Generate a concise packing checklist for a {duration}-day trip to "
        f"{goal['destination']} from {goal['origin']} for {travellers} people.\n"
        f"Current weather there: {weather or 'unknown'}.\n"
        f"Diet preference: {diet}.\n"
        f"Budget style: {goal.get('budget_style', 'mid-range')}.\n\n"
        "Group items under these headings and use checkboxes (- [ ]):\n"
        "**Documents & Money** | **Clothing** | **Toiletries** | "
        "**Tech & Gadgets** | **Health & Safety** | **Group Essentials** | **Extras**\n"
        "Keep it practical and destination-specific. Max 8 items per group.\n"
        "Add one 'Group Essentials' section for shared items (e.g., first-aid kit, "
        "portable charger, shared snacks)."
    )
    fallback = (
        "**Documents & Money**\n- [ ] Passport / ID\n- [ ] Travel insurance\n- [ ] Local currency\n\n"
        "**Clothing**\n- [ ] Weather-appropriate layers\n- [ ] Comfortable walking shoes\n\n"
        "**Tech & Gadgets**\n- [ ] Phone charger & power bank\n- [ ] Universal adapter\n\n"
        "**Health & Safety**\n- [ ] Basic first-aid kit\n- [ ] Personal medications\n\n"
        "**Group Essentials**\n- [ ] Shared power bank\n- [ ] Group WhatsApp with hotel/itinerary info"
    )
    packing = llm_call(prompt, fallback)
    if looks_bad_output(packing):
        packing = fallback
    ms = int((datetime.datetime.now() - t0).total_seconds() * 1000)
    logger.log("PLAN_TRIP", "Packing list ready", packing[:100], duration_ms=ms)
    return {**state, "packing_list": packing, "logs": [{"node": "PLAN_TRIP"}]}


def reflect_node(state: TravelState) -> TravelState:
    t0          = datetime.datetime.now()
    activities  = state.get("activities", "")
    destination = state["parsed_goal"].get("destination", "")
    needs_replan = (
        looks_bad_output(activities)
        or output_conflicts_with_destination(activities, destination)
    )
    note = (
        "REPLAN: Activity output looked unsafe or mismatched."
        if needs_replan
        else "APPROVED: Plan looks consistent."
    )
    ms = int((datetime.datetime.now() - t0).total_seconds() * 1000)
    logger.log("PLAN_TRIP", "Reflection complete", note, duration_ms=ms)
    return {
        **state,
        "reflect_notes": note,
        "needs_replan":  needs_replan,
        "logs":          [{"node": "PLAN_TRIP"}],
    }


def replan_node(state: TravelState) -> TravelState:
    t0    = datetime.datetime.now()
    fixed = build_fallback_activities(state["parsed_goal"])
    ms    = int((datetime.datetime.now() - t0).total_seconds() * 1000)
    logger.log("PLAN_TRIP", "Fallback replan applied", fixed[:100], duration_ms=ms)
    return {
        **state,
        "activities":    fixed,
        "reflect_notes": "APPROVED: Replaced unreliable itinerary with deterministic fallback.",
        "needs_replan":  False,
        "logs":          [{"node": "PLAN_TRIP"}],
    }


def assemble_node(state: TravelState) -> TravelState:
    t0            = datetime.datetime.now()
    goal          = state["parsed_goal"]
    destination   = goal.get("destination",  "Unknown")
    duration_days = parse_int(goal.get("duration_days"), 3)
    start_date    = valid_date_string(goal.get("start_date"))
    travellers    = max(1, parse_int(goal.get("travellers"), 1))
    logger.log("ASSEMBLE", "Assembling final output", destination)

    # ── Infeasible path ──────────────────────────────────────────────────────
    if state.get("is_infeasible"):
        summary    = state.get("budget_summary", {})
        total_over = summary.get("total", 0)
        per_person = round(total_over / travellers) if total_over else 0
        output = "\n".join([
            f"# Trip Feasibility Report — {destination}",
            "",
            f"**Your budget:** ₹{goal.get('budget_inr', 0):,} (for {travellers} person{'s' if travellers > 1 else ''})",
            f"**Estimated cost:** ₹{total_over:,} total (≈ ₹{per_person:,}/person)",
            "",
            "## Why it exceeds your budget",
            state.get("budget_plan", "Budget details unavailable."),
            "",
            "## Alternatives to make it work",
            f"1. **Increase budget** to at least ₹{total_over:,} total",
            f"2. **Shorten trip** to {max(1, duration_days - 1)} day(s)",
            "3. **Choose a lower-cost destination** nearby",
            "4. **Travel off-season** for cheaper flights and hotels",
            f"5. **Split costs smartly** — each person contributes ₹{per_person:,}",
        ])
        ms = int((datetime.datetime.now() - t0).total_seconds() * 1000)
        logger.log("ASSEMBLE", "Infeasible output ready", output[:80], duration_ms=ms)
        return {
            **state,
            "final_itinerary":  output,
            "ics_path":         "",
            "calendar_events":  [],
            "logs":             [{"node": "ASSEMBLE"}],
        }

    # ── Happy path ───────────────────────────────────────────────────────────
    activities = sanitize_output(
        state.get("activities", ""),
        destination,
        build_fallback_activities(goal),
    )
    budget_summary = state.get("budget_summary", {})
    safety_info = sanitize_output(
        state.get("safety_info", ""),
        destination,
        f"Verify visa and safety guidance for {destination} with official sources.",
    )
    weather_info = state.get("weather_info", "")
    packing_list = state.get("packing_list", "")
    overlap_warning = state.get("overlap_warning", "")

    final_itinerary = format_final_itinerary(
        goal, activities, budget_summary, safety_info, weather_info, packing_list
    )

    # Prepend overlap warning if present
    if overlap_warning:
        final_itinerary = (
            "> " + overlap_warning.replace("\n", "\n> ") +
            "\n\n" + final_itinerary
        )

    calendar_events = extract_calendar_events(activities, start_date)

    ics_events_payload = [
        {
            "summary":     ev.get("summary", f"Day {i+1}"),
            "description": ev.get("info", ""),
        }
        for i, ev in enumerate(calendar_events)
    ]
    ics_path = tool_ics_writer(json.dumps({
        "destination":   destination,
        "start_date":    start_date,
        "duration_days": duration_days,
        "events":        ics_events_payload,
    }))

    updated_memory = save_memory(
        destination,
        goal.get("diet", "Any"),
        goal.get("budget_inr", 0),
    )

    ms = int((datetime.datetime.now() - t0).total_seconds() * 1000)
    logger.log("ASSEMBLE", f"Done — {len(calendar_events)} events", ics_path, duration_ms=ms)
    return {
        **state,
        "final_itinerary":   final_itinerary,
        "ics_path":          ics_path,
        "calendar_events":   calendar_events,
        "memory":            updated_memory,
        "logs":              [{"node": "ASSEMBLE"}],
    }



# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

def route_feasibility(state: TravelState) -> str:
    return "assemble" if state.get("is_infeasible") else "curator"


def route_reflect(state: TravelState) -> str:
    return "replan" if state.get("needs_replan") else "packing"


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(TravelState)

    # Nodes
    graph.add_node("parse_goal",  parse_goal_node)
    graph.add_node("weather",     weather_node)
    graph.add_node("researcher",  researcher_node)
    graph.add_node("budgeter",    budgeter_node)
    graph.add_node("feasibility", feasibility_node)
    graph.add_node("curator",     curator_node)
    graph.add_node("safety",      safety_node)
    graph.add_node("reflect",     reflect_node)
    graph.add_node("replan",      replan_node)
    graph.add_node("packing",     packing_node)
    graph.add_node("assemble",    assemble_node)

    # Edges
    graph.set_entry_point("parse_goal")
    graph.add_edge("parse_goal",  "weather")
    graph.add_edge("weather",     "researcher")
    graph.add_edge("researcher",  "budgeter")
    graph.add_edge("budgeter",    "feasibility")
    graph.add_conditional_edges(
        "feasibility",
        route_feasibility,
        {"curator": "curator", "assemble": "assemble"},
    )
    graph.add_edge("curator",  "safety")
    graph.add_edge("safety",   "reflect")
    graph.add_conditional_edges(
        "reflect",
        route_reflect,
        {"replan": "replan", "packing": "packing"},
    )
    graph.add_edge("replan",  "packing")
    graph.add_edge("packing", "assemble")
    graph.add_edge("assemble", END)
    return graph.compile()


travel_graph = build_graph()


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Static / pages ────────────────────────────────────────────────────────────

@app.route("/")
def index_page():
    vite_page = BASE_DIR / "travel-agent-ai" / "dist" / "index.html"
    if vite_page.exists():
        return vite_page.read_text(encoding="utf-8")
    legacy_page = BASE_DIR / "index.html"
    if legacy_page.exists():
        return legacy_page.read_text(encoding="utf-8")
    return "<h1>Place index.html in the project root or build Vite app.</h1>", 404


@app.route("/calendar")
def calendar_page():
    page = BASE_DIR / "calendar.html"
    if page.exists():
        return page.read_text(encoding="utf-8")
    return "<h1>Place calendar.html in the project root.</h1>", 404


# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def api_health():
    uptime = (datetime.datetime.utcnow() - _STARTUP_TIME).total_seconds()
    return jsonify({
        "status":         "ok",
        "version":        VERSION,
        "model":          MODEL,
        "uptime_seconds": round(uptime, 1),
    })


# ── Plan ──────────────────────────────────────────────────────────────────────

@app.route("/api/plan", methods=["POST"])
def api_plan():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"success": False, "error": "No message provided"}), 400

    logger.reset()
    logger.log("API", "Request received", message[:120])

    initial: TravelState = {
        "user_request":     message,
        "parsed_goal":      {},
        "research":         "",
        "weather_info":     "",
        "budget_plan":      "",
        "budget_summary":   {},
        "is_infeasible":    False,
        "activities":       "",
        "packing_list":     "",
        "safety_info":      "",
        "reflect_notes":    "",
        "needs_replan":     False,
        "final_itinerary":  "",
        "ics_path":         "",
        "calendar_events":  [],
        "memory":           load_memory(),
        "overlap_warning":  "",
        "logs":             [],
    }

    try:
        final = travel_graph.invoke(initial)

        trip_record = normalize_trip_record({
            "id":          uuid.uuid4().hex[:8],
            "destination": final["parsed_goal"].get("destination"),
            "start_date":  final["parsed_goal"].get("start_date"),
            "itinerary":   final["final_itinerary"],
            "ics_path":    final["ics_path"],
            "events":      final["calendar_events"],
        })

        history = load_normalized_history(persist=True)
        if trip_record:
            history.append(trip_record)
            save_history(history)

        ics_url = f"/itineraries/{final['ics_path']}" if final.get("ics_path") else None

        return jsonify({
            "success":       True,
            "trip":          trip_record,
            "ics_url":       ics_url,
            "weather":       final.get("weather_info", ""),
            "logs":          logger.get_logs(),
            "is_infeasible": final.get("is_infeasible", False),
            "budget_summary": final.get("budget_summary", {}),
            "memory":        final.get("memory", load_memory()),
            "travellers":    final["parsed_goal"].get("travellers", 1),
        })
    except Exception as exc:
        logger.log("API", "Unhandled error", str(exc))
        return jsonify({"success": False, "error": str(exc), "logs": logger.get_logs()}), 500


# ── History ───────────────────────────────────────────────────────────────────

@app.route("/api/history")
def api_history():
    return jsonify(load_normalized_history(persist=True))


@app.route("/api/trip/<trip_id>", methods=["GET"])
def api_get_trip(trip_id: str):
    history = load_normalized_history()
    for trip in history:
        if trip["id"] == trip_id:
            return jsonify(trip)
    return jsonify({"error": "Trip not found"}), 404


@app.route("/api/trip/<trip_id>", methods=["DELETE"])
def api_delete_trip(trip_id: str):
    history     = load_normalized_history()
    new_history = [t for t in history if t["id"] != trip_id]
    if len(new_history) == len(history):
        return jsonify({"error": "Trip not found"}), 404
    save_history(new_history)
    return jsonify({"success": True, "deleted_id": trip_id})


# ── Memory ────────────────────────────────────────────────────────────────────

@app.route("/api/memory", methods=["GET"])
def api_get_memory():
    return jsonify(load_memory())


@app.route("/api/memory", methods=["PUT"])
def api_update_memory():
    payload     = request.get_json(silent=True) or {}
    memory      = load_memory()
    allowed_keys = {"diet", "budget_style", "home_city", "nationality"}
    for key in allowed_keys:
        if key in payload:
            memory[key] = str(payload[key]).strip()
    _atomic_write(MEMORY_FILE, json.dumps(memory, indent=2))
    return jsonify({"success": True, "memory": memory})


# ── Search ────────────────────────────────────────────────────────────────────

@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Missing query parameter ?q="}), 400
    results = tool_web_search(f"travel destination {q} India INR")
    return jsonify({"query": q, "results": results})


# ── Logs ──────────────────────────────────────────────────────────────────────

@app.route("/api/logs")
def api_logs():
    return jsonify({"logs": logger.get_logs()})


# ── Calendar Events ───────────────────────────────────────────────────────────

@app.route("/api/calendar-events")
def api_calendar_events():
    try:
        page = max(1, int(request.args.get("page", 1)))
        per  = min(500, max(1, int(request.args.get("per", 100))))
    except ValueError:
        page, per = 1, 100
    return jsonify(build_calendar_feed(page=page, per=per))


# ── Convenience aliases (spec-requested) ──────────────────────────────────────

@app.route("/plan-trip", methods=["POST"])
def plan_trip_alias():
    return api_plan()

@app.route("/history")
def history_alias():
    return api_history()

@app.route("/events")
def events_alias():
    return api_calendar_events()

@app.route("/logs")
def logs_alias():
    return api_logs()

@app.route("/memory")
def memory_alias():
    return api_get_memory()


# ── ICS download ──────────────────────────────────────────────────────────────

@app.route("/itineraries/<filename>")
def download_ics(filename: str):
    safe_name = Path(filename).name  # Prevent path traversal
    return send_from_directory(str(ITINERARY_DIR), safe_name, as_attachment=True)


# ── Currency helper (standalone) ──────────────────────────────────────────────

@app.route("/api/convert")
def api_convert():
    """Quick currency conversion endpoint. ?from=USD&to=INR&amount=100"""
    from_cur = request.args.get("from", "USD").upper()
    to_cur   = request.args.get("to",   "INR").upper()
    amount   = request.args.get("amount", "1")
    result   = tool_currency_convert(f"{from_cur} {amount} to {to_cur}")
    return jsonify({"query": f"{from_cur} {amount} to {to_cur}", "result": result})


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    routes = [
        "  GET  /                      → Frontend UI",
        "  GET  /calendar              → Calendar page",
        "  GET  /api/health            → Health check",
        "  POST /api/plan              → Plan a trip",
        "  POST /plan-trip             → Plan a trip (alias)",
        "  GET  /api/history           → All past trips",
        "  GET  /api/trip/<id>         → Single trip detail",
        "  DEL  /api/trip/<id>         → Delete a trip",
        "  GET  /api/memory            → User preferences",
        "  PUT  /api/memory            → Update preferences",
        "  GET  /api/search?q=         → Destination search",
        "  GET  /api/logs              → Agent step logs",
        "  GET  /api/calendar-events   → Calendar events",
        "  GET  /api/convert           → Currency conversion",
        "  GET  /itineraries/<file>    → Download .ics file",
    ]
    print()
    print("═" * 62)
    print(f"  Travel Concierge Agent  v{VERSION}")
    print("═" * 62)
    print(f"  URL      : http://{HOST}:{PORT}/")
    print(f"  Model    : {MODEL}")
    print("─" * 62)
    print("  Routes:")
    for r in routes:
        print(r)
    print("═" * 62)
    print()
    app.run(host=HOST, port=PORT, debug=True, use_reloader=False)
