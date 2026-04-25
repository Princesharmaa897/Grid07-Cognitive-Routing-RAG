"""
agent.py — LangGraph Agent Workflow  (Phase 2)
================================================
Each bot autonomously generates a social-media post by flowing
through a state-machine built with LangGraph:

    decide_topic → web_search → draft_post → END
"""

from __future__ import annotations

import json
import random
from typing import TypedDict

from langgraph.graph import StateGraph, END

from llm import get_llm
from personas import get_persona_by_id
from utils import get_logger, safe_parse_json

logger = get_logger(__name__)


class AgentState(TypedDict, total=False):
    bot_id: str
    persona: str
    persona_name: str
    topic: str
    search_query: str
    context: str
    post_content: str
    raw_output: str


# ── Mock Web Search ─────────────────────────────────────────

_MOCK_HEADLINES = {
    "ai": [
        "OpenAI announces GPT-5 with multimodal reasoning",
        "EU passes landmark AI Act enforcement rules",
        "Google DeepMind achieves protein-folding breakthrough",
        "NVIDIA H200 GPUs sell out in 48 hours",
        "Anthropic raises $7.5B Series E",
    ],
    "finance": [
        "Fed signals two rate cuts in Q3 as inflation cools",
        "Bitcoin surges past $150K after spot ETF inflows hit record",
        "NVIDIA surpasses Apple as most valuable company at $4.8T",
        "US 10-year Treasury yield drops below 3.5%",
        "Goldman Sachs launches AI robo-advisory platform",
    ],
    "doom": [
        "Global temps breach 1.7°C above pre-industrial levels",
        "WHO warns of novel H5N1 variant with pandemic potential",
        "Deepfake election ads flood social media",
        "UN report: 40% of topsoil degraded beyond recovery by 2030",
        "Cybersecurity breach exposes 800M medical records",
    ],
    "general": [
        "SpaceX Starship completes first orbital cargo delivery",
        "India UPI processes 30B transactions in a month",
        "Quantum computing startup IonQ reports first profit",
        "Reddit IPO debuts at $52 per share",
        "MIT develops room-temperature superconductor prototype",
    ],
}


def mock_searxng_search(query: str) -> list[str]:
    """Simulate SearXNG search with keyword-matched headlines."""
    q = query.lower()
    if any(k in q for k in ("ai", "gpt", "llm", "model", "robot")):
        pool = _MOCK_HEADLINES["ai"]
    elif any(k in q for k in ("stock", "market", "fed", "bitcoin", "crypto", "finance")):
        pool = _MOCK_HEADLINES["finance"]
    elif any(k in q for k in ("risk", "climate", "doom", "collapse", "threat")):
        pool = _MOCK_HEADLINES["doom"]
    else:
        pool = _MOCK_HEADLINES["general"]
    return random.sample(pool, min(len(pool), random.randint(3, 5)))


# ── Graph Nodes ─────────────────────────────────────────────

def decide_topic(state: AgentState) -> dict:
    llm = get_llm(temperature=0.8)
    prompt = (
        f"You are an AI social media bot with this persona:\n"
        f"{state['persona']}\n\n"
        f"Pick ONE trending topic to post about. "
        f"Reply with ONLY the topic (5-10 words)."
    )
    resp = llm.invoke(prompt)
    topic = resp.content.strip().strip('"').strip("'")
    logger.info("[%s] Topic: %s", state["bot_id"], topic)
    return {"topic": topic, "search_query": topic}


def web_search(state: AgentState) -> dict:
    headlines = mock_searxng_search(state["search_query"])
    context = "\n".join(f"• {h}" for h in headlines)
    logger.info("[%s] Search returned %d headlines", state["bot_id"], len(headlines))
    return {"context": context}


def draft_post(state: AgentState) -> dict:
    llm = get_llm(temperature=0.9)
    prompt = (
        f"Persona: {state['persona']}\n"
        f"Topic: {state['topic']}\n"
        f"Headlines:\n{state['context']}\n\n"
        f"Write a punchy, opinionated 280-char social media post in character.\n"
        f"Return ONLY valid JSON:\n"
        f'{{"bot_id":"{state["bot_id"]}","topic":"{state["topic"]}",'
        f'"post_content":"<your post>"}}'
    )
    resp = llm.invoke(prompt)
    raw = resp.content.strip()
    parsed = safe_parse_json(raw)
    if parsed and "post_content" in parsed:
        return {"post_content": parsed["post_content"], "raw_output": json.dumps(parsed, indent=2)}
    logger.warning("[%s] JSON parse failed, using raw", state["bot_id"])
    fb = {"bot_id": state["bot_id"], "topic": state["topic"], "post_content": raw[:280]}
    return {"post_content": fb["post_content"], "raw_output": json.dumps(fb, indent=2)}


# ── Graph Construction ──────────────────────────────────────

def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("decide_topic", decide_topic)
    graph.add_node("web_search", web_search)
    graph.add_node("draft_post", draft_post)
    graph.set_entry_point("decide_topic")
    graph.add_edge("decide_topic", "web_search")
    graph.add_edge("web_search", "draft_post")
    graph.add_edge("draft_post", END)
    return graph.compile()


def run_agent_for_bot(persona_id: str) -> dict:
    """Execute full LangGraph workflow for a persona."""
    persona = get_persona_by_id(persona_id)
    if persona is None:
        raise ValueError(f"Unknown persona: {persona_id}")
    initial: AgentState = {
        "bot_id": persona["id"], "persona": persona["description"],
        "persona_name": persona["name"], "topic": "", "search_query": "",
        "context": "", "post_content": "", "raw_output": "",
    }
    final = build_agent_graph().invoke(initial)
    logger.info("[%s] Agent workflow complete", persona_id)
    return {
        "bot_id": final["bot_id"], "topic": final.get("topic", ""),
        "post_content": final.get("post_content", ""),
        "raw_output": final.get("raw_output", ""),
    }
