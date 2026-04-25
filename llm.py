"""
llm.py — LLM Provider Abstraction
===================================
Provides a single factory function ``get_llm()`` that returns a
LangChain-compatible chat model.  Supports:

  • OpenAI  (default, uses gpt-4o-mini)
  • Ollama  (opt-in via USE_OLLAMA=true in .env)

All downstream modules import their LLM instance from here,
ensuring a single point of configuration.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from utils import get_logger

# ── Load environment variables ──────────────────────────────
load_dotenv()

logger = get_logger(__name__)

# ────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.7):
    """
    Return a LangChain chat model based on environment config.

    When USE_OLLAMA is truthy, an Ollama-backed model is returned;
    otherwise an OpenAI model is used.

    Parameters
    ----------
    temperature : float, default 0.7
        Sampling temperature — higher → more creative.

    Returns
    -------
    BaseChatModel
        A LangChain chat model instance.
    """
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() in ("true", "1", "yes")

    if use_ollama:
        # ── Ollama (local) ─────────────────────────────────
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "Install langchain-ollama to use Ollama: "
                "pip install langchain-ollama"
            ) from exc

        model = os.getenv("OLLAMA_MODEL", "llama3")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        logger.info("Using Ollama  model=%s  base_url=%s", model, base_url)

        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )

    # ── OpenAI (default) ───────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        logger.warning(
            "OPENAI_API_KEY is not set. LLM calls will fail. "
            "Set the key in .env or as an environment variable."
        )

    logger.info("Using OpenAI  model=%s", model)

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
    )
