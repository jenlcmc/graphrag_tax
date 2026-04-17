"""Tax Filing Assistant — interactive CLI chatbot.

Combines hybrid GraphRAG retrieval with an LLM to answer federal income tax
questions for individual filers (Tax Year 2024).  Run with --compare to see
how GraphRAG-augmented responses differ from LLM-only responses side by side.

Prerequisites:
  python scripts/build_pipeline.py   (build data/processed/ artifacts first)
    For Claude/Gemini: set API keys in .env
    For Ollama: run local server and pull a model

Usage:
  # Default: hybrid GraphRAG + Claude
  python chatbot.py

  # Choose retrieval mode
  python chatbot.py --mode graph
  python chatbot.py --mode vector
  python chatbot.py --mode none          # LLM only, no retrieval

  # Compare GraphRAG vs LLM-only side by side
  python chatbot.py --compare

  # Use Gemini instead of Claude
  python chatbot.py --model gemini

    # Use a local open-source model via Ollama
        OLLAMA_MODEL=qwen3.5:2b python chatbot.py --model ollama

In-session commands:
  /mode <none|vector|graph|hybrid>   switch retrieval mode mid-session
  /sources                            show sources retrieved for last query
  /clear                              clear conversation history
  /quit  or  exit                     exit the chatbot
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from urllib import error as urlerror
from urllib import request as urlrequest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import config as cfg
from src.retrieval.hybrid_retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a knowledgeable U.S. federal tax filing assistant specializing in \
individual income tax returns (Form 1040, Tax Year 2024).

Guidelines:
- Cite the specific statute or IRS publication for every substantive claim \
(e.g., "Under 26 USC §63...", "Per IRS Pub. 17, Chapter 4...", \
"See Sch. D Instructions, Line 1...").
- When a question involves dollar amounts or calculations, show each step \
explicitly (income → deductions → taxable income → tax brackets → final tax).
- If the retrieved context does not cover the question, say so clearly rather \
than guessing.
- Keep answers factual and grounded in the provided excerpts.

{context_block}"""

NO_RETRIEVAL_SYSTEM_PROMPT = """\
You are a knowledgeable U.S. federal tax filing assistant specializing in \
individual income tax returns (Form 1040, Tax Year 2024).

Answer using your training knowledge. Cite statutory sections and IRS \
publications where relevant. Show calculation steps for numeric questions."""


# ---------------------------------------------------------------------------
# Core chatbot class
# ---------------------------------------------------------------------------

class TaxChatbot:
    """Multi-turn tax assistant backed by hybrid GraphRAG and an LLM."""

    def __init__(
        self,
        retriever: HybridRetriever,
        model: str,
        provider: str,
        mode: str = "hybrid",
        top_k: int = 10,
    ) -> None:
        self.retriever   = retriever
        self.model       = model
        self.provider    = provider
        self.mode        = mode
        self.top_k       = top_k
        self.history: list[dict] = []      # OpenAI-style [{role, content}]
        self.last_sources: list[dict] = [] # chunks from last retrieval

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Process one user turn. Returns the assistant reply string."""
        chunks  = self.retriever.query(user_message, k=self.top_k, mode=self.mode)
        self.last_sources = chunks

        system  = self._build_system(chunks)
        reply   = self._call_llm(system, user_message)

        self.history.append({"role": "user",      "content": user_message})
        self.history.append({"role": "assistant",  "content": reply})
        return reply

    def clear_history(self) -> None:
        self.history.clear()
        self.last_sources.clear()

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system(self, chunks: list[dict]) -> str:
        if not chunks:
            return NO_RETRIEVAL_SYSTEM_PROMPT

        excerpts = []
        for chunk in chunks:
            sid  = chunk["section_id"]
            text = chunk.get("text", chunk.get("snippet", ""))
            excerpts.append(f"[{sid}]\n{text}")

        context_block = (
            "Relevant statutory and IRS guidance excerpts:\n\n"
            + "\n\n---\n\n".join(excerpts)
        )
        return SYSTEM_PROMPT.format(context_block=context_block)

    # ------------------------------------------------------------------
    # LLM backends
    # ------------------------------------------------------------------

    def _call_llm(self, system: str, user_message: str) -> str:
        if self.provider == "claude":
            return self._call_claude(system, user_message)
        if self.provider == "gemini":
            return self._call_gemini(system, user_message)
        if self.provider == "ollama":
            return self._call_ollama(system, user_message)
        raise ValueError(f"Unknown provider: {self.provider!r}")

    def _call_claude(self, system: str, user_message: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)

        # Build full message list including conversation history
        messages = list(self.history) + [{"role": "user", "content": user_message}]

        response = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=messages,
        )
        return response.content[0].text

    def _call_gemini(self, system: str, user_message: str) -> str:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=cfg.GEMINI_API_KEY)

        # Build the full turn sequence including conversation history.
        contents: list[types.Content] = []
        for msg in self.history:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=msg["content"])])
            )
        contents.append(
            types.Content(role="user", parts=[types.Part(text=user_message)])
        )

        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=2048,
            ),
        )
        return response.text

    def _call_ollama(self, system: str, user_message: str) -> str:
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": cfg.OLLAMA_THINK,
        }
        body = json.dumps(payload).encode("utf-8")
        endpoint = cfg.OLLAMA_BASE_URL.rstrip("/") + "/api/chat"
        req = urlrequest.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=cfg.OLLAMA_TIMEOUT_SECONDS) as resp:
                raw = resp.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = ""

            if exc.code == 404:
                raise RuntimeError(
                    f"Ollama model '{self.model}' was not found. "
                    f"Run: ollama pull {self.model}."
                ) from exc

            raise RuntimeError(
                f"Ollama HTTP error {exc.code}. "
                f"Endpoint={cfg.OLLAMA_BASE_URL}. Details={detail[:200]}"
            ) from exc
        except urlerror.URLError as exc:
            raise RuntimeError(
                "Failed to reach Ollama server. "
                f"Check OLLAMA_BASE_URL ({cfg.OLLAMA_BASE_URL}) and run 'ollama serve'."
            ) from exc

        data = json.loads(raw)
        message = data.get("message", {})
        content = str(message.get("content", "")).strip()
        if not content:
            raise RuntimeError(f"Ollama returned empty response: {raw[:200]}")
        return content


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

TERM_WIDTH = 80

def _header(text: str, char: str = "=") -> str:
    return f"\n{char * TERM_WIDTH}\n  {text}\n{char * TERM_WIDTH}"

def _wrap(text: str, indent: int = 0) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=TERM_WIDTH, initial_indent=prefix,
                         subsequent_indent=prefix)

def _print_response(reply: str, mode: str, sources: list[dict]) -> None:
    print()
    print(f"[{mode.upper()} mode]")
    print("-" * TERM_WIDTH)
    # Print reply preserving paragraph breaks
    for para in reply.split("\n"):
        print(para)
    print()

def _print_sources(sources: list[dict]) -> None:
    if not sources:
        print("  (no sources retrieved)")
        return
    print(f"  {len(sources)} sources retrieved:")
    for i, s in enumerate(sources, 1):
        score = s.get("score", s.get("vector_score", s.get("graph_score", 0.0)))
        print(f"  [{i}] {s['section_id']}  (score={score:.3f})")

def _print_comparison(
    user_message: str,
    bot_graph: TaxChatbot,
    bot_none: TaxChatbot,
) -> None:
    """Run the same query through GraphRAG and LLM-only, print side by side."""
    print(_header("RESPONSE — LLM only (no retrieval)", "-"))
    reply_none = bot_none.chat(user_message)
    _print_response(reply_none, "none", [])

    print(_header("RESPONSE — Hybrid GraphRAG + LLM", "-"))
    reply_graph = bot_graph.chat(user_message)
    _print_response(reply_graph, "hybrid", bot_graph.last_sources)

    print(_header("Sources used by GraphRAG", "-"))
    _print_sources(bot_graph.last_sources)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive tax filing assistant powered by Hybrid GraphRAG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["none", "vector", "graph", "hybrid"],
        default="hybrid",
        help="Retrieval mode (default: hybrid). 'none' = LLM only.",
    )
    parser.add_argument(
        "--model",
        choices=["claude", "gemini", "ollama"],
        default="claude",
        help="LLM backend (default: claude).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show LLM-only vs. hybrid GraphRAG response for each query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        metavar="K",
        help="Number of chunks to retrieve per query (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model == "claude":
        model_id = cfg.CLAUDE_MODEL
    elif args.model == "gemini":
        model_id = cfg.GEMINI_MODEL
    else:
        model_id = cfg.OLLAMA_MODEL

    print(_header("Tax Filing Assistant  (TY 2024  |  Individual Filers)"))
    print("Powered by Hybrid GraphRAG over federal tax law and IRS publications.")
    print(f"Model: {model_id}  |  Mode: {args.mode}")
    print("Type your tax question, or use /help for commands.\n")

    print("Loading retrieval index...", end=" ", flush=True)
    retriever = HybridRetriever.load(cfg)
    print("ready.")

    bot = TaxChatbot(retriever, model_id, provider=args.model, mode=args.mode, top_k=args.top_k)

    # Second bot for compare mode (LLM-only baseline)
    bot_none: TaxChatbot | None = None
    if args.compare:
        bot_none = TaxChatbot(
            retriever,
            model_id,
            provider=args.model,
            mode="none",
            top_k=args.top_k,
        )

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        # --- in-session commands ---
        if user_input.lower() in ("/quit", "quit", "exit"):
            print("Goodbye.")
            break

        if user_input.lower() == "/help":
            print(
                "\nCommands:\n"
                "  /mode <none|vector|graph|hybrid>  — switch retrieval mode\n"
                "  /sources                           — show sources from last query\n"
                "  /clear                             — reset conversation history\n"
                "  /quit                              — exit\n"
            )
            continue

        if user_input.lower().startswith("/mode "):
            new_mode = user_input.split(maxsplit=1)[1].strip().lower()
            if new_mode in ("none", "vector", "graph", "hybrid"):
                bot.mode = new_mode
                print(f"  Switched to {new_mode} mode.")
            else:
                print("  Unknown mode. Choose: none, vector, graph, hybrid.")
            continue

        if user_input.lower() == "/sources":
            print()
            _print_sources(bot.last_sources)
            continue

        if user_input.lower() == "/clear":
            bot.clear_history()
            if bot_none:
                bot_none.clear_history()
            print("  Conversation history cleared.")
            continue

        # --- normal turn ---
        try:
            if args.compare and bot_none is not None:
                _print_comparison(user_input, bot, bot_none)
            else:
                reply = bot.chat(user_input)
                _print_response(reply, bot.mode, bot.last_sources)

                if bot.last_sources:
                    print("Sources  (type /sources for details):")
                    for s in bot.last_sources[:5]:
                        score = s.get("score", s.get("vector_score", s.get("graph_score", 0.0)))
                        print(f"  · {s['section_id']}  ({score:.2f})")
                    if len(bot.last_sources) > 5:
                        print(f"  · ... and {len(bot.last_sources) - 5} more")

        except Exception as exc:
            print(f"\nError calling LLM: {exc}")
            print("Check your API key in .env and that the model is accessible.")


if __name__ == "__main__":
    main()
