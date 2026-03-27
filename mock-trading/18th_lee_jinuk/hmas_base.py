# """
# H-MAS Base Classes
# ==================
# Shared data structures, LLM client wrapper, and base Agent class
# used by all agents in the hierarchy.
# """

# from __future__ import annotations

# import json
# import os
# import re
# from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
# from typing import Any, Literal


# # ──────────────────────────────────────────────────────────────────────────────
# # LLM client — OpenAI 또는 Anthropic (Claude) 선택 가능
# # ──────────────────────────────────────────────────────────────────────────────

# Provider = Literal["openai", "anthropic"]


# class LLMClient:
#     """
#     Unified wrapper supporting OpenAI and Anthropic (Claude) backends.

#     Usage examples
#     --------------
#     # GPT-4o (기존 방식 그대로)
#     llm = LLMClient(model="gpt-4o")

#     # Claude — Anthropic SDK (권장)
#     llm = LLMClient(model="claude-sonnet-4-5", provider="anthropic")

#     # Claude — OpenAI-compatible endpoint (SDK 없이)
#     llm = LLMClient(
#         model="claude-opus-4-5",
#         provider="openai",
#         base_url="https://api.anthropic.com/v1",
#         api_key=os.getenv("ANTHROPIC_API_KEY"),
#     )

#     Environment variables
#     ---------------------
#     OPENAI_API_KEY    : OpenAI key (provider="openai", default)
#     ANTHROPIC_API_KEY : Anthropic key (provider="anthropic")
#     """

#     def __init__(
#         self,
#         model: str = "claude-sonnet-4-5",
#         provider: Provider = "anthropic",
#         base_url: str | None = None,
#         api_key: str | None = None,
#         temperature: float = 0.0,
#     ) -> None:
#         self.model = model
#         self.provider = provider
#         self.temperature = temperature

#         if provider == "anthropic":
#             self._client = self._init_anthropic(api_key)
#         else:
#             self._client = self._init_openai(api_key, base_url)

#     # ── 초기화 헬퍼 ──────────────────────────────────────────────────────────

#     def _init_openai(self, api_key: str | None, base_url: str | None):
#         try:
#             from openai import OpenAI
#             return OpenAI(
#                 api_key=api_key or os.getenv("OPENAI_API_KEY", "sk-placeholder"),
#                 base_url=base_url,
#             )
#         except ImportError:
#             return None  # stub mode

#     def _init_anthropic(self, api_key: str | None):
#         try:
#             import anthropic
#             return anthropic.Anthropic(
#                 api_key=api_key or os.getenv("ANTHROPIC_API_KEY", ""),
#             )
#         except ImportError:
#             # Anthropic SDK 미설치 → openai-compat fallback 시도
#             import warnings
#             warnings.warn(
#                 "anthropic package not installed. "
#                 "Falling back to OpenAI-compatible endpoint for Claude. "
#                 "Install with: pip install anthropic",
#                 stacklevel=2,
#             )
#             try:
#                 from openai import OpenAI
#                 return OpenAI(
#                     api_key=api_key or os.getenv("ANTHROPIC_API_KEY", ""),
#                     base_url="https://api.anthropic.com/v1",
#                 )
#             except ImportError:
#                 return None

#     # ── 핵심 chat 메서드 ─────────────────────────────────────────────────────

#     def chat(self, system: str, user: str) -> str:
#         """Send a chat request; return the assistant text."""
#         if self._client is None:
#             raise RuntimeError(
#                 "No LLM client available. "
#                 "Install 'openai' or 'anthropic' package."
#             )

#         if self.provider == "anthropic" and _is_anthropic_client(self._client):
#             return self._chat_anthropic(system, user)
#         else:
#             return self._chat_openai(system, user)

#     def _chat_openai(self, system: str, user: str) -> str:
#         """OpenAI chat completions (JSON mode)."""
#         response = self._client.chat.completions.create(
#             model=self.model,
#             temperature=self.temperature,
#             messages=[
#                 {"role": "system", "content": system},
#                 {"role": "user",   "content": user},
#             ],
#             response_format={"type": "json_object"},
#         )
#         return response.choices[0].message.content

#     def _chat_anthropic(self, system: str, user: str) -> str:
#         """
#         Anthropic Messages API.

#         Claude는 response_format=json_object를 지원하지 않으므로
#         시스템 프롬프트에 JSON-only 지시를 추가하고,
#         응답에서 JSON 블록을 추출합니다.
#         """
#         system_with_json = (
#             system
#             + "\n\nIMPORTANT: Respond with ONLY a valid JSON object. "
#             "No markdown fences, no explanation, no preamble."
#         )
#         message = self._client.messages.create(
#             model=self.model,
#             max_tokens=1024,
#             temperature=self.temperature,
#             system=system_with_json,
#             messages=[{"role": "user", "content": user}],
#         )
#         return message.content[0].text

#     # ── JSON 파싱 ────────────────────────────────────────────────────────────

#     def parse_json(self, system: str, user: str) -> dict[str, Any]:
#         """chat() → JSON 파싱. 마크다운 fence 자동 제거."""
#         raw = self.chat(system, user)
#         cleaned = _strip_json_fence(raw)
#         try:
#             return json.loads(cleaned)
#         except json.JSONDecodeError as exc:
#             raise ValueError(f"LLM returned invalid JSON: {raw!r}") from exc


# # ── 유틸리티 ──────────────────────────────────────────────────────────────────

# def _is_anthropic_client(client) -> bool:
#     """anthropic.Anthropic 인스턴스인지 확인 (import 없이)."""
#     return type(client).__module__.startswith("anthropic")


# def _strip_json_fence(text: str) -> str:
#     """```json ... ``` 또는 ``` ... ``` 펜스 제거."""
#     text = text.strip()
#     # ```json\n{...}\n``` 또는 ```\n{...}\n```
#     pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
#     match = re.match(pattern, text, re.DOTALL)
#     if match:
#         return match.group(1).strip()
#     return text


# # ──────────────────────────────────────────────────────────────────────────────
# # 편의 팩토리 함수
# # ──────────────────────────────────────────────────────────────────────────────

# def make_llm(
#     model: str | None = None,
#     provider: Provider | None = None,
#     temperature: float = 0.0,
# ) -> LLMClient:
#     """
#     환경변수 기반 자동 설정 팩토리.

#     우선순위:
#       1. 명시적 provider 인자
#       2. ANTHROPIC_API_KEY 있으면 → anthropic
#       3. OPENAI_API_KEY 있으면    → openai
#       4. 기본값 → openai (gpt-4o)
#     """
#     if provider is None:
#         if os.getenv("ANTHROPIC_API_KEY"):
#             provider = "anthropic"
#         else:
#             provider = "openai"

#     defaults = {
#         "openai":    "gpt-4o",
#         "anthropic": "claude-sonnet-4-5",
#     }
#     resolved_model = model or defaults[provider]

#     return LLMClient(model=resolved_model, provider=provider, temperature=temperature)


# # ──────────────────────────────────────────────────────────────────────────────
# # Shared data structures
# # ──────────────────────────────────────────────────────────────────────────────

# @dataclass
# class QualAgentOutput:
#     """Output of the Qualitative (Qual) Agent (B.3)."""
#     business_momentum: int      # 1–5
#     immediate_risk_severity: int  # 1–5 (5 = low risk)
#     management_trust: int       # 1–5
#     insight: str                # Korean professional briefing ~150 chars
#     raw: dict = field(default_factory=dict)

#     @property
#     def composite_score(self) -> float:
#         """Simple average mapped to 0–100."""
#         avg = (self.business_momentum + self.immediate_risk_severity + self.management_trust) / 3
#         return round((avg - 1) / 4 * 100, 1)


# @dataclass
# class NewsAgentOutput:
#     """Output of the News Agent (B.4)."""
#     return_outlook: int   # 1–5 (positive momentum)
#     risk_outlook: int     # 1–5 (potential downside; higher = more risk)
#     reason: str           # Korean ~100 chars
#     raw: dict = field(default_factory=dict)

#     @property
#     def net_score(self) -> float:
#         """Return outlook minus risk outlook, mapped 0–100."""
#         net = (self.return_outlook - self.risk_outlook + 4) / 8  # range [0,1]
#         return round(net * 100, 1)


# @dataclass
# class SectorAgentOutput:
#     """Output of the Sector Agent (B.5)."""
#     conviction_score: int    # 0–100
#     investment_thesis: str   # Korean ~200 words
#     raw: dict = field(default_factory=dict)


# @dataclass
# class MacroAgentOutput:
#     """Output of the Macro Agent (B.6)."""
#     market_trend: dict    # {"label": str, "score": 0-100}
#     risk: dict
#     economy: dict
#     rates: dict
#     inflation: dict
#     summary: str          # Korean ~200 chars
#     raw: dict = field(default_factory=dict)

#     @property
#     def composite_score(self) -> float:
#         metrics = [self.market_trend, self.risk, self.economy, self.rates, self.inflation]
#         scores = [m.get("score", 50) for m in metrics]
#         return round(sum(scores) / len(scores), 1)


# @dataclass
# class PMAgentOutput:
#     """Output of the Portfolio Manager Agent (B.7)."""
#     final_score: int   # 0–100
#     reason: str        # Korean 150–200 chars
#     raw: dict = field(default_factory=dict)


# # ──────────────────────────────────────────────────────────────────────────────
# # Abstract base agent
# # ──────────────────────────────────────────────────────────────────────────────

# class BaseAgent(ABC):
#     """All H-MAS agents inherit from this."""

#     def __init__(self, llm: LLMClient) -> None:
#         self.llm = llm

#     @property
#     @abstractmethod
#     def system_prompt(self) -> str:
#         ...

#     @abstractmethod
#     def run(self, **kwargs) -> Any:
#         ...

#     def _call(self, user_prompt: str) -> dict[str, Any]:
#         return self.llm.parse_json(self.system_prompt, user_prompt)

"""
H-MAS Base Classes
==================
Shared data structures, LLM client wrapper, and base Agent class
used by all agents in the hierarchy.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# LLM client (OpenAI-compatible; works with Anthropic, Azure, local, etc.)
# ──────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Thin wrapper around the OpenAI chat-completions API.
    Set OPENAI_API_KEY (or ANTHROPIC_API_KEY for claude via openai-compat).
    Override base_url for other providers.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY", "sk-placeholder"),
                base_url=base_url,
            )
        except ImportError:
            self._client = None  # stub mode for offline testing
        self.model = model
        self.temperature = temperature

    def chat(self, system: str, user: str) -> str:
        """Send a chat request; return the assistant text."""
        if self._client is None:
            raise RuntimeError("openai package not installed")
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def parse_json(self, system: str, user: str) -> dict[str, Any]:
        """chat() then parse JSON; raises ValueError on bad JSON."""
        raw = self.chat(system, user)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {raw!r}") from exc


# ──────────────────────────────────────────────────────────────────────────────
# Shared data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QualAgentOutput:
    """Output of the Qualitative (Qual) Agent (B.3)."""
    business_momentum: int      # 1–5
    immediate_risk_severity: int  # 1–5 (5 = low risk)
    management_trust: int       # 1–5
    insight: str                # Japanese professional briefing ~150 chars
    raw: dict = field(default_factory=dict)

    @property
    def composite_score(self) -> float:
        """Simple average mapped to 0–100."""
        avg = (self.business_momentum + self.immediate_risk_severity + self.management_trust) / 3
        return round((avg - 1) / 4 * 100, 1)


@dataclass
class NewsAgentOutput:
    """Output of the News Agent (B.4)."""
    return_outlook: int   # 1–5 (positive momentum)
    risk_outlook: int     # 1–5 (potential downside; higher = more risk)
    reason: str           # Japanese ~100 chars
    raw: dict = field(default_factory=dict)

    @property
    def net_score(self) -> float:
        """Return outlook minus risk outlook, mapped 0–100."""
        net = (self.return_outlook - self.risk_outlook + 4) / 8  # range [0,1]
        return round(net * 100, 1)


@dataclass
class SectorAgentOutput:
    """Output of the Sector Agent (B.5)."""
    conviction_score: int    # 0–100
    investment_thesis: str   # Japanese ~200 words
    raw: dict = field(default_factory=dict)


@dataclass
class MacroAgentOutput:
    """Output of the Macro Agent (B.6)."""
    market_trend: dict    # {"label": str, "score": 0-100}
    risk: dict
    economy: dict
    rates: dict
    inflation: dict
    summary: str          # Japanese ~200 chars
    raw: dict = field(default_factory=dict)

    @property
    def composite_score(self) -> float:
        metrics = [self.market_trend, self.risk, self.economy, self.rates, self.inflation]
        scores = [m.get("score", 50) for m in metrics]
        return round(sum(scores) / len(scores), 1)


@dataclass
class PMAgentOutput:
    """Output of the Portfolio Manager Agent (B.7)."""
    final_score: int   # 0–100
    reason: str        # Japanese 150–200 chars
    raw: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base agent
# ──────────────────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """All H-MAS agents inherit from this."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @abstractmethod
    def run(self, **kwargs) -> Any:
        ...

    def _call(self, user_prompt: str) -> dict[str, Any]:
        return self.llm.parse_json(self.system_prompt, user_prompt)