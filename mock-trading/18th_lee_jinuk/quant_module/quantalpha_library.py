"""
QuantAlpha Factor Library
==========================
Persistent JSON store of promoted alpha factors + utilities to convert
them into H-MAS TechnicalAgent weights and quant signals.

Mirrors quantaalpha/factors/library.py :: FactorLibraryManager
but is a standalone, Qlib-free implementation.

Schema (factor_library.json)
----------------------------
{
  "metadata": { "total_factors": N, "last_updated": "ISO8601" },
  "factors": {
    "factor_id": {
      "factor_name":       str,
      "factor_expression": str,
      "factor_description": str,
      "hypothesis":        str,
      "quality":           "high" | "medium" | "low",
      "backtest_metrics": {
        "IC": float, "ICIR": float, "Rank IC": float, "Rank ICIR": float
      },
      "round": int,
      "promoted_at": ISO8601
    }
  }
}
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_LIBRARY_PATH = Path(__file__).parent / "quantalpha_factor_library.json"


# ──────────────────────────────────────────────────────────────────────────────
# Stored factor entry
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StoredFactor:
    factor_id:          str
    factor_name:        str
    factor_expression:  str
    factor_description: str
    hypothesis:         str
    quality:            str          # high / medium / low
    backtest_metrics:   dict
    round_idx:          int = 0
    promoted_at:        str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def rank_ic(self) -> float:
        return float(self.backtest_metrics.get("Rank IC", 0.0))

    @property
    def ic(self) -> float:
        return float(self.backtest_metrics.get("IC", 0.0))


# ──────────────────────────────────────────────────────────────────────────────
# Library manager
# ──────────────────────────────────────────────────────────────────────────────

class FactorLibrary:
    """
    Load / save / query the promoted factor library.
    Thread-unsafe — single-process use only (fine for H-MAS single-run).
    """

    def __init__(self, path: Path | str = DEFAULT_LIBRARY_PATH) -> None:
        self._path = Path(path)
        self._factors: dict[str, StoredFactor] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for fid, fd in data.get("factors", {}).items():
                    self._factors[fid] = StoredFactor(
                        factor_id=fid,
                        factor_name=fd["factor_name"],
                        factor_expression=fd["factor_expression"],
                        factor_description=fd.get("factor_description", ""),
                        hypothesis=fd.get("hypothesis", ""),
                        quality=fd.get("quality", "low"),
                        backtest_metrics=fd.get("backtest_metrics", {}),
                        round_idx=fd.get("round", 0),
                        promoted_at=fd.get("promoted_at", ""),
                    )
                logger.info(f"Loaded {len(self._factors)} factors from {self._path}")
            except Exception as e:
                logger.warning(f"Could not load factor library: {e}")

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "total_factors": len(self._factors),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            },
            "factors": {
                fid: {
                    "factor_name":        f.factor_name,
                    "factor_expression":  f.factor_expression,
                    "factor_description": f.factor_description,
                    "hypothesis":         f.hypothesis,
                    "quality":            f.quality,
                    "backtest_metrics":   f.backtest_metrics,
                    "round":              f.round_idx,
                    "promoted_at":        f.promoted_at,
                }
                for fid, f in self._factors.items()
            },
        }
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Saved {len(self._factors)} factors → {self._path}")

    # ── mutation ──────────────────────────────────────────────────────────────

    def add_factor(
        self,
        factor_name:        str,
        factor_expression:  str,
        factor_description: str,
        hypothesis:         str,
        backtest_metrics:   dict,
        quality:            str,
        round_idx:          int = 0,
    ) -> StoredFactor:
        """Add or update a factor in the library. Returns the stored entry."""
        # Use name as ID (de-duplicate by name)
        fid = factor_name.lower().replace(" ", "_")
        sf = StoredFactor(
            factor_id=fid,
            factor_name=factor_name,
            factor_expression=factor_expression,
            factor_description=factor_description,
            hypothesis=hypothesis,
            quality=quality,
            backtest_metrics=backtest_metrics,
            round_idx=round_idx,
        )
        existing = self._factors.get(fid)
        if existing:
            # only replace if new Rank IC is strictly better
            if sf.rank_ic > existing.rank_ic:
                self._factors[fid] = sf
                logger.info(f"  Updated {fid}: RankIC {existing.rank_ic:.4f} → {sf.rank_ic:.4f}")
            else:
                logger.debug(f"  Skipped {fid}: existing RankIC {existing.rank_ic:.4f} ≥ new {sf.rank_ic:.4f}")
        else:
            self._factors[fid] = sf
            logger.info(f"  Added   {fid}: RankIC {sf.rank_ic:.4f}  quality={quality}")
        return self._factors[fid]

    # ── queries ───────────────────────────────────────────────────────────────

    def get_all(self, quality: Optional[str] = None) -> list[StoredFactor]:
        factors = list(self._factors.values())
        if quality:
            factors = [f for f in factors if f.quality == quality]
        return sorted(factors, key=lambda f: f.rank_ic, reverse=True)

    def get_top(self, n: int = 10, quality: Optional[str] = None) -> list[StoredFactor]:
        return self.get_all(quality)[:n]

    def __len__(self) -> int:
        return len(self._factors)

    def summary(self) -> str:
        total = len(self._factors)
        high   = sum(1 for f in self._factors.values() if f.quality == "high")
        medium = sum(1 for f in self._factors.values() if f.quality == "medium")
        low    = sum(1 for f in self._factors.values() if f.quality == "low")
        return (
            f"FactorLibrary: {total} factors "
            f"(high={high}, medium={medium}, low={low})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# H-MAS bridge: convert library → agent weights / signals
# ──────────────────────────────────────────────────────────────────────────────

def build_technical_weights(library: FactorLibrary) -> dict:
    """
    Derive IC-proportional sub-signal weights for TechnicalAgent.

    The TechnicalAgent uses ScoringWeights(macd, rsi, bollinger, roc5, roc21).
    We augment these with promoted alpha factors, allocating weight
    proportional to each factor's Rank IC.

    Returns a dict consumed by QuantAlphaTechnicalAgent:
    {
      "base_weight":   float,   # fraction kept for legacy MACD/RSI/BB signals
      "alpha_weight":  float,   # fraction allocated to promoted alpha factors
      "alpha_factors": [
        {"factor_expression": str, "weight": float, "rank_ic": float}, ...
      ]
    }
    """
    high_factors  = library.get_top(5, quality="high")
    mid_factors   = library.get_top(3, quality="medium")
    alpha_factors = high_factors + mid_factors

    if not alpha_factors:
        return {
            "base_weight": 1.0,
            "alpha_weight": 0.0,
            "alpha_factors": [],
        }

    # Weights proportional to Rank IC (only positive IC factors)
    rics = [max(f.rank_ic, 0.0) for f in alpha_factors]
    total_ric = sum(rics)
    if total_ric < 1e-8:
        return {"base_weight": 1.0, "alpha_weight": 0.0, "alpha_factors": []}

    # Total alpha weight grows with library quality but capped at 0.5
    alpha_weight = min(0.5, total_ric * 5.0)
    base_weight  = 1.0 - alpha_weight

    return {
        "base_weight":  round(base_weight, 4),
        "alpha_weight": round(alpha_weight, 4),
        "alpha_factors": [
            {
                "factor_name":       f.factor_name,
                "factor_expression": f.factor_expression,
                "weight":            round((ric / total_ric) * alpha_weight, 6),
                "rank_ic":           round(f.rank_ic, 6),
                "quality":           f.quality,
            }
            for f, ric in zip(alpha_factors, rics)
            if ric > 0
        ],
    }


def build_quant_signals(library: FactorLibrary,
                        price_data: dict[str, pd.DataFrame]) -> dict[str, float]:
    """
    Compute latest cross-sectional signal scores for each ticker
    from the top promoted factors.  Returns {ticker: score_0_100}.
    """
    try:
        import pandas as pd
        from quantalpha_factor_engine import FactorEngineMulti
    except ImportError:
        return {}
    top_factors = library.get_top(5)
    if not top_factors or not price_data:
        return {}

    multi = FactorEngineMulti(price_data)
    signals: list[pd.Series] = []

    for sf in top_factors:
        try:
            last = multi.eval_last(sf.factor_expression)
            # weight by Rank IC
            signals.append(last * sf.rank_ic)
        except Exception as e:
            logger.debug(f"Signal eval failed for {sf.factor_name}: {e}")

    if not signals:
        return {}

    combined = pd.concat(signals, axis=1).sum(axis=1)
    total_w  = sum(sf.rank_ic for sf in top_factors if sf.rank_ic > 0) or 1.0
    combined /= total_w

    # Normalise to 0-100
    lo, hi = combined.min(), combined.max()
    if hi - lo < 1e-8:
        return {t: 50.0 for t in combined.index}

    normed = (combined - lo) / (hi - lo) * 100.0
    return {str(t): round(float(v), 2) for t, v in normed.items()}
