"""
QuantAlpha R&D Loop for H-MAS
================================
Main orchestrator: runs N iterations of the hypothesis → factor → backtest
→ feedback cycle, writing promoted factors to the library.

Mirrors quantaalpha/pipeline/loop.py :: AlphaAgentLoop

Usage (no LLM — runs entirely on fallback factors, good for testing):
    python quantalpha_loop.py --n-iter 5 --no-llm

Usage (with Claude via Anthropic API):
    ANTHROPIC_API_KEY=sk-... python quantalpha_loop.py --n-iter 10

Usage (dry-run on synthetic data — no real KOSPI download):
    python quantalpha_loop.py --n-iter 3 --no-llm --synthetic
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from quantalpha_scenario   import KOSPIScenario
from quantalpha_hypothesis import (
    AlphaHypothesisGen, AlphaHypothesis2Factor, Trace,
    Hypothesis, _fallback_hypothesis, _fallback_factors,
)
from quantalpha_runner     import QuantAlphaRunner
from quantalpha_library    import FactorLibrary, build_technical_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quantalpha_loop")


# ──────────────────────────────────────────────────────────────────────────────
# LLM client builder
# ──────────────────────────────────────────────────────────────────────────────

def build_llm_fn(use_llm: bool = True):
    """
    Returns a callable (system_prompt, user_prompt) → str.
    Falls back to None-returning stub if no key or use_llm=False.
    """
    if not use_llm:
        logger.info("LLM disabled — using fallback factor sets")
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            def _anthropic_fn(system: str, user: str) -> str:
                msg = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1500,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return msg.content[0].text

            logger.info("Using Anthropic Claude for hypothesis + factor generation")
            return _anthropic_fn
        except Exception as e:
            logger.warning(f"Anthropic setup failed: {e} — using fallback")
            return None

    # Try OpenAI-compat
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        try:
            from openai import OpenAI
            oa = OpenAI(api_key=openai_key)

            def _openai_fn(system: str, user: str) -> str:
                resp = oa.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.7,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                )
                return resp.choices[0].message.content

            logger.info("Using OpenAI GPT-4o for hypothesis + factor generation")
            return _openai_fn
        except Exception as e:
            logger.warning(f"OpenAI setup failed: {e} — using fallback")

    logger.info("No LLM API key found — using built-in fallback factor sets")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generator (for tests without real market data)
# ──────────────────────────────────────────────────────────────────────────────

def make_synthetic_price_data(
    tickers: list[str],
    n_days:  int = 756,          # ~3 years
    seed:    int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)

    data: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(tickers):
        # Each ticker has slightly different drift + vol
        mu  = 0.0003 + rng.uniform(-0.0002, 0.0004)
        sig = 0.015  + rng.uniform(-0.005,  0.010)
        ret = rng.normal(mu, sig, n_days)

        close = 100_000 * np.exp(np.cumsum(ret))
        high  = close * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low   = close * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        open_ = close * (1 + rng.normal(0, 0.003, n_days))
        vol   = rng.integers(100_000, 2_000_000, n_days).astype(float)

        df = pd.DataFrame({
            "open":   open_,
            "high":   high,
            "low":    low,
            "close":  close,
            "volume": vol,
        }, index=dates)
        data[ticker] = df

    logger.info(f"Synthetic data: {len(tickers)} tickers × {n_days} days")
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Real KOSPI data fetch (yfinance)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_kospi_price_data(
    tickers:    list[str],
    start_date: str,
    end_date:   str,
) -> dict[str, pd.DataFrame]:
    """Download KOSPI OHLCV from yfinance. Falls back to synthetic on failure."""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — using synthetic data")
        return make_synthetic_price_data(tickers)

    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        yf_sym = f"{ticker}.KS"
        try:
            df = yf.download(yf_sym, start=start_date, end=end_date,
                             auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError("empty")
            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index)
            data[ticker] = df[["open","high","low","close","volume"]]
            logger.debug(f"  {ticker}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"  {ticker} download failed ({e}), using synthetic")
            data[ticker] = make_synthetic_price_data([ticker], seed=hash(ticker) % 9999)[ticker]

    n_real = sum(1 for t in data if len(data[t]) > 100)
    logger.info(f"KOSPI data: {n_real}/{len(tickers)} real, rest synthetic")
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Feedback builder
# ──────────────────────────────────────────────────────────────────────────────

def build_feedback(result, round_idx: int) -> dict:
    """
    Convert BacktestResult into a feedback dict for the Trace history.
    Mirrors QuantaAlpha AlphaAgentQlibFactorHypothesisExperiment2Feedback.
    """
    best = result.best_factor()
    promoted = result.promoted_factors(rank_ic_threshold=0.02)

    decision = len(promoted) > 0

    if decision:
        new_hypothesis = (
            f"Build on: {best.factor_name} (RankIC={best.rank_ic:.4f}). "
            f"Explore variations with different lookback windows or combined with volume."
        )
        evaluation = f"Positive — {len(promoted)}/{len(result.factor_metrics)} factors promoted."
    else:
        new_hypothesis = (
            "Prior hypothesis did not yield significant IC. "
            "Try contrarian signals or interaction terms with volume."
        )
        evaluation = f"Negative — ensemble RankIC={result.ensemble_rank_ic:.4f} below threshold."

    return {
        "rank_ic":           result.ensemble_rank_ic,
        "annualized_return": result.annualized_return,
        "max_drawdown":      result.max_drawdown,
        "decision":          decision,
        "n_promoted":        len(promoted),
        "evaluation":        evaluation,
        "new_hypothesis":    new_hypothesis,
        "round":             round_idx,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

class QuantAlphaLoop:
    """
    Full R&D loop: hypothesis → factor → backtest → feedback → library update.

    Parameters
    ----------
    scenario     : KOSPIScenario
    price_data   : {ticker: df_ohlcv}
    llm_fn       : callable (system, user) → str  or None for fallback
    library_path : path to JSON factor library
    pred_horizon : forward return horizon in days
    topk         : portfolio construction K
    promote_threshold : minimum Rank IC for promotion
    """

    def __init__(
        self,
        scenario:          KOSPIScenario,
        price_data:        dict[str, pd.DataFrame],
        llm_fn             = None,
        library_path:      str | Path = "quantalpha_factor_library.json",
        pred_horizon:      int   = 21,
        topk:              int   = 3,
        promote_threshold: float = 0.02,
    ) -> None:
        self.scenario  = scenario
        self.llm_fn    = llm_fn
        self.trace     = Trace(scen=scenario)
        self.library   = FactorLibrary(library_path)
        self.runner    = QuantAlphaRunner(
            price_data, pred_horizon=pred_horizon, topk=topk)
        self.promote_threshold = promote_threshold

        if llm_fn is not None:
            self.hyp_gen   = AlphaHypothesisGen(scenario, llm_fn)
            self.fac_gen   = AlphaHypothesis2Factor(scenario, llm_fn)
        else:
            self.hyp_gen   = None
            self.fac_gen   = None

        logger.info(
            f"QuantAlphaLoop ready | library={len(self.library)} factors | "
            f"llm={'on' if llm_fn else 'fallback'}"
        )

    # ── single iteration ─────────────────────────────────────────────────────

    def step(self, round_idx: int) -> dict:
        """Run one full R&D iteration. Returns summary dict."""
        logger.info(f"\n{'='*60}")
        logger.info(f"  Round {round_idx + 1}")
        logger.info(f"{'='*60}")

        t0 = time.time()

        # 1. Hypothesis
        logger.info("Stage 1: Hypothesis generation")
        if self.hyp_gen is not None:
            hypothesis = self.hyp_gen.gen(self.trace)
        else:
            hypothesis = Hypothesis(**_fallback_hypothesis(round_idx))
        logger.info(f"  → {hypothesis.hypothesis}")

        # 2. Factor construction
        logger.info("Stage 2: Factor construction")
        if self.fac_gen is not None:
            factors = self.fac_gen.convert(hypothesis, self.trace)
        else:
            factors = _fallback_factors(hypothesis)
        for f in factors:
            logger.info(f"  → {f.factor_name}: {f.factor_expression[:80]}")

        # 3. Backtest
        logger.info("Stage 3: Backtest")
        result = self.runner.run(hypothesis, factors)

        # 4. Feedback
        logger.info("Stage 4: Feedback")
        feedback = build_feedback(result, round_idx)
        logger.info(
            f"  Ensemble RankIC={result.ensemble_rank_ic:+.4f}  "
            f"AnnRet={result.annualized_return:+.4f}  "
            f"MaxDD={result.max_drawdown:.4f}  "
            f"decision={'PROMOTE' if feedback['decision'] else 'REJECT'}"
        )

        # 5. Library update
        promoted_this_round: list[str] = []
        for m in result.promoted_factors(self.promote_threshold):
            self.library.add_factor(
                factor_name=m.factor_name,
                factor_expression=m.expression,
                factor_description=next(
                    (f.factor_description for f in factors if f.factor_name == m.factor_name), ""
                ),
                hypothesis=str(hypothesis),
                backtest_metrics=m.as_dict(),
                quality=m.quality,
                round_idx=round_idx,
            )
            promoted_this_round.append(m.factor_name)

        if promoted_this_round:
            self.library.save()
            logger.info(f"  Promoted: {promoted_this_round}")

        # 6. Update trace
        self.trace.hist.append((hypothesis, factors, feedback))

        elapsed = time.time() - t0
        summary = {
            "round":             round_idx,
            "hypothesis":        str(hypothesis),
            "n_factors_tested":  len(factors),
            "n_promoted":        len(promoted_this_round),
            "promoted":          promoted_this_round,
            "ensemble_rank_ic":  result.ensemble_rank_ic,
            "annualized_return": result.annualized_return,
            "max_drawdown":      result.max_drawdown,
            "elapsed_s":         round(elapsed, 1),
        }
        return summary

    # ── full run ─────────────────────────────────────────────────────────────

    def run(self, n_iter: int = 5) -> list[dict]:
        """Run n_iter rounds. Returns list of per-round summaries."""
        summaries: list[dict] = []
        for i in range(n_iter):
            try:
                s = self.step(i)
                summaries.append(s)
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Round {i} failed: {e}", exc_info=True)
                summaries.append({"round": i, "error": str(e)})

        self._print_final_report(summaries)
        return summaries

    def _print_final_report(self, summaries: list[dict]) -> None:
        print("\n" + "=" * 65)
        print("  QuantAlpha R&D Loop — Final Report")
        print("=" * 65)
        for s in summaries:
            if "error" in s:
                print(f"  Round {s['round']+1:2d}  ERROR: {s['error']}")
            else:
                print(
                    f"  Round {s['round']+1:2d}  "
                    f"RankIC={s['ensemble_rank_ic']:+.4f}  "
                    f"Promoted={s['n_promoted']}  "
                    f"({s['elapsed_s']:.1f}s)"
                )

        print(f"\n{self.library.summary()}")
        weights = build_technical_weights(self.library)
        print(
            f"TechnicalAgent alpha_weight={weights['alpha_weight']:.3f}  "
            f"base_weight={weights['base_weight']:.3f}  "
            f"n_alpha_factors={len(weights['alpha_factors'])}"
        )
        print("=" * 65)

    def get_technical_weights(self) -> dict:
        """Return IC-weighted signal allocation for TechnicalAgent."""
        return build_technical_weights(self.library)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QuantAlpha R&D Loop for H-MAS")
    parser.add_argument("--n-iter",    type=int,   default=5,
                        help="Number of R&D iterations (default 5)")
    parser.add_argument("--no-llm",   action="store_true",
                        help="Use built-in fallback factors, no LLM calls")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic price data (no yfinance download)")
    parser.add_argument("--tickers",  nargs="+",
                        default=["005930","000660","005380","035420","373220",
                                 "105560","051910","207940","035720","006400"],
                        help="KOSPI ticker codes")
    parser.add_argument("--start",    default="2021-01-01")
    parser.add_argument("--end",      default="2024-12-31")
    parser.add_argument("--horizon",  type=int, default=21,
                        help="Forward return horizon (trading days, default 21)")
    parser.add_argument("--topk",     type=int, default=3,
                        help="TopK portfolio construction (default 3)")
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="Rank IC promotion threshold (default 0.02)")
    parser.add_argument("--library",  default="quantalpha_factor_library.json",
                        help="Path to factor library JSON")
    parser.add_argument("--out",      default="quantalpha_results.json",
                        help="Path to save run summary JSON")
    args = parser.parse_args()

    scenario = KOSPIScenario(
        universe=args.tickers,
        start_date=args.start,
        end_date=args.end,
        pred_horizon=args.horizon,
    )

    # Price data
    if args.synthetic:
        price_data = make_synthetic_price_data(args.tickers)
    else:
        price_data = fetch_kospi_price_data(args.tickers, args.start, args.end)

    # LLM
    llm_fn = build_llm_fn(use_llm=not args.no_llm)

    loop = QuantAlphaLoop(
        scenario=scenario,
        price_data=price_data,
        llm_fn=llm_fn,
        library_path=args.library,
        pred_horizon=args.horizon,
        topk=args.topk,
        promote_threshold=args.threshold,
    )

    summaries = loop.run(n_iter=args.n_iter)

    # Save summary
    out_path = Path(args.out)
    out_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Summary saved → {out_path}")


if __name__ == "__main__":
    main()
