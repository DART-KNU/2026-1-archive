"""
QuantAlpha Factor Engine
=========================
Evaluates Qlib-style factor expression strings over a pandas OHLCV DataFrame.

Supported ops (subset of Qlib Alpha158 / QuantaAlpha expression language):
  Time-series : TS_MEAN, TS_STD, TS_MAX, TS_MIN, TS_RANK, TS_CORR,
                TS_SUM, DELTA, DELAY, EMA, SMA
  Cross-section: RANK, ZSCORE, MEAN, STD
  Math         : LOG, ABS, SIGN, SQRT
  Arithmetic   : +, -, *, /
  Functions    : MAX(a,b), MIN(a,b)

Input DataFrame columns (case-sensitive): open, high, low, close, volume
Also derived: return_ (daily pct return), vwap

Usage
-----
    engine = FactorEngine(df_ohlcv)           # df indexed by date, columns=OHLCV
    result = engine.eval("RANK(TS_MEAN($return,20)/(TS_STD($return,20)+1e-8))")
    # result: pd.Series of factor values (cross-sectional rank, last available bar)
"""

from __future__ import annotations

import logging
import re
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────────────

def prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names and add derived features.
    Input columns (any case): Open, High, Low, Close, Volume
    Returns df with lowercase cols + return_, vwap added.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # rename common variants
    renames = {
        "adj close": "close", "adj_close": "close",
        "vol": "volume",
    }
    df.rename(columns=renames, inplace=True)

    if "close" not in df.columns:
        raise ValueError("DataFrame must have a 'close' column")

    # derived
    df["return_"] = df["close"].pct_change().fillna(0.0)
    if "high" in df.columns and "low" in df.columns:
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
    else:
        df["vwap"] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 1.0

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Expression evaluator
# ──────────────────────────────────────────────────────────────────────────────

class FactorEngine:
    """
    Evaluates one factor expression string over a prepared OHLCV DataFrame.

    The DataFrame (self._df) has rows = trading dates, columns = OHLCV + derived.
    eval() returns a pd.Series (index=dates, values=factor values for a SINGLE
    ticker). For multi-ticker cross-sectional ops, pass a dict of engines or
    use FactorEngineMulti.
    """

    # Maps $feature_name → df column
    _FEATURE_MAP = {
        "$open":     "open",
        "$high":     "high",
        "$low":      "low",
        "$close":    "close",
        "$volume":   "volume",
        "$return":   "return_",
        "$vwap":     "vwap",
        "$turnover": "volume",   # fallback approximation
    }

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = prepare_ohlcv(df)

    # ── public API ─────────────────────────────────────────────────────────

    def eval(self, expression: str) -> pd.Series:
        """
        Evaluate expression string.
        Returns pd.Series (dates as index) of factor values.
        """
        try:
            return self._eval_expr(expression.strip())
        except Exception as e:
            logger.warning(f"Factor eval failed for '{expression}': {e}")
            return pd.Series(np.nan, index=self._df.index, dtype=float)

    # ── tokeniser / parser ─────────────────────────────────────────────────

    def _eval_expr(self, expr: str) -> pd.Series:
        """Recursively evaluate an expression string."""
        expr = expr.strip()

        # Numeric literal (including scientific notation like 1e-8)
        try:
            val = float(expr)
            return pd.Series(val, index=self._df.index, dtype=float)
        except ValueError:
            pass

        # Unary minus:  -(...)  or  -FUNC(...)  or  -$feature
        if expr.startswith("-") and len(expr) > 1:
            inner = expr[1:].strip()
            # make sure it's genuinely unary (not a subtraction)
            try:
                return -self._eval_expr(inner)
            except Exception:
                pass

        # Parenthesised sub-expression: (expr)
        if expr.startswith("(") and expr.endswith(")"):
            # verify the parens are balanced around the whole expr
            depth = 0
            balanced = False
            for i, ch in enumerate(expr):
                if ch == "(": depth += 1
                elif ch == ")": depth -= 1
                if depth == 0:
                    if i == len(expr) - 1:
                        balanced = True
                    break
            if balanced:
                return self._eval_expr(expr[1:-1])

        # Base feature  $close, $return, …
        if expr in self._FEATURE_MAP:
            col = self._FEATURE_MAP[expr]
            if col in self._df.columns:
                return self._df[col].astype(float)
            return pd.Series(np.nan, index=self._df.index, dtype=float)

        # Arithmetic  (+, -, *, /) — lowest precedence, split on outermost op
        for series in self._try_binary_arith(expr):
            if series is not None:
                return series

        # Function calls  NAME(args...)
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\((.+)\)$", expr, re.DOTALL)
        if m:
            fname = m.group(1).upper()
            args_str = m.group(2)
            return self._call_func(fname, args_str)

        raise ValueError(f"Cannot parse expression: '{expr}'")

    def _split_top_level_args(self, args_str: str) -> list[str]:
        """Split args_str by commas at depth 0 (not inside parentheses)."""
        depth = 0
        parts: list[str] = []
        current: list[str] = []
        for ch in args_str:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current).strip())
        return parts

    def _try_binary_arith(self, expr: str):
        """
        Try to split expression on outermost binary operator.
        Returns [pd.Series] on success, [None] on failure.
        """
        # Operators in ascending precedence (we split on lowest first)
        for ops in [("+-",), ("*/",)]:
            result = self._split_on_op(expr, ops[0])
            if result is not None:
                return [result]
        return [None]

    def _split_on_op(self, expr: str, ops: str) -> pd.Series | None:
        """Find outermost binary op in `ops`, split and evaluate."""
        depth = 0
        # scan right-to-left so we get left-associativity
        i = len(expr) - 1
        while i >= 0:
            ch = expr[i]
            if ch == ")":
                depth += 1
            elif ch == "(":
                depth -= 1
            elif depth == 0 and ch in ops and i > 0:
                # make sure it's not a unary minus at start
                left_str  = expr[:i].strip()
                right_str = expr[i+1:].strip()
                if not left_str:
                    i -= 1
                    continue
                try:
                    left  = self._eval_expr(left_str)
                    right = self._eval_expr(right_str)
                    if ch == "+":   return left + right
                    elif ch == "-": return left - right
                    elif ch == "*": return left * right
                    elif ch == "/": return left / right.replace(0, np.nan)
                except Exception:
                    pass
            i -= 1
        return None

    # ── function dispatcher ────────────────────────────────────────────────

    def _call_func(self, fname: str, args_str: str) -> pd.Series:
        args = self._split_top_level_args(args_str)

        def _s(i: int) -> pd.Series:
            return self._eval_expr(args[i])

        def _n(i: int) -> int:
            return max(1, int(float(args[i])))

        # ── time-series ──────────────────────────────────────────────────
        if fname == "TS_MEAN":  return _s(0).rolling(_n(1), min_periods=1).mean()
        if fname == "TS_STD":   return _s(0).rolling(_n(1), min_periods=2).std().fillna(0.0)
        if fname == "TS_MAX":   return _s(0).rolling(_n(1), min_periods=1).max()
        if fname == "TS_MIN":   return _s(0).rolling(_n(1), min_periods=1).min()
        if fname == "TS_SUM":   return _s(0).rolling(_n(1), min_periods=1).sum()
        if fname == "TS_RANK":
            n = _n(1)
            return _s(0).rolling(n, min_periods=1).apply(
                lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=True)
        if fname == "TS_ZSCORE":
            n = _n(1); s = _s(0)
            mu  = s.rolling(n, min_periods=2).mean()
            std = s.rolling(n, min_periods=2).std().replace(0, np.nan)
            return (s - mu) / std
        if fname == "TS_CORR":
            n = _n(2)
            return _s(0).rolling(n, min_periods=5).corr(_s(1))
        if fname == "DELTA":
            return _s(0).diff(_n(1))
        if fname == "DELAY":
            return _s(0).shift(_n(1))
        if fname == "EMA":
            n = _n(1); alpha = 2.0 / (n + 1)
            return _s(0).ewm(alpha=alpha, adjust=False).mean()
        if fname == "SMA":
            return _s(0).rolling(_n(1), min_periods=1).mean()
        if fname == "Ref":  # Qlib Ref ≡ DELAY
            return _s(0).shift(_n(1))

        # ── cross-section (single ticker: these are per-date, trivially) ──
        # For single ticker, RANK/ZSCORE collapse to identity.
        # The multi-ticker engine overrides these with cross-sectional versions.
        if fname == "RANK":     return _s(0).rank(pct=True)   # time-series rank as proxy
        if fname == "ZSCORE":
            mu = _s(0).mean(); std = _s(0).std()
            return (_s(0) - mu) / (std if std else 1.0)
        if fname == "MEAN":     return pd.Series(_s(0).mean(), index=self._df.index)
        if fname == "STD":      return pd.Series(_s(0).std(),  index=self._df.index)

        # ── math ──────────────────────────────────────────────────────────
        if fname == "LOG":      return np.log(_s(0).clip(lower=1e-9))
        if fname == "ABS":      return _s(0).abs()
        if fname == "SIGN":     return np.sign(_s(0))
        if fname == "SQRT":     return np.sqrt(_s(0).clip(lower=0.0))

        # ── two-arg math ──────────────────────────────────────────────────
        if fname == "MAX":      return pd.concat([_s(0), _s(1)], axis=1).max(axis=1)
        if fname == "MIN":      return pd.concat([_s(0), _s(1)], axis=1).min(axis=1)

        # ── Qlib Alpha158 extras ──────────────────────────────────────────
        if fname == "Corr":   return _s(0).rolling(_n(2), min_periods=5).corr(_s(1))
        if fname == "Std":    return _s(0).rolling(_n(1), min_periods=2).std().fillna(0)
        if fname == "Mean":   return _s(0).rolling(_n(1), min_periods=1).mean()
        if fname == "Max":    return _s(0).rolling(_n(1), min_periods=1).max()
        if fname == "Min":    return _s(0).rolling(_n(1), min_periods=1).min()
        if fname == "Slope":  # linear slope via rolling regression beta
            n = _n(1); s = _s(0)
            x = np.arange(n, dtype=float)
            xm = x.mean()
            denom = ((x - xm) ** 2).sum()
            def _slope(w):
                ym = w.mean()
                return ((x[-len(w):] - xm) * (w - ym)).sum() / (denom + 1e-12)
            return s.rolling(n, min_periods=2).apply(_slope, raw=True)
        if fname == "Rsquare":
            n = _n(1); s = _s(0)
            x = np.arange(n, dtype=float)
            xm = x.mean()
            def _rsq(w):
                ym = w.mean()
                ss_res = ((w - ym) ** 2).sum()
                ss_tot = ss_res + 1e-12
                b = ((x[-len(w):] - xm) * (w - ym)).sum() / (((x[-len(w):] - xm)**2).sum() + 1e-12)
                ss_reg = b**2 * ((x[-len(w):] - xm)**2).sum()
                return ss_reg / (ss_tot + 1e-12)
            return s.rolling(n, min_periods=2).apply(_rsq, raw=True)

        raise ValueError(f"Unknown function: {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# Multi-ticker cross-sectional engine
# ──────────────────────────────────────────────────────────────────────────────

class FactorEngineMulti:
    """
    Evaluates a factor expression across multiple tickers, returning a
    DataFrame (index=dates, columns=tickers) with cross-sectional rankings.

    This is the production engine used by the backtest runner.
    """

    def __init__(self, price_data: dict[str, pd.DataFrame]) -> None:
        """
        price_data : {ticker: df_ohlcv}  each df_ohlcv has rows=dates
        """
        self._engines: dict[str, FactorEngine] = {
            t: FactorEngine(df) for t, df in price_data.items()
        }
        # common date index
        all_dates = sorted(set().union(*[set(e._df.index) for e in self._engines.values()]))
        self._dates = pd.DatetimeIndex(all_dates)

    def eval(self, expression: str) -> pd.DataFrame:
        """
        Evaluate expression for every ticker; return DataFrame.
        Cross-sectional RANK/ZSCORE are computed across all tickers per date.
        """
        per_ticker: dict[str, pd.Series] = {}
        for ticker, engine in self._engines.items():
            per_ticker[ticker] = engine.eval(expression)

        df = pd.DataFrame(per_ticker).reindex(self._dates)

        # Cross-sectional rank if expression starts with RANK(
        if expression.strip().startswith("RANK("):
            df = df.rank(axis=1, pct=True)
        elif expression.strip().startswith("ZSCORE("):
            df = df.subtract(df.mean(axis=1), axis=0).divide(
                df.std(axis=1).replace(0, np.nan), axis=0)

        return df

    def eval_last(self, expression: str) -> pd.Series:
        """Return most-recent cross-sectional factor values (latest date)."""
        df = self.eval(expression)
        return df.iloc[-1].dropna()
