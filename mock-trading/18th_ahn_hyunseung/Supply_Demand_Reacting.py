from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from pykrx import stock

try:
    from pykrx.website.comm import build_krx_session, set_session
except ImportError:
    build_krx_session = None
    set_session = None


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "codex_swing_0323_outputs"
DAILY_CACHE_DIR = OUT_DIR / "daily_snapshots"
KOSPI_INDEX_TICKER = "1001"
CALENDAR_PROXY_TICKERS = ("005930", "000660", "069500")
DEFAULT_COMPETITION_UNIVERSE_CANDIDATES = [
    BASE_DIR / "RFM-11-sector.csv",
    Path.home() / "Documents" / "RFM-11-sector.csv",
]
DEFAULT_MULTI_NAME_SECTOR_CODES = ("IT", "In")

INITIAL_CAPITAL = 100_000_000
BUY_FEE = 0.00015
SELL_FEE = 0.00015
SELL_TAX = 0.0018
ENTRY_SLIPPAGE_RATE = 0.0005
EXIT_SLIPPAGE_RATE = 0.0005
GAP_EXIT_EXTRA_SLIPPAGE = 0.0010
MAX_ENTRY_TURNOVER_PARTICIPATION = 0.03

COMPETITION_DEFAULT_SECTOR_CAPS = {
    "En": 0.10,
    "Ma": 0.10,
    "In": 0.438,
    "CD": 0.138,
    "CS": 0.10,
    "He": 0.153,
    "Fi": 0.176,
    "IT": 0.868,
    "Co": 0.10,
    "Ut": 0.10,
    "Re": 0.10,
}

RECIPE_LIBRARY = {
    "base_big_short_mr20": [
        ("big_rank", 0.40),
        ("short_rank", 0.35),
        ("mr20_rank", 0.25),
    ],
    "base_big_short_mr5_mr20": [
        ("big_rank", 0.25),
        ("short_rank", 0.30),
        ("mr5_rank", 0.20),
        ("mr20_rank", 0.25),
    ],
    "base_big_short_turnover": [
        ("big_rank", 0.25),
        ("short_rank", 0.25),
        ("turnover_surge_rank", 0.20),
        ("mr20_rank", 0.30),
    ],
    "squeeze_reversal": [
        ("short_rank", 0.25),
        ("mr5_rank", 0.20),
        ("mr20_rank", 0.20),
        ("turnover_surge_rank", 0.15),
        ("dd60_rank", 0.20),
    ],
    "deep_pullback": [
        ("dd60_rank", 0.30),
        ("mr10_rank", 0.18),
        ("short10_rank", 0.18),
        ("low_vol_rank", 0.12),
        ("big_rank", 0.22),
    ],
    "sector_rebound": [
        ("sector_alpha20_rank", 0.20),
        ("sec_mr20_rank", 0.20),
        ("mr5_rank", 0.15),
        ("short_rank", 0.20),
        ("big_rank", 0.25),
    ],
    "smallcap_shock": [
        ("small_rank", 0.22),
        ("short_rank", 0.18),
        ("mr10_rank", 0.14),
        ("turnover_surge_rank", 0.16),
        ("dd60_rank", 0.16),
        ("short10_rank", 0.14),
    ],
    "liquidity_relaunch": [
        ("liquidity_rank", 0.16),
        ("turnover_surge_rank", 0.18),
        ("mr20_rank", 0.18),
        ("short_rank", 0.18),
        ("big_rank", 0.15),
        ("breakout20_rank", 0.15),
    ],
}

DEFAULT_RECIPE = "base_big_short_mr5_mr20"
SELECTION_FOLDS = [
    ("fold_2025_early", pd.Timestamp("2025-01-02"), pd.Timestamp("2025-04-30")),
    ("fold_2025_mid", pd.Timestamp("2025-05-01"), pd.Timestamp("2025-08-31")),
    ("fold_2025_late", pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31")),
]


@dataclass
class StrategyConfig:
    recipe: str = DEFAULT_RECIPE
    min_price: int = 1000
    min_turnover_krw: float = 5_000_000_000.0
    min_component_rank: float = 0.45
    top_n: int = 7
    max_positions: int = 3
    max_hold_days: int = 7
    stop_loss: float = -0.08
    take_profit: float = 0.18

    train_start: str = "2025-01-02"
    train_end: str = "2025-12-31"
    holdout_start: str = "2026-01-02"
    holdout_end: Optional[str] = None
    warmup_days: int = 120
    top_k: int = 20
    strategy_capital_fraction: float = 0.30
    max_weight_per_name: Optional[float] = 0.10
    max_names_per_sector: Optional[int] = None


@dataclass
class Position:
    code: str
    name: str
    sector: str
    sector_code: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    cost: float
    hold_days: int = 0
    score: float = 0.0


DEFAULT_CONFIG = StrategyConfig()


def load_optional_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_code_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    codes = []
    for token in str(raw).split(","):
        code = normalize_ticker_code(token)
        if code:
            codes.append(code)
    return sorted(set(codes))


def parse_token_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    tokens = []
    for token in str(raw).split(","):
        item = token.strip()
        if item:
            tokens.append(item)
    return sorted(set(tokens))


def discover_default_competition_universe_path() -> Optional[Path]:
    for path in DEFAULT_COMPETITION_UNIVERSE_CANDIDATES:
        if path.exists():
            return path
    documents_dir = Path.home() / "Documents"
    if documents_dir.exists():
        matches = sorted(documents_dir.rglob("RFM-11-sector.csv"))
        if matches:
            return matches[0]
    return None


def list_cached_snapshot_dates() -> List[pd.Timestamp]:
    if not DAILY_CACHE_DIR.exists():
        return []
    dates: List[pd.Timestamp] = []
    for path in DAILY_CACHE_DIR.glob("*.csv"):
        try:
            dates.append(pd.Timestamp(path.stem))
        except Exception:
            continue
    return sorted(set(dates))


def prompt_for_krx_credentials(krx_id: Optional[str], krx_pw: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if krx_id and krx_pw:
        return krx_id, krx_pw
    if not sys.stdin.isatty():
        return krx_id, krx_pw

    try:
        entered_id = (krx_id or input("KRX login id (press Enter to skip): ").strip() or None)
        if not entered_id:
            return None, None
        entered_pw = krx_pw or getpass.getpass("KRX login password: ").strip() or None
        if not entered_pw:
            return None, None
        return entered_id, entered_pw
    except (EOFError, KeyboardInterrupt):
        print()
        return None, None


def parse_weight_mapping(raw: Optional[str]) -> Dict[str, float]:
    if not raw:
        return {}

    path = Path(str(raw))
    if path.exists():
        text = path.read_text(encoding="utf-8")
        loaded = json.loads(text)
        return {str(k): float(v) for k, v in loaded.items()}

    mapping: Dict[str, float] = {}
    for token in str(raw).split(","):
        item = token.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid mapping item: {item}")
        key, value = item.split(":", 1)
        mapping[key.strip()] = float(value.strip())
    return mapping


def normalize_ticker_code(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    match = re.search(r"(\d{6})", text)
    return match.group(1) if match else None


def load_competition_universe(csv_path: Path) -> pd.DataFrame:
    read_errors = []
    for encoding in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            raw = pd.read_csv(csv_path, encoding=encoding)
            break
        except Exception as exc:
            read_errors.append(f"{encoding}: {exc}")
    else:
        joined = " | ".join(read_errors)
        raise RuntimeError(f"Could not read competition universe CSV: {csv_path} / {joined}")

    required = {"섹터코드", "섹터명", "종목코드", "종목명"}
    missing = required.difference(raw.columns)
    if missing:
        raise RuntimeError(f"Competition universe CSV missing columns: {sorted(missing)}")

    out = raw.rename(
        columns={
            "섹터코드": "sector_code",
            "섹터명": "sector_name",
            "종목코드": "raw_code",
            "종목명": "competition_name",
        }
    ).copy()
    out["code"] = out["raw_code"].map(normalize_ticker_code)
    dropped = int(out["code"].isna().sum())
    out = out.dropna(subset=["code"]).copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["sector_code"] = out["sector_code"].astype(str).str.strip()
    out["sector_name"] = out["sector_name"].astype(str).str.strip()
    out["competition_name"] = out["competition_name"].astype(str).str.strip()
    out = out.drop_duplicates("code").reset_index(drop=True)
    print(
        f"[INFO] competition universe loaded: {len(out)} tradable codes "
        f"(dropped {dropped} rows with non-6-digit identifiers)"
    )
    return out[["code", "competition_name", "sector_code", "sector_name"]]


def apply_competition_universe(
    panel: pd.DataFrame,
    competition_universe: Optional[pd.DataFrame],
    excluded_codes: Iterable[str],
) -> pd.DataFrame:
    work = panel.copy()
    excluded = {code for code in excluded_codes if code}

    if competition_universe is not None and not competition_universe.empty:
        work = work.merge(competition_universe, on="code", how="inner", suffixes=("", "_competition"))
        work["name"] = work["competition_name"].fillna(work["name"])
        competition_sector_code_col = "sector_code_competition" if "sector_code_competition" in work.columns else "sector_code"
        work["sector_code"] = work[competition_sector_code_col].fillna("")
        work["sector"] = work["sector_name"].fillna(work["sector"])
        work = work.drop(columns=["competition_name", "sector_name"])
    else:
        work["sector_code"] = ""

    if excluded:
        before = len(work)
        work = work.loc[~work["code"].isin(excluded)].copy()
        print(f"[INFO] excluded {before - len(work)} rows for codes: {', '.join(sorted(excluded))}")

    work["sector_code"] = work["sector_code"].fillna("").astype(str)
    work = work.sort_values(["code", "date"]).reset_index(drop=True)
    return work


def resolve_sector_caps(competition_mode: bool, overrides: Dict[str, float]) -> Dict[str, float]:
    caps = dict(COMPETITION_DEFAULT_SECTOR_CAPS if competition_mode else {})
    caps.update({str(k): float(v) for k, v in overrides.items()})
    return caps


def get_sector_cap(params: dict, sector_code: str, sector_name: str) -> float:
    caps = params.get("sector_caps", {}) or {}
    if sector_code and sector_code in caps:
        return float(caps[sector_code])
    if sector_name and sector_name in caps:
        return float(caps[sector_name])
    return float(params.get("default_sector_cap", 1.0))


def resolve_sector_name_limit(params: dict, sector_code: str, sector_name: str) -> int:
    multi_name_allowed = {
        str(code).strip().casefold()
        for code in params.get("multi_name_sector_codes", DEFAULT_MULTI_NAME_SECTOR_CODES)
    }
    if str(sector_code).strip().casefold() in multi_name_allowed or str(sector_name).strip().casefold() in multi_name_allowed:
        return int(params.get("max_names_per_sector", 9999))
    return int(params.get("default_non_exempt_sector_name_limit", 9999))


def parse_args() -> argparse.Namespace:
    default_competition_csv = discover_default_competition_universe_path()
    parser = argparse.ArgumentParser(description="Standalone pykrx-only runner for the current optimal codex swing strategy")
    parser.add_argument("--recipe", default=DEFAULT_CONFIG.recipe, choices=sorted(RECIPE_LIBRARY.keys()))
    parser.add_argument("--min-price", type=int, default=DEFAULT_CONFIG.min_price)
    parser.add_argument("--min-turnover-krw", type=float, default=DEFAULT_CONFIG.min_turnover_krw)
    parser.add_argument("--min-component-rank", type=float, default=DEFAULT_CONFIG.min_component_rank)
    parser.add_argument("--top-n", type=int, default=DEFAULT_CONFIG.top_n)
    parser.add_argument("--max-positions", type=int, default=DEFAULT_CONFIG.max_positions)
    parser.add_argument("--max-hold-days", type=int, default=DEFAULT_CONFIG.max_hold_days)
    parser.add_argument("--stop-loss", type=float, default=DEFAULT_CONFIG.stop_loss)
    parser.add_argument("--take-profit", type=float, default=DEFAULT_CONFIG.take_profit)
    parser.add_argument("--train-start", default=DEFAULT_CONFIG.train_start)
    parser.add_argument("--train-end", default=DEFAULT_CONFIG.train_end)
    parser.add_argument("--holdout-start", default=DEFAULT_CONFIG.holdout_start)
    parser.add_argument("--holdout-end", default=DEFAULT_CONFIG.holdout_end, help="holdout end date, defaults to latest business day")
    parser.add_argument("--warmup-days", type=int, default=DEFAULT_CONFIG.warmup_days)
    parser.add_argument("--top-k", type=int, default=DEFAULT_CONFIG.top_k)
    parser.add_argument("--strategy-capital-fraction", type=float, default=DEFAULT_CONFIG.strategy_capital_fraction)
    parser.add_argument("--screen-only", action="store_true", help="build the latest watchlist only")
    parser.add_argument("--refresh-panel", action="store_true", help="ignore cached daily snapshots and redownload them")
    parser.add_argument("--screen-date", default=None, help="override latest business day, format YYYYMMDD")
    parser.add_argument("--run-search", action="store_true", help="search pykrx-only alpha strategies")
    parser.add_argument("--search-samples", type=int, default=1200, help="number of parameter combinations to sample")
    parser.add_argument("--search-fixed-max-positions", type=int, default=None, help="when set, search only this max_positions value")
    parser.add_argument("--competition-universe-csv", default=str(default_competition_csv) if default_competition_csv else None, help="competition tradable universe CSV path")
    parser.add_argument("--exclude-codes", default="005930,000660", help="comma-separated 6-digit codes to exclude")
    parser.add_argument(
        "--max-weight-per-name",
        type=float,
        default=DEFAULT_CONFIG.max_weight_per_name,
        help="per-name max portfolio weight",
    )
    parser.add_argument("--max-names-per-sector", type=int, default=None, help="max simultaneous names per sector")
    parser.add_argument("--multi-name-sector-codes", default="IT,In", help="sector codes that may hold multiple names simultaneously")
    parser.add_argument("--default-non-exempt-sector-name-limit", type=int, default=1, help="simultaneous name limit for sectors not in multi-name-sector-codes")
    parser.add_argument("--sector-cap-overrides", default=None, help="comma-separated sector cap overrides like IT:0.55,He:0.10")
    parser.add_argument("--krx-id", default=None, help="optional KRX login id")
    parser.add_argument("--krx-pw", default=None, help="optional KRX login password")
    parser.add_argument("--no-cache", action="store_true", help="ignore cached daily snapshots and fetch all data live")
    parser.add_argument("--skip-krx-login-prompt", action="store_true", help="do not interactively prompt for KRX login credentials")
    return parser.parse_args()


def build_params(args: argparse.Namespace) -> dict:
    if args.max_positions > args.top_n:
        raise ValueError("max_positions cannot exceed top_n")
    if not (0.0 < float(args.strategy_capital_fraction) <= 1.0):
        raise ValueError("strategy_capital_fraction must be in (0, 1].")

    competition_csv = args.competition_universe_csv or discover_default_competition_universe_path()
    competition_mode = bool(competition_csv)
    sector_caps = resolve_sector_caps(competition_mode, parse_weight_mapping(args.sector_cap_overrides))
    max_weight_per_name = (
        float(args.max_weight_per_name)
        if args.max_weight_per_name is not None
        else (
            float(DEFAULT_CONFIG.max_weight_per_name)
            if DEFAULT_CONFIG.max_weight_per_name is not None
            else (0.15 if competition_mode else 1.0)
        )
    )
    max_names_per_sector = int(args.max_names_per_sector) if args.max_names_per_sector is not None else (int(args.max_positions) if competition_mode else 9999)
    multi_name_sector_codes = parse_token_list(args.multi_name_sector_codes)

    return {
        "recipe": str(args.recipe),
        "min_price": int(args.min_price),
        "min_turnover_krw": float(args.min_turnover_krw),
        "min_component_rank": float(args.min_component_rank),
        "top_n": int(args.top_n),
        "max_positions": int(args.max_positions),
        "max_hold_days": int(args.max_hold_days),
        "stop_loss": float(args.stop_loss),
        "take_profit": float(args.take_profit),
        "strategy_capital_fraction": float(args.strategy_capital_fraction),
        "competition_mode": competition_mode,
        "competition_universe_csv": str(competition_csv) if competition_csv else None,
        "excluded_codes": parse_code_list(args.exclude_codes),
        "max_weight_per_name": max_weight_per_name,
        "max_names_per_sector": max_names_per_sector,
        "multi_name_sector_codes": multi_name_sector_codes,
        "default_non_exempt_sector_name_limit": int(args.default_non_exempt_sector_name_limit),
        "sector_caps": sector_caps,
        "default_sector_cap": 1.0,
    }


def build_periods(args: argparse.Namespace, latest_business_day: pd.Timestamp) -> dict[str, pd.Timestamp]:
    holdout_end_raw = args.holdout_end
    holdout_end = (
        latest_business_day
        if holdout_end_raw in {None, "", "latest", "LATEST"}
        else min(pd.Timestamp(holdout_end_raw), latest_business_day)
    )
    return {
        "train_start": pd.Timestamp(args.train_start),
        "train_end": pd.Timestamp(args.train_end),
        "holdout_start": pd.Timestamp(args.holdout_start),
        "holdout_end": holdout_end,
    }


def initialize_pykrx_session(krx_id: Optional[str], krx_pw: Optional[str], allow_prompt: bool = True) -> bool:
    if build_krx_session is None or set_session is None:
        print("[WARN] pykrx login helper is unavailable in this environment.")
        return False
    if allow_prompt:
        krx_id, krx_pw = prompt_for_krx_credentials(krx_id, krx_pw)
    if not (krx_id and krx_pw):
        print("[WARN] KRX login not provided. Continuing without login.")
        return False

    try:
        session = build_krx_session(login_id=krx_id, login_pw=krx_pw)
        set_session(session)
        ok = session is not None
        print(f"[INFO] pykrx login session initialized = {ok}")
        return ok
    except Exception as exc:
        print(f"[WARN] pykrx login failed: {exc}")
        return False


def yyyymmdd(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")


def get_proxy_ohlcv(date_from: pd.Timestamp, date_to: pd.Timestamp) -> pd.DataFrame:
    for ticker in CALENDAR_PROXY_TICKERS:
        try:
            df = stock.get_market_ohlcv_by_date(yyyymmdd(date_from), yyyymmdd(date_to), ticker)
            if not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def resolve_latest_business_day(screen_date: Optional[str] = None) -> pd.Timestamp:
    if screen_date:
        return pd.Timestamp(screen_date)
    today = pd.Timestamp.today().normalize()
    start = today - pd.Timedelta(days=14)
    proxy = get_proxy_ohlcv(start, today)
    if not proxy.empty:
        return pd.Timestamp(proxy.index[-1])
    raise RuntimeError("Could not resolve latest business day from pykrx proxy ticker data.")


def get_trading_calendar(date_from: pd.Timestamp, date_to: pd.Timestamp) -> List[pd.Timestamp]:
    proxy = get_proxy_ohlcv(date_from, date_to)
    if not proxy.empty:
        return [pd.Timestamp(x) for x in proxy.index]
    raise RuntimeError(f"No trading calendar data from pykrx proxy tickers for {date_from.date()} ~ {date_to.date()}.")


def get_sector_frame(date_str: str) -> pd.DataFrame:
    frames = []
    for market in ["KOSPI", "KOSDAQ"]:
        try:
            df = stock.get_market_sector_classifications(date_str, market=market).reset_index()
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] sector classification fetch failed for {market} {date_str}: {exc}")
    if not frames:
        raise RuntimeError(f"Could not load sector classifications for {date_str}.")

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"종목코드": "code", "종목명": "name", "업종명": "sector"})
    out["code"] = out["code"].astype(str).str.zfill(6)
    return out[["code", "name", "sector"]].drop_duplicates("code")


def get_shorting_frame(date_str: str) -> pd.DataFrame:
    frames = []
    for market in ["KOSPI", "KOSDAQ"]:
        try:
            df = stock.get_shorting_value_by_ticker(date_str, market=market).reset_index()
            frames.append(df)
        except Exception as exc:
            print(f"[WARN] shorting value fetch failed for {market} {date_str}: {exc}")
    if not frames:
        return pd.DataFrame(columns=["code", "short_sell_value_krw"])

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"티커": "code", "공매도": "short_sell_value_krw"})
    out["code"] = out["code"].astype(str).str.zfill(6)
    return out[["code", "short_sell_value_krw"]]


def fetch_daily_snapshot(date: pd.Timestamp, refresh: bool = False) -> pd.DataFrame:
    DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DAILY_CACHE_DIR / f"{yyyymmdd(date)}.csv"
    if cache_path.exists() and not refresh:
        snap = pd.read_csv(cache_path, parse_dates=["date"], dtype={"code": str})
        snap["code"] = snap["code"].astype(str).str.zfill(6)
        if "sector_code" not in snap.columns:
            snap["sector_code"] = ""
        return snap

    date_str = yyyymmdd(date)
    try:
        ohlcv = stock.get_market_ohlcv_by_ticker(date_str, market="ALL").reset_index()
    except Exception as exc:
        raise RuntimeError(
            "Live pykrx daily snapshot fetch failed. "
            "Run with --krx-id/--krx-pw or use the interactive login prompt. "
            f"date={date_str} / error={exc}"
        ) from exc
    if ohlcv.empty:
        raise RuntimeError(f"OHLCV snapshot is empty for {date_str}.")

    ohlcv = ohlcv.rename(
        columns={
            "티커": "code",
            "시가": "adj_open",
            "고가": "adj_high",
            "저가": "adj_low",
            "종가": "adj_close",
            "거래량": "volume",
            "거래대금": "turnover_krw",
            "시가총액": "market_cap_krw",
        }
    )
    ohlcv["code"] = ohlcv["code"].astype(str).str.zfill(6)

    sector = get_sector_frame(date_str)
    shorting = get_shorting_frame(date_str)

    snap = ohlcv.merge(sector, on="code", how="left").merge(shorting, on="code", how="left")
    snap["short_sell_value_krw"] = pd.to_numeric(snap["short_sell_value_krw"], errors="coerce").fillna(0.0)
    snap["date"] = pd.Timestamp(date)

    keep_cols = [
        "date",
        "code",
        "name",
        "sector",
        "sector_code",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "volume",
        "turnover_krw",
        "market_cap_krw",
        "short_sell_value_krw",
    ]
    snap["sector_code"] = ""
    snap = snap[keep_cols].sort_values("code").reset_index(drop=True)
    snap.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return snap


def build_panel(date_from: pd.Timestamp, date_to: pd.Timestamp, refresh: bool = False) -> pd.DataFrame:
    trading_dates = get_trading_calendar(date_from, date_to)
    frames: List[pd.DataFrame] = []
    total = len(trading_dates)
    for idx, date in enumerate(trading_dates, start=1):
        frames.append(fetch_daily_snapshot(date, refresh=refresh))
        if idx == 1 or idx % 20 == 0 or idx == total:
            print(f"[panel] loaded {idx}/{total} trading days")

    panel = pd.concat(frames, ignore_index=True)
    numeric_cols = [c for c in panel.columns if c not in {"date", "code", "name", "sector", "sector_code"}]
    for col in numeric_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
    return panel


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "sector_code" not in work.columns:
        work["sector_code"] = ""
    g = work.groupby("code", group_keys=False)

    work["ret1"] = g["adj_close"].pct_change(fill_method=None)
    work["ret10"] = g["adj_close"].pct_change(10, fill_method=None)
    work["ret5"] = g["adj_close"].pct_change(5, fill_method=None)
    work["ret20"] = g["adj_close"].pct_change(20, fill_method=None)
    work["ret60"] = g["adj_close"].pct_change(60, fill_method=None)
    work["turn_ma20"] = g["turnover_krw"].transform(lambda s: s.rolling(20).mean())
    work["turn_ma5"] = g["turnover_krw"].transform(lambda s: s.rolling(5).mean())
    work["turnover_surge"] = work["turn_ma5"] / work["turn_ma20"] - 1.0
    work["short5_value"] = g["short_sell_value_krw"].transform(lambda s: s.rolling(5).sum())
    work["short10_value"] = g["short_sell_value_krw"].transform(lambda s: s.rolling(10).sum())
    denom = (work["turn_ma20"] * 5.0).replace(0, np.nan)
    work["short5_to_turn"] = work["short5_value"] / denom
    work["short10_to_turn"] = work["short10_value"] / ((work["turn_ma20"] * 10.0).replace(0, np.nan))

    work["ma20"] = g["adj_close"].transform(lambda s: s.rolling(20).mean())
    work["price_vs_ma20"] = work["adj_close"] / work["ma20"] - 1.0
    work["high20_prev"] = g["adj_high"].transform(lambda s: s.rolling(20).max().shift(1))
    work["high60"] = g["adj_high"].transform(lambda s: s.rolling(60).max())
    work["breakout20"] = work["adj_close"] / work["high20_prev"] - 1.0
    work["dd60"] = work["adj_close"] / work["high60"] - 1.0
    work["vol20"] = g["ret1"].transform(lambda s: s.rolling(20).std()) * np.sqrt(252)

    work["sector_bucket"] = np.where(work["sector_code"].astype(str).str.len() > 0, work["sector_code"], work["sector"])
    sector_daily = (
        work.groupby(["date", "sector_bucket"], observed=False)
        .agg(
            sec_ret5=("ret5", "mean"),
            sec_ret10=("ret10", "mean"),
            sec_ret20=("ret20", "mean"),
        )
        .reset_index()
    )
    market_daily = work.groupby("date").agg(mkt_ret20=("ret20", "mean")).reset_index()
    work = work.merge(sector_daily, on=["date", "sector_bucket"], how="left")
    work = work.merge(market_daily, on="date", how="left")
    work["sector_alpha20"] = work["sec_ret20"] - work["mkt_ret20"]
    g = work.groupby("code", group_keys=False)

    signal_cols = [
        "adj_close",
        "turn_ma20",
        "ret5",
        "ret10",
        "ret20",
        "ret60",
        "market_cap_krw",
        "short5_to_turn",
        "short10_to_turn",
        "turnover_surge",
        "price_vs_ma20",
        "breakout20",
        "dd60",
        "vol20",
        "sec_ret10",
        "sec_ret20",
        "sector_alpha20",
    ]
    for col in signal_cols:
        work[f"{col}_sig"] = g[col].shift(1)

    rank_specs = [
        ("mr5_rank", "ret5_sig", "ret5", False),
        ("mr10_rank", "ret10_sig", "ret10", False),
        ("mr20_rank", "ret20_sig", "ret20", False),
        ("mr60_rank", "ret60_sig", "ret60", False),
        ("big_rank", "market_cap_krw_sig", "market_cap_krw", True),
        ("small_rank", "market_cap_krw_sig", "market_cap_krw", False),
        ("short_rank", "short5_to_turn_sig", "short5_to_turn", True),
        ("short10_rank", "short10_to_turn_sig", "short10_to_turn", True),
        ("turnover_surge_rank", "turnover_surge_sig", "turnover_surge", True),
        ("liquidity_rank", "turn_ma20_sig", "turn_ma20", True),
        ("price_vs_ma20_rank", "price_vs_ma20_sig", "price_vs_ma20", True),
        ("breakout20_rank", "breakout20_sig", "breakout20", True),
        ("dd60_rank", "dd60_sig", "dd60", False),
        ("low_vol_rank", "vol20_sig", "vol20", False),
        ("sec_mr20_rank", "sec_ret20_sig", "sec_ret20", False),
        ("sector_alpha20_rank", "sector_alpha20_sig", "sector_alpha20", True),
    ]
    for rank_name, trade_col, live_col, higher in rank_specs:
        work[rank_name] = work.groupby("date")[trade_col].rank(pct=True, ascending=higher, method="average")
        work[f"live_{rank_name}"] = work.groupby("date")[live_col].rank(pct=True, ascending=higher, method="average")
    return work


def build_period_context(df: pd.DataFrame, date_from: pd.Timestamp, date_to: pd.Timestamp) -> dict:
    work = df.loc[(df["date"] >= date_from) & (df["date"] <= date_to)].copy()
    row_cols = ["date", "code", "name", "sector", "sector_code", "adj_open", "adj_high", "adj_low", "adj_close"]
    row_map = {(row["code"], pd.Timestamp(row["date"])): row for _, row in work[row_cols].iterrows()}
    trading_dates = [pd.Timestamp(x) for x in work["date"].drop_duplicates().sort_values().tolist()]
    return {"df": work, "row_map": row_map, "trading_dates": trading_dates}


def build_candidate_map(df: pd.DataFrame, params: dict) -> Dict[pd.Timestamp, pd.DataFrame]:
    work = df.copy()
    recipe = RECIPE_LIBRARY[params["recipe"]]
    cond = (
        (work["adj_close_sig"] >= params["min_price"])
        & (work["turn_ma20_sig"] >= params["min_turnover_krw"])
    )
    for factor, _ in recipe:
        cond &= work[factor] >= params["min_component_rank"]

    work = work.loc[cond].copy()
    if work.empty:
        return {}

    work["score"] = 0.0
    for factor, weight in recipe:
        work["score"] += work[factor] * weight

    keep_cols = [
        "date",
        "code",
        "name",
        "sector",
        "sector_code",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "score",
        "turn_ma20_sig",
        "ret5_sig",
        "ret20_sig",
        "market_cap_krw_sig",
        "short5_to_turn_sig",
        "short10_to_turn_sig",
        "turnover_surge_sig",
        "dd60_sig",
        "breakout20_sig",
        "sector_alpha20_sig",
        "sec_ret20_sig",
    ]
    work = work[keep_cols].sort_values(["date", "score"], ascending=[True, False])

    daily: Dict[pd.Timestamp, pd.DataFrame] = {}
    for date, group in work.groupby("date"):
        pool = group.copy()
        if params.get("competition_mode"):
            sector_limited = []
            for _, sector_group in pool.groupby(["sector_code", "sector"], group_keys=False, dropna=False):
                first = sector_group.iloc[0]
                sector_code = str(first.get("sector_code", "") or "")
                sector_name = str(first.get("sector", "") or "")
                sector_limit = resolve_sector_name_limit(params, sector_code, sector_name)
                sector_limited.append(sector_group.head(max(1, sector_limit)))
            if sector_limited:
                pool = pd.concat(sector_limited, ignore_index=False).sort_values("score", ascending=False)
        daily[pd.Timestamp(date)] = pool.head(params["top_n"]).copy()
    return daily


def get_exit_slippage(exit_reason: Optional[str]) -> float:
    slip = EXIT_SLIPPAGE_RATE
    if exit_reason in {"gap_stop", "gap_take"}:
        slip += GAP_EXIT_EXTRA_SLIPPAGE
    return slip


def build_metric_equity_frame(equity_curve: pd.DataFrame) -> pd.DataFrame:
    if equity_curve.empty:
        return pd.DataFrame(columns=["date", "metric_equity"])

    work = equity_curve.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values("date").reset_index(drop=True)

    if "equity" in work.columns:
        metric_equity = pd.to_numeric(work["equity"], errors="coerce")
    elif "active_equity" in work.columns:
        metric_equity = pd.to_numeric(work["active_equity"], errors="coerce")
    else:
        raise KeyError("equity_curve must contain an 'equity' column or fallback 'active_equity' column.")

    work["metric_equity"] = metric_equity
    return work[["date", "metric_equity"]]


def calc_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame) -> dict:
    if equity_curve.empty:
        return {
            "total_return": np.nan,
            "cagr": np.nan,
            "mdd": np.nan,
            "volatility": np.nan,
            "sharpe": np.nan,
            "trade_count": 0,
            "win_rate": np.nan,
            "avg_trade_return": np.nan,
            "profit_factor": np.nan,
        }

    eq = build_metric_equity_frame(equity_curve)
    eq["daily_return"] = eq["metric_equity"].pct_change().fillna(0.0)
    eq["cummax"] = eq["metric_equity"].cummax()
    eq["drawdown"] = eq["metric_equity"] / eq["cummax"] - 1.0

    n_days = len(eq)
    total_return = eq["metric_equity"].iloc[-1] / eq["metric_equity"].iloc[0] - 1.0
    cagr = (
        (eq["metric_equity"].iloc[-1] / eq["metric_equity"].iloc[0]) ** (252 / n_days) - 1.0
        if n_days > 0
        else np.nan
    )
    volatility = eq["daily_return"].std() * np.sqrt(252)
    sharpe = (
        eq["daily_return"].mean() / eq["daily_return"].std() * np.sqrt(252)
        if eq["daily_return"].std() > 0
        else np.nan
    )

    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum() if not trades.empty else 0.0
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum() if not trades.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    return {
        "total_return": total_return,
        "cagr": cagr,
        "mdd": eq["drawdown"].min(),
        "volatility": volatility,
        "sharpe": sharpe,
        "trade_count": int(len(trades)),
        "win_rate": float((trades["return"] > 0).mean()) if not trades.empty else np.nan,
        "avg_trade_return": float(trades["return"].mean()) if not trades.empty else np.nan,
        "profit_factor": float(profit_factor) if pd.notna(profit_factor) else np.nan,
    }


def prepare_performance_series(equity_curve: pd.DataFrame, label: str) -> pd.DataFrame:
    if equity_curve.empty:
        return pd.DataFrame(columns=["date", "equity", "cum_return", "drawdown", "series"])

    work = build_metric_equity_frame(equity_curve).rename(columns={"metric_equity": "equity"})
    work["cum_return"] = work["equity"] / work["equity"].iloc[0] - 1.0
    work["drawdown"] = work["equity"] / work["equity"].cummax() - 1.0
    work["series"] = label
    return work


def build_performance_comparison_frame(strategy_eq: pd.DataFrame, benchmark_eq: pd.DataFrame) -> pd.DataFrame:
    strategy_series = prepare_performance_series(strategy_eq, "strategy").rename(
        columns={
            "equity": "strategy_equity",
            "cum_return": "strategy_cum_return",
            "drawdown": "strategy_drawdown",
        }
    )
    bench_label = str(benchmark_eq["symbol"].iloc[0]) if not benchmark_eq.empty and "symbol" in benchmark_eq.columns else "benchmark"
    benchmark_series = prepare_performance_series(benchmark_eq, bench_label).rename(
        columns={
            "equity": "benchmark_equity",
            "cum_return": "benchmark_cum_return",
            "drawdown": "benchmark_drawdown",
        }
    )

    keep_strategy = ["date", "strategy_equity", "strategy_cum_return", "strategy_drawdown"]
    keep_benchmark = ["date", "benchmark_equity", "benchmark_cum_return", "benchmark_drawdown"]
    merged = strategy_series[keep_strategy].merge(
        benchmark_series[keep_benchmark],
        on="date",
        how="outer",
    )
    return merged.sort_values("date").reset_index(drop=True)


def plot_performance_comparison(strategy_eq: pd.DataFrame, benchmark_eq: pd.DataFrame, out_path: Path) -> None:
    strategy_series = prepare_performance_series(strategy_eq, "Strategy")
    if strategy_series.empty:
        return

    benchmark_label = str(benchmark_eq["symbol"].iloc[0]) if not benchmark_eq.empty and "symbol" in benchmark_eq.columns else "Benchmark"
    benchmark_series = prepare_performance_series(benchmark_eq, benchmark_label)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(strategy_series["date"], strategy_series["cum_return"], linewidth=2.2, label="Strategy")
    if not benchmark_series.empty:
        axes[0].plot(benchmark_series["date"], benchmark_series["cum_return"], linewidth=1.8, label=benchmark_label)
    axes[0].set_title("Cumulative Return Comparison")
    axes[0].set_ylabel("Cumulative Return")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(strategy_series["date"], strategy_series["drawdown"], linewidth=2.0, label="Strategy MDD")
    if not benchmark_series.empty:
        axes[1].plot(benchmark_series["date"], benchmark_series["drawdown"], linewidth=1.6, label=f"{benchmark_label} MDD")
    axes[1].set_title("Drawdown Comparison")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def calc_yearly_returns(equity_curve: pd.DataFrame) -> pd.DataFrame:
    if equity_curve.empty:
        return pd.DataFrame(columns=["year", "start_equity", "end_equity", "return"])

    temp = build_metric_equity_frame(equity_curve).rename(columns={"metric_equity": "equity"})
    temp["year"] = temp["date"].dt.year
    yearly = (
        temp.groupby("year")
        .agg(start_equity=("equity", "first"), end_equity=("equity", "last"))
        .reset_index()
    )
    yearly["return"] = yearly["end_equity"] / yearly["start_equity"] - 1.0
    return yearly


def snapshot_open_positions(
    positions: Dict[str, Position],
    row_map: dict,
    current_date: pd.Timestamp,
) -> tuple[float, Dict[str, float], Dict[str, int]]:
    market_value = 0.0
    sector_values: Dict[str, float] = {}
    sector_counts: Dict[str, int] = {}
    for code, pos in positions.items():
        row = row_map.get((code, current_date))
        if row is None:
            continue
        value = float(row["adj_close"]) * pos.shares
        market_value += value
        sector_key = pos.sector_code or pos.sector
        sector_values[sector_key] = sector_values.get(sector_key, 0.0) + value
        sector_counts[sector_key] = sector_counts.get(sector_key, 0) + 1
    return market_value, sector_values, sector_counts


def run_backtest(context: dict, params: dict) -> dict:
    row_map = context["row_map"]
    trading_dates = context["trading_dates"]
    candidate_map = build_candidate_map(context["df"], params)

    strategy_capital_fraction = float(params.get("strategy_capital_fraction", 1.0))
    reserve_cash = INITIAL_CAPITAL * (1.0 - strategy_capital_fraction)
    cash = INITIAL_CAPITAL * strategy_capital_fraction
    positions: Dict[str, Position] = {}
    trades: List[dict] = []
    equity_rows: List[dict] = []

    for current_date in trading_dates:
        period_end = current_date == trading_dates[-1]
        exits: List[str] = []

        for code, pos in list(positions.items()):
            row = row_map.get((code, current_date))
            if row is None:
                continue

            open_p = float(row["adj_open"])
            high_p = float(row["adj_high"])
            low_p = float(row["adj_low"])
            close_p = float(row["adj_close"])
            pos.hold_days += 1

            stop_price = pos.entry_price * (1.0 + params["stop_loss"])
            take_price = pos.entry_price * (1.0 + params["take_profit"])

            exit_reason: Optional[str] = None
            exit_price: Optional[float] = None

            if open_p <= stop_price:
                exit_price = open_p
                exit_reason = "gap_stop"
            elif open_p >= take_price:
                exit_price = open_p
                exit_reason = "gap_take"
            elif low_p <= stop_price:
                exit_price = stop_price
                exit_reason = "stop_loss"
            elif high_p >= take_price:
                exit_price = take_price
                exit_reason = "take_profit"
            elif pos.hold_days >= params["max_hold_days"]:
                exit_price = close_p
                exit_reason = "max_hold"
            elif period_end:
                exit_price = close_p
                exit_reason = "end_of_backtest"

            if exit_reason is None or exit_price is None:
                continue

            exec_price = exit_price * (1.0 - get_exit_slippage(exit_reason))
            gross = exec_price * pos.shares
            sell_cost = gross * (SELL_FEE + SELL_TAX)
            net = gross - sell_cost
            pnl = net - pos.cost
            ret = pnl / pos.cost if pos.cost > 0 else np.nan

            cash += net
            trades.append(
                {
                    "code": code,
                    "name": pos.name,
                    "sector": pos.sector,
                    "sector_code": pos.sector_code,
                    "entry_date": pos.entry_date,
                    "exit_date": current_date,
                    "entry_price": pos.entry_price,
                    "exit_price": exec_price,
                    "shares": pos.shares,
                    "cost": pos.cost,
                    "net_proceeds": net,
                    "pnl": pnl,
                    "return": ret,
                    "hold_days": pos.hold_days,
                    "entry_score": pos.score,
                    "exit_reason": exit_reason,
                }
            )
            exits.append(code)

        for code in exits:
            positions.pop(code, None)

        slots = params["max_positions"] - len(positions)
        if slots > 0 and not period_end:
            candidates = candidate_map.get(current_date)
            if candidates is not None and not candidates.empty:
                candidates = candidates.loc[~candidates["code"].isin(positions.keys())].copy()
                current_market_value, sector_values, sector_counts = snapshot_open_positions(positions, row_map, current_date)
                current_equity = reserve_cash + cash + current_market_value
                max_weight_per_name = float(params.get("max_weight_per_name", 1.0))
                opened = 0
                for _, row in candidates.iterrows():
                    if opened >= slots:
                        break
                    open_price = float(row["adj_open"])
                    if not np.isfinite(open_price) or open_price <= 0:
                        continue

                    sector_code = str(row.get("sector_code", "") or "")
                    sector_name = str(row["sector"])
                    sector_key = sector_code or sector_name
                    sector_name_limit = resolve_sector_name_limit(params, sector_code, sector_name)
                    if sector_counts.get(sector_key, 0) >= sector_name_limit:
                        continue

                    exec_price = open_price * (1.0 + ENTRY_SLIPPAGE_RATE)
                    liquidity_cap = float(row["turn_ma20_sig"]) * MAX_ENTRY_TURNOVER_PARTICIPATION
                    if not np.isfinite(liquidity_cap) or liquidity_cap <= 0:
                        continue

                    name_cap_cash = current_equity * max_weight_per_name
                    sector_cap_cash = current_equity * get_sector_cap(params, sector_code, sector_name)
                    sector_room_cash = max(0.0, sector_cap_cash - sector_values.get(sector_key, 0.0))
                    alloc_cash = min(
                        cash / max(1, slots - opened),
                        liquidity_cap,
                        name_cap_cash,
                        sector_room_cash,
                    )
                    shares = int(alloc_cash / (exec_price * (1.0 + BUY_FEE)))
                    if shares <= 0:
                        continue

                    gross = shares * exec_price
                    buy_cost = gross * BUY_FEE
                    total_cost = gross + buy_cost
                    if total_cost > cash:
                        continue

                    cash -= total_cost
                    sector_values[sector_key] = sector_values.get(sector_key, 0.0) + gross
                    sector_counts[sector_key] = sector_counts.get(sector_key, 0) + 1
                    positions[row["code"]] = Position(
                        code=row["code"],
                        name=str(row["name"]),
                        sector=str(row["sector"]),
                        sector_code=sector_code,
                        entry_date=current_date,
                        entry_price=exec_price,
                        shares=shares,
                        cost=total_cost,
                        hold_days=0,
                        score=float(row["score"]),
                    )
                    opened += 1

        market_value, _, _ = snapshot_open_positions(positions, row_map, current_date)

        equity_rows.append(
            {
                "date": current_date,
                "cash": cash,
                "reserved_cash": reserve_cash,
                "market_value": market_value,
                "active_equity": cash + market_value,
                "equity": reserve_cash + cash + market_value,
                "n_positions": len(positions),
            }
        )

    trades_df = pd.DataFrame(trades)
    equity_curve = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
    metrics = calc_metrics(equity_curve, trades_df)
    yearly = calc_yearly_returns(equity_curve)
    return {
        "params": params,
        "trades": trades_df,
        "equity_curve": equity_curve,
        "metrics": metrics,
        "yearly_returns": yearly,
    }


def align_benchmark_to_trading_dates(
    bench: pd.DataFrame,
    trading_dates: Iterable[pd.Timestamp],
    capital_fraction: float = 1.0,
) -> pd.DataFrame:
    if bench.empty:
        return pd.DataFrame(columns=["date", "close", "daily_return", "equity", "symbol", "capital_fraction"])

    work = bench.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
    calendar = pd.DataFrame({"date": pd.to_datetime(pd.Index(list(trading_dates)))})
    calendar = calendar.drop_duplicates().sort_values("date").reset_index(drop=True)
    work = calendar.merge(work, on="date", how="left")
    work["close"] = work["close"].ffill().bfill()
    work = work.dropna(subset=["close"]).reset_index(drop=True)
    capital_fraction = float(np.clip(capital_fraction, 0.0, 1.0))
    work["index_return"] = work["close"].pct_change().fillna(0.0)
    work["index_equity"] = INITIAL_CAPITAL * (1.0 + work["index_return"]).cumprod()
    reserve_cash = INITIAL_CAPITAL * (1.0 - capital_fraction)
    invested_capital = INITIAL_CAPITAL * capital_fraction
    work["reserved_cash"] = reserve_cash
    work["active_equity"] = invested_capital * (work["index_equity"] / work["index_equity"].iloc[0])
    work["equity"] = reserve_cash + invested_capital * (work["index_equity"] / work["index_equity"].iloc[0])
    work["daily_return"] = work["equity"].pct_change().fillna(0.0)
    work["symbol"] = f"KOSPI x{capital_fraction:.2f}"
    work["capital_fraction"] = capital_fraction
    return work


def benchmark_kospi_index(context: dict, capital_fraction: float = 1.0) -> pd.DataFrame:
    trading_dates = pd.to_datetime(context["trading_dates"])
    if len(trading_dates) == 0:
        return pd.DataFrame(columns=["date", "close", "daily_return", "equity", "symbol", "capital_fraction"])

    raw = stock.get_index_ohlcv_by_date(
        yyyymmdd(trading_dates.min()),
        yyyymmdd(trading_dates.max()),
        KOSPI_INDEX_TICKER,
        name_display=False,
    )
    if raw.empty:
        raise RuntimeError("Could not download KOSPI index data from pykrx.")
    bench = raw.reset_index().rename(columns={"날짜": "date", "종가": "close"})
    return align_benchmark_to_trading_dates(bench[["date", "close"]], trading_dates, capital_fraction=capital_fraction)


def calc_relative_metrics(strategy_eq: pd.DataFrame, benchmark_eq: pd.DataFrame) -> dict:
    if strategy_eq.empty or benchmark_eq.empty:
        return {
            "benchmark_total_return": np.nan,
            "benchmark_mdd": np.nan,
            "excess_total_return": np.nan,
            "alpha_annualized": np.nan,
            "beta": np.nan,
            "tracking_error": np.nan,
            "information_ratio": np.nan,
        }

    strategy_metric = build_metric_equity_frame(strategy_eq).rename(columns={"metric_equity": "equity"})
    benchmark_metric = build_metric_equity_frame(benchmark_eq).rename(columns={"metric_equity": "equity"})
    merged = strategy_metric.merge(
        benchmark_metric,
        on="date",
        how="inner",
        suffixes=("_strategy", "_benchmark"),
    )
    if merged.empty:
        return {
            "benchmark_total_return": np.nan,
            "benchmark_mdd": np.nan,
            "excess_total_return": np.nan,
            "alpha_annualized": np.nan,
            "beta": np.nan,
            "tracking_error": np.nan,
            "information_ratio": np.nan,
        }

    merged["strategy_return"] = merged["equity_strategy"].pct_change().fillna(0.0)
    merged["benchmark_return"] = merged["equity_benchmark"].pct_change().fillna(0.0)
    merged["active_return"] = merged["strategy_return"] - merged["benchmark_return"]

    bench_var = merged["benchmark_return"].var()
    beta = (
        merged[["strategy_return", "benchmark_return"]].cov().iloc[0, 1] / bench_var
        if bench_var and np.isfinite(bench_var) and bench_var > 0
        else np.nan
    )
    alpha_daily = (
        merged["strategy_return"].mean() - beta * merged["benchmark_return"].mean()
        if np.isfinite(beta)
        else np.nan
    )
    tracking_error = merged["active_return"].std() * np.sqrt(252)
    information_ratio = (
        merged["active_return"].mean() / merged["active_return"].std() * np.sqrt(252)
        if merged["active_return"].std() > 0
        else np.nan
    )

    strategy_total_return = merged["equity_strategy"].iloc[-1] / merged["equity_strategy"].iloc[0] - 1.0
    benchmark_total_return = merged["equity_benchmark"].iloc[-1] / merged["equity_benchmark"].iloc[0] - 1.0
    benchmark_drawdown = merged["equity_benchmark"] / merged["equity_benchmark"].cummax() - 1.0
    return {
        "benchmark_total_return": benchmark_total_return,
        "benchmark_mdd": benchmark_drawdown.min(),
        "excess_total_return": strategy_total_return - benchmark_total_return,
        "alpha_annualized": alpha_daily * 252 if np.isfinite(alpha_daily) else np.nan,
        "beta": beta,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
    }


def parameter_space(competition_mode: bool = False, fixed_max_positions: Optional[int] = None) -> List[dict]:
    if competition_mode:
        min_prices = [1000, 2000]
        min_turnovers = [2_000_000_000.0, 3_000_000_000.0, 5_000_000_000.0, 10_000_000_000.0]
        min_component_ranks = [0.30, 0.35, 0.45, 0.55]
        top_ns = [5, 7, 10, 12, 16]
        max_positions_list = [3, 4, 6, 8]
        max_hold_days_list = [5, 7, 10]
        stop_losses = [-0.06, -0.08, -0.10]
        take_profits = [0.10, 0.12, 0.18]
    else:
        min_prices = [1000, 2000]
        min_turnovers = [3_000_000_000.0, 5_000_000_000.0, 10_000_000_000.0]
        min_component_ranks = [0.35, 0.45, 0.55]
        top_ns = [3, 5, 7]
        max_positions_list = [2, 3]
        max_hold_days_list = [5, 7, 10]
        stop_losses = [-0.06, -0.08, -0.10]
        take_profits = [0.12, 0.18, 0.24]

    if fixed_max_positions is not None:
        max_positions_list = [int(fixed_max_positions)]

    rows = []
    for recipe in RECIPE_LIBRARY:
        for min_price in min_prices:
            for min_turnover_krw in min_turnovers:
                for min_component_rank in min_component_ranks:
                    for top_n in top_ns:
                        for max_positions in max_positions_list:
                            for max_hold_days in max_hold_days_list:
                                for stop_loss in stop_losses:
                                    for take_profit in take_profits:
                                        if max_positions > top_n:
                                            continue
                                        rows.append(
                                            {
                                                "recipe": recipe,
                                                "min_price": min_price,
                                                "min_turnover_krw": min_turnover_krw,
                                                "min_component_rank": min_component_rank,
                                                "top_n": top_n,
                                                "max_positions": max_positions,
                                                "max_hold_days": max_hold_days,
                                                "stop_loss": stop_loss,
                                                "take_profit": take_profit,
                                            }
                                        )
    return rows


def calc_walkforward_stats(fold_metrics: List[tuple[str, dict]]) -> dict:
    returns = np.array([metrics["total_return"] for _, metrics in fold_metrics], dtype=float)
    mdds = np.array([metrics["mdd"] for _, metrics in fold_metrics], dtype=float)
    sharpes = np.array([metrics["sharpe"] for _, metrics in fold_metrics], dtype=float)
    trades = np.array([metrics["trade_count"] for _, metrics in fold_metrics], dtype=float)

    cv_total_return = float(np.prod(1.0 + returns) - 1.0)
    cv_mean_return = float(np.nanmean(returns))
    cv_worst_return = float(np.nanmin(returns))
    cv_mean_mdd = float(np.nanmean(mdds))
    cv_mean_sharpe = float(np.nanmean(sharpes))
    cv_positive_folds = int(np.sum(returns > 0))
    cv_trade_count = int(np.nansum(trades))

    drawdown_penalty = max(0.0, abs(min(cv_mean_mdd, 0.0)) - 0.25)
    tail_penalty = max(0.0, abs(min(cv_worst_return, 0.0)) - 0.10)
    return_score = (
        cv_total_return * 0.78
        + cv_mean_return * 0.18
        + cv_mean_sharpe * 0.03
        + cv_positive_folds * 0.03
        - drawdown_penalty * 0.70
        - tail_penalty * 0.50
    )
    sharpe_score = (
        cv_mean_sharpe * 0.72
        + cv_total_return * 0.12
        + cv_mean_return * 0.08
        + cv_positive_folds * 0.03
        - drawdown_penalty * 0.35
        - tail_penalty * 0.15
    )

    stats = {
        "return_score": return_score,
        "sharpe_score": sharpe_score,
        "cv_total_return": cv_total_return,
        "cv_mean_return": cv_mean_return,
        "cv_worst_return": cv_worst_return,
        "cv_mean_mdd": cv_mean_mdd,
        "cv_mean_sharpe": cv_mean_sharpe,
        "cv_positive_folds": cv_positive_folds,
        "cv_trade_count": cv_trade_count,
    }
    for fold_name, metrics in fold_metrics:
        stats[f"{fold_name}_return"] = metrics["total_return"]
        stats[f"{fold_name}_mdd"] = metrics["mdd"]
        stats[f"{fold_name}_sharpe"] = metrics["sharpe"]
        stats[f"{fold_name}_trade_count"] = metrics["trade_count"]
    return stats


def search_alpha_strategies(
    feat: pd.DataFrame,
    periods: dict[str, pd.Timestamp],
    n_samples: int,
    static_params: Optional[dict] = None,
    fixed_max_positions: Optional[int] = None,
) -> pd.DataFrame:
    competition_mode = bool((static_params or {}).get("competition_mode"))
    all_params = parameter_space(
        competition_mode=competition_mode,
        fixed_max_positions=fixed_max_positions,
    )
    if n_samples < len(all_params):
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(all_params), size=n_samples, replace=False)
        params_to_run = [all_params[i] for i in sample_idx]
    else:
        params_to_run = all_params

    train_context = build_period_context(feat, periods["train_start"], periods["train_end"])
    fold_contexts = [
        (fold_name, build_period_context(feat, fold_start, fold_end))
        for fold_name, fold_start, fold_end in SELECTION_FOLDS
    ]

    rows = []
    total = len(params_to_run)
    print(f"[search] evaluating {total} parameter combinations")
    for idx, candidate_params in enumerate(params_to_run, start=1):
        runtime_params = {**(static_params or {}), **candidate_params}
        train = run_backtest(train_context, runtime_params)
        train_metrics = train["metrics"]
        if train_metrics["trade_count"] < 25:
            continue

        fold_metrics: List[tuple[str, dict]] = []
        for fold_name, fold_context in fold_contexts:
            fold_result = run_backtest(fold_context, runtime_params)
            fold_metrics.append((fold_name, fold_result["metrics"]))

        walkforward = calc_walkforward_stats(fold_metrics)
        if walkforward["cv_trade_count"] < 20 or walkforward["cv_positive_folds"] < 1:
            continue

        rows.append(
            {
                **candidate_params,
                **walkforward,
                "train_total_return": train_metrics["total_return"],
                "train_cagr": train_metrics["cagr"],
                "train_mdd": train_metrics["mdd"],
                "train_sharpe": train_metrics["sharpe"],
                "train_trade_count": train_metrics["trade_count"],
                "train_win_rate": train_metrics["win_rate"],
                "train_avg_trade_return": train_metrics["avg_trade_return"],
                "train_profit_factor": train_metrics["profit_factor"],
            }
        )

        if idx == 1 or idx % 100 == 0 or idx == total:
            print(f"[search] processed {idx}/{total}")

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No pykrx-only alpha strategy survived the search.")
    return result.sort_values(["return_score", "cv_total_return", "cv_mean_return"], ascending=[False, False, False]).reset_index(drop=True)


def extract_params_from_row(row: pd.Series) -> dict:
    return {
        "recipe": str(row["recipe"]),
        "min_price": int(row["min_price"]),
        "min_turnover_krw": float(row["min_turnover_krw"]),
        "min_component_rank": float(row["min_component_rank"]),
        "top_n": int(row["top_n"]),
        "max_positions": int(row["max_positions"]),
        "max_hold_days": int(row["max_hold_days"]),
        "stop_loss": float(row["stop_loss"]),
        "take_profit": float(row["take_profit"]),
    }


def choose_best_row(search_df: pd.DataFrame, objective: str) -> pd.Series:
    shortlist = search_df.loc[
        (search_df["cv_total_return"] > 0)
        & (search_df["cv_mean_mdd"] > -0.35)
        & (search_df["cv_positive_folds"] >= 2)
    ].copy()
    if shortlist.empty:
        shortlist = search_df.copy()

    sort_cols = (
        ["return_score", "cv_total_return", "cv_mean_return", "train_sharpe"]
        if objective == "return"
        else ["sharpe_score", "cv_mean_sharpe", "cv_total_return", "train_sharpe"]
    )
    return shortlist.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0]


def choose_best_params(search_df: pd.DataFrame, objective: str) -> dict:
    return extract_params_from_row(choose_best_row(search_df, objective))


def build_live_watchlist(feat: pd.DataFrame, params: dict, screen_date: pd.Timestamp) -> pd.DataFrame:
    latest = feat.loc[feat["date"] == screen_date].copy()
    if latest.empty:
        return latest

    recipe = RECIPE_LIBRARY[params["recipe"]]
    cond = (
        (latest["adj_close"] >= params["min_price"])
        & (latest["turn_ma20"] >= params["min_turnover_krw"])
    )
    for factor, _ in recipe:
        cond &= latest[f"live_{factor}"] >= params["min_component_rank"]

    latest = latest.loc[cond].copy()
    if latest.empty:
        return latest

    latest["score"] = 0.0
    for factor, weight in recipe:
        latest["score"] += latest[f"live_{factor}"] * weight

    cols = [
        "date",
        "code",
        "name",
        "sector",
        "sector_code",
        "score",
        "adj_close",
        "turn_ma20",
        "ret5",
        "ret20",
        "market_cap_krw",
        "short5_to_turn",
        "short10_to_turn",
        "turnover_surge",
        "dd60",
        "sector_alpha20",
    ]
    latest = latest[cols].sort_values("score", ascending=False).reset_index(drop=True)
    latest["watch_rank"] = np.arange(1, len(latest) + 1)
    latest["max_name_weight"] = float(params.get("max_weight_per_name", 1.0))
    latest["sector_cap_weight"] = latest.apply(
        lambda row: get_sector_cap(params, str(row.get("sector_code", "")), str(row["sector"])),
        axis=1,
    )
    return latest


def build_screen_text(screen_date: pd.Timestamp, params: dict, watchlist: pd.DataFrame, top_k: int) -> str:
    lines = []
    lines.append("Codex Swing 03_23 Standalone Realtime Screen")
    lines.append("")
    lines.append(f"signal_date: {screen_date.date()}")
    lines.append("selection_basis: current close data for next session")
    lines.append("")
    lines.append("params")
    for key, value in params.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("top_candidates")
    if watchlist.empty:
        lines.append("- no candidates")
    else:
        for _, row in watchlist.head(top_k).iterrows():
            lines.append(
                f"- {int(row['watch_rank'])}. {row['code']} {row['name']} / "
                f"{row.get('sector_code', '')} {row['sector']} / "
                f"score={row['score']:.4f} / ret5={row['ret5']:.2%} / ret20={row['ret20']:.2%} / "
                f"name_cap={row.get('max_name_weight', np.nan):.1%} / sector_cap={row.get('sector_cap_weight', np.nan):.1%}"
            )
    return "\n".join(lines)


def evaluate_strategy_bundle(
    feat: pd.DataFrame,
    params: dict,
    periods: dict[str, pd.Timestamp],
    latest_business_day: pd.Timestamp,
    top_k: int,
) -> dict:
    train_context = build_period_context(feat, periods["train_start"], periods["train_end"])
    holdout_context = build_period_context(feat, periods["holdout_start"], periods["holdout_end"])
    full_context = build_period_context(feat, periods["train_start"], periods["holdout_end"])

    train_result = run_backtest(train_context, params)
    holdout_result = run_backtest(holdout_context, params)
    full_result = run_backtest(full_context, params)
    benchmark = benchmark_kospi_index(full_context, capital_fraction=float(params.get("strategy_capital_fraction", 1.0)))

    train_bench = benchmark.loc[benchmark["date"].between(periods["train_start"], periods["train_end"])].reset_index(drop=True)
    holdout_bench = benchmark.loc[
        benchmark["date"].between(periods["holdout_start"], periods["holdout_end"])
    ].reset_index(drop=True)
    relative_metrics = pd.DataFrame(
        [
            {"period": "train", **calc_relative_metrics(train_result["equity_curve"], train_bench)},
            {"period": "holdout", **calc_relative_metrics(holdout_result["equity_curve"], holdout_bench)},
            {"period": "full", **calc_relative_metrics(full_result["equity_curve"], benchmark)},
        ]
    )

    watchlist = build_live_watchlist(feat, params, latest_business_day)
    screen_text = build_screen_text(latest_business_day, params, watchlist, top_k)
    empty_trades = pd.DataFrame(columns=["pnl", "return"])
    metrics_table = pd.DataFrame(
        [
            {
                "period": "train",
                "strategy_total_return": train_result["metrics"]["total_return"],
                "strategy_cagr": train_result["metrics"]["cagr"],
                "strategy_mdd": train_result["metrics"]["mdd"],
                "strategy_volatility": train_result["metrics"]["volatility"],
                "strategy_sharpe": train_result["metrics"]["sharpe"],
                "strategy_trade_count": train_result["metrics"]["trade_count"],
                "strategy_win_rate": train_result["metrics"]["win_rate"],
                "strategy_avg_trade_return": train_result["metrics"]["avg_trade_return"],
                "strategy_profit_factor": train_result["metrics"]["profit_factor"],
                **{
                    f"benchmark_{k}": v
                    for k, v in calc_metrics(train_bench, empty_trades).items()
                    if k in {"total_return", "cagr", "mdd", "volatility", "sharpe"}
                },
                **calc_relative_metrics(train_result["equity_curve"], train_bench),
            },
            {
                "period": "holdout",
                "strategy_total_return": holdout_result["metrics"]["total_return"],
                "strategy_cagr": holdout_result["metrics"]["cagr"],
                "strategy_mdd": holdout_result["metrics"]["mdd"],
                "strategy_volatility": holdout_result["metrics"]["volatility"],
                "strategy_sharpe": holdout_result["metrics"]["sharpe"],
                "strategy_trade_count": holdout_result["metrics"]["trade_count"],
                "strategy_win_rate": holdout_result["metrics"]["win_rate"],
                "strategy_avg_trade_return": holdout_result["metrics"]["avg_trade_return"],
                "strategy_profit_factor": holdout_result["metrics"]["profit_factor"],
                **{
                    f"benchmark_{k}": v
                    for k, v in calc_metrics(holdout_bench, empty_trades).items()
                    if k in {"total_return", "cagr", "mdd", "volatility", "sharpe"}
                },
                **calc_relative_metrics(holdout_result["equity_curve"], holdout_bench),
            },
            {
                "period": "full",
                "strategy_total_return": full_result["metrics"]["total_return"],
                "strategy_cagr": full_result["metrics"]["cagr"],
                "strategy_mdd": full_result["metrics"]["mdd"],
                "strategy_volatility": full_result["metrics"]["volatility"],
                "strategy_sharpe": full_result["metrics"]["sharpe"],
                "strategy_trade_count": full_result["metrics"]["trade_count"],
                "strategy_win_rate": full_result["metrics"]["win_rate"],
                "strategy_avg_trade_return": full_result["metrics"]["avg_trade_return"],
                "strategy_profit_factor": full_result["metrics"]["profit_factor"],
                **{
                    f"benchmark_{k}": v
                    for k, v in calc_metrics(benchmark, empty_trades).items()
                    if k in {"total_return", "cagr", "mdd", "volatility", "sharpe"}
                },
                **calc_relative_metrics(full_result["equity_curve"], benchmark),
            },
        ]
    )
    return {
        "params": params,
        "train_result": train_result,
        "holdout_result": holdout_result,
        "full_result": full_result,
        "benchmark": benchmark,
        "relative_metrics": relative_metrics,
        "metrics_table": metrics_table,
        "watchlist": watchlist,
        "screen_text": screen_text,
    }


def make_summary_text(
    login_ok: bool,
    params: dict,
    periods: dict[str, pd.Timestamp],
    train_result: dict,
    holdout_result: dict,
    full_result: dict,
    relative_metrics: pd.DataFrame,
    watchlist: pd.DataFrame,
    title: str = "Codex Swing 03_23 Standalone Summary",
    objective: Optional[str] = None,
    search_row: Optional[pd.Series] = None,
) -> str:
    rel_map = relative_metrics.set_index("period").to_dict("index") if not relative_metrics.empty else {}
    benchmark_fraction = float(params.get("strategy_capital_fraction", 1.0))
    benchmark_label = "benchmark_kospi_return" if benchmark_fraction >= 0.9999 else f"benchmark_kospi_x{benchmark_fraction:.1%}_return"
    excess_label = "excess_return_vs_kospi" if benchmark_fraction >= 0.9999 else f"excess_return_vs_kospi_x{benchmark_fraction:.1%}"
    lines = []
    lines.append(title)
    lines.append("")
    lines.append(f"pykrx_login_checked: {login_ok}")
    lines.append("")
    lines.append("Strategy params")
    lines.append(json.dumps(params, ensure_ascii=False, indent=2))
    if objective:
        lines.append("")
        lines.append(f"selection_objective: {objective}")
    if search_row is not None:
        lines.append("")
        lines.append("Search stats")
        lines.append(f"- return_score: {search_row.get('return_score', np.nan):.4f}")
        lines.append(f"- sharpe_score: {search_row.get('sharpe_score', np.nan):.4f}")
        lines.append(f"- cv_total_return: {search_row.get('cv_total_return', np.nan):.2%}")
        lines.append(f"- cv_mean_return: {search_row.get('cv_mean_return', np.nan):.2%}")
        lines.append(f"- cv_mean_sharpe: {search_row.get('cv_mean_sharpe', np.nan):.2f}")
        lines.append(f"- cv_mean_mdd: {search_row.get('cv_mean_mdd', np.nan):.2%}")
        lines.append(f"- cv_positive_folds: {int(search_row.get('cv_positive_folds', 0))}")
        lines.append(f"- cv_trade_count: {int(search_row.get('cv_trade_count', 0))}")
    lines.append("")
    lines.append("Date ranges")
    lines.append(f"- train: {periods['train_start'].date()} ~ {periods['train_end'].date()}")
    lines.append(f"- holdout: {periods['holdout_start'].date()} ~ {periods['holdout_end'].date()}")
    lines.append(f"- full: {periods['train_start'].date()} ~ {periods['holdout_end'].date()}")
    lines.append("")
    lines.append("Execution assumptions")
    lines.append("- pykrx-only cross-sectional snapshots")
    lines.append("- signal features are shifted before trading")
    lines.append("- 0.05% entry slippage, 0.05% exit slippage, gap exits use extra slippage")
    lines.append("- max entry participation capped at 3% of 20-day average turnover")
    lines.append(
        f"- benchmark is modeled as KOSPI with {benchmark_fraction:.1%} capital exposure and the rest held in cash"
    )
    if float(params.get("strategy_capital_fraction", 1.0)) < 1.0:
        reserved_fraction = 1.0 - float(params["strategy_capital_fraction"])
        lines.append(
            f"- only {float(params['strategy_capital_fraction']):.1%} of total capital is deployed by this strategy; "
            f"remaining {reserved_fraction:.1%} is reserved for other sleeves"
        )
        lines.append(
            f"- performance metrics below keep the reserved cash included, same as the benchmark convention"
        )
    if params.get("competition_mode"):
        lines.append(f"- competition universe filter enabled: {params.get('competition_universe_csv')}")
        lines.append(f"- per-name cap: {params.get('max_weight_per_name', np.nan):.1%}")
        lines.append(f"- max names per sector: {int(params.get('max_names_per_sector', 0))}")
        exempt_codes = ",".join(params.get("multi_name_sector_codes", [])) or "-"
        lines.append(f"- multi-name sector exemptions: {exempt_codes}")
        lines.append(
            f"- non-exempt sectors can hold at most {int(params.get('default_non_exempt_sector_name_limit', 0))} name(s)"
        )
    lines.append("")
    for label, result in [("train", train_result), ("holdout", holdout_result), ("full", full_result)]:
        metrics = result["metrics"]
        rel = rel_map.get(label, {})
        lines.append(label)
        lines.append(f"- total_return: {metrics['total_return']:.2%}")
        lines.append(f"- cagr: {metrics['cagr']:.2%}")
        lines.append(f"- mdd: {metrics['mdd']:.2%}")
        lines.append(f"- volatility: {metrics['volatility']:.2%}")
        lines.append(f"- sharpe: {metrics['sharpe']:.2f}")
        lines.append(f"- trade_count: {metrics['trade_count']}")
        lines.append(f"- win_rate: {metrics['win_rate']:.2%}")
        lines.append(f"- avg_trade_return: {metrics['avg_trade_return']:.2%}")
        lines.append(f"- profit_factor: {metrics['profit_factor']:.2f}")
        lines.append(f"- {benchmark_label}: {rel.get('benchmark_total_return', np.nan):.2%}")
        lines.append(f"- benchmark_mdd: {rel.get('benchmark_mdd', np.nan):.2%}")
        lines.append(f"- {excess_label}: {rel.get('excess_total_return', np.nan):.2%}")
        lines.append("")

    lines.append("Latest watchlist")
    if watchlist.empty:
        lines.append("- no candidates")
    else:
        for _, row in watchlist.head(15).iterrows():
            lines.append(
                f"- {int(row['watch_rank'])}. {row['code']} {row['name']} / "
                f"{row.get('sector_code', '')} {row['sector']} / "
                f"score={row['score']:.4f} / ret5={row['ret5']:.2%} / ret20={row['ret20']:.2%} / "
                f"name_cap={row.get('max_name_weight', np.nan):.1%} / sector_cap={row.get('sector_cap_weight', np.nan):.1%}"
            )
    return "\n".join(lines)


def save_strategy_bundle(
    prefix: str,
    bundle: dict,
    summary_text: str,
    out_dir: Path,
    latest_business_day: pd.Timestamp,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_df = build_performance_comparison_frame(bundle["full_result"]["equity_curve"], bundle["benchmark"])
    bundle["watchlist"].to_csv(out_dir / f"{prefix}_watchlist_latest.csv", index=False, encoding="utf-8-sig")
    bundle["watchlist"].to_csv(
        out_dir / f"{prefix}_watchlist_{yyyymmdd(latest_business_day)}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (out_dir / f"{prefix}_watchlist_latest.txt").write_text(bundle["screen_text"], encoding="utf-8")
    (out_dir / f"{prefix}_watchlist_{yyyymmdd(latest_business_day)}.txt").write_text(
        bundle["screen_text"],
        encoding="utf-8",
    )
    (out_dir / f"{prefix}_params.json").write_text(
        json.dumps(bundle["params"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    bundle["full_result"]["trades"].to_csv(out_dir / f"{prefix}_trades.csv", index=False, encoding="utf-8-sig")
    bundle["full_result"]["equity_curve"].to_csv(
        out_dir / f"{prefix}_equity_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bundle["full_result"]["yearly_returns"].to_csv(
        out_dir / f"{prefix}_yearly_returns.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison_df.to_csv(
        out_dir / f"{prefix}_performance_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bundle["benchmark"].to_csv(out_dir / f"{prefix}_kospi_benchmark.csv", index=False, encoding="utf-8-sig")
    bundle["relative_metrics"].to_csv(
        out_dir / f"{prefix}_relative_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bundle["metrics_table"].to_csv(
        out_dir / f"{prefix}_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_performance_comparison(
        bundle["full_result"]["equity_curve"],
        bundle["benchmark"],
        out_dir / f"{prefix}_performance_comparison.png",
    )
    (out_dir / f"{prefix}_summary.txt").write_text(summary_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    params = build_params(args)
    competition_universe: Optional[pd.DataFrame] = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] initializing pykrx session")
    login_ok = initialize_pykrx_session(args.krx_id, args.krx_pw, allow_prompt=not args.skip_krx_login_prompt)

    print("[2/5] resolving dates and building panel")
    latest_business_day = resolve_latest_business_day(args.screen_date)
    periods = build_periods(args, latest_business_day)
    if args.screen_only:
        panel_start = latest_business_day - pd.Timedelta(days=args.warmup_days)
        panel_end = latest_business_day
    else:
        panel_start = min(periods["train_start"], latest_business_day) - pd.Timedelta(days=args.warmup_days)
        panel_end = max(periods["holdout_end"], latest_business_day)
    panel = build_panel(panel_start, panel_end, refresh=(args.refresh_panel or args.no_cache))
    if params.get("competition_universe_csv"):
        competition_universe = load_competition_universe(Path(params["competition_universe_csv"]))
    panel = apply_competition_universe(panel, competition_universe, params.get("excluded_codes", []))
    print(f"[INFO] active panel rows: {len(panel):,} / codes: {panel['code'].nunique():,}")

    print("[3/5] building features")
    feat = build_features(panel)

    if args.run_search:
        print("[4/6] running pykrx alpha search")
        static_search_params = {
            "strategy_capital_fraction": params.get("strategy_capital_fraction", 1.0),
            "competition_mode": params.get("competition_mode", False),
            "competition_universe_csv": params.get("competition_universe_csv"),
            "excluded_codes": params.get("excluded_codes", []),
            "max_weight_per_name": params.get("max_weight_per_name", 1.0),
            "max_names_per_sector": params.get("max_names_per_sector", 9999),
            "sector_caps": params.get("sector_caps", {}),
            "default_sector_cap": params.get("default_sector_cap", 1.0),
        }
        search_df = search_alpha_strategies(
            feat,
            periods,
            args.search_samples,
            static_params=static_search_params,
            fixed_max_positions=args.search_fixed_max_positions,
        )
        best_return_row = choose_best_row(search_df, "return")
        best_sharpe_row = choose_best_row(search_df, "sharpe")
        best_return_params = {**static_search_params, **extract_params_from_row(best_return_row)}
        best_sharpe_params = {**static_search_params, **extract_params_from_row(best_sharpe_row)}

        print("[5/6] evaluating selected strategies")
        best_return_bundle = evaluate_strategy_bundle(
            feat=feat,
            params=best_return_params,
            periods=periods,
            latest_business_day=latest_business_day,
            top_k=args.top_k,
        )
        best_sharpe_bundle = evaluate_strategy_bundle(
            feat=feat,
            params=best_sharpe_params,
            periods=periods,
            latest_business_day=latest_business_day,
            top_k=args.top_k,
        )

        best_return_summary = make_summary_text(
            login_ok=login_ok,
            params=best_return_params,
            periods=periods,
            train_result=best_return_bundle["train_result"],
            holdout_result=best_return_bundle["holdout_result"],
            full_result=best_return_bundle["full_result"],
            relative_metrics=best_return_bundle["relative_metrics"],
            watchlist=best_return_bundle["watchlist"],
            title="Codex Swing 03_23 Pykrx Search Summary - Best Return",
            objective="maximize_return",
            search_row=best_return_row,
        )
        best_sharpe_summary = make_summary_text(
            login_ok=login_ok,
            params=best_sharpe_params,
            periods=periods,
            train_result=best_sharpe_bundle["train_result"],
            holdout_result=best_sharpe_bundle["holdout_result"],
            full_result=best_sharpe_bundle["full_result"],
            relative_metrics=best_sharpe_bundle["relative_metrics"],
            watchlist=best_sharpe_bundle["watchlist"],
            title="Codex Swing 03_23 Pykrx Search Summary - Best Sharpe",
            objective="maximize_sharpe",
            search_row=best_sharpe_row,
        )

        comparison_rows = []
        for objective_name, row, bundle in [
            ("best_return", best_return_row, best_return_bundle),
            ("best_sharpe", best_sharpe_row, best_sharpe_bundle),
        ]:
            rel_map = bundle["relative_metrics"].set_index("period").to_dict("index")
            comparison_rows.append(
                {
                    "objective": objective_name,
                    **extract_params_from_row(row),
                    "return_score": float(row["return_score"]),
                    "sharpe_score": float(row["sharpe_score"]),
                    "cv_total_return": float(row["cv_total_return"]),
                    "cv_mean_return": float(row["cv_mean_return"]),
                    "cv_mean_sharpe": float(row["cv_mean_sharpe"]),
                    "cv_mean_mdd": float(row["cv_mean_mdd"]),
                    "train_total_return": float(bundle["train_result"]["metrics"]["total_return"]),
                    "train_sharpe": float(bundle["train_result"]["metrics"]["sharpe"]),
                    "holdout_total_return": float(bundle["holdout_result"]["metrics"]["total_return"]),
                    "holdout_sharpe": float(bundle["holdout_result"]["metrics"]["sharpe"]),
                    "full_total_return": float(bundle["full_result"]["metrics"]["total_return"]),
                    "full_sharpe": float(bundle["full_result"]["metrics"]["sharpe"]),
                    "full_mdd": float(bundle["full_result"]["metrics"]["mdd"]),
                    "full_excess_return_vs_kospi": float(rel_map.get("full", {}).get("excess_total_return", np.nan)),
                    "full_information_ratio": float(rel_map.get("full", {}).get("information_ratio", np.nan)),
                }
            )
        comparison_df = pd.DataFrame(comparison_rows)

        overview_lines = [
            "Codex Swing 03_23 Pykrx Alpha Search Overview",
            "",
            f"search_samples_requested: {args.search_samples}",
            f"search_candidates_survived: {len(search_df)}",
            f"signal_date_for_watchlist: {latest_business_day.date()}",
            "",
        ]
        for objective_name, row, bundle in [
            ("best_return", best_return_row, best_return_bundle),
            ("best_sharpe", best_sharpe_row, best_sharpe_bundle),
        ]:
            metrics = bundle["full_result"]["metrics"]
            rel_map = bundle["relative_metrics"].set_index("period").to_dict("index")
            overview_lines.extend(
                [
                    objective_name,
                    f"- recipe: {row['recipe']}",
                    f"- total_return_full: {metrics['total_return']:.2%}",
                    f"- sharpe_full: {metrics['sharpe']:.2f}",
                    f"- mdd_full: {metrics['mdd']:.2%}",
                    f"- excess_return_vs_kospi_full: {rel_map.get('full', {}).get('excess_total_return', np.nan):.2%}",
                    "",
                ]
            )
        overview_text = "\n".join(overview_lines)

        print("[6/6] saving search outputs")
        search_df.to_csv(OUT_DIR / "pykrx_alpha_search_results.csv", index=False, encoding="utf-8-sig")
        comparison_df.to_csv(OUT_DIR / "pykrx_alpha_objective_compare.csv", index=False, encoding="utf-8-sig")
        save_strategy_bundle("pykrx_best_return", best_return_bundle, best_return_summary, OUT_DIR, latest_business_day)
        save_strategy_bundle("pykrx_best_sharpe", best_sharpe_bundle, best_sharpe_summary, OUT_DIR, latest_business_day)
        (OUT_DIR / "pykrx_alpha_search_overview.txt").write_text(overview_text, encoding="utf-8")
        print(overview_text)
        print("")
        print(best_return_summary)
        print("")
        print(best_sharpe_summary)
        print(f"[INFO] search outputs saved to: {OUT_DIR}")
        return

    print("[4/5] building latest watchlist")
    watchlist = build_live_watchlist(feat, params, latest_business_day)
    screen_text = build_screen_text(latest_business_day, params, watchlist, args.top_k)

    watchlist.to_csv(OUT_DIR / "watchlist_latest.csv", index=False, encoding="utf-8-sig")
    watchlist.to_csv(OUT_DIR / f"watchlist_{yyyymmdd(latest_business_day)}.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "watchlist_latest.txt").write_text(screen_text, encoding="utf-8")
    (OUT_DIR / f"watchlist_{yyyymmdd(latest_business_day)}.txt").write_text(screen_text, encoding="utf-8")
    (OUT_DIR / "strategy_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(screen_text)

    if args.screen_only:
        print(f"[INFO] screen-only mode complete. outputs saved to: {OUT_DIR}")
        return

    print("[5/5] running backtests and saving outputs")
    bundle = evaluate_strategy_bundle(
        feat=feat,
        params=params,
        periods=periods,
        latest_business_day=latest_business_day,
        top_k=args.top_k,
    )

    summary_text = make_summary_text(
        login_ok=login_ok,
        params=params,
        periods=periods,
        train_result=bundle["train_result"],
        holdout_result=bundle["holdout_result"],
        full_result=bundle["full_result"],
        relative_metrics=bundle["relative_metrics"],
        watchlist=bundle["watchlist"],
    )
    save_strategy_bundle("strategy", bundle, summary_text, OUT_DIR, latest_business_day)
    bundle["watchlist"].to_csv(OUT_DIR / "watchlist_latest.csv", index=False, encoding="utf-8-sig")
    bundle["watchlist"].to_csv(OUT_DIR / f"watchlist_{yyyymmdd(latest_business_day)}.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "watchlist_latest.txt").write_text(bundle["screen_text"], encoding="utf-8")
    (OUT_DIR / f"watchlist_{yyyymmdd(latest_business_day)}.txt").write_text(bundle["screen_text"], encoding="utf-8")
    (OUT_DIR / "strategy_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    bundle["full_result"]["trades"].to_csv(OUT_DIR / "strategy_trades.csv", index=False, encoding="utf-8-sig")
    bundle["full_result"]["equity_curve"].to_csv(OUT_DIR / "strategy_equity_curve.csv", index=False, encoding="utf-8-sig")
    bundle["full_result"]["yearly_returns"].to_csv(OUT_DIR / "strategy_yearly_returns.csv", index=False, encoding="utf-8-sig")
    bundle["benchmark"].to_csv(OUT_DIR / "strategy_kospi_benchmark.csv", index=False, encoding="utf-8-sig")
    bundle["relative_metrics"].to_csv(OUT_DIR / "strategy_relative_metrics.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "strategy_summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(f"[INFO] outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
