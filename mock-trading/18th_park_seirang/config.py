"""Global configuration for the KOSPI Modified Market-Cap Weighting strategy."""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Backtest period
# ---------------------------------------------------------------------------
START_DATE: str = "2019-01-01"
END_DATE: str = "2026-03-23"

# pykrx uses YYYYMMDD format
START_DATE_KRX: str = "20190101"
END_DATE_KRX: str = "20260323"

# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
INITIAL_CAPITAL: float = 1_000_000_000.0  # 10억 원

# ---------------------------------------------------------------------------
# Universe filter
# ---------------------------------------------------------------------------
ADTV_THRESHOLD: float = 5_000_000_000.0  # 50억 원 (30-day avg trading value)
ADTV_WINDOW: int = 30
FREE_FLOAT_THRESHOLD: float = 0.15  # 15%

# ---------------------------------------------------------------------------
# Weight constraints
# ---------------------------------------------------------------------------
CAP_LIMIT: float = 0.15          # 15% individual cap
MIN_WEIGHT: float = 0.0005       # 0.05% minimum weight
SECTOR_PENALTY_LAMBDA: float = 1000.0  # Sector neutrality soft-penalty strength

# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------
COMMISSION_RATE: float = 0.00015  # 0.015% one-way
SECURITIES_TAX: float = 0.0025    # 0.25% sell-side only
SLIPPAGE_RATE: float = 0.0005     # 5 bp per trade

# ---------------------------------------------------------------------------
# Rebalancing schedule
# ---------------------------------------------------------------------------
REBAL_MONTHS: List[int] = [3, 6, 9, 12]

# ---------------------------------------------------------------------------
# Benchmark index tickers (pykrx)
# Try in order; fall back to regular KOSPI if TR is unavailable
# ---------------------------------------------------------------------------
BENCHMARK_TICKERS: List[str] = ["1028", "1001"]
# 1028 = KOSPI Total Return (dividend reinvested) — may not be available
# 1001 = KOSPI (price return) — fallback

# ---------------------------------------------------------------------------
# KRX industry name → sector group mapping
# Source: pykrx get_market_sector_classifications returns KRX 업종명
# This maps ~30 KRX industry labels to 11 GICS-like sector groups.
# No per-ticker manual work needed.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# KSIC (한국표준산업분류) code prefix → sector group
# Used when DART returns induty_code (e.g. "C2621", "K6410")
# ---------------------------------------------------------------------------
KSIC_TO_SECTOR: Dict[str, str] = {
    # 제조업 — 소재
    "C10": "필수소비재", "C11": "필수소비재", "C12": "필수소비재",
    "C13": "경기소비재", "C14": "경기소비재", "C15": "경기소비재",
    "C16": "소재", "C17": "소재", "C18": "소재",
    "C19": "에너지",
    "C20": "소재", "C21": "헬스케어", "C22": "소재", "C23": "소재",
    "C24": "소재", "C25": "소재",
    # 전기전자/반도체
    "C26": "IT/반도체", "C27": "헬스케어",
    "C28": "산업재", "C29": "경기소비재",
    "C30": "산업재", "C31": "경기소비재", "C32": "경기소비재",
    "C33": "헬스케어",
    # 인프라
    "D":   "유틸리티",
    "E":   "기타",
    "F":   "산업재",
    "G":   "경기소비재",
    "H":   "산업재",
    "I":   "경기소비재",
    # IT/통신
    "J58": "IT/반도체", "J59": "커뮤니케이션",
    "J60": "커뮤니케이션", "J61": "커뮤니케이션",
    "J62": "IT/반도체", "J63": "IT/반도체",
    # 금융
    "K64": "금융", "K65": "금융", "K66": "금융",
    # 부동산
    "L":   "부동산",
    # 기타 서비스
    "M":   "기타", "N":   "기타", "O":   "기타",
    "P":   "기타", "Q":   "헬스케어", "R":   "기타",
    "S":   "기타", "T":   "기타", "U":   "기타",
}

# KRX 업종명(한국어) → sector group (fallback when DART returns Korean name)
KRX_TO_SECTOR: Dict[str, str] = {
    # IT / Semiconductor
    "전기전자": "IT/반도체",
    "반도체": "IT/반도체",
    "소프트웨어": "IT/반도체",
    "IT서비스": "IT/반도체",
    "디스플레이": "IT/반도체",
    "인터넷": "IT/반도체",
    # Healthcare
    "의약품": "헬스케어",
    "의료정밀": "헬스케어",
    "바이오": "헬스케어",
    "제약": "헬스케어",
    # Consumer Discretionary
    "운수장비": "경기소비재",
    "자동차부품": "경기소비재",
    "유통업": "경기소비재",
    "섬유의복": "경기소비재",
    "오락문화": "경기소비재",
    # Consumer Staples
    "음식료품": "필수소비재",
    "음식료": "필수소비재",
    # Materials
    "화학": "소재",
    "철강금속": "소재",
    "비금속광물": "소재",
    "종이목재": "소재",
    "고무": "소재",
    # Industrials
    "기계": "산업재",
    "건설업": "산업재",
    "운수창고업": "산업재",
    "항공": "산업재",
    "조선": "산업재",
    # Financials
    "금융업": "금융",
    "은행": "금융",
    "증권": "금융",
    "보험": "금융",
    "기타금융": "금융",
    # Communication
    "통신업": "커뮤니케이션",
    "방송서비스": "커뮤니케이션",
    "미디어": "커뮤니케이션",
    # Energy
    "에너지": "에너지",
    "정유": "에너지",
    # Utilities
    "전기가스업": "유틸리티",
    # Real Estate
    "부동산": "부동산",
    # Other
    "서비스업": "기타",
    "제조업": "기타",
    "기타": "기타",
}

# ---------------------------------------------------------------------------
# DART API key (OpenDartReader) — for PBR/ROE factor data
# Set via environment variable:  export DART_API_KEY=your_key
# Or fill in directly below (keep quotes).
# ---------------------------------------------------------------------------
import os as _os
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()  # loads .env file automatically
except ImportError:
    pass
DART_API_KEY: str = _os.environ.get("DART_API_KEY", "")

# Output directory for charts
OUTPUT_DIR: str = "output"
