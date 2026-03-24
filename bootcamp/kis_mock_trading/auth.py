# KIS 접근 토큰 발급 및 파일 캐싱
# 토큰은 24시간 유효하며, 만료 1시간 전부터 자동 갱신

import json
import os
import requests
from datetime import datetime, timedelta

import config

# 토큰 캐시 파일 경로 (프로젝트 루트에 저장)
_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".token_cache")


def get_token():
    """유효한 접근 토큰 반환 (캐시에서 읽거나 새로 발급)"""
    cached = _load_cached_token()
    if cached:
        return cached
    return _issue_token()


def _load_cached_token():
    """캐시 파일에서 토큰 읽기, 만료 1시간 전부터 갱신"""
    if not os.path.exists(_CACHE_FILE):
        return None
    try:
        with open(_CACHE_FILE, "r") as f:
            data = json.load(f)
        expires_at = datetime.fromisoformat(data["expires_at"])
        # 만료 1시간 전이면 재발급 (여유 있게)
        if datetime.now() < expires_at - timedelta(hours=1):
            return data["token"]
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


def _issue_token():
    """KIS OAuth2 접근 토큰 신규 발급"""
    url = f"{config.BASE_URL}/oauth2/tokenP"
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": config.APP_KEY,
        "appsecret": config.APP_SECRET,
    }

    resp = requests.post(url, headers=headers, json=body)
    if resp.status_code != 200:
        raise Exception(f"토큰 발급 실패 ({resp.status_code}): {resp.text}")

    token = resp.json().get("access_token")
    if not token:
        raise Exception(f"토큰 발급 응답 오류: {resp.text}")

    _save_token(token)
    return token


def _save_token(token):
    """토큰을 캐시 파일에 저장 (24시간 유효)"""
    expires_at = datetime.now() + timedelta(hours=24)
    with open(_CACHE_FILE, "w") as f:
        json.dump({"token": token, "expires_at": expires_at.isoformat()}, f)


if __name__ == "__main__":
    token = get_token()
    print(f"액세스 토큰: {token[:20]}...")  # 앞 20자만 출력
