from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


IN_PATH_DEFAULT = Path("krx_data.xlsx")  # 기존에 만들어둔 최신 티커
OUT_DIR = Path("data")

OUT_FILTERED_CSV = OUT_DIR / "tickers_filtered.csv"
OUT_FILTERED_XLSX = OUT_DIR / "tickers_filtered.xlsx"
OUT_REMOVED_CSV = OUT_DIR / "tickers_removed.csv"


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}


def _read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    else:
        # csv 인코딩은 케이스가 많아서 utf-8-sig 우선, 실패하면 cp949 시도
        try:
            df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype=str, encoding="cp949")
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    columns를 code/name/market로 맞춤
    너의 파일이 어떤 헤더든 최대한 흡수하도록 후보들을 지원
    """
    col_map_candidates = [
        # (code, name, market)
        ("code", "name", "market"),
        ("종목코드", "종목명", "시장구분"),
        ("단축코드", "한글 종목명", "시장구분"),
        ("종목코드", "종목명", "시장"),
        ("Code", "Name", "Market"),
    ]

    cols = set(df.columns)
    for c_code, c_name, c_market in col_map_candidates:
        if c_code in cols and c_name in cols and c_market in cols:
            out = df[[c_code, c_name, c_market]].copy()
            out.columns = ["code", "name", "market"]
            return out

    # 최소한 code/market만 있는 경우도 허용
    if "code" in cols and "market" in cols:
        out = df[["code", "market"]].copy()
        out["name"] = df["name"] if "name" in cols else ""
        out = out[["code", "name", "market"]]
        return out

    raise RuntimeError(
        f"컬럼을 인식하지 못했습니다. 현재 컬럼: {list(df.columns)}\n"
        "필요 컬럼 후보: (종목코드/종목명/시장구분) 또는 (code/name/market)"
    )


def _safe_get(url: str, *, timeout: Tuple[int, int] = (5, 10), retries: int = 2) -> Optional[str]:
    last = None
    for i in range(retries):
        try:
            with requests.Session() as s:
                s.headers.update(HEADERS)
                r = s.get(url, timeout=timeout, allow_redirects=True)
                r.encoding = r.apparent_encoding or r.encoding
                return r.text
        except Exception as e:
            last = e
            if i == retries - 1:
                return None
            time.sleep(0.6 * (2 ** i) + random.uniform(0.0, 0.3))
    return None


def exists_on_naver_finance(code: str) -> bool:
    """
    네이버 금융 종목 존재 여부
    """
    url = f"https://finance.naver.com/item/main.naver?code={code}"
    html = _safe_get(url)
    if not html:
        # 네트워크 오류면 "존재하지 않는다"로 단정하지 말고 False 처리(removed로 보냄)
        return False

    if "존재하지 않는 종목" in html:
        return False

    # 페이지에 종목명 영역이 보통 있으므로 약한 조건 추가
    # (너무 빡세게 하면 오탐이 생길 수 있어 약하게만)
    if "종목명" in html or "시세" in html or "현재가" in html:
        return True

    # 애매하면 False
    return False


def _check_one(row) -> Tuple[str, bool]:
    code = str(row["code"]).zfill(6)
    ok = exists_on_naver_finance(code)
    return code, ok


def main(in_path: str = str(IN_PATH_DEFAULT), workers: int = 8) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    src = Path(in_path)
    df = _read_any(src)
    df = _normalize_columns(df)

    # code 정규화
    df["code"] = df["code"].astype(str).str.strip().str.zfill(6)
    df["market"] = df["market"].astype(str).str.strip()

    # ✅ KOSPI/KOSDAQ만 남김 (KONEX 등 제거)
    df = df[df["market"].isin(["KOSPI", "KOSDAQ"])].copy()
    df = df.drop_duplicates(subset=["code"]).reset_index(drop=True)

    print(f"INPUT rows: {len(df)} markets: {df['market'].value_counts().to_dict()}")

    # ✅ 존재 여부 병렬 체크
    ok_map = {}
    removed = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_check_one, row): idx for idx, row in df.iterrows()}

        done = 0
        total = len(futures)

        for fut in as_completed(futures):
            done += 1
            code, ok = fut.result()
            ok_map[code] = ok

            if done % 50 == 0 or done == total:
                print(f"CHECK {done}/{total}")

    df["naver_exists"] = df["code"].map(ok_map).fillna(False)

    df_ok = df[df["naver_exists"] == True].drop(columns=["naver_exists"]).copy()
    df_bad = df[df["naver_exists"] == False].drop(columns=["naver_exists"]).copy()

    # ✅ 저장
    df_ok.to_csv(OUT_FILTERED_CSV, index=False, encoding="utf-8-sig")
    df_ok.to_excel(OUT_FILTERED_XLSX, index=False, engine="openpyxl")
    df_bad.to_csv(OUT_REMOVED_CSV, index=False, encoding="utf-8-sig")

    print(f"[SAVED] {OUT_FILTERED_CSV}")
    print(f"[SAVED] {OUT_FILTERED_XLSX}")
    print(f"[SAVED] {OUT_REMOVED_CSV}")
    print(f"OK rows: {len(df_ok)} / REMOVED rows: {len(df_bad)}")


if __name__ == "__main__":
    main()
