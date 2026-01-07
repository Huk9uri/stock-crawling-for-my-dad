from __future__ import annotations

from io import StringIO
import random
import re
import time
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter


# =========================
# Session (차단 대응)
# =========================
_SESSION: requests.Session | None = None


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    s = requests.Session()

    # ⚠️ urllib3 Retry는 여기서 쓰지 않음(중복 재시도로 전체가 오래 걸림)
    # 우리가 deadline 기반으로 직접 제어한다.
    adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    _SESSION = s
    return s


def _headers(referer: str | None = None) -> dict:
    h = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.7,en;q=0.6",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    if referer:
        h["Referer"] = referer
    return h


def _time_left(deadline: float | None) -> float:
    if deadline is None:
        return 999999.0
    return deadline - time.monotonic()


def _sleep_with_deadline(sec: float, deadline: float | None) -> None:
    if deadline is None:
        time.sleep(sec)
        return
    left = _time_left(deadline)
    if left <= 0:
        return
    time.sleep(min(sec, max(0.0, left)))


def _fetch_text(
    url: str,
    referer: str | None = None,
    timeout: float = 10.0,
    deadline: float | None = None,
) -> str:
    """
    - deadline(종목 전체 예산)을 넘기지 않도록 제어
    - ConnectionReset/차단 등은 "짧게" 재시도하되, 남은 시간 안에서만 수행
    """
    s = _get_session()

    # 워밍업: finance.naver.com 한번 찔러서 쿠키/세션 준비(차단 완화)
    if not hasattr(s, "_warmed"):
        try:
            left = _time_left(deadline)
            if left > 0.2:
                s.get(
                    "https://finance.naver.com/",
                    headers=_headers(),
                    timeout=min(2.5, left),
                )
        except Exception:
            pass
        setattr(s, "_warmed", True)

    last_exc: Exception | None = None

    # ✅ deadline이 있으면 재시도 횟수도 줄임(전체 3초 예산 보호)
    max_attempt = 3 if deadline is not None else 8

    for attempt in range(1, max_attempt + 1):
        left = _time_left(deadline)
        if left <= 0:
            raise TimeoutError(f"deadline exceeded while fetching: {url}")

        try:
            # 요청 timeout도 남은 시간에 맞춰 축소
            req_timeout = min(timeout, max(0.2, left))
            r = s.get(url, headers=_headers(referer), timeout=req_timeout)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            return r.text

        except requests.exceptions.RequestException as e:
            last_exc = e

            left2 = _time_left(deadline)
            if left2 <= 0:
                raise TimeoutError(f"deadline exceeded while retrying: {url}") from e

            # ✅ 짧은 백오프(남은 시간 안에서만)
            # attempt=1: 0.15~0.35, attempt=2: 0.25~0.55, attempt=3: 0.35~0.75
            backoff = 0.10 + attempt * 0.15 + random.uniform(0.05, 0.25)
            _sleep_with_deadline(backoff, deadline)

    raise last_exc if last_exc else RuntimeError("요청 실패")


def _save_debug_html(debug_dir: Path, code: str, name: str, html: str) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / f"{code}_{name}.html").write_text(html, encoding="utf-8", errors="ignore")


# =========================
# encparam/id 추출
# =========================
def _find_encparam_id_pairs_in_html(html: str) -> tuple[str, str] | None:
    enc_patterns = [
        r"encparam\s*:\s*'([^']+)'",
        r'encparam\s*:\s*"([^"]+)"',
        r"encparam\s*[:=]\s*[\"']([^\"']+)[\"']",
    ]
    id_patterns = [
        r"\bid\s*:\s*'([a-zA-Z0-9]+)'",
        r'\bid\s*:\s*"([a-zA-Z0-9]+)"',
        r"\bid\s*[:=]\s*[\"']([a-zA-Z0-9]+)[\"']",
    ]

    enc = None
    _id = None

    for pat in enc_patterns:
        m = re.search(pat, html, flags=re.I)
        if m:
            enc = m.group(1)
            break

    for pat in id_patterns:
        m = re.search(pat, html, flags=re.I)
        if m:
            _id = m.group(1)
            break

    if enc and _id:
        return unquote(enc), unquote(_id)

    return None


def _get_encparam_and_id(
    code: str,
    debug: bool = False,
    debug_dir: Path | None = None,
    deadline: float | None = None,
) -> tuple[str, str, str]:
    """
    return: (encparam, id, referer_url)
    """
    code = str(code).zfill(6)
    debug_dir = debug_dir or (Path("output") / "debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    c101_candidates = [
        f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={code}&target=finsum_more",
        f"https://companyinfo.stock.naver.com/v1/company/c1010001.aspx?cmp_cd={code}&target=finsum_more",
        f"https://companyinfo.stock.naver.com/v1/company/c1010001.aspx?cmp_cd={code}",
    ]

    for idx, url in enumerate(c101_candidates):
        # deadline 체크
        if _time_left(deadline) <= 0:
            raise TimeoutError("deadline exceeded while getting encparam/id")

        try:
            html = _fetch_text(url, referer="https://finance.naver.com/", timeout=6.0, deadline=deadline)
        except Exception as e:
            if debug:
                print(f"DEBUG[{code}] c101 cand[{idx}] fetch fail -> {e}")
            continue

        pair = _find_encparam_id_pairs_in_html(html)
        if pair:
            enc, _id = pair
            if debug:
                print(f"DEBUG[{code}] encparam/id from c101 cand[{idx}] OK")
            return enc, _id, url

        if debug:
            _save_debug_html(debug_dir, code, f"c101_{idx}", html)

    raise RuntimeError("encparam/id를 찾지 못했습니다.")


# =========================
# 진짜 재무테이블 선별 (9393/4122 제거)
# =========================
def _pick_real_fin_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    KEYWORDS = [
        "매출액", "영업이익", "당기순이익", "EPS", "BPS", "PER", "PBR", "ROE", "부채비율",
        "영업이익률", "순이익률", "자산", "부채", "자본"
    ]

    def numeric_like_count(df: pd.DataFrame) -> int:
        pat = re.compile(r"^-?\d[\d,]*\.?\d*$")
        cnt = 0
        for v in df.astype(str).values.ravel():
            s = v.strip()
            if s == "" or s.lower() == "nan":
                continue
            if pat.match(s):
                cnt += 1
        return cnt

    def looks_like_constant_fill(df: pd.DataFrame) -> bool:
        sample = df.head(10).astype(str).values.ravel().tolist()
        cleaned = [x.strip() for x in sample if x.strip() and x.strip().lower() != "nan"]
        return len(set(cleaned)) <= 2

    best = None
    best_score = -1

    for t in tables:
        if t is None or t.empty:
            continue
        if t.shape[0] < 5 or t.shape[1] < 3:
            continue
        if looks_like_constant_fill(t):
            continue

        first_col = t.iloc[:, 0].astype(str)
        kw_hits = sum(1 for kw in KEYWORDS if first_col.str.contains(kw, na=False).any())
        if kw_hits == 0:
            continue

        ncnt = numeric_like_count(t)
        if ncnt < 10:
            continue

        score = kw_hits * 10 + min(ncnt, 300)
        if score > best_score:
            best = t
            best_score = score

    if best is None:
        raise RuntimeError("재무 테이블을 찾지 못했습니다(값 있는 표가 없음).")

    return best


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join([str(x).strip() for x in col if str(x).strip().lower() != "nan" and str(x).strip() != ""])
            for col in out.columns
        ]
    else:
        out.columns = [str(c).strip() for c in out.columns]
    return out


# =========================
# cF1001(Y/Q) URL 빌드 + 호출
# =========================
def _build_cf1001_urls(code: str, fin_typ: int, freq: str, encparam: str, _id: str, referer_url: str) -> list[str]:
    qs = f"cmp_cd={code}&fin_typ={fin_typ}&freq_typ={freq}&encparam={encparam}&id={_id}"
    urls: list[str] = []

    if "wisereport.co.kr" in referer_url:
        urls.append(f"https://navercomp.wisereport.co.kr/v2/company/ajax/cF1001.aspx?{qs}")
        urls.append(f"https://navercomp.wisereport.co.kr/v1/company/ajax/cF1001.aspx?{qs}")

    urls.append(f"https://companyinfo.stock.naver.com/v1/company/ajax/cF1001.aspx?{qs}")
    return urls


def _fetch_fin_table(
    code: str,
    freq: str,
    fin_typ: int,
    encparam: str,
    _id: str,
    referer_url: str,
    debug: bool = False,
    deadline: float | None = None,
) -> pd.DataFrame:
    urls = _build_cf1001_urls(code, fin_typ, freq, encparam, _id, referer_url)
    last_exc: Exception | None = None

    for u in urls:
        if _time_left(deadline) <= 0:
            raise TimeoutError(f"deadline exceeded before fetching fin table (freq={freq})")

        try:
            html = _fetch_text(u, referer=referer_url, timeout=6.0, deadline=deadline)
            tables = pd.read_html(StringIO(html))
            if not tables:
                raise RuntimeError(f"read_html 결과가 비었습니다. URL={u}")

            raw = _pick_real_fin_table(tables)
            df = _flatten_columns(raw)

            idx_col = df.columns[0]
            df[idx_col] = df[idx_col].astype(str).str.strip()
            df = df.set_index(idx_col)

            df = df.dropna(axis=0, how="all")
            df = df.dropna(axis=1, how="all")

            non_null = int(df.notna().sum().sum())
            if non_null < 10:
                raise RuntimeError(f"재무표가 비어있습니다(freq={freq}, fin_typ={fin_typ}). URL={u}")

            if debug:
                print(f"DEBUG[{code}] freq={freq} fin_typ={fin_typ} OK shape={df.shape} non_null={non_null}")

            return df

        except Exception as e:
            last_exc = e
            if debug:
                print(f"DEBUG[{code}] freq={freq} fin_typ={fin_typ} URL fail -> {e}")

            # ✅ 너무 길게 쉬지 않음(남은 시간 안에서만)
            _sleep_with_deadline(0.25 + random.uniform(0.05, 0.35), deadline)

    raise last_exc if last_exc else RuntimeError("재무 테이블 호출 실패")


# =========================
# 외부 공개 함수
# =========================
def get_financials_yq(
    code: str,
    fin_typ: int = 0,
    debug: bool = False,
    deadline: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    연간(Y), 분기(Q) 재무테이블 반환
    - deadline이 주어지면 그 시간 안에서만 시도하고 초과 시 TimeoutError
    - fin_typ=0(연결) 실패하면 fin_typ=1(별도) fallback
    """
    code = str(code).zfill(6)
    encparam, _id, referer_url = _get_encparam_and_id(code, debug=debug, deadline=deadline)

    last_exc = None

    for ft in [fin_typ, 1 if fin_typ == 0 else 0]:
        if _time_left(deadline) <= 0:
            raise TimeoutError("deadline exceeded before Y/Q fetch")

        try:
            y_df = _fetch_fin_table(code, "Y", ft, encparam, _id, referer_url, debug=debug, deadline=deadline)
            q_df = _fetch_fin_table(code, "Q", ft, encparam, _id, referer_url, debug=debug, deadline=deadline)
            return y_df, q_df
        except Exception as e:
            last_exc = e
            if debug:
                print(f"DEBUG[{code}] fin_typ={ft} failed -> {e}")

    raise last_exc if last_exc else RuntimeError("Y/Q 재무테이블을 가져오지 못했습니다.")


# =========================
# Period normalize
# =========================
_DATE_RE = re.compile(r"(\d{4})/(\d{2})")


def normalize_period_columns(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    - freq="Y": 2024/03, 2024/12(E) -> 2024, 2024E
    - freq="Q": 2024/06 -> 2024Q2
    """
    out = df.copy()
    new_cols = {}

    for c in out.columns:
        s = str(c)
        m = _DATE_RE.search(s)
        if not m:
            continue
        year = int(m.group(1))
        month = int(m.group(2))
        is_est = "(E" in s or "E)" in s or "(E)" in s

        if freq.upper() == "Y":
            key = f"{year}{'E' if is_est else ''}"
        else:
            q = (month - 1) // 3 + 1
            key = f"{year}Q{q}{'E' if is_est else ''}"

        new_cols[c] = key

    out = out.rename(columns=new_cols)

    # 같은 이름이 생기면 bfill로 합치기
    if out.columns.duplicated().any():
        cols = list(out.columns)
        uniq = []
        for col in cols:
            if col in uniq:
                continue
            uniq.append(col)

        merged = {}
        for col in uniq:
            same = out.loc[:, out.columns == col]
            if same.shape[1] == 1:
                merged[col] = same.iloc[:, 0]
            else:
                merged[col] = same.bfill(axis=1).iloc[:, 0]

        out = pd.DataFrame(merged, index=out.index)

    return out
