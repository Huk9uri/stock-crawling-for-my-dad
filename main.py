from __future__ import annotations

import random
import re
import time
from pathlib import Path

import pandas as pd

from crawler.financials import get_financials_yq, normalize_period_columns


# ===== 경로 =====
DATA_DIR = Path("data")
OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS_CSV = DATA_DIR / "tickers_filtered.csv"  # 필요시 파일명 변경
OUT_XLSX = OUT_DIR / "financials_result.xlsx"

# ✅ 항목(카테고리) 고정 순서
ITEM_ORDER = [
    "매출액",
    "영업이익",
    "영업이익(발표기준)",
    "세전계속사업이익",
    "당기순이익",
    "당기순이익(지배)",
    "당기순이익(비지배)",
    "자산총계",
    "부채총계",
    "자본총계",
    "자본총계(지배)",
    "자본총계(비지배)",
    "자본금",
    "영업활동현금흐름",
    "투자활동현금흐름",
    "재무활동현금흐름",
    "CAPEX",
    "FCF",
    "이자발생부채",
    "영업이익률",
    "순이익률",
    "ROE(%)",
    "ROA(%)",
    "부채비율",
    "자본유보율",
    "EPS(원)",
    "PER(배)",
    "BPS(원)",
    "PBR(배)",
    "현금DPS(원)",
    "현금배당수익률",
    "현금배당성향(%)",
    "발행주식수(보통주)",
]
ITEM_RANK = {name: i for i, name in enumerate(ITEM_ORDER)}


def read_tickers_csv(path: Path) -> pd.DataFrame:
    """
    CSV에서 종목코드/종목명/시장구분만 읽어서 표준 컬럼(code,name,market)으로 맞춤
    - utf-8-sig 기준
    """
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")

    rename_map = {}
    if "종목코드" in df.columns and "code" not in df.columns:
        rename_map["종목코드"] = "code"
    if "종목명" in df.columns and "name" not in df.columns:
        rename_map["종목명"] = "name"
    if "시장구분" in df.columns and "market" not in df.columns:
        rename_map["시장구분"] = "market"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "name" not in df.columns:
        df["name"] = ""
    if "market" not in df.columns:
        df["market"] = ""

    df["code"] = df["code"].astype(str).str.strip().str.zfill(6)
    df["name"] = df["name"].astype(str).str.strip()
    df["market"] = df["market"].astype(str).str.strip()

    df = df.drop_duplicates(subset=["code"]).sort_values(["market", "code"]).reset_index(drop=True)
    return df[["code", "name", "market"]]


def _period_key(p: str):
    """
    기간 정렬용 키
    - Y: 2020, 2025E
    - Q: 2024Q3, 2026Q2E
    """
    s = str(p)

    m = re.match(r"^(\d{4})Q([1-4])(E)?$", s)
    if m:
        y = int(m.group(1))
        q = int(m.group(2))
        e = 1 if m.group(3) else 0
        return (y, 1, q, e)

    m = re.match(r"^(\d{4})(E)?$", s)
    if m:
        y = int(m.group(1))
        e = 1 if m.group(2) else 0
        return (y, 0, 0, e)

    return (9999, 9, 9, 9)


def sort_columns_fixed_item_order(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    컬럼 포맷: {항목}__{Y/Q}__{기간}
    정렬: (고정 항목 순서) -> Y(연간) -> Q(분기) -> 기간 오름차순
    """
    meta_cols = ["종목코드", "종목명", "시장구분"]
    meta_cols = [c for c in meta_cols if c in result_df.columns]

    data_cols = [c for c in result_df.columns if c not in meta_cols]

    def col_key(c: str):
        parts = c.split("__")
        if len(parts) >= 3:
            item, freq, period = parts[0], parts[1], parts[2]
            item_rank = ITEM_RANK.get(item, len(ITEM_ORDER) + 999)
            freq_rank = 0 if freq == "Y" else 1  # ✅ Y 먼저, Q 나중
            return (item_rank, item, freq_rank, _period_key(period), c)

        return (len(ITEM_ORDER) + 999, "~~~~", 9, (9999, 9, 9, 9), c)

    data_cols_sorted = sorted(data_cols, key=col_key)
    return result_df[meta_cols + data_cols_sorted]


def to_one_row_grouped_keepna(df: pd.DataFrame, *, freq: str) -> dict:
    """
    df(index=항목, columns=기간) -> dict
    컬럼명 규칙: {항목}__{freq}__{기간}
    ✅ 모든 셀을 포함(값이 NaN이어도 key는 생성) => 엑셀에서 빈칸 유지
    """
    out: dict = {}

    # 안전한 문자열화
    idx = [str(x).strip() for x in df.index.tolist()]
    cols = [str(x).strip() for x in df.columns.tolist()]

    # 빠른 접근을 위해 values 사용
    values = df.values

    for i_item, item in enumerate(idx):
        for j_col, period in enumerate(cols):
            key = f"{item}__{freq}__{period}"
            out[key] = values[i_item, j_col]

    return out


def main():
    tickers = read_tickers_csv(TICKERS_CSV)
    print("tickers rows:", len(tickers))

    tickers = tickers[tickers["market"].isin(["KOSPI", "KOSDAQ"])].reset_index(drop=True)
    print("filtered rows:", len(tickers), "markets:", tickers["market"].value_counts().to_dict())

    total = len(tickers)
    rows: list[dict] = []
    fails: list[dict] = []

    # 종목당 예산(초)
    BUDGET_SEC = 3.0

    for i, r in tickers.iterrows():
        code = r["code"]
        name = r["name"]
        market = r["market"]

        print(f"START[{i+1}/{total}] {code} {name} {market}")

        start = time.monotonic()
        deadline = start + BUDGET_SEC

        try:
            # ✅ 연간/분기 가져오기 (deadline 적용)
            y_df, q_df = get_financials_yq(code, fin_typ=0, debug=False, deadline=deadline)

            # ✅ 결산월 차이 제거
            y_df = normalize_period_columns(y_df, "Y")
            q_df = normalize_period_columns(q_df, "Q")

            # ✅ 1행 dict (빈칸/NaN 컬럼 유지)
            y_map = to_one_row_grouped_keepna(y_df, freq="Y")
            q_map = to_one_row_grouped_keepna(q_df, freq="Q")

            row = {
                "종목코드": code,
                "종목명": name,
                "시장구분": market,
                **y_map,
                **q_map,
            }

            rows.append(row)
            print(f"OK  [{i+1}/{total}] {code} {name} {market}")

        except TimeoutError as e:
            fails.append({"종목코드": code, "종목명": name, "시장구분": market, "error": f"timeout>{BUDGET_SEC}s"})
            print(f"FAIL[{i+1}/{total}] {code} -> timeout>{BUDGET_SEC}s")

        except Exception as e:
            fails.append({"종목코드": code, "종목명": name, "시장구분": market, "error": str(e)})
            print(f"FAIL[{i+1}/{total}] {code} -> {e}")

        # ✅ 차단 방지 텀(조금 더 빠르게)
        time.sleep(0.22 + random.uniform(0.03, 0.25))

    # ✅ 루프 끝나고 1번만 저장
    result_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    fail_df = pd.DataFrame(fails) if fails else pd.DataFrame()

    # ✅ 원하는 컬럼 정렬: 고정 카테고리 순서 + (Y -> Q)
    if not result_df.empty:
        result_df = sort_columns_fixed_item_order(result_df)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="RESULT", index=False)
        fail_df.to_excel(writer, sheet_name="FAIL", index=False)

    print("[SAVED]", OUT_XLSX.resolve())
    print("success:", len(rows), "fail:", len(fails))


if __name__ == "__main__":
    main()
