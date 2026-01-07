from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import pandas as pd
from pykrx import stock  # 스크래핑 기반 라이브러리이므로 과도 호출은 자제 권장 :contentReference[oaicite:1]{index=1}


@dataclass(frozen=True)
class TickerPaths:
    out_dir: Path = Path("data")
    latest_csv: Path = Path("data/tickers_latest.csv")
    latest_xlsx: Path = Path("data/tickers_latest.xlsx")


def fetch_kospi_kosdaq_tickers() -> pd.DataFrame:
    # date 미지정 시 가장 최근 영업일 기준 목록 :contentReference[oaicite:2]{index=2}
    kospi = stock.get_market_ticker_list(market="KOSPI")
    kosdaq = stock.get_market_ticker_list(market="KOSDAQ")

    df1 = pd.DataFrame({"code": kospi, "market": "KOSPI"})
    df2 = pd.DataFrame({"code": kosdaq, "market": "KOSDAQ"})
    df = pd.concat([df1, df2], ignore_index=True)

    # 혹시 모를 중복 제거 + 정렬
    df = df.drop_duplicates(subset=["code"]).sort_values(["market", "code"]).reset_index(drop=True)
    return df


def save_tickers(df: pd.DataFrame, paths: TickerPaths = TickerPaths()) -> tuple[Path, Path]:
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    # “오늘 갱신본”도 남기고 싶으면 날짜 파일도 같이 저장
    today = datetime.now().strftime("%Y%m%d")
    dated_csv = paths.out_dir / f"tickers_{today}.csv"

    df.to_csv(paths.latest_csv, index=False, encoding="utf-8-sig")
    df.to_excel(paths.latest_xlsx, index=False, engine="openpyxl")
    df.to_csv(dated_csv, index=False, encoding="utf-8-sig")

    return paths.latest_csv, paths.latest_xlsx
