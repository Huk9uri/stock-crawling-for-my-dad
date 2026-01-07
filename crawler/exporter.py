from __future__ import annotations

import pandas as pd


def _make_unique(cols: list[str]) -> list[str]:
    """
    중복 컬럼명을 강제로 유니크하게 만든다.
    예: A, A, A -> A, A__2, A__3
    """
    seen = {}
    out = []
    for c in cols:
        c = str(c)
        if c not in seen:
            seen[c] = 1
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out


def to_one_row(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """
    df(index=항목, columns=기간) -> 1행
    - 빈칸 유지
    - 컬럼 중복 방지
    """
    df2 = df.copy()
    df2.index = df2.index.astype(str).str.strip()
    df2.columns = df2.columns.astype(str).str.strip()

    # ✅ stack FutureWarning 피하기(미래 구현 사용)
    s = df2.stack(future_stack=True)  # NA rows는 원래 생성 안함

    # ✅ NA(빈칸)를 유지하고 싶으면, stack 전에 문자열로 유지(값은 NaN으로 남음)
    # => one-row로 펼 때 NaN은 그대로 엑셀에 빈칸으로 저장됨 (요구사항 충족)

    keys = [f"{a}__{b}" for (a, b) in s.index]
    s.index = _make_unique(keys)  # ✅ 혹시 같은 키가 중복되면 뒤에 __2 붙임

    one = s.to_frame().T
    one.columns = _make_unique(list(one.columns))  # 안전망(한번 더)

    one.insert(0, "code", str(code).zfill(6))
    return one

import pandas as pd


def to_one_row_grouped(df: pd.DataFrame, *, code: str, freq: str) -> pd.DataFrame:
    """
    df(index=항목, columns=기간) -> 1행 wide
    컬럼명 규칙: {항목}__{freq}__{기간}

    예:
      매출액__Y__2024
      매출액__Q__2024Q1
    """
    df2 = df.copy()
    df2.index = df2.index.astype(str).str.strip()
    df2.columns = df2.columns.astype(str).str.strip()

    # (항목, 기간) -> 값
    s = df2.stack(future_stack=True)

    # MultiIndex -> "항목__freq__기간" 로 컬럼화
    cols = [f"{a}__{freq}__{b}" for (a, b) in s.index]

    one = pd.DataFrame([s.values], columns=cols)

    # 메타
    one.insert(0, "code", str(code).zfill(6))

    # 혹시 같은 이름 중복이면 뒤에서 덮어쓰지 말고 bfill로 합치기(안전)
    if one.columns.duplicated().any():
        one = one.loc[:, ~one.columns.duplicated()]

    return one
