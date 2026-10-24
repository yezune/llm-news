"""'7B', '15T', '128K', '3.8e25', '3.8×10^25' 같은 수량 표기를 숫자로 바꾼다."""

from __future__ import annotations

import re
from decimal import Decimal

SUFFIX = {
    "k": 10**3,
    "thousand": 10**3,
    "m": 10**6,
    "million": 10**6,
    "b": 10**9,
    "billion": 10**9,
    "t": 10**12,
    "trillion": 10**12,
}

_NUM = r"(\d+(?:[.,]\d+)*)"
_PATTERN = re.compile(
    rf"^\s*{_NUM}\s*(?:(?:[x×*]\s*10\s*\^\s*|e)([+-]?\d+)|\s*(k|m|b|t|thousand|million|billion|trillion))?\s*$",
    re.IGNORECASE,
)


def parse_quantity(text: str | int | float) -> float:
    if isinstance(text, (int, float)):
        return float(text)
    m = _PATTERN.match(text)
    if not m:
        raise ValueError(f"수량을 해석할 수 없음: {text!r}")
    num, exp, suffix = m.groups()
    value = float(num.replace(",", ""))
    if exp is not None:
        value *= 10 ** int(exp)
    if suffix:
        value *= SUFFIX[suffix.lower()]
    return value


def rounding_tolerance(text: str | int | float) -> float:
    """표기의 유효숫자가 허용하는 상대 오차. '7B'는 6.5B~7.5B, '6.74B'는 ±0.005B.

    이름에 붙은 크기 표기('7B')는 반올림 값이므로, 이 구간 안이면 같은 주장으로 본다.
    """
    if isinstance(text, (int, float)):
        return 0.0
    m = _PATTERN.match(text)
    if not m:
        return 0.0
    num = m.group(1).replace(",", "")
    d = Decimal(num)
    exponent = d.as_tuple().exponent
    assert isinstance(exponent, int)
    half_unit = Decimal(5) * Decimal(10) ** (exponent - 1)
    return float(half_unit / d) if d else 0.0


def human(n: float) -> str:
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        # 999.9M 처럼 반올림하면 윗단위가 되는 값은 윗단위로 표기 ('1e+03M' 방지)
        if abs(n) >= scale * 0.9995:
            return f"{n / scale:.3g}{unit}"
    return f"{n:.3g}"
