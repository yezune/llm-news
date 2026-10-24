"""llm-news 명령줄.

llm-news daily [--date YYYY-MM-DD] [--force]   일일 큐레이션+검증 (하루 1회)
llm-news ledger [--check]                      claims/ 명세만 검증 (CI용)
llm-news verify-model REPO --params 7B [--active 1B] [--context 128K] [--tokens 15T]
llm-news verify-math --lhs ... --rhs ... --var x=positive
llm-news algorithms                            등록된 알고리즘 재현 목록
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

from .models import Claim, Verdict
from .pipeline import Paths, run_daily, run_ledger
from .render import ICON, LABEL
from .verify import verify
from .verify.algorithms import ALGORITHMS, load_reproductions


def _print(v, as_json: bool) -> None:
    if as_json:
        print(json.dumps(asdict(v), ensure_ascii=False, indent=1, default=str))
    else:
        print(f"{ICON[v.verdict]} {LABEL[v.verdict]} [{v.method}/{v.tier.value}] {v.claim.statement}")
        print(f"   {v.summary}")
        for e in v.evidence:
            rev = f" @{e.revision}" if e.revision else ""
            print(f"   - {e.description}{rev} {e.url or ''}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="llm-news")
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("daily", help="일일 큐레이션 + 검증")
    d.add_argument("--date", type=date.fromisoformat)
    d.add_argument("--force", action="store_true", help="같은 날짜 다이제스트가 있어도 다시 생성")

    lg = sub.add_parser("ledger", help="claims/ 명세 검증")
    lg.add_argument("--check", action="store_true", help="기대 판정과 다르면 실패(종료 코드 1)")
    lg.add_argument("--json", action="store_true")

    vm = sub.add_parser("verify-model", help="HF 모델 주장을 원본과 대조")
    vm.add_argument("repo")
    vm.add_argument("--params")
    vm.add_argument("--active")
    vm.add_argument("--context")
    vm.add_argument("--tokens")
    vm.add_argument("--json", action="store_true")

    vd = sub.add_parser("verify-dataset", help="HF 데이터셋 주장을 원본과 대조")
    vd.add_argument("repo")
    vd.add_argument("--rows")
    vd.add_argument("--tokens")
    vd.add_argument("--config")
    vd.add_argument("--json", action="store_true")

    mm = sub.add_parser("verify-math", help="수학 항등식/부등식 검증")
    mm.add_argument("--lhs", required=True)
    mm.add_argument("--rhs", required=True)
    mm.add_argument("--op", default="==", choices=["==", "<=", "<", ">=", ">"])
    mm.add_argument("--var", action="append", default=[], help="이름=가정 (예: x=positive)")
    mm.add_argument("--json", action="store_true")

    sub.add_parser("algorithms", help="등록된 알고리즘 재현 목록")

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    paths = Paths(args.root)

    if args.cmd == "daily":
        out = run_daily(args.root, args.date, args.force)
        print(f"생성됨: {out}" if out else "오늘 다이제스트가 이미 있어 건너뜀 (--force로 재생성)")
        return 0

    if args.cmd == "ledger":
        results = run_ledger(paths)
        drifted = [r for r in results if not r.as_expected]
        for r in results:
            mark = "" if r.as_expected else f"  ⚠️ 기대 {r.expected}"
            if not args.json:
                print(f"{r.file}{mark}")
                _print(r.verification, False)
        if args.json:
            print(json.dumps([asdict(r.verification) for r in results], ensure_ascii=False, indent=1))
        if args.check and drifted:
            print(f"\n기대 판정과 다른 주장 {len(drifted)}건", file=sys.stderr)
            return 1
        return 0

    if args.cmd == "verify-model":
        claims = []
        if args.params:
            claims.append(
                Claim(
                    "hf_model_params",
                    f"{args.repo} 총 파라미터 {args.params}",
                    {"repo": args.repo, "value": args.params},
                )
            )
        if args.active:
            claims.append(
                Claim(
                    "hf_model_params",
                    f"{args.repo} 활성 파라미터 {args.active}",
                    {"repo": args.repo, "value": args.active, "scope": "active"},
                )
            )
        if args.context:
            claims.append(
                Claim(
                    "hf_model_context",
                    f"{args.repo} 컨텍스트 {args.context}",
                    {"repo": args.repo, "value": args.context},
                )
            )
        if args.tokens:
            claims.append(
                Claim(
                    "hf_training_tokens",
                    f"{args.repo} 훈련 토큰 {args.tokens}",
                    {"repo": args.repo, "value": args.tokens},
                )
            )
        if not claims:
            ap.error("--params/--active/--context/--tokens 중 하나 이상 필요")
        return _run_claims(claims, args.json)

    if args.cmd == "verify-dataset":
        params = {"repo": args.repo}
        for key in ("rows", "tokens", "config"):
            if getattr(args, key):
                params[key] = getattr(args, key)
        return _run_claims([Claim("hf_dataset", f"데이터셋 {args.repo}", params)], args.json)

    if args.cmd == "verify-math":
        variables = {k: a.split(",") for k, a in (v.split("=", 1) for v in args.var)}
        if args.op == "==":
            claim = Claim(
                "math_identity",
                f"{args.lhs} = {args.rhs}",
                {"lhs": args.lhs, "rhs": args.rhs, "variables": variables},
            )
        else:
            claim = Claim(
                "math_inequality",
                f"{args.lhs} {args.op} {args.rhs}",
                {"lhs": args.lhs, "rhs": args.rhs, "op": args.op, "variables": variables},
            )
        return _run_claims([claim], args.json)

    if args.cmd == "algorithms":
        if paths.reproductions.exists():
            load_reproductions(paths.reproductions)
        for name, (_, desc) in sorted(ALGORITHMS.items()):
            print(f"{name:34s} {desc}")
        return 0
    return 2


def _run_claims(claims: list[Claim], as_json: bool) -> int:
    results = [verify(c) for c in claims]
    for v in results:
        _print(v, as_json)
    return 1 if any(v.verdict == Verdict.REFUTED for v in results) else 0


if __name__ == "__main__":
    sys.exit(main())
