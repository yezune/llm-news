# llm-news — 검증하는 LLM 데일리

매일 한 번 LLM 관련 최신 소식(논문 · 새 모델 · 연구소 블로그)을 큐레이션하고,
소식에 담긴 **주장(claim)을 하나씩 검증**해서 판정과 증거를 함께 기록하는 저장소입니다.

<!-- LATEST:START -->
**최신 다이제스트: [2026-11-14](digests/2026/2026-11-14.md)** — 소식 19건 · ✅ 검증됨 5 · ❌ 반박됨 1 · ⚪ 검증 불가(자기보고) 4

최근 14일:

- [2026-11-14](digests/2026/2026-11-14.md)
- [2026-11-13](digests/2026/2026-11-13.md)
- [2026-11-12](digests/2026/2026-11-12.md)
- [2026-11-11](digests/2026/2026-11-11.md)
- [2026-11-09](digests/2026/2026-11-09.md)
- [2026-11-08](digests/2026/2026-11-08.md)
- [2026-11-07](digests/2026/2026-11-07.md)
- [2026-11-06](digests/2026/2026-11-06.md)
- [2026-11-05](digests/2026/2026-11-05.md)
- [2026-11-04](digests/2026/2026-11-04.md)
- [2026-11-03](digests/2026/2026-11-03.md)
- [2026-11-02](digests/2026/2026-11-02.md)
- [2026-11-01](digests/2026/2026-11-01.md)
- [2026-10-31](digests/2026/2026-10-31.md)
<!-- LATEST:END -->

<!-- REPORTS:START -->
**기간 리포트:** 주간 [2026-W45](reports/weekly/2026/2026-W45.md) · 월간 [2026-10](reports/monthly/2026/2026-10.md) · 연간 [2026](reports/yearly/2026.md) ⏳
<!-- REPORTS:END -->

## 무엇을 어떻게 검증하나

| 주장 종류 | 계층 | 방법 | 예 |
|---|---|---|---|
| 수학 (항등식·부등식·값) | **A. 내부** | sympy 기호 증명 + 시드 고정 반례 탐색 | `softmax(x+c) = softmax(x)` |
| 알고리즘 성질 | **A. 내부** | 소규모 재현 구현 ↔ 정의대로 계산하는 기준 구현 차분 테스트 | FlashAttention 타일링 = 정확한 어텐션 |
| 훈련 연산량 · 메모리 산술 | **A. 내부** | 6ND, KV 캐시 공식 계산 | "8B × 15T 토큰 = 7.2e23 FLOPs" |
| 모델 크기 · 활성 파라미터 · 컨텍스트 | **B. 원본** | safetensors 실측 + 샤드 헤더(Range) 텐서 집계 + config.json 독립 계산 교차 검증 | "Qwen3-30B-A3B는 활성 3B" |
| 훈련 토큰 수 | **B. 원본** | 원본 모델 카드 선언과 대조 (재훈련 불가 → `attested`까지만) | "15T 토큰으로 훈련" |
| 데이터셋 | **B. 원본** | HF 데이터셋 뷰어가 실제 parquet에서 센 행 수 · 바이트 | "Alpaca는 52K 예제" |
| 벤치마크 점수 등 | C. 자기보고 | 근거 없음 → `unverifiable`로 명시 | "MMLU 85%" |

대규모 훈련 자체는 재현하지 않습니다. 대신 **공개된 원본 아티팩트(Hugging Face의 가중치·설정·데이터)**와
주장을 대조하고, 모든 증거를 조회한 저장소 커밋 sha로 고정해 나중에 같은 데이터로 다시 검증할 수 있게 합니다.
자세한 판정 기준은 [METHODOLOGY.md](METHODOLOGY.md)를 보세요.

판정: ✅ `verified` · ❌ `refuted` · 📄 `attested`(원본 선언과 일치, 독립 검증 아님) · ❔ `inconclusive` · ⚪ `unverifiable`

## 빠른 시작

```bash
uv sync --extra dev               # Python 3.11+, 의존성은 uv.lock으로 고정
uv run pytest                     # 오프라인 테스트 (수학·알고리즘 대장 전체 재검증 포함)
uv run llm-news daily             # 오늘(Asia/Seoul) 큐레이션 + 검증 → digests/, data/
uv run llm-news daily --date 2026-10-23 --force   # 특정 날짜 재생성

uv run llm-news report week       # 이번 주(진행 중) 주간 리포트 미리 보기 → reports/
uv run llm-news report month --period 2026-10     # 특정 월 · year --period 2026 연간
```

단건 검증 도구:

```bash
# 모델 주장을 HF 원본과 대조 (총/활성 파라미터, 컨텍스트, 훈련 토큰)
uv run llm-news verify-model Qwen/Qwen3-30B-A3B --params 30B --active 3B
uv run llm-news verify-model HuggingFaceTB/SmolLM3-3B --context 64K --tokens 11T

# 데이터셋 주장을 실제 parquet 집계와 대조
uv run llm-news verify-dataset tatsu-lab/alpaca --rows 52K

# 수학 주장: 기호 증명 + 반례 탐색
uv run llm-news verify-math --lhs "log(1+x)" --op "<=" --rhs "x" --var x=positive

# claims/ 대장 전체 검증 (기대 판정과 다르면 종료 코드 1)
uv run llm-news ledger --check
uv run llm-news algorithms        # 등록된 알고리즘 재현 목록
```

`HF_TOKEN` 환경 변수가 있으면 gated 모델(config·카드)도 대조할 수 있습니다.

## 하루 1회 실행

- **GitHub Actions** (`.github/workflows/daily.yml`): 매일 22:17 UTC(07:17 KST)에
  오프라인 테스트 → 큐레이션·검증 → 대장 드리프트 확인 → 결과 커밋을 수행합니다.
  저장소 Secrets에 `HF_TOKEN`을 넣으면 gated 모델까지 확인합니다. 수동 실행(`workflow_dispatch`)도 됩니다.
- **로컬**: `scripts/daily.sh`를 cron이나 launchd에 등록합니다.
- 같은 날짜 다이제스트가 이미 있으면 파이프라인이 스스로 건너뛰므로, 재시도해도 하루 1건만 생깁니다.

## 주간 · 월간 · 연간 리포트

일일 실행이 끝날 때마다 **이미 끝난 기간**(ISO 주 월~일, 달력 월, 연도)의 리포트를 확정합니다.
월요일 실행이 지난주를, 매월 1일 실행이 지난달을, 1월 1일 실행이 지난해를 확정하고,
실행을 놓친 날이 있어도 다음 실행에서 따라잡습니다. 리포트는 `data/`의 일일 기록만 다시 모으므로
새로 수집하거나 재검증하지 않고, 판정과 증거(커밋 sha)는 그날 기록 그대로입니다.

- 📈 흐름: 주간은 일별, 월간은 주별, 연간은 월별 소식·주장·판정 수
- 🏆 주요 소식: 섹션별 점수 상위 (주 5 · 월 10 · 년 20건)
- ❌ 반박된 주장 전체, 🔬 검증 계층별 판정, 🧩 많이 다룬 주제, 🔗 여러 소식에 연결된 HF 저장소
- 🧮 검증 대장 드리프트(기대 판정에서 벗어난 명세와 그 기간), ⚠️ 출처별 수집 오류 일수

진행 중인 기간은 `llm-news report week|month|year`로 미리 볼 수 있습니다(서버에서는 Actions의 `period-reports` 수동 실행). 이렇게 만든 리포트에는 ⏳ 표시가 붙고,
기간이 끝나면 일일 실행이 확정본으로 다시 만듭니다. 과거 날짜를 `daily --force`로 재생성했다면
`report <종류> --period <기간>`으로 해당 리포트도 다시 만드세요.

## 저장소 구조

```
config/sources.yaml        출처·가중치·키워드·섹션별 상한 (코드 수정 없이 큐레이션 조정)
src/llm_news/
  sources/                 HF Daily Papers, arXiv, HF 트렌딩 모델, 연구소 RSS
  curate.py                관련도 필터 · 중복 병합 · 점수 · 출처별 상한
  extract.py               규칙 기반 주장 추출 + 원본 아티팩트 연결
  verify/
    symbolic.py            Tier A 수학: sympy 증명 + 반례 탐색
    algorithms.py          Tier A 알고리즘: 소규모 재현 차분 테스트
    compute.py             Tier A 훈련 산술: 6ND, KV 캐시
    hf.py                  Tier B 원본 대조: 모델·데이터셋
    params.py              config.json → 파라미터 수 독립 계산
    tensors.py             safetensors 헤더(Range)만 읽어 텐서 집계
  pipeline.py · cli.py     일일 실행 · 명령줄
  reports.py               주간·월간·연간 리포트 (data/ 집계)
claims/                    사람이 쓰는 검증 명세 (검증 대장)
reproductions/             새 논문 알고리즘 재현 플러그인
digests/YYYY/YYYY-MM-DD.md 일일 다이제스트 (사람용)
data/YYYY/YYYY-MM-DD.json  같은 내용의 구조화 기록 (주장·판정·증거·커밋 sha)
reports/weekly/YYYY/YYYY-Www.md · monthly/YYYY/YYYY-MM.md · yearly/YYYY.md   기간 리포트
state/seen.json            이미 실린 소식 (날짜 간 중복 방지)
```

## 기여: 주장 검증 추가하기

**1. 명세만 추가** — `claims/<분류>/<이름>.yaml`

```yaml
title: 논문 또는 소식 제목
source: https://arxiv.org/abs/XXXX.XXXXX
claims:
  - kind: math_identity            # math_inequality / math_value / algorithm / training_flops
    statement: 사람이 읽는 주장       # kv_cache_bytes / hf_model_params / hf_model_context
    params:                          # hf_training_tokens / hf_dataset
      lhs: exp(x + c) / (exp(x + c) + exp(y + c))
      rhs: exp(x) / (exp(x) + exp(y))
      variables: {x: real, y: real, c: real}
    expected: verified               # 회귀 기준 (선택)
```

**2. 새 알고리즘 재현** — `reproductions/<이름>.py`

```python
from llm_news.verify.algorithms import CheckResult, algorithm

@algorithm("my_trick_is_exact")
def my_trick_is_exact(rng, n=32):
    """논문의 트릭이 정의대로 계산한 값과 같다."""
    x = rng.standard_normal(n)
    err = float(abs(fast(x) - reference(x)).max())
    return CheckResult(err < 1e-10, err)
```

그다음 명세에서 `kind: algorithm`, `params: {name: my_trick_is_exact}`로 참조합니다.
실패하는 시행 번호와 시드가 증거에 남으므로 누구나 같은 반례를 재현할 수 있습니다.
