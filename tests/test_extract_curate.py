from llm_news.curate import curate, merge_duplicates
from llm_news.extract import extract, name_sizes, repo_has_size
from llm_news.models import Item


def paper(summary, models=(), datasets=(), id="arxiv:1", **kw):
    return Item(
        "hf_papers",
        "paper",
        id,
        kw.pop("title", "T"),
        "u",
        "",
        summary=summary,
        artifacts={"models": list(models), "datasets": list(datasets)},
        **kw,
    )


def kinds(claims):
    return sorted(c.kind for c in claims)


def test_size_claim_binds_to_repo_with_matching_name():
    it = paper(
        "We release a 7B parameter model pretrained on 2T tokens with 128K context.",
        models=["org/Model-1B", "org/Model-7B-Instruct"],
    )
    claims = extract(it)
    p = next(c for c in claims if c.kind == "hf_model_params")
    assert p.params == {"repo": "org/Model-7B-Instruct", "value": "7B", "scope": "total"}
    assert {"hf_training_tokens", "hf_model_context"} <= {c.kind for c in claims}


def test_unbound_numbers_are_reported_not_verified():
    claims = extract(paper("Our 35B parameter model was trained on 15 trillion tokens."))
    assert kinds(claims) == ["reported_params", "reported_training_tokens"]


def test_benchmark_scores_are_extracted_as_self_reported():
    claims = extract(paper("The best model achieves 39.0% accuracy on OpenDS-World."))
    assert kinds(claims) == ["benchmark_score"]


def test_linked_dataset_gets_existence_claim():
    claims = extract(paper("A new benchmark.", datasets=["org/bench"]))
    assert claims[0].kind == "hf_dataset" and claims[0].params == {"repo": "org/bench"}


def test_math_candidate_tag():
    it = paper("We prove a tight lower bound on the regret.")
    extract(it)
    assert "math-candidate" in it.tags


def test_training_flops_consistency_claim():
    claims = extract(paper("A 70B parameter model trained on 1.4T tokens using 5.76e23 FLOPs."))
    flops = next(c for c in claims if c.kind == "training_flops")
    assert flops.params == {"flops": "5.76e23", "params": "70B", "tokens": "1.4T"}


def test_name_sizes():
    assert name_sizes("Qwen/Qwen3-30B-A3B") == ("30B", "3B")
    assert name_sizes("org/Llama-3.1-8B-Instruct") == ("8B", None)
    assert name_sizes("org/flint") == (None, None)
    assert repo_has_size("org/Qwen2.5-0.5B", "0.5B") and not repo_has_size("org/Qwen2.5-0.5B", "5B")


def test_model_card_declarations_become_claims():
    card = """---
license: mit
---
| Total Parameters | 28.8B |
| Activated Parameters | 4.3B |
- **Mid-training**: 2T tokens at 32K context
- **Context length**: 131,072 tokens
```python
model = load("x-7B")  # 코드 블록 속 숫자는 무시
```
"""
    it = Item(
        "hf_models", "model", "hf-model:org/raon", "org/raon", "u", "", artifacts={"models": ["org/raon"]}
    )
    claims = {c.statement: c.params for c in extract(it, card)}
    assert claims["모델 카드 선언: 총 파라미터 28.8B"]["scope"] == "total"
    assert claims["모델 카드 선언: 활성 파라미터 4.3B"]["scope"] == "active"
    assert claims["모델 카드 선언: 컨텍스트 길이 131,072"]["value"] == "131,072"
    assert len(claims) == 3


CONFIG = {
    "keywords": {"language model": 2, "llm": 3},
    "min_relevance": 2,
    "limits": {"paper": 2, "model": 5, "blog": 5, "max_per_source": {"blog": 1}},
    "sources": {
        "hf_papers": {"weight": 2, "require_keywords": True},
        "arxiv": {"weight": 1, "require_keywords": True},
        "rss": {"weight": 1, "feeds": [{"name": "A"}, {"name": "B"}]},
    },
}


def test_merge_duplicates_keeps_signals_and_artifacts():
    a = paper("LLM", id="arxiv:9", signals={"upvotes": 10})
    b = Item("arxiv", "paper", "arxiv:9", "T", "u", "", summary="LLM", artifacts={"models": ["m/x"]})
    (merged,) = merge_duplicates([b, a])
    assert merged.signals["upvotes"] == 10 and merged.artifacts["models"] == ["m/x"]


def test_curate_filters_irrelevant_and_previously_seen():
    items = [
        paper("A large LLM study", id="arxiv:1", signals={"upvotes": 5}),
        paper("Robot grasping with diffusion policies", id="arxiv:2", signals={"upvotes": 500}),
        paper("Another LLM paper", id="arxiv:3"),
    ]
    out = curate(items, CONFIG, seen={"arxiv:3": "2026-10-01"}, run_date="2026-10-23")
    assert [i.id for i in out] == ["arxiv:1"]


def test_curate_keeps_todays_items_on_rerun():
    items = [paper("LLM", id="arxiv:1")]
    assert curate(items, CONFIG, seen={"arxiv:1": "2026-10-23"}, run_date="2026-10-23")


def test_upvotes_beat_keyword_stuffing():
    stuffed = paper("LLM language model LLM " * 10, id="arxiv:1", signals={"upvotes": 1})
    popular = paper("An LLM paper", id="arxiv:2", signals={"upvotes": 200})
    out = curate([stuffed, popular], CONFIG, {}, "2026-10-23")
    assert out[0].id == "arxiv:2"


def test_per_source_cap_diversifies_blogs():
    blogs = [Item(f"rss:{src}", "blog", f"url:{src}{i}", "t", "u", "") for src in "AB" for i in range(3)]
    out = curate(blogs, CONFIG, {}, "2026-10-23")
    assert sorted(i.source for i in out) == ["rss:A", "rss:B"]
