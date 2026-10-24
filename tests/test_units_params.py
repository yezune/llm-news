import pytest
from conftest import fixture_json

from llm_news.units import human, parse_quantity, rounding_tolerance
from llm_news.verify.params import context_lengths, count_params


@pytest.mark.parametrize(
    "text,value",
    [
        ("7B", 7e9),
        ("0.5B", 5e8),
        ("1.2T", 1.2e12),
        ("128K", 128e3),
        ("15 trillion", 15e12),
        ("6.3e24", 6.3e24),
        ("3.8×10^25", 3.8e25),
        ("131,072", 131072),
        ("52K", 52e3),
        (405, 405.0),
    ],
)
def test_parse_quantity(text, value):
    assert parse_quantity(text) == pytest.approx(value)


def test_parse_quantity_rejects_garbage():
    with pytest.raises(ValueError):
        parse_quantity("seven billion-ish")


def test_rounding_tolerance_follows_significant_digits():
    assert rounding_tolerance("7B") == pytest.approx(0.5 / 7)
    assert rounding_tolerance("6.74B") == pytest.approx(0.005 / 6.74)
    assert rounding_tolerance(7e9) == 0.0


def test_human_never_prints_exponent_in_unit():
    assert human(999_885_952) == "1B"
    assert human(494_032_768) == "494M"


# 기대값은 Hugging Face Hub의 safetensors 실측 파라미터 수 (2026-10 조회)
@pytest.mark.parametrize(
    "fixture,total,active",
    [
        ("qwen2.5-0.5b.config.json", 494_032_768, 494_032_768),
        ("mixtral-8x7b.config.json", 46_702_792_704, 12_879_925_248),
        ("qwen3-30b-a3b.config.json", 30_532_122_624, 3_353_032_704),
    ],
)
def test_count_params_matches_real_weights_exactly(fixture, total, active):
    pc = count_params(fixture_json(fixture))
    assert pc.total == total
    assert pc.active == active


def test_count_params_refuses_unknown_architecture():
    assert (
        count_params(
            {
                "model_type": "deepseek_v3",
                "vocab_size": 10,
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
            }
        )
        is None
    )


def test_count_params_uses_text_config_for_multimodal():
    cfg = {"model_type": "llava", "text_config": fixture_json("qwen2.5-0.5b.config.json")}
    assert count_params(cfg).total == 494_032_768


def test_context_lengths_include_rope_scaling():
    cfg = {
        "model_type": "llama",
        "max_position_embeddings": 32768,
        "rope_scaling": {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768},
    }
    assert context_lengths(cfg) == [32768, 131072]
