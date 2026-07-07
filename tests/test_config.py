"""Tests for the config system (Phase 2).

These are deliberately about *invariants*, not values: we check that derived
fields are computed correctly, that bad configs are rejected loudly, and that a
config survives a YAML round-trip unchanged. That last property is the whole
foundation of reproducibility.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyllm.config import DataConfig, ModelConfig, TrainConfig

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


# --------------------------------------------------------------------------- #
# Derived fields
# --------------------------------------------------------------------------- #
def test_head_dim_is_derived():
    cfg = ModelConfig(d_model=768, n_heads=12)
    assert cfg.head_dim == 64


def test_n_kv_heads_defaults_to_n_heads():
    cfg = ModelConfig(n_heads=12, n_kv_heads=None)
    assert cfg.n_kv_heads == 12  # None => MHA


def test_ffn_hidden_swiglu_formula_and_rounding():
    # 2/3 * 4 * 768 = 2048 exactly, already a multiple of 256.
    cfg = ModelConfig(d_model=768, ffn_mult=4.0, ffn_multiple_of=256)
    assert cfg.ffn_hidden == 2048
    # A size that must round UP: 2/3 * 4 * 1024 = 2730 -> next multiple of 256 = 2816.
    # (n_heads=16 so d_model is divisible; ffn sizing is independent of n_heads.)
    cfg2 = ModelConfig(d_model=1024, n_heads=16, ffn_mult=4.0, ffn_multiple_of=256)
    assert cfg2.ffn_hidden == 2816


def test_explicit_ffn_hidden_is_respected():
    cfg = ModelConfig(ffn_hidden=999)
    assert cfg.ffn_hidden == 999


# --------------------------------------------------------------------------- #
# Validation: bad configs must fail at construction
# --------------------------------------------------------------------------- #
def test_d_model_must_divide_n_heads():
    with pytest.raises(ValueError, match="divisible by n_heads"):
        ModelConfig(d_model=768, n_heads=10)


def test_gqa_grouping_must_be_even():
    with pytest.raises(ValueError, match="divisible by n_kv_heads"):
        ModelConfig(n_heads=12, n_kv_heads=5)


def test_bad_dropout_rejected():
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=1.5)


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="Unknown config keys"):
        ModelConfig.from_dict({"d_model": 128, "not_a_real_field": 7})


def test_train_dtype_validation():
    with pytest.raises(ValueError, match="dtype"):
        TrainConfig(dtype="float64")


def test_warmup_cannot_exceed_max_steps():
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainConfig(warmup_steps=10, max_steps=5)


# --------------------------------------------------------------------------- #
# YAML round-trip (reproducibility)
# --------------------------------------------------------------------------- #
def test_model_config_yaml_roundtrip(tmp_path):
    cfg = ModelConfig(d_model=256, n_heads=8, n_kv_heads=2, vocab_size=1000)
    path = tmp_path / "m.yaml"
    cfg.to_yaml(path)
    reloaded = ModelConfig.from_yaml(path)
    assert reloaded == cfg


def test_train_config_yaml_roundtrip(tmp_path):
    cfg = TrainConfig(lr=1e-3, max_steps=200, warmup_steps=20)
    path = tmp_path / "t.yaml"
    cfg.to_yaml(path)
    assert TrainConfig.from_yaml(path) == cfg


def test_derived_head_dim_excluded_from_yaml(tmp_path):
    cfg = ModelConfig(d_model=256, n_heads=8)
    path = tmp_path / "m.yaml"
    cfg.to_yaml(path)
    text = path.read_text(encoding="utf-8")
    assert "head_dim" not in text  # derived, must not be serialized


# --------------------------------------------------------------------------- #
# Shipped config files load and have the expected scale
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["toy", "model_125m", "model_300m"])
def test_shipped_configs_load(name):
    cfg = ModelConfig.from_yaml(CONFIGS_DIR / f"{name}.yaml")
    assert cfg.head_dim > 0
    assert cfg.ffn_hidden and cfg.ffn_hidden > 0


def test_train_default_loads():
    TrainConfig.from_yaml(CONFIGS_DIR / "train_default.yaml")


def test_param_estimates_are_in_the_right_ballpark():
    m125 = ModelConfig.from_yaml(CONFIGS_DIR / "model_125m.yaml")
    m300 = ModelConfig.from_yaml(CONFIGS_DIR / "model_300m.yaml")
    assert 120_000_000 <= m125.estimate_params() <= 130_000_000
    # ~322M: GQA (n_kv_heads=4) shrinks the K/V projections vs full MHA.
    assert 300_000_000 <= m300.estimate_params() <= 350_000_000


def test_tokens_per_step():
    t = TrainConfig(micro_batch_size=8, grad_accum_steps=4)
    assert t.tokens_per_step(seq_len=1024) == 8 * 4 * 1024


def test_dataconfig_defaults():
    d = DataConfig()
    assert d.num_workers == 0
    with pytest.raises(ValueError):
        DataConfig(num_workers=-1)
