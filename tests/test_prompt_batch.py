"""Prompt 批推理语义：eval 逐样本选择 / train 保留 batchwise 多数投票。

批推理（B>1）数值一致性的前提：eval 下不用整 batch 多数 prompt 强加给
所有样本（否则少数派样本特征与逐片推理不一致）。top_k=1 下 B=1 两分支
严格等价（现网单批行为锚点）。
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core" / "algo_packages"))

from models.prompt import Prompt  # noqa: E402


def _make_prompt(pool_size: int = 5) -> Prompt:
    torch.manual_seed(7)
    prompt = Prompt(
        length=4, embed_dim=8, embedding_key="mean", prompt_init="uniform",
        prompt_pool=True, prompt_key=True, pool_size=pool_size,
        top_k=1, batchwise_prompt=True,
    )
    # 键替换为单位基向量：x_embed 均值 = e_i → 样本 i 选 prompt i（确定性）
    with torch.no_grad():
        prompt.prompt_key.copy_(torch.eye(pool_size, 8))
    return prompt


def _batch(prompt: Prompt, ids: list) -> torch.Tensor:
    # 每样本 token 均值恰为 e_id：tokens 全取 e_id 即可
    return torch.stack([torch.stack([torch.eye(5, 8)[i] for _ in range(3)]) for i in ids])


def test_eval_batch_uses_per_sample_prompt():
    """eval：B=4 各选各的 prompt（合批推理的等价性前提）。"""
    prompt = _make_prompt()
    prompt.eval()
    with torch.no_grad():
        out = prompt(_batch(prompt, [0, 1, 2, 3]))
    idx = out["prompt_idx"]
    assert idx.shape == (4, 1)
    assert idx.flatten().tolist() == [0, 1, 2, 3], f"eval 应逐样本选择, got {idx.flatten().tolist()}"


def test_train_batch_keeps_majority_vote():
    """train：batchwise 多数投票语义原样保留（训练损失的分母行为）。"""
    prompt = _make_prompt()
    prompt.train()
    with torch.no_grad():
        out = prompt(_batch(prompt, [0, 1, 2, 3]))
    idx = out["prompt_idx"].flatten().tolist()
    assert len(set(idx)) == 1, f"train 下全 batch 应共用多数 prompt, got {idx}"


def test_batch1_train_eval_equivalent():
    """B=1 时两分支 idx 相等（top_k=1 下现网单批行为不变锚点）。"""
    prompt = _make_prompt()
    x = _batch(prompt, [2])
    with torch.no_grad():
        prompt.train()
        train_idx = prompt(x)["prompt_idx"].clone()
        prompt.eval()
        eval_idx = prompt(x)["prompt_idx"].clone()
    assert train_idx.shape == (1, 1)
    assert train_idx.equal(eval_idx) and eval_idx.flatten().tolist() == [2]
