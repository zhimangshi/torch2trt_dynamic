from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import torch


_ShapeLike = Union[torch.Size, Sequence[int], Iterable[int]]


def scatter_nd(
    indices: torch.Tensor,
    updates: torch.Tensor,
    shape: _ShapeLike,
    *,
    reduction: str = "replace",
    validate_indices: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    PyTorch 实现的 scatter_nd（对齐 TensorFlow scatter_nd 的常用语义）。

    给定:
    - indices: shape = (..., K)，最后一维 K 表示索引的维度数
    - updates: shape = indices.shape[:-1] + shape[K:]
    - shape: 输出张量形状（长度 >= K）

    返回:
    - output: shape = shape，初始化为 0，然后把 updates 写入 indices 指定的位置

    参数:
    - reduction:
        - "replace": 覆盖写入（默认）
        - "add"/"sum": 对重复 indices 做累加
    - validate_indices: 为 True 时检查 indices 是否越界（会带来少量同步开销）
    """
    if not isinstance(indices, torch.Tensor) or not isinstance(updates,
                                                              torch.Tensor):
        raise TypeError("indices 和 updates 必须是 torch.Tensor")

    out_shape: Tuple[int, ...] = tuple(int(x) for x in shape)
    if indices.dim() == 0:
        raise ValueError("indices 至少需要 1 个维度（... , K）")
    if indices.size(-1) <= 0:
        raise ValueError("indices 的最后一维 K 必须 > 0")

    k = int(indices.size(-1))
    if k > len(out_shape):
        raise ValueError(
            f"indices 的最后一维 K={k} 不能大于输出维度数 len(shape)={len(out_shape)}"
        )

    expected_updates_shape = tuple(indices.shape[:-1]) + out_shape[k:]
    if tuple(updates.shape) != expected_updates_shape:
        raise ValueError(
            f"updates.shape 应为 {expected_updates_shape}，实际为 {tuple(updates.shape)}"
        )

    if reduction not in ("replace", "add", "sum"):
        raise ValueError('reduction 仅支持 "replace" / "add" / "sum"')
    accumulate = reduction in ("add", "sum")

    if device is None:
        device = updates.device
    if dtype is None:
        dtype = updates.dtype

    out = torch.zeros(out_shape, dtype=dtype, device=device)
    if indices.numel() == 0:
        return out

    # torch 的高级索引需要 long
    idx = indices.to(device=device, dtype=torch.long)
    idx_flat = idx.reshape(-1, k).transpose(0, 1).contiguous()  # (K, N)

    if validate_indices:
        for d in range(k):
            if torch.any(idx_flat[d] < 0) or torch.any(idx_flat[d] >= out_shape[d]):
                raise IndexError(
                    f"indices 在维度 {d} 上越界，合法范围 [0, {out_shape[d]})"
                )

    # value 形状需要与高级索引结果匹配: (N, *out_shape[k:])
    updates_flat = updates.to(device=device, dtype=dtype).reshape(
        -1, *out_shape[k:])

    out.index_put_(tuple(idx_flat[d] for d in range(k)),
                   updates_flat,
                   accumulate=accumulate)
    return out

