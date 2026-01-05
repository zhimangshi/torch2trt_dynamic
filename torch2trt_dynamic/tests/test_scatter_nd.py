import pytest
import torch

from torch2trt_dynamic.scatter_nd import scatter_nd


def test_scatter_nd_1d_replace():
    indices = torch.tensor([[0], [2], [3]], dtype=torch.int64)
    updates = torch.tensor([10.0, 20.0, 30.0])
    out = scatter_nd(indices, updates, shape=(4, ))
    assert torch.allclose(out, torch.tensor([10.0, 0.0, 20.0, 30.0]))


def test_scatter_nd_2d_replace():
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    updates = torch.tensor([5, 7], dtype=torch.int32)
    out = scatter_nd(indices, updates, shape=(2, 2))
    expected = torch.tensor([[0, 5], [7, 0]], dtype=torch.int32)
    assert torch.equal(out, expected)


def test_scatter_nd_prefix_indices_with_slice_updates():
    # shape=(3,2,2), K=1：对第 0 维做 scatter，每个 update 是 (2,2) 的块
    indices = torch.tensor([[0], [2]], dtype=torch.int64)
    updates = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    out = scatter_nd(indices, updates, shape=(3, 2, 2))

    expected = torch.zeros((3, 2, 2), dtype=torch.float32)
    expected[0] = updates[0]
    expected[2] = updates[1]
    assert torch.allclose(out, expected)


def test_scatter_nd_reduction_add_for_duplicates():
    indices = torch.tensor([[1], [1], [2]], dtype=torch.int64)
    updates = torch.tensor([1.5, 2.5, 4.0], dtype=torch.float32)
    out = scatter_nd(indices, updates, shape=(4, ), reduction="add")
    assert torch.allclose(out, torch.tensor([0.0, 4.0, 4.0, 0.0]))


def test_scatter_nd_validate_indices_raises():
    indices = torch.tensor([[3]], dtype=torch.int64)
    updates = torch.tensor([1.0], dtype=torch.float32)
    with pytest.raises(IndexError):
        scatter_nd(indices, updates, shape=(3, ), validate_indices=True)

