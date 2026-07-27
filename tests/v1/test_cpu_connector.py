# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure-PyTorch CPU KV connector (``VLLMCPUConnector``).

Covers, without any GPU / c_ops kernels:
1. round-trip store/load correctness on CPU,
2. both HND and NHD physical layouts,
3. negative sentinel slots (``-1``) being skipped on load and zeroed on store.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.cpu_connector import VLLMCPUConnector
from lmcache.v1.gpu_connector.interface import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryFormat

NUM_LAYERS = 2
NB = 4  # num blocks
BS = 8  # block size
NH = 2  # num kv heads
HS = 4  # head size
HIDDEN = NH * HS


def _make_kvcaches(layout: str, fill, mode: str = "5d") -> list:
    """Per-layer paged caches in the requested physical layout.

    mode="5d": split K/V on a leading dim  -> [2, NB, NH, BS, HS] (HND) /
               [2, NB, BS, NH, HS] (NHD).
    mode="4d": vLLM CPU fused K/V on the last dim -> [NB, NH, BS, 2*HS] (HND) /
               [NB, BS, NH, 2*HS] (NHD).
    """
    caches = []
    for layer in range(NUM_LAYERS):
        if mode == "5d":
            shape = (2, NB, NH, BS, HS) if layout == "HND" else (2, NB, BS, NH, HS)
        else:  # 4d fused
            shape = (NB, NH, BS, 2 * HS) if layout == "HND" else (NB, BS, NH, 2 * HS)
        caches.append(fill(shape, layer))
    return caches


def _make_memory_obj(num_tokens: int) -> SimpleNamespace:
    """Duck-typed KV_2LTD MemoryObj: [2, num_layers, T, hidden]."""
    tensor = torch.zeros(2, NUM_LAYERS, num_tokens, HIDDEN)
    return SimpleNamespace(
        tensor=tensor, metadata=SimpleNamespace(fmt=MemoryFormat.KV_2LTD)
    )


def test_is_gpu_connector_interface_subclass():
    assert issubclass(VLLMCPUConnector, GPUConnectorInterface)


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("mode", ["5d", "4d"])
def test_round_trip(layout, mode):
    torch.manual_seed(0)
    conn = VLLMCPUConnector(
        hidden_dim_size=HIDDEN,
        num_layers=NUM_LAYERS,
        kv_layout=layout,
        head_size=HS,
    )
    src = _make_kvcaches(layout, lambda shape, layer: torch.randn(*shape), mode)
    dst = _make_kvcaches(layout, lambda shape, layer: torch.zeros(*shape), mode)

    tokens = NB * BS
    slots = torch.arange(tokens, dtype=torch.int64)
    mo = _make_memory_obj(tokens)

    conn.from_gpu(mo, 0, tokens, kvcaches=src, slot_mapping=slots)
    conn.to_gpu(mo, 0, tokens, kvcaches=dst, slot_mapping=slots)

    # Every scattered slot in dst must match the gathered value from src.
    conn.kvcaches = src
    src_views = [conn._kv_token_views(l) for l in range(NUM_LAYERS)]
    conn.kvcaches = dst
    dst_views = [conn._kv_token_views(l) for l in range(NUM_LAYERS)]
    block = torch.div(slots, BS, rounding_mode="floor")
    off = slots - block * BS
    for layer in range(NUM_LAYERS):
        ks, vs = src_views[layer]
        kd, vd = dst_views[layer]
        assert torch.allclose(kd[block, off], ks[block, off])
        assert torch.allclose(vd[block, off], vs[block, off])


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("mode", ["5d", "4d"])
def test_sentinel_slots_skipped(layout, mode):
    torch.manual_seed(1)
    conn = VLLMCPUConnector(
        hidden_dim_size=HIDDEN,
        num_layers=NUM_LAYERS,
        kv_layout=layout,
        head_size=HS,
    )
    src = _make_kvcaches(layout, lambda shape, layer: torch.randn(*shape), mode)
    dst = _make_kvcaches(layout, lambda shape, layer: torch.zeros(*shape), mode)

    # token index 2 is a PAD_SLOT sentinel (-1) and must be skipped.
    slots = torch.tensor([0, 1, -1, 3], dtype=torch.int64)
    tokens = slots.numel()
    mo = _make_memory_obj(tokens)

    conn.from_gpu(mo, 0, tokens, kvcaches=src, slot_mapping=slots)
    # The stored row for the sentinel token must be zeroed (not garbage).
    assert torch.count_nonzero(mo.tensor[0, :, 2, :]) == 0
    assert torch.count_nonzero(mo.tensor[1, :, 2, :]) == 0

    conn.to_gpu(mo, 0, tokens, kvcaches=dst, slot_mapping=slots)
    # Valid slots are written; the sentinel wrote nowhere, so dst stays
    # consistent with src only at the valid positions.
    conn.kvcaches = src
    src_views = [conn._kv_token_views(l) for l in range(NUM_LAYERS)]
    conn.kvcaches = dst
    dst_views = [conn._kv_token_views(l) for l in range(NUM_LAYERS)]
    for layer in range(NUM_LAYERS):
        ks, _ = src_views[layer]
        kd, _ = dst_views[layer]
        for slot in (0, 1, 3):
            b, o = slot // BS, slot % BS
            assert torch.allclose(kd[b, o], ks[b, o])


def test_mla_not_supported():
    with pytest.raises(NotImplementedError):
        VLLMCPUConnector(hidden_dim_size=HIDDEN, num_layers=NUM_LAYERS, use_mla=True)
