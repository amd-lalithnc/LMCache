# SPDX-License-Identifier: Apache-2.0
"""
Pure-PyTorch CPU KV connector for vLLM.

This is the CPU counterpart of ``VLLMPagedMemGPUConnectorV2``. The GPU/XPU/HPU
connectors move KV between vLLM's paged buffer and an LMCache ``MemoryObj`` via a
compiled CUDA/SYCL kernel (``lmcache.c_ops.multi_layer_kv_transfer``) driven by
``torch.cuda.Stream``. Neither the kernel nor CUDA streams exist on a CPU-only
build, so this connector reimplements the same gather/scatter with plain torch
advanced indexing — no native extension, no streams, no device pointers.

It deliberately does NOT import ``lmcache.c_ops`` (absent in NO_GPU_EXT builds)
nor ``gpu_connector.gpu_connectors``/``gpu_connector.utils`` (which import
``c_ops`` at module load). It is fully self-contained.

Supported KV layout: vLLM non-MLA per-layer caches, i.e. a list of per-layer
tensors. vLLM's CPU attention backend uses HND physical layout
``[2, num_blocks, num_heads, block_size, head_size]``; NHD
``[2, num_blocks, block_size, num_heads, head_size]`` is also handled.

MemoryObj format produced/consumed: ``KV_2LTD`` = ``[2, num_layers, T, NH*HS]``.
"""

import os
from typing import List, Optional, Union

import torch

from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


def _contiguous_view(t: torch.Tensor) -> torch.Tensor:
    """Return a view whose .shape reflects the physical layout.

    vLLM may expose an HND cache as a stride-permuted view of an NHD-allocated
    tensor (or vice-versa). Sorting dims by descending stride recovers the
    physical order without copying. No-op if already contiguous.
    """
    if t.is_contiguous():
        return t
    strides = t.stride()
    perm = sorted(range(t.ndim), key=lambda i: strides[i], reverse=True)
    return t.permute(perm)


class VLLMCPUConnector:
    """CPU gather/scatter between vLLM paged KV and an LMCache MemoryObj."""

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_mla: bool = False,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_mla = use_mla
        # vLLM CPU backend uses HND. Override via env if a build differs.
        self.kv_layout = os.environ.get("LMCACHE_CPU_KV_LAYOUT", "HND").upper()
        self.kvcaches: Optional[List[torch.Tensor]] = None
        self._logged = False

        if self.use_mla:
            raise NotImplementedError(
                "VLLMCPUConnector does not support MLA models yet."
            )

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints=None,
    ) -> "VLLMCPUConnector":
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size
        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_mla=metadata.use_mla,
        )

    # ---- interface methods ---------------------------------------------------

    def get_shape(self, num_tokens: int) -> torch.Size:
        kv_size = 1 if self.use_mla else 2
        return torch.Size(
            [kv_size, self.num_layers, num_tokens, self.hidden_dim_size]
        )

    def initialize_kvcaches_ptr(self, **kwargs) -> None:
        if "kvcaches" in kwargs:
            self.kvcaches = kwargs["kvcaches"]

    def _kv_token_views(self, layer: int):
        """Return (k_view, v_view), each a [NB, BS, NH, HS] view of the layer's
        paged cache, indexable as ``view[block_idx, off_idx] -> [.., NH, HS]``.
        Views alias storage so assignment writes through.
        """
        t = _contiguous_view(self.kvcaches[layer])
        # t: [2, NB, *, *, HS]
        k = t[0]
        v = t[1]
        if self.kv_layout == "HND":
            # [NB, NH, BS, HS] -> [NB, BS, NH, HS]
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
        # NHD already [NB, BS, NH, HS]
        return k, v

    def _block_offsets(self, slot_mapping: torch.Tensor, start: int, end: int, bs: int):
        # Negative slots are sentinels (PAD_SLOT_ID == -1, prefix-cache padding).
        # The CUDA kernel and python fallback SKIP them; we must too. Without a
        # guard, slot<0 makes block/off negative which silently wraps to the last
        # block/offset in torch indexing -> reads/writes the WRONG KV. We build a
        # validity mask and clamp indices to 0 so advanced indexing stays in-bounds;
        # callers must only touch positions where `valid` is True.
        slots = slot_mapping[start:end].to(torch.int64)
        valid = slots >= 0
        safe = slots.clamp(min=0)
        block = torch.div(safe, bs, rounding_mode="floor")
        off = safe - block * bs
        return block, off, valid

    def _log_once(self):
        if self._logged:
            return
        self._logged = True
        t = self.kvcaches[0]
        ct = _contiguous_view(t)
        logger.info(
            "VLLMCPUConnector active: layers=%d hidden=%d layout=%s "
            "layer0 raw shape=%s stride=%s -> view shape=%s",
            self.num_layers,
            self.hidden_dim_size,
            self.kv_layout,
            tuple(t.shape),
            tuple(t.stride()),
            tuple(ct.shape),
        )

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """STORE: gather vLLM paged KV (tokens [start,end)) into memory_obj."""
        assert memory_obj.tensor is not None
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' must be provided")
        self._log_once()

        slot_mapping = kwargs["slot_mapping"]
        mt = memory_obj.tensor
        T = end - start
        k0, _ = self._kv_token_views(0)
        bs = k0.shape[1]
        block, off, valid = self._block_offsets(slot_mapping, start, end, bs)
        all_valid = bool(valid.all())

        for layer in range(self.num_layers):
            kview, vview = self._kv_token_views(layer)
            g_k = kview[block, off]  # [T, NH, HS] (clamped; invalid rows are bogus)
            g_v = vview[block, off]
            if not all_valid:
                # do not store KV for sentinel/padding tokens
                g_k = g_k.clone(); g_v = g_v.clone()
                g_k[~valid] = 0; g_v[~valid] = 0
            mt[0, layer, :T, :].copy_(g_k.reshape(T, -1))
            mt[1, layer, :T, :].copy_(g_v.reshape(T, -1))

        memory_obj.metadata.fmt = MemoryFormat.KV_2LTD

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """LOAD: scatter memory_obj KV back into vLLM paged buffer."""
        assert memory_obj.tensor is not None
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' must be provided")
        if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
            raise ValueError(
                "memory object must be in KV_2LTD format for VLLMCPUConnector"
            )
        self._log_once()

        slot_mapping = kwargs["slot_mapping"]
        mt = memory_obj.tensor
        T = end - start

        # Skip tokens already present in vLLM's own paged cache (matches the
        # CUDA path's skip_prefix_n_tokens), avoids redundant writes.
        vllm_cached = kwargs.get("vllm_cached_tokens", 0)
        skip = min(T, max(0, vllm_cached - start))

        k0, _ = self._kv_token_views(0)
        bs = k0.shape[1]
        block, off, valid = self._block_offsets(slot_mapping, start, end, bs)
        # only scatter real positions (skip prefix + sentinel/padding slots)
        sel = valid.clone()
        sel[:skip] = False
        b_sel, o_sel = block[sel], off[sel]
        # row indices into mt for the selected (valid, non-skipped) tokens
        rows = torch.nonzero(sel, as_tuple=True)[0]

        for layer in range(self.num_layers):
            kview, vview = self._kv_token_views(layer)
            nh, hs = kview.shape[2], kview.shape[3]
            kview[b_sel, o_sel] = mt[0, layer, rows, :].reshape(-1, nh, hs)
            vview[b_sel, o_sel] = mt[1, layer, rows, :].reshape(-1, nh, hs)

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)
