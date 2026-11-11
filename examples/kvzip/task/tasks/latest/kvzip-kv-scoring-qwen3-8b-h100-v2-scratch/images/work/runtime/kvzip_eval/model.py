"""Frozen Qwen3-8B: prefill (plain or capturing) and masked greedy decoding.

The layer loop below is the Hugging Face Qwen3 decoder layer written out (same
modules, same SDPA call), so the prefill KV cache equals transformers' own and
the scorer sees exactly the tensors the cache was built from.
"""

from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn.functional as F

from . import protocol as P
from .attention import merged_attention


def load_model(model_dir: str, device: torch.device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
        device_map={"": device})
    model = model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


CaptureFn = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None]


@torch.inference_mode()
def prefill(model, ids: torch.Tensor, capture: CaptureFn | None = None, keep_cache: bool = True):
    """Prefill one sequence ids [1, n]. Returns per-layer (K, V) lists of [8, n, 128].

    capture(layer, h [n, 4096], q [32, n, 128], k [8, n, 128], v [8, n, 128]) is called
    for every layer before its attention, with the attention input (after the
    layer's input RMSNorm) and the post-norm, post-RoPE queries and keys.
    """
    m = model.model
    n = ids.shape[1]
    x = m.embed_tokens(ids)
    pos = torch.arange(n, device=ids.device).unsqueeze(0)
    cos, sin = m.rotary_emb(x, pos)
    keys, values = [], []
    for layer_idx, layer in enumerate(m.layers):
        attn = layer.self_attn
        residual = x
        h = layer.input_layernorm(x)
        q = attn.q_norm(attn.q_proj(h).view(1, n, -1, P.HEAD_DIM)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(h).view(1, n, -1, P.HEAD_DIM)).transpose(1, 2)
        v = attn.v_proj(h).view(1, n, -1, P.HEAD_DIM).transpose(1, 2)
        q, k = _rope(q, k, cos, sin)
        if keep_cache:
            keys.append(k[0])
            values.append(v[0])
        if capture is not None:
            capture(layer_idx, h[0], q[0], k[0], v[0])
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True, scale=attn.scaling)
        x = residual + attn.o_proj(o.transpose(1, 2).reshape(1, n, -1))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return keys, values


def timed_prefill(model, ids: torch.Tensor):
    """Plain prefill timed with the device synchronized before and after."""
    torch.cuda.synchronize(ids.device)
    t0 = time.perf_counter()
    keys, values = prefill(model, ids)
    torch.cuda.synchronize(ids.device)
    return keys, values, time.perf_counter() - t0


class RowDecoder:
    """Greedy decoding of every (question, setting) element of one row in one batch.

    k_ctx/v_ctx: per-layer [8, N, 128] prefill cache (prefix + context);
    keep: [36, S, 8, N] uint8 visibility of each cache entry under each setting.
    """

    def __init__(self, model, k_ctx, v_ctx, keep, use_graphs: bool = True):
        self.model = model
        self.m = model.model
        self.k_ctx = k_ctx
        self.v_ctx = v_ctx
        self.keep = keep
        self.n_ctx_total = k_ctx[0].shape[1]
        self.device = k_ctx[0].device
        self.use_graphs = use_graphs
        self.eos = torch.tensor(P.EOS_TOKEN_IDS, device=self.device)

    def _layers(self, x, positions, rowset, ks, vs, slots, limit):
        """x [B, T, 4096]; writes K/V of these tokens into ks/vs[layer][b, :, slots[b, t]]."""
        m = self.m
        b, t = x.shape[0], x.shape[1]
        cos, sin = m.rotary_emb(x, positions)
        bidx = torch.arange(b, device=x.device).unsqueeze(1).expand(b, t)
        for layer_idx, layer in enumerate(m.layers):
            attn = layer.self_attn
            residual = x
            h = layer.input_layernorm(x)
            q = attn.q_norm(attn.q_proj(h).view(b, t, -1, P.HEAD_DIM)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(h).view(b, t, -1, P.HEAD_DIM)).transpose(1, 2)
            v = attn.v_proj(h).view(b, t, -1, P.HEAD_DIM).transpose(1, 2)
            q, k = _rope(q, k, cos, sin)
            ks[layer_idx][bidx, :, slots] = k.transpose(1, 2)
            vs[layer_idx][bidx, :, slots] = v.transpose(1, 2)
            o = merged_attention(q, self.k_ctx[layer_idx], self.v_ctx[layer_idx], self.keep[layer_idx],
                                 rowset, ks[layer_idx], vs[layer_idx], limit, attn.scaling)
            x = residual + attn.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return x

    @torch.inference_mode()
    def run(self, prompts: list[list[int]], settings: list[int], max_new: int) -> list[list[int]]:
        dev = self.device
        nb = len(prompts)
        lengths = torch.tensor([len(p) for p in prompts], device=dev)
        t_max = max(len(p) for p in prompts)
        s_max = t_max + max_new
        nl = len(self.m.layers)
        ks = torch.zeros(nl, nb, P.NUM_KV_HEADS, s_max, P.HEAD_DIM, device=dev, dtype=torch.bfloat16)
        vs = torch.zeros_like(ks)
        ids = torch.zeros(nb, t_max, dtype=torch.long, device=dev)
        for i, p in enumerate(prompts):
            ids[i, :len(p)] = torch.tensor(p, device=dev)
        setting_t = torch.tensor(settings, device=dev, dtype=torch.int32)
        group = P.NUM_Q_HEADS // P.NUM_KV_HEADS

        # 1. the appended prompt tokens of every element (eager, once per row)
        slots = torch.arange(t_max, device=dev).unsqueeze(0).expand(nb, t_max)
        positions = self.n_ctx_total + slots
        rowset = setting_t.repeat_interleave(group * t_max)
        x = self._layers(self.m.embed_tokens(ids), positions, rowset, ks, vs, slots, slots)
        last = x[torch.arange(nb, device=dev), lengths - 1]
        first = self.model.lm_head(self.m.norm(last)).argmax(-1)

        out = torch.zeros(nb, max_new, dtype=torch.long, device=dev)
        finished = torch.zeros(nb, dtype=torch.bool, device=dev)
        out[:, 0] = first
        finished |= torch.isin(first, self.eos)
        if max_new > 1 and not bool(finished.all()):
            tok = first.clone()
            spos = lengths.clone()
            step = torch.ones(1, dtype=torch.long, device=dev)
            rowset1 = setting_t.repeat_interleave(group)
            arange_b = torch.arange(nb, device=dev)

            def decode_step():
                xs = self.m.embed_tokens(tok.unsqueeze(1))
                sl = spos.unsqueeze(1)
                xs = self._layers(xs, self.n_ctx_total + sl, rowset1, ks, vs, sl, sl)
                nxt = self.model.lm_head(self.m.norm(xs[:, 0])).argmax(-1)
                out[arange_b, step.expand(nb)] = nxt
                finished.logical_or_(torch.isin(nxt, self.eos))
                tok.copy_(nxt)
                spos.add_(1)
                step.add_(1)

            remaining = max_new - 1
            graph = None
            if self.use_graphs and remaining > 4:
                stream = torch.cuda.Stream(device=dev)
                stream.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(stream):
                    decode_step()  # warm-up doubles as step 1
                torch.cuda.current_stream(dev).wait_stream(stream)
                remaining -= 1
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    decode_step()
            done_steps = 0
            while remaining > 0:
                burst = min(8, remaining)
                for _ in range(burst):
                    if graph is not None:
                        graph.replay()
                    else:
                        decode_step()
                remaining -= burst
                done_steps += burst
                if bool(finished.all()):
                    break
            del graph
        result = []
        out_l = out.tolist()
        for i in range(nb):
            toks = []
            for tkn in out_l[i]:
                if tkn in P.EOS_TOKEN_IDS:
                    break
                toks.append(tkn)
            result.append(toks)
        return result
