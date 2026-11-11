"""Masked attention used for every decoded token (full cache and compressed alike).

The prefilled cache (protected prefix + context) is shared by all decoding
elements of a row; each element (question x setting) sees it through the
retention mask of its setting.  Tokens appended after the context (question,
chat postfix, answer prefix, generated tokens) are always visible to their own
element.  Scores and softmax are float32; inputs and P@V operands are bf16.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit(do_not_specialize=["R", "N", "chunk"])
def _ctx_attn_kernel(Q, K, V, MASK, ROWSET, O, ML,
                     s_qh, s_qr, s_kh, s_kn, s_vh, s_vn, s_ms, s_mh,
                     s_oh, s_or, s_os, s_mlh, s_mlr, s_mls,
                     R, N, chunk, qk_scale,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.program_id(2)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < R
    d = tl.arange(0, D)
    q = tl.load(Q + h * s_qh + rows[:, None] * s_qr + d[None, :], mask=rmask[:, None], other=0.0)
    rs = tl.load(ROWSET + rows, mask=rmask, other=0)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    start = sp * chunk
    end = tl.minimum(start + chunk, N)
    for n0 in range(start, end, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        cmask = cols < end
        k = tl.load(K + h * s_kh + cols[:, None] * s_kn + d[None, :], mask=cmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        keep = tl.load(MASK + rs[:, None] * s_ms + h * s_mh + cols[None, :],
                       mask=rmask[:, None] & cmask[None, :], other=0)
        s = tl.where(keep != 0, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + h * s_vh + cols[:, None] * s_vn + d[None, :], mask=cmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    tl.store(O + h * s_oh + rows[:, None] * s_or + sp * s_os + d[None, :], acc, mask=rmask[:, None])
    tl.store(ML + h * s_mlh + rows * s_mlr + sp * s_mls, m_i, mask=rmask)
    tl.store(ML + h * s_mlh + rows * s_mlr + sp * s_mls + 1, l_i, mask=rmask)


def _config(rows: int):
    """(BLOCK_M, BLOCK_N, num_warps, num_stages, target programs), tuned on H100."""
    if rows <= 32:
        return 16, 64, 4, 3, 264
    if rows <= 256:
        return 32, 128, 4, 2, 528
    if rows <= 2048:
        return 128, 128, 8, 2, 528
    return 128, 128, 8, 2, 264


def ctx_attention(q, K, V, keep, rowset, scale):
    """Context attention with a per-row retention mask.

    q: [Hkv, R, D] bf16 (rows = element x group-head x token), K/V: [Hkv, N, D] bf16,
    keep: [S, Hkv, N] uint8 (1 = visible) for each setting, rowset: [R] int32 setting of
    each row.  Returns o [Hkv, R, D] float32 and the natural-log LSE [Hkv, R].
    """
    hkv, rows, dim = q.shape
    n = K.shape[1]
    block_m, block_n, num_warps, num_stages, target_programs = _config(rows)
    nblk = triton.cdiv(rows, block_m)
    splits = max(1, min(triton.cdiv(n, block_n), triton.cdiv(target_programs, nblk * hkv)))
    chunk = triton.cdiv(triton.cdiv(n, splits), block_n) * block_n
    splits = triton.cdiv(n, chunk)
    out = torch.empty((hkv, rows, splits, dim), device=q.device, dtype=torch.float32)
    ml = torch.empty((hkv, rows, splits, 2), device=q.device, dtype=torch.float32)
    _ctx_attn_kernel[(nblk, hkv, splits)](
        q, K, V, keep, rowset, out, ml,
        q.stride(0), q.stride(1), K.stride(0), K.stride(1), V.stride(0), V.stride(1),
        keep.stride(0), keep.stride(1),
        out.stride(0), out.stride(1), out.stride(2), ml.stride(0), ml.stride(1), ml.stride(2),
        rows, n, chunk, scale * LOG2E,
        BLOCK_M=block_m, BLOCK_N=block_n, D=dim, num_warps=num_warps, num_stages=num_stages)
    m = ml[..., 0]
    l = ml[..., 1]
    mmax = m.amax(-1, keepdim=True)
    mmax = torch.where(torch.isinf(mmax), torch.zeros_like(mmax), mmax)
    w = torch.exp2(m - mmax)
    total = (w * l).sum(-1)
    o = (out * w[..., None]).sum(2) / total[..., None]
    lse = (mmax[..., 0] + torch.log2(total)) / LOG2E
    return o, lse


def suffix_attention(q, ks, vs, limit, scale):
    """Each element's own appended tokens.

    q: [B, Hq, T, D]; ks/vs: [B, Hkv, S, D]; limit: [B, T] last visible suffix slot of
    each query token (causal).  Returns o [B, Hq, T, D] float32 and LSE [B, Hq, T].
    """
    b, hq, t, d = q.shape
    hkv, s = ks.shape[1], ks.shape[2]
    g = hq // hkv
    qg = q.view(b, hkv, g, t, d).float()
    scores = torch.einsum("bhgtd,bhsd->bhgts", qg, ks.float()) * scale
    slots = torch.arange(s, device=q.device)
    allowed = slots.view(1, 1, s) <= limit.unsqueeze(-1)  # [B, T, S]
    scores = scores.masked_fill(~allowed.view(b, 1, 1, t, s), float("-inf"))
    lse = torch.logsumexp(scores, -1)
    p = torch.exp(scores - lse.unsqueeze(-1))
    o = torch.einsum("bhgts,bhsd->bhgtd", p, vs.float())
    return o.reshape(b, hq, t, d), lse.reshape(b, hq, t)


def merged_attention(q, k_ctx, v_ctx, keep, rowset, ks, vs, limit, scale):
    """Full attention output [B, T, Hq*D] bf16 for queries q [B, Hq, T, D]."""
    b, hq, t, d = q.shape
    hkv = k_ctx.shape[0]
    g = hq // hkv
    qr = q.view(b, hkv, g, t, d).permute(1, 0, 2, 3, 4).reshape(hkv, b * g * t, d)
    o1, lse1 = ctx_attention(qr, k_ctx, v_ctx, keep, rowset, scale)
    o1 = o1.view(hkv, b, g, t, d).permute(1, 0, 2, 3, 4).reshape(b, hq, t, d)
    lse1 = lse1.view(hkv, b, g, t).permute(1, 0, 2, 3).reshape(b, hq, t)
    o2, lse2 = suffix_attention(q, ks, vs, limit, scale)
    lse = torch.logaddexp(lse1, lse2)
    o = o1 * torch.exp(lse1 - lse).unsqueeze(-1) + o2 * torch.exp(lse2 - lse).unsqueeze(-1)
    return o.to(q.dtype).transpose(1, 2).reshape(b, t, hq * d)
