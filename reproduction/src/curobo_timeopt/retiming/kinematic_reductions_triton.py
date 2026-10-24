"""Fused Triton reductions for dense kinematic validation."""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


FUSED_BASIS_TIMING_MAX_COEFFICIENTS = 64


def supports_fused_basis_timing_shape(coefficients: int, dof: int, samples: int) -> bool:
    """Return whether the fused basis-to-kinematics Triton kernel supports a shape.

    The basis-major kernel currently supports at most 64 coefficients. Callers
    can use evaluated-timing kernels above this bound without changing checks.
    """
    return (
        1 <= int(coefficients) <= FUSED_BASIS_TIMING_MAX_COEFFICIENTS
        and 1 <= int(dof) <= 16
        and int(samples) >= 2
    )


if triton is not None:

    @triton.jit
    def _kinematic_node_reduce_kernel(
        qd,
        qdd,
        qddd,
        velocity_limit,
        acceleration_limit,
        jerk_limit,
        node_max,
        tile_metrics,
        S: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        Q_STRIDE_B: tl.constexpr,
        Q_STRIDE_S: tl.constexpr,
        Q_STRIDE_D: tl.constexpr,
        NODE_STRIDE_B: tl.constexpr,
        TILE_STRIDE_B: tl.constexpr,
        TILE_STRIDE_T: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        candidate = tl.program_id(0)
        tile = tl.program_id(1)

        sample = tile * BLOCK_S + tl.arange(0, BLOCK_S)
        joint = tl.arange(0, BLOCK_D)
        mask_s = sample < S
        mask_d = joint < D
        mask = mask_s[:, None] & mask_d[None, :]

        base = (
            candidate * Q_STRIDE_B
            + sample[:, None] * Q_STRIDE_S
            + joint[None, :] * Q_STRIDE_D
        )
        qd_v = tl.load(qd + base, mask=mask, other=0.0)
        qdd_v = tl.load(qdd + base, mask=mask, other=0.0)
        qddd_v = tl.load(qddd + base, mask=mask, other=0.0)

        inf = float("inf")
        v_lim = tl.load(
            velocity_limit + joint,
            mask=mask_d,
            other=inf,
        )[None, :]
        a_lim = tl.load(
            acceleration_limit + joint,
            mask=mask_d,
            other=inf,
        )[None, :]
        j_lim = tl.load(
            jerk_limit + joint,
            mask=mask_d,
            other=inf,
        )[None, :]

        zero = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
        v_excess = tl.maximum(tl.abs(qd_v) - v_lim, zero)
        a_excess = tl.maximum(tl.abs(qdd_v) - a_lim, zero)
        j_excess = tl.maximum(tl.abs(qddd_v) - j_lim, zero)

        v_node = tl.max(v_excess, axis=1)
        a_node = tl.max(a_excess, axis=1)
        j_node = tl.max(j_excess, axis=1)
        combined = tl.maximum(v_node, tl.maximum(a_node, j_node))

        tl.store(
            node_max
            + candidate * NODE_STRIDE_B
            + sample,
            combined,
            mask=mask_s,
        )

        invalid = -float("inf")
        v_tile = tl.max(tl.where(mask_s, v_node, invalid), axis=0)
        a_tile = tl.max(tl.where(mask_s, a_node, invalid), axis=0)
        j_tile = tl.max(tl.where(mask_s, j_node, invalid), axis=0)
        out = tile_metrics + candidate * TILE_STRIDE_B + tile * TILE_STRIDE_T
        tl.store(out + 0, v_tile)
        tl.store(out + 1, a_tile)
        tl.store(out + 2, j_tile)


    @triton.jit
    def _kinematic_finalize_kernel(
        node_max,
        tile_metrics,
        interval,
        output,
        S: tl.constexpr,
        TILES: tl.constexpr,
        NODE_STRIDE_B: tl.constexpr,
        TILE_STRIDE_B: tl.constexpr,
        TILE_STRIDE_T: tl.constexpr,
        INTERVAL_STRIDE_B: tl.constexpr,
        OUT_STRIDE_M: tl.constexpr,
        OUT_STRIDE_B: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        candidate = tl.program_id(0)

        sample = tl.arange(0, BLOCK_S)
        mask_i = sample < (S - 1)
        current = tl.load(
            node_max + candidate * NODE_STRIDE_B + sample,
            mask=mask_i,
            other=0.0,
        )
        nxt = tl.load(
            node_max + candidate * NODE_STRIDE_B + sample + 1,
            mask=mask_i,
            other=0.0,
        )
        tl.store(
            interval + candidate * INTERVAL_STRIDE_B + sample,
            tl.maximum(current, nxt),
            mask=mask_i,
        )

        tile = tl.arange(0, BLOCK_T)
        mask_t = tile < TILES
        base = (
            tile_metrics
            + candidate * TILE_STRIDE_B
            + tile * TILE_STRIDE_T
        )
        invalid = -float("inf")
        v = tl.load(base + 0, mask=mask_t, other=invalid)
        a = tl.load(base + 1, mask=mask_t, other=invalid)
        j = tl.load(base + 2, mask=mask_t, other=invalid)
        tl.store(
            output + 0 * OUT_STRIDE_M + candidate * OUT_STRIDE_B,
            tl.max(v, axis=0),
        )
        tl.store(
            output + 1 * OUT_STRIDE_M + candidate * OUT_STRIDE_B,
            tl.max(a, axis=0),
        )
        tl.store(
            output + 2 * OUT_STRIDE_M + candidate * OUT_STRIDE_B,
            tl.max(j, axis=0),
        )


    @triton.jit
    def _basis_timing_kinematic_node_reduce_kernel(
        coefficients,
        B,
        B_s,
        B_ss,
        q_s,
        q_ss,
        q_sss,
        velocity_limit,
        acceleration_limit,
        jerk_limit,
        x_out,
        qd,
        qdd,
        node_max,
        tile_metrics,
        S: tl.constexpr,
        K: tl.constexpr,
        D: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        C_STRIDE_B: tl.constexpr,
        B_STRIDE_S: tl.constexpr,
        B_STRIDE_K: tl.constexpr,
        PATH_STRIDE_S: tl.constexpr,
        PATH_STRIDE_D: tl.constexpr,
        X_STRIDE_B: tl.constexpr,
        Q_STRIDE_B: tl.constexpr,
        Q_STRIDE_S: tl.constexpr,
        Q_STRIDE_D: tl.constexpr,
        NODE_STRIDE_B: tl.constexpr,
        TILE_STRIDE_B: tl.constexpr,
        TILE_STRIDE_T: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        candidate = tl.program_id(0)
        tile = tl.program_id(1)

        sample = tile * BLOCK_S + tl.arange(0, BLOCK_S)
        coeff = tl.arange(0, BLOCK_K)
        joint = tl.arange(0, BLOCK_D)
        mask_s = sample < S
        mask_k = coeff < K
        mask_d = joint < D

        c = tl.load(
            coefficients + candidate * C_STRIDE_B + coeff,
            mask=mask_k,
            other=0.0,
        )
        basis_base = (
            sample[:, None] * B_STRIDE_S
            + coeff[None, :] * B_STRIDE_K
        )
        basis_mask = mask_s[:, None] & mask_k[None, :]
        b0 = tl.load(B + basis_base, mask=basis_mask, other=0.0)
        b1 = tl.load(B_s + basis_base, mask=basis_mask, other=0.0)
        b2 = tl.load(B_ss + basis_base, mask=basis_mask, other=0.0)
        xv = tl.sum(b0 * c[None, :], axis=1)
        xsv = tl.sum(b1 * c[None, :], axis=1)
        xssv = tl.sum(b2 * c[None, :], axis=1)
        tl.store(
            x_out + candidate * X_STRIDE_B + sample,
            xv,
            mask=mask_s,
        )

        speed = tl.sqrt(tl.maximum(xv, 0.0))
        sddot = 0.5 * xsv
        sdddot = 0.5 * xssv * speed

        path_mask = mask_s[:, None] & mask_d[None, :]
        path_base = (
            sample[:, None] * PATH_STRIDE_S
            + joint[None, :] * PATH_STRIDE_D
        )
        qs = tl.load(q_s + path_base, mask=path_mask, other=0.0)
        qss = tl.load(q_ss + path_base, mask=path_mask, other=0.0)
        qsss = tl.load(q_sss + path_base, mask=path_mask, other=0.0)

        speed2 = speed * speed
        speed3 = speed2 * speed
        qd_v = qs * speed[:, None]
        qdd_v = qss * xv[:, None] + qs * sddot[:, None]
        qddd_v = (
            qsss * speed3[:, None]
            + 3.0 * qss * (sddot * speed)[:, None]
            + qs * sdddot[:, None]
        )

        q_base = (
            candidate * Q_STRIDE_B
            + sample[:, None] * Q_STRIDE_S
            + joint[None, :] * Q_STRIDE_D
        )
        tl.store(qd + q_base, qd_v, mask=path_mask)
        tl.store(qdd + q_base, qdd_v, mask=path_mask)

        inf = float("inf")
        v_lim = tl.load(
            velocity_limit + joint,
            mask=mask_d,
            other=inf,
        )[None, :]
        a_lim = tl.load(
            acceleration_limit + joint,
            mask=mask_d,
            other=inf,
        )[None, :]
        j_lim = tl.load(
            jerk_limit + joint,
            mask=mask_d,
            other=inf,
        )[None, :]

        zero = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
        v_excess = tl.maximum(tl.abs(qd_v) - v_lim, zero)
        a_excess = tl.maximum(tl.abs(qdd_v) - a_lim, zero)
        j_excess = tl.maximum(tl.abs(qddd_v) - j_lim, zero)

        v_node = tl.max(v_excess, axis=1)
        a_node = tl.max(a_excess, axis=1)
        j_node = tl.max(j_excess, axis=1)
        combined = tl.maximum(v_node, tl.maximum(a_node, j_node))

        tl.store(
            node_max + candidate * NODE_STRIDE_B + sample,
            combined,
            mask=mask_s,
        )

        invalid = -float("inf")
        v_tile = tl.max(tl.where(mask_s, v_node, invalid), axis=0)
        a_tile = tl.max(tl.where(mask_s, a_node, invalid), axis=0)
        j_tile = tl.max(tl.where(mask_s, j_node, invalid), axis=0)
        out = (
            tile_metrics
            + candidate * TILE_STRIDE_B
            + tile * TILE_STRIDE_T
        )
        tl.store(out + 0, v_tile)
        tl.store(out + 1, a_tile)
        tl.store(out + 2, j_tile)


    @triton.jit
    def _timing_kinematic_node_reduce_kernel(
        x,
        x_s,
        x_ss,
        q_s,
        q_ss,
        q_sss,
        velocity_limit,
        acceleration_limit,
        jerk_limit,
        qd,
        qdd,
        node_max,
        tile_metrics,
        S: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        X_STRIDE_B: tl.constexpr,
        PATH_STRIDE_S: tl.constexpr,
        PATH_STRIDE_D: tl.constexpr,
        Q_STRIDE_B: tl.constexpr,
        Q_STRIDE_S: tl.constexpr,
        Q_STRIDE_D: tl.constexpr,
        NODE_STRIDE_B: tl.constexpr,
        TILE_STRIDE_B: tl.constexpr,
        TILE_STRIDE_T: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        candidate = tl.program_id(0)
        tile = tl.program_id(1)

        sample = tile * BLOCK_S + tl.arange(0, BLOCK_S)
        joint = tl.arange(0, BLOCK_D)
        mask_s = sample < S
        mask_d = joint < D
        mask = mask_s[:, None] & mask_d[None, :]

        timing_base = candidate * X_STRIDE_B + sample
        xv = tl.load(x + timing_base, mask=mask_s, other=0.0)
        xsv = tl.load(x_s + timing_base, mask=mask_s, other=0.0)
        xssv = tl.load(x_ss + timing_base, mask=mask_s, other=0.0)
        speed = tl.sqrt(tl.maximum(xv, 0.0))
        sddot = 0.5 * xsv
        sdddot = 0.5 * xssv * speed

        path_base = (
            sample[:, None] * PATH_STRIDE_S
            + joint[None, :] * PATH_STRIDE_D
        )
        qs = tl.load(q_s + path_base, mask=mask, other=0.0)
        qss = tl.load(q_ss + path_base, mask=mask, other=0.0)
        qsss = tl.load(q_sss + path_base, mask=mask, other=0.0)

        speed2 = speed * speed
        speed3 = speed2 * speed
        qd_v = qs * speed[:, None]
        qdd_v = (
            qss * xv[:, None]
            + qs * sddot[:, None]
        )
        qddd_v = (
            qsss * speed3[:, None]
            + 3.0 * qss * (sddot * speed)[:, None]
            + qs * sdddot[:, None]
        )

        q_base = (
            candidate * Q_STRIDE_B
            + sample[:, None] * Q_STRIDE_S
            + joint[None, :] * Q_STRIDE_D
        )
        tl.store(qd + q_base, qd_v, mask=mask)
        tl.store(qdd + q_base, qdd_v, mask=mask)

        inf = float("inf")
        v_lim = tl.load(
            velocity_limit + joint,
            mask=mask_d,
            other=inf,
        )[None, :]
        a_lim = tl.load(
            acceleration_limit + joint,
            mask=mask_d,
            other=inf,
        )[None, :]
        j_lim = tl.load(
            jerk_limit + joint,
            mask=mask_d,
            other=inf,
        )[None, :]

        zero = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
        v_excess = tl.maximum(tl.abs(qd_v) - v_lim, zero)
        a_excess = tl.maximum(tl.abs(qdd_v) - a_lim, zero)
        j_excess = tl.maximum(tl.abs(qddd_v) - j_lim, zero)

        v_node = tl.max(v_excess, axis=1)
        a_node = tl.max(a_excess, axis=1)
        j_node = tl.max(j_excess, axis=1)
        combined = tl.maximum(v_node, tl.maximum(a_node, j_node))

        tl.store(
            node_max
            + candidate * NODE_STRIDE_B
            + sample,
            combined,
            mask=mask_s,
        )

        invalid = -float("inf")
        v_tile = tl.max(tl.where(mask_s, v_node, invalid), axis=0)
        a_tile = tl.max(tl.where(mask_s, a_node, invalid), axis=0)
        j_tile = tl.max(tl.where(mask_s, j_node, invalid), axis=0)
        out = (
            tile_metrics
            + candidate * TILE_STRIDE_B
            + tile * TILE_STRIDE_T
        )
        tl.store(out + 0, v_tile)
        tl.store(out + 1, a_tile)
        tl.store(out + 2, j_tile)


    @triton.jit
    def _timing_kinematic_finalize_kernel(
        x,
        s,
        node_max,
        tile_metrics,
        dt,
        interval,
        output,
        S: tl.constexpr,
        TILES: tl.constexpr,
        X_STRIDE_B: tl.constexpr,
        NODE_STRIDE_B: tl.constexpr,
        TILE_STRIDE_B: tl.constexpr,
        TILE_STRIDE_T: tl.constexpr,
        DT_STRIDE_B: tl.constexpr,
        INTERVAL_STRIDE_B: tl.constexpr,
        OUT_STRIDE_M: tl.constexpr,
        OUT_STRIDE_B: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        candidate = tl.program_id(0)

        sample = tl.arange(0, BLOCK_S)
        mask_i = sample < (S - 1)
        current = tl.load(
            node_max + candidate * NODE_STRIDE_B + sample,
            mask=mask_i,
            other=0.0,
        )
        nxt = tl.load(
            node_max + candidate * NODE_STRIDE_B + sample + 1,
            mask=mask_i,
            other=0.0,
        )
        tl.store(
            interval
            + candidate * INTERVAL_STRIDE_B
            + sample,
            tl.maximum(current, nxt),
            mask=mask_i,
        )

        x0 = tl.load(
            x + candidate * X_STRIDE_B + sample,
            mask=mask_i,
            other=0.0,
        )
        x1 = tl.load(
            x + candidate * X_STRIDE_B + sample + 1,
            mask=mask_i,
            other=0.0,
        )
        s0 = tl.load(s + sample, mask=mask_i, other=0.0)
        s1 = tl.load(s + sample + 1, mask=mask_i, other=0.0)
        speed0 = tl.sqrt(tl.maximum(x0, 0.0))
        speed1 = tl.sqrt(tl.maximum(x1, 0.0))
        dt_v = 2.0 * (s1 - s0) / (speed0 + speed1)
        tl.store(
            dt + candidate * DT_STRIDE_B + sample,
            dt_v,
            mask=mask_i,
        )

        tile = tl.arange(0, BLOCK_T)
        mask_t = tile < TILES
        base = (
            tile_metrics
            + candidate * TILE_STRIDE_B
            + tile * TILE_STRIDE_T
        )
        invalid = -float("inf")
        v = tl.load(base + 0, mask=mask_t, other=invalid)
        a = tl.load(base + 1, mask=mask_t, other=invalid)
        j = tl.load(base + 2, mask=mask_t, other=invalid)
        tl.store(
            output + 0 * OUT_STRIDE_M + candidate * OUT_STRIDE_B,
            tl.max(v, axis=0),
        )
        tl.store(
            output + 1 * OUT_STRIDE_M + candidate * OUT_STRIDE_B,
            tl.max(a, axis=0),
        )
        tl.store(
            output + 2 * OUT_STRIDE_M + candidate * OUT_STRIDE_B,
            tl.max(j, axis=0),
        )


    @triton.jit
    def _torque_final_tile_kernel(
        tau,
        x,
        dt,
        kinematic_interval,
        torque_limit,
        torque_rate_limit,
        tile_stats,
        interval_out,
        violating,
        tolerance,
        S: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        TAU_STRIDE_B: tl.constexpr,
        TAU_STRIDE_S: tl.constexpr,
        TAU_STRIDE_D: tl.constexpr,
        X_STRIDE_B: tl.constexpr,
        DT_STRIDE_B: tl.constexpr,
        KIN_INTERVAL_STRIDE_B: tl.constexpr,
        TILE_STRIDE_B: tl.constexpr,
        TILE_STRIDE_T: tl.constexpr,
        INTERVAL_STRIDE_B: tl.constexpr,
        VIOLATING_STRIDE_B: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        candidate = tl.program_id(0)
        tile = tl.program_id(1)
        sample = tile * BLOCK_S + tl.arange(0, BLOCK_S)
        joint = tl.arange(0, BLOCK_D)

        mask_s = sample < S
        mask_i = sample < (S - 1)
        mask_d = joint < D
        mask_tau = mask_s[:, None] & mask_d[None, :]
        mask_rate = mask_i[:, None] & mask_d[None, :]

        tau_base = (
            candidate * TAU_STRIDE_B
            + sample[:, None] * TAU_STRIDE_S
            + joint[None, :] * TAU_STRIDE_D
        )
        tau0 = tl.load(tau + tau_base, mask=mask_tau, other=0.0)
        tau1 = tl.load(
            tau + tau_base + TAU_STRIDE_S,
            mask=mask_rate,
            other=0.0,
        )

        inf = float("inf")
        tq_lim = tl.load(
            torque_limit + joint,
            mask=mask_d,
            other=inf,
        )[None, :]
        tr_lim = tl.load(
            torque_rate_limit + joint,
            mask=mask_d,
            other=inf,
        )[None, :]
        zero = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)

        torque_excess0 = tl.maximum(tl.abs(tau0) - tq_lim, zero)
        torque_node0 = tl.max(torque_excess0, axis=1)

        torque_excess1 = tl.maximum(tl.abs(tau1) - tq_lim, zero)
        torque_node1 = tl.max(torque_excess1, axis=1)

        dtv = tl.load(
            dt + candidate * DT_STRIDE_B + sample,
            mask=mask_i,
            other=1.0,
        )
        tau_rate = (tau1 - tau0) / dtv[:, None]
        rate_excess = tl.maximum(tl.abs(tau_rate) - tr_lim, zero)
        rate_node = tl.max(rate_excess, axis=1)

        kin = tl.load(
            kinematic_interval
            + candidate * KIN_INTERVAL_STRIDE_B
            + sample,
            mask=mask_i,
            other=0.0,
        )
        combined = tl.maximum(
            kin,
            tl.maximum(
                rate_node,
                tl.maximum(torque_node0, torque_node1),
            ),
        )
        tl.store(
            interval_out
            + candidate * INTERVAL_STRIDE_B
            + sample,
            combined,
            mask=mask_i,
        )
        tl.store(
            violating
            + candidate * VIOLATING_STRIDE_B
            + sample,
            combined > tolerance,
            mask=mask_i,
        )

        xv = tl.load(
            x + candidate * X_STRIDE_B + sample,
            mask=mask_s,
            other=0.0,
        )
        negative = tl.maximum(-xv, 0.0)
        invalid_x = (~(xv == xv)) | (tl.abs(xv) == float("inf"))
        invalid_dt = (
            (~(dtv == dtv))
            | (tl.abs(dtv) == float("inf"))
            | (dtv <= 0.0)
        )

        invalid = -float("inf")
        torque_tile = tl.max(
            tl.where(mask_s, torque_node0, invalid),
            axis=0,
        )
        rate_tile = tl.max(
            tl.where(mask_i, rate_node, invalid),
            axis=0,
        )
        negative_tile = tl.max(
            tl.where(mask_s, negative, invalid),
            axis=0,
        )
        invalid_tile = tl.max(
            tl.where(
                mask_s,
                invalid_x.to(tl.int32),
                0,
            )
            | tl.where(
                mask_i,
                invalid_dt.to(tl.int32),
                0,
            ),
            axis=0,
        )

        out = (
            tile_stats
            + candidate * TILE_STRIDE_B
            + tile * TILE_STRIDE_T
        )
        tl.store(out + 0, torque_tile)
        tl.store(out + 1, rate_tile)
        tl.store(out + 2, negative_tile)
        tl.store(out + 3, invalid_tile.to(tl.float32))


    @triton.jit
    def _torque_final_finish_kernel(
        x,
        tile_stats,
        requested_endpoints,
        velocity,
        acceleration,
        jerk,
        torque,
        torque_rate,
        maximum,
        feasible,
        tolerance,
        S: tl.constexpr,
        TILES: tl.constexpr,
        X_STRIDE_B: tl.constexpr,
        TILE_STRIDE_B: tl.constexpr,
        TILE_STRIDE_T: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        candidate = tl.program_id(0)
        tile = tl.arange(0, BLOCK_T)
        mask_t = tile < TILES
        base = (
            tile_stats
            + candidate * TILE_STRIDE_B
            + tile * TILE_STRIDE_T
        )
        invalid = -float("inf")
        tq_tiles = tl.load(
            base + 0,
            mask=mask_t,
            other=invalid,
        )
        tr_tiles = tl.load(
            base + 1,
            mask=mask_t,
            other=invalid,
        )
        neg_tiles = tl.load(
            base + 2,
            mask=mask_t,
            other=invalid,
        )
        invalid_tiles = tl.load(
            base + 3,
            mask=mask_t,
            other=0.0,
        )

        tq = tl.max(tq_tiles, axis=0)
        tr = tl.max(tr_tiles, axis=0)
        negative = tl.max(neg_tiles, axis=0)
        invalid_any = tl.max(invalid_tiles, axis=0) > 0.0

        x0 = tl.load(x + candidate * X_STRIDE_B)
        x1 = tl.load(x + candidate * X_STRIDE_B + (S - 1))
        r0 = tl.load(requested_endpoints + 0)
        r1 = tl.load(requested_endpoints + 1)
        endpoint = tl.maximum(tl.abs(x0 - r0), tl.abs(x1 - r1))

        v = tl.load(velocity + candidate)
        a = tl.load(acceleration + candidate)
        j = tl.load(jerk + candidate)

        value = tl.maximum(v, a)
        value = tl.maximum(value, j)
        value = tl.maximum(value, tq)
        value = tl.maximum(value, tr)
        value = tl.maximum(value, negative)
        value = tl.maximum(value, endpoint)
        value = tl.where(invalid_any, float("inf"), value)

        tl.store(torque + candidate, tq)
        tl.store(torque_rate + candidate, tr)
        tl.store(maximum + candidate, value)
        tl.store(feasible + candidate, value <= tolerance)


    @triton.jit
    def _final_validation_tile_kernel(
        x,
        dt,
        interval,
        tile_stats,
        violating,
        tolerance,
        S: tl.constexpr,
        X_STRIDE_B: tl.constexpr,
        DT_STRIDE_B: tl.constexpr,
        INTERVAL_STRIDE_B: tl.constexpr,
        TILE_STRIDE_B: tl.constexpr,
        TILE_STRIDE_T: tl.constexpr,
        VIOLATING_STRIDE_B: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        candidate = tl.program_id(0)
        tile = tl.program_id(1)
        sample = tile * BLOCK_S + tl.arange(0, BLOCK_S)

        mask_x = sample < S
        xv = tl.load(
            x + candidate * X_STRIDE_B + sample,
            mask=mask_x,
            other=0.0,
        )
        negative = tl.maximum(-xv, 0.0)
        invalid_x = (~(xv == xv)) | (tl.abs(xv) == float("inf"))
        neg_tile = tl.max(
            tl.where(mask_x, negative, -float("inf")),
            axis=0,
        )
        invalid_x_tile = tl.max(
            tl.where(mask_x, invalid_x.to(tl.int32), 0),
            axis=0,
        )

        mask_i = sample < (S - 1)
        dtv = tl.load(
            dt + candidate * DT_STRIDE_B + sample,
            mask=mask_i,
            other=1.0,
        )
        invalid_dt = (
            (~(dtv == dtv))
            | (tl.abs(dtv) == float("inf"))
            | (dtv <= 0.0)
        )
        invalid_dt_tile = tl.max(
            tl.where(mask_i, invalid_dt.to(tl.int32), 0),
            axis=0,
        )

        interval_v = tl.load(
            interval + candidate * INTERVAL_STRIDE_B + sample,
            mask=mask_i,
            other=0.0,
        )
        tl.store(
            violating
            + candidate * VIOLATING_STRIDE_B
            + sample,
            interval_v > tolerance,
            mask=mask_i,
        )

        out = (
            tile_stats
            + candidate * TILE_STRIDE_B
            + tile * TILE_STRIDE_T
        )
        tl.store(out + 0, neg_tile)
        tl.store(
            out + 1,
            (invalid_x_tile | invalid_dt_tile).to(tl.float32),
        )


    @triton.jit
    def _final_validation_finish_kernel(
        x,
        tile_stats,
        requested_endpoints,
        velocity,
        acceleration,
        jerk,
        torque,
        torque_rate,
        maximum,
        feasible,
        tolerance,
        S: tl.constexpr,
        TILES: tl.constexpr,
        X_STRIDE_B: tl.constexpr,
        TILE_STRIDE_B: tl.constexpr,
        TILE_STRIDE_T: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        candidate = tl.program_id(0)
        tile = tl.arange(0, BLOCK_T)
        mask_t = tile < TILES
        base = (
            tile_stats
            + candidate * TILE_STRIDE_B
            + tile * TILE_STRIDE_T
        )
        neg = tl.load(
            base + 0,
            mask=mask_t,
            other=-float("inf"),
        )
        invalid = tl.load(
            base + 1,
            mask=mask_t,
            other=0.0,
        )
        negative = tl.max(neg, axis=0)
        invalid_any = tl.max(invalid, axis=0) > 0.0

        x0 = tl.load(x + candidate * X_STRIDE_B)
        x1 = tl.load(x + candidate * X_STRIDE_B + (S - 1))
        r0 = tl.load(requested_endpoints + 0)
        r1 = tl.load(requested_endpoints + 1)
        endpoint = tl.maximum(tl.abs(x0 - r0), tl.abs(x1 - r1))

        v = tl.load(velocity + candidate)
        a = tl.load(acceleration + candidate)
        j = tl.load(jerk + candidate)
        tq = tl.load(torque + candidate)
        tr = tl.load(torque_rate + candidate)

        value = tl.maximum(v, a)
        value = tl.maximum(value, j)
        value = tl.maximum(value, tq)
        value = tl.maximum(value, tr)
        value = tl.maximum(value, negative)
        value = tl.maximum(value, endpoint)
        value = tl.where(invalid_any, float("inf"), value)

        tl.store(maximum + candidate, value)
        tl.store(feasible + candidate, value <= tolerance)


@torch.no_grad()
def fused_kinematic_reductions(
    qd: Tensor,
    qdd: Tensor,
    qddd: Tensor,
    velocity_limit: Tensor,
    acceleration_limit: Tensor,
    jerk_limit: Tensor,
    *,
    block_samples: int = 256,
) -> tuple[Tensor, Tensor]:
    """Return exact node-family maxima and interval V/A/J violation envelope.

    qd/qdd/qddd are consumed exactly as provided.  The routine changes only
    how max/clamp reductions are scheduled.
    """

    if triton is None:
        raise RuntimeError("Triton is not installed")
    if qd.shape != qdd.shape or qd.shape != qddd.shape:
        raise ValueError("qd/qdd/qddd shapes must match")
    if qd.ndim != 3:
        raise ValueError("qd/qdd/qddd must be [batch, samples, dof]")
    if qd.dtype != torch.float32 or not qd.is_cuda:
        raise ValueError("fused kinematic reductions require CUDA FP32")
    if not qd.is_contiguous() or not qdd.is_contiguous() or not qddd.is_contiguous():
        raise ValueError("fused kinematic reductions require contiguous inputs")

    batch, samples, dof = map(int, qd.shape)
    if dof < 1 or samples < 2:
        raise ValueError("invalid dense kinematic shape")
    if dof > 16:
        raise ValueError("fused kinematic reductions currently support <=16 dof")

    tiles = (samples + block_samples - 1) // block_samples
    block_d = triton.next_power_of_2(dof)
    node_max = torch.empty(
        (batch, samples),
        device=qd.device,
        dtype=qd.dtype,
    )
    tile_metrics = torch.empty(
        (batch, tiles, 3),
        device=qd.device,
        dtype=qd.dtype,
    )
    interval = torch.empty(
        (batch, samples - 1),
        device=qd.device,
        dtype=qd.dtype,
    )
    # Metric-major layout keeps velocity/acceleration/jerk outputs
    # contiguous as output[0], output[1], output[2]. This avoids strided
    # gathers in the validator's final reduction stage.
    output = torch.empty(
        (3, batch),
        device=qd.device,
        dtype=qd.dtype,
    )

    _kinematic_node_reduce_kernel[(batch, tiles)](
        qd,
        qdd,
        qddd,
        velocity_limit,
        acceleration_limit,
        jerk_limit,
        node_max,
        tile_metrics,
        S=samples,
        D=dof,
        BLOCK_D=block_d,
        Q_STRIDE_B=qd.stride(0),
        Q_STRIDE_S=qd.stride(1),
        Q_STRIDE_D=qd.stride(2),
        NODE_STRIDE_B=node_max.stride(0),
        TILE_STRIDE_B=tile_metrics.stride(0),
        TILE_STRIDE_T=tile_metrics.stride(1),
        BLOCK_S=block_samples,
        num_warps=4,
    )
    _kinematic_finalize_kernel[(batch,)](
        node_max,
        tile_metrics,
        interval,
        output,
        S=samples,
        TILES=tiles,
        NODE_STRIDE_B=node_max.stride(0),
        TILE_STRIDE_B=tile_metrics.stride(0),
        TILE_STRIDE_T=tile_metrics.stride(1),
        INTERVAL_STRIDE_B=interval.stride(0),
        OUT_STRIDE_M=output.stride(0),
        OUT_STRIDE_B=output.stride(1),
        BLOCK_S=triton.next_power_of_2(samples - 1),
        BLOCK_T=triton.next_power_of_2(tiles),
        num_warps=4,
    )
    return output, interval




@torch.no_grad()
def fused_basis_timing_kinematics_reductions(
    coefficients: Tensor,
    B: Tensor,
    B_s: Tensor,
    B_ss: Tensor,
    s: Tensor,
    q_s: Tensor,
    q_ss: Tensor,
    q_sss: Tensor,
    velocity_limit: Tensor,
    acceleration_limit: Tensor,
    jerk_limit: Tensor,
    *,
    block_samples: int = 256,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fuse B-spline timing evaluation with dense kinematics and V/A/J reductions.

    Returns x for downstream profile/final checks, contiguous qd/qdd for RNEA,
    interval durations dt, metric-major V/A/J maxima, and the interval
    kinematic violation envelope. x_s/x_ss/qddd are never materialized.
    """

    if triton is None:
        raise RuntimeError("Triton is not installed")
    if coefficients.ndim != 2:
        raise ValueError("coefficients must be [batch, K]")
    batch, k = map(int, coefficients.shape)
    if B.ndim != 2 or B_s.shape != B.shape or B_ss.shape != B.shape:
        raise ValueError("B/B_s/B_ss must be matching [samples, K]")
    samples, basis_k = map(int, B.shape)
    if basis_k != k:
        raise ValueError("basis coefficient count must match coefficients")
    if q_s.shape != q_ss.shape or q_s.shape != q_sss.shape or q_s.ndim != 2:
        raise ValueError("path derivatives must be matching [samples, dof]")
    path_samples, dof = map(int, q_s.shape)
    if path_samples != samples:
        raise ValueError("timing and path sample counts must match")
    if not supports_fused_basis_timing_shape(k, dof, samples):
        raise ValueError("unsupported fused basis timing-kinematic shape")
    if coefficients.dtype != torch.float32 or not coefficients.is_cuda:
        raise ValueError("fused basis timing kinematics require CUDA FP32")

    tensors = (
        B,
        B_s,
        B_ss,
        s,
        q_s,
        q_ss,
        q_sss,
        velocity_limit,
        acceleration_limit,
        jerk_limit,
    )
    if any(t.device != coefficients.device for t in tensors):
        raise ValueError("all fused basis timing inputs must share device")
    if any(t.dtype != coefficients.dtype for t in tensors):
        raise ValueError("all fused basis timing numeric inputs must share dtype")
    if (
        not coefficients.is_contiguous()
        or not B.is_contiguous()
        or not B_s.is_contiguous()
        or not B_ss.is_contiguous()
        or not q_s.is_contiguous()
        or not q_ss.is_contiguous()
        or not q_sss.is_contiguous()
    ):
        raise ValueError("fused basis timing inputs must be contiguous")
    if s.shape != (samples,):
        raise ValueError("s must have shape [samples]")
    if any(limit.shape != (dof,) for limit in (
        velocity_limit,
        acceleration_limit,
        jerk_limit,
    )):
        raise ValueError("V/A/J limits must have shape [dof]")

    x = torch.empty(
        (batch, samples),
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    qd = torch.empty(
        (batch, samples, dof),
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    qdd = torch.empty_like(qd)
    dt = torch.empty(
        (batch, samples - 1),
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    node_max = torch.empty(
        (batch, samples),
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    tiles = (samples + block_samples - 1) // block_samples
    tile_metrics = torch.empty(
        (batch, tiles, 3),
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    interval = torch.empty(
        (batch, samples - 1),
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    output = torch.empty(
        (3, batch),
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    block_k = triton.next_power_of_2(k)
    block_d = triton.next_power_of_2(dof)

    _basis_timing_kinematic_node_reduce_kernel[(batch, tiles)](
        coefficients,
        B,
        B_s,
        B_ss,
        q_s,
        q_ss,
        q_sss,
        velocity_limit,
        acceleration_limit,
        jerk_limit,
        x,
        qd,
        qdd,
        node_max,
        tile_metrics,
        S=samples,
        K=k,
        D=dof,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        C_STRIDE_B=coefficients.stride(0),
        B_STRIDE_S=B.stride(0),
        B_STRIDE_K=B.stride(1),
        PATH_STRIDE_S=q_s.stride(0),
        PATH_STRIDE_D=q_s.stride(1),
        X_STRIDE_B=x.stride(0),
        Q_STRIDE_B=qd.stride(0),
        Q_STRIDE_S=qd.stride(1),
        Q_STRIDE_D=qd.stride(2),
        NODE_STRIDE_B=node_max.stride(0),
        TILE_STRIDE_B=tile_metrics.stride(0),
        TILE_STRIDE_T=tile_metrics.stride(1),
        BLOCK_S=block_samples,
        num_warps=4,
    )
    _timing_kinematic_finalize_kernel[(batch,)](
        x,
        s,
        node_max,
        tile_metrics,
        dt,
        interval,
        output,
        S=samples,
        TILES=tiles,
        X_STRIDE_B=x.stride(0),
        NODE_STRIDE_B=node_max.stride(0),
        TILE_STRIDE_B=tile_metrics.stride(0),
        TILE_STRIDE_T=tile_metrics.stride(1),
        DT_STRIDE_B=dt.stride(0),
        INTERVAL_STRIDE_B=interval.stride(0),
        OUT_STRIDE_M=output.stride(0),
        OUT_STRIDE_B=output.stride(1),
        BLOCK_S=triton.next_power_of_2(samples - 1),
        BLOCK_T=triton.next_power_of_2(tiles),
        num_warps=4,
    )
    return x, qd, qdd, dt, output, interval


@torch.no_grad()
def fused_timing_kinematics_reductions(
    x: Tensor,
    x_s: Tensor,
    x_ss: Tensor,
    s: Tensor,
    q_s: Tensor,
    q_ss: Tensor,
    q_sss: Tensor,
    velocity_limit: Tensor,
    acceleration_limit: Tensor,
    jerk_limit: Tensor,
    *,
    block_samples: int = 256,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fuse dense kinematics and V/A/J reductions from timing derivatives.

    Returns contiguous qd and qdd for direct RNEA, interval durations dt,
    metric-major [velocity, acceleration, jerk] maxima, and the exact
    interval V/A/J violation envelope.  qddd is evaluated on the fly and is
    never materialized.
    """

    if triton is None:
        raise RuntimeError("Triton is not installed")
    if x.shape != x_s.shape or x.shape != x_ss.shape or x.ndim != 2:
        raise ValueError("x/x_s/x_ss must be matching [batch, samples]")
    if q_s.shape != q_ss.shape or q_s.shape != q_sss.shape or q_s.ndim != 2:
        raise ValueError("path derivatives must be matching [samples, dof]")
    batch, samples = map(int, x.shape)
    path_samples, dof = map(int, q_s.shape)
    if path_samples != samples:
        raise ValueError("timing and path sample counts must match")
    if dof < 1 or dof > 16 or samples < 2:
        raise ValueError("unsupported fused timing-kinematic shape")
    if x.dtype != torch.float32 or not x.is_cuda:
        raise ValueError("fused timing kinematics require CUDA FP32")
    tensors = (
        x_s,
        x_ss,
        s,
        q_s,
        q_ss,
        q_sss,
        velocity_limit,
        acceleration_limit,
        jerk_limit,
    )
    if any(t.device != x.device for t in tensors):
        raise ValueError("all fused timing-kinematic inputs must share device")
    if any(t.dtype != x.dtype for t in tensors):
        raise ValueError("all fused timing-kinematic inputs must share dtype")
    if (
        not x.is_contiguous()
        or not x_s.is_contiguous()
        or not x_ss.is_contiguous()
        or not q_s.is_contiguous()
        or not q_ss.is_contiguous()
        or not q_sss.is_contiguous()
    ):
        raise ValueError("fused timing-kinematic inputs must be contiguous")
    if s.shape != (samples,):
        raise ValueError("s must have shape [samples]")
    if any(limit.shape != (dof,) for limit in (
        velocity_limit,
        acceleration_limit,
        jerk_limit,
    )):
        raise ValueError("V/A/J limits must have shape [dof]")

    qd = torch.empty(
        (batch, samples, dof),
        device=x.device,
        dtype=x.dtype,
    )
    qdd = torch.empty_like(qd)
    dt = torch.empty(
        (batch, samples - 1),
        device=x.device,
        dtype=x.dtype,
    )
    node_max = torch.empty(
        (batch, samples),
        device=x.device,
        dtype=x.dtype,
    )
    tiles = (samples + block_samples - 1) // block_samples
    tile_metrics = torch.empty(
        (batch, tiles, 3),
        device=x.device,
        dtype=x.dtype,
    )
    interval = torch.empty(
        (batch, samples - 1),
        device=x.device,
        dtype=x.dtype,
    )
    output = torch.empty(
        (3, batch),
        device=x.device,
        dtype=x.dtype,
    )
    block_d = triton.next_power_of_2(dof)

    _timing_kinematic_node_reduce_kernel[(batch, tiles)](
        x,
        x_s,
        x_ss,
        q_s,
        q_ss,
        q_sss,
        velocity_limit,
        acceleration_limit,
        jerk_limit,
        qd,
        qdd,
        node_max,
        tile_metrics,
        S=samples,
        D=dof,
        BLOCK_D=block_d,
        X_STRIDE_B=x.stride(0),
        PATH_STRIDE_S=q_s.stride(0),
        PATH_STRIDE_D=q_s.stride(1),
        Q_STRIDE_B=qd.stride(0),
        Q_STRIDE_S=qd.stride(1),
        Q_STRIDE_D=qd.stride(2),
        NODE_STRIDE_B=node_max.stride(0),
        TILE_STRIDE_B=tile_metrics.stride(0),
        TILE_STRIDE_T=tile_metrics.stride(1),
        BLOCK_S=block_samples,
        num_warps=4,
    )
    _timing_kinematic_finalize_kernel[(batch,)](
        x,
        s,
        node_max,
        tile_metrics,
        dt,
        interval,
        output,
        S=samples,
        TILES=tiles,
        X_STRIDE_B=x.stride(0),
        NODE_STRIDE_B=node_max.stride(0),
        TILE_STRIDE_B=tile_metrics.stride(0),
        TILE_STRIDE_T=tile_metrics.stride(1),
        DT_STRIDE_B=dt.stride(0),
        INTERVAL_STRIDE_B=interval.stride(0),
        OUT_STRIDE_M=output.stride(0),
        OUT_STRIDE_B=output.stride(1),
        BLOCK_S=triton.next_power_of_2(samples - 1),
        BLOCK_T=triton.next_power_of_2(tiles),
        num_warps=4,
    )
    return qd, qdd, dt, output, interval


@torch.no_grad()
def fused_torque_final_validation_reductions(
    tau: Tensor,
    x: Tensor,
    dt: Tensor,
    kinematic_interval: Tensor,
    velocity: Tensor,
    acceleration: Tensor,
    jerk: Tensor,
    torque_limit: Tensor,
    torque_rate_limit: Tensor,
    requested_endpoints: Tensor,
    *,
    tolerance: float,
    block_samples: int = 256,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fuse torque, torque-rate, interval, and final validation reductions."""

    if triton is None:
        raise RuntimeError("Triton is not installed")
    if tau.ndim != 3:
        raise ValueError("tau must be [batch, samples, dof]")
    batch, samples, dof = map(int, tau.shape)
    if x.shape != (batch, samples):
        raise ValueError("x shape is incompatible with tau")
    if dt.shape != (batch, samples - 1):
        raise ValueError("dt shape is incompatible with tau")
    if kinematic_interval.shape != (batch, samples - 1):
        raise ValueError("kinematic interval shape is incompatible with tau")
    if any(v.shape != (batch,) for v in (velocity, acceleration, jerk)):
        raise ValueError("V/A/J violation vectors must have shape [batch]")
    if torque_limit.shape != (dof,) or torque_rate_limit.shape != (dof,):
        raise ValueError("torque limits must have shape [dof]")
    if requested_endpoints.shape != (2,):
        raise ValueError("requested_endpoints must have shape [2]")
    if tau.dtype != torch.float32 or not tau.is_cuda:
        raise ValueError("fused torque final reductions require CUDA FP32")
    tensors = (
        x,
        dt,
        kinematic_interval,
        velocity,
        acceleration,
        jerk,
        torque_limit,
        torque_rate_limit,
        requested_endpoints,
    )
    if any(t.device != tau.device for t in tensors):
        raise ValueError("all fused torque-final inputs must share device")
    if any(t.dtype != tau.dtype for t in tensors if t.dtype != torch.bool):
        raise ValueError("all fused torque-final numeric inputs must share dtype")
    if (
        not tau.is_contiguous()
        or not x.is_contiguous()
        or not dt.is_contiguous()
        or not kinematic_interval.is_contiguous()
    ):
        raise ValueError("fused torque-final dense inputs must be contiguous")
    if dof < 1 or dof > 16 or samples < 2:
        raise ValueError("unsupported fused torque-final shape")

    tiles = (samples + block_samples - 1) // block_samples
    tile_stats = torch.empty(
        (batch, tiles, 4),
        device=tau.device,
        dtype=tau.dtype,
    )
    interval = torch.empty(
        (batch, samples - 1),
        device=tau.device,
        dtype=tau.dtype,
    )
    violating = torch.empty(
        (batch, samples - 1),
        device=tau.device,
        dtype=torch.bool,
    )
    torque = torch.empty(
        (batch,),
        device=tau.device,
        dtype=tau.dtype,
    )
    torque_rate = torch.empty_like(torque)
    maximum = torch.empty_like(torque)
    feasible = torch.empty(
        (batch,),
        device=tau.device,
        dtype=torch.bool,
    )
    block_d = triton.next_power_of_2(dof)

    _torque_final_tile_kernel[(batch, tiles)](
        tau,
        x,
        dt,
        kinematic_interval,
        torque_limit,
        torque_rate_limit,
        tile_stats,
        interval,
        violating,
        float(tolerance),
        S=samples,
        D=dof,
        BLOCK_D=block_d,
        TAU_STRIDE_B=tau.stride(0),
        TAU_STRIDE_S=tau.stride(1),
        TAU_STRIDE_D=tau.stride(2),
        X_STRIDE_B=x.stride(0),
        DT_STRIDE_B=dt.stride(0),
        KIN_INTERVAL_STRIDE_B=kinematic_interval.stride(0),
        TILE_STRIDE_B=tile_stats.stride(0),
        TILE_STRIDE_T=tile_stats.stride(1),
        INTERVAL_STRIDE_B=interval.stride(0),
        VIOLATING_STRIDE_B=violating.stride(0),
        BLOCK_S=block_samples,
        num_warps=4,
    )
    _torque_final_finish_kernel[(batch,)](
        x,
        tile_stats,
        requested_endpoints,
        velocity,
        acceleration,
        jerk,
        torque,
        torque_rate,
        maximum,
        feasible,
        float(tolerance),
        S=samples,
        TILES=tiles,
        X_STRIDE_B=x.stride(0),
        TILE_STRIDE_B=tile_stats.stride(0),
        TILE_STRIDE_T=tile_stats.stride(1),
        BLOCK_T=triton.next_power_of_2(tiles),
        num_warps=1,
    )
    return (
        torque,
        torque_rate,
        interval,
        maximum,
        feasible,
        violating,
    )


@torch.no_grad()
def fused_final_validation_reductions(
    x: Tensor,
    dt: Tensor,
    interval: Tensor,
    velocity: Tensor,
    acceleration: Tensor,
    jerk: Tensor,
    torque: Tensor,
    torque_rate: Tensor,
    requested_endpoints: Tensor,
    *,
    tolerance: float,
    block_samples: int = 256,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fuse the final physical-validation max/finite/mask reductions.

    Arithmetic matches the existing PyTorch tail: negative-x and endpoint
    violations are folded into the family maxima, invalid x/dt states force
    +inf, and interval thresholding is emitted in the same pass.
    """

    if triton is None:
        raise RuntimeError("Triton is not installed")
    if x.ndim != 2 or dt.ndim != 2 or interval.ndim != 2:
        raise ValueError("x/dt/interval must be batched 2-D tensors")
    batch, samples = map(int, x.shape)
    if dt.shape != (batch, samples - 1):
        raise ValueError("dt shape is incompatible with x")
    if interval.shape != (batch, samples - 1):
        raise ValueError("interval shape is incompatible with x")
    vectors = (
        velocity,
        acceleration,
        jerk,
        torque,
        torque_rate,
    )
    if any(v.shape != (batch,) for v in vectors):
        raise ValueError("violation vectors must have shape [batch]")
    if requested_endpoints.shape != (2,):
        raise ValueError("requested_endpoints must have shape [2]")
    if x.dtype != torch.float32 or not x.is_cuda:
        raise ValueError("fused final reductions require CUDA FP32")
    if (
        dt.device != x.device
        or interval.device != x.device
        or requested_endpoints.device != x.device
        or any(v.device != x.device for v in vectors)
    ):
        raise ValueError("all final-reduction inputs must share a CUDA device")
    if not x.is_contiguous() or not dt.is_contiguous() or not interval.is_contiguous():
        raise ValueError("x/dt/interval must be contiguous")

    tiles = (samples + block_samples - 1) // block_samples
    tile_stats = torch.empty(
        (batch, tiles, 2),
        device=x.device,
        dtype=x.dtype,
    )
    violating = torch.empty(
        (batch, samples - 1),
        device=x.device,
        dtype=torch.bool,
    )
    maximum = torch.empty(
        (batch,),
        device=x.device,
        dtype=x.dtype,
    )
    feasible = torch.empty(
        (batch,),
        device=x.device,
        dtype=torch.bool,
    )

    _final_validation_tile_kernel[(batch, tiles)](
        x,
        dt,
        interval,
        tile_stats,
        violating,
        float(tolerance),
        S=samples,
        X_STRIDE_B=x.stride(0),
        DT_STRIDE_B=dt.stride(0),
        INTERVAL_STRIDE_B=interval.stride(0),
        TILE_STRIDE_B=tile_stats.stride(0),
        TILE_STRIDE_T=tile_stats.stride(1),
        VIOLATING_STRIDE_B=violating.stride(0),
        BLOCK_S=block_samples,
        num_warps=4,
    )
    _final_validation_finish_kernel[(batch,)](
        x,
        tile_stats,
        requested_endpoints,
        velocity,
        acceleration,
        jerk,
        torque,
        torque_rate,
        maximum,
        feasible,
        float(tolerance),
        S=samples,
        TILES=tiles,
        X_STRIDE_B=x.stride(0),
        TILE_STRIDE_B=tile_stats.stride(0),
        TILE_STRIDE_T=tile_stats.stride(1),
        BLOCK_T=triton.next_power_of_2(tiles),
        num_warps=1,
    )
    return maximum, feasible, violating


__all__ = [
    "FUSED_BASIS_TIMING_MAX_COEFFICIENTS",
    "supports_fused_basis_timing_shape",
    "fused_kinematic_reductions",
    "fused_basis_timing_kinematics_reductions",
    "fused_timing_kinematics_reductions",
    "fused_torque_final_validation_reductions",
    "fused_final_validation_reductions",
]
