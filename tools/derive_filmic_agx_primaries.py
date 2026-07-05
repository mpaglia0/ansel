#!/usr/bin/env python3
"""
Fit the inset/rotation anchor constants of filmic RGB's "AgX" color science
(v8) — see doc/filmic-agx.md and filmic_agx_prepare_bracket() in
src/iop/filmicrgb.c, whose PROVISIONAL constants this script is meant to replace.

Model (mirrors the C pixel path, working profile = linear Rec2020):
  work RGB -> inset matrix -> per-channel [log2 encoding -> sigmoid spline ->
  hardness power] -> exact-inverse outset -> measure chroma / hue in Kirk Yrg.

Objectives:
  1. purity-vs-exposure: colors on the working-gamut boundary must reach
     achromatic (chroma ratio ~ 0) at the white end of the curve, monotonically;
  2. hue drift: dh/dEV in Yrg matches a chosen uniform target across the wheel
     (0 for a neutral character, or a uniform warm bias);
  3. hard constraints as penalties: displaced primaries stay inside the working
     gamut triangle with a margin (positivity of the bracket) and away from
     degeneracy (conditioning).

Usage:
  python3 tools/derive_filmic_agx_primaries.py --max-desat FRAC   (recommended)
  python3 tools/derive_filmic_agx_primaries.py --fit-priority
  python3 tools/derive_filmic_agx_primaries.py [--drift-target DEG_PER_EV]

The recommended mode is --max-desat : you state the maximum average chroma loss
over the priority set (skin + reflective) you accept, and the solver returns the
bracket with the best hue match at or below that budget. Sweep it to trace the
hue/chroma frontier. See its --help for the full method.

Prints the C arrays to paste into filmic_agx_prepare_bracket(). Requires
numpy + scipy (run with python3.12). Constants below are copied verbatim from the
C code so the model matches the pipeline bit-for-bit in spirit (float64 here,
float32 there).
"""

import argparse
import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------- constants
# copied from src/common/chromatic_adaptation.h and colorspaces_inline_conversions.h

XYZ_D50_to_D65_CAT16 = np.array([
    [9.89466254e-01, -4.00304626e-02, 4.40530317e-02],
    [-5.40518733e-03, 1.00666069e+00, -1.75551955e-03],
    [-4.03920992e-04, 1.50768030e-02, 1.30210211e+00]])

XYZ_D65_to_LMS_2006 = np.array([
    [0.257085, 0.859943, -0.031061],
    [-0.394427, 1.175800, 0.106423],
    [0.064856, -0.076250, 0.559067]])

filmlightRGB_to_LMS = np.array([
    [0.95, 0.38, 0.00],
    [0.05, 0.62, 0.03],
    [0.00, 0.00, 0.97]])
LMS_to_filmlightRGB = np.linalg.inv(filmlightRGB_to_LMS)

Y_2006_COEFFS = np.array([0.68990272, 0.34832189, 0.0])

# linear Rec2020, D50-adapted (Bradford), as computed by the LCMS path for the
# working profile. Close enough for anchor fitting; regenerate from the pipeline
# if exactness matters.
REC2020_TO_XYZ_D50 = np.array([
    [0.6734241, 0.1656411, 0.1251286],
    [0.2790177, 0.6753402, 0.0456377],
    [-0.0019300, 0.0299784, 0.7973330]])

# filmic defaults the anchors are fitted against — keep in sync with the C
# $DEFAULT values. The curve model is shared with the appearance-match harness
# (C-exact v3 node geometry + perceptual-sigmoid segments).
from derive_filmic_default_curve import curve_factory, GREY  # noqa: E402

BLACK_EV, WHITE_EV = -8.0, 4.0
CURVE_DEFAULTS = (1.18, 10.0, 0.0, 1.5, 1.5)  # contrast, latitude %, balance %, toe power, (shoulder is slope-matched)
CURVE = curve_factory(BLACK_EV, WHITE_EV, *CURVE_DEFAULTS, shoulder_slope_matched=True)

# ---------------------------------------------------------------- color helpers

def xyz_D50_to_Yrg(xyz):
    lms = XYZ_D65_to_LMS_2006 @ (XYZ_D50_to_D65_CAT16 @ xyz)
    Y = Y_2006_COEFFS @ lms
    a = lms.sum()
    rgb = LMS_to_filmlightRGB @ (lms / a if a != 0 else lms)
    return np.array([Y, rgb[0], rgb[1]])

def rgb_work_to_Yrg(rgb):
    return xyz_D50_to_Yrg(REC2020_TO_XYZ_D50 @ rgb)

WHITE_YRG = rgb_work_to_Yrg(np.ones(3))

def chroma_hue(Yrg):
    r, g = Yrg[1] - WHITE_YRG[1], Yrg[2] - WHITE_YRG[2]
    return np.hypot(r, g), np.arctan2(g, r)

# ---------------------------------------------------------------- curve model

def tone_map(rgb):
    dr = WHITE_EV - BLACK_EV
    out = np.empty(3)
    for c in range(3):
        x = (np.log2(max(rgb[c], 1e-10) / GREY) - BLACK_EV) / dr
        out[c] = CURVE(x)
    return out

# ---------------------------------------------------------------- bracket model

def bracket_matrices(insets, rotations):
    """inset matrix M (work->rendering) and its exact inverse, exactly as the C code."""
    white_xyz = REC2020_TO_XYZ_D50 @ np.ones(3)
    P_prime = np.empty((3, 3))
    for i in range(3):
        p_yrg = xyz_D50_to_Yrg(REC2020_TO_XYZ_D50[:, i])
        d = p_yrg[1:] - WHITE_YRG[1:]
        s = 1.0 - insets[i]
        ca, sa = np.cos(rotations[i]), np.sin(rotations[i])
        rot = np.array([[ca, -sa], [sa, ca]])
        rg = WHITE_YRG[1:] + s * (rot @ d)
        # Yrg -> XYZ D50 (inverse path of xyz_D50_to_Yrg)
        rgbn = np.array([rg[0], rg[1], 1.0 - rg[0] - rg[1]])
        lms = filmlightRGB_to_LMS @ rgbn
        lms *= p_yrg[0] / (Y_2006_COEFFS @ lms)
        P_prime[:, i] = np.linalg.inv(XYZ_D50_to_D65_CAT16) @ \
                        np.linalg.inv(XYZ_D65_to_LMS_2006) @ lms
    scale = np.linalg.solve(P_prime, white_xyz)
    P_inset = P_prime * scale[None, :]
    M = np.linalg.inv(REC2020_TO_XYZ_D50) @ P_inset
    return M, np.linalg.inv(M)

# ---------------------------------------------------------------- objectives

EVS = np.arange(-2.0, WHITE_EV + 0.51, 0.5)
HUE_STEPS = 24

def boundary_colors():
    """saturated colors on the working-gamut boundary, one channel at zero"""
    out = []
    for k in range(HUE_STEPS):
        h = 2 * np.pi * k / HUE_STEPS
        rgb = np.array([np.cos(h), np.cos(h - 2 * np.pi / 3), np.cos(h + 2 * np.pi / 3)])
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # in [0,1], min channel 0
        out.append(rgb)
    return out

SHOULDER_EV = BLACK_EV + CURVE.sh_x * (WHITE_EV - BLACK_EV)  # where compression starts

# ---------------------------------------------------------------- priority colors
# The colors whose fidelity is non-negotiable : human skin tones (database from
# src/common/color_vocabulary.c, CIE Lab under D65, avg ± 1.5 std corners) and
# in-gamut diffuse reflectances, both swept over the tonal placements a
# photographer may give them. Used to fit the outset recovery (see --fit-outset).

_SKIN_LAB = [  # L avg,std, a avg,std, b avg,std — see color_vocabulary.c for sources
    (60.9, 3.4, 7.0, 1.7, 15.0, 1.8), (61.9, 3.7, 7.1, 1.7, 17.4, 2.0),
    (60.6, 4.8, 6.5, 1.6, 16.4, 2.3), (63.0, 5.5, 5.6, 1.9, 14.0, 2.9),
    (56.4, 3.2, 11.7, 2.1, 16.3, 1.4), (56.8, 4.1, 11.6, 2.2, 17.7, 1.8),
    (56.1, 4.5, 11.3, 2.1, 16.4, 2.2), (59.2, 5.1, 11.6, 2.8, 15.1, 2.3),
    (44.0, 2.0, 14.0, 1.0, 19.0, 1.0), (58.0, 1.0, 15.0, 1.0, 21.0, 1.0),
    (58.9, 3.1, 11.4, 2.1, 14.2, 1.5), (60.7, 4.0, 10.5, 2.3, 17.2, 2.1),
    (58.0, 4.4, 11.7, 2.3, 15.8, 2.1), (59.6, 5.5, 11.8, 3.1, 14.6, 2.6),
    (48.0, 1.0, 15.0, 1.0, 20.0, 1.0), (63.0, 1.0, 16.0, 1.0, 21.0, 1.0)]

_D65_WHITE = np.array([0.95047, 1.0, 1.08883])

def _lab_d65_to_work(L, a, b):
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0
    f = lambda t: t**3 if t**3 > 0.008856 else (t - 16.0 / 116.0) / 7.787
    xyz_d65 = _D65_WHITE * np.array([f(fx), f(fy), f(fz)])
    xyz_d50 = np.linalg.inv(XYZ_D50_to_D65_CAT16) @ xyz_d65
    return np.linalg.inv(REC2020_TO_XYZ_D50) @ xyz_d50

def priority_samples():
    """skin corners + diffuse hue circle, each swept over tonal placements"""
    y_row = REC2020_TO_XYZ_D50[1]
    def place(rgb, evs):
        lum = y_row @ rgb
        return [rgb * (GREY * 2.0**e / lum) for e in evs]
    out = []
    for (L, Ls, a, As, b, Bs) in _SKIN_LAB:
        for dL in (-1.5, 0.0, 1.5):
            for da in (-1.5, 1.5):
                for db in (-1.5, 1.5):
                    rgb = _lab_d65_to_work(L + dL * Ls, a + da * As, b + db * Bs)
                    if rgb.min() > 0:
                        out += place(rgb, (-1.5, -0.75, 0.0, 0.75, 1.5))
    for k in range(12):
        h = 2 * np.pi * k / 12
        base = np.array([np.cos(h), np.cos(h - 2 * np.pi / 3), np.cos(h + 2 * np.pi / 3)])
        base = (base - base.min()) / (base.max() - base.min())
        for p in (0.3, 0.5, 0.7):
            out += place(p * base + (1 - p) * 0.5, (-2, -1, 0, 1, 2, 2.5))
    return out

def chroma_trajectory(sample, M, M_inv):
    """chroma ratio and hue drift vs exposure, through the bracketed curve"""
    c_in, h_in = chroma_hue(rgb_work_to_Yrg(sample))
    ratios, drifts = [], []
    for ev in EVS:
        rgb = sample * GREY * 2.0 ** ev
        mapped = M_inv @ tone_map(M @ rgb)
        c_out, h_out = chroma_hue(rgb_work_to_Yrg(np.maximum(mapped, 1e-10)))
        ratios.append(c_out / max(c_in, 1e-6))
        drifts.append(np.remainder(h_out - h_in + np.pi, 2 * np.pi) - np.pi)
    return np.array(ratios), np.array(drifts)

IDENTITY = np.eye(3)
BASELINES = {tuple(s): chroma_trajectory(s, IDENTITY, IDENTITY)[0] for s in boundary_colors()}

def target_profile(baseline):
    """Design target for the bleaching : transparent (baseline) below the shoulder,
    then a smooth ramp to full achromatic exactly at the white endpoint. This pins
    the bleach *rate* through the shoulder — without it the fit has no optimum and
    rails the insets (endpoint-only objectives are monotone in the inset amount)."""
    t = np.clip((EVS - SHOULDER_EV) / (WHITE_EV - SHOULDER_EV), 0.0, 1.0)
    ramp = 1.0 - t * t * (3.0 - 2.0 * t)  # smoothstep, 1 -> 0
    return baseline * ramp

def residuals(params, drift_target_rad_per_ev):
    insets, rotations = params[:3], params[3:]
    M, M_inv = bracket_matrices(insets, rotations)

    res = []
    # hard constraints as strong penalties : positivity with margin, over the whole
    # user ray t in [1, 2] (the runtime slider scales insets and rotations together)
    M2, _ = bracket_matrices(np.clip(2.0 * insets, 0.0, 0.9), 2.0 * rotations)
    res.append(500.0 * np.maximum(0.0, 0.005 - M).sum())
    res.append(500.0 * np.maximum(0.0, 0.005 - M2).sum())
    res.append(10.0 * max(0.0, np.linalg.cond(M) - 8.0))            # conditioning
    for sample in boundary_colors():
        ratios, drifts = chroma_trajectory(sample, M, M_inv)
        target = target_profile(BASELINES[tuple(sample)])
        res.extend(3.0 * (ratios - target))                          # purity-vs-exposure profile
        res.extend(2.0 * np.maximum(0.0, np.diff(ratios)))           # monotone bleaching
        d = np.gradient(drifts, EVS)
        res.extend(0.5 * (d - drift_target_rad_per_ev))              # uniform drift
    return np.array(res, dtype=float)

# ---------------------------------------------------------------- shipped variants + vectorized report
# The three v8 "AgX" colorscience variants (see filmic_agx_prepare_bracket in
# src/iop/filmicrgb.c). SINGLE SOURCE OF TRUTH — these must equal the C constants.
# Regenerate each with the fit mode noted; --report measures them for the doc tables.
SHIPPED_VARIANTS = {  # inset is uniform ; outset is per-primary (over-expanding)
    "no-bleach":   dict(fit="--min-bleach",     inset=0.335562,
                        irot=[-0.0092314, +0.0979124, +0.0034991],
                        outset=[0.349067, 0.737216, 0.396854], orot=[+0.0047636, +0.1445774, +0.0011608]),
    "low-bleach":  dict(fit="--fit-low-bleach", inset=0.487623,
                        irot=[-0.0176159, +0.0650293, +0.0044292],
                        outset=[0.479379, 0.746078, 0.369679], orot=[-0.0050757, +0.1018535, +0.0119466]),
    "medium-bleach": dict(fit="--fit-medium-bleach", inset=0.595334,
                          irot=[-0.0273940, +0.0323704, +0.0236516],
                          outset=[0.576994, 0.768241, 0.402336], orot=[-0.0167965, +0.0642740, +0.0419658]),
    "high-bleach": dict(fit="--max-desat 0.05", inset=0.747987,
                        irot=[-0.0515563, -0.0375649, +0.0222773],
                        outset=[0.724651, 0.828507, 0.550322], orot=[-0.0438769, -0.0095878, +0.0521530]),
}

# tone curve baked into a monotone LUT once, so the whole measurement vectorizes
# (np.interp) instead of calling the scalar-branching curve per sample
_LUT_X = np.linspace(0.0, 1.0, 8193)
_LUT_Y = np.array([CURVE(v) for v in _LUT_X])

def chroma_hue_batch(RGB):
    """(N,3) working-linear-Rec2020 -> (chroma, hue) in Kirk Yrg, relative to white."""
    LMS = ((RGB @ REC2020_TO_XYZ_D50.T) @ XYZ_D50_to_D65_CAT16.T) @ XYZ_D65_to_LMS_2006.T
    a = LMS.sum(axis=1, keepdims=True)
    rg = (LMS / np.where(a == 0.0, 1.0, a)) @ LMS_to_filmlightRGB.T
    dr, dg = rg[:, 0] - WHITE_YRG[1], rg[:, 1] - WHITE_YRG[2]
    return np.hypot(dr, dg), np.arctan2(dg, dr)

def variant_bracket(v):
    """(inset matrix M, applied outset matrix) for a SHIPPED_VARIANTS entry."""
    M, _ = bracket_matrices(np.full(3, v["inset"]), v["irot"])
    Mo, _ = bracket_matrices(v["outset"], v["orot"])
    return M, np.linalg.inv(Mo)

def measure_batch(S, C_in, H_in, M, Mo):
    """(N,3) samples through inset -> per-channel curve -> outset ; returns
    (chroma_ratio post output<=input clamp, hue_drift_deg), both (N,)."""
    x = (np.log2(np.maximum(S @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
    y = np.interp(np.clip(x, 0.0, 1.0).ravel(), _LUT_X, _LUT_Y).reshape(x.shape)
    O = np.maximum(y @ Mo.T, 1e-10)
    c_f, h_f = chroma_hue_batch(O)
    cr = np.minimum(c_f / np.maximum(C_in, 1e-9), 1.0)
    dh = np.rad2deg(np.remainder(h_f - H_in + np.pi, 2 * np.pi) - np.pi)
    return cr, dh

def skin_and_reflective_sets():
    """Two disjoint sample arrays, each swept over tonal placements : the skin-tone
    database, and the diffuse 'reflective'/memory-color hue circle (NOT skin)."""
    y_row = REC2020_TO_XYZ_D50[1]
    def place(rgb, evs):
        lum = y_row @ rgb
        return [rgb * (GREY * 2.0 ** e / lum) for e in evs]
    skin = []
    for (L, Ls, a, As, b, Bs) in _SKIN_LAB:
        for dL in (-1.5, 0.0, 1.5):
            for da in (-1.5, 1.5):
                for db in (-1.5, 1.5):
                    rgb = _lab_d65_to_work(L + dL * Ls, a + da * As, b + db * Bs)
                    if rgb.min() > 0:
                        skin += place(rgb, (-1.5, -0.75, 0.0, 0.75, 1.5))
    refl = []
    for k in range(12):
        h = 2 * np.pi * k / 12
        base = np.array([np.cos(h), np.cos(h - 2 * np.pi / 3), np.cos(h + 2 * np.pi / 3)])
        base = (base - base.min()) / (base.max() - base.min())
        for p in (0.3, 0.5, 0.7):
            refl += place(p * base + (1 - p) * 0.5, (-2, -1, 0, 1, 2, 2.5))
    return np.array(skin), np.array(refl)

def rec2020_worst_boundary_luminance(M, Mo):
    """Worst output luminance over the Rec2020 primaries and secondaries across the
    tonal range. Rec2020 IS the working space, so its primaries are the most saturated
    colors that can occur — the worst case. A strongly over-expanding outset can push
    them to NEGATIVE luminance (the pixel renders BLACK); this must stay > 0. This is
    the gamut-safety check that caught the no-bleach blue-goes-black bug."""
    y_row = REC2020_TO_XYZ_D50[1]
    bnd = [np.maximum(np.array(c, float), 1e-6) for c in
           ([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0])]
    worst = 1e9
    for s in bnd:
        lum0 = y_row @ s
        for ev in np.arange(-12.0, 8.01, 0.5):
            mr = M @ (s * GREY * 2.0 ** ev / lum0)
            x = (np.log2(np.maximum(mr, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
            cv = np.interp(np.clip(x, 0.0, 1.0), _LUT_X, _LUT_Y)
            worst = min(worst, float(y_row @ (Mo @ cv)))
    return worst

def fit_midpoint(lo_key, hi_key, inset_lo, inset_hi, seed_insets):
    """Solve for the bracket that best reproduces the AVERAGE of two variants' processed
    outputs — the perceptual midpoint. Target = 0.5*(post_bracket(lo) + post_bracket(hi))
    over a skin + reflective + Rec2020-boundary sample set (skin weighted x2), fit in
    least squares under Rec2020 gamut safety, positivity and conditioning. Returns the
    10-parameter bracket (uniform inset, inset rot[3], outset[3], outset rot[3]) or None."""
    from scipy.optimize import minimize
    y_row = REC2020_TO_XYZ_D50[1]

    def post_bracket(S, M, Mo):
        x = (np.log2(np.maximum(S @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
        y = np.interp(np.clip(x, 0.0, 1.0).ravel(), _LUT_X, _LUT_Y).reshape(x.shape)
        return y @ Mo.T

    def brkp(p):
        M, _ = bracket_matrices(np.full(3, p[0]), p[1:4])
        Mo, _ = bracket_matrices(p[4:7], p[7:10])
        return M, np.linalg.inv(Mo)

    S_sk, S_rf = skin_and_reflective_sets()

    def place(rgb, evs):
        lum = y_row @ rgb
        return [rgb * (GREY * 2.0 ** e / lum) for e in evs]
    bnd = []
    for c in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]):
        bnd += place(np.maximum(np.array(c, float), 1e-6), (-4, -2, 0, 2))
    S = np.vstack([S_sk, S_rf, np.array(bnd)])
    wt = np.ones(len(S)); wt[:len(S_sk)] = 2.0           # weight skin (portraits)

    def input_chroma(RGB):                               # module chroma_hue_batch is shadowed inside main()
        LMS = ((RGB @ REC2020_TO_XYZ_D50.T) @ XYZ_D50_to_D65_CAT16.T) @ XYZ_D65_to_LMS_2006.T
        a = LMS.sum(axis=1, keepdims=True)
        rg = (LMS / np.where(a == 0.0, 1.0, a)) @ LMS_to_filmlightRGB.T
        return np.hypot(rg[:, 0] - WHITE_YRG[1], rg[:, 1] - WHITE_YRG[2])
    keep = input_chroma(S) > 0.03
    S, wt = S[keep], wt[keep]

    Mlo, Molo = variant_bracket(SHIPPED_VARIANTS[lo_key])
    Mhi, Mohi = variant_bracket(SHIPPED_VARIANTS[hi_key])
    TARGET = 0.5 * (post_bracket(S, Mlo, Molo) + post_bracket(S, Mhi, Mohi))

    BIG = 1e6

    def objective(p):
        if not (inset_lo <= p[0] <= inset_hi):
            return BIG
        if p[4:7].min() < 0.02 or p[4:7].max() > 0.98:
            return BIG
        if np.abs(np.concatenate([p[1:4], p[7:10]])).max() > np.deg2rad(25):
            return BIG
        M, Mo = brkp(p)
        if M.min() < 0.004 or max(np.linalg.cond(M), np.linalg.cond(Mo)) > 6.5:
            return BIG
        if rec2020_worst_boundary_luminance(M, Mo) <= 0.0:   # gamut safety : no black
            return BIG
        O = post_bracket(S, M, Mo)
        return float(np.mean(wt[:, None] * (O - TARGET) ** 2))

    best = None
    for i0 in seed_insets:
        x0 = [i0, -0.03, 0.16, 0.01, min(0.9, i0), min(0.9, i0 * 1.25), min(0.9, i0 * 0.9), -0.02, 0.17, 0.0]
        r = minimize(objective, x0, method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 9000, "maxfev": 9000})
        if r.fun < BIG and (best is None or r.fun < best.fun):
            best = r
    return best.x if best is not None else None

def print_midpoint_constants(p, mode, endpoints):
    """Print the C constant block for a fitted midpoint bracket (or a failure note)."""
    if p is None:
        print("// no feasible %s under the constraints." % mode)
        return
    M, _ = bracket_matrices(np.full(3, p[0]), p[1:4])
    Mo, _ = bracket_matrices(p[4:7], p[7:10])
    Mo = np.linalg.inv(Mo)
    print("// fitted by tools/derive_filmic_agx_primaries.py %s" % mode)
    print("// perceptual midpoint of %s (average of processed outputs)" % endpoints)
    print("// Rec2020 gamut safety : worst boundary luminance %+.4f ; cond %.1f"
          % (rec2020_worst_boundary_luminance(M, Mo), max(np.linalg.cond(M), np.linalg.cond(Mo))))
    print("static const float inset_anchor[3]    = { %.6ff, %.6ff, %.6ff };" % (p[0], p[0], p[0]))
    print("static const float rotation_anchor[3] = { %+.7ff, %+.7ff, %+.7ff }; // %+.2f°, %+.2f°, %+.2f°"
          % (p[1], p[2], p[3], *np.rad2deg(p[1:4])))
    print("static const float outset_anchor[3]   = { %.6ff, %.6ff, %.6ff };" % (p[4], p[5], p[6]))
    print("static const float outset_rotation[3] = { %+.7ff, %+.7ff, %+.7ff }; // %+.2f°, %+.2f°, %+.2f°"
          % (p[7], p[8], p[9], *np.rad2deg(p[7:10])))

def report_variants():
    """Print the {avg,max} x {desaturation,hue drift} x {skin,reflective} table for
    the four shipped variants, plus a Rec2020 gamut-safety check. Desaturation is over
    all colors that carry chroma ; hue drift is measured where chroma survives (ratio >
    0.2 — a bleached color has no meaningful hue). This is the source of the tables in
    doc/filmic-agx.md and the user docs."""
    S_sk, S_rf = skin_and_reflective_sets()
    Csk, Hsk = chroma_hue_batch(S_sk)
    Crf, Hrf = chroma_hue_batch(S_rf)
    mk = Csk > 0.04                                          # drop near-neutral inputs (ratio is noise)
    S_sk, Csk, Hsk = S_sk[mk], Csk[mk], Hsk[mk]
    mr = Crf > 0.05
    S_rf, Crf, Hrf = S_rf[mr], Crf[mr], Hrf[mr]

    def stat(S, C, H, M, Mo):
        cr, dh = measure_batch(S, C, H, M, Mo)
        d = (1.0 - cr) * 100.0
        g = np.abs(dh[cr > 0.2])
        return d.mean(), d.max(), g.mean(), g.max()

    print("# {avg ; max} desaturation and hue drift, skin tones vs reflective colors")
    print("# skin database (De Rigal/Xiao) and diffuse memory-color hue circle, over tonal placements")
    print("%-11s | %-31s | %-31s" % ("", "SKIN TONES", "REFLECTIVE COLORS"))
    print("%-11s | %-13s %-16s | %-13s %-16s"
          % ("variant", "desat a/max", "hue drift a/max", "desat a/max", "hue drift a/max"))
    for name, v in SHIPPED_VARIANTS.items():
        M, Mo = variant_bracket(v)
        sda, sdm, sha, shm = stat(S_sk, Csk, Hsk, M, Mo)
        rda, rdm, rha, rhm = stat(S_rf, Crf, Hrf, M, Mo)
        print("%-11s | %4.1f%% %5.1f%%   %4.1f° %5.1f°  | %4.1f%% %5.1f%%   %4.1f° %5.1f°"
              % (name, sda, sdm, sha, shm, rda, rdm, rha, rhm))

    print("\n# Rec2020 gamut safety : worst output luminance over the Rec2020 primaries")
    print("# and secondaries, EV -12..+8. Must be > 0 — a negative value renders BLACK.")
    for name, v in SHIPPED_VARIANTS.items():
        M, Mo = variant_bracket(v)
        wl = rec2020_worst_boundary_luminance(M, Mo)
        print("%-11s worst boundary luminance %+.4f   %s" % (name, wl, "OK" if wl > 0 else "*** BLACK ***"))

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift-target", type=float, default=0.0,
                    help="uniform hue drift target in degrees per EV of compression "
                         "(0 = neutral/hue-stable character, e.g. -0.5 for a warm bias)")
    ap.add_argument("--inset", type=float, default=0.25,
                    help="uniform inset anchor. NOT fitted: bleach depth saturates with the "
                         "inset (the exact-inverse outset re-expands what the curve did not "
                         "equalize, and the Yrg gamut mapper owns the very endpoint), so "
                         "endpoint objectives rail it against any bound. 0.25 sits at the "
                         "knee of diminishing returns with ~0.96 midtone transparency at "
                         "+1 EV and leaves conditioning headroom for the user ray t <= 2.")
    ap.add_argument("--fit-insets", action="store_true",
                    help="fit all 6 parameters anyway (diagnostic; expect railed insets)")
    ap.add_argument("--fit-outset", action="store_true",
                    help="fit the outset recovery factor kappa. The outset is the inverse of "
                         "the bracket built with kappa-scaled insets (kappa = 1 : exact "
                         "inverse). An exact inverse mandatorily bleaches every color the "
                         "curve touches — with the low-latitude sigmoid default that is the "
                         "whole tonal range, washing out valid midtone colors (skin tones). "
                         "kappa > 1 over-expands so that priority colors (skin database + "
                         "diffuse reflectances, over their tonal placements) reach the "
                         "output-chroma-equals-input clamp : the clamp then trims recovery "
                         "to exactly 1.0 per pixel, tone-adaptively, which is what makes one "
                         "fixed kappa portable across curves and dynamic ranges (verified "
                         "6.5-16 EV). Then re-fits the inset-blue and outset rotations for the "
                         "RECOVERED-chroma regime (the kappa recovery re-exposes drift the "
                         "bleaching used to hide), under the maintainer's priority ordering. "
                         "This is the CURRENT production fit ; it supersedes --minimax, which "
                         "was correct only for the earlier bleached (exact-inverse) regime.")
    ap.add_argument("--minimax", action="store_true",
                    help="fit rotations for the WORST-CASE hue drift (Chebyshev) over "
                         "EV <= +3.5 instead of zero-mean L2. This is the right objective "
                         "when the hue is NOT anchored in Ych afterwards (see "
                         "FILMIC_AGX_UNANCHORED_HUE_TEST in filmicrgb.c) : without a "
                         "backstop, only the worst case matters. Floor is ~28.5° — the "
                         "drift field reverses direction between toe and shoulder, which "
                         "no constant matrix pair can serve on both sides. SUPERSEDED by "
                         "--fit-outset : minimax was fit for the bleached exact-inverse regime "
                         "and its blue rotation now CAUSES purple on recovered blues.")
    ap.add_argument("--fit-priority", action="store_true",
                    help="CURRENT production fit. Joint Nelder-Mead fine-tune of the whole "
                         "bracket (per-primary inset chroma + rotation, per-primary outset "
                         "chroma + rotation = 12 params) started from the uniform-0.25 / "
                         "kappa=2 config, minimizing hue drift over the PRIORITY set (skin "
                         "database + diffuse reflectances) at preserved chroma. Supersedes "
                         "--fit-outset : the outset is now a per-primary expansion, not a "
                         "scalar kappa. Hard constraints (barriers the optimizer cannot "
                         "cross) : skin red-ward drift <= -1.5 deg, skin chroma >= 92%%, "
                         "diffuse recovery p5 >= 0.97 (bleaching cannot game accuracy), inset "
                         "positivity >= 0.004, conditioning <= 6.5. Inset capped at 0.35 to "
                         "keep rendering character (the free optimum runs 0.55 for a sub-JND "
                         "gain at a visible highlight-desaturation cost). sRGB blue at high EV "
                         "is not targeted — a structural DoF limit, left to the Ych recovery.")
    ap.add_argument("--max-desat", type=float, default=None, metavar="FRAC",
                    help="Desaturation-budgeted fit — the recommended way to set the bracket. "
                         "FRAC is the MAXIMUM average chroma loss (1 - output/input chroma, post "
                         "output<=input clamp) over the priority set (skin database + diffuse "
                         "reflectances) you are willing to trade for hue accuracy — e.g. 0.05 = "
                         "at most 5%% average desaturation. The solver minimizes hue drift and "
                         "spends desaturation on hue up to FRAC : a hard barrier caps the "
                         "average, and a tiny chroma-preference term keeps chroma as high as the "
                         "hue optimum allows, so a slack budget is not wasted (the 'best chroma "
                         "match' half). The inset is a single UNIFORM scalar (10 params : inset "
                         "chroma + 3 inset rotations + 3 outset chroma + 3 outset rotations) — a "
                         "per-primary inset is the lever the optimizer abuses to game the metric "
                         "(green rails to 0.7, wrecking an unseen blue), and the good fits use a "
                         "uniform inset with the per-primary action in the outset anyway. The "
                         "whole priority set is evaluated every step (vectorized, no subsample), "
                         "so no color can hide ; the worst single hue is hard-capped at 16 deg ; "
                         "skin red-ward drift stays vetoed (<= -1.5 deg) regardless of budget. "
                         "The desaturation is mostly the bright-color / highlight bleach (the AgX "
                         "wash-out look), so individual highlights bleaching hard is expected and "
                         "not penalized. Sweep FRAC to trace the hue/chroma frontier : ~0.02 "
                         "gives inset 0.63 / skin 1.3 deg / refl max 10 deg ; ~0.05 gives inset "
                         "0.75 / skin 0.8 deg / refl max 6 deg (saturated colors visibly bleach) ; "
                         "beyond ~0.05 hue stops improving. Supersedes --fit-priority (which is "
                         "roughly FRAC 0.02 on a uniform 0.35 inset).")
    ap.add_argument("--min-bleach", action="store_true",
                    help="Fit the NO-BLEACH variant : minimize the bracket's own desaturation, "
                         "letting hue drift (which the downstream Ych hue-recovery restores — "
                         "there is no downstream saturation recovery, so chroma is what must be "
                         "protected here). COUNTERINTUITIVE result this encodes : a hard 0%% "
                         "inset is the WORST case for saturation (~7.7%% avg desat), because "
                         "with no inset the outset cannot over-expand without wrecking "
                         "conditioning, so bright colors bleach from the raw per-channel curve "
                         "with nothing to recover them. The over-expanding outset is what un-"
                         "bleaches (pulls chroma back up to the output<=input clamp), and it "
                         "only becomes well-conditioned at inset >= ~0.2. So minimum bleach sits "
                         "at a MODERATE inset ~0.25 (avg desat < 0.5%%, conditioning ~3, very "
                         "stable) — not near zero. Objective : minimize avg + worst-color "
                         "desaturation over the priority set, inset in [0.20, 0.30], worst hue "
                         "<= 24 deg (recoverable), conditioning <= 4.5 (stability), skin red-ward "
                         "drift <= -2.5 deg.")
    ap.add_argument("--fit-low-bleach", action="store_true",
                    help="Fit the LOW-BLEACH variant as the PERCEPTUAL MIDPOINT of no-bleach and "
                         "high-bleach. Rather than a desaturation budget (which put low-bleach's "
                         "hue too close to high-bleach — the visible gap no->low was larger than "
                         "low->high), the target is the straight average of the no-bleach and "
                         "high-bleach PROCESSED OUTPUTS (post-bracket display RGB) over a skin + "
                         "reflective + Rec2020-boundary sample set (skin weighted x2 for "
                         "portraits) ; the low-bleach bracket is solved to reproduce that average "
                         "in least squares. Result bisects the hue drift evenly on both sides and "
                         "keeps skin chroma (avoids high-bleach's skin whitening). Constrained to "
                         "Rec2020 gamut safety, skin red-ward veto, positivity, conditioning "
                         "<= 6.5. Reads no/high-bleach from SHIPPED_VARIANTS ; refit if either "
                         "endpoint changes.")
    ap.add_argument("--fit-medium-bleach", action="store_true",
                    help="Fit the MEDIUM-BLEACH variant as the PERCEPTUAL MIDPOINT of low-bleach and "
                         "high-bleach. See --fit-low-bleach")
    ap.add_argument("--report", action="store_true",
                    help="Measure the three SHIPPED variants (no/low/high bleach) and print the "
                         "{avg ; max} desaturation and hue-shift table for skin tones vs "
                         "reflective colors — the source of the tables in doc/filmic-agx.md and "
                         "the user docs. Reads the constants from SHIPPED_VARIANTS (which must "
                         "mirror filmic_agx_prepare_bracket in the C), so it doubles as a drift "
                         "check that the shipped brackets still measure as documented.")
    args = ap.parse_args()

    if args.fit_low_bleach:
        p = fit_midpoint("no-bleach", "high-bleach", 0.20, 0.75, (0.35, 0.45, 0.55))
        if p is not None:
            print_midpoint_constants(p, "--fit-low-bleach", "no-bleach and high-bleach")
        else:
            print("// no feasible low-bleach midpoint under the constraints.")
        return
    
    if args.fit_medium_bleach:
        p = fit_midpoint("low-bleach", "high-bleach", 0.20, 0.75, (0.35, 0.45, 0.55))
        if p is not None:
            print_midpoint_constants(p, "--fit-medium-bleach", "low-bleach and high-bleach")
        else:
            print("// no feasible medium-bleach midpoint under the constraints.")
        return

    if args.report:
        report_variants()
        return

    if args.max_desat is not None:
        from scipy.optimize import minimize
        budget = float(args.max_desat)
        y_row = REC2020_TO_XYZ_D50[1]

        def place(rgb, evs):
            lum = y_row @ rgb
            return [rgb * (GREY * 2.0 ** e / lum) for e in evs]

        refl = priority_samples()
        skin_base = [_lab_d65_to_work(L, a + da * As, b + db * Bs)
                     for (L, Ls, a, As, b, Bs) in _SKIN_LAB
                     for da in (-1, 1) for db in (-1, 1)]
        skin_base = [s for s in skin_base if s.min() > 0]
        skin = [s for w in skin_base for s in place(w, (-1.5, -0.5, 0.5, 1.5))]

        # Vectorized pipeline : NO subsampling. Every color in the priority set is
        # evaluated on every objective call, so the optimizer can never game the
        # average by wrecking a hue it cannot see. The tone curve branches on scalars,
        # so bake it into a monotone lookup table and apply it with np.interp.
        LUT_X = np.linspace(0.0, 1.0, 8193)
        LUT_Y = np.array([CURVE(v) for v in LUT_X])
        S_refl = np.array(refl)                              # (Nr, 3) working linear Rec2020
        S_skin = np.array(skin)                              # (Ns, 3)

        def yrg_rg_batch(RGB):                               # RGB (N,3) -> (r, g) chromaticity
            LMS = ((RGB @ REC2020_TO_XYZ_D50.T) @ XYZ_D50_to_D65_CAT16.T) @ XYZ_D65_to_LMS_2006.T
            a = LMS.sum(axis=1, keepdims=True)
            rg = (LMS / np.where(a == 0.0, 1.0, a)) @ LMS_to_filmlightRGB.T
            return rg[:, 0], rg[:, 1]

        def chroma_hue_batch(RGB):
            r, g = yrg_rg_batch(RGB)
            dr, dg = r - WHITE_YRG[1], g - WHITE_YRG[2]
            return np.hypot(dr, dg), np.arctan2(dg, dr)

        C_refl, H_refl = chroma_hue_batch(S_refl)            # input chroma/hue, invariant
        C_skin, H_skin = chroma_hue_batch(S_skin)
        # Drop near-neutral inputs : a gray has no saturation to preserve and no
        # meaningful hue, so its chroma RATIO (c_out/c_in) is numerical noise that
        # otherwise inflates the desaturation average and the worst-color cap — which
        # is exactly what the green-heavy inset was exploiting.
        rkeep = C_refl > 0.05
        S_refl, C_refl, H_refl = S_refl[rkeep], C_refl[rkeep], H_refl[rkeep]
        skeep = C_skin > 0.04                                # keep pale skin, drop the near-grays
        S_skin, C_skin, H_skin = S_skin[skeep], C_skin[skeep], H_skin[skeep]

        # Parameterization : the inset is a single UNIFORM scalar (p[0]), and the
        # per-primary action lives entirely in the outset — which is exactly the
        # structure the good fits converge to anyway (the shipped inset is 0.35 on all
        # three). A per-primary inset is the lever the optimizer abused to game the
        # gated metrics (green railed to 0.7, wrecking an unseen blue) ; removing that
        # DoF structurally forbids the pathology. 10 params :
        #   p[0]      uniform inset chroma
        #   p[1:4]    inset rotations (R, G, B)
        #   p[4:7]    outset chroma   (R, G, B)
        #   p[7:10]   outset rotations (R, G, B)
        def brk(p):
            M, _ = bracket_matrices(np.full(3, p[0]), p[1:4])
            Mo, _ = bracket_matrices(p[4:7], p[7:10])
            return M, np.linalg.inv(Mo)

        def measure(S, C_in, H_in, M, Mo):                   # -> chroma_ratio, drift_deg (N,)
            x = (np.log2(np.maximum(S @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
            y = np.interp(np.clip(x, 0.0, 1.0).ravel(), LUT_X, LUT_Y).reshape(x.shape)
            O = np.maximum(y @ Mo.T, 1e-10)
            c_f, h_f = chroma_hue_batch(O)
            cr = np.minimum(c_f / np.maximum(C_in, 1e-9), 1.0)
            dh = np.rad2deg(np.remainder(h_f - H_in + np.pi, 2 * np.pi) - np.pi)
            return cr, dh

        BIG = 1e5

        def evaluate(p):
            """(skin_cr, skin_dh, refl_cr, refl_dh, cond) or None if degenerate."""
            if not (0.05 <= p[0] <= 0.85):                   # uniform inset chroma
                return None
            for i in (4, 5, 6):                              # outset chroma
                if not (0.02 <= p[i] <= 0.95):
                    return None
            for i in (1, 2, 3, 7, 8, 9):                     # rotations |.| <= 25 deg
                if abs(p[i]) > np.deg2rad(25):
                    return None
            M, Mo = brk(p)
            if M.min() < 0.004:
                return None
            cond = max(np.linalg.cond(M), np.linalg.cond(Mo))
            if cond > 6.5:                                   # reject near-degenerate brackets
                return None
            skin_cr, skin_dh = measure(S_skin, C_skin, H_skin, M, Mo)
            refl_cr, refl_dh = measure(S_refl, C_refl, H_refl, M, Mo)
            return skin_cr, skin_dh, refl_cr, refl_dh, cond

        def objective(p):
            ev = evaluate(p)
            if ev is None:
                return BIG
            skin_cr, skin_dh, refl_cr, refl_dh, cond = ev
            if skin_dh.min() < -1.5:                         # skin red-ward veto (absolute)
                return BIG + abs(skin_dh.min())
            # desaturation = the chroma the bracket costs (mostly the bright-color /
            # highlight bleach — the AgX wash-out look — which IS the cost the budget
            # trades ; individual highlights bleaching hard is fine, so no per-color cap).
            desats = np.concatenate([1.0 - skin_cr, 1.0 - refl_cr])
            mean_desat = float(desats.mean())
            gated = np.abs(refl_dh[refl_cr > 0.2])           # hue where chroma survives
            worst_hue = float(gated.max())
            # hue fidelity : mean AND worst single color ('best hue match' = no color
            # badly off, not a good average hiding an outlier).
            hue_err = np.mean(gated) + 0.5 * worst_hue + 1.5 * np.mean(np.abs(skin_dh))
            return (hue_err
                    + 0.10 * mean_desat                      # prefer chroma when hue is tied (best-chroma-match)
                    + 1e4 * max(0.0, mean_desat - budget)    # HARD average-desaturation budget
                    + 20.0 * max(0.0, worst_hue - 16.0))     # HARD worst-case hue cap (deg)

        # start from the shipped config (uniform inset 0.35), plus stronger-inset
        # restarts — a larger budget affords a stronger bracket for better hue
        p0 = [0.35, -0.0437436, 0.1580839, 0.0177711,
              0.457525, 0.621106, 0.349832, -0.0346110, 0.2086015, 0.0753169]
        best = None
        for di in (0.0, 0.12, 0.24):
            start = list(p0)
            start[0] += di
            r = minimize(objective, start, method="Nelder-Mead",
                         options={"xatol": 1e-5, "fatol": 1e-4, "maxiter": 8000, "maxfev": 8000})
            if r.fun < BIG and (best is None or r.fun < best.fun):
                best = r
        if best is None:
            print("// no feasible bracket under the constraints (skin red veto / positivity / conditioning).")
            print("// the desaturation budget is not the binding constraint — check --max-desat is >= 0.")
            return
        p = best.x
        skin_cr, skin_dh, refl_cr, refl_dh, cond = evaluate(p)
        desats = np.concatenate([1.0 - skin_cr, 1.0 - refl_cr])
        mean_desat = float(desats.mean())
        gated = np.abs(refl_dh[refl_cr > 0.2])
        M, Mo = brk(p)
        inset = p[0]
        print("// fitted by tools/derive_filmic_agx_primaries.py --max-desat %.3f (obj %.3f)" % (budget, best.fun))
        print("// achieved avg desaturation %.1f%% (budget %.1f%%) ; worst single color %.1f%% (bright-color bleach)"
              % (100 * mean_desat, 100 * budget, 100 * desats.max()))
        print("// priority set : skin |mean| %.1f deg [%+.1f..%+.1f] ; reflective mean %.1f max %.1f ; cond %.1f"
              % (np.mean(np.abs(skin_dh)), skin_dh.min(), skin_dh.max(),
                 gated.mean(), gated.max(), cond))
        print("static const float inset_anchor[3]    = { %.6ff, %.6ff, %.6ff };" % (inset, inset, inset))
        print("static const float rotation_anchor[3] = { %+.7ff, %+.7ff, %+.7ff }; // %+.2f°, %+.2f°, %+.2f°"
              % (p[1], p[2], p[3], *np.rad2deg(p[1:4])))
        print("static const float outset_anchor[3]   = { %.6ff, %.6ff, %.6ff };" % (p[4], p[5], p[6]))
        print("static const float outset_rotation[3] = { %+.7ff, %+.7ff, %+.7ff }; // %+.2f°, %+.2f°, %+.2f°"
              % (p[7], p[8], p[9], *np.rad2deg(p[7:10])))
        print("// kappa-equivalent per-primary outset/inset ratios : %.3f %.3f %.3f"
              % (p[4] / inset, p[5] / inset, p[6] / inset))
        return

    if args.min_bleach:
        from scipy.optimize import minimize
        y_row = REC2020_TO_XYZ_D50[1]

        def place(rgb, evs):
            lum = y_row @ rgb
            return [rgb * (GREY * 2.0 ** e / lum) for e in evs]

        refl = priority_samples()
        skin_base = [_lab_d65_to_work(L, a + da * As, b + db * Bs)
                     for (L, Ls, a, As, b, Bs) in _SKIN_LAB
                     for da in (-1, 1) for db in (-1, 1)]
        skin_base = [s for s in skin_base if s.min() > 0]
        skin = [s for w in skin_base for s in place(w, (-1.5, -0.5, 0.5, 1.5))]

        LUT_X = np.linspace(0.0, 1.0, 8193)
        LUT_Y = np.array([CURVE(v) for v in LUT_X])
        S_refl = np.array(refl)
        S_skin = np.array(skin)

        def chroma_hue_batch(RGB):
            LMS = ((RGB @ REC2020_TO_XYZ_D50.T) @ XYZ_D50_to_D65_CAT16.T) @ XYZ_D65_to_LMS_2006.T
            a = LMS.sum(axis=1, keepdims=True)
            rg = (LMS / np.where(a == 0.0, 1.0, a)) @ LMS_to_filmlightRGB.T
            dr, dg = rg[:, 0] - WHITE_YRG[1], rg[:, 1] - WHITE_YRG[2]
            return np.hypot(dr, dg), np.arctan2(dg, dr)

        C_refl, H_refl = chroma_hue_batch(S_refl)
        C_skin, H_skin = chroma_hue_batch(S_skin)
        rkeep = C_refl > 0.05
        S_refl, C_refl, H_refl = S_refl[rkeep], C_refl[rkeep], H_refl[rkeep]
        skeep = C_skin > 0.04
        S_skin, C_skin, H_skin = S_skin[skeep], C_skin[skeep], H_skin[skeep]
        S_all = np.vstack([S_skin, S_refl])
        C_all = np.concatenate([C_skin, C_refl])
        H_all = np.concatenate([H_skin, H_refl])

        def brk(p):                                          # p[0] inset, p[1:4] irot, p[4:7] outset, p[7:10] orot
            M, _ = bracket_matrices(np.full(3, p[0]), p[1:4])
            Mo, _ = bracket_matrices(p[4:7], p[7:10])
            return M, np.linalg.inv(Mo)

        def measure(S, C_in, H_in, M, Mo):
            x = (np.log2(np.maximum(S @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
            y = np.interp(np.clip(x, 0.0, 1.0).ravel(), LUT_X, LUT_Y).reshape(x.shape)
            O = np.maximum(y @ Mo.T, 1e-10)
            c_f, h_f = chroma_hue_batch(O)
            cr = np.minimum(c_f / np.maximum(C_in, 1e-9), 1.0)
            dh = np.rad2deg(np.remainder(h_f - H_in + np.pi, 2 * np.pi) - np.pi)
            return cr, dh

        # Rec2020 GAMUT SAFETY. The working space IS linear Rec2020, so its primaries
        # and secondaries are the most saturated colors representable — the true worst
        # case (more so than sRGB's). A strong outset over-expansion, which minimum-
        # desaturation wants, pushes these to NEGATIVE luminance -> black pixels : the
        # no-bleach blue-ramp-goes-black failure. Require the outset to retain at least
        # 25% of the pre-outset luminance for every primary/secondary across the tonal
        # range. (Deep shadows keep near-zero luminance either way — the ratio scales.)
        _BND = [np.maximum(np.array(c, float), 1e-6) for c in
                ([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0])]
        _BND_EV = (-8, -6, -4, -2, -1, 0, 1, 2, 3)

        def gamut_violation(M, Mo):
            worst = 0.0
            for s in _BND:
                lum0 = y_row @ s
                for ev in _BND_EV:
                    mr = M @ (s * GREY * 2.0 ** ev / lum0)
                    x = (np.log2(np.maximum(mr, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
                    cv = np.interp(np.clip(x, 0.0, 1.0), LUT_X, LUT_Y)
                    worst = max(worst, 0.25 * (y_row @ cv) - (y_row @ (Mo @ cv)))
            return worst

        BIG = 1e5

        def objective(p):
            if not (0.15 <= p[0] <= 0.45):
                return BIG
            if p[4:7].min() < 0.02 or p[4:7].max() > 0.98:
                return BIG
            if np.abs(np.concatenate([p[1:4], p[7:10]])).max() > np.deg2rad(25):
                return BIG
            M, Mo = brk(p)
            if M.min() < 0.004:
                return BIG
            cond = max(np.linalg.cond(M), np.linalg.cond(Mo))
            if cond > 5.0:                                   # conditioning : stability
                return BIG
            gv = gamut_violation(M, Mo)
            if gv > 0.0:                                     # Rec2020 gamut safety (hard) — no black
                return BIG + gv * 100.0
            skin_cr, skin_dh = measure(S_skin, C_skin, H_skin, M, Mo)
            if skin_dh.min() < -2.5:                         # skin red-ward veto (looser : hue recovered)
                return BIG + abs(skin_dh.min())
            cr, _ = measure(S_all, C_all, H_all, M, Mo)
            refl_cr, refl_dh = measure(S_refl, C_refl, H_refl, M, Mo)
            worst_hue = np.abs(refl_dh[refl_cr > 0.2]).max()
            d = 1.0 - cr
            # minimize the bracket's desaturation (avg + a touch of worst-color) ; hue is free
            # up to 24 deg (recovered downstream) ; a small skin-hue term keeps skin sane.
            return (d.mean() * 100.0 + 0.15 * d.max() * 100.0
                    + 0.3 * max(0.0, worst_hue - 24.0)
                    + 0.5 * np.mean(np.abs(skin_dh)))

        best = None
        for seed in ([0.18, 0.0, 0.06, 0.0, 0.44, 0.58, 0.28, 0.0, 0.10, 0.02],
                     [0.25, 0.0, 0.05, 0.0, 0.50, 0.58, 0.35, 0.0, 0.08, 0.02],
                     [0.30, 0.0, 0.03, 0.0, 0.55, 0.60, 0.40, 0.0, 0.06, 0.02]):
            r = minimize(objective, seed, method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 8000, "maxfev": 8000})
            if r.fun < BIG and (best is None or r.fun < best.fun):
                best = r
        if best is None:
            print("// no feasible no-bleach bracket under the constraints.")
            return
        p = best.x
        skin_cr, skin_dh = measure(S_skin, C_skin, H_skin, *brk(p))
        cr, _ = measure(S_all, C_all, H_all, *brk(p))
        refl_cr, refl_dh = measure(S_refl, C_refl, H_refl, *brk(p))
        g = np.abs(refl_dh[refl_cr > 0.2])
        M, Mo = brk(p)
        inset = p[0]
        print("// fitted by tools/derive_filmic_agx_primaries.py --min-bleach (no-bleach variant, obj %.3f)" % best.fun)
        print("// avg desaturation %.2f%% ; worst single color %.0f%% ; refl hue mean %.1f max %.1f (recovered downstream)"
              % (100 * (1 - cr).mean(), 100 * (1 - cr).max(), g.mean(), g.max()))
        print("// skin |mean| %.1f deg [%+.1f..%+.1f] ; cond %.1f"
              % (np.mean(np.abs(skin_dh)), skin_dh.min(), skin_dh.max(), max(np.linalg.cond(M), np.linalg.cond(Mo))))
        # Rec2020 gamut-safety proof : worst luminance retention over primaries/secondaries
        worst_ret = min((y_row @ (Mo @ np.interp(np.clip((np.log2(np.maximum(M @ (s * GREY * 2.0 ** ev / (y_row @ s)), 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV), 0.0, 1.0), LUT_X, LUT_Y)))
                         for s in _BND for ev in _BND_EV)
        print("// Rec2020 gamut safety : worst boundary output luminance %+.4f (must be > 0 : no black)" % worst_ret)
        print("static const float inset_anchor[3]    = { %.6ff, %.6ff, %.6ff };" % (inset, inset, inset))
        print("static const float rotation_anchor[3] = { %+.7ff, %+.7ff, %+.7ff }; // %+.2f°, %+.2f°, %+.2f°"
              % (p[1], p[2], p[3], *np.rad2deg(p[1:4])))
        print("static const float outset_anchor[3]   = { %.6ff, %.6ff, %.6ff };" % (p[4], p[5], p[6]))
        print("static const float outset_rotation[3] = { %+.7ff, %+.7ff, %+.7ff }; // %+.2f°, %+.2f°, %+.2f°"
              % (p[7], p[8], p[9], *np.rad2deg(p[7:10])))
        print("// kappa-equivalent per-primary outset/inset ratios : %.3f %.3f %.3f"
              % (p[4] / inset, p[5] / inset, p[6] / inset))
        return

    if args.fit_priority:
        from scipy.optimize import minimize
        y_row = REC2020_TO_XYZ_D50[1]

        def place(rgb, evs):
            lum = y_row @ rgb
            return [rgb * (GREY * 2.0 ** e / lum) for e in evs]

        # PRIORITY set : reflective/memory colors (already EV-spread) + skin database
        refl = priority_samples()
        skin_base = [_lab_d65_to_work(L, a + da * As, b + db * Bs)
                     for (L, Ls, a, As, b, Bs) in _SKIN_LAB
                     for da in (-1, 1) for db in (-1, 1)]
        skin_base = [s for s in skin_base if s.min() > 0]
        skin = [s for w in skin_base for s in place(w, (-1.5, -0.5, 0.5, 1.5))]

        def brk(p):
            M, _ = bracket_matrices(p[0:3], p[3:6])
            Mo, _ = bracket_matrices(p[6:9], p[9:12])
            return M, np.linalg.inv(Mo)

        def meas(rgb, M, Mo):
            c_o, h_o = chroma_hue(rgb_work_to_Yrg(rgb))
            x = (np.log2(np.maximum(M @ rgb, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
            o = Mo @ np.array([CURVE(v) for v in x])
            c_f, h_f = chroma_hue(rgb_work_to_Yrg(np.maximum(o, 1e-10)))
            return min(c_f / max(c_o, 1e-9), 1.0), np.rad2deg(np.remainder(h_f - h_o + np.pi, 2 * np.pi) - np.pi)

        BIG = 1e5

        def objective(p):
            if any(p[i] < 0.20 or p[i] > 0.35 for i in range(3)):  # inset cap : keep character
                return BIG
            M, Mo = brk(p)
            if M.min() < 0.004:
                return BIG - M.min() * 1e4
            if max(np.linalg.cond(M), np.linalg.cond(Mo)) > 6.5:
                return BIG
            sdh = []
            for s in skin:
                cr, dh = meas(s, M, Mo)
                if dh < -1.5 or cr < 0.92:            # skin red-ward veto + chroma floor
                    return BIG + abs(dh) + (1 - cr) * 100
                sdh.append(abs(dh))
            rd = [meas(s, M, Mo) for s in refl]
            if np.percentile([c for c, d in rd], 5) < 0.97:   # no bleaching
                return BIG
            rdh = [abs(d) for c, d in rd if c > 0.5]
            return np.mean(rdh) + 0.2 * np.max(rdh) + 1.5 * np.mean(sdh)

        p0 = [0.25, 0.25, 0.25, np.deg2rad(-2.50), np.deg2rad(9.29), np.deg2rad(3.11),
              0.50, 0.50, 0.50, np.deg2rad(-2.00), np.deg2rad(11.93), np.deg2rad(5.20)]
        res = minimize(objective, p0, method="Nelder-Mead",
                       options={"xatol": 2e-5, "fatol": 2e-4, "maxiter": 8000, "maxfev": 12000})
        p = res.x
        M, Mo = brk(p)
        sd = [meas(s, M, Mo) for s in skin]
        rd = [meas(s, M, Mo) for s in refl]
        sdh = [d for c, d in sd]
        rdh = [abs(d) for c, d in rd if c > 0.5]
        print("// fitted by tools/derive_filmic_agx_primaries.py --fit-priority (obj %.3f)" % res.fun)
        print("// priority set : skin |mean| %.1f deg [%+.1f..%+.1f] ; reflective mean %.1f max %.1f ; cond %.1f/%.1f"
              % (np.mean(np.abs(sdh)), min(sdh), max(sdh), np.mean(rdh), np.max(rdh),
                 np.linalg.cond(M), np.linalg.cond(Mo)))
        print("static const float inset_anchor[3]    = { %.6ff, %.6ff, %.6ff };" % (p[0], p[1], p[2]))
        print("static const float rotation_anchor[3] = { %+.7ff, %+.7ff, %+.7ff }; // %+.2f°, %+.2f°, %+.2f°"
              % (p[3], p[4], p[5], *np.rad2deg(p[3:6])))
        print("static const float outset_anchor[3]   = { %.6ff, %.6ff, %.6ff };" % (p[6], p[7], p[8]))
        print("static const float outset_rotation[3] = { %+.7ff, %+.7ff, %+.7ff }; // %+.2f°, %+.2f°, %+.2f°"
              % (p[9], p[10], p[11], *np.rad2deg(p[9:12])))
        print("// kappa-equivalent per-primary outset/inset ratios : %.3f %.3f %.3f"
              % (p[6] / p[0], p[7] / p[1], p[8] / p[2]))
        return

    insets0 = np.full(3, args.inset)
    rot0 = np.zeros(3)
    if args.fit_outset:
        # production rotations (from --minimax) ; keep in sync with the C anchors
        rot = np.array([-0.0436910, 0.1621254, 0.4199733])
        M, _ = bracket_matrices(insets0, rot)
        samples = priority_samples()

        def p5_ratio(kappa, curve=CURVE, black=BLACK_EV, white=WHITE_EV):
            Mk, _ = bracket_matrices(np.clip(kappa * insets0, 0.0, 0.9), rot)
            M_out = np.linalg.inv(Mk)
            ratios = []
            for s in samples:
                c_o, _ = chroma_hue(rgb_work_to_Yrg(s))
                x = (np.log2(np.maximum(M @ s, 1e-10) / GREY) - black) / (white - black)
                out = M_out @ np.array([curve(v) for v in x])
                c_f, _ = chroma_hue(rgb_work_to_Yrg(np.maximum(out, 1e-10)))
                ratios.append(c_f / max(c_o, 1e-9))
            return np.percentile(ratios, 5)

        lo, hi = 1.0, 2.5
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if p5_ratio(mid) < 1.0 else (lo, mid)
        kappa = 0.5 * (lo + hi)
        print(f"// OUTSET_RECOVERY (kappa) = {kappa:.3f} : smallest outset over-expansion")
        print(f"// whose 5th-percentile pre-clamp chroma ratio on the priority set reaches 1")

        # ---- rotations for the RECOVERED-CHROMA regime ----------------------
        # kappa recovery keeps the drifting pixels saturated, so the drift is now
        # visible where it used to be bleached invisible. Crucially, the old
        # minimax blue inset rotation (+24 deg, tuned to counter BLEACHED-pixel
        # drift) is measured to CAUSE most of the purple on RECOVERED saturated
        # blues (isolation : +13.7 deg with it, +1.5 without, at ~95% chroma).
        # So re-fit for the recovered regime. Structure that works (a free 6-param
        # search wanders into skin-wrecking minima) : keep inset red/green at their
        # diffuse-validated values (they protect skin, sunset and foliage — zeroing
        # them sends green to -25 deg and sunsets to +37 deg), fit inset BLUE plus
        # the three outset deltas. Priority weights, from the maintainer's stated
        # ordering (skin/reflective = absolute priority ; blue LED purple = the
        # worst real complaint ; sunset should drift yellow not red) :
        from scipy.optimize import minimize
        M_srgb = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                           [0.0193, 0.1192, 0.9505]])
        M_2020 = np.array([[0.6370, 0.1446, 0.1689], [0.2627, 0.6780, 0.0593],
                           [0.0000, 0.0281, 1.0610]])
        TO2020 = np.linalg.inv(M_2020) @ M_srgb
        blue_srgb = np.maximum(TO2020 @ np.array([0.0, 0.0, 1.0]), 1e-6)
        y_row = REC2020_TO_XYZ_D50[1]

        def place(rgb, evs):
            lum = y_row @ rgb
            return [rgb * (GREY * 2.0**e / lum) for e in evs]
        skin_set, sunset_set = [], []
        for (L, Ls, a, As, b, Bs) in _SKIN_LAB:
            for da in (-1.5, 1.5):
                for db in (-1.5, 1.5):
                    rgb = _lab_d65_to_work(L, a + da * As, b + db * Bs)
                    if rgb.min() > 0:
                        skin_set += place(rgb, (-1.5, 0.0, 1.5))
        for base in ([1, .45, .12], [1, .6, .2]):
            for pu in (.6, .9):
                sunset_set += place(np.maximum(TO2020 @ (pu * np.array(base) + (1 - pu) * .5), 1e-6),
                                    (0.0, 1.5, 3.0))

        def run_px(rgb, M_in, M_out):
            c_o, h_o = chroma_hue(rgb_work_to_Yrg(rgb))
            x = (np.log2(np.maximum(M_in @ rgb, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
            out = M_out @ np.array([CURVE(v) for v in x])
            c_f, h_f = chroma_hue(rgb_work_to_Yrg(np.maximum(out, 1e-10)))
            dh = np.rad2deg(np.remainder(h_f - h_o + np.pi, 2 * np.pi) - np.pi)
            return min(c_f, c_o) / max(c_o, 1e-9), dh, c_f / max(c_o, 1e-9)

        # sign convention : drift > 0 is yellow-ward for warm hues (skin/sunset),
        # drift < 0 is red-ward. Red-ward on skin/sunset is the maintainer's veto.
        def objective(p):
            irot = np.array([rot[0], rot[1], p[0]])       # inset red/green fixed, blue free
            orot = irot + p[1:]                           # outset = inset + deltas
            M_in, _ = bracket_matrices(insets0, irot)
            Mk, _ = bracket_matrices(np.clip(kappa * insets0, 0.0, 0.9), orot)
            M_out = np.linalg.inv(Mk)
            blue = max(abs(run_px(blue_srgb * GREY * 2.0**ev, M_in, M_out)[1]) for ev in (3.5, 4.0, 4.5))
            skd = [run_px(s, M_in, M_out)[1] for s in skin_set]
            snd = [run_px(s, M_in, M_out)[1] for s in sunset_set]
            p5 = np.percentile([run_px(s, M_in, M_out)[2] for s in samples[::4]], 5)
            posmin = M_in.min()
            return (blue                                        # #1 : blue purple
                    + 30.0 * max(0.0, -min(skd) - 1.5)          # skin red-ward veto (cap 1.5)
                    + 10.0 * max(0.0, max(skd) - 4.0)           # skin excess yellow
                    + 10.0 * max(0.0, -min(snd) - 1.0)          # sunset must not go red
                    + 100.0 * max(0.0, 0.9 - p5)                # diffuse recovery preserved
                    + 1e4 * max(0.0, 0.005 - posmin))           # inset positivity margin

        x0 = np.deg2rad([3.0, 0.4, 3.1, 4.5])
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"xatol": 2e-4, "fatol": 5e-3, "maxiter": 500})
        irot = np.array([rot[0], rot[1], res.x[0]])
        orot = irot + res.x[1:]
        print("static const float rotation_anchor[3] = { %+.7ff, %+.7ff, %+.7ff }; // %+.2f°, %+.2f°, %+.2f°"
              % (*irot, *np.rad2deg(irot)))
        print("static const float outset_rotation[3] = { %+.7ff, %+.7ff, %+.7ff }; // %+.2f°, %+.2f°, %+.2f°"
              % (*orot, *np.rad2deg(orot)))
        M_in, _ = bracket_matrices(insets0, irot)
        Mk, _ = bracket_matrices(np.clip(kappa * insets0, 0.0, 0.9), orot)
        M_out = np.linalg.inv(Mk)
        b4 = run_px(blue_srgb * GREY * 16.0, M_in, M_out)
        skd = [run_px(s, M_in, M_out)[1] for s in skin_set]
        snd = [run_px(s, M_in, M_out)[1] for s in sunset_set]
        print("// sRGB blue @+4EV drift %+.1f deg (%.0f%% chroma) ; skin [%+.1f..%+.1f] ; sunset [%+.1f..%+.1f]"
              % (b4[1], 100 * b4[0], min(skd), max(skd), min(snd), max(snd)))
        print("// inset positivity min %.4f ; portability (post-clamp p5) :" % M_in.min())
        for (bl, wh, name) in ((-4.0, 2.5, "studio 6.5EV"), (-8.0, 4.0, "default 12EV"), (-10.0, 6.0, "HDR 16EV")):
            c = curve_factory(bl, wh, *CURVE_DEFAULTS, shoulder_slope_matched=True)
            ratios = []
            for s in samples:
                c_o, _ = chroma_hue(rgb_work_to_Yrg(s))
                x = (np.log2(np.maximum(M_in @ s, 1e-10) / GREY) - bl) / (wh - bl)
                out = M_out @ np.array([c(v) for v in x])
                c_f, _ = chroma_hue(rgb_work_to_Yrg(np.maximum(out, 1e-10)))
                ratios.append(min(c_f / max(c_o, 1e-9), 1.0))
            print("//   %-13s p5 %.3f" % (name, np.percentile(ratios, 5)))
        return
    if args.minimax:
        from scipy.optimize import minimize
        evs = np.arange(-6.0, 3.51, 0.5)  # visible range : beyond, the gamut mapper crushes chroma

        def worst_drift(r):
            M, M_inv = bracket_matrices(insets0, r)
            M2, _ = bracket_matrices(np.clip(2.0 * insets0, 0.0, 0.9), 2.0 * r)
            pen = 1e4 * (np.maximum(0.0, 0.005 - M).sum() + np.maximum(0.0, 0.005 - M2).sum())
            worst = 0.0
            for s in boundary_colors():
                c_in, h_in = chroma_hue(rgb_work_to_Yrg(s))
                for ev in evs:
                    out = M_inv @ tone_map(M @ (s * GREY * 2.0 ** ev))
                    _, h = chroma_hue(rgb_work_to_Yrg(np.maximum(out, 1e-10)))
                    worst = max(worst, abs(np.remainder(h - h_in + np.pi, 2 * np.pi) - np.pi))
            return np.rad2deg(worst) + pen

        res = minimize(worst_drift, np.deg2rad([-2.0, 4.0, 11.0]), method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 400})
        insets, rotations = insets0, res.x
        fit_note = f"minimax, worst-case drift {res.fun:.2f} deg over EV <= +3.5"
    elif args.fit_insets:
        x0 = np.concatenate([insets0, rot0])
        bounds = ([0.02] * 3 + [np.deg2rad(-15)] * 3,
                  [0.45] * 3 + [np.deg2rad(15)] * 3)
        fit = least_squares(residuals, x0, bounds=bounds,
                            args=(np.deg2rad(args.drift_target),), verbose=1)
        insets, rotations = fit.x[:3], fit.x[3:]
    else:
        # rotations only : well-posed 3-parameter fit for the hue drift objective
        fit = least_squares(lambda r, tgt: residuals(np.concatenate([insets0, r]), tgt),
                            rot0, bounds=(np.deg2rad(-15), np.deg2rad(15)),
                            args=(np.deg2rad(args.drift_target),), verbose=1)
        insets, rotations = insets0, fit.x
    if not args.minimax:
        fit_note = f"drift target {args.drift_target} deg/EV, cost {fit.cost:.4f}"
    print(f"\n// fitted by tools/derive_filmic_agx_primaries.py ({fit_note})")
    print("static const float inset_anchor[3] = { %.6ff, %.6ff, %.6ff };" % tuple(insets))
    print("static const float rotation_anchor[3] = { %+.7ff, %+.7ff, %+.7ff }; // %+.2f°, %+.2f°, %+.2f°"
          % (tuple(rotations) + tuple(np.rad2deg(rotations))))
    M, _ = bracket_matrices(insets, rotations)
    print("// inset matrix (work RGB -> rendering), rows sum to 1:")
    for row in M:
        print("//   [ %+.6f %+.6f %+.6f ]  (sum %.6f)" % (*row, row.sum()))

if __name__ == "__main__":
    main()
