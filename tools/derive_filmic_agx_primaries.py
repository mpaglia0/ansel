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
    "no-bleach": dict(
        fit="--min-bleach --ab-pull 200",
        inset=[0.5991055, 0.6000000, 0.3300009],
        irot=[0.0571015, 0.1999891, 0.0886110],
        outset=[0.761433, 0.752267, 0.465293],
        orot=[-0.0034297, 0.1952448, -0.0480109]
    ),
    "low-bleach": dict(
        fit="--fit-bisect no-bleach medium-bleach",
        inset=[0.6410825, 0.6898110, 0.3194529],
        irot=[0.0405734, 0.1631286, 0.0350584],
        outset=[0.784757, 0.789387, 0.445403],
        orot=[-0.0057845, 0.1593207, -0.0592955]
    ),
    "medium-bleach": dict(
        fit="--fit-bisect no-bleach extra-bleach",
        inset=[0.6509540, 0.7488775, 0.3517703],
        irot=[0.0278602, 0.1214671, -0.0228829],
        outset=[0.793082, 0.815169, 0.460318],
        orot=[-0.0053781, 0.1187604, -0.0794801]
    ),
    "high-bleach": dict(
        fit="--fit-bisect medium-bleach extra-bleach",
        inset=[0.6379749, 0.7878689, 0.3753822],
        irot=[0.0106096, 0.0582598, -0.0696729],
        outset=[0.790237, 0.831376, 0.465406],
        orot=[-0.0080070, 0.0571100, -0.0912220]
    ),
    "extra-bleach": dict(
        fit="--fit-extra-bleach --ab-stabilize 70 --ab-level 10 --bleach-nudge 0.5",
        inset=[0.5770235, 0.8102094, 0.4000390],
        irot=[-0.0081060, -0.0034008, -0.1035236],
        outset=[0.766420, 0.838020, 0.465130],
        orot=[-0.0122011, -0.0021732, -0.0971215]
    ),
}

# Vectorized pipeline : NO subsampling. Every color in the priority set is
# evaluated on every objective call, so the optimizer can never game the
# average by wrecking a hue it cannot see. The tone curve branches on scalars,
# so bake it into a monotone lookup table and apply it with np.interp.
LUT_X = np.linspace(0.0, 1.0, 8193)
LUT_Y = np.array([CURVE(v) for v in LUT_X])

# Parameterization : the inset is a single UNIFORM scalar (p[0]), and the
# per-primary action lives entirely in the outset — which is exactly the
# structure the good fits converge to anyway (the shipped inset is 0.35 on all
# three). A per-primary inset is the lever the optimizer abused to game the
# gated metrics (green railed to 0.7, wrecking an unseen blue) ; removing that
# DoF structurally forbids the pathology. 10 params :
#   p[0:3]    inset chroma (R, G, B)
#   p[3:6]    inset rotations (R, G, B)
#   p[6:9]    outset chroma   (R, G, B)
#   p[9:12]   outset rotations (R, G, B)
def brk(p):
    M, _ = bracket_matrices(p[0:3], p[3:6])
    Mo, _ = bracket_matrices(p[6:9], p[9:12])
    return M, np.linalg.inv(Mo)

def chroma_hue_batch(RGB):
    """(N,3) working-linear-Rec2020 -> (chroma, hue) in Kirk Yrg, relative to white."""
    LMS = ((RGB @ REC2020_TO_XYZ_D50.T) @ XYZ_D50_to_D65_CAT16.T) @ XYZ_D65_to_LMS_2006.T
    a = LMS.sum(axis=1, keepdims=True)
    rg = (LMS / np.where(a == 0.0, 1.0, a)) @ LMS_to_filmlightRGB.T
    dr, dg = rg[:, 0] - WHITE_YRG[1], rg[:, 1] - WHITE_YRG[2]
    return np.hypot(dr, dg), np.arctan2(dg, dr)

def variant_bracket(v):
    """(inset matrix M, applied outset matrix) for a SHIPPED_VARIANTS entry."""
    M, _ = bracket_matrices(v["inset"], v["irot"])
    Mo, _ = bracket_matrices(v["outset"], v["orot"])
    return M, np.linalg.inv(Mo)

def measure_batch(S, C_in, H_in, M, Mo):
    """(N,3) samples through inset -> per-channel curve -> outset ; returns
    (chroma_ratio post output<=input clamp, hue_drift_deg), both (N,)."""
    x = (np.log2(np.maximum(S @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
    y = np.interp(np.clip(x, 0.0, 1.0).ravel(), LUT_X, LUT_Y).reshape(x.shape)
    O = np.maximum(y @ Mo.T, 1e-10)
    c_f, h_f = chroma_hue_batch(O)
    cr = np.minimum(c_f / np.maximum(C_in, 1e-9), 1.0)
    dh = np.rad2deg(np.remainder(h_f - H_in + np.pi, 2 * np.pi) - np.pi)
    return cr, dh

def hk_drift_batch(S, M, Mo):
    """Helmholtz-Kohlrausch apparent-brightness drift : excess(output) - excess(input),
    per sample, through the same inset -> curve -> outset path as measure_batch. Positive
    = the bracket made the colour read brighter-for-its-luminance than the original (H-K
    inflation) ; negative = deflation. The look wants this as close to 0 as it can, so the
    rendered colours keep the SAME apparent-brightness balance as the scene."""
    x = (np.log2(np.maximum(S @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
    y = np.interp(np.clip(x, 0.0, 1.0).ravel(), LUT_X, LUT_Y).reshape(x.shape)
    O = np.maximum(y @ Mo.T, 1e-10)
    return nayatani_hk_excess(O) - nayatani_hk_excess(S)

def delta_e_yrg(cr, dh_deg):
    """Perceptual color-shift distance in the chroma-NORMALIZED Yrg plane : the input
    sits at (1, 0) — chroma ratio 1, zero hue drift — and the output at (cr*cos, cr*sin),
    so the distance between them folds chroma loss AND hue drift into one number.
    0 = colour unchanged ; ~1 = fully bleached ; up to 2 = hue-flipped at full chroma.

    NOTE: subtracting the (1, 0) reference is the whole point. An earlier version
    computed hypot(cr*cos, cr*sin), which reduces algebraically to just cr — the output
    vector's LENGTH, not its distance from the input — so it was not a real delta-E and
    cancelled the desaturation term to a constant. Both --min-bleach and --max-desat now
    use THIS function."""
    r = np.deg2rad(dh_deg)
    return np.hypot(cr * np.cos(r) - 1.0, cr * np.sin(r))

def nayatani_hk_excess(RGB):
    """Helmholtz-Kohlrausch apparent-brightness excess (Gamma - 1), Nayatani (1997) VAC
    model, for a batch of working-linear-Rec2020 colours. This is the FRACTIONAL amount
    by which a chromatic colour looks brighter than an equally-luminous grey — a real
    perceptual effect that per-channel tone mapping amplifies unevenly by hue. It is
    ~0 for neutrals and yellow-greens and largest for saturated blue / red / magenta
    (measured : gray 0.00, red 0.32, green 0.12, blue 0.43 at equal luminance), which is
    exactly why an over-saturated red reads brighter than an equally-bright green.
    Luminance-independent (a fractional boost), so penalizing it targets chroma+hue only."""
    XYZ = RGB @ REC2020_TO_XYZ_D50.T
    X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
    denom = np.maximum(X + 15.0 * Y + 3.0 * Z, 1e-12)
    u, v = 4.0 * X / denom, 9.0 * Y / denom
    Xw, Yw, Zw = REC2020_TO_XYZ_D50 @ np.ones(3)                 # working white (D50)
    dw = Xw + 15.0 * Yw + 3.0 * Zw
    un, vn = 4.0 * Xw / dw, 9.0 * Yw / dw
    du, dv = u - un, v - vn
    s_uv = 13.0 * np.hypot(du, dv)                               # CIELUV saturation
    th = np.arctan2(dv, du)                                      # CIELUV hue angle
    q = (-0.01585 - 0.03017 * np.cos(th) - 0.04556 * np.cos(2 * th)
         - 0.02667 * np.cos(3 * th) - 0.00295 * np.cos(4 * th)
         + 0.14592 * np.sin(th) + 0.05084 * np.sin(2 * th)
         - 0.01944 * np.sin(3 * th) - 0.00776 * np.sin(4 * th))
    L_a = 63.66                                                  # adapting luminance -> K_Br ~= 1
    K_Br = 0.2717 * (6.469 + 6.362 * L_a ** 0.4495) / (6.469 + L_a ** 0.4495)
    return (0.0872 * K_Br - 0.1340 * q) * s_uv                  # Gamma - 1

def scene_ab_target(S_refl, hk_retention=1.0):
    """PRINCIPLED uniform apparent-brightness target for the apparent-brightness stabilizers
    (--ab-pull on min-bleach, and the reference level for --ab-stabilize on extra), replacing a
    hand-tuned constant. = mean over the reflective set of
        curve(L_in) * (1 + hk_retention * H-K_excess(input))
    i.e. the apparent brightness a colour gets if the ACHROMATIC tone curve maps its luminance and
    it keeps `hk_retention` of its OWN natural (input) Helmholtz-Kohlrausch excess :
        hk_retention = 1   -> SCENE-PRESERVING ceiling (full natural H-K pop, ~0.379)
        hk_retention = 0   -> GRAY-EQUIVALENT floor    (H-K fully neutralized, ~0.336)
        hk_retention = 0.5 -> midway (~0.357) ; LINEAR in hk_retention, so 0.5 IS the average of the two.
    Holding every hue at this value preserves the scene's apparent-brightness STRUCTURE (no hue
    over/under-brightened relative to the others) at the chosen H-K-retention level. Hue-independent
    (the set places every hue at luminance GREY*2^EV, matched), curve-relative (recomputes if the
    default curve or the set changes). A chroma-preserving end (min-bleach) wants retention 1 ; a
    bleaching end (extra), whose colours lose chroma/H-K, sits lower in the [floor, ceiling] band."""
    y_row = REC2020_TO_XYZ_D50[1]
    Lin = S_refl @ y_row
    x = (np.log2(np.maximum(Lin, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
    ach = np.interp(np.clip(x, 0.0, 1.0), LUT_X, LUT_Y)          # achromatic (gray) tone response
    return float((ach * (1.0 + hk_retention * nayatani_hk_excess(S_refl))).mean())

def skin_and_reflective_sets():
    """The SINGLE canonical colour-constancy evaluation set — shared by --report AND every
    fit mode, so a "desaturation %" (or hue / delta-E / H-K drift) means the same thing
    everywhere. Two disjoint arrays, each swept over tonal placements : the skin-tone
    database, and the reflective hue circle. The reflective circle spans a purity sweep
    from LOW (0.3 : diffuse matte reflectances) to HIGH (0.9 : high-chroma memory colours) —
    these used to live in two separate builders (this one and priority_samples), which is
    what let the modes disagree on what "reflective" meant ; they are merged here."""
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
        for p in (0.3, 0.5, 0.7, 0.9):   # 0.3 diffuse reflectance -> 0.9 high-chroma memory colour
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
            cv = np.interp(np.clip(x, 0.0, 1.0), LUT_X, LUT_Y)
            worst = min(worst, float(y_row @ (Mo @ cv)))
    return worst

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
    y_row = REC2020_TO_XYZ_D50[1]
    for s in _BND:
        lum0 = y_row @ s
        for ev in _BND_EV:
            mr = M @ (s * GREY * 2.0 ** ev / lum0)
            x = (np.log2(np.maximum(mr, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
            cv = np.interp(np.clip(x, 0.0, 1.0), LUT_X, LUT_Y)
            worst = max(worst, 0.25 * (y_row @ cv) - (y_row @ (Mo @ cv)))
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
        y = np.interp(np.clip(x, 0.0, 1.0).ravel(), LUT_X, LUT_Y).reshape(x.shape)
        return y @ Mo.T

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

def print_variant_entry(name, fit, inset, irot, outset, orot):
    """Print a SHIPPED_VARIANTS entry that can be pasted directly into the script."""
    print('    "%s": dict(' % name)
    print('        fit="%s",' % fit)
    print('        inset=[%.7f, %.7f, %.7f],' % tuple(inset))
    print('        irot=[%.7f, %.7f, %.7f],' % tuple(irot))
    print('        outset=[%.6f, %.6f, %.6f],' % tuple(outset))
    print('        orot=[%.7f, %.7f, %.7f]' % tuple(orot))
    print('    ),')
    
def print_c_case(case_name, fit, inset, irot, outset, orot):
    """Print a C switch-case block for the fitted bracket constants."""
    print('    case %s: // %s : %s' % (case_name, case_name, fit))
    print('      // fitted by tools/derive_filmic_agx_primaries.py %s' % fit)
    print('      inset_anchor[0] = %+.7ff; inset_anchor[1] = %+.7ff; inset_anchor[2] = %+.7ff;' % tuple(inset))
    print('      rotation_anchor[0] = %+.7ff; rotation_anchor[1] = %+.7ff; rotation_anchor[2] = %+.7ff;' % tuple(irot))
    print('      outset_anchor[0]   = %.6ff; outset_anchor[1] = %.6ff; outset_anchor[2] = %.6ff;' % tuple(outset))
    print('      outset_rotation[0] = %+.7ff; outset_rotation[1] = %+.7ff; outset_rotation[2] = %+.7ff;' % tuple(orot))
    print('      break;')

def report_variants():
    """Print the {avg,max} x {desaturation, hue drift, delta-E} x {skin,reflective} table
    for the shipped variants, plus a Rec2020 gamut-safety check. Desaturation is over all
    colors that carry chroma ; hue drift is measured where chroma survives (ratio > 0.2 —
    a bleached color has no meaningful hue) ; delta-E is the combined chroma+hue fidelity
    (delta_e_yrg) over ALL samples, the single tie-breaker metric. Source of the tables in
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
        de = delta_e_yrg(cr, dh)                            # combined chroma+hue fidelity, ALL samples
        hk = hk_drift_batch(S, M, Mo)                       # H-K excess(out) - excess(in), signed
        hk_ext = hk[np.argmax(np.abs(hk))]                  # signed drift of largest magnitude
        return (d.mean(), d.max(), g.mean(), g.max(),
                de.mean(), de.max(), hk.mean(), hk_ext)

    def row(name, s):
        # one Markdown row : "desat avg / max | hue avg / max | ΔE avg / max | H-K avg / max"
        return ("| %-13s | %4.1f / %4.1f | %4.1f / %4.1f | %.2f / %.2f | %+.3f / %+.3f |"
                % (name, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]))

    hdr = ("| Variant | Desat. % (avg / max) | Hue drift ° (avg / max) "
           "| ΔE (avg / max) | H-K drift (avg / max) |")
    sep = "|---|---|---|---|---|"

    print("<!-- Auto-generated by tools/derive_filmic_agx_primaries.py --report. Do not edit by hand. -->")
    print("<!-- Metrics over the skin database (De Rigal/Xiao) and the diffuse memory-colour hue")
    print("     circle, across tonal placements. Desaturation = 1 - output/input chroma (%). Hue")
    print("     drift = |output - input| hue in Kirk Yrg, where chroma survives. ΔE = combined")
    print("     chroma+hue move in the chroma-normalized Yrg plane. H-K drift = signed change in")
    print("     Nayatani Helmholtz-Kohlrausch apparent-brightness excess, output vs input. -->\n")

    print("### Skin tones\n")
    print(hdr); print(sep)
    for name, v in SHIPPED_VARIANTS.items():
        M, Mo = variant_bracket(v)
        print(row(name, stat(S_sk, Csk, Hsk, M, Mo)))

    print("\n### Reflective colours\n")
    print(hdr); print(sep)
    for name, v in SHIPPED_VARIANTS.items():
        M, Mo = variant_bracket(v)
        print(row(name, stat(S_rf, Crf, Hrf, M, Mo)))

    print("\n### Rec2020 gamut safety\n")
    print("<!-- Worst output luminance over the Rec2020 primaries and secondaries, EV -12..+8.")
    print("     Must stay > 0 — a negative value renders BLACK. -->\n")
    print("| Variant | Worst boundary luminance | Status |")
    print("|---|---|---|")
    for name, v in SHIPPED_VARIANTS.items():
        M, Mo = variant_bracket(v)
        wl = rec2020_worst_boundary_luminance(M, Mo)
        print("| %-13s | %+.4f | %s |" % (name, wl, "OK" if wl > 0 else "**BLACK**"))

def per_hue_ab_and_drift(v, S, C, H, bin_idx, nbins):
    """For one variant, return (apparent_brightness[nbins], signed_hue_drift_deg[nbins],
    output_chroma[nbins]) binned over the reflective hue circle. APPARENT BRIGHTNESS = output
    luminance x (1 + Nayatani H-K excess) — how bright a colour READS, not just its luminance.
    OUTPUT CHROMA = Yrg saturation — must DECREASE monotonically no->extra (the non-monotone-
    saturation bug lived here). Hue drift is signed, only where output chroma survives (ratio > 0.2)."""
    y_row = REC2020_TO_XYZ_D50[1]
    M, Mo = variant_bracket(v)
    x = (np.log2(np.maximum(S @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
    O = np.maximum(np.interp(np.clip(x, 0.0, 1.0).ravel(), LUT_X, LUT_Y).reshape(x.shape) @ Mo.T, 1e-10)
    ab = (O @ y_row) * (1.0 + nayatani_hk_excess(O))
    c_f, h_f = chroma_hue_batch(O)
    cr = np.minimum(c_f / np.maximum(C, 1e-9), 1.0)
    dh = np.rad2deg(np.remainder(h_f - H + np.pi, 2 * np.pi) - np.pi)
    AB = np.array([ab[bin_idx == b].mean() if (bin_idx == b).any() else np.nan for b in range(nbins)])
    HD = np.array([dh[(bin_idx == b) & (cr > 0.2)].mean() if ((bin_idx == b) & (cr > 0.2)).any()
                   else np.nan for b in range(nbins)])
    CH = np.array([c_f[bin_idx == b].mean() if (bin_idx == b).any() else np.nan for b in range(nbins)])
    return AB, HD, CH

def report_hue_continuity():
    """Per-hue colour-CONTINUITY diagnostic across the bleach ladder (reflective set). For each
    of 12 hue bins it prints, for every shipped variant, the APPARENT BRIGHTNESS (output luminance
    x (1 + H-K excess)) and the SIGNED hue drift, with the per-step deltas. A well-behaved ladder
    is MONOTONE with EVEN steps per hue ; a big jump (e.g. reds/magentas over-brightening at one
    step) or a sign reversal (a hue rotating against the ramp) marks a variant that left the ramp.
    This is the tool behind the no-bleach red-darkening + low-bleach hue-divergence fixes."""
    S_sk, S_rf = skin_and_reflective_sets()
    Crf, Hrf = chroma_hue_batch(S_rf)
    mr = Crf > 0.05
    S, C, H = S_rf[mr], Crf[mr], Hrf[mr]
    nbins = 12
    bin_idx = (np.floor(np.remainder(H, 2 * np.pi) / (2 * np.pi) * nbins).astype(int)) % nbins
    labels = ["red", "red-orange", "orange", "yellow-green", "green", "green-cyan",
              "cyan", "cyan-blue", "blue", "blue-magenta", "magenta", "magenta-red"]
    names = list(SHIPPED_VARIANTS.keys())
    data = {n: per_hue_ab_and_drift(SHIPPED_VARIANTS[n], S, C, H, bin_idx, nbins) for n in names}
    hdr = "%-13s" % "hue" + "".join("%9s" % n[:8] for n in names) + "  | per-step Δ"

    print("# PER-HUE COLOUR CONTINUITY across the bleach ladder (reflective set ; --diagnose).")
    print("# Smooth ladder = MONOTONE with EVEN per-step Δ. A big jump or a sign reversal marks")
    print("# a variant off the ramp (over-brightened reds/magentas, or a hue rotating the wrong way).\n")
    for title, idx, fmt, dfmt in [
            ("APPARENT BRIGHTNESS  (output luminance x (1 + Nayatani H-K excess))", 0, "%9.3f", "%+.3f"),
            ("OUTPUT CHROMA  (Yrg saturation ; must DECREASE monotonically no->extra)", 2, "%9.4f", "%+.4f"),
            ("SIGNED HUE DRIFT deg  (where output chroma survives, ratio > 0.2)", 1, "%9.1f", "%+.1f")]:
        print("## " + title)
        print(hdr)
        for b in range(nbins):
            vals = [data[n][idx][b] for n in names]
            if any(np.isnan(vals)):
                continue
            steps = " ".join(dfmt % (vals[i + 1] - vals[i]) for i in range(len(vals) - 1))
            print("%-13s" % labels[b] + "".join(fmt % v for v in vals) + "  | " + steps)
        print()

def print_diagnostics(p, message):
    inset = p[0:3]
    irot = p[3:6]
    outset = p[6:9]
    orot = p[9:12]
    SHIPPED_VARIANTS["new fitting"] = dict(
        fit=message,
        inset=inset,
        irot=irot,
        outset=outset,
        orot=orot
    )
    report_variants()
    print("// paste this into SHIPPED_VARIANTS:")
    print_variant_entry("new fitting", message, inset, irot, outset, orot)
    print("// paste this into C code:")
    print_c_case("new fitting", message, inset, irot, outset, orot)


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
    ap.add_argument("--interpolate-fits", action="store_true",
                    help="Fit the MEDIUM-BLEACH variant as the PERCEPTUAL MIDPOINT of low-bleach and "
                         "high-bleach. See --fit-low-bleach")
    ap.add_argument("--report", action="store_true",
                    help="Measure the three SHIPPED variants (no/low/high bleach) and print the "
                         "{avg ; max} desaturation and hue-shift table for skin tones vs "
                         "reflective colors — the source of the tables in doc/filmic-agx.md and "
                         "the user docs. Reads the constants from SHIPPED_VARIANTS (which must "
                         "mirror filmic_agx_prepare_bracket in the C), so it doubles as a drift "
                         "check that the shipped brackets still measure as documented.")
    ap.add_argument("--diagnose", action="store_true",
                    help="Per-hue colour-CONTINUITY diagnostic across the whole bleach ladder : for "
                         "each of 12 hue bins, the APPARENT BRIGHTNESS (output luminance x (1 + H-K "
                         "excess)) and the SIGNED hue drift of every shipped variant, with per-step "
                         "deltas. A smooth ladder is monotone with even steps ; a jump or sign "
                         "reversal flags a variant off the ramp (reds/magentas over-brightening, or "
                         "a hue rotating the wrong way). The tool behind the continuity fixes.")
    ap.add_argument("--hk-weight", type=float, default=0.0, metavar="W",
                    help="OPTIONAL Helmholtz-Kohlrausch FIDELITY term for --min-bleach (default 0 = "
                         "off). Saturated colours look brighter than an equally-luminous grey, hue-"
                         "dependently (strongest blue/red/magenta, weakest yellow-green ; Nayatani "
                         "1997 VAC model). A bracket that bleaches some hues more than others "
                         "therefore SHIFTS their apparent brightness and can amplify e.g. the "
                         "red<->green brightness gap. With W > 0, --min-bleach adds "
                         "W * mean(|H-K excess(output) - H-K excess(input)|) over the reflective "
                         "set : it penalizes the apparent-brightness CHANGE the bracket introduces, "
                         "keeping each colour's perceived brightness as close to the original as "
                         "possible (a fidelity term, complementary to the delta_e_yrg chroma/hue "
                         "distance). It is the DIFFERENCE before vs after, NOT the absolute output "
                         "excess — penalizing the absolute would flatten the vivid hues toward "
                         "neutral, i.e. AWAY from the original. Applied to reflective colours only, "
                         "so it does not fight skin-chroma protection. The per-set mean change is "
                         "small (the fit already preserves chroma) and the other objective terms "
                         "sum to ~1, so try W ~ 5-20 ; sweep to taste. Higher W holds apparent "
                         "brightness closer to the original.")
    ap.add_argument("--desat-frac", type=float, default=0.5, metavar="F",
                    help="For --fit-medium-bleach : position the interior variant at reflective-"
                         "desaturation fraction F between the no-bleach (0) and extra-bleach (1) "
                         "ends. 0.25 -> low-bleach, 0.5 -> medium-bleach (default), 0.75 -> high-"
                         "bleach. Same per-hue-ramp / soft-gamut / interpolation-anchor fit, so the "
                         "whole ladder stays a continuous, gamut-safe ramp between the settled ends.")
    ap.add_argument("--fit-extra-bleach", action="store_true",
                    help="Fit the EXTRA-BLEACH end : minimize the REFLECTIVE colours' hue shift "
                         "(bleach is allowed, hue must stay correct) AND the SKIN delta-E (skin "
                         "stays faithful) — both as the objective, not caps. No reflective desat "
                         "term, so the fit bleaches as hard as Rec2020 gamut safety + conditioning "
                         "allow, flattening per-channel hue drift : the extreme end of the look "
                         "axis. Hue is taken in radians to match the delta-E scale. Skin red-ward "
                         "drift stays vetoed. Reads nothing from SHIPPED_VARIANTS (a true end).")
    ap.add_argument("--ab-pull", type=float, default=0.0, metavar="W",
                    help="--min-bleach only : pull each hue's APPARENT BRIGHTNESS (output luminance "
                         "x (1 + Nayatani H-K excess)) toward the MIDPOINT of the shipped no-bleach "
                         "and low-bleach, so no-bleach stops darkening reds/magentas off the ladder "
                         "('true red sits between no and low'). Fixed reference read from "
                         "SHIPPED_VARIANTS once (not circular). 0 = off (pure min-delta-E).")
    ap.add_argument("--bleach-nudge", type=float, default=0.5, metavar="W",
                    help="Soft reflective-desaturation reward on --fit-extra-bleach (default 0.5). "
                         "Tips the hue-vs-bleach trade-off toward MORE bleaching without a hard "
                         "desat target — a nudge, not a requirement (extreme bleach is deferred to "
                         "creative grading). Bounded by the gamut penalty, skin delta-E and the "
                         "conditioning cap, so it stays gamut-safe. W=0 is the pure hue/skin-"
                         "faithful fit (the theoretically-sound end) ; raise it for a bolder end.")
    ap.add_argument("--ab-stabilize", type=float, default=0.0, metavar="W",
                    help="--fit-extra-bleach only : weight of per-hue APPARENT-BRIGHTNESS UNIFORMITY "
                         "(output luminance x (1 + H-K excess)) at the extra end — keep every hue at the "
                         "SAME apparent brightness so bleaching does NOT over-brighten reds/magentas "
                         "relative to other hues. This term is TARGET-FREE (penalizes spread around the "
                         "mean), so it does not make the solver sensitive to the exact target level. "
                         "0 = off. Pairs with --ab-level (the gentle absolute-level pull).")
    ap.add_argument("--ab-level", type=float, default=10.0, metavar="W",
                    help="--fit-extra-bleach only : weight of the GENTLE pull of the MEAN apparent "
                         "brightness toward the target LEVEL (scene_ab_target average, ~0.357), SEPARATE "
                         "from --ab-stabilize (uniformity). Kept LOW by design : folding the level into "
                         "the uniformity term (the old W*sum((ab-target)^2)) put ~45%% of the weight on "
                         "the absolute level, so a 0.003 target change flipped the fit into a blue-"
                         "distorting basin. 0 = let the level float entirely (uniformity only ; the "
                         "--bleach-nudge desat reward then sets the level).")
    ap.add_argument("--fit-bisect", nargs=2, metavar=("LO", "HI"),
                    help="Fit the PERCEPTUAL MIDPOINT between two shipped variants LO and HI : targets, "
                         "per reflective hue, the MIDPOINT of their apparent brightness AND signed hue "
                         "drift (+ skin faithfulness, gamut safety). Build the interior by SUCCESSIVE "
                         "BISECTION so every step is confined between its neighbours (monotone, even "
                         "steps) : medium = bisect(no-bleach, extra-bleach), then low = bisect(no-bleach, "
                         "medium-bleach), high = bisect(medium-bleach, extra-bleach). Re-fit inner "
                         "steps after either bounding variant changes.")
    args = ap.parse_args()
    
    if args.report:
        report_variants()
        return

    if args.diagnose:
        report_hue_continuity()
        return

    if args.fit_bisect:
        from scipy.optimize import minimize
        lo_key, hi_key = args.fit_bisect
        if lo_key not in SHIPPED_VARIANTS or hi_key not in SHIPPED_VARIANTS:
            print("// --fit-bisect needs two existing SHIPPED_VARIANTS keys (e.g. no-bleach extra-bleach).")
            return
        y_row = REC2020_TO_XYZ_D50[1]
        def _cah(RGB):                                       # module chroma_hue_batch is shadowed inside main()
            LMS = ((RGB @ REC2020_TO_XYZ_D50.T) @ XYZ_D50_to_D65_CAT16.T) @ XYZ_D65_to_LMS_2006.T
            a = LMS.sum(axis=1, keepdims=True)
            rg = (LMS / np.where(a == 0.0, 1.0, a)) @ LMS_to_filmlightRGB.T
            return np.hypot(rg[:, 0] - WHITE_YRG[1], rg[:, 1] - WHITE_YRG[2]), \
                   np.arctan2(rg[:, 1] - WHITE_YRG[2], rg[:, 0] - WHITE_YRG[1])
        S_sk, S_rf = skin_and_reflective_sets()
        Csk, Hsk = _cah(S_sk); Crf, Hrf = _cah(S_rf)
        mk = Csk > 0.04; S_sk, Csk, Hsk = S_sk[mk], Csk[mk], Hsk[mk]
        mr = Crf > 0.05; S_rf, Crf, Hrf = S_rf[mr], Crf[mr], Hrf[mr]
        nbins = 12
        bin_idx = (np.floor(np.remainder(Hrf, 2 * np.pi) / (2 * np.pi) * nbins).astype(int)) % nbins

        def ph_ab_hd(M, Mo):                                 # per-hue apparent brightness, hue drift, output chroma
            x = (np.log2(np.maximum(S_rf @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
            O = np.maximum(np.interp(np.clip(x, 0.0, 1.0).ravel(), LUT_X, LUT_Y).reshape(x.shape) @ Mo.T, 1e-10)
            ab = (O @ y_row) * (1.0 + nayatani_hk_excess(O))
            c_f, h_f = _cah(O)
            cr = np.minimum(c_f / np.maximum(Crf, 1e-9), 1.0)
            dh = np.rad2deg(np.remainder(h_f - Hrf + np.pi, 2 * np.pi) - np.pi)
            AB = np.array([ab[bin_idx == b].mean() if (bin_idx == b).any() else np.nan for b in range(nbins)])
            HD = np.array([dh[(bin_idx == b) & (cr > 0.2)].mean() if ((bin_idx == b) & (cr > 0.2)).any()
                           else np.nan for b in range(nbins)])
            CH = np.array([c_f[bin_idx == b].mean() if (bin_idx == b).any() else np.nan for b in range(nbins)])
            return AB, HD, CH

        lo_v, hi_v = SHIPPED_VARIANTS[lo_key], SHIPPED_VARIANTS[hi_key]
        AB_lo, HD_lo, CH_lo = ph_ab_hd(*variant_bracket(lo_v))
        AB_hi, HD_hi, CH_hi = ph_ab_hd(*variant_bracket(hi_v))
        ab_tgt = 0.5 * (AB_lo + AB_hi)                       # per-hue MIDPOINT of the two neighbours :
        hd_tgt = 0.5 * (HD_lo + HD_hi)                       # apparent brightness, signed hue drift,
        ch_tgt = 0.5 * (CH_lo + CH_hi)                       # AND output CHROMA (saturation) -> monotone ladder

        _bnd = [np.maximum(np.array(c, float), 1e-6) for c in
                ([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0])]
        BND = np.array([s * GREY * 2.0 ** ev / (y_row @ s) for s in _bnd for ev in np.arange(-12.0, 8.01, 0.5)])
        def worst_lum(M, Mo):
            x = (np.log2(np.maximum(BND @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
            O = np.interp(np.clip(x, 0.0, 1.0).ravel(), LUT_X, LUT_Y).reshape(x.shape) @ Mo.T
            return float((O @ y_row).min())
        BIG = 1e5
        # W_AB : hold each hue's apparent brightness on its neighbour-midpoint (nailed ; ~1e-4 scale).
        # W_CH : likewise nail each hue's output CHROMA (saturation) on the midpoint. WITHOUT it,
        # chroma is a FREE variable — only refl_dE nudges it toward the INPUT, not the midpoint — so
        # the 2nd-stage bisections (low, high) undershoot and saturation ZIG-ZAGS (reds/magentas end up
        # more muted at low/high than at medium/extra : the non-monotone-saturation bug). Same ~1e-4
        # scale as AB, so the same weight nails it → monotone saturation ladder.
        # W_HD : DAMPENING FACTOR for even hue-shift spacing between steps (deg^2 scale). At the fitted
        # optimum the colour-fidelity term refl_dE (~0.11) out-weighs the hue-midpoint term, which is
        # why hue steps are slightly front-loaded ; raise W_HD toward ~refl_dE to pull each step's hue
        # drift onto its exact midpoint = MORE EVEN hue steps, at a little input-hue fidelity. Tunable:
        # 0.30 = original (fidelity-first), 1.0 = even-spacing-first. Sweep to taste in --diagnose.
        W_AB, W_CH, W_HD = 4000.0, 4000.0, 1.0

        def objective(p):
            M, Mo = brk(p)
            if M.min() < 0.0:
                return BIG
            if max(np.linalg.cond(M), np.linalg.cond(Mo)) > 7.1:
                return BIG
            wl = worst_lum(M, Mo)
            if wl < -0.02:
                return BIG
            skin_cr, skin_dh = measure_batch(S_sk, Csk, Hsk, M, Mo)
            if skin_dh.min() < -3.0:                          # loose skin red-ward veto (racial bias)
                return BIG
            AB, HD, CH = ph_ab_hd(M, Mo)
            ab_err = float(np.nanmean((AB - ab_tgt) ** 2))    # per-hue apparent-brightness midpoint
            hd_err = float(np.nanmean((HD - hd_tgt) ** 2))    # per-hue signed hue-drift midpoint
            ch_err = float(np.nanmean((CH - ch_tgt) ** 2))    # per-hue output-chroma midpoint (monotone saturation)
            skin_dE = delta_e_yrg(skin_cr, skin_dh)
            gamut_pen = 1.0e4 * max(0.0, 0.0002 - wl)
            refl_cr, refl_dh = measure_batch(S_rf, Crf, Hrf, M, Mo)
            refl_dE = delta_e_yrg(refl_cr, refl_dh)
            return (W_AB * ab_err + W_CH * ch_err + W_HD * hd_err
                    + skin_dE.mean() + skin_dE.max() + refl_dE.mean() + gamut_pen)

        def interp_prim(c0, a0, c1, a1, t):                  # chroma-plane (qualia-preserving) seed
            oc, oa = [], []
            for i in range(3):
                vx = (1 - t) * c0[i] * np.cos(a0[i]) + t * c1[i] * np.cos(a1[i])
                vy = (1 - t) * c0[i] * np.sin(a0[i]) + t * c1[i] * np.sin(a1[i])
                oc.append(float(np.hypot(vx, vy))); oa.append(float(np.arctan2(vy, vx)))
            return oc, oa
        def interp_at(t):
            i_in, i_ir = interp_prim(lo_v["inset"], lo_v["irot"], hi_v["inset"], hi_v["irot"], t)
            i_out, i_or = interp_prim(lo_v["outset"], lo_v["orot"], hi_v["outset"], hi_v["orot"], t)
            return i_in + i_ir + i_out + i_or
        def flat(v):
            return [*v["inset"], *v["irot"], *v["outset"], *v["orot"]]

        seeds = [interp_at(t) for t in np.linspace(0.2, 0.8, 5)] + [flat(lo_v), flat(hi_v)]
        best = None
        for s in seeds:
            r = minimize(objective, list(s), method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-7, "maxiter": 9000, "maxfev": 9000})
            if r.fun < BIG and (best is None or r.fun < best.fun):
                best = r
        if best is None:
            print("// no feasible bisection bracket (gamut-safe) between %s and %s." % (lo_key, hi_key))
            return
        p = best.x
        AB, HD, CH = ph_ab_hd(*brk(p))
        print("// bisect(%s, %s) : per-hue RMS-to-midpoint  AB %.4f  chroma %.4f  hue-drift %.2f° ; gamut %+.4f ; cond %.1f"
              % (lo_key, hi_key, np.sqrt(np.nanmean((AB - ab_tgt) ** 2)),
                 np.sqrt(np.nanmean((CH - ch_tgt) ** 2)),
                 np.sqrt(np.nanmean((HD - hd_tgt) ** 2)),
                 rec2020_worst_boundary_luminance(*brk(p)), max(np.linalg.cond(brk(p)[0]), np.linalg.cond(brk(p)[1]))))
        print_diagnostics(p, "--fit-bisect %s %s" % (lo_key, hi_key))
        return

    if args.fit_extra_bleach:
        from scipy.optimize import minimize
        def _cah(RGB):                                       # module chroma_hue_batch is shadowed inside main()
            LMS = ((RGB @ REC2020_TO_XYZ_D50.T) @ XYZ_D50_to_D65_CAT16.T) @ XYZ_D65_to_LMS_2006.T
            a = LMS.sum(axis=1, keepdims=True)
            rg = (LMS / np.where(a == 0.0, 1.0, a)) @ LMS_to_filmlightRGB.T
            return np.hypot(rg[:, 0] - WHITE_YRG[1], rg[:, 1] - WHITE_YRG[2]), \
                   np.arctan2(rg[:, 1] - WHITE_YRG[2], rg[:, 0] - WHITE_YRG[1])
        S_sk, S_rf = skin_and_reflective_sets()
        Csk, Hsk = _cah(S_sk)
        Crf, Hrf = _cah(S_rf)
        mk = Csk > 0.04; S_sk, Csk, Hsk = S_sk[mk], Csk[mk], Hsk[mk]
        mr = Crf > 0.05; S_rf, Crf, Hrf = S_rf[mr], Crf[mr], Hrf[mr]
        BIG = 1e5

        # VECTORIZED boundary luminance for the gamut penalty : the module
        # rec2020_worst_boundary_luminance is a 246-iteration Python loop, far too slow to call
        # every objective eval in a sweep. Precompute the Rec2020 primary/secondary samples at
        # each EV (the danger zone is dark BLUE ~EV -5.5) and batch the tone-map.
        _yr = REC2020_TO_XYZ_D50[1]
        _bnd = [np.maximum(np.array(c, float), 1e-6) for c in
                ([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0])]
        BND = np.array([s * GREY * 2.0 ** ev / (_yr @ s) for s in _bnd for ev in np.arange(-12.0, 8.01, 0.5)])
        def worst_lum(M, Mo):
            x = (np.log2(np.maximum(BND @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
            O = np.interp(np.clip(x, 0.0, 1.0).ravel(), LUT_X, LUT_Y).reshape(x.shape) @ Mo.T
            return float((O @ _yr).min())

        COND_CAP = 7.0             # extra-bleach's bleach intensity dial : the objective has no
                                   # desat term, so it bleaches until conditioning binds (~6.5 is the
                                   # current extra-bleach level ; raise it for an even more extreme end).

        # per-hue APPARENT BRIGHTNESS reference (no-bleach) for --ab-stabilize : keep the extra end
        # from over-brightening reds/magentas — apparent brightness should stay ~put across the
        # bleach axis (only chroma/hue change). Reflective, 12 hue bins.
        _abnb = 12
        _abin = (np.floor(np.remainder(Hrf, 2 * np.pi) / (2 * np.pi) * _abnb).astype(int)) % _abnb
        def _ph_ab(M, Mo):
            x = (np.log2(np.maximum(S_rf @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
            O = np.maximum(np.interp(np.clip(x, 0.0, 1.0).ravel(), LUT_X, LUT_Y).reshape(x.shape) @ Mo.T, 1e-10)
            ab = (O @ _yr) * (1.0 + nayatani_hk_excess(O))
            return np.array([ab[_abin == b].mean() if (_abin == b).any() else np.nan for b in range(_abnb)])
        # Uniform apparent-brightness target = the DATA-DERIVED AVERAGE of the H-K-neutral
        # gray-equivalent FLOOR (scene_ab_target(...,0.0) ~0.336) and the scene-preserving CEILING
        # (scene_ab_target(...,1.0) ~0.379) = ~0.357 (equivalently scene_ab_target(...,0.5)). Below
        # min-bleach's full-retention value : the extra end bleaches chroma/H-K OUT, so a lower
        # target is faithful AND permits more desaturation (verified: lower desaturates better).
        # Replaces the hand-tuned 0.360 it lands on.
        ab_no_ref = [0.5 * (scene_ab_target(S_rf, 0.0) + scene_ab_target(S_rf, 1.0))] * _abnb

        # per-hue OUTPUT-CHROMA CEILING (no-bleach) : extra-bleach MUST be LESS saturated than
        # no-bleach at EVERY hue — more bleach => less chroma. The colour-fidelity term (global_dE)
        # rewards chroma retention toward the INPUT, which lets the outset over-recover MAGENTA (R+B)
        # ABOVE the no-bleach level : a saturation-ORDER inversion (extra magenta > no magenta) that
        # then propagates through the bisections. One-sided penalty on any hue whose extra output
        # chroma exceeds no-bleach's. Fixed reference (read from SHIPPED once).
        def _ph_chroma(M, Mo):
            x = (np.log2(np.maximum(S_rf @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
            O = np.maximum(np.interp(np.clip(x, 0.0, 1.0).ravel(), LUT_X, LUT_Y).reshape(x.shape) @ Mo.T, 1e-10)
            c_f, _h = _cah(O)
            return np.array([c_f[_abin == b].mean() if (_abin == b).any() else np.nan for b in range(_abnb)])
        ch_no_ref = _ph_chroma(*variant_bracket(SHIPPED_VARIANTS["no-bleach"]))

        def objective(p):
            M, Mo = brk(p)
            if M.min() < 0.0:
                return BIG
            if max(np.linalg.cond(M), np.linalg.cond(Mo)) > COND_CAP:
                return BIG
            wl = worst_lum(M, Mo)                            # vectorized boundary luminance (fast)
            if wl < -0.02:                                   # real black — reject outright
                return BIG
            skin_cr, skin_dh = measure_batch(S_sk, Csk, Hsk, M, Mo)
            refl_cr, refl_dh = measure_batch(S_rf, Crf, Hrf, M, Mo)
            if skin_dh.min() < -3.0:                         # loose skin red-ward safety veto
                return BIG
            # EXTRA-BLEACH objective : minimize REFLECTIVE hue shift (in radians, to match the
            # delta-E scale) + SKIN delta-E, directly (not caps). No reflective desat term, so
            # the bracket bleaches as hard as conditioning allows to flatten hue — the extreme,
            # hue-faithful end. Skin delta-E keeps skin from washing out with it.
            # GAMUT as a SOFT penalty (not a hard wall) : minimizing reflective hue wants strong
            # blue bleach, which pushes dark BLUE (~EV -5.5) negative -> black. A hard barrier
            # cannot be gradient-navigated at that edge (the 512-start sweep found nothing), so
            # trade hue against blue-gamut smoothly, driving the worst luminance up to +0.0005.
            # NO H-K term here : minimizing reflective H-K drift on the extreme end pushes the
            # optimizer to over-brighten saturated red/magenta (the "self-luminous"/neon lipstick
            # artifact — skin stays stable but red-fuchsia reads too luminous). The extra end is
            # deliberately hue-faithful + skin-faithful ONLY ; H-K fidelity is left to the tamer
            # variants, where the shallower bracket does not amplify saturated-red apparent brightness.
            refl_hue = np.deg2rad(np.abs(refl_dh[refl_cr > 0.2]))
            skin_dE = delta_e_yrg(skin_cr, skin_dh)
            global_dE = delta_e_yrg(refl_cr, refl_dh)
            gamut_pen = 1.0e4 * max(0.0, 0.0002 - wl)        # keep dark blue clearly positive (achievable)
            # SOFT bleach preference : a gentle reward (NOT a hard desat target) that tips the
            # hue-vs-bleach trade-off toward MORE reflective desaturation when fidelity is roughly
            # tied — extreme creative bleach is left to grading, this only leans the character.
            # Bounded by gamut_pen (1e4), skin_dE and the conditioning cap, so it cannot push the
            # bracket gamut-unsafe ; --bleach-nudge 0 recovers the pure hue/skin-faithful fit.
            bleach_reward = args.bleach_nudge * float(np.mean(1.0 - refl_cr))
            # SOFT per-hue apparent-brightness stabilization toward no-bleach (--ab-stabilize) :
            # keeps reds/magentas from over-brightening at the extreme end (bleach = desaturate at
            # ~constant apparent brightness). 0 = off.
            ab_stab = 0.0
            if args.ab_stabilize > 0.0 or args.ab_level > 0.0:
                ab = _ph_ab(M, Mo)
                ab_mean = np.nanmean(ab)
                nvalid = int(np.sum(~np.isnan(ab)))
                # DECOUPLED apparent-brightness stabilisation (fixes the target sensitivity) :
                #  - UNIFORMITY (--ab-stabilize) : spread of per-hue AB around its own mean. TARGET-FREE,
                #    strong — the real goal (no hue over/under-bright relative to the others).
                #  - LEVEL (--ab-level) : gentle pull of the MEAN toward the target. Target-sensitive,
                #    so kept weak. The old coupled sum((ab-target)^2) == var + nvalid*(mean-target)^2 put
                #    ~45%% of W on the level, so a 0.003 target shift moved the objective ~0.03 and flipped
                #    the winning seed into a blue-distorting basin. ab_no_ref is uniform -> [0] is the level.
                ab_stab = (args.ab_stabilize * float(np.nansum((ab - ab_mean) ** 2))
                           + args.ab_level * nvalid * float((ab_mean - ab_no_ref[0]) ** 2))
            # CHROMA CEILING vs no-bleach : one-sided penalty forcing extra output chroma STRICTLY
            # BELOW no-bleach at every hue (fixes the magenta saturation-order inversion). The 0.98
            # margin makes it clearly LOWER (not merely equal) — "more bleach => less chroma". Strong
            # so it firmly binds ; only bites the hue(s) near/above the ceiling (the rest sit well below).
            chroma_ceiling = 5.0e3 * float(np.nansum(np.maximum(0.0, _ph_chroma(M, Mo) - 0.98 * ch_no_ref) ** 2))
            return (refl_hue.mean() + refl_hue.max()
                    + skin_dE.mean() + global_dE.mean() + skin_dE.max() + skin_dE.mean()
                    + gamut_pen - bleach_reward + ab_stab + chroma_ceiling)

        bounds = [(0.40, 0.98)] * 3 + [(-0.2, 0.2)] * 3 + [(0.30, 0.98)] * 3 + [(-0.2, 0.2)] * 3
        # TARGETED seed set (was a blind 4^3 x 4^3 = 4096-start grid). The strictly-gamut-safe
        # extreme bracket is a SINGLE known basin : it rails the inset floor (~0.40 red/blue, ~0.78
        # green) at cond 7.0, and every grid start converged there. So seed from the shipped extra
        # + a handful of structured points with the right per-channel shape (blue/red deep, green
        # shallow) and refine locally — same optimum, ~500x fewer evals. Widen again only if the
        # objective changes basin (watch the printed best).
        ex = SHIPPED_VARIANTS["extra-bleach"]
        seeds = [[*ex["inset"], *ex["irot"], *ex["outset"], *ex["orot"]]]
        for ins in ([0.40, 0.78, 0.40], [0.45, 0.75, 0.43], [0.55, 0.80, 0.45]):
            for out in ([0.34, 0.83, 0.37], [0.50, 0.80, 0.45]):
                seeds.append([*ins, 0.0, 0.0, 0.0, *out, 0.0, 0.0, 0.0])
        best = None
        for guess in seeds:
            r = minimize(objective, guess, method="Nelder-Mead", bounds=bounds,
                         options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 8000, "maxfev": 8000})
            if r.fun < BIG and (best is None or r.fun < best.fun):
                best = r
                Mb, Mob = brk(r.x)
                print("// extra best obj %.4f  inset %.2f/%.2f/%.2f  gamut %+.4f  cond %.1f"
                      % (r.fun, r.x[0], r.x[1], r.x[2],
                         rec2020_worst_boundary_luminance(Mb, Mob),
                         max(np.linalg.cond(Mb), np.linalg.cond(Mob))))
        if best is None:
            print("// no feasible extra-bleach bracket under the constraints.")
            return
        print_diagnostics(best.x, "--fit-extra-bleach")
        return

    if args.interpolate_fits:
        n_fits = 3  # intermediate steps between bounds
        bounds = [SHIPPED_VARIANTS["no-bleach"], SHIPPED_VARIANTS["extra-bleach"]]

        def compute_vector(coeff, angle):
            return coeff * np.cos(angle), coeff * np.sin(angle)

        def vector_to_coeff_angle(vec):
            coeff = np.hypot(vec[0], vec[1])
            angle = np.arctan2(vec[1], vec[0])
            return coeff, angle

        def build_variant_from_vectors(name, fit, in_vectors, out_vectors):
            inset = []
            irot = []
            outset = []
            orot = []
            for vec in in_vectors:
                coeff, angle = vector_to_coeff_angle(vec)
                inset.append(float(coeff))
                irot.append(float(angle))
            for vec in out_vectors:
                coeff, angle = vector_to_coeff_angle(vec)
                outset.append(float(coeff))
                orot.append(float(angle))
            variant = dict(fit=fit, inset=inset, irot=irot, outset=outset, orot=orot)
            SHIPPED_VARIANTS[name] = variant
            return variant

        def interpolate_vectors(vec0, vec1, t):
            return ((1.0 - t) * vec0[0] + t * vec1[0],
                    (1.0 - t) * vec0[1] + t * vec1[1])

        primaries_in = []
        primaries_out = []
        for bound in bounds:
            for i in range(3):
                primaries_in.append(compute_vector(bound["inset"][i], bound["irot"][i]))
                primaries_out.append(compute_vector(bound["outset"][i], bound["orot"][i]))

        print("// interpolated variants between no-bleach and extra-bleach")
        for k in range(1, n_fits + 1):
            t = float(k) / (n_fits + 1)
            in_vectors = [interpolate_vectors(primaries_in[i], primaries_in[i + 3], t)
                          for i in range(3)]
            out_vectors = [interpolate_vectors(primaries_out[i], primaries_out[i + 3], t)
                           for i in range(3)]
            name = "interp-%d" % k
            fit = "--interpolate-fits %d/%d" % (k, n_fits)
            variant = build_variant_from_vectors(name, fit, in_vectors, out_vectors)
            print("// %s t=%.3f" % (name, t))
            print_variant_entry(name, fit, variant["inset"], variant["irot"], variant["outset"], variant["orot"])
            print_c_case("DT_FILMIC_COLORSCIENCE_V%d" % (8 + k - 1), fit,
                         variant["inset"], variant["irot"], variant["outset"], variant["orot"])

        report_variants()
        return

    if args.fit_low_bleach:
        p = fit_midpoint("no-bleach", "high-bleach", 0.20, 0.75, (0.35, 0.45, 0.55))
        if p is not None:
            print_midpoint_constants(p, "--fit-low-bleach", "no-bleach and high-bleach")
        else:
            print("// no feasible low-bleach midpoint under the constraints.")
        return
    
    if args.fit_medium_bleach:
        from scipy.optimize import minimize
        if "no-bleach" not in SHIPPED_VARIANTS or "extra-bleach" not in SHIPPED_VARIANTS:
            print("// medium needs the two ENDS in SHIPPED_VARIANTS first — refit no-bleach and extra-bleach.")
            return
        y_row = REC2020_TO_XYZ_D50[1]

        def _cah(RGB):                                       # module chroma_hue_batch is shadowed inside main()
            LMS = ((RGB @ REC2020_TO_XYZ_D50.T) @ XYZ_D50_to_D65_CAT16.T) @ XYZ_D65_to_LMS_2006.T
            a = LMS.sum(axis=1, keepdims=True)
            rg = (LMS / np.where(a == 0.0, 1.0, a)) @ LMS_to_filmlightRGB.T
            return np.hypot(rg[:, 0] - WHITE_YRG[1], rg[:, 1] - WHITE_YRG[2]), \
                   np.arctan2(rg[:, 1] - WHITE_YRG[2], rg[:, 0] - WHITE_YRG[1])
        S_sk, S_rf = skin_and_reflective_sets()
        Csk, Hsk = _cah(S_sk)
        mk = Csk > 0.04; S_sk, Csk, Hsk = S_sk[mk], Csk[mk], Hsk[mk]
        Crf, Hrf = _cah(S_rf)
        mr = Crf > 0.05; S_rf, Crf, Hrf = S_rf[mr], Crf[mr], Hrf[mr]
        no_v, ex_v = SHIPPED_VARIANTS["no-bleach"], SHIPPED_VARIANTS["extra-bleach"]

        def flat(v):
            return [*v["inset"], *v["irot"], *v["outset"], *v["orot"]]

        def interp_prim(c0, a0, c1, a1, t):                  # chroma-plane interpolation (qualia)
            oc, oa = [], []
            for i in range(3):
                vx = (1 - t) * c0[i] * np.cos(a0[i]) + t * c1[i] * np.cos(a1[i])
                vy = (1 - t) * c0[i] * np.sin(a0[i]) + t * c1[i] * np.sin(a1[i])
                oc.append(float(np.hypot(vx, vy))); oa.append(float(np.arctan2(vy, vx)))
            return oc, oa
        def interp_at(t):
            i_in, i_ir = interp_prim(no_v["inset"], no_v["irot"], ex_v["inset"], ex_v["irot"], t)
            i_out, i_or = interp_prim(no_v["outset"], no_v["orot"], ex_v["outset"], ex_v["orot"], t)
            return i_in + i_ir + i_out + i_or

        # per-hue APPARENT BRIGHTNESS = output luminance x (1 + H-K excess). "Darkening reds/
        # magentas" is a drop in this ; keeping every hue's value inside the ends' [min,max]
        # bracket forbids any hue leaving the ramp (the exact bug). Reflective, 12 hue bins.
        nbins = 12
        bin_idx = (np.floor(np.remainder(Hrf, 2 * np.pi) / (2 * np.pi) * nbins).astype(int)) % nbins

        def per_hue_ab(M, Mo):
            x = (np.log2(np.maximum(S_rf @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
            O = np.maximum(np.interp(np.clip(x, 0.0, 1.0).ravel(), LUT_X, LUT_Y).reshape(x.shape) @ Mo.T, 1e-10)
            ab = (O @ y_row) * (1.0 + nayatani_hk_excess(O))
            return np.array([ab[bin_idx == b].mean() if (bin_idx == b).any() else np.nan for b in range(nbins)])

        def desat_of(M, Mo):
            rcr, _ = measure_batch(S_rf, Crf, Hrf, M, Mo)
            return float((1.0 - rcr).mean())

        Mno, Mono_ = variant_bracket(no_v)
        Mex, Moex = variant_bracket(ex_v)
        ab_no, ab_ex = per_hue_ab(Mno, Mono_), per_hue_ab(Mex, Moex)
        ab_lo, ab_hi = np.minimum(ab_no, ab_ex), np.maximum(ab_no, ab_ex)
        frac = float(args.desat_frac)                                   # 0.25 low, 0.5 medium, 0.75 high
        d_no, d_ex = desat_of(Mno, Mono_), desat_of(Mex, Moex)
        target = d_no + frac * (d_ex - d_no)                            # reflective-desat position on the ramp

        # smooth-character anchor : the interpolation point at the desat midpoint (itself may be
        # gamut-unsafe, but we only PULL toward it while enforcing gamut safety hard below).
        ts = np.linspace(0.0, 1.0, 61)
        dpath = np.array([desat_of(*brk(interp_at(t))) for t in ts])
        anchor = interp_at(float(ts[int(np.argmin(np.abs(dpath - target)))]))

        # vectorized boundary luminance for the SOFT gamut penalty (rec2020_worst_boundary_
        # luminance is a slow Python loop) — Rec2020 primaries/secondaries over EV -12..+8.
        _bnd = [np.maximum(np.array(c, float), 1e-6) for c in
                ([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0])]
        BND = np.array([s * GREY * 2.0 ** ev / (y_row @ s) for s in _bnd for ev in np.arange(-12.0, 8.01, 0.5)])
        def worst_lum(M, Mo):
            x = (np.log2(np.maximum(BND @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
            O = np.interp(np.clip(x, 0.0, 1.0).ravel(), LUT_X, LUT_Y).reshape(x.shape) @ Mo.T
            return float((O @ y_row).min())
        BIG = 1e5

        def objective(p):
            M, Mo = brk(p)
            if M.min() < 0.0:
                return BIG
            if max(np.linalg.cond(M), np.linalg.cond(Mo)) > 7.1:        # include the ends (extra ~7.0)
                return BIG
            wl = worst_lum(M, Mo)
            if wl < -0.02:                                              # real black — reject outright
                return BIG
            skin_cr, skin_dh = measure_batch(S_sk, Csk, Hsk, M, Mo)
            if skin_dh.min() < -3.0:
                return BIG
            pos = (desat_of(M, Mo) - target) ** 2                        # position at bleach midpoint
            ab = per_hue_ab(M, Mo)
            mono = float(np.nansum(np.maximum(0.0, ab_lo - ab) + np.maximum(0.0, ab - ab_hi)))  # per-hue ramp
            skin_dE = delta_e_yrg(skin_cr, skin_dh)
            qualia = float(np.sum((np.array(p) - np.array(anchor)) ** 2))
            gamut_pen = 1.0e4 * max(0.0, 0.0002 - wl)                    # SOFT : keep dark blue clearly positive
            return 100.0 * pos + 20.0 * mono + skin_dE.mean() + skin_dE.max() + 0.30 * qualia + gamut_pen

        # bruteforce-ish seed set : the two ends + several interpolation-path points (the soft
        # gamut penalty guides infeasible seeds back, so no need to pre-filter for gamut).
        seeds = [flat(no_v), flat(ex_v)] + [interp_at(tt) for tt in np.linspace(0.2, 0.9, 6)]
        best = None
        for s in seeds:
            r = minimize(objective, list(s), method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-7, "maxiter": 9000, "maxfev": 9000})
            if r.fun < BIG and (best is None or r.fun < best.fun):
                best = r
        if best is None:
            print("// no feasible medium-bleach bracket (gamut-safe, inside the per-hue ramp).")
            return
        p = best.x
        M, Mo = brk(p)
        ab = per_hue_ab(M, Mo)
        n_out = int(np.nansum((ab < ab_lo - 1e-3) | (ab > ab_hi + 1e-3)))
        print("// interior fit frac %.2f : reflective desat %.1f%% (target %.1f%%, ends %.1f..%.1f) ; "
              "hues out-of-ramp %d/%d ; gamut %+.4f ; cond %.1f"
              % (frac, 100 * desat_of(M, Mo), 100 * target, 100 * d_no, 100 * d_ex, n_out, nbins,
                 rec2020_worst_boundary_luminance(M, Mo), max(np.linalg.cond(M), np.linalg.cond(Mo))))
        print_diagnostics(p, "--fit-medium-bleach --desat-frac %.2f" % frac)
        return

    if args.max_desat is not None:
        from scipy.optimize import minimize
        budget = float(args.max_desat)
        # UNIFIED colour-constancy set (single source of truth) : the SAME skin + reflective
        # samples --report and every other fit mode use, so a "desaturation %" is comparable
        # across modes. This mode used to conflate skin into the reflective bucket
        # (priority_samples returns skin+diffuse combined) and build a divergent ±1-std / 4-EV
        # skin set, which is exactly why its "reflective desat" disagreed with the report's.
        S_skin, S_refl = skin_and_reflective_sets()

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
        hk_in_refl = nayatani_hk_excess(S_refl)             # input H-K excess (invariant)

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
            M, Mo = brk(p)
            
            # reject negative input matrices
            if M.min() < 0.0:
                return None
            
            # reject near-degenerate brackets
            cond = max(np.linalg.cond(M), np.linalg.cond(Mo))
            if cond > 6.5:                                   
                return None
            
            # reject brackets that escape Rec2020 gamut
            if gamut_violation(M, Mo) > 0.0:                                     
                return None
            
            skin_cr, skin_dh = measure(S_skin, C_skin, H_skin, M, Mo)
            refl_cr, refl_dh = measure(S_refl, C_refl, H_refl, M, Mo)
            return skin_cr, skin_dh, refl_cr, refl_dh, cond

        def objective(p):
            ev = evaluate(p)
            if ev is None:
                return BIG
            skin_cr, skin_dh, refl_cr, refl_dh, cond = ev
            # desaturation = the chroma the bracket costs (mostly the bright-color /
            # highlight bleach — the AgX wash-out look — which IS the cost the budget
            # trades ; individual highlights bleaching hard is fine, so no per-color cap).
            desats = np.concatenate([1.0 - skin_cr, 1.0 - refl_cr])
            mean_desat = float(desats.mean())
            gated = np.abs(refl_dh[refl_cr > 0.2])           # hue where chroma survives
            worst_hue = float(gated.max())
            # hue fidelity : mean AND worst single color ('best hue match' = no color
            # badly off, not a good average hiding an outlier).
            hue_err = np.mean(gated) + worst_hue + np.mean(np.abs(skin_dh)) + np.max(np.abs(skin_dh))
            # delta-E SAFETY JACKET (the shared delta_e_yrg, same metric as --min-bleach).
            # It folds skin chroma-loss AND skin
            # hue-drift into one perceptual shift. A THRESHOLD CAP, not a co-objective : it
            # stays slack (zero) until the worst skin colour shifts past SKIN_DE_CAP, then
            # bites hard — so the aggressive reflective bleaching this mode trades for hue
            # stability keeps its character, but skin cannot be quietly whitened past the
            # cap (a racial-bias failure). NOTE the coupling : skin is red/yellow, so
            # sparing it forces the red/green inset down (the per-channel inset keeps blue
            # high) — a tighter cap therefore softens the reflective bleaching too. Only
            # SKIN is jacketed : reflective colours are MEANT to bleach, so their delta-E
            # saturates near 1 by design and cannot tell an intended wash-out from a
            # pathological one. SKIN_DE_CAP ~= worst tolerated skin chroma loss (0.12 ~ 12%).
            SKIN_DE_CAP = 0.12
            skin_dE = delta_e_yrg(skin_cr, skin_dh)
            skin_jacket = 200.0 * max(0.0, float(skin_dE.max()) - SKIN_DE_CAP)

            # OPTIONAL H-K fidelity term, OFF by default (--hk-weight 0). On the heavy-bleach
            # top end minimizing reflective H-K drift over-brightens saturated red/magenta —
            # the "self-luminous"/neon lipstick artifact — so leave it disabled here ; the
            # tamer variants carry H-K fidelity, where the shallower bracket does not amplify it.
            hk_term = 0.0
            if args.hk_weight > 0.0:
                xr = (np.log2(np.maximum(S_refl @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
                Or = np.maximum(np.interp(np.clip(xr, 0.0, 1.0).ravel(), LUT_X, LUT_Y).reshape(xr.shape) @ Mo.T, 1e-10)
                hk_term = args.hk_weight * float(np.abs(nayatani_hk_excess(Or) - hk_in_refl).mean())
            return (hue_err
                    + skin_jacket                            # delta-E safety jacket on skin
                    + hk_term                                # H-K fidelity (0 unless --hk-weight > 0)
                    + 0.10 * mean_desat                      # prefer chroma when hue is tied (best-chroma-match)
                    + 1e4 * max(0.0, mean_desat - budget)    # HARD average-desaturation budget
                    + max(0.0, worst_hue - 16.0)             # HARD worst-case hue cap (deg)
                    + 0.10 * cond)                           # favour well-conditionned matrices

        p0 = [0.7, 0.7, 0.7,  # inset anchor
              0.0, 0.0, 0.0,  # inset rotation
              0.5, 0.5, 0.5,  # outset anchor
              0.0, 0.0, 0.0]  # outset rotation
        best = None
        
        # Bruteforce parametric sweep on the objective function coeffs
        # At the end we only want the min hue drift at requested desaturation
        # that doesn't fuck up Rec2020 gamut.
        insets = np.linspace(0.5, 0.9, 4)
        outsets = np.linspace(0.5, 0.9, 4)
        for di1 in insets:
            p0[0] = di1
            for di2 in insets:
                p0[1] = di2
                for di3 in insets:
                    p0[2] = di3
                    for do1 in outsets:
                        p0[6] = do1
                        for do2 in outsets:
                            p0[7] = do2
                            for do3 in outsets:
                                p0[8] = do3
                                
                                r = minimize(objective, p0, method="Nelder-Mead",
                                            bounds=[
                                                (0.50, 0.9), (0.50, 0.9), (0.50, 0.9),       # inset
                                                (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), # rotation anchor: limit to +/- 5.7°
                                                (0.4, 0.9), (0.4, 0.9), (0.4, 0.9),          # outset
                                                (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), # rotation anchor: limit to +/- 5.7°
                                            ],
                                            options={"xatol": 1e-5, "fatol": 1e-4, "maxiter": 8000, "maxfev": 8000})
                                if r.fun < BIG and (best is None or r.fun < best.fun):
                                    best = r
                                    p = best.x
                                    skin_cr, skin_dh, refl_cr, refl_dh, cond = evaluate(p)
                                    desats = np.concatenate([1.0 - skin_cr, 1.0 - refl_cr])
                                    mean_desat = float(desats.mean())
                                    gated = np.abs(refl_dh[refl_cr > 0.2])
                                    M, Mo = brk(p)
                                    inset = p[0]
                                    print("// achieved avg desaturation %.1f%% (budget %.1f%%) ; worst single color %.1f%% (bright-color bleach)"
                                        % (100 * mean_desat, 100 * budget, 100 * desats.max()))
                                    print("// priority set : skin |mean| %.1f deg [%+.1f..%+.1f] ; reflective mean %.1f max %.1f ; cond %.1f"
                                        % (np.mean(np.abs(skin_dh)), skin_dh.min(), skin_dh.max(),
                                            gated.mean(), gated.max(), cond))
                        
        if best is None:
            print("// no feasible bracket under the constraints (skin red veto / positivity / conditioning).")
            print("// the desaturation budget is not the binding constraint — check --max-desat is >= 0.")
            return
        
        p = best.x
        message = "--max-desat %f" % budget
        print_diagnostics(p, message)
        return

    if args.min_bleach:
        from scipy.optimize import minimize
        hk_weight = float(args.hk_weight)                   # optional Helmholtz-Kohlrausch term (0 = off)

        # UNIFIED colour-constancy set (single source of truth), same as --report / --max-desat /
        # --fit-extra-bleach. Previously conflated skin into the reflective bucket + divergent skin.
        S_skin, S_refl = skin_and_reflective_sets()

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
        hk_in_refl = nayatani_hk_excess(S_refl)             # input H-K excess (invariant) for the optional correction

        # PER-HUE APPARENT-BRIGHTNESS PULL (--ab-pull, 0 = off) : the min-delta-E objective has
        # NO luminance term, so high-H-K hues (red, magenta) DARKEN off the ladder — no-bleach
        # kinks reds dark and flips the green<->magenta apparent-brightness balance vs low-bleach,
        # a jarring no->low step. Pull each hue's apparent brightness (output luminance x
        # (1 + H-K excess)) toward the MIDPOINT of what the CURRENT shipped no-bleach and low-bleach
        # produce ("true red sits between the two"). FIXED reference (read from SHIPPED once,
        # before the re-fit), so it is not circular.
        ab_pull_w = float(args.ab_pull)
        ab_y_row = REC2020_TO_XYZ_D50[1]
        ab_nbins = 12
        ab_bin = (np.floor(np.remainder(H_refl, 2 * np.pi) / (2 * np.pi) * ab_nbins).astype(int)) % ab_nbins
        def per_hue_ab(M, Mo):
            x = (np.log2(np.maximum(S_refl @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
            O = np.maximum(np.interp(np.clip(x, 0.0, 1.0).ravel(), LUT_X, LUT_Y).reshape(x.shape) @ Mo.T, 1e-10)
            ab = (O @ ab_y_row) * (1.0 + nayatani_hk_excess(O))
            return np.array([ab[ab_bin == b].mean() if (ab_bin == b).any() else np.nan for b in range(ab_nbins)])
        
        # PRINCIPLED uniform apparent-brightness target : the scene's own apparent brightness
        # carried through the achromatic tone curve (scene_ab_target ~0.379) — one DERIVED value,
        # replacing the hand-tuned 0.380 it matches (and the earlier no/low midpoint). min-bleach
        # keeps full chroma/H-K, so it sits at the H-K-preserving top of the target band.
        ab_target = [scene_ab_target(S_refl)] * ab_nbins

        def measure(S, C_in, H_in, M, Mo):                  # same signature as --max-desat's
            x = (np.log2(np.maximum(S @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
            y = np.interp(np.clip(x, 0.0, 1.0).ravel(), LUT_X, LUT_Y).reshape(x.shape)
            O = np.maximum(y @ Mo.T, 1e-10)
            c_f, h_f = chroma_hue_batch(O)
            cr = np.minimum(c_f / np.maximum(C_in, 1e-9), 1.0)
            dh = np.remainder(h_f - H_in + np.pi, 2 * np.pi) - np.pi
            return cr, np.rad2deg(dh)

        BIG = 1e5

        def objective(p):
            M, Mo = brk(p)
            if M.min() < 0.0:
                return BIG
            cond = max(np.linalg.cond(M), np.linalg.cond(Mo))
            if cond > 6:                                     # conditioning : stability
                return BIG
            gv = gamut_violation(M, Mo)
            if gv > 0.0:                                     # Rec2020 gamut safety (hard) — no black
                return BIG + gv * 100.0
            skin_cr, skin_dh = measure(S_skin, C_skin, H_skin, M, Mo)
            refl_cr, refl_dh = measure(S_refl, C_refl, H_refl, M, Mo)
            if skin_dh.min() < -3.0:                         # loose skin red-ward safety veto (racial bias)
                return BIG
            # NO-BLEACH objective (delta-E redesign) : minimize the REFLECTIVE colours'
            # combined move delta_e_yrg AND their desaturation, DIRECTLY as the objective —
            # not as caps. This is the "keep colours vivid and faithful" end of the look axis.
            # delta_e_yrg already couples chroma+hue with chroma-weighted hue damping (a hue
            # error on a near-grey barely counts) ; the extra desat term doubles the chroma-
            # preservation pressure that defines "no bleach". Skin is diffuse and rides along
            # (recovered by the outset), guarded only by the loose red veto above. mean AND
            # max so no single colour is left badly off. Optional H-K fidelity term (--hk-weight).
            refl_dE = delta_e_yrg(refl_cr, refl_dh)
            refl_desat = 1.0 - refl_cr
            hk_term = 0.0
            if hk_weight > 0.0:
                xr = (np.log2(np.maximum(S_refl @ M.T, 1e-10) / GREY) - BLACK_EV) / (WHITE_EV - BLACK_EV)
                Or = np.maximum(np.interp(np.clip(xr, 0.0, 1.0).ravel(), LUT_X, LUT_Y).reshape(xr.shape) @ Mo.T, 1e-10)
                hk_term = hk_weight * float(np.abs(nayatani_hk_excess(Or) - hk_in_refl).mean())
            ab_pull = 0.0
            if ab_pull_w > 0.0:
                ab_pull = ab_pull_w * float(np.nansum((per_hue_ab(M, Mo) - ab_target) ** 2))
            return (refl_dE.mean() + refl_dE.max()
                    + refl_desat.mean() + refl_desat.max()
                    + hk_term + ab_pull)

        best = None
        guess = [
            0.2, 0.2, 0.2,
            -0.0045667, -0.0085405, +0.0070037, # from extra bleach rotations
            0.5, 0.5, 0.5, 
            -0.0007132, -0.0099789, +0.0057890, # from extra bleach rotations
        ]
        
        outsets = [0.35, 0.65]
        
        # Bruteforce parameters sweep for initial parameters because
        # the solution space is full of
        # local minima and we can't know in which one we fall until
        # we do a full scan.
        for di1 in np.linspace(0.35, 0.6, 4):
            guess[0] = di1
            for di2 in np.linspace(0.35, 0.6, 4):
                guess[1] = di2
                for di3 in np.linspace(0.35, 0.6, 4):
                    guess[2] = di3
                    for do1 in outsets:
                        guess[6] = do1
                        for do2 in outsets:
                            guess[7] = do2
                            for do3 in outsets:
                                guess[8] = do3
                        
                                print((di1, di2, di3), (do1, do2, do3))
                                
                                r = minimize(objective, guess, method="Nelder-Mead",
                                            bounds=[
                                                (0.33, 0.6), (0.33, 0.6), (0.33, 0.6), # inset
                                                (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), # rotation anchor: limit to +/- 11°
                                                (0.15, 0.8), (0.15, 0.8), (0.15, 0.8), # outset
                                                (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), # rotation anchor: limit to +/- 11°
                                            ],
                                            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 8000, "maxfev": 8000})
                                
                                if r.fun < BIG and (best is None or r.fun < best.fun):
                                    best = r
                                    p = best.x
                                    skin_cr, skin_dh = measure(S_skin, C_skin, H_skin, *brk(p))
                                    cr, _ = measure(S_all, C_all, H_all, *brk(p))
                                    refl_cr, refl_dh = measure(S_refl, C_refl, H_refl, *brk(p))
                                    g = np.abs(refl_dh[refl_cr > 0.2])
                                    M, Mo = brk(p)
                                    print("// avg desaturation %.2f%% ; worst single color %.0f%% ; refl hue mean %.1f max %.1f (recovered downstream)"
                                        % (100 * (1 - cr).mean(), 100 * (1 - cr).max(), g.mean(), g.max()))
                                    print("// skin |mean| %.1f deg [%+.1f..%+.1f] ; cond %.1f"
                                        % (np.mean(np.abs(skin_dh)), skin_dh.min(), skin_dh.max(), max(np.linalg.cond(M), np.linalg.cond(Mo))))
                            
        if best is None:
            print("// no feasible no-bleach bracket under the constraints.")
            return
        
        p = best.x
        message = "--min-bleach"
        print_diagnostics(p, message)
        return


if __name__ == "__main__":
    main()
