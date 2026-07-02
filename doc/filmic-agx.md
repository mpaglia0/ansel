# Filmic RGB "AgX-like" rendering — design notes

Status: implemented (CPU + OpenCL), 2026-07.
Code: `src/iop/filmicrgb.c` (`filmic_agx*`, sigmoid spline v4), `data/kernels/filmic.cl`
(sigmoid evaluator + `filmic_agx` device function dispatched by the chroma kernel),
`tools/derive_filmic_agx_primaries.py` (anchor fit),
`tools/derive_filmic_default_curve.py` (default curve appearance match).

This documents what was taken from Blender/darktable AgX, what was deliberately
changed and why, and which user parameters could be added later if real-world use
demonstrates the need. It exists so that future "why doesn't this match AgX?"
questions have an answer on record.

## Vocabulary for the road

The rest of this document uses these terms constantly. If you can read C but
not color-science papers, read this first — every section below assumes it.

**EV, dynamic range, mid-gray.** Exposure values: a difference of 1 EV means
twice or half the light. Mid-gray (~18% of a white diffuse surface) is the
reference; "+4 EV" means 16× brighter than mid-gray. The dynamic range is the
span between the darkest and brightest EV the curve maps to the display.

**Tone curve, toe, shoulder, latitude, pivot.** The S-shaped curve that
compresses the scene's dynamic range into what a display can show. The middle
straight-ish segment is the *latitude*; its slope is the *contrast*; the bend
easing into black is the *toe*, the bend easing into white the *shoulder*; the
*pivot* is where mid-gray sits.

**Hardness ("gamma").** A power function `y^p` applied after the curve. It is
required plumbing (display encoding), but it also darkens shadows a lot on its
own — a recurring plot point below.

**Primaries, working profile, gamut.** The three "purest" red, green and blue
a color space can express are its *primaries*; every color is a mix of them.
The *working profile* (linear Rec2020 here) is the color space the pipeline
computes in. The *gamut* is the set of colors a space or device can represent
— picture a triangle in a 2D "map of all colors" (chromaticity diagram), with
the primaries at the corners and white in the middle.

**Per-channel vs norm-based tone mapping.** Two ways to apply the curve.
*Per-channel*: run R, G and B through the curve independently — colors near
white converge and bleach (the "film look"), but hues also shift as a side
effect. *Norm-based*: compute one brightness number per pixel (a "norm"), run
only that through the curve, scale R, G, B by the result — colors stay exactly
as they were, but perceived lightness and local contrast can look off. Neither
is right; they are the two ends of a trade-off.

**Chroma, hue, Ych.** Chroma ≈ how far a color is from gray; hue ≈ which color
it is (the angle on the color wheel). *Ych* is a coordinate system (from
Richard Kirk / Filmlight) storing brightness (Y), chroma (c) and hue (h),
designed so that equal hue-angle steps look like equal hue changes to a human —
which plain RGB or HSV do **not** guarantee. When we "measure hue drift in
degrees", it is in this system, so the degrees mean something perceptually.

**Inset / outset matrices, the "bracket".** The core AgX trick. Before the
curve, a 3×3 matrix ("inset") nudges the working primaries toward white —
imagine gently shrinking the gamut triangle and slightly twisting it. This
makes the per-channel curve bleach saturated colors sooner and steers the hue
shifts. After the curve, the inverse matrix ("outset") expands everything
back. Because our outset is the *exact* inverse, the pair is invisible
wherever the curve is straight — its effects appear only where tones are
actually being compressed.

**Gamut mapping.** The final safety net: if a pixel's color cannot exist on
the display (out of gamut), reduce its chroma — at constant hue — until it
fits. Filmic has done this since v6, against both the working and the export
profile.

**Monotone, C1.** A *monotone* curve never goes back down — no wiggles, so
brighter input is always brighter output (a curve that isn't monotone creates
banding and hue inversions). *C1* means value **and** slope are continuous at
the segment joints — no kinks.

**Least-squares vs minimax; "railing".** Two ways to fit parameters.
Least-squares minimizes the *average* error; minimax minimizes the *worst
single* error. "Railing" is when an optimizer slams a parameter against its
allowed limit — a smell that the objective doesn't actually constrain that
parameter (several appearances below).

**CIECAM16 J, appearance match, surround, veiling flare.** CIECAM16 is a
standard model predicting how bright/light something *looks* given the viewing
conditions (J is its perceived-lightness output). An *appearance match* asks:
what display luminance looks the same as this scene luminance, to an observer
adapted to each? The *surround* (dim room vs daylight) changes the answer —
this is why images need more contrast on a screen in a dim room. *Veiling
flare* is stray light reflecting off the screen: it puts a floor under how
dark anything can look, which decides how much shadow detail is even visible.

**JND.** Just-noticeable difference: the smallest change a human can see. Used
as the yardstick for "is this transition too abrupt to be acceptable".

**Conditioning.** How much a matrix inversion amplifies small errors. A
well-conditioned pair (small number, ~1–8 here) is numerically safe; a
near-singular one turns rounding noise into visible artifacts.

## The limitations of Filmic and of its strategies to preserve chromaticity

Working in RGB makes lightness/brightness/luminance a by-product of the trichromatic signal,
which binds all perceptual color attributes (hue, colorfulness, lightness/brightness) in 
non-trivial ways.

Color-science research on methods to reduce dynamic range from any given scene to what the display can render
(tone-mapping), while preserving the perceptual color attributes unchanged, have produced so far only 
brittle and contextually-valid HDR color appearance models that are useless for any image-processing 
pipeline. In addition, these fail to account for Bezold-Brücke shift, Abney effect and Helmholtz-Kohlrausch effect.

Tone-mapping on individual RGB channels produces chroma/saturation and hue shifts as a by-product.
While _some_ saturation shifts and _some_ hue shifts are deemed pleasing and desirable
(highlights desaturation being seen as a "filmic" signature – more on that later), the tone-mapping core issue
is in controlling __how much__ of those shifts we let in for the amount of dynamic range compression
we impose. As such, there is no control at all. The scene-referred workflow makes this issue a bit more manageable, by dealing with color
_before_ tone-mapping, which reduces a lot of oversaturation issues we had with display-referred.
However, that doesn't solve the issue of hue shifts, that are still largely unpredictable.

While saturation is kind of reduceable to some arbitrary "intensity" or "vividness" of color, 
hues carry meaning and often give their name to colors, so they are semantic in nature. In addition, tonemapping RGB channels on portrait photography desaturates skin tones but also shifts
them toward yellow/green, which might be ok and even flattering for Caucasian (white-skinned) people, 
because it might soften redness and blemishes,
but it can be unflattering and uncalled for Arabian and Asian (olive-skinned) people. 
Beyond aesthetics, this is a racially-charged matter. Let's not reproduce [Kodak racial biases](https://www.youtube.com/watch?v=d16LNHIEJzs) when we can avoid it.

Then, what people associate with "film signature look" is often exaggerated. Movies shot on Technicolor,
like _[Lawrence of Arabia](https://filmcolors.org/galleries/lawrence-of-arabia-1962/)_, show deep blue skies at rather high luminance, that are properly impossible to get with
the forced-desaturation of independent RGB channels tone-mapping:

![](https://filmcolors.org/wp-content/uploads/2013/05/LoC_FGC-4398_LawrenceOfArabia_1962_TechnicolorV_R3_R4_JK_Img1319_anamorph.jpg)
![](https://filmcolors.org/wp-content/uploads/2013/05/AFA_3507-2-3_LawrenceOfArabia_1962_TechnicolorV_R3_BF_Img0862_anamorph.jpg)

So the _bleached highlights_ look is only one possible film look among others, and there is no reason to trap every photographer into this look, especially if it is going to be justified by a misinformed call to tradition. Moreover, there is no reason why all highlights should be bleached if they otherwise fit within the display colour gamut. Just because film couldn't render those colours is no justification for digital to forfeit them too. What matters most is the smoothness of local color gradients, and giving creative options to artists.

Handling tone-mapping on individual RGB channels is therefore the guaranty to drive any exigeant colorist insane.

For this reason, filmic historically resorted to tone-mapping a single metric for each pixel : the norm of the RGB vector. Across this mapping, the RGB ratios (as the ratio between each RGB channel and the pixel norm) were preserved unchanged, which gave a radiometrically-accurate light spectrum mapping. The rationale was: given the impossibility of ensuring the constancy of visual colour attributes, for lack of proper science, at least light physics provided a predictable ground-truth we could hold onto. No approximation required, no broken colour model in the mix.

This method had the benefit of preserving deep blue skies, as well as sunset colors, which, combined with gamut-mapping, ensured we made the best use of all available display gamut while preserving the scene light information. 

There were two problems though:

1. users complained that their sunsets registered a lot more red than what they remembered on scene and what their out-of-camera JPG registered: the hue shift toward yellow that came with individual-channels tone-mapping actually came closer to emulating Bezold-Brücke shift than the radiometrically-accurate constant RGB ratios,
2. RGB norms all failed to properly encode "lightness", so they flattened local contrast in sometimes weird and unpleasing ways.

After 6 iterations of design, using different mapping strategies for tonality, hue and chroma, in 2023, for filmic v7, I resorted to a weighted (user-parametric) mix between the max RGB norm tone-mapping and the channel-wise (independant) mapping at constant hue. Those were the 2 opposites of the spectrum, from full color saturation (albeit with weird local contrast) to full bleaching. But the whole thing still happens at constant hue, which still makes some users unhappy about how their sunsets look.

## Why Darktable AgX was a no-go in Ansel

Filmic got a lot of backlash among Darktable users when it came out (2018-2019), because it was "too complicated". Then in 2025 comes a port of Blender AgX that: 

1. has 33 user parameters, 
2. duplicates features from the color calibration (channelmixerrgb.c) through the "primaries" (which is nothing but a channel mixer) and gamut compression features,
3. duplicates features from the color balance (colorbalancergb.c) trough the ASC CDL (look slope/offset/power),
4. duplicates features from filmic (white/black relative exposure, piece-wise tone curve, autoset colorpicker, etc.).

AgX is not that different from Filmic that it requires its own, different module: this only confuses users. But then, the whole feature split is a nonsense too: it is a pipeline _within the pipeline_, with fine-grained color controls that don't belong there since the feature split approach of the scene-referred workflow was "divide and conquer" (chromaticity handled apart from lightness) to better manage it, and the typical modules duplicated in AgX support masking and blending with several instances for better control in difficult situations.

So, AgX is yet another example of overengineering by committee, only leading to overwhelming users and newcomers for no valid reason, except "it was new and cool so we implemented it".

### Magic constants and precision theater

Read `darktable/src/iop/agx.c` and trace where its numbers come from. The
"scene-referred default" primaries are:

```c
p->red_inset   = 0.29462451f;   p->red_rotation   =  0.03540329f;
p->green_inset = 0.25861925f;   p->green_rotation = -0.02108586f;
p->blue_inset  = 0.14641371f;   p->blue_rotation  = -0.06305724f;
```

Eight significant digits, cited to a LUT-generation script in a personal
GitHub repository (`EaryChow/AgX_LUT_Gen`) and a 1000+ post forum thread. There
is no objective function, no error metric, no derivation, and no regression
harness anywhere in that lineage: the digits are precise because a script
printed them, not because anything constrains them. That is precision theater
— nobody, including their authors, can re-derive these numbers or say what
they optimize. The outset values (`0.290776401758f`…) are equally unexplained,
and Blender ships `master_unrotation_ratio = 0` — the inset rotations are
*not* undone after the curve, which bakes a permanent skew and midtone
desaturation into a transform marketed as technical. The tone curve defaults
are documented in their own comments as *"about halfway between values required
to match sigmoid'd scene-referred defaults and those used by its 'smooth'
preset"* — defaults defined by splitting the difference between two other sets
of hand-tuned defaults. The "read exposure from metadata" feature computes
`black = (−8 + 0.5·exposure)`, `white = (4 + 0.8·exposure)`: two linear
regressions with unexplained coefficients. The curve's authoritative
specification is a Desmos calculator link.

None of this is disqualifying for a *creative preset* — hand-tuned looks are
legitimate. It is disqualifying for what AgX claims to be: a technically
grounded view transform. Two claims deserve specific debunking:

- **"Abney compensation."** The rotations are fixed angles in the CIE xy plane
  — a perceptually non-uniform space where the same angle means different
  perceived hue shifts at different hues — applied identically everywhere,
  although the per-channel drift field *reverses direction between toe and
  shoulder* (channels converge toward opposite ends of the curve; we measured
  it). We also measured what any constant matrix pair can do against that
  field: the worst-case drift floor is ~23–28° — more than a full perceptual
  hue category — no matter how the three angles are chosen. A static rotation
  is structurally incapable of "compensating Abney"; calling it that is
  marketing. Actual hue management needs a per-pixel mechanism in a perceptual
  hue metric, which is what Ansel's Ych recovery is (and darktable AgX's own
  fallback, tellingly, is a hue mix **in HSV** — a device-space angle with no
  perceptual meaning whatsoever).
- **The warm shift is a canned look, not color science.** The rotation signs
  are chosen to skew brights toward yellow. That is a creative decision — a
  perfectly defensible one — hard-coded as *the* rendering and presented as a
  property of the algorithm. In Ansel the drift direction is an explicit,
  stated input of the anchor fit (`--drift-target`, default 0 = neutral,
  maintainer decision); a warm variant is a one-flag refit away, not an
  identity baked into the module.

### Constants from first principles, or it does not ship

The point of the Ansel port is not that our constants are *different* — it is
that they are *derivable*. Replacing filmic's alleged suboptimality with AgX's
constants would have swapped one un-re-derivable point solution for another,
which is no progress at all. Instead, every shipped number has a committed
derivation with a stated objective:

- **Tone curve defaults** (`tools/derive_filmic_default_curve.py`): the curve
  is the luminance mapping between two *stated* viewing conditions, computed
  as a CIECAM16 lightness match (scene: ~1000 cd/m², average surround →
  display: 100 cd/m², dim surround, 0.1% veiling flare) with a local
  JND-visibility bound on the roll-off harshness. The method self-validates:
  it recovers Bartleson–Breneman surround compensation without being told, and
  confirms filmic's historical contrast 1.18 to three digits.
- **Inset/outset matrices** (`tools/derive_filmic_agx_primaries.py`): insets
  set at the measured knee of bleach effectiveness; rotations fitted by
  minimax hue drift in Kirk Yrg (a hue metric actually fitted to perceptual
  data) under explicit positivity and conditioning constraints, enforced over
  the whole user-reachable parameter ray.
- **Falsifiability in practice**: the derivation process caught its own errors
  — a mis-modeled contrast normalization (retracted with the numbers), a
  global-vs-local JND tolerance that crushed blacks, a flare assumption that
  hid the crush — each found by visual testing, diagnosed by measurement, and
  fixed in the objective, with the failure mode documented in this file.
  Hand-tuned constants cannot be debugged this way; there is nothing to check
  them against.

Where the AgX heritage values were right, the derivations *recovered* them —
the safe toe power converges to 1.5 (their default), the soft preset lands on
their gentle roll-off feel, filmic's contrast is confirmed — which is what
independent validation looks like. Where they weren't, we know exactly why and
in which viewing conditions. That is the difference between engineering and
democracy-by-forum-thread.

## What AgX is, reduced to its transferable core

Strip away the GUI and the branding, and one genuinely useful idea remains.

Run each RGB channel through the S-curve *independently* and something
interesting happens in the shoulder: for a bright saturated pixel, the highest
channel gets compressed hard while the lower ones catch up, so the three values
converge — and equal RGB values *are* white. Highlights therefore bleach toward
white automatically, and they bleach **in proportion to how much they are being
tonally compressed**. The same happens in mirror at the toe: channels converge
toward black. This coupling between tone compression and desaturation is what
people recognize as the "film look", and it is exactly what norm-based tone
mapping cannot do (a norm preserves the color ratios by construction, so a
compressed highlight keeps its full saturation and reads as a flat colored
patch).

The cost of per-channel is that the *hues* also shift while the channels
converge, in directions dictated by where the primaries happen to sit — not by
anything perceptual. AgX's real contribution is the **inset/outset bracket**
around the curve: pull the primaries toward white before the curve (which
controls *how fast* colors bleach and pre-steers *which way* hues drift), and
expand back after. That is 6 meaningful numbers — a chroma compression and a
rotation per primary — steering the whole behavior.

Everything else in AgX — the specific eight-decimal constants, the
partial-only outset, the HSV hue mix, the built-in color grading block — is
one particular hand-tuned point in that design space. The bracket is the
method; the rest is somebody's taste frozen into code.

## Architecture in Ansel

- **Color science `v8 (AgX-like)`** — a new value of
  `dt_iop_filmicrgb_colorscience_type_t`. Pixel path (`filmic_agx()`), in
  plain terms:

  1. sanitize the input (NaN → 0, clamp to ±1e6);
  2. lift any negative RGB values to zero while preserving the pixel's
     luminance (negative channels happen with out-of-gamut or clipped input,
     and the upcoming log encoding cannot represent them — this is Blender's
     "luminance compensation", generalized so it works in any working profile,
     not just Rec2020);
  3. record the pixel's original hue and chroma in Ych — this is the
     "what color was it supposed to be" reference used at the end. Measured
     *after* step 2, because before it, out-of-gamut pixels don't have a
     meaningful color to record;
  4. multiply by the inset matrix — enter the rendering space where the curve
     will behave nicely (see the constraints section below);
  5. run each channel independently through the ordinary filmic machinery:
     log encoding → spline → hardness;
  6. multiply by the outset matrix (the exact inverse of the inset) — back to
     the working profile;
  7. clamp the resulting chroma so it never exceeds the original (bleaching is
     allowed, spontaneous saturation boosts are not);
  8. **color recovery**: blend the result's hue and chroma back toward the
     step-3 reference by the user-controlled amount β. Hue is stored as a
     (cos, sin) pair, so the blend is a cheap vector interpolation with no
     trigonometry — for the small angles involved it is equivalent to walking
     the shortest way around the color wheel;
  9. hand the pixel to the existing filmic gamut mapping (working + export
     profiles), which fits the chosen hue into what the display can show.

- **Spline `v4 (2026)`** — a new value of
  `dt_iop_filmicrgb_spline_version_type_t`, usable with *any* color science,
  replacing the polynomial toe/shoulder segments with generalized sigmoids
  (`u/(1+u^p)^(1/p)` — an S-shaped function whose exponent `p` sets how long it
  hugs the straight line before rolling off). Why it is better than the
  polynomials: it is monotone for **any** setting (polynomials can wiggle when
  the constraints get tight — that is what the orange warnings on the filmic
  graph were about), it passes *exactly* through the black and white endpoints,
  it joins the latitude without a kink (C1), and when the requested geometry is
  impossible (the straight line would have to bend backwards to reach the
  endpoint) it degrades into a clean power curve instead of a broken fit.
  hard/soft/safe select fixed exponents per side. Implementation detail: the
  sigmoid's runtime parameters travel in the unused slots of the internal
  spline coefficient vectors, so the OpenCL kernels needed no new arguments.

- **Zero new params members.** `dt_iop_filmicrgb_params_t` layout and
  `DT_MODULE_INTROSPECTION` version are unchanged: two enum *values* were added
  to existing members, and the `saturation` slider is reinterpreted under v8.
  Databases, XMP, styles and presets remain byte-compatible in both directions.

## Design constraints of the inset/outset bracket

The inset matrix `M` has a friendly interpretation: its three columns are
simply the *new, displaced primaries* written as colors of the working space.
(Multiply `M` by pure red `(1,0,0)ᵀ` and you get the first column — the
slightly desaturated, slightly rotated red that the rendering space uses.)
Reading the matrix that way makes every constraint easy to state:

1. **Gray must stay gray** (hard constraint). The displaced primaries keep the
   same white point, which translates to `M·(1,1,1)ᵀ = (1,1,1)ᵀ` — each row of
   the matrix sums to 1. Consequence worth appreciating: since the outset is
   the exact inverse and the latitude part of the curve treats all channels
   identically, the whole bracket **cancels out perfectly for any pixel whose
   three channels sit in the latitude**. The color machinery only touches
   pixels that are actually being tone-compressed. This is provable, not
   hoped-for.
2. **No negative channels** (hard constraint). The log encoding in step 5
   cannot take a negative input. `M` keeps positive pixels positive exactly
   when every displaced primary stays *inside* the working gamut triangle.
   Geometrically: insetting pulls a primary toward the center (safe), rotating
   swings it sideways (risky near the triangle's edge) — so **the more you
   inset, the more rotation you can afford**. This coupling is why the user
   slider scales inset and rotation *together*: if the fitted anchor point is
   valid with margin, every point along the slider's range is too.
3. **The inverse must be numerically safe** (soft constraint). The outset is a
   matrix inverse, and inverting an almost-degenerate matrix amplifies every
   rounding error. Amplification grows roughly like 1/(1−inset): the code caps
   the inset at 0.9, and in practice it stays ≤ 0.5 at the extreme end of the
   user slider.
4. A neat consequence of 1+2: a matrix with non-negative entries whose rows
   sum to 1 means each output channel is a weighted average of the input
   channels — i.e. **the entire "primaries" apparatus of darktable AgX, its
   twelve sliders included, is mathematically nothing more than a conservative
   channel mixer**. That is why Ansel's color calibration module can emulate
   it (see the user documentation).

## Deviations from Blender / darktable AgX, with justification

| # | darktable/Blender AgX | Ansel v8 | Why |
|---|---|---|---|
| 1 | Separate `agx` module (33 params, 2 notebook pages, 2845 LoC) | Filmic color science value, adds 457 LoC | Same purpose as filmic (view transform); a second tone mapper splits users and duplicates the log/curve/picker scaffolding wholesale. |
| 2 | Base primaries selector (export/work/Rec2020/P3/AdobeRGB/sRGB) | Working profile, always | The base folds into the constants; "export profile" as a base made the render depend on an output setting while we *already* gamut-map to the export profile at the end. |
| 3 | Primaries displaced in CIE xy, rotations in xy radians | Displaced in Kirk/Filmlight Yrg chromaticity, rotations in Yrg angle | A degree of rotation then means the same perceptual hue shift for every primary (Yrg is fitted to even Munsell hue spacing); xy radians are perceptually uneven across the wheel. |
| 4 | Hand-tuned constants (Blender forum lineage), 12 user sliders + 2 master ratios + reverse toggle | 6 fixed anchors fitted offline by `tools/derive_filmic_agx_primaries.py`; single strength scalar at runtime | The sliders exist upstream so users can fit their own rendering space; the shipped configurations are what everyone uses. The residual empiricism moves into a stated, reproducible objective function instead of thirty floats in a params struct. Positivity coupling (constraint 2) makes one shared scalar the *safe* parameterization. |
| 5 | Outset partially restores purity, rotations not reversed by default (Blender), separately editable (darktable) | Outset = exact inverse of the inset, not editable | With the exact inverse, desaturation/drift are strictly a function of curve nonlinearity (transparent latitude — provable, see constraints). Blender's asymmetric outset bakes midtone desaturation into the transform; in Ansel that is a grading decision and Color Balance RGB owns it. |
| 6 | Hue restore mix in **HSV** (device-space hue), default 60% | Shortest-arc hue mix in **Kirk Ych**, inside the existing gamut-mapping stage | HSV hue is not a perceptual quantity. Filmic already computes Ych per pixel and its gamut mapper reduces chroma *at constant hue*, so deciding the hue first in the same metric is both cheaper and coherent. The recovery is exact per pixel, not a first-order matrix approximation ("Abney compensation" limited only by the Yrg Munsell fit — weakest in deep blue-violet). |
| 7 | "Look" block: lift/slope/power/saturation + brightness | Dropped | Strict subset of Color Balance RGB (lift↔shadows lift/offset, slope↔gain+contrast, brightness↔power, saturation↔saturation global), computed there in a proper perceptual space with per-range masks. |
| 8 | Own log encoding fixed to 0.18 mid-gray, contrast renormalized to the 16.5 EV Blender range, gamma-compensated slope, `auto_gamma` | Filmic's existing log encoding, grey/DR params, contrast, `auto_hardness` | Identical math where it overlaps; the renormalizations only existed to keep Blender's slider values meaningful. Filmic's parameterization is authoritative here; no values were imported. |
| 9 | Own pivot/toe/shoulder curve machinery (13 params) | Filmic spline geometry + sigmoid family as spline v4 | The curve DoF map 1:1 onto existing filmic params (pivot↔grey, linear ratios↔latitude+balance, curve gamma↔hardness). The genuinely new DoF — continuous toe/shoulder powers — is quantized onto hard/soft/safe, derived from the appearance match and perceptual-sharpness spacing (see the default-curve section): safe (1.5, 9.0), soft (1.1, 4.5), hard (2.1, 12.7). |
| 10 | Gamut compression hardcodes Rec2020 luminance coefficients | Generalized to the working profile's Y row | Works for any matrix working profile; identical result when working in Rec2020. |
| 11 | No output/export gamut mapping | Existing filmic Yrg gamut mapping runs after the bracket | dt AgX output can leave the export gamut freely; ours cannot. Strictly an improvement. |
| 12 | "Read exposure from metadata" heuristics (−8+0.5·exposure …) | Not ported | Magic constants; filmic's pickers and auto-tune cover the workflow. |
| 13 | `dynamic_range_scaling` | Not ported | Identical to filmic's existing `security_factor`. |
| 14 | Matrices rebuilt per `process()` call from params | Anchors are compile-time constants; the (cheap) bracket construction still runs in `process()` because the working profile is only known there | Same cost class (3×3 algebra per frame, not per pixel). |
| 15 | OpenCL kernel (`agx.cl`) | `filmic_agx` device function dispatched by the existing chroma kernel (4 extra args: inset/outset matrices, luma coefficients, recovery mix) | Reuses the shared Ych/gamut-mapping CL machinery instead of duplicating it in a standalone kernel. |

Known accepted cost of placing v8 in the color science enum (rather than a
`preserve_color` value): an **older Ansel build** reading a v8 edit falls through
every branch of its dispatch and produces an undefined output buffer for that
image. Params/DB/styles remain structurally compatible (no migration, no
introspection bump); only the *render* of v8-edits is undefined on old builds.
The `preserve_color` placement would have degraded to a defined v7 render
instead, but was rejected as semantically wrong (v8 replaces the whole color
handling, exactly like v7 does) — maintainer decision, 2026-07.

## The single runtime axis (saturation slider under v8)

Under v8, the old saturation slider becomes the one creative control of the
rendering: **film character on the left, color fidelity on the right**. This
section explains why it is designed that way — the answer came from a failed
first attempt.

The first version had no way to recover color, and visual testing immediately
flagged it: bright legitimate colors — sunsets, blue sky against orange clouds
— washed out irrecoverably. The obvious suspect was the inset matrices
(they exist to bleach things, after all), so we measured it: take a
half-saturated blue, scale it so its *brightness* sits at +2.5 EV like a bright
sky, run it through the pipeline. It keeps 18% of its chroma **with the
matrices — and also 18% with the matrices removed entirely.** The matrices were
innocent. The washing comes from the per-channel shoulder itself: for a blue
pixel to be *bright*, its blue channel must be enormous (blue contributes
almost nothing to luminance), which parks it deep in the shoulder where all
channels converge. Orange, whose channels contribute more evenly, keeps 58%
under the same test. So the insets kept their real job — taming clipped and
out-of-gamut colors, where they demonstrably help — and the cure for washed
sunsets became a *recovery* control instead.

The recovery has a pleasant mathematical property. Ych chroma is
luminance-normalized (it measures "how colorful relative to how bright"), so
taking the tone-mapped brightness and re-imposing the *original* chroma and hue
on it reconstructs exactly what norm-based tone mapping would have produced,
color-wise — while keeping the per-channel *tonality*. In other words, the
slider's right end gives you norm-like color without norm-based tone mapping's
flattened local contrast; its left end gives you the full film character. Any
overshoot near display white is caught by the gamut mapper, so peak display
brightness stays reachable.

The slider value `s ∈ [−100, +100]` (hard limits ±200, clamped) maps to:

- `β = (s + 100) / 200` — how much of the original color (chroma + hue) is
  restored: 0 = none (full character), 1 = all of it (full fidelity);
- `t = 1 + max(0, −s)/100` — on the left half only, the inset/rotation
  strength grows up to 2× the fitted anchors (stronger bleach and drift).

So `s = +100` → norm-like color fidelity; `s = 0` → **an equal mix, the
default** (deliberately the same convention as the v7 slider, whose zero also
means "equal mix"); `s = −100` → full film character.

## Anchor constants and what the fit taught us

The shipped constants are: **insets 0.25 for all three primaries**, and
**rotations {−2.50° red, +9.29° green, +24.06° blue}**, measured as hue angles
in Yrg. They come from `tools/derive_filmic_agx_primaries.py --minimax`
(2026-07, drift target 0 — the maintainer explicitly rejected a baked-in warm
bias), fitted against the appearance-matched default curve of the next section.

Why *minimax* (minimize the single worst hue error) rather than least-squares
(minimize the average)? Because at the character end of the slider there is no
safety net behind the matrices — the one number a viewer will notice is the
worst skew, not the average. The fit lands at a worst-case drift of 23.8° over
the visible exposure range. Two properties of that solution to know about:
the blue rotation is not free — it sits where the no-negative-channels
constraint stops it (constraint 2 above) — and the minimax landscape is a
plateau, so re-running the fit can return a *different but equally good* set
of angles; don't be alarmed by that. The least-squares alternative remains one
flag away in the script. The positivity margin is enforced over the entire
slider range (t ∈ [1, 2]) inside the fit itself.

One maintenance rule: the anchors are only optimal *for the curve they were
fitted against*. `CURVE_DEFAULTS` in the script must be kept in sync with the
C `$DEFAULT`s, and any change to the default curve requires re-running the
anchor fit.

Two findings from the fitting work reshaped the methodology and are worth
keeping on record:

1. **You cannot fit the insets — bleach depth saturates.** Because the outset
   is the exact inverse of the inset, the only desaturation that survives the
   round trip is what the curve achieves by making the channels *equal*;
   everything else gets expanded right back. Practical consequence: pushing
   the inset harder always bleaches a *little* more, forever, with no optimum
   — so every fitting objective based on bleach depth slammed the insets
   against whatever limit we set (tried three times with different
   counterweights, railed every time). The deep-bleach endpoint is also
   redundant: the gamut mapper already forces chroma to zero at display white.
   So the insets are **chosen, not fitted**: 0.25 sits at the measured knee of
   diminishing returns, with near-transparent midtones. With the final anchors
   the bracket's conditioning is 2.1 at the default and 6.4 at the far left of
   the slider (within the fit's safety bound of 8 — meaning the outset
   amplifies curve-output errors by at most ~6× at full character).
2. **Rotations fix the worst case, not the average.** Fitting the three
   rotations barely moves the *average* hue drift (~8.3° either way) but cuts
   the *worst case* — the classic "blue turns purple" per-channel failure — by
   about 17°. That is what the large blue rotation is for: it is a targeted
   counter to one specific, well-known defect, not a general beautifier.

The same script is the regression harness for any future retune. The sigmoid
power presets (hard/soft/safe) are derived by the default-curve harness — see
the appearance-matching section below.

## Investigated: fully unanchored hue (chroma-only recovery)

Question asked (2026-07): what if we trusted the matrices with hue entirely —
no hue restoration at all — and let the slider recover only chroma? In other
words: are the fitted rotations good enough to manage hue on their own?
Measured answer, sweeping saturated colors across the exposure range:

- **The worst hue deviation a viewer can actually see is 32.7°** (average ~7°).
  For scale: ~20–30° is roughly the difference between two named colors —
  azure drifting to blue, orange to red. Surprise: the worst case is *not* in
  the highlights but in colorful **shadows** (cyan-azure around −2.5 EV). The
  scarier-looking near-white numbers (48°) never reach the screen, because the
  gamut mapper strips almost all chroma next to display white anyway — a hue
  error on a nearly-gray pixel is invisible.
- **No set of rotations can fix this.** Refitting them specifically to
  minimize the worst case only lowers the floor to ~28.5°. The structural
  reason: hue drifts in *opposite directions* in the toe and in the shoulder
  (channels converge toward black at one end and toward white at the other),
  and a single constant matrix applies one correction to both. Three fixed
  angles against a direction-reversing error field simply cannot get the worst
  case below a full perceptual hue category.
- Worse, chroma-only recovery is self-defeating: restoring chroma onto a
  drifted hue makes the drift *more* visible (a washed-out wrong hue is barely
  noticeable; a saturated wrong hue is glaring).

Conclusion: shipping unanchored hue means accepting ~30° hue-category shifts in
colorful shadows as part of the look. The coupled recovery (β restores chroma
*and* hue together) offers the same character at the slider's left end while
shrinking the worst case linearly as the slider moves right — ~16° at the
default position.

**Verdict (2026-07, after visual testing)**: the coupled chroma+hue recovery was
restored as the production behavior, but the **minimax rotations were kept** —
the worst-case-bounded character is the better β = 0 baseline. The temporary
compile-time switch was deleted; the minimax fit lives on as the script's
`--minimax` mode and is now the production anchor derivation.

## Runtime adaptation to the actual curve (assessed, deferred)

The anchors are fitted against the *default* curve, and the fit showed the
coupling is real: what the bracket does depends on the shoulder length in EV
(dynamic range × latitude split), which the user changes per image. Assessment:

- **Full runtime re-fit: rejected.** The fitting objective needed hand-shaped
  counterweights to avoid railing (see above) — an unsupervised optimizer does
  not belong in `commit_params`. It would also make hue rendering shift while
  dragging contrast, and land in different local minima for different curve
  configurations, so the same slider value would mean different colors.
- **1-D deterministic adaptation along the fitted ray: worthwhile, deferred.**
  Because positivity/conditioning hold along the ray, a scalar `t_auto` can be
  derived from the actual spline at commit time — either closed-form
  (`t_auto ≈ reference_shoulder_EV / actual_shoulder_EV`, clamped to [0.5, 2])
  or by a ~20-evaluation bisection on a probe color through the actual curve —
  and composed with the user slider (`t = t_auto · t_user`). Deterministic,
  smooth in the params, microsecond-cheap. Since the gamut mapper owns the
  endpoint anyway, the residual mismatch is second-order; implement only if
  visual testing on short-DR edits shows late/abrupt bleaching.

## Future user parameters (only if demonstrated necessary)

Each of these requires bumping `DT_MODULE_INTROSPECTION` (new params members),
which breaks the back-and-forth database compatibility contract — hence the
bar is "users demonstrate a real limitation", not "would be nice".

1. **Independent purity strength and hue fidelity** (2 floats). The bipolar
   slider makes `t > 1` and `β > 0` mutually exclusive; someone may legitimately
   want strong bleaching *and* partial hue recovery (e.g. concert photography:
   heavy LED compression, but skin hues held). This is the most likely first
   request. Until then, the coupling is defensible: the two halves are one
   perceptual character↔fidelity axis.
2. **Continuous toe/shoulder powers** (2 floats). If the 3-step hard/soft/safe
   quantization of the sigmoid spline proves too coarse. Blender itself ships
   fixed powers, and the powers sit in "advanced" territory upstream, so demand
   is expected to be low.
3. **Hue-recovery weighting by compression** (1 float or a fixed design change).
   `β` is currently uniform; a compression-weighted mix would hold midtone hues
   fully while letting the shoulder drift. Adds a second perceptual decision to
   explain; revisit only with concrete examples where uniform β fails.
4. **Drift-direction variants** (no params cost). A "v8 neutral" color science
   value with zero-mean-drift anchors next to the warm default — enum values are
   free and this is the cheap escape valve before any of the above. Could also
   be shipped as fitted alternates behind presets if anchors ever become
   commit-time-selectable.
5. **Per-primary anchor overrides** (6 floats + validation). The darktable
   scenario — users fitting their own rendering space. Rejected for now: it
   re-imports the GUI surface this design exists to avoid, and runtime
   positivity/conditioning validation would need to clamp user input anyway.
   Users with this need are better served contributing an anchor fit.

## Deriving the default curve from appearance matching

The anchors are fitted against the default curve, which raised the obvious
question: where does the *default curve* come from? Historically: taste. It is
however a solvable problem — the tone curve is the luminance mapping between
two known viewing states, so `tools/derive_filmic_default_curve.py` computes it
as an appearance match: CIECAM16 lightness J of the scene state (diffuse white
~1000 cd/m², observer adapted, average surround, no flare) matched to the
display state (SDR ~100 cd/m², dim surround, veiling flare 0.5% of white),
then the curve family least-squares fitted onto the match with a content-mass
prior and a **JND-visibility term** (the rendering may not introduce
perceptual-lightness curvature d²J/dEV² sharper than the match itself contains
anywhere). The curve model replicates the C v3 geometry *exactly* — including
the DR/8 normalization and gamma compensation of the user contrast — so fitted
values are directly the module's parameters.

Results (2026-07), stable across scene 1000→5000 cd/m², display 100→200 cd/m²
and JND weights 1–3:

- The method self-validates: fitted end-to-end log-log midtone slope 1.16 ≈
  Bartleson–Breneman dim-surround compensation, never given to the harness.
- **The shipped contrast 1.18 is confirmed to three digits** (fit: 1.178–1.189).
  A first-pass model that skipped the DR/8+gamma normalization wrongly reported
  the shipped default as "flatter than neutral (0.889)" — retracted; the actual
  spline slope of the shipped default is 1.53, exactly what the appearance
  match asks for. The historically hand-tuned value was already the
  appearance-correct one; it now has a derivation.
- **Under spline v4, the latitude is a tension control and belongs in the fit.**
  An earlier revision of this doc called latitude/balance "editing-ergonomics
  conventions, not appearance-derivable" and fixed them at 33%/0 — retracted
  (maintainer correction, 2026-07). Small latitude hands the whole transition to
  the sigmoids (soft, global curve); large latitude forces them into short, hard
  turns: it participates in the contrast-transition strength exactly like the
  sigmoid powers. The CIECAM16 match has no linear segment (cone compression is
  smooth everywhere), and a fit started near zero latitude confirms it: the
  **latitude converges to ~0** (cost flat below 2%, ship 1%), and the **balance
  is unidentified there** (its translation is proportional to the latitude span;
  Δcost 0.007% over its full range — ship 0).
- **The JND tolerance must be local, not global** (first shadow-crush fix,
  2026-07). The first formulation bounded the rendering's curvature by the
  *global* maximum of the match's own curvature — which sits in the scene's
  deep-shadow compression (~22 J/EV²), so the toe could turn arbitrarily hard
  without penalty (toe-side curvature 5 vs tolerance 22: the term never bound).
  Measured crush: log-log shadow slope 0.16 at −7 EV. The tolerance is now
  per-exposure (match curvature at the same EV + 1 J/EV² margin).
- **The flare assumption owns the toe** (second shadow-crush fix, 2026-07 —
  still "a bit crushing" after the local-JND fix). Two structural facts: the
  hardness/gamma function alone imposes log-log slope 4.0 at −6.5 EV (a
  *straight* spline is far from a shadow no-op — the toe's actual job is to
  counteract the gamma), and the model considers shadow differences below the
  veiling-flare floor invisible, so at 0.5% office flare, crushing below −4 EV
  is free. The default fit now assumes **demanding viewing (0.1% flare, dim
  room)**: safe toe converges to **1.5 — the AgX heritage value recovered from
  first principles** — and the latitude rises to **~10%, genuinely identified**
  (with a soft toe the fit wants linear range to hold the midtone slope; the
  near-zero-latitude result was an artifact of the crushing toe). Balance is
  identified but flattens above 0 (gain past 0 is ~0.5% of its range, railing
  on a weak gradient) → ships at 0.
- **Safe = (1.5, 9.0), latitude 10%.** Shadow slope at −7 EV: 1.19 — open.
  **Hard/soft are derived, not hand-picked**: since the sigmoid powers are not
  exposed in the GUI, the presets and the latitude slider together must tile
  the range of transition strengths. "Soft" halves the peak perceptual shoulder
  sharpness of safe → (1.1, 4.5); "hard" is the usable ceiling (~1.4× safe) →
  (2.1, 12.7). Toe powers follow the shoulder's power ratios. Shadow slopes at
  −7 EV: soft 1.42, safe 1.19, hard 0.92 — crushing is nobody's default. The
  preset × latitude sweep tiles shoulder sharpness from ~11 to ~72 J/EV² with
  consistent ordering at every latitude (verified 1–60%).
- Consequently `spline_version` now **defaults to v4** for new edits (old edits
  keep their stored params), with default latitude 10% and balance 0.
- Composition order: the anchors are curve-relative, so
  `derive_filmic_agx_primaries.py` (which imports this harness's curve model,
  `CURVE_DEFAULTS` kept in sync with the C `$DEFAULT`s) was re-run against the
  new defaults — see above for the resulting anchors.

## Follow-ups

- Add the sweep plots (purity-vs-EV, drift-vs-EV per hue, preset × latitude
  sharpness grid) to CI or at least to the PR.
- Factory presets ("AgX-like base/punchy": v8 + black −10 EV / white +6.5 EV +
  matched contrast).
- Visual pass on the final defaults (latitude 10%, safe 1.5/9.0, refitted
  anchors) — numbers are derived, eyes are not optional.
- User documentation: see `ansel-doc/content/views/darkroom/modules/filmic.md`
  (v8 + spline v4 + darktable-AgX emulation guide).
