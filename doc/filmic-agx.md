# Filmic RGB "AgX" rendering — design notes

Status: implemented (CPU + OpenCL), 2026-07.
Code: `src/iop/filmicrgb.c` (`filmic_agx*`, perceptual sigmoid curve type), `data/kernels/filmic.cl`
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
shifts. After the curve, another matrix ("outset") expands everything back —
in Ansel deliberately a bit *more* than the inset compressed (see the
self-recovery finding in the slider section), so that valid diffuse colors get
their saturation back while extreme highlights stay bleached.

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
  stated input of the anchor fit (`--drift-target`, default 0 = neutral); 
  a warm variant is a one-flag refit away, not an
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

- **Color science `v8 (AgX)`** — three new values of
  `dt_iop_filmicrgb_colorscience_type_t` (`V6`/`V7`/`V8` = no / low / high
  bleach; `_filmic_is_agx()` dispatches them identically, they differ only by
  the bracket constants — see "Three variants on one axis"). Pixel path
  (`filmic_agx()`), in plain terms:

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
  6. multiply by the outset matrix — back to the working profile. The outset
     over-expands relative to the inset (per-primary, κ-equivalent {1.31, 1.78,
     1.01}, jointly fitted) so
     that valid diffuse colors — skin tones, product colors — recover their
     chroma on their own; the clamp of step 7 trims any excess to exactly the
     original, per pixel;
  7. clamp the resulting chroma so it never exceeds the original (bleaching is
     allowed, spontaneous saturation boosts are not);
  8. **color recovery**: blend the result's color back toward the step-3
     reference, with *separate* user-driven weights for hue and chroma (hue
     recovers first — fully restored from the slider center upward). The hue
     blend interpolates the *chromaticity vectors* — each hue weighted by its
     own chroma — not the hue angles: a bleached pixel whose output hue is
     numerically meaningless (near-zero chroma) then contributes no direction
     to the mix. See the two bug notes at the end of the slider section for
     what taught us both rules;
  9. hand the pixel to the existing filmic gamut mapping (working + export
     profiles), which fits the chosen hue into what the display can show.

- **Curve type `perceptual`** — a new value of
  `dt_iop_filmicrgb_curve_type_t` (the `shadows`/`highlights` control),
  selectable per side alongside the legacy hard/soft/safe, usable with *any*
  color science and the default for new edits. It replaces the polynomial
  toe/shoulder segment with a generalized sigmoid (`u/(1+u^p)^(1/p)` — an
  S-shaped function whose exponent `p` sets how long it hugs the straight line
  before rolling off). Why it is better than the polynomials: it is monotone
  for **any** setting (polynomials can wiggle when the constraints get tight —
  that is what the orange warnings on the filmic graph were about), it passes
  *exactly* through the black and white endpoints, it joins the latitude
  without a kink (C1), and when the requested geometry is impossible (the
  straight line would have to bend backwards to reach the endpoint) it degrades
  into a clean power curve instead of a broken fit. Unlike the legacy types it
  has **one fixed exponent per side** (not hard/soft/safe): the CIECAM16-J
  appearance match (see the default-curve section). Implementation detail: the
  sigmoid's runtime parameters travel in the unused slots of the internal
  spline coefficient vectors, so the OpenCL kernels needed no new arguments.

  *History note:* this first shipped (for ~24h) mislabelled as a *spline
  version* `v4 (2026)` — wrong, because a spline version is the node
  *geometry* (how latitude/balance place the toe/shoulder nodes) while the
  sigmoid is the segment *shape* between them; they are orthogonal. It was
  moved to a curve type; the `v4` spline-version enum value was removed and any
  history that stored it falls back to `v3` node geometry silently (no
  migration — maintainer decision given the 24h exposure).

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
   the matrix sums to 1. Historical note: an earlier design used an outset that
   was the *exact* inverse of the inset, on the strength of a provable theorem
   (the bracket cancels perfectly for any pixel whose three channels sit in the
   latitude). The theorem was true and became worthless: when the default
   latitude dropped to 10%, the sigmoid owns the whole range, the curve is
   nonlinear everywhere, and "transparent in the latitude" protects almost no
   pixel. The exact inverse then *mandatorily bleached every valid color* —
   including midtone skin tones, which is a racial-bias issue, not a style (see
   the limitations section). The production outset is therefore fitted for
   self-recovery instead (slider section).
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
| 4 | Hand-tuned constants (Blender forum lineage), 12 user sliders + 2 master ratios + reverse toggle | Fixed anchors (uniform inset scalar + per-primary outset chroma & rotation) fitted offline by `tools/derive_filmic_agx_primaries.py`; three preset variants (no/low/high bleach), no runtime bracket sliders | The sliders exist upstream so users can fit their own rendering space; the shipped configurations are what everyone uses. The residual empiricism moves into a stated, reproducible objective (a desaturation budget) instead of thirty floats in a params struct. A uniform inset is the *safe* parameterization — per-primary insets let the fit game the metric. |
| 5 | Outset partially restores purity (under-expansion: bakes midtone *desaturation* in), rotations not reversed by default (Blender), separately editable (darktable) | Outset over-expands per-primary (ratios > 1), jointly fitted with the inset, not editable | Fitted so valid diffuse colors (skin database + reflectances) recover their chroma through the bracket itself, with the output≤input chroma clamp trimming the excess per pixel. Blender's under-expansion bleaches midtones by construction — the exact opposite of what skin tones need; an exact inverse was tried and retired (see constraints). |
| 6 | Hue restore mix in **HSV** (device-space hue), default 60% | Shortest-arc hue mix in **Kirk Ych**, inside the existing gamut-mapping stage | HSV hue is not a perceptual quantity. Filmic already computes Ych per pixel and its gamut mapper reduces chroma *at constant hue*, so deciding the hue first in the same metric is both cheaper and coherent. The recovery is exact per pixel, not a first-order matrix approximation ("Abney compensation" limited only by the Yrg Munsell fit — weakest in deep blue-violet). |
| 7 | "Look" block: lift/slope/power/saturation + brightness | Dropped | Strict subset of Color Balance RGB (lift↔shadows lift/offset, slope↔gain+contrast, brightness↔power, saturation↔saturation global), computed there in a proper perceptual space with per-range masks. |
| 8 | Own log encoding fixed to 0.18 mid-gray, contrast renormalized to the 16.5 EV Blender range, gamma-compensated slope, `auto_gamma` | Filmic's existing log encoding, grey/DR params, contrast, `auto_hardness` | Identical math where it overlaps; the renormalizations only existed to keep Blender's slider values meaningful. Filmic's parameterization is authoritative here; no values were imported. |
| 9 | Own pivot/toe/shoulder curve machinery (13 params) | Filmic spline geometry + sigmoid as a new curve type ("perceptual") | The curve DoF map 1:1 onto existing filmic params (pivot↔grey, linear ratios↔latitude+balance, curve gamma↔hardness). The genuinely new DoF — the toe/shoulder roll-off shape — is not a user control: the toe is a fixed appearance-matched exponent (1.5) and the shoulder is a slope-matched power roll-off computed at runtime from the geometry (see the default-curve section). |
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
handling, exactly like v7 does).

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

The slider is **hue-only**: it recovers none,
some or all of the per-channel hue drift and never touches chroma. With
`a = s/100 ∈ [−1, +1]`:

- `β_hue = (a + 1) / 2` — a single linear ramp across the whole slider:
  **0 at −100%, 0.5 at the center, 1 at +100%**;
- **chroma is not user-controlled.** It is entirely the bracket's own
  κ-recovery + clamp: valid diffuse colors reach the clamp (original chroma,
  kept), strongly-compressed colors bleach along the bracket's smooth roll-off.

The three anchor points:
- `s = −100` → **pure AgX**: full per-channel hue drift (the film character),
  strongly-compressed colors bleached;
- `s = 0` (default) → **half the hue drift removed**, chroma unchanged;
- `s = +100` → **original hues restored exactly**, chroma still
  bracket-bleached (the transform contributes the per-channel luminance and the
  bleach roll-off, but no hue shift).

A short-lived earlier version also offered *chroma* recovery on the positive
half (`β_chroma = max(0, a)`, for speculars/jewelry). It was removed: mixing
any original chroma back on top of the bracket's output kinks highlight
gradients — the recovered value `β·c_orig` fights the bracket's smooth bleach
roll-off at the `min(c_orig, bracket)` clamp, and the visible seam in a bright
colored gradient was worse than the muted highlight it rescued. Chroma fidelity
of *valid* colors never needed the slider (it is in the bracket); recovering
the chroma of *strongly-compressed* colors is better left to a downstream
grading module, where it does not fight the tone map. (Also gone, from an even
earlier design: the `t` strength multiplier that scaled the anchors up on the
left half — it doubled the worst-case hue drift for no benefit once bleaching
stopped being the point of that half.)

The purple-band failure mode (restored chroma landing on a drifted hue) is now
closed *by construction*: there is no restored chroma at all, so a
partially-recovered hue can never carry extra saturation onto a wrong angle.

**The mix must be chroma-weighted** (bug found in visual testing, 2026-07: a
blue gradient swept from −10 to +15 EV turned magenta at the default slider
position, while both slider extremes were fine). The first implementation
blended the two *hue angles* as unit vectors and the two chromas separately.
But a pixel fully converged by the curve leaves it with (near-)zero chroma and
therefore no meaningful hue — exactly achromatic pixels even get a placeholder
hue pointing at the red axis from `pipe_RGB_to_Ych`. The unit-vector blend
weighted that placeholder as much as the pixel's real original hue (blue +
red placeholder → magenta), and the chroma half of the recovery then restored
half of the original chroma *onto the wrong hue* — a saturated magenta out of
nowhere. Endpoints never showed it: at β = 0 the garbage hue carries no chroma
(invisible), at β = 1 it is ignored entirely. The fix blends the chromaticity
vectors, i.e. each hue enters the mix scaled by its own chroma: meaningless
hues carry zero weight by construction, near-antipodal hues mix through gray
instead of through the perpendicular hue, and the aligned-hue behavior (and
both endpoints) are numerically unchanged. Validated in the harness: bright
blue at +6…+15 EV keeps hue 235° through the mix instead of drifting to 256°.

**And hue must recover before chroma** (second half of the same bug, 2026-07:
a fainter purple cast remained at the slider center after the vector-mix fix,
still absent at both extremes). The residual mechanism is not a degenerate
value but a property of *any* blend that couples hue and chroma recovery: the
visible error is hue-drift × chroma, and the two are anti-correlated along the
slider — at the character end the drift is maximal but the color is bleached
away (pale, reads as the look); at the fidelity end the chroma is full but the
drift is zero. Their product therefore peaks mid-slider. Measured on a true
sRGB blue: the per-channel skew reaches +29.6° toward violet around +4…+6 EV
while 34–84% of the chroma survives, and the coupled mix at the center left a
+8…+11° violet cast on 0.27–0.37 chroma — a visible purple band. The fix
split the two weights so they were never both partial-and-rising on a drifting
color. (This whole class of bug was later made moot — §"The single runtime
axis" — when chroma recovery was removed from the slider entirely: with no
restored chroma at all, a partially-recovered hue can never carry extra
saturation onto a wrong angle.) Diffuse colors are immune throughout regardless.

**And the bracket must self-recover diffuse chroma** (third finding, 2026-07:
at −100% *everything* washed out, midtones included — olive skin tones bleached
until they read as Caucasian, which is a racial-bias defect, not a style; see
the limitations section). Root cause: the exact-inverse outset only returns
what the curve did not converge, and with the 10%-latitude sigmoid the curve
converges *everywhere* — the transparency theorem that justified the exact
inverse had quietly become vacuous. The fix moves chroma fidelity off the
slider and into the bracket: the outset is the inverse of a **κ-scaled** inset
(κ = 1.51, same rotations), i.e. it over-expands, and the mandatory
output≤input chroma clamp trims the excess to exactly 1.0 per pixel. The
fitting questions this raised, with their measured answers
(`--fit-outset` in the anchor script):

- *Staged or joint fit?* Staged. Hue drift is **insensitive to κ** (29.7° max
  at every κ tested), so the outset stage is orthogonal to the rotations fit
  and a joint 12-parameter fit would add degeneracy for nothing — we have
  railed three objectives already; no need to build a fourth.
- *Start from the exact inverse and enlarge?* Yes: κ is searched upward from 1
  by bisection for the smallest value whose 5th-percentile pre-clamp chroma
  ratio on the **priority set** reaches 1. The priority set is the point of
  the fit: the human skin database from `src/common/color_vocabulary.c`
  (Lab D65, avg ± 1.5σ corners) plus in-gamut diffuse reflectances, each swept
  over the tonal placements a photographer may give them (±1.5 EV for skin,
  −2…+2.5 EV for objects).
- *Portability across curves and dynamic ranges?* This is where the
  clamp-based design pays: the fitted target is an *inequality* ("reach the
  clamp"), and the clamp is the tone-adaptive stage, applied per pixel. Any
  over-recovery in easier conditions is trimmed to exactly 1.0. Measured with
  one fixed κ: post-clamp p5 ≥ 0.97 and median = 1.000 across studio 6.5 EV,
  studio 8 EV, default 12 EV and HDR 16 EV curves. No runtime adaptation
  needed, and no overfitting surface — the fit cannot "overshoot usefully".
- *Does the bleaching job survive?* Yes: gamut-boundary colors at the white
  endpoint still bleach to 0.59 chroma ratio (vs 0.52 with the exact inverse),
  before the gamut mapper's own near-white crush — speculars and clipped
  lights keep the film behavior.
- *The recovery re-exposes hue drift, and the rotations themselves are
  re-fitted* (fourth + fifth finding, 2026-07: visual testing showed sRGB blue
  skewing to purple and green to cyan at −100%, "not there before"). The drift
  *was* there before — the ~30° worst-case from the unanchored-hue study — but
  the exact inverse bleached those pixels invisible; κ keeps them saturated
  (sRGB blue +17.6° at 94% chroma).
  - *First attempt — outset-only rotation deltas:* gave the outset its own
    rotation set, fitted by chroma-weighted minimax over the sRGB gamut
    boundary (a first fit over the Rec2020 boundary refused to move — its
    extremes are clipped/emissive content that dominated the max). Note the
    outset rotations are *not* bound by the log-positivity budget that pins
    the inset: they act after the curve. Peak sRGB blue skew 17.6° → 13.7°.
  - *The real cause — the blue INSET rotation:* an isolation probe (drift with
    each rotation stage zeroed in turn) showed the +24° blue inset rotation —
    fitted by `--minimax` to counter the drift of *bleached* pixels — was
    itself producing +13.7° of the +13.7° once those pixels are recovered
    (turn it off: +1.5°). The minimax rotations were *correct for a regime we
    no longer run*. Re-fitting for the recovered regime drops the blue inset
    to +1° and cuts the visible blue skew to **−0.3° at 96% chroma** — better
    than any outset delta could do, because it removes the drift at its source
    rather than counter-rotating it back.
  - *The whack-a-mole, made explicit:* the bracket is one 3×3 matrix inverted,
    so a primary's rotation couples into every hue. The +24° blue was *also*
    suppressing the sunset's yellow drift as a side effect; removing it returns
    sunsets to full yellow-ward (which is the requested direction,
    but strong — up to +32° at the saturated extreme). Zeroing rotations
    entirely wrecks green (−25°) and over-yellows sunsets (+37°). So the
    production fit **keeps the diffuse-validated red/green inset rotations**
    (they protect skin, sunset and foliage) and re-fits **only blue inset +
    the three outset deltas**, under the priority ordering
    (skin/reflective absolute; blue-LED purple the worst real complaint;
    sunset must drift yellow not red). Result: inset {−2.50°, +9.29°, +1.09°},
    outset {−2.04°, +13.04°, +6.21°}; sRGB blue −0.3°, skin red-ward capped at
    −1.5° (was −3.0°), sunsets all yellow-ward, diffuse recovery intact
    (post-clamp p5 = 1.000 across 6.5–16 EV). Reproducible via `--fit-outset`,
    which supersedes `--minimax`.
  - *Structural residual:* the near-white drift is EV-dependent and no static
    matrix cancels a direction that changes along the tone axis. The
    real-pipeline gamut mapper crushes chroma in exactly that zone, so the
    model numbers overstate what reaches the screen. The one region the slider
    cannot rescue is over-range blue LEDs (a blown blue spot has no
    well-defined hue); there, bleaching is the honest treatment. Remaining
    lever if a case still objects: a reliability weight on the recovered
    chromaticity *direction* (near-white residuals are numerically unreliable
    before the κ amplification — same lesson as the chroma-weighted hue mix),
    a governor-side fix.

Is this "an Ych tone-mapping in disguise"? In the diffuse range the bracket is
now deliberately *near-neutral* — which is the requirement, not a
contradiction: grading happens before the tone mapper, so the tone mapper
should touch valid colors as little as possible. What the RGB bracket still
does that Ych operations cannot: it *generates* the roll-off into bleach as a
smooth tristimulus process (dye bleaching is a tristimulus phenomenon, and
perceptual-space chroma manipulation is exactly what produced the
yellow-leaves/blue-sky fringe artifacts documented for the color equalizer —
light mixes in RGB, not in Ych). The Ych machinery downstream is the
*governor*: it measures, clamps and anchors, but never generates. Generator in
RGB, governor in Ych — that division is the design.

## Anchor constants and what the fit taught us

### Three variants on one axis (2026-07)

v8 AgX ships as **three colorscience variants** — *no bleach* (default), *low
bleach*, *high bleach*. They share the entire pixel path and differ **only** by
the bracket constants. All three are points on a single axis: **inset strength**,
which trades bright-color desaturation ("bleach") for in-bracket hue accuracy.
More inset → the per-channel curve acts on more-compressed channels (less hue
drift) but the round trip loses more chroma. The inset is **uniform** (a
per-primary inset is the lever the optimizer abuses — it rails green to 0.7 and
wrecks an unseen blue; the good fits use a uniform inset with per-primary action
in the outset anyway), so each variant is one scalar plus the per-primary outset:

**Measured behaviour** (`tools/derive_filmic_agx_primaries.py --report`), as
{avg ; max} over the skin-tone database and the diffuse reflective/memory-color
hue circle, each swept over tonal placements. Desaturation is chroma loss
(1 − output/input chroma, post clamp); hue shift is the bracket's raw drift
*before* the Ych recovery slider mitigates it:

| variant | skin desat avg/max | skin hue drift avg/max | reflective desat avg/max | reflective hue drift avg/max |
|---|---|---|---|---|
| **no bleach** | **0.0% / 0.0%** | 2.9° / 9.0° | 3.7% / 63.4% | 3.5° / 19.7° |
| **low bleach** | 0.0% / 0.0% | 2.0° / 6.3° | 7.1% / 71.4% | 2.6° / 12.9° |
| **medium bleach** | 0.4% / 9.2% | 1.4° / 4.4° | 10.6% / 75.1% | 2.1° / 10.0° |
| **high bleach** | 2.6% / 19.0% | **0.8° / 2.1°** | 15.4% / 78.4% | 1.6° / 6.3° |

The single axis is visible across every column: from no→high bleach, hue drift
falls (skin 2.9°→0.8°) and desaturation rises (skin avg 0.0%→2.6%, reflective avg
3.7%→15.4%). The reflective *max* desaturation is high in all three (63–78%)
because it is the intended bleach of near-clipping bright colors — that is
per-channel tone mapping working, not a defect, and it is the same order in every
variant.

- **no bleach — protect the irrecoverable.** Hue drift *is* recoverable downstream
  (the Ych governor); saturation is *not* — nothing puts back chroma the bracket
  bled. So the default protects chroma (skin desaturation literally 0.0%, reflective
  avg 2.1%) and accepts the largest hue drift, which the "color preservation" slider
  restores. *Use case:* the safe default — colourful subjects, product/skin work
  where washing out is the worst failure; pair with the slider toward +100% if the
  hue character needs pulling back. *Mitigation of its cost (hue):* the slider, exact
  at +100%.
- **high bleach — protect hue in-bracket.** Spends chroma (reflective avg 15.4%) to
  drive the raw hue drift to near nothing (skin 0.8°, reflective max 6°), so the look
  is hue-correct even with the slider at −100% (pure film character). *Use case:* when
  you want the AgX highlight wash-out as an aesthetic and demand accurate hues without
  touching the slider. *Mitigation of its cost (chroma):* none downstream — this is a
  deliberate, irreversible trade, which is why it is not the default.
- **low bleach — the perceptual midpoint.** Not a desaturation budget (that put it
  too close to high-bleach — the visible gap no→low exceeded low→high) but the bracket
  that best reproduces the **average of the no-bleach and high-bleach processed
  outputs** (`--fit-low-bleach`). It bisects the hue drift evenly (skin 2.9°→**2.0°**→
  0.8°, reflective 3.5°→**2.6°**→1.6°) with intermediate reflective desaturation
  (7.1%), and — because the least-squares target lands where the output-chroma clamp
  holds skin at full chroma — it keeps skin saturation (0% desat), *not* inheriting
  high-bleach's skin whitening. *Use case:* the general-purpose middle, a even step
  between the two extremes.

The outset always **over-expands** (ratios > 1) so priority colours reach the
output-chroma ≤ input-chroma clamp and are trimmed to exactly 1.0 per pixel —
portable across dynamic ranges. The reproducible constants (mirror
`filmic_agx_prepare_bracket`, and `SHIPPED_VARIANTS` in the script):

| variant | inset | outset chroma (R/G/B) | cond | fit |
|---|---|---|---|---|
| no bleach | 0.336 | 0.349 / 0.737 / 0.397 | 4.7 | `--min-bleach` |
| low bleach | 0.488 | 0.479 / 0.746 / 0.370 | 4.7 | `--fit-low-bleach` |
| high bleach | 0.748 | 0.725 / 0.829 / 0.550 | 6.5 | `--max-desat 0.05` |

**Rec2020 gamut safety (a hard constraint on the no-bleach fit).** The working
space *is* linear Rec2020, so its primaries are the most saturated colors that can
occur — the worst case. Minimum desaturation wants a strongly over-expanding outset
(an early no-bleach fit ran inset 0.20 with outset ratios ~3), but that expansion
pushes the **Rec2020 blue primary to negative luminance** across roughly −10…+1 EV:
the outset's R and G rows have large negative off-diagonals, blue's still-saturated
channels (after the weak inset) hit them, and blue's tiny luminance weight (0.046)
means R,G ≈ −0.3 flips the luminance negative → **the pixel renders black**. Low and
high bleach are immune because their strong insets (0.63, 0.75) pre-equalize the
channels, so their (even more negative) outsets stay luminance-positive. The
`--min-bleach` fit therefore constrains the outset to retain ≥ 25% of the pre-outset
luminance for every Rec2020 primary/secondary across the tonal range; the shipped
no-bleach (inset 0.336, milder outset ratios 1.0/2.2/1.2) keeps every boundary color
luminance-positive (verified −12…+8 EV) at a small cost in reflective desaturation
(2→3.7% avg). This was a real shipped bug — the fit's chroma metric clamped negative
RGB to a tiny positive, so it never saw the black.

**The `--max-desat FRAC` tool** is the recommended way to set a bracket: state the
maximum average chroma loss over the priority set you'll accept, and the solver
returns the best-hue bracket at that budget. The budget binds (achieved ≈ FRAC)
and hue improves monotonically until it plateaus (~5%). The metric evaluates the
whole priority set every step (vectorized via a curve LUT) so no color can hide,
hard-caps the worst single hue at 16°, and vetoes skin red-ward drift — the
guards that stop the optimizer gaming an average by wrecking one color.

**The no-bleach surprise.** A hard 0% inset is the **worst** case for saturation,
not the best: with no inset the outset cannot over-expand without wrecking
conditioning, so bright colors bleach from the raw per-channel curve with nothing
to recover them (**7.7% avg desat** at inset 0, vs < 1% at a moderate inset). The
over-expanding outset is what *un-bleaches* — it pulls chroma back up to the clamp
— and it only becomes well-conditioned once the inset is ≥ ~0.2. There is a sharp
phase transition there: below it the fit is stuck at exact-inverse (~8% desat),
above it desat collapses. So minimum bleach sits at a **moderate** inset, never
near zero — the shipped no-bleach lands at 0.336 once Rec2020 gamut safety (below)
rules out the still-lower-inset / stronger-outset optima.

**The DoF limit (why the inset is uniform and blue is conceded).** A full
12-parameter *global* fit (differential evolution) over the entire sRGB hue circle
× exposure proved structural: with skin protected and bleaching forbidden, the
linear bracket tops out at ~8° mean full-circle drift — no better than a uniform
inset — and cannot hold saturated sRGB blue at +5–6 EV (above the white point)
without driving skin red or bleaching. A pair of 3×3 matrices around a per-channel
curve has too few DoF to hold every hue at every exposure; fixing one region
borrows from another, and a per-primary inset just gives the optimizer more rope
to hang an unseen color. So the inset is uniform, and blue-at-high-EV is left to
the per-pixel Ych hue recovery (exact at full strength).

**Methodology evolution.** The fit passed through `--minimax` (worst-case hue for
the bleached exact-inverse regime; parked blue at +24° against the positivity
wall), `--fit-outset` (added the κ recovery, which re-exposed the drift bleaching
hid and revealed the +24° blue now *caused* purple), `--fit-priority` (joint
12-param fit, uniform-target, capped inset 0.35), and finally the desaturation-
budget frontier (`--max-desat`) that spans the three shipped variants. All earlier
modes are kept in the script as the historical record.

One maintenance rule: the anchors are only optimal *for the curve they were
fitted against*. `CURVE_DEFAULTS` in the script must be kept in sync with the
C `$DEFAULT`s, and any change to the default curve requires re-running the three
fits — `--min-bleach` (no bleach) and `--max-desat 0.05` (high bleach) first, then
`--fit-low-bleach` (low bleach is the midpoint of the other two, so it must be
refit *after* them).

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
toe/shoulder exponents are derived by the default-curve harness — see the
appearance-matching section below.

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

The anchors were fitted against the *default* curve, but users reshape the
curve on every image (dynamic range, latitude, contrast). How much does that
mismatch matter, and should the matrices adapt to the user's actual curve?
Two options were assessed:

- **Re-running the full fit at runtime: rejected.** The fitting sessions above
  showed the objective only behaves when hand-shaped counterweights keep it
  from railing — an optimizer that needs supervision has no business running
  unattended inside `commit_params`. It would also make color rendering
  wander while the user drags the contrast slider, and settle into different
  local minima for different curve setups — the same slider position would
  mean different colors on different images. Unacceptable for predictability.
- **A single self-adjusting strength factor: worthwhile, deferred.** The safe
  middle ground: keep the fitted matrix *shape* and only auto-scale its
  *strength* to the actual curve. Because validity (constraints 2 and 3) holds
  along the whole strength axis, a scalar can be computed cheaply when
  parameters change — either from a simple ratio (fitted shoulder length vs.
  actual shoulder length, in EV) or by probing ~20 curve evaluations.
  Deterministic, smooth, no optimizer. The κ-recovery redesign lowered the
  stakes further: the diffuse range is clamp-protected under *any* curve
  (measured 6.5–16 EV), so a curve mismatch can only mis-time the bleaching of
  extremes, whose endpoint the gamut mapper owns anyway. Implement only if
  edits with small dynamic ranges visibly bleach too late or too abruptly.

## Future user parameters (only if demonstrated necessary)

Each of these requires bumping `DT_MODULE_INTROSPECTION` (new params members),
which breaks the back-and-forth database compatibility contract — hence the
bar is "users demonstrate a real limitation", not "would be nice".

1. **Chroma recovery of strongly-compressed colors** (1 float). Removed from
   the slider (it kinked highlight gradients, see the slider section), so
   there is currently no way to un-bleach a specular or a clipped colored
   light *inside* filmic — e.g. jewelry work where a highlight's colored glint
   must hold. The intended answer is a downstream grading module (color balance
   / saturation), where re-saturating does not fight the tone map's roll-off;
   only add a param here if that workflow proves inadequate. Valid diffuse
   colors need nothing (their chroma is in the bracket).
2. **User-exposed sigmoid toe/shoulder exponents** (2 floats). The perceptual
   sigmoid ships one fixed pair (1.5, 7.8) with no user control; a legacy
   polynomial/rational type is the only escape to a different roll-off. If that
   proves too coarse, expose the two exponents. Blender itself ships fixed
   powers and keeps them in "advanced" territory, so demand is expected low.
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
however a solvable problem, because a tone curve is secretly an answer to a
physical question: *"this patch of the scene had luminance X — what luminance
should the display emit so that it __looks__ the same to the viewer?"* The
scene is bright and the viewer is adapted to daylight; the display is ~10×
dimmer, sits in a dim room, and reflects some stray light. Those are known,
stateable conditions, and CIECAM16 exists precisely to predict perceived
lightness under given conditions. So `tools/derive_filmic_default_curve.py`
computes, for every exposure level, the display luminance whose *perceived*
lightness matches the scene's (scene: diffuse white ~1000 cd/m², average
surround → display: 100 cd/m², dim surround, 0.1% veiling flare — see the
flare finding below), then finds the filmic curve parameters that best
reproduce that mapping. Two weights steer the fit: exposures where photographs
actually hold detail count more, and a **JND term** forbids the curve from
introducing perceptual transitions sharper than the ideal mapping itself has
anywhere — that is what keeps the roll-offs from turning into visible cliffs.
The Python curve model replicates the C geometry *exactly*, including the
internal renormalizations of the contrast slider, so the fitted values are
directly the module's user parameters.

A recurring measurement below is the **shadow slope**: how many stops the
display output changes per stop of scene exposure, at a given point of the
range. Slope 1 means shadow gradations are fully preserved; slope near 0 means
whole stops of scene detail collapse into nothing on screen — that is what
"crushed blacks" measures.

Results (2026-07), stable across scene 1000→5000 cd/m², display 100→200 cd/m²
and JND weights 1–3:

- **The method validates itself.** The fitted curve's mid-tone slope comes out
  at ≈1.16 — textbooks (Bartleson–Breneman, 1967) say images viewed in a dim
  surround need their contrast raised by ≈1.2× to look right, and the harness
  rediscovered that number without ever being told it.
- **The shipped contrast 1.18 is confirmed to three digits** (fit:
  1.178–1.189). Embarrassing detour worth recording: a first-pass model
  ignored that the module internally rescales the contrast slider, wrongly
  concluded the shipped default was "flatter than neutral", and was retracted
  once the model replicated the C geometry exactly. The historically
  hand-tuned value was already the appearance-correct one; it now has a
  derivation instead of a reputation.
- **Under the perceptual sigmoid, the latitude is a tension control and
  belongs in the fit.** An earlier revision of this doc declared latitude and balance
  "editing-ergonomics conventions, not derivable" and pinned them at 33%/0. 
  With sigmoid segments,
  a small latitude hands the whole curve to the smooth roll-offs (soft,
  progressive transitions) while a large one forces short, hard turns — so the
  latitude shapes the transition strength exactly like the sigmoid exponents
  do, and must be fitted with them. The perceptual ideal mapping has no
  straight segment anywhere (human brightness perception is smoothly
  compressive), and the fit agrees: started near zero, the latitude stays
  there, and the balance stops mattering at all at small latitudes (its effect
  is proportional to the latitude's width; over its whole range it changes the
  fit error by 0.007% — so it ships at its neutral 0).
- **The JND tolerance must be local, not global** (first shadow-crush fix,
  2026-07). The "no transitions sharper than the ideal mapping" rule was first
  implemented against the *sharpest transition anywhere* in the ideal mapping
  — which happens to sit in the deep shadows and is quite sharp. Measured
  against that lax ceiling, the toe never triggered the penalty at all and the
  optimizer crushed the blacks for free: shadow slope 0.16 at −7 EV, i.e. a
  full stop of scene detail squeezed into a sliver of output. The rule now
  compares like with like — the curve's sharpness *at each exposure* against
  the ideal mapping's sharpness *at that same exposure* — and bites on both
  ends of the curve.
- **The flare assumption owns the toe** (second shadow-crush fix, 2026-07 —
  the blacks were still "a bit crushed" after the first fix). Two structural
  facts surfaced. First, the hardness power function alone darkens shadows
  enormously (slope 4.0 at −6.5 EV for a *straight* spline): the toe's actual
  job is to *fight the hardness*, not to add compression of its own. Second,
  the model treats any shadow difference below the screen's stray-light floor
  as invisible — so with office-grade flare (0.5%), crushing everything below
  −4 EV was literally free, and the optimizer took the deal. The viewer
  judging the result worked in better conditions than the model assumed. The
  default fit now assumes demanding viewing (0.1% flare, dim room), and two
  things fall out: the safe toe exponent converges to **1.5 — exactly the AgX
  heritage value, recovered from first principles** — and the latitude rises
  to **~10%, now genuinely pinned by the data** (with a soft toe the fit wants
  some straight segment to hold the mid-tone slope; the earlier "latitude ~0"
  answer was an artifact of the crushing toe).
- **Perceptual toe = fixed 1.5, shoulder = slope-matched at runtime.** The two
  ends are grounded on *different physical limits*, so they are shaped
  differently — this asymmetry is the whole point.
  - *Toe: fixed exponent 1.5.* Shadows have a perceptual floor — veiling flare
    hides detail below ~0.1% output — so the toe is a JND-bounded appearance
    fit, tuned to keep *visible* shadow gradients open (slope at −7 EV: 1.19).
  - *Shoulder: no fixed exponent — slope-matched power roll-off.* This
    corrects a real methodological error. The lightness match is **indifferent**
    to the shoulder power (its RMS *monotonically* falls as the exponent rises,
    9.96 → 8.87 across p = 1.5 → 16, with no interior optimum) because blown
    highlights are physically *unmatchable* — the display maxes at white, so a
    +4 EV scene highlight can never reach its scene lightness, and the match
    "gives up" on the top stop and just rewards holding mid-tone contrast
    longer (a hard shoulder). Fitting it therefore only **rails** against
    whatever bound you set; the shipped 7.8/9.0 sat on the JND ceiling — i.e.
    "as harsh as barely permissible," which is exactly why it read as harsh
    (top-stop local contrast 0.022 out/EV vs ~0.06 gentle). The grounded answer
    (maintainer's, 2026-07) is to **match the latitude slope**: the shoulder is
    a power curve `y = white − c·(1−x)^q` with `q = slope·dx/dy`, which leaves
    the latitude node at *exactly* the latitude slope and glides to white. No
    magic number; `q` is pure geometry and **adapts to the dynamic range** —
    q ≈ 1.0 for a 6.5 EV studio curve (barely any roll-off, nothing to
    compress), 1.6 for the 12 EV default, 2.5 for a 14 EV ETTR curve (more
    compression). Highlights hold detail: out at +3 EV is 85.6% (vs 96.2% at
    7.8), i.e. clearly below white with tonal separation intact.
  - *Coupling discovered in the process:* in v8 the shoulder is overloaded —
    it is both the tone roll-off **and** the per-channel bleaching driver, so a
    harsh shoulder was silently masking blue hue-drift by bleaching it away. A
    *fixed* gentle shoulder (1.5) re-exposed the drift (sRGB blue +14.8° at
    +4 EV) and broke HDR chroma recovery (needing κ = 2.1, still p5 0.865).
    The slope-matched shoulder resolves it almost for free: it **hardens
    exactly at high dynamic range** (q = 2.4 at HDR 16 EV), restoring the
    bleaching there — blue drops back to +2.1°, κ = 2.0, default-DR portability
    back to 1.000 (HDR 0.894).
- `shadows`/`highlights` now **default to "perceptual"** for new edits (old
  edits keep their stored curve types); `spline_version` stays at **v3** (its
  job is only node geometry). The short-lived `v4 (2026)` spline version — which
  conflated the sigmoid segment with the node geometry — was removed; histories
  that stored it fall back to v3 geometry silently.
- Order of operations for maintainers: the anchors are curve-relative, so
  `derive_filmic_agx_primaries.py` (which imports this harness's curve model;
  keep `CURVE_DEFAULTS` in sync with the C `$DEFAULT`s) was re-run against
  these defaults — the anchors quoted earlier are the result.

## Follow-ups

- Add the sweep plots (purity-vs-EV, drift-vs-EV per hue, preset × latitude
  sharpness grid) to CI or at least to the PR.
- Factory presets ("AgX base/punchy": v8 + black −10 EV / white +6.5 EV +
  matched contrast).
- Visual pass on the final defaults (latitude 10%, toe 1.5, slope-matched shoulder,
  refitted anchors) — numbers are derived, eyes are not optional.
- User documentation: see `ansel-doc/content/views/darkroom/modules/filmic.md`
  (v8 + spline v4 + darktable-AgX emulation guide).
