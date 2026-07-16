# Ansel — Developer Notes for AI Assistants

This file captures non-obvious architectural rules and hard-won bug knowledge for the Ansel
codebase. Read it before touching the areas it covers.

---

## Architectural rules

### No SQL in GUI modules

`src/libs/` and `src/views/` modules must contain no raw SQL. Database access belongs behind
named functions in `src/common/` (e.g. `common/collection.c`, `common/film.c`). When a GUI
module needs data, add or extend a `dt_collection_*` / `dt_film_*` / `dt_tag_*` function and
call it. Reuse existing helpers (`dt_collection_get_extended_where`, `dt_film_get_id`,
`dt_selection_select_list`) rather than re-issuing SQL.

Examples added during the collect rewrite: `dt_collection_get_property_values()`,
`dt_collection_get_images_for_rule()`, `dt_film_relocate()`.

### Pipeline↔module interface is history

The ONLY thread-safe interface between the pixel pipeline and an IOP module is **history**
(guarded by `dev->history_mutex`). `module->params` and `module->blend_params` belong to the
GUI thread and are NOT thread-safe — the pipeline thread must never read or write them.

Do NOT call `dt_iop_commit_params(module, module->params, ...)` from pipeline code. Commit
from the history snapshot (`hist->params`), never the live module params.

To push live/transient state to the pipe (e.g. drawlayer realtime stroke, ashift edit mode),
either (a) write it through history under `history_mutex`, or (b) use the transient-resync
interface `dt_dev_transient_params_{set,clear,get,active}` in `dev_history.{h,c}`.

See `doc/reorganisation.md` for the threading model (GUI diamond nodes vs. pipeline round nodes).

---

## Preferences system

Adding a Preferences entry requires **three** edits, not one:

1. `data/anselconfig.xml.in` — the `<dtconfig prefs="..." section="...">` entry.
2. `data/anselconfig.dtd` — the `section` attribute is an enumerated list; a new section value
   must be added here or `xmllint` fails the build (`USE_XMLLINT=ON`).
3. `tools/generate_prefs.xsl` — the GUI is generated into `build/src/preferences_gen.h`; each
   tab renders only **explicitly enumerated** `<xsl:for-each select="...@section='X'">` blocks.
   A section not listed in the XSL is silently dropped from the UI even if valid in XML/DTD.

Conf defaults: `dt_conf_key_exists()` returns true for any confgen key even on first run (defaults
are loaded at startup). To detect "user has never decided", use a **non-confgen** sentinel key
written only after the user acts — see the Sentry consent flow in `src/common/sentry.c`
(`sentry/consent_asked`).

---

## Pipeline & cache system

### GUI backbuf must use the published hash, not the planned hash

For final-backbuffer display (`dt_dev_pixelpipe_cache_peek_gui` with `piece == NULL`), GUI
consumers (center view, navigation thumbnail, scopes) must key the lookup on the **published**
`pipe->backbuf.hash`, not the **planned** `dt_dev_pixelpipe_get_hash(pipe)` (`pipe->hash`).

The pipeline plans the next frame's global hash before publishing pixels, so `pipe->hash` runs
ahead of `pipe->backbuf.hash` whenever a recompute is in flight. Realtime drawing makes this the
steady state. Peeking the planned hash misses the perfectly valid published frame → the main-surface
lock fails → darkroom expose falls back to the paused preview pipe → flicker.

In `peek_gui`, for `piece == NULL` use `dt_dev_backbuf_get_hash(&pipe->backbuf)` as the display
lookup hash when valid.

### OpenCL vRAM flush must not drop live entries

`darktable.pixelpipe_cache` is shared across all pipes. `dt_dev_pixelpipe_cache_flush_clmem`
iterates EVERY entry on a device, not just the calling pipe's own. The bug (issue #817): it
released the `cl_mem` of a buffer another pipe was mid-recursion on, leaving a husk (no RAM,
no vRAM) keyed in the cache, which then aborted downstream consumers → skull thumbnails.

The correct flush predicate: skip any entry where `dt_atomic_get_int(&entry->refcount) > 0` OR
`dt_pthread_rwlock_trywrlock` fails (never wait on writer locks). Idle entries (refcount 0,
unlocked) get their vRAM reclaimed. The flush must hold `cache->lock` for the entire iteration
so no consumer can mid-acquire a refcount==0 entry.

If a flushed entry is then empty (no host data + no vRAM on any device), remove it from the
hash table via `g_hash_table_iter_remove` — do NOT subtract `current_memory` manually, the
`_free_cache_entry` GDestroyNotify handles it.

### Mipmap invalidation is explicit, not hash-driven

The mipmap cache get path (`_generate_blocking` in `common/mipmap_cache.c`) does NOT compare
`history_hash` vs `mipmap_hash` to detect staleness. Regeneration only happens after an explicit
`dt_mipmap_cache_remove(cache, imgid, TRUE)`.

Every operation that mutates an image's history/development MUST explicitly:
1. `dt_mipmap_cache_remove`
2. Refresh the cached image metadata so `history_items` is correct (`_write_mipmap_to_disk` uses
   `img->history_items > 0` as the "altered" flag for the embedded-JPEG-vs-raw decision)
3. `dt_thumbtable_refresh_thumbnail`

The darkroom/paste path does this via `dt_dev_history_notify_change` (`dev_history.c`). Paths that
write history straight to DB (XMP load, `dt_image_set_flip`) bypass it and need the fix pattern:
`dt_image_cache_get_reload`, then remove mipmap + refresh thumbnail.

Do NOT refresh the filmstrip from darkroom write paths — it competes with the realtime main
preview pipeline. Lighttable ops may refresh both.

### OpenCL GUI-thread materialization hazard

`dt_dev_pixelpipe_cache_peek_gui` must pass `preferred_devid = -1` (CPU caller signal). Passing
a real GPU id causes the GUI thread to enqueue a GPU read without owning the device, racing the
pipeline's OpenCL events → SIGSEGV in `clReleaseEvent`. Device-only entries then report a miss
to the GUI, which waits for the pipeline to publish a host copy instead.

### The darkroom worker thread must be joined before view `leave()` tears down pipe state

`dt_dev_darkroom_pipeline()` runs forever in the dedicated `DT_CTL_WORKER_DARKROOM` job thread,
servicing `dev->preview_pipe` then `dev->pipe` in a loop. `views/darkroom.c`'s `leave()` sets
`dev->exit = 1` and each pipe's `shutdown` atomic, then — still from the GUI thread — calls
`dt_dev_pixelpipe_cleanup_nodes()` on `dev->pipe`/`dev->preview_pipe`/`dev->virtual_pipe` and frees
`dev->iop`/`dev->history`. Neither flag actually preempts the worker: `dev->exit` is only checked
between loop iterations and between servicing each pipe, and `pipe->shutdown` is never polled inside
`dt_dev_pixelpipe_process()` to abort mid-flight — it's only read afterwards, in
`dt_dev_darkroom_pipeline()`, to decide whether the just-produced result is valid.

`leave()`'s `busy_mutex` locks around the pipe-nodes teardown look like they serialize against the
worker, but `busy_mutex` carries a comment that it must "NEVER be used from the GUI thread" for
exactly this reason: two worker-thread accesses bypass it entirely — `dt_dev_pixelpipe_set_input()`
(iterates `pipe->nodes` every loop tick to refresh `piece->iwidth`/`iheight`) and the history-hash
resync at the top of `_resync_pipe_with_history()` (can re-trigger `dt_dev_pixelpipe_change()`,
which rebuilds `pipe->nodes` from `pipe->dev->iop`). Either can still be touching a pipe's nodes, or
`dev->iop`, after `leave()`'s mutex-guarded section already freed them. The resulting heap corruption
does not crash where it happens — it crashes wherever the next unlucky reader lands (Sentry issue
133807805: a garbage `xform_cam_Lab` pointer inside `iop/colorin.c`'s `cleanup_pipe()`, reached via
the worker's *own* next `resync_pipe_with_history()` call, nowhere near the actual race).

Fixed by making `dt_dev_pixelpipe_t.running` (set at the very top/bottom of
`dt_dev_darkroom_pipeline()`) an actual `dt_atomic_int` — it existed before but was write-only.
`leave()` now polls it for both `dev->pipe` and `dev->preview_pipe` right after setting
`dev->exit`/`shutdown`, and blocks until both read `FALSE` before touching any node/iop/history
teardown. Any other GUI-thread code that tears down darkroom pipe state must wait on this flag the
same way — `dev->exit`/`pipe->shutdown` alone do not guarantee the worker has stopped touching a
pipe.

### History items are refcounted; `history_mutex` resync contention is a known issue

`dt_dev_history_item_t` now carries a `refcount` and must be constructed exclusively through
`dt_dev_history_item_create()` — never a bare `calloc` (mirrors the masks-forms rule below;
`dt_dev_history_cow_touch()` clones a shared item before an in-place mutation, mirroring
`dt_masks_cow_touch()`).

`dt_dev_pixelpipe_change()` (worker thread, called from `dt_dev_darkroom_pipeline()`) holds
`dev->history_mutex` as **reader** for the entire O(nodes × history) pipe resync — measured over
200ms under mask-heavy history during active editing — which starves the GUI-thread writer
(`dt_dev_add_history_item_ext()`) for the same duration on every edit (a scroll on exposure, a
mask drag). Still open. See `doc/reorganisation.md` ("History item refcounting and the
`history_mutex` contention") for the full diagnosis and status, and the named-rwlock diagnostic
in `common/dtpthread.h` (`dt_pthread_rwlock_set_name()`, opt-in per lock, combine with
`-d history`) to reproduce the measurement.

### `piece->iwidth`/`iheight` go stale on the export pipe specifically

`dt_dev_pixelpipe_create_nodes()` copies `pipe->iwidth`/`iheight` into each `piece->iwidth`/`iheight`
once, at node-creation time — it is not refreshed on later ROI passes. Darkroom pipes call
`dt_dev_pixelpipe_set_input()` (which sets `pipe->iwidth`/`iheight`) before creating nodes; the
export pipe (`common/imageio.c`) does the reverse, so every piece was permanently stuck at 0 there
(issue #967: `iop/toneequal.c`'s blending radius and `iop/soften.c`'s glow radius silently collapsed
to 0 on export only, regardless of the module's params, while darkroom rendered correctly). Fixed by
having `dt_dev_pixelpipe_set_input()` re-sync `iwidth`/`iheight` onto any already-created nodes. See
`doc/resizing-scaling.md` for the full write-up; any other per-piece field seeded from `pipe->*` at
node-creation time is exposed to the same ordering hazard.

### `basebuffer` must crop using `roi_out`, not `roi_in`

`iop/basebuffer.c` is the first module in the pipe: it slices the requested window out of the
full-resolution mipmap-cache payload. Its `modify_roi_in()` unconditionally requests the whole
image (`{0, 0, pipe->iwidth, pipe->iheight}`) — `roi_in` never carries an offset, since basebuffer
needs the full frame available to crop from. The window actually requested downstream lives in
`piece->roi_out`, not `piece->roi_in`. `process()`/`process_cl()` must read the crop offset (and
the destination copy size) from `roi_out`, and use `pipe->iwidth`/`iheight` — not `roi_in->width`/
`height`, which is always the full frame too — for the source row stride. Reading the offset from
`roi_in` instead always crops from the sensor's true `(0,0)`: harmless whenever the requested
window is itself near `(0,0)` (a fit-to-screen view, a barely-cropping module), silently wrong by
the full requested offset otherwise (e.g. `iop/lens.cc`'s `scale` slider, whose backward-pass
`roi_in.x/y` grows with the zoom amount) — every downstream module still looks internally
consistent (sizes match, ROI planning round-trips cleanly), because each of them only reads
buffer-relative pixels and never re-derives its own absolute position from `pipe->iwidth`/
`iheight`. Parametric masks/forms don't go through this buffer-cropping path at all, so they stay
correctly positioned even when the base image content is offset — a mismatch between a mask and
the image it's drawn on is a symptom of this class of bug, not of the masking code.

---

## Masks / forms history

### Forms are refcounted, not deep-copied

`dev->forms` (`dt_develop_t`) is the live, mutable `GList` of every mask shape and group
(`dt_masks_form_t*`) in the current image. Groups don't nest forms directly — a group's `points`
is a list of `dt_masks_form_group_t` entries (`{formid, parentid, state, opacity}`) referencing
sibling forms in the same flat `dev->forms` list by ID.

Every history commit that touches masks used to deep-copy the *entire* `dev->forms` list into
`hist->forms` (`dt_dev_history_item_t`), even when only one shape on one module changed. Forms
are now refcounted (`dt_masks_form_t.refcount`, `src/develop/masks/masks_history.{h,c}`) instead:

- `dt_masks_snapshot_current_forms()` takes a reference on each current `dev->forms` element
  instead of copying it. Multiple `hist->forms` snapshots (and `dev->forms` itself) can share the
  exact same `dt_masks_form_t*`.
- `dt_masks_cow_touch(dev, form)` is the copy-on-write gate: before *mutating* a form (move,
  resize, remove a group member...), check its refcount. If it's 1 (only `dev->forms` holds it),
  mutate in place. If it's shared (an undo/redo or history snapshot also references it), clone it
  first, splice the clone into `dev->forms` in place of the original, and mutate the clone —
  never mutate a form that might be observed by a frozen snapshot. Every mutation call site
  (mouse/keyboard event dispatchers in `masks.c`, `dt_masks_form_delete`, group add/move/ungroup,
  `blend_gui.c` group operations, the shape-manager panel in `libs/masks.c`) must route through
  this before touching `form->points` or any other field. `dt_masks_cow_touch` also re-points
  `dev->form_gui->form_visible` if it was the form that got cloned — that's the only other raw
  `dt_masks_form_t*` cached outside `dev->forms`.
- `dt_masks_replace_current_forms()` swaps `dev->forms` wholesale (used when history navigation
  rebuilds it) by releasing the old references and taking new ones — never a raw deep copy.
- `pipe->forms` (the pixel-pipeline's own snapshot, taken once per `dt_dev_pixelpipe_process()`
  call, `pixelpipe_hb.c`) is shared by reference the same way. It has exactly one real consumer
  (`iop/retouch.c`, read-only), so no COW gate is needed on that side — `dt_masks_cow_touch`
  already guarantees a GUI-side edit clones instead of mutating a form an in-flight pipeline run
  is holding.

### A form mutation that never reaches a history commit is invisible to undo/redo

`dt_dev_add_history_item_ext()` (`dev_history.c`) is the only place that turns the current
`dev->forms` state into a `hist->forms` snapshot, and only when
`dt_iop_module_needs_mask_history(module)` is true for the committing module. **Any code path
that mutates `dev->forms` (directly or via `dt_masks_form_delete`/group helpers) must be followed
by a `dt_dev_add_history_item()` call**, or the mutation only ever exists in live memory.

Undo/redo (`_pop_undo`, `dev_history.c`) replaces `dev->history` with a duplicate of the
recorded `before_snapshot`/`after_snapshot` (`dt_history_duplicate`, itself ref-sharing) and calls
`dt_dev_pop_history_items_ext()`, which rebuilds `dev->forms` from the `hist->forms` of the
**last history item that actually has one** — walking backwards over items with
`hist->forms == NULL`. If a mutation was never committed, every subsequent history navigation
silently falls back to whatever was last actually recorded and the live edit is lost. Confirmed
bug instances, found by auditing every handler in `libs/masks.c` for a trailing
`dt_dev_add_history_item()`/`_add_masks_history_item()` call: `_tree_delete_shape` (delete),
`_tree_moveup`/`_tree_movedown` (reorder inside a group — silently lost on next undo/redo), and
`_tree_duplicate_shape` (the duplicate was also never attached to the source shape's parent group
via `dt_masks_group_add_form`, so it was an orphan on top of being uncommitted). All four are
fixed; audit any *new* handler in `libs/masks.c` / `blend_gui.c` that mutates forms without a
trailing commit before trusting its undo/redo behavior.

### Same-thread rwlock reentrancy

Committing masks more often surfaces a pre-existing, unrelated hazard: `dt_dev_pixelpipe_change()`
can be re-entered by the same thread while it already holds `history_mutex` as writer (a
history-commit path resyncing the virtual pipe mid-commit). glibc's default
`PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP` policy self-deadlocks such a thread as soon as a
second thread is queued for the write lock. Fixed by porting the same-thread recursive-writer
tracking that already existed in the `_DEBUG` build of `dt_pthread_rwlock_t`
(`common/dtpthread.h`: `writer` + `writer_depth` fields) into the release path too — a thread
that already holds the write lock cannot race itself, so letting it re-enter (as reader or
writer) is safe. `try*` locks keep their "is it locked by anyone?" probe contract and still
report busy on same-thread reentry, so callers relying on that semantic are unaffected.

### DB/XMP persistence still duplicates content per history step

`masks_history` (SQL table) and `Xmp.darktable.masks_history[N]` (XMP array) store one
row/entry per (history step, formid), with no dedup — the in-memory refcounting above stops at
the persistence boundary, so a form shared unchanged across 100 history steps still gets its
points BLOB serialized 100 times on every commit (`dt_dev_write_history_ext` rewrites the whole
image's history + masks_history every time). Known, not yet fixed — see
`doc/masks_history_dedup.md` for the full design (developed on a dedicated branch, merged only
when Ansel 1.0 is prepared, per explicit instruction not to migrate any user's live DB
prematurely).

---

## IOP modules

### ashift: preview buffer and crop geometry

The reference for ashift edit-mode is the **crop module** (`src/iop/crop.c`) — same "show the
full uncropped image while editing a clipping module" problem.

**Show the full image by neutralizing the crop in `commit_params()`** (crop: `cx=cy=0,cw=ch=1`;
ashift: `cl=0,cr=1,ct=0,cb=1`). Output, input, view, and size caches must all describe the same
full frame. Do NOT widen only `roi_in` while leaving `roi_out` cropped — the pipe renders a
cropped output while the view wants the full image → preview aborts at `initialscale` and
restarts forever.

**`g->buf` comes from `process()` capture, not `peek_gui`.** During edit ashift is cache-bypass,
so `process()` runs on every render and copies its input into `g->buf`. Do NOT use
`dt_dev_pixelpipe_cache_peek_gui()` for ashift's own input — it never runs `process()`, the
intermediate is evicted before the GUI reads it → re-request loop.

**Auto-crop geometry needs only size, not pixels.** Use `piece->buf_in.width/height`
(crop-independent), NOT `roi_in`/`g->buf` dims. Use the preview pipe's `buf_in`, not the virtual
pipe.

**`has_preview_output()` requires matching both portrait and landscape.** ashift runs before
`flip` (iop_order 16 < 20), so on portrait images its `roi_out` is landscape while preview dims
are post-flip portrait. The guard must also accept the swapped match
(`width==preview_height && height==preview_width`).

### drawlayer: realtime stroke correctness

**Stroke truncation:** `dt_drawlayer_commit_dabs` must guard on `painting_active` for BOTH quiet
and record-history commits. `_build_runtime_schedule` schedules quiet commits for `GUI_SCROLL`
and `GUI_SYNC_TEMP_BUFFERS` — these fire during active strokes and will truncate the path if
`commit_dabs` does not early-return. The fix: `if(g->manager.painting_active){ ... return TRUE; }`.

**Realtime trigger / hover thrash:** `_update_realtime_state` must track `painting_active` only.
The `GUI_RAW_INPUT` / `SAMPLE` override that set `realtime_active=TRUE` regardless of
`painting_active` caused hover mouse-moves to toggle realtime ON/OFF on every pixel → ~44ms
`resync_history_main` at each stroke boundary. Only an actual stroke (`STROKE_BEGIN`/`STROKE_END`)
should enter/leave realtime.

**Partial composite gate:** gate the damage-limited resample on the stable per-layer identity
`process->base_patch.cache_hash` (NOT on `piece->global_hash` — the heartbeat bumps
`stroke_commit_hash` every frame, so `global_hash` changes every realtime frame).

**Transient-params channel:** the realtime heartbeat (`_publish_backend_progress` in `worker.c`)
publishes via `dt_dev_transient_params_set` instead of `add_history_item`, avoiding per-heartbeat
undo/DB churn. History is written only at the real commit. Crop/ashift use `resync_history_all`
(full, all pipes); drawlayer heartbeat raises `TOP_CHANGED` + redraw (fast, non-geometry). The
two must NOT be mixed — routing crop's geometry through `_sync_focused_in_place` (partial)
mishandles the warm cropped→uncropped geometry change.

---

## Collection / Library module

`src/libs/collect.c` is the left-panel "Library" GUI. It does NOT build the collection query —
that is `src/common/collection.c`. The GUI's only job is to write the conf keys
`plugins/lighttable/collect/{num_rules, item<N>, mode<N>, string<N>, tab}` and call
`dt_collection_update_query()`.

Three tabs: **Folders** (film-roll list / folder tree; relocate + remove in batches),
**Collections** (tag browser + delete + rename), **Queries** (multi-rule builder + raw SQL via
`DT_COLLECTION_PROP_QUERY`).

Drag-and-drop of lighttable images onto tree rows was attempted and abandoned — a GtkTreeView
with a manual `gtk_drag_dest_set` reliably receives motion but does not deliver the drop on tree
models. DnD was removed entirely at the maintainer's request; do not re-add without a
non-tree drop target or `tagging.c`-style full source+dest.

---

## GTK / UI

### Thumbtable scrolled-window sizing

Three separate, mostly-invisible per-cell overheads must be budgeted for the thumbtable
`GtkFixed` grid to fit flush:

1. **scrollbar-spacing** — GtkScrolledWindow legacy GtkWidget *style property* (default 3px),
   NOT a CSS box property. Zero it via `-GtkScrolledWindow-scrollbar-spacing: 0` in CSS on
   `#thumbtable-scroll` / `#panel-scroll`.
2. **frame borders** — `GTK_SHADOW_ETCHED_IN` + the implicit GtkViewport's `GTK_SHADOW_IN` both
   add a `.frame` class. Set both to `GTK_SHADOW_NONE`.
3. **per-cell decoration** — `.thumb-cell { border: 4px transparent; margin: -2px }` makes each
   cell ~4px wider than the `thumb_width` stride. Budget it:
   `thumb_width = floor((new_width - deco) / cols)`.

**Critical:** `dt_thumbtable_configure` is the single source of truth for thumb geometry. Pass
the already-computed `new_thumbs_per_row/new_thumb_width/new_thumb_height` to `_grid_configure`,
which must only STORE them, never re-derive. If two code paths compute thumb geometry with
different formulas, `thumbs_changed` is true on every idle tick → full grid repopulate every
tick → ~20% idle CPU.

**Filmstrip-specific:** the filmstrip `scroll_window` must be the MAIN child of `parent_overlay`
(via `gtk_container_add`), NOT an overlay child. Overlay children on Wayland use an offscreen
path and go stale/blank until a pointer event invalidates them. The filmstrip vertical scroll
policy must be `GTK_POLICY_EXTERNAL` + `set_min_content_height(1)` +
`set_propagate_natural_height(FALSE)` to allow the resize handle to shrink the panel.

**Re-entry init:** `dt_thumbtable_show` must reset `last_parent_width/height` and
`last_h_scrollbar_height/last_v_scrollbar_width` to -1 so the next `size-allocate` always
reconfigures (the table persists across view enter/leave; the guard would otherwise skip the
reconfigure on same-size re-entry).

---

## Interpolation

Mitchell-Netravali (B=C=1/3) is the pipeline interpolator. Lanczos has been removed entirely.
Rationale: Lanczos has large negative side-lobes → halos at high-contrast edges and pushes
premultiplied alpha out of [0,1]. Mitchell is near-halo-free (~3% residual undershoot), sharp,
and a separable partition-of-unity kernel that fits the existing tap machinery for CPU and GPU.

The pipeline's interpolation architecture in `src/common/interpolation.c` is separable — each
kernel registers a 1D `maketaps`, and both `dt_interpolation_resample` (CPU) and
`dt_interpolation_resample_cl` (GPU) consume the same CPU-computed taps. A new separable kernel
is automatically CPU+GPU.

The drawlayer brush matte still forces `BILINEAR` explicitly — premultiplied alpha wants strictly
zero overshoot.

Config option strings in `anselconfig.xml.in` MUST equal the kernel `.name` field exactly;
`USERPREF` resolves by strcmp. A mismatch (e.g. `"bicubic (Catmull-Rom)"` vs `"bicubic"`)
silently falls back to default instead of erroring.

---

## Tools

**Sentry crash issues:** `tools/sentry-fetch-issue.sh <issue-id|url>` pulls a Sentry issue's
backtrace locally (writes `summary.txt`, `event.json`, attachments). The region host is
`https://de.sentry.io` (EU data residency) — `sentry.io`/`us.sentry.io` give 403/401.
Reading issues needs a **User Auth Token** (not the org token used for symbol upload).
See `doc/sentry.md` for setup details.

---

## General engineering

Ansel carries the burden of Darktable legacy, which made it a principle to entangle all
application layers (GUI, pipeline, history, database) and imported the whole software
into the whole software through `#include "common/darktable.h"`. This voids the modularity
principle, creates many bugs, data races, and makes any maintenance tedious and prone to 
edge effects, since the app is heavily asynchronous and parallel.

The Ansel codebase should move toward more enclosed modularity, making data structure private
to each translation unit and exposing only API to the outside (getters/setters/init/cleanup). 
Direct value changes on data not owned by the current TU are forbidden. The dependency graph 
should be simplified and only a minimal set of `#include` should be kept per TU. In particular,
`src/common/darktable.h` should inherit from lower-level modules, but lower-level modules
should not inherit it, so it should stop being the glue of all common helpers throughout
the software.

CRUD operations should have one central entry point for the whole software and run only
once, for as long as user didn't send new input, so the data lifecycle is legible and
cacheable.

Since every data flow in the software is a pipeline, issues should be tracked to their root
cause by climbing the call tree up until the source is found, instead of being fixed where
they are visible.