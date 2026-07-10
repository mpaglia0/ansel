# Masks history deduplication (planned, not yet implemented) {#masks_history_dedup}

[TOC]

## Status

**Design only — no code has been written yet.** This document exists so the design survives
between sessions and contributors; it is not a description of current behavior. See
`CLAUDE.md`'s "Masks / forms history" section for what *is* currently implemented (the in-memory
refcounting/copy-on-write refactor this design builds on).

## Problem

The in-memory forms-history refactor (`src/develop/masks/masks_history.{h,c}`) made `dev->forms`/
`hist->forms` share the same `dt_masks_form_t*` objects by reference across history steps,
cloning only on write (copy-on-write). This sharing **stops at the persistence boundary**. Two
places still serialize one row/entry **per (history step, formid)**, with no dedup, even when the
exact same form is unchanged across dozens of consecutive steps:

- SQL table `masks_history` (`src/common/database.c:1855`):
  `(imgid, num, formid, form, name, version, points, points_count, source)`, no `UNIQUE`
  constraint. Written by `dt_masks_write_masks_history_item()`
  (`src/develop/masks/masks.c:2181`), called once per form per history item by
  `dt_dev_write_history_item()` (`src/develop/dev_history.c:1355-1359`), itself called for
  **every** history step by `dt_dev_write_history_ext()` (`src/develop/dev_history.c:1372`).
  That function deletes and rewrites the **entire** history + masks_history for the image on
  **every single commit** (`_cleanup_history` → `dt_history_db_delete_dev_history` →
  `dt_history_db_delete_masks_history`). A form shared unchanged across 100 history steps (the
  common case, enabled by the refactor above) still gets its full points BLOB serialized 100
  times, on every commit.
- XMP array `Xmp.darktable.masks_history[N]` (`src/common/exif.cc:3515-3553`): reads directly
  from the SQL table above and writes one XMP array entry per row — same duplication, downstream.

`dt_dev_mask_history_overload()` (`src/develop/dev_history.c:1297`) already warns users about
this via a toast ("consider compressing history...") without fixing it. The maintainer already
flagged the intended direction as a code comment near `dt_masks_read_masks_history()`
(`src/develop/masks/masks.c:2159-2163`): attach the forms snapshot to its own object, linked by ID
to the history item, instead of duplicating it inline.

**Historical precedent**: before XMP format version 3, `read_masks()` (`src/common/exif.cc`,
legacy path) stored a single entry per `mask_id` for the whole image — already deduplicated, with
no `num` dimension. The current "one row per step" scheme is a v3 regression. This design
reverses it.

## Hard constraint: must not touch any user's live database before Ansel 1.0 ships

This is a persistent-schema change to a table every existing user already has data in. It is
developed and merged on a **dedicated branch**, kept out of `master` until the Ansel 1.0 release
is actually being prepared — never landed on the branch users' current builds track. This avoids
any risk of migrating a live database prematurely, without needing conditional compilation.

## Design

### SQL schema

Split the current table in two (same rename → create → copy-transform → drop migration pattern
already used elsewhere in this file, e.g. `src/common/database.c:1851-1872`):

```sql
CREATE TABLE masks_history_forms (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  imgid INTEGER,
  formid INTEGER, form INTEGER, name VARCHAR(256), version INTEGER,
  points BLOB, points_count INTEGER, source BLOB,
  UNIQUE(imgid, formid, form, name, version, points, points_count, source),
  FOREIGN KEY(imgid) REFERENCES images(id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE masks_history (   -- becomes a thin reference table
  imgid INTEGER, num INTEGER, formid INTEGER, content_id INTEGER,
  FOREIGN KEY(imgid) REFERENCES images(id) ON DELETE CASCADE ON UPDATE CASCADE,
  FOREIGN KEY(content_id) REFERENCES masks_history_forms(id) ON DELETE CASCADE
);
CREATE INDEX masks_history_imgid_index ON masks_history(imgid, num);
```

The `UNIQUE` constraint on the content columns does the dedup **at the SQL level**, no
application-side hash needed (no new dependency, no collision risk): `INSERT OR IGNORE` into
`masks_history_forms`, then `SELECT id` (via the unique index, so fast) to get the `content_id`
to reference. This works correctly **regardless of caller** — important because
`dt_masks_write_masks_history_item()` has **two call sites**: the main per-commit loop
(`src/develop/dev_history.c:1355`) and the legacy "spots" compatibility path
(`src/iop/spots.c:189`) — both benefit automatically, with no cache to thread between them.

Since `dt_dev_write_history_ext()` deletes and rewrites the whole image's history on every
commit, dedup happens naturally on every full rewrite: `_cleanup_history` must empty **both**
tables for that `imgid`, then the rewrite loop repopulates `masks_history_forms` only with
content actually in use — self-cleaning, no separate GC pass needed.

Migration: new `else if(version == 36)` block in `_upgrade_library_schema_step`
(`src/common/database.c:485`), bumping `CURRENT_DATABASE_VERSION_LIBRARY` 36 → 37. Pure SQL,
no C-driven transform needed: `INSERT ... SELECT DISTINCT` populates `masks_history_forms`, then
`INSERT ... SELECT ... JOIN` populates the new thin `masks_history` resolving `content_id`.

The ephemeral `memory.undo_masks_history` table (`src/common/database.c:2545`, recreated on
every startup, **not** subject to the version counter — no migration needed, just update its
`CREATE TABLE`) needs a `memory.undo_masks_history_forms` mirror. `src/common/history_snapshot.c`
(create/restore/clear around lines 95, 168, 228) must copy **both** tables for lighttable
undo/redo snapshots.

### Read path (in-memory dedup bonus)

`dt_masks_read_masks_history()` (`src/develop/masks/masks.c:2075`): join the two tables on
`content_id`, ordered by `num`, as today. Keep a local `GHashTable` (`content_id` →
`dt_masks_form_t*`) while looping: if a `content_id` was already materialized, call
`dt_masks_form_ref()` on the existing object instead of allocating a new one. This extends the
in-memory refactor's benefit to **loading** an image, not just editing an already-open session.

### XMP (new v4 format, selected by file version)

The XMP reader already supports multiple mask-format generations, selected by the file's
declared version (legacy `read_masks()` vs `read_masks_v3()`, `src/common/exif.cc:~2850-2918`).
Add `read_masks_v4()` next to the existing readers, dispatched the same way. Writing switches to
the new format outright (no old-format writer to keep around, since this all lives on the
dedicated branch until merge):

```
Xmp.darktable.masks_history_forms[K]/darktable:{mask_id, mask_type, mask_name, mask_version, mask_points, mask_nb, mask_src}
Xmp.darktable.masks_history_refs[N]/darktable:{mask_num, mask_content_ref}   -- mask_content_ref = index K
```

Direct mirror of the SQL split (content array + thin reference array), reusing the same
in-memory dedup pass (one walk over `dev->history`, same pointer/content_id keyed hash table as
the SQL side). Touches `src/common/exif.cc:3515-3553` (write) and needs the new `read_masks_v4()`
next to `read_masks_v3()` (`src/common/exif.cc:~2913`).

## Files to touch

- `src/common/database.c` — new `version==36` migration block, bump
  `CURRENT_DATABASE_VERSION_LIBRARY`, new `memory.undo_masks_history_forms` ephemeral table.
- `src/develop/masks/masks.c` — rewrite `dt_masks_write_masks_history_item()` and
  `dt_masks_read_masks_history()` for the two-table schema.
- `src/common/history.c` — `dt_history_db_delete_masks_history()` must empty both tables.
- `src/common/history_snapshot.c` — lighttable undo snapshot create/restore/clear, mirrored to
  two tables.
- `src/common/exif.cc` — new XMP v4 writer; new `read_masks_v4()` selected by file version
  (v3 reader stays, for files written by older Ansel/darktable versions).
- `src/iop/spots.c` — no signature change needed (benefits automatically via
  `dt_masks_write_masks_history_item()`), verify in testing only.

## Verification plan

1. On the dedicated branch, against a **copy** of a test database (never a real user DB): run the
   migration, verify `SELECT COUNT(*) FROM masks_history_forms` is far below the old row count for
   an image with deep history, and that the image reloads identically (compare rendering
   before/after migration). Test: lighttable undo/redo, copying history to another image,
   `iop/spots.c` (image with legacy "spots"), XMP export then reimport, history compression.
2. Confirm XMP files written by older Ansel/darktable versions (v3 format) still import correctly
   through the existing `read_masks_v3()` path.
3. Measure `dt_dev_write_history_ext()` time before/after on a heavily-masked image (e.g. the
   140-shape retouch test image used for pipeline performance work), to quantify the win.
4. Merge to `master` only when Ansel 1.0 is actually being prepared.
