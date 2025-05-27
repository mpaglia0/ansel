/*
    This file is part of darktable,
    Copyright (C) 2009-2020 darktable developers.
    Copyright (C) 2022-2025 Aurélien PIERRE.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <inttypes.h>

struct dt_dev_pixelpipe_t;
struct dt_iop_buffer_dsc_t;
struct dt_iop_roi_t;


/**
 * @file pixelpipe_cache.h
 * @brief Pixelpipe cache for storing intermediate results in the pixelpipe.
 *
 * This cache can be used locally (in the pixelpipe) or globally (in the whole app).
 * Current implementation is global, using `darktable.pipeline_threadsafe` mutex lock
 * to protect cache entries addition/removal accross threads. The mutex lock
 * protects the whole recursive pixelpipe, so no internal locking is needed nor implemented here.
 */

typedef struct dt_dev_pixelpipe_cache_t
{
  GHashTable *entries;
  uint64_t queries;
  uint64_t hits;
  size_t max_memory;
  size_t current_memory;
  dt_pthread_mutex_t lock; // mutex to protect the cache entries
} dt_dev_pixelpipe_cache_t;

/** constructs a new cache with given cache line count (entries) and float buffer entry size in bytes.
  \param[out] returns 0 if fail to allocate mem cache.
*/
dt_dev_pixelpipe_cache_t * dt_dev_pixelpipe_cache_init(size_t max_memory);
void dt_dev_pixelpipe_cache_cleanup(dt_dev_pixelpipe_cache_t *cache);

/**
 * @brief Get a cache line from the cache.
 *
 * @param cache
 * @param hash State checksum of the cache line.
 * @param size Buffer size in bytes.
 * @param name Name of the cache line (for debugging).
 * @param id ID of the pipeline owning the cache line.
 * @param data Pointer to the buffer pointer (returned).
 * @param dsc Pointer to the buffer descriptor (returned).
 * @return int 1 if the cache line was freshly allocated, 0 if it was found in the cache.
 */
int dt_dev_pixelpipe_cache_get(dt_dev_pixelpipe_cache_t *cache, const uint64_t hash,
                               const size_t size, const char *name, const int id,
                               void **data, struct dt_iop_buffer_dsc_t **dsc);

/**
* @brief Get an existing cache line from the cache. This is similar to `dt_dev_pixelpipe_cache_get`,
* but it does not create a new cache line if it is not found.
*
* @param cache
* @param hash
* @param data
* @param dsc
* @return int TRUE if found, FALSE if not found.
*/
int dt_dev_pixelpipe_cache_get_existing(dt_dev_pixelpipe_cache_t *cache, const uint64_t hash,
                                        void **data, struct dt_iop_buffer_dsc_t **dsc);

/** test availability of a cache line without destroying another, if it is not found. */
int dt_dev_pixelpipe_cache_available(dt_dev_pixelpipe_cache_t *cache, const uint64_t hash);

/**
 * @brief Remove cache lines matching id
 *
 * @param cache
 * @param id ID of the pipeline owning the cache line, or -1 to remove all lines.
 */
void dt_dev_pixelpipe_cache_flush(dt_dev_pixelpipe_cache_t *cache, const int id);

/**
 * @brief Force-delete a cache line without checking if it's locked.
 * Reserve this to garbled outputs that can't be used at all.
 *
 * @param cache
 * @param data
 */
void dt_dev_pixelpipe_cache_invalidate(dt_dev_pixelpipe_cache_t *cache, void *data);

/** print out cache lines/hashes (debug). */
void dt_dev_pixelpipe_cache_print(dt_dev_pixelpipe_cache_t *cache);

/** remove the least used cache entry
 * @return 0 on success, 1 on error
 */
int dt_dev_pixel_pipe_cache_remove_lru(dt_dev_pixelpipe_cache_t *cache);

/**
 * @brief Increase/Decrease the reference count on the cache line as to prevent
 * LRU item removal.
 *
 * @param cache
 * @param hash
 * @param lock TRUE to lock, FALSE to unlock
 */
void dt_dev_pixelpipe_cache_lock_entry_hash(dt_dev_pixelpipe_cache_t *cache, const uint64_t hash, gboolean lock);

/**
 * @brief Find the hash of the cache entry holding the buffer data
 *
 * @param cache
 * @param data
 * @return uint64_t defaults to 0 if nothing was found.
 */
uint64_t dt_dev_pixelpipe_cache_get_hash_data(dt_dev_pixelpipe_cache_t *cache, void *data);

/**
 * @brief Chains `dt_dev_pixelpipe_cache_lock_entry_hash` with
 * `dt_dev_pixelpipe_cache_get_hash_data`
 *
 * @param cache
 * @param data
 * @param lock TRUE to lock, FALSE to unlock
 */
void dt_dev_pixelpipe_cache_lock_entry_data(dt_dev_pixelpipe_cache_t *cache, void *data, gboolean lock);


/**
 * @brief Lock or release the write lock on the entry
 *
 * @param cache
 * @param hash
 * @param lock TRUE to lock, FALSE to release
 */
void dt_dev_pixelpipe_cache_wrlock_entry(dt_dev_pixelpipe_cache_t *cache, const uint64_t hash, gboolean lock);


/**
 * @brief Lock or release the read lock on the entry
 *
 * @param cache
 * @param hash
 * @param lock TRUE to lock, FALSE to release
 */
void dt_dev_pixelpipe_cache_rdlock_entry(dt_dev_pixelpipe_cache_t *cache, const uint64_t hash, gboolean lock);


// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
