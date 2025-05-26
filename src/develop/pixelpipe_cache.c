/*
    This file is part of Ansel
    Copyright (C) 2025 - Aurélien PIERRE

    Ansel is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Ansel is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "develop/pixelpipe_cache.h"
#include "common/darktable.h"
#include "common/debug.h"
#include "develop/format.h"
#include "develop/pixelpipe_hb.h"
#include <glib.h>
#include <stdlib.h>


typedef struct dt_pixel_cache_entry_t
{
  uint64_t hash;
  void *data;
  size_t size;
  dt_iop_buffer_dsc_t dsc;
  int age;
  char *name; // name of the cache entry, for debugging
  int id;
  dt_atomic_int refcount; // reference count for the cache entry, to avoid freeing it while still in use
  dt_pthread_rwlock_t lock;
} dt_pixel_cache_entry_t;


size_t dt_pixel_cache_get_size(dt_pixel_cache_entry_t *cache_entry)
{
  return cache_entry->size / (1024 * 1024);
}

void dt_pixel_cache_message(dt_pixel_cache_entry_t *cache_entry, const char *message)
{
  if(!((darktable.unmuted & DT_DEBUG_PIPE) && (darktable.unmuted & DT_DEBUG_VERBOSE))) return;
  dt_print(DT_DEBUG_PIPE, "[pixelpipe] cache entry %lu: %s (%lu MiB - age %i) %s\n", cache_entry->hash, cache_entry->name,
           dt_pixel_cache_get_size(cache_entry), cache_entry->age, message);
}

// remove the cache entry with the given hash and update the cache memory usage
// WARNING: not internally thread-safe, protect its calls with mutex lock
void dt_dev_pixel_pipe_cache_remove(dt_dev_pixelpipe_cache_t *cache, const uint64_t hash)
{
  dt_pixel_cache_entry_t *cache_entry = (dt_pixel_cache_entry_t *)g_hash_table_lookup(cache->entries, GINT_TO_POINTER(hash));
  if(cache_entry)
  {
    // Returns 1 if the lock is captured by another thread
    // 0 if WE capture the lock, and then need to release it
    gboolean locked = dt_pthread_rwlock_trywrlock(&cache_entry->lock);
    if(!locked) dt_pthread_rwlock_unlock(&cache_entry->lock);
    gboolean used = dt_atomic_get_int(&cache_entry->refcount) > 0;

    if(!used && !locked)
    {
      cache->current_memory -= cache_entry->size;
      g_hash_table_remove(cache->entries, GINT_TO_POINTER(hash));
    }
    else if(used)
      dt_pixel_cache_message(cache_entry, "cannot remove: used");
    else if(locked)
      dt_pixel_cache_message(cache_entry, "cannot remove: locked");
  }
  else
  {
    dt_print(DT_DEBUG_PIPE, "[pixelpipe] cache entry %lu not found, will not be removed\n", hash);
  }
}

typedef struct _cache_lru_t
{
  int max_age;
  uint64_t hash;
} _cache_lru_t;


// find the cache entry hash with the oldest use
void _cache_get_oldest(gpointer key, gpointer value, gpointer user_data)
{
  dt_pixel_cache_entry_t *cache_entry = (dt_pixel_cache_entry_t *)value;
  _cache_lru_t *lru = (_cache_lru_t *)user_data;

  // Don't remove LRU entries that are still in use
  // NOTE: with all the killswitches mechanisms and safety measures,
  // we might have more things decreasing refcount than increasing it.
  // It's no big deal though, as long as the (final output) backbuf
  // is checked for NULL and not reused if pipeline is DIRTY.
  if(cache_entry->age > lru->max_age)
  {
    // Returns 1 if the lock is captured by another thread
    // 0 if WE capture the lock, and then need to release it
    gboolean locked = dt_pthread_rwlock_trywrlock(&cache_entry->lock);
    if(!locked) dt_pthread_rwlock_unlock(&cache_entry->lock);
    gboolean used = dt_atomic_get_int(&cache_entry->refcount) > 0;

    if(!locked && !used)
    {
      lru->max_age = cache_entry->age;
      lru->hash = cache_entry->hash;
      dt_pixel_cache_message(cache_entry, "candidate for deletion");
    }
    else if(used)
      dt_pixel_cache_message(cache_entry, "cannot be deleted: used");
    else if(locked)
      dt_pixel_cache_message(cache_entry, "cannot be deleted: locked");
  }
}


// remove the least used cache entry
static void _non_thread_safe_pixel_pipe_cache_remove_lru(dt_dev_pixelpipe_cache_t *cache)
{
  _cache_lru_t *lru = (_cache_lru_t *)malloc(sizeof(_cache_lru_t));
  lru->max_age = -1;
  lru->hash = 0;
  g_hash_table_foreach(cache->entries, _cache_get_oldest, lru);
  dt_dev_pixel_pipe_cache_remove(cache, lru->hash);
  free(lru);
}

void dt_dev_pixel_pipe_cache_remove_lru(dt_dev_pixelpipe_cache_t *cache)
{
  dt_pthread_mutex_lock(&cache->lock);
  _non_thread_safe_pixel_pipe_cache_remove_lru(cache);
  dt_pthread_mutex_unlock(&cache->lock);
}

// WARNING: not thread-safe, protect its calls with mutex lock
static dt_pixel_cache_entry_t *dt_pixel_cache_new_entry(const uint64_t hash, const size_t size,
                                                        const dt_iop_buffer_dsc_t dsc, const char *name, const int id,
                                                        dt_dev_pixelpipe_cache_t *cache)
{
  // Free up space if needed to match the max memory limit
  while(cache->current_memory + size > cache->max_memory && g_hash_table_size(cache->entries) > 0)
    _non_thread_safe_pixel_pipe_cache_remove_lru(cache);

  dt_pixel_cache_entry_t *cache_entry = (dt_pixel_cache_entry_t *)malloc(sizeof(dt_pixel_cache_entry_t));
  if(!cache_entry) return NULL;

  // allocate the data buffer
  cache_entry->data = dt_alloc_align(size);

  // if allocation failed, remove the least recently used cache entry, then try again
  while(cache_entry->data == NULL && g_hash_table_size(cache->entries) > 0)
  {
    _non_thread_safe_pixel_pipe_cache_remove_lru(cache);
    cache_entry->data = dt_alloc_align(size);
  }

  if(!cache_entry->data)
  {
    free(cache_entry);
    return NULL;
  }

  cache_entry->size = size;
  cache_entry->age = 0;
  cache_entry->dsc = dsc;
  cache_entry->hash = hash;
  cache_entry->id = id;
  cache_entry->name = g_strdup(name);
  cache_entry->refcount = 0;
  dt_pthread_rwlock_init(&cache_entry->lock, NULL);

  g_hash_table_insert(cache->entries, GINT_TO_POINTER(hash), cache_entry);
  cache->current_memory += size;

  return cache_entry;
}


static void _free_cache_entry(dt_pixel_cache_entry_t *cache_entry)
{
  if(!cache_entry) return;
  dt_pixel_cache_message(cache_entry, "freed");
  dt_free_align(cache_entry->data);
  cache_entry->data = NULL;
  dt_pthread_rwlock_destroy(&cache_entry->lock);
  g_free(cache_entry->name);
  free(cache_entry);
}


dt_dev_pixelpipe_cache_t * dt_dev_pixelpipe_cache_init(size_t max_memory)
{
  dt_dev_pixelpipe_cache_t *cache = (dt_dev_pixelpipe_cache_t *)malloc(sizeof(dt_dev_pixelpipe_cache_t));
  dt_pthread_mutex_init(&cache->lock, NULL);
  cache->entries = g_hash_table_new_full(g_direct_hash, g_direct_equal, NULL, (GDestroyNotify)_free_cache_entry);
  cache->max_memory = max_memory;
  cache->current_memory = 0;
  cache->queries = cache->hits = 0;
  return cache;
}


void dt_dev_pixelpipe_cache_cleanup(dt_dev_pixelpipe_cache_t *cache)
{
  if(!cache) return;
  dt_pthread_mutex_destroy(&cache->lock);
  g_hash_table_destroy(cache->entries);
  cache->entries = NULL;
}


int dt_dev_pixelpipe_cache_available(dt_dev_pixelpipe_cache_t *cache, const uint64_t hash)
{
  dt_pthread_mutex_lock(&cache->lock);
  gboolean result = g_hash_table_lookup(cache->entries, GINT_TO_POINTER(hash)) != NULL;
  dt_pthread_mutex_unlock(&cache->lock);

  return result;
}


void _age_cache_entry(gpointer key, gpointer value, gpointer user_data)
{
  dt_pixel_cache_entry_t *cache_entry = (dt_pixel_cache_entry_t *)value;
  cache_entry->age++;
}


int dt_dev_pixelpipe_cache_get(dt_dev_pixelpipe_cache_t *cache, const uint64_t hash,
                               const size_t size, const char *name, const int id,
                               void **data, dt_iop_buffer_dsc_t **dsc)
{
  dt_pthread_mutex_lock(&cache->lock);

  cache->queries++;

  // Age all cache entries
  g_hash_table_foreach(cache->entries, _age_cache_entry, NULL);

  // Find the cache entry for this hash, if any
  dt_pixel_cache_entry_t *cache_entry = (dt_pixel_cache_entry_t *)g_hash_table_lookup(cache->entries, GINT_TO_POINTER(hash));
  gboolean cache_entry_found = (cache_entry != NULL);

  if(cache_entry)
    cache->hits++;
  else
    cache_entry = dt_pixel_cache_new_entry(hash, size, **dsc, name, id, cache);

  dt_pthread_mutex_unlock(&cache->lock);

  if(cache_entry)
  {
    cache_entry->age = 0; // this is the MRU entry
    *data = cache_entry->data;
    *dsc = &cache_entry->dsc;
    dt_pixel_cache_message(cache_entry, (cache_entry_found) ? "found" : "created");

    // This will become the input for the next module, lock it until next module process ends
    dt_dev_pixelpipe_cache_lock_entry_hash(darktable.pixelpipe_cache, hash, TRUE);
    return !cache_entry_found;
  }
  else
  {
    *data = NULL;
    *dsc = NULL;
    dt_print(DT_DEBUG_PIPE, "couldn't allocate new cache entry %lu\n", hash);
    return 1;
  }
}

gboolean _for_each_remove(gpointer key, gpointer value, gpointer user_data)
{
  dt_pixel_cache_entry_t *cache_entry = (dt_pixel_cache_entry_t *)value;
  const int id = GPOINTER_TO_INT(user_data);

  // Returns 1 if the lock is captured by another thread
  // 0 if WE capture the lock, and then need to release it
  gboolean locked = dt_pthread_rwlock_trywrlock(&cache_entry->lock);
  if(!locked) dt_pthread_rwlock_unlock(&cache_entry->lock);
  gboolean used = dt_atomic_get_int(&cache_entry->refcount) > 0;

  return (cache_entry->id == id || id == -1) && !used && !locked;
}

void dt_dev_pixelpipe_cache_flush(dt_dev_pixelpipe_cache_t *cache, const int id)
{
  dt_pthread_mutex_lock(&cache->lock);
  g_hash_table_foreach_remove(cache->entries, _for_each_remove, GINT_TO_POINTER(id));
  cache->current_memory = 0;
  dt_pthread_mutex_unlock(&cache->lock);
}

typedef struct _cache_invalidate_t
{
  void *data;
  size_t size;
} _cache_invalidate_t;


gboolean _cache_invalidate(gpointer key, gpointer value, gpointer user_data)
{
  dt_pixel_cache_entry_t *cache_entry = (dt_pixel_cache_entry_t *)value;
  _cache_invalidate_t *cache_invalidate = (_cache_invalidate_t *)user_data;

  if(cache_entry->data == cache_invalidate->data)
  {
    cache_invalidate->size = cache_entry->size;
    return TRUE; // remove this entry
  }

  return FALSE; // keep this entry
}


void dt_dev_pixelpipe_cache_invalidate(dt_dev_pixelpipe_cache_t *cache, void *data)
{
  _cache_invalidate_t *cache_invalidate = (_cache_invalidate_t *)malloc(sizeof(_cache_invalidate_t));
  cache_invalidate->data = data;
  cache_invalidate->size = 0;

  dt_pthread_mutex_lock(&cache->lock);

  // Remove the cache entry with the given data
  g_hash_table_foreach_remove(cache->entries, _cache_invalidate, cache_invalidate);

  // Update the current memory usage
  cache->current_memory -= cache_invalidate->size;

  dt_pthread_mutex_unlock(&cache->lock);

  free(cache_invalidate);
}

uint64_t dt_dev_pixelpipe_cache_get_hash_data(dt_dev_pixelpipe_cache_t *cache, void *data)
{
  GHashTableIter iter;
  gpointer key, value;
  uint64_t hash = 0;

  dt_pthread_mutex_lock(&cache->lock);
  g_hash_table_iter_init(&iter, cache->entries);
  while(g_hash_table_iter_next(&iter, &key, &value))
  {
    dt_pixel_cache_entry_t *cache_entry = (dt_pixel_cache_entry_t *)value;
    if(cache_entry->data == data)
    {
      hash = cache_entry->hash;
      break;
    }
  }
  dt_pthread_mutex_unlock(&cache->lock);

  return hash;
}


void dt_dev_pixelpipe_cache_lock_entry_data(dt_dev_pixelpipe_cache_t *cache, void *data, gboolean lock)
{
  uint64_t hash = dt_dev_pixelpipe_cache_get_hash_data(cache, data);
  dt_dev_pixelpipe_cache_lock_entry_hash(cache, hash, lock);
}


void dt_dev_pixelpipe_cache_lock_entry_hash(dt_dev_pixelpipe_cache_t *cache, const uint64_t hash, gboolean lock)
{
  dt_pthread_mutex_lock(&cache->lock);
  dt_pixel_cache_entry_t *cache_entry = (dt_pixel_cache_entry_t *)g_hash_table_lookup(cache->entries, GINT_TO_POINTER(hash));
  if(cache_entry)
  {
    if(lock)
    {
      dt_atomic_add_int(&cache_entry->refcount, 1);
      dt_pixel_cache_message(cache_entry, "ref count ++");
    }
    else
    {
      dt_atomic_sub_int(&cache_entry->refcount, 1);
      dt_pixel_cache_message(cache_entry, "ref count --");
    }
  }
  dt_pthread_mutex_unlock(&cache->lock);
}


void dt_dev_pixelpipe_cache_wrlock_entry(dt_dev_pixelpipe_cache_t *cache, const uint64_t hash, gboolean lock)
{
  dt_pthread_mutex_lock(&cache->lock);
  dt_pixel_cache_entry_t *cache_entry = (dt_pixel_cache_entry_t *)g_hash_table_lookup(cache->entries, GINT_TO_POINTER(hash));
  if(cache_entry)
  {
    if(lock)
    {
      dt_pthread_rwlock_wrlock(&cache_entry->lock);
      dt_pixel_cache_message(cache_entry, "write lock");
    }
    else
    {
      dt_pthread_rwlock_unlock(&cache_entry->lock);
      dt_pixel_cache_message(cache_entry, "write unlock");
    }
  }
  dt_pthread_mutex_unlock(&cache->lock);
}

void dt_dev_pixelpipe_cache_rdlock_entry(dt_dev_pixelpipe_cache_t *cache, const uint64_t hash, gboolean lock)
{
  dt_pthread_mutex_lock(&cache->lock);
  dt_pixel_cache_entry_t *cache_entry = (dt_pixel_cache_entry_t *)g_hash_table_lookup(cache->entries, GINT_TO_POINTER(hash));
  if(cache_entry)
  {
    if(lock)
    {
      dt_pthread_rwlock_rdlock(&cache_entry->lock);
      dt_pixel_cache_message(cache_entry, "read lock");
    }
    else
    {
      dt_pthread_rwlock_unlock(&cache_entry->lock);
      dt_pixel_cache_message(cache_entry, "read unlock");
    }
  }
  dt_pthread_mutex_unlock(&cache->lock);
}

void dt_dev_pixelpipe_cache_print(dt_dev_pixelpipe_cache_t *cache)
{
  if(!(darktable.unmuted & DT_DEBUG_PIPE)) return;

  dt_print(DT_DEBUG_PIPE, "[pixelpipe] cache hit rate so far: %.3f%% - size: %lu MiB over %lu MiB\n", 100. * (cache->hits) / (float)cache->queries, cache->current_memory / (1024 * 1024), cache->max_memory / (1024 * 1024));
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
