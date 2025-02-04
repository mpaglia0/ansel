/*
    This file is part of darktable,
    Copyright (C) 2009-2021 darktable developers.

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
#include <assert.h>
#include <glib/gprintf.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include "common/atomic.h"
#include "common/debug.h"
#include "common/history.h"
#include "common/image_cache.h"
#include "common/imageio.h"
#include "common/mipmap_cache.h"
#include "common/opencl.h"
#include "common/tags.h"
#include "control/conf.h"
#include "control/control.h"
#include "control/jobs.h"
#include "develop/blend.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/lightroom.h"
#include "develop/masks.h"
#include "gui/gtk.h"
#include "gui/presets.h"

#define DT_DEV_AVERAGE_DELAY_START 250
#define DT_DEV_PREVIEW_AVERAGE_DELAY_START 50
#define DT_DEV_AVERAGE_DELAY_COUNT 5
#define DT_IOP_ORDER_INFO (darktable.unmuted & DT_DEBUG_IOPORDER)

void dt_dev_init(dt_develop_t *dev, int32_t gui_attached)
{
  memset(dev, 0, sizeof(dt_develop_t));
  dev->gui_module = NULL;
  dev->average_delay = DT_DEV_AVERAGE_DELAY_START;
  dev->preview_average_delay = DT_DEV_PREVIEW_AVERAGE_DELAY_START;
  dt_pthread_mutex_init(&dev->history_mutex, NULL);
  dt_pthread_mutex_init(&dev->pipe_mutex, NULL);
  dev->history_end = 0;
  dev->history = NULL; // empty list

  dev->gui_attached = gui_attached;
  dev->width = -1;
  dev->height = -1;
  dev->exit = 0;

  dt_image_init(&dev->image_storage);
  dev->image_invalid_cnt = 0;
  dev->pipe = dev->preview_pipe = NULL;
  dev->histogram_pre_tonecurve = NULL;
  dev->histogram_pre_levels = NULL;
  dev->forms = NULL;
  dev->form_visible = NULL;
  dev->form_gui = NULL;
  dev->allforms = NULL;
  dev->forms_hash = 0;
  dev->forms_changed = FALSE;

  if(dev->gui_attached)
  {
    dev->pipe = (dt_dev_pixelpipe_t *)malloc(sizeof(dt_dev_pixelpipe_t));
    dev->preview_pipe = (dt_dev_pixelpipe_t *)malloc(sizeof(dt_dev_pixelpipe_t));
    dt_dev_pixelpipe_init(dev->pipe);
    dt_dev_pixelpipe_init_preview(dev->preview_pipe);
    dev->histogram_pre_tonecurve = (uint32_t *)calloc(4 * 256, sizeof(uint32_t));
    dev->histogram_pre_levels = (uint32_t *)calloc(4 * 256, sizeof(uint32_t));

    // FIXME: these are uint32_t, setting to -1 is confusing
    dev->histogram_pre_tonecurve_max = -1;
    dev->histogram_pre_levels_max = -1;
  }

  dev->raw_histogram.buffer = NULL;
  dev->raw_histogram.op = "demosaic";
  dev->raw_histogram.height = 0;
  dev->raw_histogram.width = 0;
  dev->raw_histogram.hash = -1;
  dev->raw_histogram.bpp = 0;

  dev->output_histogram.buffer = NULL;
  dev->output_histogram.op = "colorout";
  dev->output_histogram.width = 0;
  dev->output_histogram.height = 0;
  dev->output_histogram.hash = -1;
  dev->output_histogram.bpp = 0;

  dev->display_histogram.buffer = NULL;
  dev->display_histogram.op = "gamma";
  dev->display_histogram.width = 0;
  dev->display_histogram.height = 0;
  dev->display_histogram.hash = -1;
  dev->display_histogram.bpp = 0;

  dev->auto_save_timeout = 0;
  dev->drawing_timeout = 0;

  dev->iop_instance = 0;
  dev->iop = NULL;
  dev->alliop = NULL;

  dev->allprofile_info = NULL;

  dev->iop_order_version = 0;
  dev->iop_order_list = NULL;

  dev->proxy.exposure.module = NULL;
  dev->proxy.chroma_adaptation = NULL;
  dev->proxy.wb_is_D65 = TRUE; // don't display error messages until we know for sure it's FALSE
  dev->proxy.wb_coeffs[0] = 0.f;

  dev->rawoverexposed.enabled = FALSE;
  dev->rawoverexposed.mode = dt_conf_get_int("darkroom/ui/rawoverexposed/mode");
  dev->rawoverexposed.colorscheme = dt_conf_get_int("darkroom/ui/rawoverexposed/colorscheme");
  dev->rawoverexposed.threshold = dt_conf_get_float("darkroom/ui/rawoverexposed/threshold");

  dev->overexposed.enabled = FALSE;
  dev->overexposed.mode = dt_conf_get_int("darkroom/ui/overexposed/mode");
  dev->overexposed.colorscheme = dt_conf_get_int("darkroom/ui/overexposed/colorscheme");
  dev->overexposed.lower = dt_conf_get_float("darkroom/ui/overexposed/lower");
  dev->overexposed.upper = dt_conf_get_float("darkroom/ui/overexposed/upper");

  dev->iso_12646.enabled = FALSE;

  // Init the mask lock state
  dev->mask_lock = 0;
  dev->darkroom_skip_mouse_events = 0;
}

void dt_dev_cleanup(dt_develop_t *dev)
{
  if(!dev) return;
  // image_cache does not have to be unref'd, this is done outside develop module.

  if(dev->raw_histogram.buffer) dt_free_align(dev->raw_histogram.buffer);
  if(dev->output_histogram.buffer) dt_free_align(dev->output_histogram.buffer);
  if(dev->display_histogram.buffer) dt_free_align(dev->display_histogram.buffer);

  // On dev cleanup, it is expected to force an history save
  if(dev->auto_save_timeout) g_source_remove(dev->auto_save_timeout);
  if(dev->drawing_timeout) g_source_remove(dev->drawing_timeout);

  dev->proxy.chroma_adaptation = NULL;
  dev->proxy.wb_coeffs[0] = 0.f;
  if(dev->pipe)
  {
    dt_dev_pixelpipe_cleanup(dev->pipe);
    free(dev->pipe);
  }
  if(dev->preview_pipe)
  {
    dt_dev_pixelpipe_cleanup(dev->preview_pipe);
    free(dev->preview_pipe);
  }
  while(dev->history)
  {
    dt_dev_free_history_item(((dt_dev_history_item_t *)dev->history->data));
    dev->history = g_list_delete_link(dev->history, dev->history);
  }
  while(dev->iop)
  {
    dt_iop_cleanup_module((dt_iop_module_t *)dev->iop->data);
    free(dev->iop->data);
    dev->iop = g_list_delete_link(dev->iop, dev->iop);
  }
  while(dev->alliop)
  {
    dt_iop_cleanup_module((dt_iop_module_t *)dev->alliop->data);
    free(dev->alliop->data);
    dev->alliop = g_list_delete_link(dev->alliop, dev->alliop);
  }
  g_list_free_full(dev->iop_order_list, free);
  while(dev->allprofile_info)
  {
    dt_ioppr_cleanup_profile_info((dt_iop_order_iccprofile_info_t *)dev->allprofile_info->data);
    dt_free_align(dev->allprofile_info->data);
    dev->allprofile_info = g_list_delete_link(dev->allprofile_info, dev->allprofile_info);
  }
  dt_pthread_mutex_destroy(&dev->history_mutex);
  dt_pthread_mutex_destroy(&dev->pipe_mutex);

  free(dev->histogram_pre_tonecurve);
  free(dev->histogram_pre_levels);

  g_list_free_full(dev->forms, (void (*)(void *))dt_masks_free_form);
  g_list_free_full(dev->allforms, (void (*)(void *))dt_masks_free_form);

  dt_conf_set_int("darkroom/ui/rawoverexposed/mode", dev->rawoverexposed.mode);
  dt_conf_set_int("darkroom/ui/rawoverexposed/colorscheme", dev->rawoverexposed.colorscheme);
  dt_conf_set_float("darkroom/ui/rawoverexposed/threshold", dev->rawoverexposed.threshold);

  dt_conf_set_int("darkroom/ui/overexposed/mode", dev->overexposed.mode);
  dt_conf_set_int("darkroom/ui/overexposed/colorscheme", dev->overexposed.colorscheme);
  dt_conf_set_float("darkroom/ui/overexposed/lower", dev->overexposed.lower);
  dt_conf_set_float("darkroom/ui/overexposed/upper", dev->overexposed.upper);
}

void dt_dev_process_image(dt_develop_t *dev)
{
  if(!dev->gui_attached) return;
  const int err
      = dt_control_add_job_res(darktable.control, dt_dev_process_image_job_create(dev), DT_CTL_WORKER_ZOOM_1);
  if(err) fprintf(stderr, "[dev_process_image] job queue exceeded!\n");
}

void dt_dev_process_preview(dt_develop_t *dev)
{
  if(!dev->gui_attached) return;
  const int err
      = dt_control_add_job_res(darktable.control, dt_dev_process_preview_job_create(dev), DT_CTL_WORKER_ZOOM_FILL);
  if(err) fprintf(stderr, "[dev_process_preview] job queue exceeded!\n");
}

void dt_dev_refresh_ui_images_real(dt_develop_t *dev)
{
  // We need to get the shutdown atomic set to TRUE,
  // which is handled everytime history is changed,
  // including when initing a new pipeline (from scratch or from user history).
  // Benefit is atomics are de-facto thread-safe.
  if(dt_atomic_get_int(&dev->preview_pipe->shutdown) && !dev->preview_pipe->processing)
    dt_dev_process_preview(dev);
  // else : join current pipe

  // When entering darkroom, the GUI will size itself and call the
  // configure() method of the view, which calls dev_configure() below.
  // Problem is the GUI can be glitchy and reconfigure the pipe twice with different sizes,
  // Each one starting a recompute. The shutdown mechanism should take care of stopping
  // an ongoing pipe which output we don't care about anymore.
  // But just in case, always start with the preview pipe, hoping
  // the GUI will have figured out what size it really wants when we start
  // the main preview pipe.
  if(dt_atomic_get_int(&dev->pipe->shutdown) && !dev->pipe->processing)
    dt_dev_process_image(dev);
  // else : join current pipe
}

void _dev_pixelpipe_set_dirty(dt_dev_pixelpipe_t *pipe)
{
  pipe->status = DT_DEV_PIXELPIPE_DIRTY;
}

void dt_dev_pixelpipe_rebuild(dt_develop_t *dev)
{
  if(!dev || !dev->gui_attached || !dev->pipe || !dev->preview_pipe) return;

  dt_times_t start;
  dt_get_times(&start);

  _dev_pixelpipe_set_dirty(dev->pipe);
  _dev_pixelpipe_set_dirty(dev->preview_pipe);

  dev->pipe->changed |= DT_DEV_PIPE_REMOVE;
  dev->preview_pipe->changed |= DT_DEV_PIPE_REMOVE;

  dt_atomic_set_int(&dev->pipe->shutdown, TRUE);
  dt_atomic_set_int(&dev->preview_pipe->shutdown, TRUE);

  dt_show_times(&start, "[dt_dev_invalidate] sending killswitch signal on all pipelines");
}

void dt_dev_pixelpipe_resync_main(dt_develop_t *dev)
{
  if(!dev || !dev->gui_attached || !dev->pipe) return;

  _dev_pixelpipe_set_dirty(dev->pipe);
  dev->pipe->changed |= DT_DEV_PIPE_SYNCH;
  dt_atomic_set_int(&dev->pipe->shutdown, TRUE);
}

void dt_dev_pixelpipe_resync_all(dt_develop_t *dev)
{
  if(!dev || !dev->gui_attached || !dev->pipe || !dev->preview_pipe) return;

  _dev_pixelpipe_set_dirty(dev->preview_pipe);
  dev->preview_pipe->changed |= DT_DEV_PIPE_SYNCH;
  dt_atomic_set_int(&dev->preview_pipe->shutdown, TRUE);

  dt_dev_pixelpipe_resync_main(dev);
}

void dt_dev_invalidate_real(dt_develop_t *dev)
{
  if(!dev || !dev->gui_attached || !dev->pipe) return;

  dt_times_t start;
  dt_get_times(&start);

  _dev_pixelpipe_set_dirty(dev->pipe);
  dev->pipe->changed |= DT_DEV_PIPE_TOP_CHANGED;
  dt_atomic_set_int(&dev->pipe->shutdown, TRUE);

  dt_show_times(&start, "[dt_dev_invalidate] sending killswitch signal on main image pipeline");
}

void dt_dev_invalidate_zoom_real(dt_develop_t *dev)
{
  if(!dev || !dev->gui_attached || !dev->pipe) return;

  dt_times_t start;
  dt_get_times(&start);

  _dev_pixelpipe_set_dirty(dev->pipe);
  dev->pipe->changed |= DT_DEV_PIPE_ZOOMED;
  dt_atomic_set_int(&dev->pipe->shutdown, TRUE);

  dt_show_times(&start, "[dt_dev_invalidate_zoom] sending killswitch signal on main image pipeline");
}

void dt_dev_invalidate_preview_real(dt_develop_t *dev)
{
  if(!dev || !dev->gui_attached || !dev->preview_pipe) return;

  dt_times_t start;
  dt_get_times(&start);

  _dev_pixelpipe_set_dirty(dev->preview_pipe);
  dev->preview_pipe->changed |= DT_DEV_PIPE_TOP_CHANGED;
  dt_atomic_set_int(&dev->preview_pipe->shutdown, TRUE);

  dt_show_times(&start, "[dt_dev_invalidate_preview] sending killswitch signal on preview pipeline");
}

void dt_dev_invalidate_all_real(dt_develop_t *dev)
{
  if(!dev || !dev->gui_attached || !dev->pipe || !dev->preview_pipe) return;

  dt_dev_invalidate(dev);
  dt_dev_invalidate_preview(dev);
}


static void _flag_pipe(dt_dev_pixelpipe_t *pipe, gboolean error)
{
  // If dt_dev_pixelpipe_process() returned with a state int == 1
  // and the shutdown flag is on, it means history commit activated the kill-switch.
  // Any other circomstance returning 1 is a runtime error, flag it invalid.
  if(error && !dt_atomic_get_int(&pipe->shutdown))
    pipe->status = DT_DEV_PIXELPIPE_INVALID;

  // Before calling dt_dev_pixelpipe_process(), we set the status to DT_DEV_PIXELPIPE_UNDEF.
  // If it's still set to this value and we have a backbuf, everything went well.
  else if(pipe->backbuf && pipe->status == DT_DEV_PIXELPIPE_UNDEF)
    pipe->status = DT_DEV_PIXELPIPE_VALID;

  // Otherwise, the main thread will have reset the status to DT_DEV_PIXELPIPE_DIRTY
  // and the pipe->shutdown to TRUE because history has changed in the middle of a process.
  // In that case, do nothing and do another loop
}


void dt_dev_process_preview_job(dt_develop_t *dev)
{
  dt_dev_pixelpipe_t *pipe = dev->preview_pipe;
  pipe->running = 1;

  dt_pthread_mutex_lock(&pipe->busy_mutex);

  // init pixel pipeline for preview.
  // always process the whole downsampled mipf buffer, to allow for fast scrolling and mip4 write-through.
  dt_mipmap_buffer_t buf;
  dt_mipmap_cache_get(darktable.mipmap_cache, &buf, dev->image_storage.id, DT_MIPMAP_F, DT_MIPMAP_BLOCKING, 'r');

  gboolean finish_on_error = (!buf.buf || !buf.width || !buf.height);

  if(!finish_on_error)
  {
    dt_dev_pixelpipe_set_input(pipe, dev, (float *)buf.buf, buf.width, buf.height, buf.iscale);
    dt_print(DT_DEBUG_DEV, "[pixelpipe] Started thumbnail preview recompute at %i×%i px\n", buf.width, buf.height);
  }

  pipe->processing = 1;
  while(!dev->exit && !finish_on_error && (pipe->status == DT_DEV_PIXELPIPE_DIRTY))
  {
    dt_times_t thread_start;
    dt_get_times(&thread_start);

    // We are starting fresh, reset the killswitch signal
    dt_atomic_set_int(&pipe->shutdown, FALSE);

    // this locks dev->history_mutex.
    dt_dev_pixelpipe_change(pipe, dev);

    dt_control_log_busy_enter();
    dt_control_toast_busy_enter();

    // Signal that we are starting
    pipe->status = DT_DEV_PIXELPIPE_UNDEF;

    // OpenCL devices have their own lock, called internally
    // Here we lock the whole stack, including CPU.
    // Processing pipelines in parallel triggers memory contention.
    dt_pthread_mutex_lock(&dev->pipe_mutex);

    dt_times_t start;
    dt_get_times(&start);

    int ret = dt_dev_pixelpipe_process(pipe, dev, 0, 0, pipe->processed_width,
                                       pipe->processed_height, 1.f);

    dt_show_times(&start, "[dev_process_preview] pixel pipeline processing");

    dt_pthread_mutex_unlock(&dev->pipe_mutex);
    dt_control_log_busy_leave();
    dt_control_toast_busy_leave();

    dt_show_times(&thread_start, "[dev_process_preview] pixel pipeline thread");
    dt_dev_average_delay_update(&thread_start, &dev->preview_average_delay);

    _flag_pipe(pipe, ret);

    if(pipe->status == DT_DEV_PIXELPIPE_VALID)
      DT_DEBUG_CONTROL_SIGNAL_RAISE(darktable.signals, DT_SIGNAL_DEVELOP_PREVIEW_PIPE_FINISHED);

    dt_iop_nap(200);
  }
  pipe->processing = 0;

  dt_mipmap_cache_release(darktable.mipmap_cache, &buf);

  dt_pthread_mutex_unlock(&pipe->busy_mutex);

  pipe->running = 0;
  dt_print(DT_DEBUG_DEV, "[pixelpipe] exiting preview pipe thread\n");
  dt_control_queue_redraw();
}


void dt_dev_process_image_job(dt_develop_t *dev)
{
  // -1×-1 px means the dimensions of the main preview in darkroom were not inited yet.
  // 0×0 px is not feasible.
  // Anything lower than 32 px might cause segfaults with blurs and local contrast.
  // When the window size get inited, we will get a new order to recompute with a "zoom_changed" flag.
  // Until then, don't bother computing garbage that will not be reused later.
  if(dev->width < 32 || dev->height < 32) return;

  dt_print(DT_DEBUG_DEV, "[pixelpipe] Started main preview recompute at %i×%i px\n", dev->width, dev->height);

  dt_dev_pixelpipe_t *pipe = dev->pipe;
  pipe->running = 1;

  dt_pthread_mutex_lock(&pipe->busy_mutex);

  dt_mipmap_buffer_t buf;
  dt_mipmap_cache_get(darktable.mipmap_cache, &buf, dev->image_storage.id, DT_MIPMAP_FULL, DT_MIPMAP_BLOCKING, 'r');

  gboolean finish_on_error = (!buf.buf || !buf.width || !buf.height);

  if(!finish_on_error)
    dt_dev_pixelpipe_set_input(pipe, dev, (float *)buf.buf, buf.width, buf.height, 1.0);

  float scale = 1.f, zoom_x = 1.f, zoom_y = 1.f;
  pipe->processing = 1;
  while(!dev->exit && !finish_on_error && (pipe->status == DT_DEV_PIXELPIPE_DIRTY))
  {
    dt_times_t thread_start;
    dt_get_times(&thread_start);

    // We are starting fresh, reset the killswitch signal
    dt_atomic_set_int(&pipe->shutdown, FALSE);

    dt_dev_pixelpipe_change_t pipe_changed = pipe->changed;

    // this locks dev->history_mutex
    dt_dev_pixelpipe_change(pipe, dev);

    // determine scale according to new dimensions
    dt_dev_zoom_t zoom = dt_control_get_dev_zoom();
    int closeup = dt_control_get_dev_closeup();
    zoom_x = dt_control_get_dev_zoom_x();
    zoom_y = dt_control_get_dev_zoom_y();
    // if just changed to an image with a different aspect ratio or
    // altered image orientation, the prior zoom xy could now be beyond
    // the image boundary
    if(pipe_changed != DT_DEV_PIPE_UNCHANGED)
    {
      dt_dev_check_zoom_bounds(dev, &zoom_x, &zoom_y, zoom, closeup, NULL, NULL);
      dt_control_set_dev_zoom_x(zoom_x);
      dt_control_set_dev_zoom_y(zoom_y);
    }

    scale = dt_dev_get_zoom_scale(dev, zoom, 1.0f, 0) * darktable.gui->ppd;
    int window_width = dev->width * darktable.gui->ppd;
    int window_height = dev->height * darktable.gui->ppd;
    if(closeup)
    {
      window_width /= 1<<closeup;
      window_height /= 1<<closeup;
    }
    const int wd = MIN(window_width, pipe->processed_width * scale);
    const int ht = MIN(window_height, pipe->processed_height * scale);
    int x = MAX(0, scale * pipe->processed_width  * (.5 + zoom_x) - wd / 2);
    int y = MAX(0, scale * pipe->processed_height * (.5 + zoom_y) - ht / 2);

    dt_control_log_busy_enter();
    dt_control_toast_busy_enter();

    // Signal that we are starting
    pipe->status = DT_DEV_PIXELPIPE_UNDEF;

    // OpenCL devices have their own lock, called internally
    // Here we lock the whole stack, including CPU.
    // Processing pipelines in parallel triggers memory contention.
    dt_pthread_mutex_lock(&dev->pipe_mutex);

    dt_times_t start;
    dt_get_times(&start);

    int ret = dt_dev_pixelpipe_process(pipe, dev, x, y, wd, ht, scale);

    dt_show_times(&start, "[dev_process_image] pixel pipeline processing");

    dt_pthread_mutex_unlock(&dev->pipe_mutex);

    dt_control_log_busy_leave();
    dt_control_toast_busy_leave();

    dt_show_times(&thread_start, "[dev_process_image] pixel pipeline thread");
    dt_dev_average_delay_update(&thread_start, &dev->average_delay);

    _flag_pipe(pipe, ret);

    // cool, we got a new image!
    if(pipe->status == DT_DEV_PIXELPIPE_VALID)
    {
      pipe->backbuf_scale = scale;
      pipe->backbuf_zoom_x = zoom_x;
      pipe->backbuf_zoom_y = zoom_y;
      dev->image_invalid_cnt = 0;
    }

    if(pipe->status == DT_DEV_PIXELPIPE_VALID)
      DT_DEBUG_CONTROL_SIGNAL_RAISE(darktable.signals, DT_SIGNAL_DEVELOP_UI_PIPE_FINISHED);

    dt_iop_nap(200);
  }
  pipe->processing = 0;

  dt_mipmap_cache_release(darktable.mipmap_cache, &buf);

  dt_pthread_mutex_unlock(&pipe->busy_mutex);

  pipe->running = 0;
  dt_print(DT_DEBUG_DEV, "[pixelpipe] exiting main image pipe thread\n");
  dt_control_queue_redraw_center();
}


static inline void _dt_dev_load_pipeline_defaults(dt_develop_t *dev)
{
  for(const GList *modules = g_list_first(dev->iop); modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *module = (dt_iop_module_t *)(modules->data);
    dt_iop_reload_defaults(module);
  }
}

// load the raw and get the new image struct, blocking in gui thread
static inline int _dt_dev_load_raw(dt_develop_t *dev, const uint32_t imgid)
{
  // first load the raw, to make sure dt_image_t will contain all and correct data.
  dt_times_t start;
  dt_get_times(&start);

  // Test we got images. Also that populates the cache for later.
  dt_mipmap_buffer_t buf;
  dt_mipmap_cache_get(darktable.mipmap_cache, &buf, imgid, DT_MIPMAP_FULL, DT_MIPMAP_BLOCKING, 'r');
  gboolean no_valid_image = buf.buf == NULL;
  dt_mipmap_cache_release(darktable.mipmap_cache, &buf);

  dt_mipmap_cache_get(darktable.mipmap_cache, &buf, imgid, DT_MIPMAP_F, DT_MIPMAP_BLOCKING, 'r');
  gboolean no_valid_thumb = buf.buf == NULL;
  dt_mipmap_cache_release(darktable.mipmap_cache, &buf);

  dt_show_times_f(&start, "[dev]", "to load the image.");

  const dt_image_t *image = dt_image_cache_get(darktable.image_cache, imgid, 'r');
  dev->image_storage = *image;
  dt_image_cache_read_release(darktable.image_cache, image);

  return (no_valid_image || no_valid_thumb);
}

float dt_dev_get_zoom_scale(dt_develop_t *dev, dt_dev_zoom_t zoom, int closeup_factor, int preview)
{
  float zoom_scale;

  const float w = preview ? dev->preview_pipe->processed_width : dev->pipe->processed_width;
  const float h = preview ? dev->preview_pipe->processed_height : dev->pipe->processed_height;
  const float ps = dev->pipe->backbuf_width
                       ? dev->pipe->processed_width / (float)dev->preview_pipe->processed_width
                       : dev->preview_pipe->iscale;

  switch(zoom)
  {
    case DT_ZOOM_FIT:
      zoom_scale = fminf(dev->width / w, dev->height / h);
      break;
    case DT_ZOOM_FILL:
      zoom_scale = fmaxf(dev->width / w, dev->height / h);
      break;
    case DT_ZOOM_1:
      zoom_scale = closeup_factor;
      if(preview) zoom_scale *= ps;
      break;
    default: // DT_ZOOM_FREE
      zoom_scale = dt_control_get_dev_zoom_scale();
      if(preview) zoom_scale *= ps;
      break;
  }

  return zoom_scale;
}

int dt_dev_load_image(dt_develop_t *dev, const uint32_t imgid)
{
  if(_dt_dev_load_raw(dev, imgid)) return 1;

  // we need a global lock as the dev->iop set must not be changed until read history is terminated
  dt_pthread_mutex_lock(&dev->history_mutex);
  dev->iop = dt_iop_load_modules(dev);

  dt_dev_read_history_ext(dev, dev->image_storage.id, FALSE);

  if(dev->pipe)
  {
    dev->pipe->processed_width = 0;
    dev->pipe->processed_height = 0;
  }
  if(dev->preview_pipe)
  {
    dev->preview_pipe->processed_width = 0;
    dev->preview_pipe->processed_height = 0;
  }
  dt_pthread_mutex_unlock(&dev->history_mutex);

  dt_dev_pixelpipe_rebuild(dev);

  return 0;
}

void dt_dev_configure_real(dt_develop_t *dev, int wd, int ht)
{
  // Called only from Darkroom to init and update drawing size
  // depending on sidebars and main window resizing.
  if(dev->width != wd || dev->height != ht || !dev->pipe->backbuf)
  {
    // If dimensions didn't change or we don't have a valid output image to display

    dev->width = wd;
    dev->height = ht;

    dt_print(DT_DEBUG_DEV, "[pixelpipe] Darkroom requested a %i×%i px main preview\n", wd, ht);
    dt_dev_invalidate_zoom(dev);

    if(dev->image_storage.id > -1 && darktable.mipmap_cache)
    {
      // Only if it's not our initial configure call, aka if we already have an image
      dt_control_queue_redraw_center();
      dt_dev_refresh_ui_images(dev);
    }
  }
}

// helper used to synch a single history item with db
int dt_dev_write_history_item(const int imgid, dt_dev_history_item_t *h, int32_t num)
{
  dt_print(DT_DEBUG_HISTORY, "[dt_dev_write_history_item] writing history for module %s (%s) at pipe position %i for image %i...\n", h->op_name, h->multi_name, h->iop_order, imgid);

  sqlite3_stmt *stmt;
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                              "SELECT num FROM main.history WHERE imgid = ?1 AND num = ?2", -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, imgid);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, num);
  if(sqlite3_step(stmt) != SQLITE_ROW)
  {
    sqlite3_finalize(stmt);
    DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                                "INSERT INTO main.history (imgid, num) VALUES (?1, ?2)", -1, &stmt, NULL);
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, imgid);
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, num);
    sqlite3_step(stmt);
  }
  // printf("[dev write history item] writing %d - %s params %f %f\n", h->module->instance, h->module->op,
  // *(float *)h->params, *(((float *)h->params)+1));
  sqlite3_finalize(stmt);
  // clang-format off
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                              "UPDATE main.history"
                              " SET operation = ?1, op_params = ?2, module = ?3, enabled = ?4, "
                              "     blendop_params = ?7, blendop_version = ?8, multi_priority = ?9, multi_name = ?10"
                              " WHERE imgid = ?5 AND num = ?6",
                              -1, &stmt, NULL);
  // clang-format on
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, h->module->op, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_BLOB(stmt, 2, h->params, h->module->params_size, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 3, h->module->version());
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 4, h->enabled);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 5, imgid);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 6, num);
  DT_DEBUG_SQLITE3_BIND_BLOB(stmt, 7, h->blend_params, sizeof(dt_develop_blend_params_t), SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 8, dt_develop_blend_version());
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 9, h->multi_priority);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 10, h->multi_name, -1, SQLITE_TRANSIENT);

  sqlite3_step(stmt);
  sqlite3_finalize(stmt);

  // write masks (if any)
  if(h->forms)
    dt_print(DT_DEBUG_HISTORY, "[dt_dev_write_history_item] drawn mask found for module %s (%s) for image %i\n", h->op_name, h->multi_name, imgid);

  for(GList *forms = g_list_first(h->forms); forms; forms = g_list_next(forms))
  {
    dt_masks_form_t *form = (dt_masks_form_t *)forms->data;
    if (form)
      dt_masks_write_masks_history_item(imgid, num, form);
  }

  return 0;
}


static dt_iop_module_t * _find_mask_manager(dt_develop_t *dev)
{
  for(GList *module = g_list_first(dev->iop); module; module = g_list_next(module))
  {
    dt_iop_module_t *mod = (dt_iop_module_t *)(module->data);
    if(strcmp(mod->op, "mask_manager") == 0)
      return mod;
  }
  return NULL;
}


gboolean dt_dev_add_history_item_ext(dt_develop_t *dev, struct dt_iop_module_t *module, gboolean enable,
                                     gboolean force_new_item, gboolean no_image, gboolean include_masks)
{
  // If this history item is the first for this module,
  // we need to notify the pipeline that its topology may change (aka insert a new node).
  // Since changing topology is expensive, we want to do it only when needed.
  gboolean add_new_pipe_node = FALSE;

  if(!module)
  {
    // module = NULL means a mask was changed from the mask manager
    // and that's where this function is called.
    // Find it now, even though it is not enabled and won't be.
    module = _find_mask_manager(dev);
    if(module)
    {
      // Mask manager is an IOP that never processes pixel
      // aka it's an ugly hack to record mask history
      force_new_item = FALSE;
      enable = FALSE;
    }
  }

  if(!module) return add_new_pipe_node;

  // TODO: this is probably a redundant call
  dt_iop_compute_blendop_hash(module);
  dt_iop_compute_module_hash(module);

  // look for leaks on top of history in two steps
  // first remove obsolete items above history_end
  // but keep the always-on modules
  for(GList *history = g_list_nth(dev->history, dt_dev_get_history_end(dev));
      history;
      history = g_list_next(history))
  {
    dt_dev_history_item_t *hist = (dt_dev_history_item_t *)(history->data);
    dt_print(DT_DEBUG_HISTORY, "[dt_dev_add_history_item_ext] history item %s at %i is past history limit (%i)\n", hist->module->op, g_list_index(dev->history, hist), dt_dev_get_history_end(dev) - 1);

    // Check if an earlier instance of the module exists.
    gboolean earlier_entry = FALSE;
    for(GList *prior_history = g_list_previous(history);
        prior_history && earlier_entry == FALSE;
        prior_history = g_list_previous(prior_history))
    {
      dt_dev_history_item_t *prior_hist = (dt_dev_history_item_t *)(prior_history->data);
      if(prior_hist->module->so == hist->module->so)
        earlier_entry = TRUE;
    }

    if((!hist->module->hide_enable_button && !hist->module->default_enabled)
        || earlier_entry)
    {
      dt_print(DT_DEBUG_HISTORY, "[dt_dev_add_history_item_ext] removing obsoleted history item: %s at %i\n", hist->module->op, g_list_index(dev->history, hist));
      dt_dev_free_history_item(hist);
      dev->history = g_list_delete_link(dev->history, history);
    }
    else
    {
      dt_print(DT_DEBUG_HISTORY, "[dt_dev_add_history_item_ext] obsoleted history item will be kept: %s at %i\n", hist->module->op, g_list_index(dev->history, hist));
    }
  }

  // Check if the current module to append to history is actually the same as the last one in history,
  // and the internal parameters (aka module hash) are the same.
  GList *last = g_list_last(dev->history);
  gboolean new_is_old = FALSE;
  if(last && last->data)
  {
    dt_dev_history_item_t *last_item = (dt_dev_history_item_t *)last->data;
    dt_iop_module_t *last_module = last_item->module;
    // fprintf(stdout, "history has hash %lu, new module %s has %lu\n", last_item->hash, module->op, module->hash);
    new_is_old = dt_iop_check_modules_equal(module, last_module);
    // add_new_pipe_node = FALSE
  }
  else
  {
    const dt_dev_history_item_t *previous_item = dt_dev_get_history_item(dev, module);
    // check if NULL first or prevous_item->module will segfault
    // We need to add a new pipeline node if:
    add_new_pipe_node = (previous_item == NULL)                         // it's the first history entry for this module
                        || (previous_item->enabled != module->enabled); // the previous history entry is disabled
    // if previous history entry is disabled and we don't have any other entry,
    // it is possible the pipeline will not have this node.
  }

  dt_dev_history_item_t *hist;
  uint32_t position = 0; // used only for debug strings
  if(force_new_item || !new_is_old)
  {
    // Create a new history entry
    hist = (dt_dev_history_item_t *)calloc(1, sizeof(dt_dev_history_item_t));
    hist->params = malloc(module->params_size);
    hist->blend_params = malloc(sizeof(dt_develop_blend_params_t));
    hist->num = -1;
    dev->history = g_list_append(dev->history, hist);
    position = g_list_index(dev->history, hist);
    dt_print(DT_DEBUG_HISTORY, "[dt_dev_add_history_item_ext] new history entry added for %s at position %i\n",
             module->name(), position);
  }
  else
  {
    // Reuse previous history entry
    hist = (dt_dev_history_item_t *)last->data;
    // Drawn masks are forced-resync later, free them now
    if(hist->forms) g_list_free_full(hist->forms, (void (*)(void *))dt_masks_free_form);
    position = g_list_index(dev->history, hist);
    dt_print(DT_DEBUG_HISTORY, "[dt_dev_add_history_item_ext] history entry reused for %s at position %i\n",
             module->name(), position);
  }

  // Always resync history with all module internals
  hist->module = module;
  hist->iop_order = module->iop_order;
  hist->multi_priority = module->multi_priority;
  g_strlcpy(hist->op_name, module->op, sizeof(hist->op_name));
  g_strlcpy(hist->multi_name, module->multi_name, sizeof(hist->multi_name));
  memcpy(hist->params, module->params, module->params_size);
  memcpy(hist->blend_params, module->blend_params, sizeof(dt_develop_blend_params_t));

  // Include masks if module supports blending and blending is on or if it's the mask manager
  include_masks = ((module->flags() & IOP_FLAGS_SUPPORTS_BLENDING) == IOP_FLAGS_SUPPORTS_BLENDING
                   && module->blend_params->mask_mode > DEVELOP_MASK_ENABLED)
                  || (module->flags() & IOP_FLAGS_INTERNAL_MASKS) == IOP_FLAGS_INTERNAL_MASKS;

  if(include_masks)
  {
    dt_print(DT_DEBUG_HISTORY, "[dt_dev_add_history_item_ext] committing masks for module %s at history position %i\n", module->name(), position);
    // FIXME: this copies ALL drawn masks AND masks groups used by all modules to any module history using masks.
    // Kudos to the idiots who thought it would be reasonable. Expect database bloating and perf penalty.
    hist->forms = dt_masks_dup_forms_deep(dev->forms, NULL);
    dev->forms_changed = FALSE; // reset
  }
  else
  {
    hist->forms = NULL;
  }

  if(include_masks && hist->forms)
    dt_print(DT_DEBUG_HISTORY, "[dt_dev_add_history_item_ext] masks committed for module %s at history position %i\n", module->name(), position);
  else if(include_masks)
    dt_print(DT_DEBUG_HISTORY, "[dt_dev_add_history_item_ext] masks NOT committed for module %s at history position %i\n", module->name(), position);

  if(enable) module->enabled = TRUE;
  hist->enabled = module->enabled;
  hist->hash = module->hash;

  // It is assumed that the last-added history entry is always on top
  // so its cursor index is always equal to the number of elements,
  // keeping in mind that history_end = 0 is the raw image, aka not a dev->history GList entry.
  // So dev->history_end = index of last history entry + 1 = length of history
  dt_dev_set_history_end(dev, g_list_length(dev->history));

  return add_new_pipe_node;
}

const dt_dev_history_item_t *dt_dev_get_history_item(dt_develop_t *dev, struct dt_iop_module_t *module)
{
  for(GList *l = g_list_last(dev->history); l; l = g_list_previous(l))
  {
    dt_dev_history_item_t *item = (dt_dev_history_item_t *)l->data;
    if(item->module == module)
      return item;
  }
  return NULL;
}


#define AUTO_SAVE_TIMEOUT 30000

static int _auto_save_edit(gpointer data)
{
  // TODO: put that in a parallel job to not freeze GUI mainthread
  // when writing XMP on remote storage ? But that will still lock history mutex...
  dt_develop_t *dev = (dt_develop_t *)data;
  dev->auto_save_timeout = 0;

  dt_times_t start;
  dt_get_times(&start);
  dt_toast_log(_("autosaving changes..."));

  dt_pthread_mutex_lock(&dev->history_mutex);
  dt_dev_write_history_ext(dev->history, dev->iop_order_list, dev->image_storage.id);
  dt_dev_write_history_end_ext(dt_dev_get_history_end(dev), dev->image_storage.id);
  dt_pthread_mutex_unlock(&dev->history_mutex);

  dt_control_save_xmp(dev->image_storage.id);

  dt_show_times(&start, "[_auto_save_edit] auto-saving history upon last change");

  dt_times_t end;
  dt_get_times(&end);
  dt_toast_log("autosaving completed in %.3f s", end.clock - start.clock);
  return G_SOURCE_REMOVE;
}


// The next 2 functions are always called from GUI controls setting parameters
// This is why they directly start a pipeline recompute.
// Otherwise, please keep GUI and pipeline fully separated.

void dt_dev_add_history_item_real(dt_develop_t *dev, dt_iop_module_t *module, gboolean enable)
{
  dt_atomic_set_int(&dev->pipe->shutdown, TRUE);
  dt_atomic_set_int(&dev->preview_pipe->shutdown, TRUE);

  dt_dev_undo_start_record(dev);

  // Run the delayed post-commit actions if implemented
  if(module && module->post_history_commit) module->post_history_commit(module);

  dt_pthread_mutex_lock(&dev->history_mutex);
  dt_dev_add_history_item_ext(dev, module, enable, FALSE, FALSE, FALSE);
  dt_pthread_mutex_unlock(&dev->history_mutex);

  /* signal that history has changed */
  dt_dev_undo_end_record(dev);

  // Figure out if the current history item includes masks/forms
  GList *last_history = g_list_nth(dev->history, dt_dev_get_history_end(dev) - 1);
  dt_dev_history_item_t *hist = NULL;
  gboolean has_forms = FALSE;
  if(last_history)
  {
    hist = (dt_dev_history_item_t *)last_history->data;
    has_forms = (hist->forms != NULL);
  }

  // Recompute pipeline last
  if(module && !has_forms)
  {
    // If we have a module and it doesn't have masks, we only need to resync the top-most history item with pipeline
    dt_dev_invalidate_all(dev);
  }
  else
  {
    // We either don't have a module, meaning we have the mask manager, or
    // we have a module and it has masks. Both ways, masks can affect several modules anywhere.
    // We need a full resync of all pipeline with history.
    dt_dev_pixelpipe_resync_all(dev);
  }

  dt_dev_masks_list_update(dev);
  dt_dev_refresh_ui_images(dev);

  if(darktable.gui && dev->gui_attached)
  {
    if(module) dt_iop_gui_set_enable_button(module);

    // Auto-save N s after the last change.
    // If another change is made during that delay,
    // reset the timer and restart Ns
    if(dev->auto_save_timeout)
    {
      g_source_remove(dev->auto_save_timeout);
      dev->auto_save_timeout = 0;
    }
    dev->auto_save_timeout = g_timeout_add(AUTO_SAVE_TIMEOUT, _auto_save_edit, dev);
  }
}

void dt_dev_free_history_item(gpointer data)
{
  dt_dev_history_item_t *item = (dt_dev_history_item_t *)data;
  free(item->params);
  free(item->blend_params);
  g_list_free_full(item->forms, (void (*)(void *))dt_masks_free_form);
  free(item);
}

void dt_dev_reload_history_items(dt_develop_t *dev)
{
  // Recreate the whole history from scratch
  dt_pthread_mutex_lock(&dev->history_mutex);
  g_list_free_full(dev->history, dt_dev_free_history_item);
  dev->history = NULL;
  dt_dev_read_history_ext(dev, dev->image_storage.id, FALSE);
  dt_pthread_mutex_unlock(&dev->history_mutex);

  // we have to add new module instances first
  for(GList *modules = g_list_first(dev->iop); modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *module = (dt_iop_module_t *)(modules->data);
    if(module->multi_priority > 0)
    {
      if(!dt_iop_is_hidden(module) && !module->expander)
      {
        dt_iop_gui_init(module);

        /* add module to right panel */
        dt_iop_gui_set_expander(module);
        dt_iop_gui_set_expanded(module, TRUE, FALSE);

        dt_iop_reload_defaults(module);
        dt_iop_gui_update_blending(module);
      }
    }
    else if(!dt_iop_is_hidden(module) && module->expander)
    {
      // we have to ensure that the name of the widget is correct
      dt_iop_gui_update_header(module);
    }
  }

  dt_dev_pop_history_items(dev, dt_dev_get_history_end(dev));

  dt_ioppr_resync_iop_list(dev);

  // set the module list order
  dt_dev_reorder_gui_module_list(dev);

  // we update show params for multi-instances for each other instances
  dt_dev_modules_update_multishow(dev);

  dt_dev_pixelpipe_rebuild(dev);
}

static inline void _dt_dev_modules_reload_defaults(dt_develop_t *dev)
{
  // The reason for this is modules mandatorily ON don't leave history.
  // We therefore need to init modules containers to the defaults.
  // This is of course shit because it relies on the fact that defaults will never change in the future.
  // As a result, changing defaults needs to be handled on a case-by-case basis, by first adding the affected
  // modules to history, then setting said history to the previous defaults.
  // Worse, some modules (temperature.c) grabbed their params at runtime (WB as shot in camera),
  // meaning the defaults were not even static values.
  for(GList *modules = g_list_first(dev->iop); modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *module = (dt_iop_module_t *)(modules->data);
    memcpy(module->params, module->default_params, module->params_size);
    dt_iop_commit_blend_params(module, module->default_blendop_params);
    module->enabled = module->default_enabled;

    if(module->multi_priority == 0)
      module->iop_order = dt_ioppr_get_iop_order(dev->iop_order_list, module->op, module->multi_priority);
    else
      module->iop_order = INT_MAX;
  }
}


void dt_dev_pop_history_items_ext(dt_develop_t *dev, int32_t cnt)
{
  dt_print(DT_DEBUG_HISTORY, "[dt_dev_pop_history_items_ext] loading history entries into modules...\n");

  dt_dev_set_history_end(dev, cnt);

  // reset gui params for all modules
  _dt_dev_modules_reload_defaults(dev);

  // go through history and set modules params
  GList *forms = NULL;
  GList *history = g_list_first(dev->history);
  for(int i = 0; i < cnt && history; i++)
  {
    dt_dev_history_item_t *hist = (dt_dev_history_item_t *)(history->data);
    memcpy(hist->module->params, hist->params, hist->module->params_size);
    dt_iop_commit_blend_params(hist->module, hist->blend_params);

    hist->module->iop_order = hist->iop_order;
    hist->module->enabled = hist->enabled;
    hist->module->multi_priority = hist->multi_priority;
    g_strlcpy(hist->module->multi_name, hist->multi_name, sizeof(hist->module->multi_name));

    // This needs to run after dt_iop_compute_blendop_hash()
    // which is called in dt_iop_commit_blend_params
    dt_iop_compute_module_hash(hist->module);
    hist->hash = hist->module->hash;

    if(hist->forms) forms = hist->forms;

    history = g_list_next(history);
  }

  dt_ioppr_resync_modules_order(dev);

  dt_ioppr_check_duplicate_iop_order(&dev->iop, dev->history);

  dt_ioppr_check_iop_order(dev, 0, "dt_dev_pop_history_items_ext end");

  dt_masks_replace_current_forms(dev, forms);
}

void dt_dev_pop_history_items(dt_develop_t *dev, int32_t cnt)
{
  ++darktable.gui->reset;

  dt_pthread_mutex_lock(&dev->history_mutex);
  dt_ioppr_check_iop_order(dev, 0, "dt_dev_pop_history_items");
  dt_dev_pop_history_items_ext(dev, cnt);
  dt_pthread_mutex_unlock(&dev->history_mutex);

  // update all gui modules
  for(GList *module = g_list_first(dev->iop); module; module = g_list_next(module))
  {
    dt_iop_module_t *mod = (dt_iop_module_t *)(module->data);
    dt_iop_gui_update(mod);
  }
  --darktable.gui->reset;

  dt_dev_masks_list_change(dev);
  dt_dev_pixelpipe_rebuild(dev);
  dt_dev_refresh_ui_images(dev);
}

static void _cleanup_history(const int imgid)
{
  sqlite3_stmt *stmt;
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db), "DELETE FROM main.history WHERE imgid = ?1", -1,
                              &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, imgid);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);

  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db), "DELETE FROM main.masks_history WHERE imgid = ?1", -1,
                              &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, imgid);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}

guint dt_dev_mask_history_overload(GList *dev_history, guint threshold)
{
  // Count all the mask forms used x history entries, up to a certain threshold.
  // Stop counting when the threshold is reached, for performance.
  guint states = 0;
  for(GList *history = g_list_first(dev_history); history; history = g_list_next(history))
  {
    dt_dev_history_item_t *hist_item = (dt_dev_history_item_t *)(history->data);
    states += g_list_length(hist_item->forms);
    if(states > threshold) break;
  }
  return states;
}

static void _warn_about_history_overuse(GList *dev_history)
{
  /* History stores one entry per module, everytime a parameter is changed.
  *  For modules using masks, we also store a full snapshot of masks states.
  *  All that is saved into database and XMP. When history entries x number of mask > 250,
  *  we get a really bad performance penalty.
  */
  guint states = dt_dev_mask_history_overload(dev_history, 250);

  if(states > 250)
    dt_toast_log(_("Your history is storing %d mask states. To ensure smooth operation, consider compressing "
                   "history and removing unused masks."),
                 states);
}


void dt_dev_write_history_end_ext(const int history_end, const int imgid)
{
  // update history end
  sqlite3_stmt *stmt;
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                              "UPDATE main.images SET history_end = ?1 WHERE id = ?2", -1,
                              &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, history_end);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, imgid);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}


void dt_dev_write_history_ext(GList *dev_history, GList *iop_order_list, const int imgid)
{
  _cleanup_history(imgid);
  _warn_about_history_overuse(dev_history);

  dt_print(DT_DEBUG_HISTORY, "[dt_dev_write_history_ext] writing history for image %i...\n", imgid);

  // write history entries
  int i = 0;
  for(GList *history = g_list_first(dev_history); history; history = g_list_next(history))
  {
    dt_dev_history_item_t *hist = (dt_dev_history_item_t *)(history->data);
    dt_dev_write_history_item(imgid, hist, i);
    i++;
  }

  // write the current iop-order-list for this image
  dt_ioppr_write_iop_order_list(iop_order_list, imgid);
  dt_history_hash_write_from_history(imgid, DT_HISTORY_HASH_CURRENT);
}

void dt_dev_write_history(dt_develop_t *dev)
{
  // FIXME: [CRITICAL] should lock the image history at the app level
  dt_pthread_mutex_lock(&dev->history_mutex);
  dt_dev_write_history_ext(dev->history, dev->iop_order_list, dev->image_storage.id);
  dt_dev_write_history_end_ext(dt_dev_get_history_end(dev), dev->image_storage.id);
  dt_pthread_mutex_unlock(&dev->history_mutex);
}

static int _dev_get_module_nb_records()
{
  sqlite3_stmt *stmt;
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                              "SELECT count (*) FROM  memory.history",
                              -1, &stmt, NULL);
  sqlite3_step(stmt);
  const int cnt = sqlite3_column_int(stmt, 0);
  sqlite3_finalize(stmt);
  return cnt;
}

void _dev_insert_module(dt_develop_t *dev, dt_iop_module_t *module, const int imgid)
{
  sqlite3_stmt *stmt;

  DT_DEBUG_SQLITE3_PREPARE_V2(
    dt_database_get(darktable.db),
    "INSERT INTO memory.history VALUES (?1, 0, ?2, ?3, ?4, 1, NULL, 0, 0, '')",
    -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, imgid);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, module->version());
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 3, module->op, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_BLOB(stmt, 4, module->default_params, module->params_size, SQLITE_TRANSIENT);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);

  dt_print(DT_DEBUG_PARAMS, "[history] module %s inserted to history\n", module->op);
}

static gboolean _dev_auto_apply_presets(dt_develop_t *dev)
{
  // NOTE: the presets/default iops will be *prepended* into the history.

  const int imgid = dev->image_storage.id;

  if(imgid <= 0) return FALSE;

  gboolean run = FALSE;
  dt_image_t *image = dt_image_cache_get(darktable.image_cache, imgid, 'w');
  if(!(image->flags & DT_IMAGE_AUTO_PRESETS_APPLIED)) run = TRUE;

  const gboolean is_raw = dt_image_is_raw(image);

  // Force-reload modern chromatic adaptation
  // Will be overriden below if we have no history for temperature
  dt_conf_set_string("plugins/darkroom/chromatic-adaptation", "modern");

  // flag was already set? only apply presets once in the lifetime of a history stack.
  // (the flag will be cleared when removing it).
  if(!run || image->id <= 0)
  {
    // Next section is to recover old edits where all modules with default parameters were not
    // recorded in the db nor in the .XMP.
    //
    // One crucial point is the white-balance which has automatic default based on the camera
    // and depends on the chroma-adaptation. In modern mode the default won't be the same used
    // in legacy mode and if the white-balance is not found on the history one will be added by
    // default using current defaults. But if we are in modern chromatic adaptation the default
    // will not be equivalent to the one used to develop this old edit.

    // So if the current mode is the modern chromatic-adaptation, do check the history.

    if(is_raw)
    {
      // loop over all modules and display a message for default-enabled modules that
      // are not found on the history.

      for(GList *modules = g_list_first(dev->iop); modules; modules = g_list_next(modules))
      {
        dt_iop_module_t *module = (dt_iop_module_t *)modules->data;

        if(module->default_enabled
           && !(module->flags() & IOP_FLAGS_NO_HISTORY_STACK)
           && !dt_history_check_module_exists(imgid, module->op, FALSE))
        {
          fprintf(stderr,
                  "[_dev_auto_apply_presets] missing mandatory module %s for image %d\n",
                  module->op, imgid);

          // If the module is white-balance and we are dealing with a raw file we need to add
          // one now with the default legacy parameters. And we want to do this only for
          // old edits.
          //
          // For new edits the temperature will be added back depending on the chromatic
          // adaptation the standard way.

          if(!strcmp(module->op, "temperature")
             && (image->change_timestamp == -1))
          {
            // it is important to recover temperature in this case (modern chroma and
            // not module present as we need to have the pre 3.0 default parameters used.

            dt_conf_set_string("plugins/darkroom/chromatic-adaptation", "legacy");
            dt_iop_reload_defaults(module);
            _dev_insert_module(dev, module, imgid);
            dt_conf_set_string("plugins/darkroom/chromatic-adaptation", "modern");
            dt_iop_reload_defaults(module);
          }
        }
      }
    }

    dt_image_cache_write_release(darktable.image_cache, image, DT_IMAGE_CACHE_RELAXED);
    return FALSE;
  }

  //  Add scene-referred workflow
  //  Note that we cannot use a preset for FilmicRGB as the default values are
  //  dynamically computed depending on the actual exposure compensation
  //  (see reload_default routine in filmicrgb.c)

  const gboolean has_matrix = dt_image_is_matrix_correction_supported(image);

  if(is_raw)
  {
    for(GList *modules = dev->iop; modules; modules = g_list_next(modules))
    {
      dt_iop_module_t *module = (dt_iop_module_t *)modules->data;

      if((   (strcmp(module->op, "filmicrgb") == 0)
          || (strcmp(module->op, "colorbalancergb") == 0)
          || (strcmp(module->op, "lens") == 0)
          || (has_matrix && strcmp(module->op, "channelmixerrgb") == 0) )
         && !dt_history_check_module_exists(imgid, module->op, FALSE)
         && !(module->flags() & IOP_FLAGS_NO_HISTORY_STACK))
      {
        _dev_insert_module(dev, module, imgid);
      }
    }
  }

  // FIXME : the following query seems duplicated from gui/presets.c/dt_gui_presets_autoapply_for_module()

  // select all presets from one of the following table and add them into memory.history. Note that
  // this is appended to possibly already present default modules.
  const char *preset_table[2] = { "data.presets", "main.legacy_presets" };
  const int legacy = (image->flags & DT_IMAGE_NO_LEGACY_PRESETS) ? 0 : 1;
  char query[1024];
  // clang-format off
  snprintf(query, sizeof(query),
           "INSERT INTO memory.history"
           " SELECT ?1, 0, op_version, operation, op_params,"
           "       enabled, blendop_params, blendop_version, multi_priority, multi_name"
           " FROM %s"
           " WHERE ( (autoapply=1"
           "          AND ((?2 LIKE model AND ?3 LIKE maker) OR (?4 LIKE model AND ?5 LIKE maker))"
           "          AND ?6 LIKE lens AND ?7 BETWEEN iso_min AND iso_max"
           "          AND ?8 BETWEEN exposure_min AND exposure_max"
           "          AND ?9 BETWEEN aperture_min AND aperture_max"
           "          AND ?10 BETWEEN focal_length_min AND focal_length_max"
           "          AND (format = 0 OR (format&?11 != 0 AND ~format&?12 != 0)))"
           "        OR (name = ?13))"
           "   AND operation NOT IN"
           "        ('ioporder', 'metadata', 'modulegroups', 'export', 'tagging', 'collect', 'basecurve')"
           " ORDER BY writeprotect DESC, LENGTH(model), LENGTH(maker), LENGTH(lens)",
           preset_table[legacy]);
  // clang-format on
  // query for all modules at once:
  sqlite3_stmt *stmt;
  const char *workflow_preset = has_matrix ? _("scene-referred default") : "\t\n";
  int iformat = 0;
  if(dt_image_is_rawprepare_supported(image)) iformat |= FOR_RAW;
  else iformat |= FOR_LDR;
  if(dt_image_is_hdr(image)) iformat |= FOR_HDR;

  int excluded = 0;
  if(dt_image_monochrome_flags(image)) excluded |= FOR_NOT_MONO;
  else excluded |= FOR_NOT_COLOR;

  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db), query, -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, imgid);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, image->exif_model, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 3, image->exif_maker, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 4, image->camera_alias, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 5, image->camera_maker, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 6, image->exif_lens, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_DOUBLE(stmt, 7, fmaxf(0.0f, fminf(FLT_MAX, image->exif_iso)));
  DT_DEBUG_SQLITE3_BIND_DOUBLE(stmt, 8, fmaxf(0.0f, fminf(1000000, image->exif_exposure)));
  DT_DEBUG_SQLITE3_BIND_DOUBLE(stmt, 9, fmaxf(0.0f, fminf(1000000, image->exif_aperture)));
  DT_DEBUG_SQLITE3_BIND_DOUBLE(stmt, 10, fmaxf(0.0f, fminf(1000000, image->exif_focal_length)));
  // 0: dontcare, 1: ldr, 2: raw plus monochrome & color
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 11, iformat);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 12, excluded);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 13, workflow_preset, -1, SQLITE_TRANSIENT);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);

  // now we want to auto-apply the iop-order list if one corresponds and none are
  // still applied. Note that we can already have an iop-order list set when
  // copying an history or applying a style to a not yet developed image.

  if(!dt_ioppr_has_iop_order_list(imgid))
  {
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                                "SELECT op_params"
                                " FROM data.presets"
                                " WHERE autoapply=1"
                                "       AND ((?2 LIKE model AND ?3 LIKE maker) OR (?4 LIKE model AND ?5 LIKE maker))"
                                "       AND ?6 LIKE lens AND ?7 BETWEEN iso_min AND iso_max"
                                "       AND ?8 BETWEEN exposure_min AND exposure_max"
                                "       AND ?9 BETWEEN aperture_min AND aperture_max"
                                "       AND ?10 BETWEEN focal_length_min AND focal_length_max"
                                "       AND (format = 0 OR (format&?11 != 0 AND ~format&?12 != 0))"
                                "       AND operation = 'ioporder'"
                                " ORDER BY writeprotect DESC, LENGTH(model), LENGTH(maker), LENGTH(lens)",
                                -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, imgid);
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, image->exif_model, -1, SQLITE_TRANSIENT);
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 3, image->exif_maker, -1, SQLITE_TRANSIENT);
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 4, image->camera_alias, -1, SQLITE_TRANSIENT);
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 5, image->camera_maker, -1, SQLITE_TRANSIENT);
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 6, image->exif_lens, -1, SQLITE_TRANSIENT);
    DT_DEBUG_SQLITE3_BIND_DOUBLE(stmt, 7, fmaxf(0.0f, fminf(FLT_MAX, image->exif_iso)));
    DT_DEBUG_SQLITE3_BIND_DOUBLE(stmt, 8, fmaxf(0.0f, fminf(1000000, image->exif_exposure)));
    DT_DEBUG_SQLITE3_BIND_DOUBLE(stmt, 9, fmaxf(0.0f, fminf(1000000, image->exif_aperture)));
    DT_DEBUG_SQLITE3_BIND_DOUBLE(stmt, 10, fmaxf(0.0f, fminf(1000000, image->exif_focal_length)));
    // 0: dontcare, 1: ldr, 2: raw plus monochrome & color
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 11, iformat);
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 12, excluded);
    if(sqlite3_step(stmt) == SQLITE_ROW)
    {
      const char *params = (char *)sqlite3_column_blob(stmt, 0);
      const int32_t params_len = sqlite3_column_bytes(stmt, 0);
      GList *iop_list = dt_ioppr_deserialize_iop_order_list(params, params_len);
      dt_ioppr_write_iop_order_list(iop_list, imgid);
      g_list_free_full(iop_list, free);
      dt_ioppr_set_default_iop_order(dev, imgid);
    }
    else
    {
      // we have no auto-apply order, so apply iop order, depending of the workflow
      GList *iop_list = dt_ioppr_get_iop_order_list_version(DT_IOP_ORDER_V30);
      dt_ioppr_write_iop_order_list(iop_list, imgid);
      g_list_free_full(iop_list, free);
      dt_ioppr_set_default_iop_order(dev, imgid);
    }
    sqlite3_finalize(stmt);
  }

  image->flags |= DT_IMAGE_AUTO_PRESETS_APPLIED | DT_IMAGE_NO_LEGACY_PRESETS;

  // make sure these end up in the image_cache; as the history is not correct right now
  // we don't write the sidecar here but later in dt_dev_read_history_ext
  dt_image_cache_write_release(darktable.image_cache, image, DT_IMAGE_CACHE_RELAXED);

  return TRUE;
}

static void _dev_add_default_modules(dt_develop_t *dev, const int imgid)
{
  // modules that cannot be disabled
  // or modules that can be disabled but are auto-on
  for(GList *modules = dev->iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *module = (dt_iop_module_t *)modules->data;

    if(!dt_history_check_module_exists(imgid, module->op, FALSE)
       && module->default_enabled
       && !(module->flags() & IOP_FLAGS_NO_HISTORY_STACK))
    {
      _dev_insert_module(dev, module, imgid);
    }
  }
}

static void _dev_merge_history(dt_develop_t *dev, const int imgid)
{
  sqlite3_stmt *stmt;

  // count what we found:
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                              "SELECT COUNT(*) FROM memory.history", -1,
                              &stmt, NULL);
  if(sqlite3_step(stmt) == SQLITE_ROW)
  {
    // if there is anything..
    const int cnt = sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);

    // workaround a sqlite3 "feature". The above statement to insert
    // items into memory.history is complex and in this case sqlite
    // does not give rowid a linear increment. But the following code
    // really expect that the rowid in this table starts from 0 and
    // increment one by one. So in the following code we rewrite the
    // "num" values from 0 to cnt-1.

    if(cnt > 0)
    {
      // get all rowids
      GList *rowids = NULL;

      // get the rowids in descending order since building the list will reverse the order
      DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                                  "SELECT rowid FROM memory.history ORDER BY rowid DESC",
                                  -1, &stmt, NULL);
      while(sqlite3_step(stmt) == SQLITE_ROW)
        rowids = g_list_prepend(rowids, GINT_TO_POINTER(sqlite3_column_int(stmt, 0)));
      sqlite3_finalize(stmt);

      // update num accordingly
      int v = 0;

      DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                                  "UPDATE memory.history SET num=?1 WHERE rowid=?2",
                                  -1, &stmt, NULL);

      // let's wrap this into a transaction, it might make it a little faster.
      dt_database_start_transaction(darktable.db);

      for(GList *r = rowids; r; r = g_list_next(r))
      {
        DT_DEBUG_SQLITE3_CLEAR_BINDINGS(stmt);
        DT_DEBUG_SQLITE3_RESET(stmt);
        DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, v);
        DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, GPOINTER_TO_INT(r->data));

        if(sqlite3_step(stmt) != SQLITE_DONE) break;

        v++;
      }

      dt_database_release_transaction(darktable.db);

      g_list_free(rowids);

      // advance the current history by cnt amount, that is, make space
      // for the preset/default iops that will be *prepended* into the
      // history.
      DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                                  "UPDATE main.history SET num=num+?1 WHERE imgid=?2",
                                  -1, &stmt, NULL);
      DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, cnt);
      DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, imgid);

      if(sqlite3_step(stmt) == SQLITE_DONE)
      {
        sqlite3_finalize(stmt);
        // clang-format off
        DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                                    "UPDATE main.images"
                                    " SET history_end=history_end+?1"
                                    " WHERE id=?2",
                                    -1, &stmt, NULL);
        // clang-format on
        DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, cnt);
        DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, imgid);

        if(sqlite3_step(stmt) == SQLITE_DONE)
        {
          // and finally prepend the rest with increasing numbers (starting at 0)
          sqlite3_finalize(stmt);
          // clang-format off
          DT_DEBUG_SQLITE3_PREPARE_V2(
            dt_database_get(darktable.db),
            "INSERT INTO main.history"
            " SELECT imgid, num, module, operation, op_params, enabled, "
            "        blendop_params, blendop_version, multi_priority,"
            "        multi_name"
            " FROM memory.history",
            -1, &stmt, NULL);
          // clang-format on
          sqlite3_step(stmt);
          sqlite3_finalize(stmt);
        }
      }
    }
  }
}

// helper function for debug strings
char * _print_validity(gboolean state)
{
  if(state)
    return "ok";
  else
    return "WRONG";
}


static void _init_default_history(dt_develop_t *dev, const int imgid, gboolean *first_run, gboolean *auto_apply_modules)
{
  // cleanup DB
  DT_DEBUG_SQLITE3_EXEC(dt_database_get(darktable.db), "DELETE FROM memory.history", NULL, NULL, NULL);
  dt_print(DT_DEBUG_HISTORY, "[history] temporary history deleted\n");

  // make sure all modules default params are loaded to init history
  _dt_dev_load_pipeline_defaults(dev);

  // prepend all default modules to memory.history
  _dev_add_default_modules(dev, imgid);
  const int default_modules = _dev_get_module_nb_records();

  // maybe add auto-presets to memory.history
  *first_run = _dev_auto_apply_presets(dev);
  *auto_apply_modules = _dev_get_module_nb_records() - default_modules;
  dt_print(DT_DEBUG_HISTORY, "[history] temporary history initialised with default params and presets\n");

  // now merge memory.history into main.history
  _dev_merge_history(dev, imgid);
  dt_print(DT_DEBUG_HISTORY, "[history] temporary history merged with image history\n");
}


static void _find_so_for_history_entry(dt_develop_t *dev, dt_dev_history_item_t *hist)
{
  dt_iop_module_t *match = NULL;

  for(GList *modules = g_list_first(dev->iop); modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *module = (dt_iop_module_t *)modules->data;
    if(!strcmp(module->op, hist->op_name))
    {
      if(module->multi_priority == hist->multi_priority)
      {
        // Found exact match at required priority: we are done
        hist->module = module;
        break;
      }
      else if(hist->multi_priority > 0)
      {
        // Found the right kind of module but the wrong instance.
        // Current history entry is targeting an instance that may exist later in the pipe, so keep looping/looking.
        match = module;
      }
    }
  }

  if(!hist->module && match)
  {
    // We found a module having the required name but not the required instance number:
    // add a new instance of this module by using its ->so property
    dt_iop_module_t *new_module = (dt_iop_module_t *)calloc(1, sizeof(dt_iop_module_t));
    if(!dt_iop_load_module(new_module, match->so, dev))
    {
      dev->iop = g_list_append(dev->iop, new_module);
      // Just init, it will get rewritten later by resync IOP order methods:
      new_module->instance = match->instance;
      hist->module = new_module;
    }
  }
  // else we found an already-existing instance and it's in hist->module already
}


static void _sync_blendop_params(dt_dev_history_item_t *hist, const void *blendop_params, const int bl_length,
                          const int blendop_version, int *legacy_params)
{
  const gboolean is_valid_blendop_version = (blendop_version == dt_develop_blend_version());
  const gboolean is_valid_blendop_size = (bl_length == sizeof(dt_develop_blend_params_t));

  hist->blend_params = malloc(sizeof(dt_develop_blend_params_t));

  if(blendop_params && is_valid_blendop_version && is_valid_blendop_size)
  {
    memcpy(hist->blend_params, blendop_params, sizeof(dt_develop_blend_params_t));
  }
  else if(blendop_params
          && dt_develop_blend_legacy_params(hist->module, blendop_params, blendop_version, hist->blend_params,
                                            dt_develop_blend_version(), bl_length)
                 == 0)
  {
    *legacy_params = TRUE;
  }
  else
  {
    memcpy(hist->blend_params, hist->module->default_blendop_params, sizeof(dt_develop_blend_params_t));
  }
}

static int _sync_params(dt_dev_history_item_t *hist, const void *module_params, const int param_length,
                          const int modversion, int *legacy_params)
{
  const gboolean is_valid_module_version = (modversion == hist->module->version());
  const gboolean is_valid_params_size = (param_length == hist->module->params_size);

  hist->params = malloc(hist->module->params_size);
  if(is_valid_module_version && is_valid_params_size)
  {
    memcpy(hist->params, module_params, hist->module->params_size);
  }
  else
  {
    if(!hist->module->legacy_params
        || hist->module->legacy_params(hist->module, module_params, labs(modversion),
                                      hist->params, labs(hist->module->version())))
    {
      fprintf(stderr, "[dev_read_history] module `%s' version mismatch: history is %d, dt %d.\n",
              hist->module->op, modversion, hist->module->version());

      dt_control_log(_("module `%s' version mismatch: %d != %d"), hist->module->op,
                      hist->module->version(), modversion);
      dt_dev_free_history_item(hist);
      return 1;
    }
    else
    {
      if(!strcmp(hist->module->op, "spots") && modversion == 1)
      {
        // quick and dirty hack to handle spot removal legacy_params
        memcpy(hist->blend_params, hist->module->blend_params, sizeof(dt_develop_blend_params_t));
      }
      *legacy_params = TRUE;
    }

    /*
      * Fix for flip iop: previously it was not always needed, but it might be
      * in history stack as "orientation (off)", but now we always want it
      * by default, so if it is disabled, enable it, and replace params with
      * default_params. if user want to, he can disable it.
      */
    if(!strcmp(hist->module->op, "flip") && hist->enabled == 0 && labs(modversion) == 1)
    {
      memcpy(hist->params, hist->module->default_params, hist->module->params_size);
      hist->enabled = 1;
    }
  }

  // Copy params from history entry to module internals
  memcpy(hist->module->params, hist->params, hist->module->params_size);

  return 0;
}

static int _process_history_db_entry(dt_develop_t *dev, sqlite3_stmt *stmt, const int imgid, int *legacy_params)
{
  // Unpack the DB blobs
  const int id = sqlite3_column_int(stmt, 0);
  const int num = sqlite3_column_int(stmt, 1);
  const int modversion = sqlite3_column_int(stmt, 2);
  const char *module_name = (const char *)sqlite3_column_text(stmt, 3);
  const void *module_params = sqlite3_column_blob(stmt, 4);
  int enabled = sqlite3_column_int(stmt, 5);
  const void *blendop_params = sqlite3_column_blob(stmt, 6);
  const int blendop_version = sqlite3_column_int(stmt, 7);
  const int multi_priority = sqlite3_column_int(stmt, 8);
  const char *multi_name = (const char *)sqlite3_column_text(stmt, 9);

  const int param_length = sqlite3_column_bytes(stmt, 4);
  const int bl_length = sqlite3_column_bytes(stmt, 6);

  // Sanity checks
  const gboolean is_valid_id = (id == imgid);
  const gboolean has_module_name = (module_name != NULL);

  if(!(has_module_name && is_valid_id))
  {
    fprintf(stderr, "[dev_read_history] database history for image `%s' seems to be corrupted!\n",
            dev->image_storage.filename);
    return 1;
  }

  const int iop_order = dt_ioppr_get_iop_order(dev->iop_order_list, module_name, multi_priority);

  // Init a bare minimal history entry
  dt_dev_history_item_t *hist = (dt_dev_history_item_t *)calloc(1, sizeof(dt_dev_history_item_t));
  hist->module = NULL;
  hist->enabled = (enabled != 0); // "cast" int into a clean gboolean
  hist->num = num;
  hist->iop_order = iop_order;
  hist->multi_priority = multi_priority;
  g_strlcpy(hist->op_name, module_name, sizeof(hist->op_name));
  g_strlcpy(hist->multi_name, multi_name, sizeof(hist->multi_name));

  // Find a .so file that matches our history entry, aka a module to run the params stored in DB
  _find_so_for_history_entry(dev, hist);

  if(!hist->module)
  {
    fprintf(
        stderr,
        "[dev_read_history] the module `%s' requested by image `%s' is not installed on this computer!\n",
        module_name, dev->image_storage.filename);
    free(hist);
    return 1;
  }

  // Update IOP order stuff, that applies to all modules regardless of their internals
  hist->module->iop_order = hist->iop_order;
  dt_iop_update_multi_priority(hist->module, hist->multi_priority);

  // module has no user params and won't bother us in GUI - exit early, we are done
  if(hist->module->flags() & IOP_FLAGS_NO_HISTORY_STACK)
  {
    free(hist);
    return 1;
  }

  // Last chance & desperate attempt at enabling/disabling critical modules
  // when history is garbled - This might prevent segfaults on invalid data
  if(hist->module->force_enable) enabled = hist->module->force_enable(hist->module, enabled);

  dt_print(DT_DEBUG_HISTORY, "[history] successfully loaded module %s history (enabled: %i)\n", hist->module->op, enabled);

  // Copy instance name
  g_strlcpy(hist->module->multi_name, hist->multi_name, sizeof(hist->module->multi_name));

  // Copy blending params if valid, else try to convert legacy params
  _sync_blendop_params(hist, blendop_params, bl_length, blendop_version, legacy_params);

  // Copy module params if valid, else try to convert legacy params
  if(_sync_params(hist, module_params, param_length, modversion, legacy_params))
    return 1;

  // make sure that always-on modules are always on. duh.
  if(hist->module->default_enabled == 1 && hist->module->hide_enable_button == 1)
    hist->enabled = hist->module->enabled = TRUE;

  dev->history = g_list_append(dev->history, hist);
  dt_dev_set_history_end(dev, dt_dev_get_history_end(dev) + 1);

  return 0;
}


void dt_dev_read_history_ext(dt_develop_t *dev, const int imgid, gboolean no_image)
{
  if(imgid <= 0) return;
  if(!dev->iop) return;

  int auto_apply_modules = 0;
  gboolean first_run = FALSE;
  gboolean legacy_params = FALSE;

  dt_ioppr_set_default_iop_order(dev, imgid);

  if(!no_image) _init_default_history(dev, imgid, &first_run, &auto_apply_modules);

  sqlite3_stmt *stmt;

  // Load current image history from DB
  // clang-format off
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                              "SELECT imgid, num, module, operation,"
                              "       op_params, enabled, blendop_params,"
                              "       blendop_version, multi_priority, multi_name"
                              " FROM main.history"
                              " WHERE imgid = ?1"
                              " ORDER BY num",
                              -1, &stmt, NULL);
  // clang-format on
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, imgid);

  // Strip rows from DB lookup. One row == One module in history
  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    if(_process_history_db_entry(dev, stmt, imgid, &legacy_params))
      continue;
  }
  sqlite3_finalize(stmt);

  // find the new history end
  // Note: dt_dev_set_history_end sanitizes the value with the actual history size.
  // It needs to run after dev->history is fully populated
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                              "SELECT history_end FROM main.images WHERE id = ?1",
                              -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 1, imgid);
  if(sqlite3_step(stmt) == SQLITE_ROW) // seriously, this should never fail
    if(sqlite3_column_type(stmt, 0) != SQLITE_NULL)
      dt_dev_set_history_end(dev, sqlite3_column_int(stmt, 0));
  sqlite3_finalize(stmt);

  // Sanitize and flatten module order
  dt_ioppr_resync_modules_order(dev);
  dt_ioppr_check_iop_order(dev, imgid, "dt_dev_read_history_no_image end");

  // Update masks history
  // Note: until there, we had only blendops. No masks
  dt_masks_read_masks_history(dev, imgid);

  // Copy and publish the masks on the raster stack for other modules to find
  for(GList *history = g_list_first(dev->history); history; history = g_list_next(history))
  {
    dt_dev_history_item_t *hist = (dt_dev_history_item_t *)history->data;
    dt_iop_commit_blend_params(hist->module, hist->blend_params);

    // Compute the history params hash.
    // This needs to run after dt_iop_compute_blendop_hash(),
    // which is called in dt_iop_commit_blend_params,
    // which needs to run after dt_masks_read_masks_history()
    // TL;DR: don't move this higher, it needs blendop AND mask shapes
    dt_iop_compute_module_hash(hist->module);
    hist->hash = hist->module->hash;
  }

  dt_dev_masks_list_change(dev);
  dt_dev_masks_update_hash(dev);

  dt_print(DT_DEBUG_HISTORY, "[history] dt_dev_read_history_ext completed\n");
}

void dt_dev_read_history(dt_develop_t *dev)
{
  dt_pthread_mutex_lock(&dev->history_mutex);
  dt_dev_read_history_ext(dev, dev->image_storage.id, FALSE);
  dt_pthread_mutex_unlock(&dev->history_mutex);
}

void dt_dev_reprocess_all(dt_develop_t *dev)
{
  if(darktable.gui->reset || !dev || !dev->gui_attached) return;
  dt_dev_pixelpipe_cache_flush(&(dev->pipe->cache));
  dt_dev_pixelpipe_cache_flush(&(dev->preview_pipe->cache));
  dt_dev_pixelpipe_rebuild(dev);
}

void dt_dev_check_zoom_bounds(dt_develop_t *dev, float *zoom_x, float *zoom_y, dt_dev_zoom_t zoom,
                              int closeup, float *boxww, float *boxhh)
{
  int procw = 0, proch = 0;
  dt_dev_get_processed_size(dev, &procw, &proch);
  float boxw = 1.0f, boxh = 1.0f; // viewport in normalised space
                            //   if(zoom == DT_ZOOM_1)
                            //   {
                            //     const float imgw = (closeup ? 2 : 1)*procw;
                            //     const float imgh = (closeup ? 2 : 1)*proch;
                            //     const float devw = MIN(imgw, dev->width);
                            //     const float devh = MIN(imgh, dev->height);
                            //     boxw = fminf(1.0, devw/imgw);
                            //     boxh = fminf(1.0, devh/imgh);
                            //   }
  if(zoom == DT_ZOOM_FIT)
  {
    *zoom_x = *zoom_y = 0.0f;
    boxw = boxh = 1.0f;
  }
  else
  {
    const float scale = dt_dev_get_zoom_scale(dev, zoom, 1<<closeup, 0);
    const float imgw = procw;
    const float imgh = proch;
    const float devw = dev->width;
    const float devh = dev->height;
    boxw = devw / (imgw * scale);
    boxh = devh / (imgh * scale);
  }

  if(*zoom_x < boxw / 2 - .5) *zoom_x = boxw / 2 - .5;
  if(*zoom_x > .5 - boxw / 2) *zoom_x = .5 - boxw / 2;
  if(*zoom_y < boxh / 2 - .5) *zoom_y = boxh / 2 - .5;
  if(*zoom_y > .5 - boxh / 2) *zoom_y = .5 - boxh / 2;
  if(boxw > 1.0) *zoom_x = 0.0f;
  if(boxh > 1.0) *zoom_y = 0.0f;

  if(boxww) *boxww = boxw;
  if(boxhh) *boxhh = boxh;
}

void dt_dev_get_processed_size(const dt_develop_t *dev, int *procw, int *proch)
{
  if(!dev) return;

  // if pipe is processed, lets return its size
  if(dev->pipe && dev->pipe->processed_width)
  {
    *procw = dev->pipe->processed_width;
    *proch = dev->pipe->processed_height;
    return;
  }

  // fallback on preview pipe
  if(dev->preview_pipe && dev->preview_pipe->processed_width)
  {
    const float scale = dev->preview_pipe->iscale;
    *procw = scale * dev->preview_pipe->processed_width;
    *proch = scale * dev->preview_pipe->processed_height;
    return;
  }

  // no processed pipes, lets return 0 size
  *procw = *proch = 0;
  return;
}

void dt_dev_get_pointer_zoom_pos(dt_develop_t *dev, const float px, const float py, float *zoom_x,
                                 float *zoom_y)
{
  dt_dev_zoom_t zoom;
  int closeup = 0, procw = 0, proch = 0;
  float zoom2_x = 0.0f, zoom2_y = 0.0f;
  zoom = dt_control_get_dev_zoom();
  closeup = dt_control_get_dev_closeup();
  zoom2_x = dt_control_get_dev_zoom_x();
  zoom2_y = dt_control_get_dev_zoom_y();
  dt_dev_get_processed_size(dev, &procw, &proch);
  const float scale = dt_dev_get_zoom_scale(dev, zoom, 1<<closeup, 0);
  // offset from center now (current zoom_{x,y} points there)
  const float mouse_off_x = px - .5 * dev->width, mouse_off_y = py - .5 * dev->height;
  zoom2_x += mouse_off_x / (procw * scale);
  zoom2_y += mouse_off_y / (proch * scale);
  *zoom_x = zoom2_x;
  *zoom_y = zoom2_y;
}

void dt_dev_get_history_item_label(dt_dev_history_item_t *hist, char *label, const int cnt)
{
  gchar *module_label = dt_history_item_get_name(hist->module);
  g_snprintf(label, cnt, "%s (%s)", module_label, hist->enabled ? _("on") : _("off"));
  g_free(module_label);
}

int dt_dev_is_current_image(dt_develop_t *dev, uint32_t imgid)
{
  return (dev->image_storage.id == imgid) ? 1 : 0;
}

static dt_dev_proxy_exposure_t *find_last_exposure_instance(dt_develop_t *dev)
{
  if(!dev->proxy.exposure.module) return NULL;

  dt_dev_proxy_exposure_t *instance = &dev->proxy.exposure;

  return instance;
};

gboolean dt_dev_exposure_hooks_available(dt_develop_t *dev)
{
  dt_dev_proxy_exposure_t *instance = find_last_exposure_instance(dev);

  /* check if exposure iop module has registered its hooks */
  if(instance && instance->module && instance->get_black && instance->get_exposure)
    return TRUE;

  return FALSE;
}

float dt_dev_exposure_get_exposure(dt_develop_t *dev)
{
  dt_dev_proxy_exposure_t *instance = find_last_exposure_instance(dev);

  if(instance && instance->module && instance->get_exposure) return instance->get_exposure(instance->module);

  return 0.0;
}


float dt_dev_exposure_get_black(dt_develop_t *dev)
{
  dt_dev_proxy_exposure_t *instance = find_last_exposure_instance(dev);

  if(instance && instance->module && instance->get_black) return instance->get_black(instance->module);

  return 0.0;
}

void dt_dev_modulegroups_set(dt_develop_t *dev, uint32_t group)
{
  if(dev->proxy.modulegroups.module && dev->proxy.modulegroups.set)
    dev->proxy.modulegroups.set(dev->proxy.modulegroups.module, group);
}

uint32_t dt_dev_modulegroups_get(dt_develop_t *dev)
{
  if(dev->proxy.modulegroups.module && dev->proxy.modulegroups.get)
    return dev->proxy.modulegroups.get(dev->proxy.modulegroups.module);

  return 0;
}

void dt_dev_modulegroups_switch(dt_develop_t *dev, dt_iop_module_t *module)
{
  if(dev->proxy.modulegroups.module && dev->proxy.modulegroups.switch_group)
    dev->proxy.modulegroups.switch_group(dev->proxy.modulegroups.module, module);
}

void dt_dev_modulegroups_update_visibility(dt_develop_t *dev)
{
  if(dev->proxy.modulegroups.module && dev->proxy.modulegroups.switch_group)
    dev->proxy.modulegroups.update_visibility(dev->proxy.modulegroups.module);
}

void dt_dev_modulegroups_search_text_focus(dt_develop_t *dev)
{
  if(dev->proxy.modulegroups.module && dev->proxy.modulegroups.search_text_focus)
    dev->proxy.modulegroups.search_text_focus(dev->proxy.modulegroups.module);
}

void dt_dev_masks_list_change(dt_develop_t *dev)
{
  if(dev->proxy.masks.module && dev->proxy.masks.list_change)
    dev->proxy.masks.list_change(dev->proxy.masks.module);
}
void dt_dev_masks_list_update(dt_develop_t *dev)
{
  if(dev->proxy.masks.module && dev->proxy.masks.list_update)
    dev->proxy.masks.list_update(dev->proxy.masks.module);
}
void dt_dev_masks_list_remove(dt_develop_t *dev, int formid, int parentid)
{
  if(dev->proxy.masks.module && dev->proxy.masks.list_remove)
    dev->proxy.masks.list_remove(dev->proxy.masks.module, formid, parentid);
}
void dt_dev_masks_selection_change(dt_develop_t *dev, struct dt_iop_module_t *module,
                                   const int selectid, const int throw_event)
{
  if(dev->proxy.masks.module && dev->proxy.masks.selection_change)
    dev->proxy.masks.selection_change(dev->proxy.masks.module, module, selectid, throw_event);
}

void dt_dev_snapshot_request(dt_develop_t *dev, const char *filename)
{
  dev->proxy.snapshot.filename = filename;
  dev->proxy.snapshot.request = TRUE;
  dt_control_queue_redraw_center();
}

void dt_dev_average_delay_update(const dt_times_t *start, uint32_t *average_delay)
{
  dt_times_t end;
  dt_get_times(&end);

  *average_delay += ((end.clock - start->clock) * 1000 / DT_DEV_AVERAGE_DELAY_COUNT
                     - *average_delay / DT_DEV_AVERAGE_DELAY_COUNT);
}


/** duplicate a existent module */
dt_iop_module_t *dt_dev_module_duplicate(dt_develop_t *dev, dt_iop_module_t *base)
{
  // we create the new module
  dt_iop_module_t *module = (dt_iop_module_t *)calloc(1, sizeof(dt_iop_module_t));
  if(dt_iop_load_module(module, base->so, base->dev)) return NULL;
  module->instance = base->instance;

  // we set the multi-instance priority and the iop order
  int pmax = 0;
  for(GList *modules = base->dev->iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = (dt_iop_module_t *)modules->data;
    if(mod->instance == base->instance)
    {
      if(pmax < mod->multi_priority) pmax = mod->multi_priority;
    }
  }
  // create a unique multi-priority
  pmax += 1;
  dt_iop_update_multi_priority(module, pmax);

  // add this new module position into the iop-order-list
  dt_ioppr_insert_module_instance(dev, module);

  // since we do not rename the module we need to check that an old module does not have the same name. Indeed
  // the multi_priority
  // are always rebased to start from 0, to it may be the case that the same multi_name be generated when
  // duplicating a module.
  int pname = module->multi_priority;
  char mname[128];

  do
  {
    snprintf(mname, sizeof(mname), "%d", pname);
    gboolean dup = FALSE;

    for(GList *modules = base->dev->iop; modules; modules = g_list_next(modules))
    {
      dt_iop_module_t *mod = (dt_iop_module_t *)modules->data;
      if(mod->instance == base->instance)
      {
        if(strcmp(mname, mod->multi_name) == 0)
        {
          dup = TRUE;
          break;
        }
      }
    }

    if(dup)
      pname++;
    else
      break;
  } while(1);

  // the multi instance name
  g_strlcpy(module->multi_name, mname, sizeof(module->multi_name));
  // we insert this module into dev->iop
  base->dev->iop = g_list_insert_sorted(base->dev->iop, module, dt_sort_iop_by_order);

  // always place the new instance after the base one
  if(!dt_ioppr_move_iop_after(base->dev, module, base))
  {
    fprintf(stderr, "[dt_dev_module_duplicate] can't move new instance after the base one\n");
  }

  // that's all. rest of insertion is gui work !
  return module;
}

void dt_dev_invalidate_history_module(GList *list, dt_iop_module_t *module)
{
  for(; list; list = g_list_next(list))
  {
    dt_dev_history_item_t *hitem = (dt_dev_history_item_t *)list->data;
    if (hitem->module == module)
    {
      hitem->module = NULL;
    }
  }
}

void dt_dev_module_remove(dt_develop_t *dev, dt_iop_module_t *module)
{
  // if(darktable.gui->reset) return;
  dt_pthread_mutex_lock(&dev->history_mutex);
  int del = 0;

  if(dev->gui_attached)
  {
    dt_dev_undo_start_record(dev);

    GList *elem = dev->history;
    while(elem != NULL)
    {
      GList *next = g_list_next(elem);
      dt_dev_history_item_t *hist = (dt_dev_history_item_t *)(elem->data);

      if(module == hist->module)
      {
        dt_print(DT_DEBUG_HISTORY, "[dt_module_remode] removing obsoleted history item: %s %s %p %p\n", hist->module->op, hist->module->multi_name, module, hist->module);
        dt_dev_free_history_item(hist);
        dev->history = g_list_delete_link(dev->history, elem);
        dt_dev_set_history_end(dev, dt_dev_get_history_end(dev) - 1);
        del = 1;
      }
      elem = next;
    }
  }


  // and we remove it from the list
  for(GList *modules = dev->iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = (dt_iop_module_t *)modules->data;
    if(mod == module)
    {
      dev->iop = g_list_remove_link(dev->iop, modules);
      break;
    }
  }

  dt_pthread_mutex_unlock(&dev->history_mutex);

  if(dev->gui_attached && del)
  {
    /* signal that history has changed */
    dt_dev_undo_end_record(dev);

    DT_DEBUG_CONTROL_SIGNAL_RAISE(darktable.signals, DT_SIGNAL_DEVELOP_MODULE_REMOVE, module);
  }
}

void _dev_module_update_multishow(dt_develop_t *dev, struct dt_iop_module_t *module)
{
  // We count the number of other instances
  int nb_instances = 0;
  for(GList *modules = g_list_first(dev->iop); modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = (dt_iop_module_t *)modules->data;

    if(mod->instance == module->instance) nb_instances++;
  }

  dt_iop_module_t *mod_prev = dt_iop_gui_get_previous_visible_module(module);
  dt_iop_module_t *mod_next = dt_iop_gui_get_next_visible_module(module);

  const gboolean move_next = (mod_next && mod_next->iop_order != INT_MAX) ? dt_ioppr_check_can_move_after_iop(dev->iop, module, mod_next) : -1.0;
  const gboolean move_prev = (mod_prev && mod_prev->iop_order != INT_MAX) ? dt_ioppr_check_can_move_before_iop(dev->iop, module, mod_prev) : -1.0;

  module->multi_show_new = !(module->flags() & IOP_FLAGS_ONE_INSTANCE);
  module->multi_show_close = (nb_instances > 1);
  if(mod_next)
    module->multi_show_up = move_next;
  else
    module->multi_show_up = 0;
  if(mod_prev)
    module->multi_show_down = move_prev;
  else
    module->multi_show_down = 0;
}

void dt_dev_modules_update_multishow(dt_develop_t *dev)
{
  dt_ioppr_check_iop_order(dev, 0, "dt_dev_modules_update_multishow");

  for(GList *modules = dev->iop; modules; modules = g_list_next(modules))
  {
    dt_iop_module_t *mod = (dt_iop_module_t *)modules->data;

    // only for visible modules
    GtkWidget *expander = mod->expander;
    if(expander && gtk_widget_is_visible(expander))
    {
      _dev_module_update_multishow(dev, mod);
    }
  }
}

gchar *dt_history_item_get_label(const struct dt_iop_module_t *module)
{
  gchar *label;
  /* create a history button and add to box */
  if(!module->multi_name[0] || strcmp(module->multi_name, "0") == 0)
    label = g_strdup(module->name());
  else
  {
    label = g_strdup_printf("%s %s", module->name(), module->multi_name);
  }
  return label;
}

gchar *dt_history_item_get_name(const struct dt_iop_module_t *module)
{
  gchar *label;
  /* create a history button and add to box */
  if(!module->multi_name[0] || strcmp(module->multi_name, "0") == 0)
    label = delete_underscore(module->name());
  else
  {
    gchar *clean_name = delete_underscore(module->name());
    label = g_strdup_printf("%s %s", clean_name, module->multi_name);
    g_free(clean_name);
  }
  return label;
}

gchar *dt_history_item_get_name_html(const struct dt_iop_module_t *module)
{
  gchar *clean_name = delete_underscore(module->name());
  gchar *label;
  /* create a history button and add to box */
  if(!module->multi_name[0] || strcmp(module->multi_name, "0") == 0)
    label = g_markup_escape_text(clean_name, -1);
  else
    label = g_markup_printf_escaped("%s <span size=\"smaller\">%s</span>", clean_name, module->multi_name);
  g_free(clean_name);
  return label;
}

int dt_dev_distort_transform(dt_develop_t *dev, float *points, size_t points_count)
{
  return dt_dev_distort_transform_plus(dev, dev->preview_pipe, 0.0f, DT_DEV_TRANSFORM_DIR_ALL, points, points_count);
}
int dt_dev_distort_backtransform(dt_develop_t *dev, float *points, size_t points_count)
{
  return dt_dev_distort_backtransform_plus(dev, dev->preview_pipe, 0.0f, DT_DEV_TRANSFORM_DIR_ALL, points, points_count);
}

// only call directly or indirectly from dt_dev_distort_transform_plus, so that it runs with the history locked
int dt_dev_distort_transform_locked(dt_develop_t *dev, dt_dev_pixelpipe_t *pipe, const double iop_order,
                                    const int transf_direction, float *points, size_t points_count)
{
  GList *modules = pipe->iop;
  GList *pieces = pipe->nodes;
  while(modules)
  {
    if(!pieces)
    {
      return 0;
    }
    dt_iop_module_t *module = (dt_iop_module_t *)(modules->data);
    dt_dev_pixelpipe_iop_t *piece = (dt_dev_pixelpipe_iop_t *)(pieces->data);
    if(piece->enabled
       && ((transf_direction == DT_DEV_TRANSFORM_DIR_ALL)
           || (transf_direction == DT_DEV_TRANSFORM_DIR_FORW_INCL && module->iop_order >= iop_order)
           || (transf_direction == DT_DEV_TRANSFORM_DIR_FORW_EXCL && module->iop_order > iop_order)
           || (transf_direction == DT_DEV_TRANSFORM_DIR_BACK_INCL && module->iop_order <= iop_order)
           || (transf_direction == DT_DEV_TRANSFORM_DIR_BACK_EXCL && module->iop_order < iop_order))
       && !dt_dev_pixelpipe_activemodule_disables_currentmodule(dev, module))
    {
      module->distort_transform(module, piece, points, points_count);
    }
    modules = g_list_next(modules);
    pieces = g_list_next(pieces);
  }
  return 1;
}

int dt_dev_distort_transform_plus(dt_develop_t *dev, dt_dev_pixelpipe_t *pipe, const double iop_order, const int transf_direction,
                                  float *points, size_t points_count)
{
  dt_pthread_mutex_lock(&dev->history_mutex);
  dt_dev_distort_transform_locked(dev, pipe, iop_order, transf_direction, points, points_count);
  dt_pthread_mutex_unlock(&dev->history_mutex);
  return 1;
}

// only call directly or indirectly from dt_dev_distort_transform_plus, so that it runs with the history locked
int dt_dev_distort_backtransform_locked(dt_develop_t *dev, dt_dev_pixelpipe_t *pipe, const double iop_order,
                                        const int transf_direction, float *points, size_t points_count)
{
  GList *modules = g_list_last(pipe->iop);
  GList *pieces = g_list_last(pipe->nodes);
  while(modules)
  {
    if(!pieces)
    {
      return 0;
    }
    dt_iop_module_t *module = (dt_iop_module_t *)(modules->data);
    dt_dev_pixelpipe_iop_t *piece = (dt_dev_pixelpipe_iop_t *)(pieces->data);
    if(piece->enabled
       && ((transf_direction == DT_DEV_TRANSFORM_DIR_ALL)
           || (transf_direction == DT_DEV_TRANSFORM_DIR_FORW_INCL && module->iop_order >= iop_order)
           || (transf_direction == DT_DEV_TRANSFORM_DIR_FORW_EXCL && module->iop_order > iop_order)
           || (transf_direction == DT_DEV_TRANSFORM_DIR_BACK_INCL && module->iop_order <= iop_order)
           || (transf_direction == DT_DEV_TRANSFORM_DIR_BACK_EXCL && module->iop_order < iop_order))
       && !dt_dev_pixelpipe_activemodule_disables_currentmodule(dev, module))
    {
      module->distort_backtransform(module, piece, points, points_count);
    }
    modules = g_list_previous(modules);
    pieces = g_list_previous(pieces);
  }
  return 1;
}

int dt_dev_distort_backtransform_plus(dt_develop_t *dev, dt_dev_pixelpipe_t *pipe, const double iop_order, const int transf_direction,
                                      float *points, size_t points_count)
{
  dt_pthread_mutex_lock(&dev->history_mutex);
  const int success = dt_dev_distort_backtransform_locked(dev, pipe, iop_order, transf_direction, points, points_count);
  dt_pthread_mutex_unlock(&dev->history_mutex);
  return success;
}

dt_dev_pixelpipe_iop_t *dt_dev_distort_get_iop_pipe(dt_develop_t *dev, struct dt_dev_pixelpipe_t *pipe,
                                                    struct dt_iop_module_t *module)
{
  for(const GList *pieces = g_list_last(pipe->nodes); pieces; pieces = g_list_previous(pieces))
  {
    dt_dev_pixelpipe_iop_t *piece = (dt_dev_pixelpipe_iop_t *)(pieces->data);
    if(piece->module == module)
    {
      return piece;
    }
  }
  return NULL;
}


int dt_dev_wait_hash(dt_develop_t *dev, struct dt_dev_pixelpipe_t *pipe, const double iop_order, const int transf_direction, dt_pthread_mutex_t *lock,
                     const volatile uint64_t *const hash)
{
  const int usec = 5000;
  int nloop;

#ifdef HAVE_OPENCL
  if(pipe->devid >= 0)
    nloop = darktable.opencl->opencl_synchronization_timeout;
  else
    nloop = dt_conf_get_int("pixelpipe_synchronization_timeout");
#else
  nloop = dt_conf_get_int("pixelpipe_synchronization_timeout");
#endif

  if(nloop <= 0) return TRUE;  // non-positive values omit pixelpipe synchronization

  for(int n = 0; n < nloop; n++)
  {
    if(dt_atomic_get_int(&pipe->shutdown))
      return TRUE;  // stop waiting if pipe shuts down

    uint64_t probehash;

    if(lock)
    {
      dt_pthread_mutex_lock(lock);
      probehash = *hash;
      dt_pthread_mutex_unlock(lock);
    }
    else
      probehash = *hash;

    if(probehash == dt_dev_hash(dev, pipe))
      return TRUE;

    dt_iop_nap(usec);
  }

  return FALSE;
}

int dt_dev_sync_pixelpipe_hash(dt_develop_t *dev, struct dt_dev_pixelpipe_t *pipe, const double iop_order, const int transf_direction, dt_pthread_mutex_t *lock,
                               const volatile uint64_t *const hash)
{
  // first wait for matching hash values
  if(dt_dev_wait_hash(dev, pipe, iop_order, transf_direction, lock, hash))
    return TRUE;

  // timed out. let's see if history stack has changed
  if(pipe->changed & (DT_DEV_PIPE_TOP_CHANGED | DT_DEV_PIPE_REMOVE | DT_DEV_PIPE_SYNCH))
  {
    dt_dev_invalidate(dev);
    // pretend that everything is fine
    return TRUE;
  }

  // no way to get pixelpipes in sync
  return FALSE;
}


uint64_t dt_dev_hash(dt_develop_t *dev, struct dt_dev_pixelpipe_t *pipe)
{
  uint64_t hash = 0;
  // FIXME: this should have its own hash
  // since it's available before pipeline computation
  // but after dev->history reading and pipe nodes init
  GList *pieces = g_list_last(pipe->nodes);
  if(pieces)
  {
    dt_dev_pixelpipe_iop_t *piece = (dt_dev_pixelpipe_iop_t *)(pieces->data);
    hash = piece->global_hash;
  }
  return hash;
}

// set the module list order
void dt_dev_reorder_gui_module_list(dt_develop_t *dev)
{
  int pos_module = 0;
  for(const GList *modules = g_list_last(dev->iop); modules; modules = g_list_previous(modules))
  {
    dt_iop_module_t *module = (dt_iop_module_t *)(modules->data);

    GtkWidget *expander = module->expander;
    if(expander)
    {
      gtk_box_reorder_child(dt_ui_get_container(darktable.gui->ui, DT_UI_CONTAINER_PANEL_RIGHT_CENTER), expander,
                            pos_module++);
    }
  }
}

void dt_dev_undo_start_record(dt_develop_t *dev)
{
  const dt_view_t *cv = dt_view_manager_get_current_view(darktable.view_manager);

  /* record current history state : before change (needed for undo) */
  if(dev->gui_attached && cv->view((dt_view_t *)cv) == DT_VIEW_DARKROOM)
  {
    DT_DEBUG_CONTROL_SIGNAL_RAISE
      (darktable.signals, DT_SIGNAL_DEVELOP_HISTORY_WILL_CHANGE,
       dt_history_duplicate(dev->history),
       dt_dev_get_history_end(dev),
       dt_ioppr_iop_order_copy_deep(dev->iop_order_list));
  }
}

void dt_dev_undo_end_record(dt_develop_t *dev)
{
  const dt_view_t *cv = dt_view_manager_get_current_view(darktable.view_manager);

  /* record current history state : after change (needed for undo) */
  if(dev->gui_attached && cv->view((dt_view_t *)cv) == DT_VIEW_DARKROOM)
  {
    DT_DEBUG_CONTROL_SIGNAL_RAISE(darktable.signals, DT_SIGNAL_DEVELOP_HISTORY_CHANGE);
  }
}

gboolean dt_masks_get_lock_mode(dt_develop_t *dev)
{
  if(dev->gui_attached)
  {
    dt_pthread_mutex_lock(&darktable.gui->mutex);
    const gboolean state = dev->mask_lock;
    dt_pthread_mutex_unlock(&darktable.gui->mutex);
    return state;
  }
  return FALSE;
}

void dt_masks_set_lock_mode(dt_develop_t *dev, gboolean mode)
{
  if(dev->gui_attached)
  {
    dt_pthread_mutex_lock(&darktable.gui->mutex);
    dev->mask_lock = mode;
    dt_pthread_mutex_unlock(&darktable.gui->mutex);
  }
}

int32_t dt_dev_get_history_end(dt_develop_t *dev)
{
  const int num_items = g_list_length(dev->history);
  return CLAMP(dev->history_end, 0, num_items);
}

void dt_dev_set_history_end(dt_develop_t *dev, const uint32_t index)
{
  const int num_items = g_list_length(dev->history);
  dev->history_end = CLAMP(index, 0, num_items);
}

void dt_dev_append_changed_tag(const int32_t imgid)
{
  /* attach changed tag reflecting actual change */
  guint tagid = 0;
  dt_tag_new("darktable|changed", &tagid);
  const gboolean tag_change = dt_tag_attach(tagid, imgid, FALSE, FALSE);

  /* register last change timestamp in cache */
  dt_image_cache_set_change_timestamp(darktable.image_cache, imgid);

  if(tag_change) DT_DEBUG_CONTROL_SIGNAL_RAISE(darktable.signals, DT_SIGNAL_TAG_CHANGED);
}

void dt_dev_masks_update_hash(dt_develop_t *dev)
{
  dt_times_t start;
  dt_get_times(&start);

  uint64_t hash = 5381;
  for(GList *form = g_list_first(dev->forms); form; form = g_list_next(form))
  {
    dt_masks_form_t *shape = (dt_masks_form_t *)form->data;
    hash = dt_masks_group_get_hash(hash, shape);
  }

  // Keep on accumulating "changed" states until something saves the new stack
  // and resets that to 0
  uint64_t old_hash = dev->forms_hash;
  dev->forms_changed |= (old_hash != hash);
  dev->forms_hash = hash;

  dt_show_times(&start, "[masks_update_hash] computing forms hash");
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
