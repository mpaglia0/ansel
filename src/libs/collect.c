/*
    This file is part of darktable,
    Copyright (C) 2010-2021 darktable developers.

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

#include "libs/collect.h"
#include "bauhaus/bauhaus.h"
#include "common/collection.h"
#include "common/selection.h"
#include "common/darktable.h"
#include "common/debug.h"
#include "common/film.h"
#include "common/metadata.h"
#include "common/utility.h"
#include "common/history.h"
#include "common/map_locations.h"
#include "common/datetime.h"
#include "control/conf.h"
#include "control/control.h"
#include "control/jobs.h"
#include "dtgtk/button.h"

#include "gui/gtk.h"
#include "gui/preferences_dialogs.h"
#include "libs/lib.h"
#include "libs/lib_api.h"
#include "views/view.h"
#ifndef _WIN32
#include <gio/gunixmounts.h>
#endif
#ifdef GDK_WINDOWING_QUARTZ
#include "osx/osx.h"
#endif

DT_MODULE(3)

#define MAX_RULES 10

#define PARAM_STRING_SIZE 256 // FIXME: is this enough !?

typedef struct dt_lib_collect_rule_t
{
  int num;
  GtkWidget *hbox;
  GtkWidget *combo;
  GtkWidget *text;
  GtkWidget *button;
  gboolean typing;
  gchar *searchstring;
  // Backreference the parent of the rule
  void *lib_collect;
} dt_lib_collect_rule_t;

typedef struct dt_lib_collect_t
{
  dt_lib_collect_rule_t rule[MAX_RULES];
  GtkWidget *notebook;

  int active_rule;
  int nb_rules;

  GtkTreeView *view;
  int view_rule;

  GtkTreeModel *treefilter;
  GtkTreeModel *listfilter;

  struct dt_lib_collect_params_t *params;
#ifdef _WIN32
  GVolumeMonitor *vmonitor;
#else
  GUnixMountMonitor *vmonitor;
#endif
} dt_lib_collect_t;

typedef struct dt_lib_collect_params_rule_t
{
  uint32_t item : 16;
  uint32_t mode : 16;
  char string[PARAM_STRING_SIZE];
} dt_lib_collect_params_rule_t;

typedef struct dt_lib_collect_params_t
{
  uint32_t rules;
  dt_lib_collect_params_rule_t rule[MAX_RULES];
} dt_lib_collect_params_t;

typedef enum dt_lib_collect_cols_t
{
  DT_LIB_COLLECT_COL_TEXT = 0,
  DT_LIB_COLLECT_COL_ID,
  DT_LIB_COLLECT_COL_TOOLTIP,
  DT_LIB_COLLECT_COL_PATH,
  DT_LIB_COLLECT_COL_VISIBLE,
  DT_LIB_COLLECT_COL_UNREACHABLE,
  DT_LIB_COLLECT_COL_COUNT,
  DT_LIB_COLLECT_COL_INDEX,
  DT_LIB_COLLECT_NUM_COLS
} dt_lib_collect_cols_t;

typedef struct _range_t
{
  gchar *start;
  gchar *stop;
  GtkTreePath *path1;
  GtkTreePath *path2;
} _range_t;

static void _lib_collect_gui_update(dt_lib_module_t *self);
static void _lib_folders_update_collection(const gchar *filmroll);
static void entry_changed(GtkEntry *entry, dt_lib_collect_rule_t *dr);
static void collection_updated(gpointer instance, dt_collection_change_t query_change,
                               dt_collection_properties_t changed_property, gpointer imgs, int next,
                               gpointer self);
static void row_activated_with_event(GtkTreeView *view, GtkTreePath *path, GtkTreeViewColumn *col, GdkEventButton *event, dt_lib_collect_t *d);
static int is_time_property(int property);
static void _populate_collect_combo(GtkWidget *w);

static gboolean item_is_folder_collection(dt_collection_properties_t item)
{
  return item == DT_COLLECTION_PROP_FILMROLL || item == DT_COLLECTION_PROP_FOLDERS;
}

/*
static gboolean item_is_tag_collection(dt_collection_properties_t item)
{
  return item == DT_COLLECTION_PROP_TAG;
}
*/

static gboolean item_is_numeric_collection(dt_collection_properties_t item)
{
  return item == DT_COLLECTION_PROP_DAY || is_time_property(item) || item == DT_COLLECTION_PROP_APERTURE
         || item == DT_COLLECTION_PROP_FOCAL_LENGTH || item == DT_COLLECTION_PROP_ISO
         || item == DT_COLLECTION_PROP_EXPOSURE
         || item == DT_COLLECTION_PROP_RATING;
}

const char *name(struct dt_lib_module_t *self)
{
  return _("Library");
}

void *legacy_params(struct dt_lib_module_t *self,
                    const void *const old_params, const size_t old_params_size, const int old_version,
                    int *new_version, size_t *new_size)
{
  if(old_version == 1)
  {
    /* from v1 to v2 we have reordered the filters */
    dt_lib_collect_params_t *o = (dt_lib_collect_params_t *)old_params;

    if(o->rules > MAX_RULES)
      /* preset is corrupted, return NULL and drop the preset */
      return NULL;

    dt_lib_collect_params_t *n = (dt_lib_collect_params_t *)malloc(old_params_size);

    const int table[DT_COLLECTION_PROP_LAST] =
      {
        DT_COLLECTION_PROP_FILMROLL,
        DT_COLLECTION_PROP_FOLDERS,
        DT_COLLECTION_PROP_CAMERA,
        DT_COLLECTION_PROP_TAG,
        DT_COLLECTION_PROP_DAY,
        DT_COLLECTION_PROP_TIME,
        DT_COLLECTION_PROP_HISTORY,
        DT_COLLECTION_PROP_COLORLABEL,

        // spaces for the metadata, see metadata.h
        DT_COLLECTION_PROP_COLORLABEL + 1,
        DT_COLLECTION_PROP_COLORLABEL + 2,
        DT_COLLECTION_PROP_COLORLABEL + 3,
        DT_COLLECTION_PROP_COLORLABEL + 4,
        DT_COLLECTION_PROP_COLORLABEL + 5,

        DT_COLLECTION_PROP_LENS,
        DT_COLLECTION_PROP_FOCAL_LENGTH,
        DT_COLLECTION_PROP_ISO,
        DT_COLLECTION_PROP_APERTURE,
        DT_COLLECTION_PROP_EXPOSURE,
        DT_COLLECTION_PROP_FILENAME,
        DT_COLLECTION_PROP_GEOTAGGING,
        DT_COLLECTION_PROP_GROUPING,
        DT_COLLECTION_PROP_LOCAL_COPY,
        DT_COLLECTION_PROP_MODULE,
        DT_COLLECTION_PROP_ORDER
      };

    n->rules = o->rules;

    for(int r=0; r<o->rules; r++)
    {
      n->rule[r].item = table[o->rule[r].item];
      n->rule[r].mode = o->rule[r].mode;
      memcpy(n->rule[r].string, o->rule[r].string, PARAM_STRING_SIZE);
    }

    *new_size = old_params_size;
    *new_version = 2;

    return (void *)n;
  }
  else if(old_version == 2)
  {
    /* from v2 to v3 we have added 4 new timestamp filters and 2 metadata filters */
    dt_lib_collect_params_t *old = (dt_lib_collect_params_t *)old_params;

    if(old->rules > MAX_RULES)
      /* preset is corrupted, return NULL and drop the preset */
      return NULL;

    dt_lib_collect_params_t *new = (dt_lib_collect_params_t *)malloc(old_params_size);

    const int table[DT_COLLECTION_PROP_LAST] =
      {
        DT_COLLECTION_PROP_FILMROLL,
        DT_COLLECTION_PROP_FOLDERS,
        DT_COLLECTION_PROP_FILENAME,
        DT_COLLECTION_PROP_CAMERA,
        DT_COLLECTION_PROP_LENS,
        DT_COLLECTION_PROP_APERTURE,
        DT_COLLECTION_PROP_EXPOSURE,
        DT_COLLECTION_PROP_FOCAL_LENGTH,
        DT_COLLECTION_PROP_ISO,
        DT_COLLECTION_PROP_DAY,
        DT_COLLECTION_PROP_TIME,
        DT_COLLECTION_PROP_GEOTAGGING,
        DT_COLLECTION_PROP_TAG,
        DT_COLLECTION_PROP_COLORLABEL,

        // spaces for the metadata, see metadata.h
        DT_COLLECTION_PROP_COLORLABEL + 1,
        DT_COLLECTION_PROP_COLORLABEL + 2,
        DT_COLLECTION_PROP_COLORLABEL + 3,
        DT_COLLECTION_PROP_COLORLABEL + 4,
        DT_COLLECTION_PROP_COLORLABEL + 5,

        DT_COLLECTION_PROP_GROUPING,
        DT_COLLECTION_PROP_LOCAL_COPY,
        DT_COLLECTION_PROP_HISTORY,
        DT_COLLECTION_PROP_MODULE,
        DT_COLLECTION_PROP_ORDER
      };

    new->rules = old->rules;

    for(int r = 0; r < old->rules; r++)
    {
      new->rule[r].item = table[old->rule[r].item];
      new->rule[r].mode = old->rule[r].mode;
      memcpy(new->rule[r].string, old->rule[r].string, PARAM_STRING_SIZE);
    }

    *new_size = old_params_size;
    *new_version = 3;

    return (void *)new;
  }

  return NULL;
}

void init_presets(dt_lib_module_t *self)
{
  dt_lib_collect_params_t params;

#define CLEAR_PARAMS(r) {                \
    memset(&params, 0, sizeof(params));  \
    params.rules = 1;                    \
    params.rule[0].mode = 0;             \
    params.rule[0].item = r;             \
  }

  // based on date/time
  GDateTime *now = g_date_time_new_now_local();
  char *datetime_today = g_date_time_format(now, "%Y:%m:%d");
  GDateTime *gdt = g_date_time_add_days(now, -1);
  char *datetime_24hrs = g_date_time_format(gdt, "> %Y:%m:%d %H:%M");
  g_date_time_unref(gdt);
  gdt = g_date_time_add_days(now, -30);
  char *datetime_30d = g_date_time_format(gdt, "> %Y:%m:%d");
  g_date_time_unref(gdt);
  g_date_time_unref(now);

  // presets based on import
  CLEAR_PARAMS(DT_COLLECTION_PROP_IMPORT_TIMESTAMP);
  g_strlcpy(params.rule[0].string, datetime_today, PARAM_STRING_SIZE);
  dt_lib_presets_add(_("imported: today"), self->plugin_name, self->version(),
                       &params, sizeof(params), TRUE);

  CLEAR_PARAMS(DT_COLLECTION_PROP_IMPORT_TIMESTAMP);
  g_strlcpy(params.rule[0].string, datetime_24hrs, PARAM_STRING_SIZE);
  dt_lib_presets_add(_("imported: last 24h"), self->plugin_name, self->version(),
                       &params, sizeof(params), TRUE);

  CLEAR_PARAMS(DT_COLLECTION_PROP_IMPORT_TIMESTAMP);
  g_strlcpy(params.rule[0].string, datetime_30d, PARAM_STRING_SIZE);
  dt_lib_presets_add(_("imported: last 30 days"), self->plugin_name, self->version(),
                       &params, sizeof(params), TRUE);

  // presets based on image metadata (image taken)
  CLEAR_PARAMS(DT_COLLECTION_PROP_TIME);
  g_strlcpy(params.rule[0].string, datetime_today, PARAM_STRING_SIZE);
  dt_lib_presets_add(_("taken: today"), self->plugin_name, self->version(),
                       &params, sizeof(params), TRUE);

  CLEAR_PARAMS(DT_COLLECTION_PROP_TIME);
  g_strlcpy(params.rule[0].string, datetime_24hrs, PARAM_STRING_SIZE);
  dt_lib_presets_add(_("taken: last 24h"), self->plugin_name, self->version(),
                       &params, sizeof(params), TRUE);

  CLEAR_PARAMS(DT_COLLECTION_PROP_TIME);
  g_strlcpy(params.rule[0].string, datetime_30d, PARAM_STRING_SIZE);
  dt_lib_presets_add(_("taken: last 30 days"), self->plugin_name, self->version(),
                       &params, sizeof(params), TRUE);

  g_free(datetime_today);
  g_free(datetime_24hrs);
  g_free(datetime_30d);
#undef CLEAR_PARAMS
}

/* Update the params struct with active ruleset */
static void _lib_collect_update_params(dt_lib_collect_t *d)
{
  /* reset params */
  dt_lib_collect_params_t *p = d->params;
  memset(p, 0, sizeof(dt_lib_collect_params_t));

  /* for each active rule set update params */
  const int _a = dt_conf_get_int("plugins/lighttable/collect/num_rules") - 1;
  const int active = CLAMP(_a, 0, (MAX_RULES - 1));
  char confname[200] = { 0 };
  for(int i = 0; i <= active; i++)
  {
    /* get item */
    snprintf(confname, sizeof(confname), "plugins/lighttable/collect/item%1d", i);
    p->rule[i].item = dt_conf_get_int(confname);

    /* get mode */
    snprintf(confname, sizeof(confname), "plugins/lighttable/collect/mode%1d", i);
    p->rule[i].mode = dt_conf_get_int(confname);

    /* get string */
    snprintf(confname, sizeof(confname), "plugins/lighttable/collect/string%1d", i);
    const char *string = dt_conf_get_string_const(confname);
    if(string != NULL) g_strlcpy(p->rule[i].string, string, PARAM_STRING_SIZE);
    // fprintf(stdout,"[%i] %d,%d,%s\n",i, p->rule[i].item, p->rule[i].mode,  p->rule[i].string);
  }

  p->rules = active + 1;
}

void *get_params(dt_lib_module_t *self, int *size)
{
  _lib_collect_update_params(self->data);

  /* allocate a copy of params to return, freed by caller */
  *size = sizeof(dt_lib_collect_params_t);
  void *p = malloc(*size);
  memcpy(p, ((dt_lib_collect_t *)self->data)->params, *size);
  return p;
}

int set_params(dt_lib_module_t *self, const void *params, int size)
{
  /* update conf settings from params */
  dt_lib_collect_params_t *p = (dt_lib_collect_params_t *)params;
  char confname[200] = { 0 };

  for(uint32_t i = 0; i < p->rules; i++)
  {
    /* set item */
    snprintf(confname, sizeof(confname), "plugins/lighttable/collect/item%1u", i);
    dt_conf_set_int(confname, p->rule[i].item);

    /* set mode */
    snprintf(confname, sizeof(confname), "plugins/lighttable/collect/mode%1u", i);
    dt_conf_set_int(confname, p->rule[i].mode);

    /* set string */
    snprintf(confname, sizeof(confname), "plugins/lighttable/collect/string%1u", i);
    dt_conf_set_string(confname, p->rule[i].string);
  }

  /* set number of rules */
  g_strlcpy(confname, "plugins/lighttable/collect/num_rules", sizeof(confname));
  dt_conf_set_int(confname, p->rules);

  /* update internal params */
  _lib_collect_update_params(self->data);

  /* update ui */
  _lib_collect_gui_update(self);

  /* update view */
  dt_collection_update_query(darktable.collection, DT_COLLECTION_CHANGE_NEW_QUERY, DT_COLLECTION_PROP_UNDEF, NULL);
  return 0;
}


const char **views(dt_lib_module_t *self)
{
  static const char *v[] = {"lighttable", "map", "print", NULL};
  return v;
}

uint32_t container(dt_lib_module_t *self)
{
  return DT_UI_CONTAINER_PANEL_LEFT_CENTER;
}

static void view_popup_menu_onSearchFilmroll(GtkWidget *menuitem, gpointer userdata)
{
  GtkTreeView *treeview = GTK_TREE_VIEW(userdata);
  GtkWidget *win = dt_ui_main_window(darktable.gui->ui);

  GtkTreeSelection *selection;
  GtkTreeIter iter, child;
  GtkTreeModel *model;

  gchar *tree_path = NULL;
  gchar *new_path = NULL;

  model = gtk_tree_view_get_model(treeview);
  selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(treeview));
  if (!gtk_tree_selection_get_selected(selection, &model, &iter))
    return;

  child = iter;
  gtk_tree_model_iter_parent(model, &iter, &child);
  gtk_tree_model_get(model, &child, DT_LIB_COLLECT_COL_PATH, &tree_path, -1);

  GtkFileChooserNative *filechooser = gtk_file_chooser_native_new(
          _("search filmroll"), GTK_WINDOW(win), GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER,
          _("_open"), _("_cancel"));

  if(tree_path != NULL)
    gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(filechooser), tree_path);
  else
    goto error;

  // run the dialog
  if(gtk_native_dialog_run(GTK_NATIVE_DIALOG(filechooser)) == GTK_RESPONSE_ACCEPT)
  {
    gint id = -1;
    sqlite3_stmt *stmt;
    gchar *query = NULL;

    gchar *uri = NULL;
    uri = gtk_file_chooser_get_uri(GTK_FILE_CHOOSER(filechooser));
    new_path = g_filename_from_uri(uri, NULL, NULL);
    g_free(uri);
    if(new_path)
    {
      gchar *old = NULL;

      gchar *q_tree_path = g_strdup_printf("%s%%", tree_path);
      query = "SELECT id, folder FROM main.film_rolls WHERE folder LIKE ?1";
      DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db), query, -1, &stmt, NULL);
      DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, q_tree_path, -1, SQLITE_TRANSIENT);
      g_free(q_tree_path);
      q_tree_path = NULL;
      query = NULL;

      while(sqlite3_step(stmt) == SQLITE_ROW)
      {
        id = sqlite3_column_int(stmt, 0);
        old = (gchar *)sqlite3_column_text(stmt, 1);

        query = g_strdup("UPDATE main.film_rolls SET folder=?1 WHERE id=?2");

        gchar trailing[1024] = { 0 };
        gchar final[1024] = { 0 };

        if(g_strcmp0(old, tree_path))
        {
          g_strlcpy(trailing, old + strlen(tree_path) + 1, sizeof(trailing));
          g_snprintf(final, sizeof(final), "%s/%s", new_path, trailing);
        }
        else
        {
          g_strlcpy(final, new_path, sizeof(final));
        }

        sqlite3_stmt *stmt2;
        DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db), query, -1, &stmt2, NULL);
        DT_DEBUG_SQLITE3_BIND_TEXT(stmt2, 1, final, -1, SQLITE_STATIC);
        DT_DEBUG_SQLITE3_BIND_INT(stmt2, 2, id);
        sqlite3_step(stmt2);
        sqlite3_finalize(stmt2);
      }
      sqlite3_finalize(stmt);
      g_free(query);

      // refresh the folders status
      dt_film_set_folder_status();

      /* update collection to view missing filmroll */
      _lib_folders_update_collection(new_path);

      DT_DEBUG_CONTROL_SIGNAL_RAISE(darktable.signals, DT_SIGNAL_FILMROLLS_CHANGED);
    }
    else
      goto error;
  }
  g_free(tree_path);
  g_free(new_path);
  g_object_unref(filechooser);
  return;

error:
  /* Something wrong happened */
  g_object_unref(filechooser);
  dt_control_log(_("problem selecting new path for the filmroll in %s"), tree_path);

  g_free(tree_path);
  g_free(new_path);
}

static void view_popup_menu_onRemove(GtkWidget *menuitem, gpointer userdata)
{
  GtkTreeView *treeview = GTK_TREE_VIEW(userdata);

  GtkTreeSelection *selection;
  GtkTreeIter iter, model_iter;
  GtkTreeModel *model;

  /* Get info about the filmroll (or parent) selected */
  model = gtk_tree_view_get_model(treeview);
  selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(treeview));
  if (gtk_tree_selection_get_selected(selection, &model, &iter))
  {
    gchar *filmroll_path = NULL;
    gchar *fullq = NULL;

    gtk_tree_model_get(model, &iter, DT_LIB_COLLECT_COL_PATH, &filmroll_path, -1);

    /* Clean selected images, and add to the table those which are going to be deleted */
    dt_selection_clear(darktable.selection);

    // FIXME:
    // This should directly feed a GList of imgids to remove to dt_control_remove_images() instead of directly tampering with
    // main.selected_images

    // clang-format off
    fullq = g_strdup_printf("INSERT INTO main.selected_images"
                            " SELECT id"
                            " FROM main.images"
                            " WHERE film_id IN (SELECT id FROM main.film_rolls WHERE folder LIKE '%s%%')",
                            filmroll_path);
    // clang-format on
    DT_DEBUG_SQLITE3_EXEC(dt_database_get(darktable.db), fullq, NULL, NULL, NULL);
    g_free(filmroll_path);

    dt_selection_reload_from_database(darktable.selection);

    if (dt_control_remove_images())
    {
      gtk_tree_model_filter_convert_iter_to_child_iter(GTK_TREE_MODEL_FILTER(model), &model_iter, &iter);

      if (gtk_tree_model_get_flags(model) == GTK_TREE_MODEL_LIST_ONLY)
      {
        gtk_list_store_remove(GTK_LIST_STORE(gtk_tree_model_filter_get_model(GTK_TREE_MODEL_FILTER(model))),
                              &model_iter);
      }
      else
      {
        gtk_tree_store_remove(GTK_TREE_STORE(gtk_tree_model_filter_get_model(GTK_TREE_MODEL_FILTER(model))),
                              &model_iter);
      }
    }

    g_free(fullq);
  }
}

static void view_popup_menu(GtkWidget *treeview, GdkEventButton *event, dt_lib_collect_t *d)
{
  GtkWidget *menu, *menuitem;

  menu = gtk_menu_new();

  menuitem = gtk_menu_item_new_with_label(_("search filmroll..."));
  g_signal_connect(menuitem, "activate", (GCallback)view_popup_menu_onSearchFilmroll, treeview);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), menuitem);

  menuitem = gtk_menu_item_new_with_label(_("remove..."));
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), menuitem);
  g_signal_connect(menuitem, "activate", (GCallback)view_popup_menu_onRemove, treeview);

  gtk_widget_show_all(GTK_WIDGET(menu));

  gtk_menu_popup_at_pointer(GTK_MENU(menu), (GdkEvent *)event);
}

static gboolean view_onButtonPressed(GtkWidget *treeview, GdkEventButton *event, dt_lib_collect_t *d)
{
  /* Get tree path for row that was clicked */
  GtkTreePath *path = NULL;
  int get_path = gtk_tree_view_get_path_at_pos(GTK_TREE_VIEW(treeview), (gint)event->x, (gint)event->y, &path, NULL, NULL, NULL);
  if(!get_path)
  {
    gtk_tree_path_free(path);
    return FALSE;
  }

  GtkTreeSelection *selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(treeview));

  if(item_is_folder_collection(d->view_rule) &&
      event->type == GDK_BUTTON_PRESS &&
      event->button == 3 &&
      gtk_tree_selection_path_is_selected(selection, path) &&
      FALSE)
  {
    // Single right-click on filmroll/folder: open contextual menu
    view_popup_menu(treeview, event, d);
    gtk_tree_path_free(path);
    return TRUE;
  }
  else if(event->button == 1 && event->type == GDK_BUTTON_PRESS)
  {
    // Single left-click

    if(dt_modifier_is(event->state, GDK_SHIFT_MASK) && item_is_numeric_collection(d->view_rule))
    {
      // Shift+Click and a numeric selection : trigger range selection
      gtk_tree_selection_select_path(selection, path);

      GtkTreeModel *model = NULL;
      GList *selected = gtk_tree_selection_get_selected_rows(selection, &model);
      GtkTreePath *first_path = (GtkTreePath *)g_list_first(selected)->data;
      GtkTreePath *last_path = (GtkTreePath *)g_list_last(selected)->data;
      gtk_tree_selection_select_range(selection, first_path, last_path);
      g_list_free_full(selected, (GDestroyNotify)gtk_tree_path_free);
      row_activated_with_event(GTK_TREE_VIEW(treeview), path, NULL, event, d);
    }
    else if(!dt_modifier_is(event->state, GDK_SHIFT_MASK))
    {
      // Toggle expand/collapse state except on Shift because it excludes sub-folders
      if(gtk_tree_view_row_expanded(GTK_TREE_VIEW(treeview), path))
        gtk_tree_view_collapse_row(GTK_TREE_VIEW(treeview), path);
      else
        gtk_tree_view_expand_row(GTK_TREE_VIEW(treeview), path, FALSE);
    }

    gtk_tree_path_free(path);
    return TRUE;
  }
  else if(event->type == GDK_2BUTTON_PRESS)
  {
    // Double click alone : single selection
    gtk_tree_selection_unselect_all(selection);
    gtk_tree_selection_select_path(selection, path);

    row_activated_with_event(GTK_TREE_VIEW(treeview), path, NULL, event, d);
    gtk_tree_path_free(path);
    return TRUE;
  }
  else
  {
    gtk_tree_path_free(path);
    return FALSE; /* we did not handle this */
  }
}

static gboolean view_onPopupMenu(GtkWidget *treeview, dt_lib_collect_t *d)
{
  if(d->view_rule != DT_COLLECTION_PROP_FOLDERS)
  {
    return FALSE;
  }

  view_popup_menu(treeview, NULL, d);

  return TRUE; /* we handled this */
}

static dt_lib_collect_t *get_collect(dt_lib_collect_rule_t *r)
{
  dt_lib_collect_t *d = (dt_lib_collect_t *)r->lib_collect;
  return d;
}

static dt_lib_collect_rule_t * get_active_rule(dt_lib_collect_t *d)
{
  // Get the pointer to the currently active view
  return d->rule + d->active_rule;
}

static void get_number_of_rules(dt_lib_collect_t *d)
{
  d->nb_rules = CLAMP(dt_conf_get_int("plugins/lighttable/collect/num_rules"), 1, MAX_RULES - 1);
  d->active_rule = d->nb_rules - 1;
}

static gboolean list_select(GtkTreeModel *model, GtkTreePath *path, GtkTreeIter *iter, gpointer data)
{
  dt_lib_collect_rule_t *dr = (dt_lib_collect_rule_t *)data;
  dt_lib_collect_t *d = get_collect(dr);
  gchar *str = NULL;

  gtk_tree_model_get(model, iter, DT_LIB_COLLECT_COL_PATH, &str, -1);

  gchar *haystack = g_utf8_strdown(str, -1);
  gchar *needle = g_utf8_strdown(gtk_entry_get_text(GTK_ENTRY(dr->text)), -1);

  if(strcmp(haystack, needle) == 0)
  {
    gtk_tree_selection_select_path(gtk_tree_view_get_selection(d->view), path);
    gtk_tree_view_scroll_to_cell(d->view, path, NULL, FALSE, 0.2, 0);
  }

  g_free(haystack);
  g_free(needle);
  g_free(str);

  return FALSE;
}

static gboolean range_select(GtkTreeModel *model, GtkTreePath *path, GtkTreeIter *iter, gpointer data)
{
  _range_t *range = (_range_t *)data;
  gchar *str = NULL;

  gtk_tree_model_get(model, iter, DT_LIB_COLLECT_COL_PATH, &str, -1);

  gchar *haystack = g_utf8_strdown(str, -1);
  gchar *needle;
  if(range->path1)
    needle = g_utf8_strdown(range->stop, -1);
  else
    needle = g_utf8_strdown(range->start, -1);

  if(strcmp(haystack, needle) == 0)
  {
    if(range->path1)
    {
      range->path2 = gtk_tree_path_copy(path);
      return TRUE;
    }
    else
      range->path1 = gtk_tree_path_copy(path);
  }

  g_free(haystack);
  g_free(needle);
  g_free(str);

  return FALSE;
}

int _combo_get_active_collection(GtkWidget *combo)
{
  return GPOINTER_TO_UINT(dt_bauhaus_combobox_get_data(combo)) - 1;
}

gboolean _combo_set_active_collection(GtkWidget *combo, const int property)
{
  const gboolean found = dt_bauhaus_combobox_set_from_value(combo, property + 1);
  // make sure we have a valid collection
  if(!found) dt_bauhaus_combobox_set_from_value(combo, DT_COLLECTION_PROP_FILMROLL + 1);
  return found;
}

static gboolean tree_expand(GtkTreeModel *model, GtkTreePath *path, GtkTreeIter *iter, gpointer data)
{
  dt_lib_collect_rule_t *dr = (dt_lib_collect_rule_t *)data;
  dt_lib_collect_t *d = get_collect(dr);
  gchar *str = NULL;
  gchar *txt = NULL;
  gboolean startwildcard = FALSE;
  gboolean expanded = FALSE;

  gtk_tree_model_get(model, iter, DT_LIB_COLLECT_COL_PATH, &str, DT_LIB_COLLECT_COL_TEXT, &txt, -1);

  gchar *haystack = g_utf8_strdown(str, -1);
  gchar *needle = g_utf8_strdown(gtk_entry_get_text(GTK_ENTRY(dr->text)), -1);
  gchar *txt2 = g_utf8_strdown(txt, -1);

  if(g_str_has_prefix(needle, "%")) startwildcard = TRUE;
  if(g_str_has_suffix(needle, "%")) needle[strlen(needle) - 1] = '\0';
  if(g_str_has_suffix(haystack, "%")) haystack[strlen(haystack) - 1] = '\0';
  if(_combo_get_active_collection(dr->combo) == DT_COLLECTION_PROP_TAG
     || _combo_get_active_collection(dr->combo) == DT_COLLECTION_PROP_GEOTAGGING)
  {
    if(g_str_has_suffix(needle, "|")) needle[strlen(needle) - 1] = '\0';
    if(g_str_has_suffix(haystack, "|")) haystack[strlen(haystack) - 1] = '\0';
  }
  else if(_combo_get_active_collection(dr->combo) == DT_COLLECTION_PROP_FOLDERS)
  {
    if(g_str_has_suffix(needle, "*")) needle[strlen(needle) - 1] = '\0';
    if(g_str_has_suffix(needle, "/")) needle[strlen(needle) - 1] = '\0';
    if(g_str_has_suffix(haystack, "/")) haystack[strlen(haystack) - 1] = '\0';
  }
  else
  {
    const int temp = _combo_get_active_collection(dr->combo);
    if(DT_COLLECTION_PROP_DAY == temp || is_time_property(temp))
    {
      if(g_str_has_suffix(needle, ":")) needle[strlen(needle) - 1] = '\0';
      if(g_str_has_suffix(haystack, ":")) haystack[strlen(haystack) - 1] = '\0';
    }
  }
  if(dr->typing && g_strrstr(txt2, needle) != NULL)
  {
    gtk_tree_view_expand_to_path(d->view, path);
    expanded = TRUE;
  }

  if(strlen(needle)==0)
  {
    //nothing to do, we keep the tree collapsed
  }
  else if(strcmp(haystack, needle) == 0)
  {
    gtk_tree_view_expand_to_path(d->view, path);
    gtk_tree_selection_select_path(gtk_tree_view_get_selection(d->view), path);
    gtk_tree_view_scroll_to_cell(d->view, path, NULL, FALSE, 0.2, 0);
    expanded = TRUE;
  }
  else if(startwildcard && g_strrstr(haystack, needle+1) != NULL)
  {
    gtk_tree_view_expand_to_path(d->view, path);
    expanded = TRUE;
  }
  else if((dr->typing || _combo_get_active_collection(dr->combo) != DT_COLLECTION_PROP_FOLDERS)
          && g_str_has_prefix(haystack, needle))
  {
    gtk_tree_view_expand_to_path(d->view, path);
    expanded = TRUE;
  }

  g_free(haystack);
  g_free(needle);
  g_free(txt2);
  g_free(str);
  g_free(txt);

  return expanded; // if we expanded the path, we can stop iteration (saves half on average)
}

static gboolean list_match_string(GtkTreeModel *model, GtkTreePath *path, GtkTreeIter *iter, gpointer data)
{
  dt_lib_collect_rule_t *dr = (dt_lib_collect_rule_t *)data;
  gchar *str = NULL;
  gboolean visible = FALSE;
  gboolean was_visible;
  gtk_tree_model_get(model, iter, DT_LIB_COLLECT_COL_PATH, &str, DT_LIB_COLLECT_COL_VISIBLE, &was_visible, -1);

  gchar *haystack = g_utf8_strdown(str, -1);
  const gchar *needle = dr->searchstring;

  const int property = _combo_get_active_collection(dr->combo);
  if(property == DT_COLLECTION_PROP_APERTURE
     || property == DT_COLLECTION_PROP_FOCAL_LENGTH
     || property == DT_COLLECTION_PROP_ISO
     || property == DT_COLLECTION_PROP_RATING)
  {
    // handle of numeric value, which can have some operator before the text
    visible = TRUE;
    gchar *operator, *number, *number2;
    dt_collection_split_operator_number(needle, &number, &number2, &operator);
    if(number)
    {
      const float nb1 = g_strtod(number, NULL);
      const float nb2 = g_strtod(haystack, NULL);
      if(operator && strcmp(operator, ">") == 0)
      {
        visible = (nb2 > nb1);
      }
      else if(operator && strcmp(operator, ">=") == 0)
      {
        visible = (nb2 >= nb1);
      }
      else if(operator && strcmp(operator, "<") == 0)
      {
        visible = (nb2 < nb1);
      }
      else if(operator && strcmp(operator, "<=") == 0)
      {
        visible = (nb2 <= nb1);
      }
      else if(operator && strcmp(operator, "<>") == 0)
      {
        visible = (nb1 != nb2);
      }
      else if(operator && number2 && strcmp(operator, "[]") == 0)
      {
        const float nb3 = g_strtod(number2, NULL);
        visible = (nb2 >= nb1 && nb2 <= nb3);
      }
      else
      {
        visible = (nb1 == nb2);
      }
    }
    g_free(operator);
    g_free(number);
    g_free(number2);
  }
  else if (property == DT_COLLECTION_PROP_FILENAME && strchr(needle,',') != NULL)
  {
    GList *list = dt_util_str_to_glist(",", needle);

    for (const GList *l = list; l; l = g_list_next(l))
    {
      const char *name = (char *)l->data;
      if((visible = (g_strrstr(haystack, name + (name[0]=='%')) != NULL))) break;
    }

    g_list_free_full(list, g_free);

  }
  else
  {
    if (needle[0] == '%')
      needle++;
    if (!needle[0])
    {
      // empty search string matches all
      visible = TRUE;
    }
    else if (!needle[1])
    {
      // single-char search, use faster strchr instead of strstr
      visible = (strchr(haystack, needle[0]) != NULL);
    }
    else
    {
      visible = (g_strrstr(haystack, needle) != NULL);
    }
  }

  g_free(haystack);
  g_free(str);

  if (visible != was_visible)
    gtk_list_store_set(GTK_LIST_STORE(model), iter, DT_LIB_COLLECT_COL_VISIBLE, visible, -1);
  return FALSE;
}

static gboolean tree_match_string(GtkTreeModel *model, GtkTreePath *path, GtkTreeIter *iter, gpointer data)
{
  dt_lib_collect_rule_t *dr = (dt_lib_collect_rule_t *)data;
  gchar *str = NULL;
  gboolean cur_state, visible;

  gtk_tree_model_get(model, iter, DT_LIB_COLLECT_COL_PATH, &str, DT_LIB_COLLECT_COL_VISIBLE, &cur_state, -1);

  if(dr->typing == FALSE && !cur_state)
  {
    visible = TRUE;
  }
  else
  {
    gchar *haystack = g_utf8_strdown(str, -1),
          *needle = g_utf8_strdown(gtk_entry_get_text(GTK_ENTRY(dr->text)), -1);
    visible = (g_strrstr(haystack, needle) != NULL);
    g_free(haystack);
    g_free(needle);
  }

  g_free(str);

  gtk_tree_store_set(GTK_TREE_STORE(model), iter, DT_LIB_COLLECT_COL_VISIBLE, visible, -1);
  return FALSE;
}

static gboolean tree_reveal_func(GtkTreeModel *model, GtkTreePath *path, GtkTreeIter *iter, gpointer data)
{
  gboolean state;
  GtkTreeIter parent, child = *iter;

  gtk_tree_model_get(model, iter, DT_LIB_COLLECT_COL_VISIBLE, &state, -1);
  if(!state) return FALSE;

  while(gtk_tree_model_iter_parent(model, &parent, &child))
  {
    gtk_tree_model_get(model, &parent, DT_LIB_COLLECT_COL_VISIBLE, &state, -1);
    gtk_tree_store_set(GTK_TREE_STORE(model), &parent, DT_LIB_COLLECT_COL_VISIBLE, TRUE, -1);
    child = parent;
  }

  return FALSE;
}

static void tree_set_visibility(GtkTreeModel *model, gpointer data)
{
  gtk_tree_model_foreach(model, (GtkTreeModelForeachFunc)tree_match_string, data);

  gtk_tree_model_foreach(model, (GtkTreeModelForeachFunc)tree_reveal_func, NULL);
}

static void _lib_folders_update_collection(const gchar *filmroll)
{
  /* raise signal of collection change, only if this is an original */
  dt_collection_memory_update();
  DT_DEBUG_CONTROL_SIGNAL_RAISE(darktable.signals, DT_SIGNAL_COLLECTION_CHANGED, DT_COLLECTION_CHANGE_NEW_QUERY,
                                DT_COLLECTION_PROP_UNDEF, (GList *)NULL, -1);
}

static void set_properties(dt_lib_collect_rule_t *dr)
{
  // Store GUI params into conf keys
  const int property = _combo_get_active_collection(dr->combo);
  const gchar *text = gtk_entry_get_text(GTK_ENTRY(dr->text));

  char confname[200] = { 0 };
  snprintf(confname, sizeof(confname), "plugins/lighttable/collect/string%1d", dr->num);
  dt_conf_set_string(confname, text);
  snprintf(confname, sizeof(confname), "plugins/lighttable/collect/item%1d", dr->num);
  dt_conf_set_int(confname, property);
}

static void get_properties(dt_lib_collect_rule_t *dr)
{
  // Set GUI controls from conf keys
  char confname[200] = { 0 };
  snprintf(confname, sizeof(confname), "plugins/lighttable/collect/item%1d", dr->num);
  _combo_set_active_collection(dr->combo, dt_conf_get_int(confname));
  snprintf(confname, sizeof(confname), "plugins/lighttable/collect/string%1d", dr->num);
  const char *text = dt_conf_get_string_const(confname);

  if(text)
  {
    g_signal_handlers_block_matched(dr->text, G_SIGNAL_MATCH_FUNC, 0, 0, NULL, entry_changed, NULL);
    gtk_entry_set_text(GTK_ENTRY(dr->text), text);
    gtk_editable_set_position(GTK_EDITABLE(dr->text), -1);
    dr->typing = FALSE;
    g_signal_handlers_unblock_matched(dr->text, G_SIGNAL_MATCH_FUNC, 0, 0, NULL, entry_changed, NULL);
  }
}

static GtkTreeModel *_create_filtered_model(GtkTreeModel *model, dt_lib_collect_rule_t *dr)
{
  GtkTreeModel *filter = NULL;
  GtkTreePath *path = NULL;

  if(_combo_get_active_collection(dr->combo) == DT_COLLECTION_PROP_FOLDERS)
  {
    // we search a common path to all the folders
    // we'll use it as root
    GtkTreeIter child, iter;
    int level = 0;

    while(gtk_tree_model_iter_n_children(model, level > 0 ? &iter : NULL) > 0)
    {
      if(level > 0)
      {
        sqlite3_stmt *stmt = NULL;
        gchar *pth = NULL;
        int id = -1;
        // Check if this path also matches a filmroll
        gtk_tree_model_get(model, &iter, DT_LIB_COLLECT_COL_PATH, &pth, -1);
        DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                                    "SELECT id FROM main.film_rolls WHERE folder LIKE ?1", -1, &stmt, NULL);
        DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, pth, -1, SQLITE_TRANSIENT);
        if(sqlite3_step(stmt) == SQLITE_ROW) id = sqlite3_column_int(stmt, 0);
        sqlite3_finalize(stmt);

        g_free(pth);

        if(id != -1)
        {
          // we go back to the parent, in order to show this folder
          if(!gtk_tree_model_iter_parent(model, &child, &iter)) level = 0;
          iter = child;
          break;
        }
      }
      if(gtk_tree_model_iter_n_children(model, level > 0 ? &iter : NULL) != 1) break;

      gtk_tree_model_iter_children(model, &child, level > 0 ? &iter : NULL);
      iter = child;
      level++;
    }

    if(level > 0)
    {
      if(gtk_tree_model_iter_n_children(model, &iter) == 0 &&
         gtk_tree_model_iter_parent(model, &child, &iter))
      {
        path = gtk_tree_model_get_path(model, &child);
      }
      else
      {
        path = gtk_tree_model_get_path(model, &iter);
      }
    }
  }

  // Create filter and set virtual root
  filter = gtk_tree_model_filter_new(model, path);
  gtk_tree_path_free(path);

  gtk_tree_model_filter_set_visible_column(GTK_TREE_MODEL_FILTER(filter), DT_LIB_COLLECT_COL_VISIBLE);

  return filter;
}

static int string_array_length(char **list)
{
  int length = 0;
  for(; *list; list++) length++;
  return length;
}

// returns a NULL terminated array of path components
static char **split_path(const char *path)
{
  if(!path || !*path) return NULL;

  char **result;
  char **tokens = g_strsplit(path, G_DIR_SEPARATOR_S, -1);

#ifdef _WIN32

  if(! (g_ascii_isalpha(tokens[0][0]) && tokens[0][strlen(tokens[0]) - 1] == ':') )
  {
    g_strfreev(tokens);
    tokens = NULL;
  }

  result = tokens;

#else

  // there are size + 1 elements in tokens -- the final NULL! we want to ignore it.
  const unsigned int size = g_strv_length(tokens);

  result = malloc(sizeof(char *) * size);
  for(unsigned int i = 0; i < size; i++)
    result[i] = tokens[i + 1];

  g_free(tokens[0]);
  g_free(tokens);

#endif

  return result;
}

typedef struct name_key_tuple_t
{
  char *name, *collate_key;
  int count, status;
} name_key_tuple_t;

static void free_tuple(gpointer data)
{
  name_key_tuple_t *tuple = (name_key_tuple_t *)data;
  g_free(tuple->name);
  g_free(tuple->collate_key);
  free(tuple);
}

static gint sort_folder_tag(gconstpointer a, gconstpointer b)
{
  const name_key_tuple_t *tuple_a = (const name_key_tuple_t *)a;
  const name_key_tuple_t *tuple_b = (const name_key_tuple_t *)b;

  return g_strcmp0(tuple_a->collate_key, tuple_b->collate_key);
}

// create a key such that  _("not tagged") & "darktable|" are coming first,
// and the rest is ordered such that sub tags are coming directly behind their parent
static char *tag_collate_key(char *tag)
{
  const size_t len = strlen(tag);
  char *result = g_malloc(len + 2);
  if(!g_strcmp0(tag, _("not tagged")))
    *result = '\1';
  else if(g_str_has_prefix(tag, "darktable|"))
    *result = '\2';
  else
    *result = '\3';
  memcpy(result + 1, tag, len + 1);
  for(char *iter = result + 1; *iter; iter++)
    if(*iter == '|') *iter = '\1';
  return result;
}


void tree_count_show(GtkTreeViewColumn *col, GtkCellRenderer *renderer, GtkTreeModel *model, GtkTreeIter *iter,
                     gpointer data)
{
  gchar *name;
  guint count;

  gtk_tree_model_get(model, iter, DT_LIB_COLLECT_COL_TEXT, &name, DT_LIB_COLLECT_COL_COUNT, &count, -1);
  if (!count)
  {
    g_object_set(renderer, "text", name, NULL);
  }
  else
  {
    gchar *coltext = g_strdup_printf("%s (%d)", name, count);
    g_object_set(renderer, "text", coltext, NULL);
    g_free(coltext);
  }

  g_free(name);
}

static const char *UNCATEGORIZED_TAG = N_("uncategorized");
static void tree_view(dt_lib_collect_rule_t *dr)
{
  // update related list
  dt_lib_collect_t *d = get_collect(dr);
  const int property = _combo_get_active_collection(dr->combo);
  char *format_separator = "";

  switch(property)
  {
    case DT_COLLECTION_PROP_FOLDERS:
      format_separator = "%s" G_DIR_SEPARATOR_S;
      break;
    case DT_COLLECTION_PROP_TAG:
    case DT_COLLECTION_PROP_GEOTAGGING:
      format_separator = "%s|";
      break;
    case DT_COLLECTION_PROP_DAY:
    case DT_COLLECTION_PROP_TIME:
    case DT_COLLECTION_PROP_IMPORT_TIMESTAMP:
    case DT_COLLECTION_PROP_CHANGE_TIMESTAMP:
    case DT_COLLECTION_PROP_EXPORT_TIMESTAMP:
    case DT_COLLECTION_PROP_PRINT_TIMESTAMP:
      format_separator = "%s:";
      break;
  }

  set_properties(dr);

  GtkTreeModel *model = gtk_tree_model_filter_get_model(GTK_TREE_MODEL_FILTER(d->treefilter));
  // since we'll be inserting elements in the same order that we want them displayed, there is no need for the
  // overhead of having the tree view sort itself
  gtk_tree_sortable_set_sort_column_id(GTK_TREE_SORTABLE(model),
                                       GTK_TREE_SORTABLE_UNSORTED_SORT_COLUMN_ID, GTK_SORT_ASCENDING);

  if(d->view_rule != property)
  {
    // tree creation/recreation
    sqlite3_stmt *stmt;
    GtkTreeIter uncategorized = { 0 };
    GtkTreeIter temp;

    g_object_ref(model);
    g_object_unref(d->treefilter);
    gtk_tree_view_set_model(GTK_TREE_VIEW(d->view), NULL);
    gtk_tree_store_clear(GTK_TREE_STORE(model));
    gtk_widget_hide(GTK_WIDGET(d->view));

    /* query construction */
    gchar *where_ext = dt_collection_get_extended_where(darktable.collection, dr->num);
    gchar *query = 0;
    switch (property)
    {
      case DT_COLLECTION_PROP_FOLDERS:
        // clang-format off
        query = g_strdup_printf("SELECT folder, film_rolls_id, COUNT(*) AS count, status"
                                " FROM main.images AS mi"
                                " JOIN (SELECT fr.id AS film_rolls_id, folder, status"
                                "       FROM main.film_rolls AS fr"
                                "       JOIN memory.film_folder AS ff"
                                "       ON fr.id = ff.id)"
                                "   ON film_id = film_rolls_id "
                                " WHERE %s"
                                " GROUP BY folder, film_rolls_id", where_ext);
          // clang-format on
        break;
      case DT_COLLECTION_PROP_TAG:
      {
        // clang-format off
        query = g_strdup_printf("SELECT name, 1 AS tagid, SUM(count) AS count"
                                " FROM (SELECT tagid, COUNT(*) as count"
                                "   FROM main.images AS mi"
                                "   JOIN main.tagged_images"
                                "     ON id = imgid "
                                "   WHERE %s"
                                "   GROUP BY tagid)"
                                " JOIN (SELECT LOWER(name) AS name, id AS tag_id FROM data.tags)"
                                "   ON tagid = tag_id"
                                "   GROUP BY name", where_ext);

        query = dt_util_dstrcat(query, " UNION ALL "
                                       "SELECT '%s' AS name, 0 as id, COUNT(*) AS count "
                                       "FROM main.images AS mi "
                                       "WHERE mi.id NOT IN"
                                       "  (SELECT DISTINCT imgid FROM main.tagged_images AS ti"
                                       "   WHERE ti.tagid NOT IN memory.darktable_tags)",
                                _("not tagged"));
        // clang-format on
      }
      break;
      case DT_COLLECTION_PROP_GEOTAGGING:
        // clang-format off
        query = g_strdup_printf("SELECT "
                                " CASE WHEN mi.longitude IS NULL"
                                "           OR mi.latitude IS null THEN \'%s\'"
                                "      ELSE CASE WHEN ta.imgid IS NULL THEN \'%s\'"
                                "                ELSE \'%s\' || ta.tagname"
                                "                END"
                                "      END AS name,"
                                " ta.tagid AS tag_id, COUNT(*) AS count"
                                " FROM main.images AS mi"
                                " LEFT JOIN (SELECT imgid, t.id AS tagid, SUBSTR(t.name, %d) AS tagname"
                                "   FROM main.tagged_images AS ti"
                                "   JOIN data.tags AS t"
                                "     ON ti.tagid = t.id"
                                "   JOIN data.locations AS l"
                                "     ON l.tagid = t.id"
                                "   ) AS ta ON ta.imgid = mi.id"
                                " WHERE %s"
                                " GROUP BY name, tag_id",
                                _("not tagged"), _("tagged"), _("tagged"),
                                (int)strlen(dt_map_location_data_tag_root()) + 1, where_ext);
        // clang-format on
        break;
      case DT_COLLECTION_PROP_DAY:
        // clang-format off
        query = g_strdup_printf("SELECT (datetime_taken / 86400000000) * 86400000000 AS date, 1, COUNT(*) AS count"
                                " FROM main.images AS mi"
                                " WHERE datetime_taken IS NOT NULL AND datetime_taken <> 0"
                                " AND %s"
                                " GROUP BY date", where_ext);
        // clang-format on
        break;
      case DT_COLLECTION_PROP_TIME:
      case DT_COLLECTION_PROP_IMPORT_TIMESTAMP:
      case DT_COLLECTION_PROP_CHANGE_TIMESTAMP:
      case DT_COLLECTION_PROP_EXPORT_TIMESTAMP:
      case DT_COLLECTION_PROP_PRINT_TIMESTAMP:
        {
        const int local_property = property;
        char *colname = NULL;

        switch(local_property)
        {
          case DT_COLLECTION_PROP_TIME: colname = "datetime_taken" ; break ;
          case DT_COLLECTION_PROP_IMPORT_TIMESTAMP: colname = "import_timestamp" ; break ;
          case DT_COLLECTION_PROP_CHANGE_TIMESTAMP: colname = "change_timestamp" ; break ;
          case DT_COLLECTION_PROP_EXPORT_TIMESTAMP: colname = "export_timestamp" ; break ;
          case DT_COLLECTION_PROP_PRINT_TIMESTAMP: colname = "print_timestamp" ; break ;
        }
        // clang-format off
        query = g_strdup_printf("SELECT %s AS date, 1, COUNT(*) AS count"
                                " FROM main.images AS mi"
                                " WHERE %s IS NOT NULL AND %s <> 0"
                                " AND %s"
                                " GROUP BY date", colname, colname, colname, where_ext);
        // clang-format on
        break;
        }
    }

    g_free(where_ext);

    DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db), query, -1, &stmt, NULL);

    char **last_tokens = NULL;
    int last_tokens_length = 0;
    GtkTreeIter last_parent = { 0 };

    // we need to sort the names ourselves and not let sqlite handle this
    // because it knows nothing about path separators.
    GList *sorted_names = NULL;
    guint index = 0;
    while(sqlite3_step(stmt) == SQLITE_ROW)
    {
      char *name;
      if(is_time_property(property) || property == DT_COLLECTION_PROP_DAY)
      {
        char sdt[DT_DATETIME_EXIF_LENGTH] = {0};
        dt_datetime_gtimespan_to_exif(sdt, sizeof(sdt), sqlite3_column_int64(stmt, 0));
        if(property == DT_COLLECTION_PROP_DAY)
          sdt[10] = '\0';
        name = g_strdup(sdt);
      }
      else
      {
        const char* sqlite_name = (const char *)sqlite3_column_text(stmt, 0);
        name = sqlite_name == NULL ? g_strdup("") : g_strdup(sqlite_name);
      }
      gchar *collate_key = NULL;

      const int count = sqlite3_column_int(stmt, 2);

      if(property == DT_COLLECTION_PROP_FOLDERS)
      {
        char *name_folded = g_utf8_casefold(name, -1);
        char *name_folded_slash = g_strconcat(name_folded, G_DIR_SEPARATOR_S, NULL);
        collate_key = g_utf8_collate_key_for_filename(name_folded_slash, -1);
        g_free(name_folded_slash);
        g_free(name_folded);
      }
      else
        collate_key = tag_collate_key(name);

      name_key_tuple_t *tuple = (name_key_tuple_t *)malloc(sizeof(name_key_tuple_t));
      tuple->name = name;
      tuple->collate_key = collate_key;
      tuple->count = count;
      tuple->status = property == DT_COLLECTION_PROP_FOLDERS ? sqlite3_column_int(stmt, 3) : -1;
      sorted_names = g_list_prepend(sorted_names, tuple);
    }
    sqlite3_finalize(stmt);
    g_free(query);
    // this order should not be altered. the right feeding of the tree relies on it.
    sorted_names = g_list_sort(sorted_names, sort_folder_tag);
    const gboolean sort_descend = dt_conf_get_bool("plugins/collect/descending");
    if (!sort_descend)
      sorted_names = g_list_reverse(sorted_names);

    gboolean no_uncategorized = (property == DT_COLLECTION_PROP_TAG) ?
                                dt_conf_get_bool("plugins/lighttable/tagging/no_uncategorized")
                                : TRUE;

    for(GList *names = sorted_names; names; names = g_list_next(names))
    {
      name_key_tuple_t *tuple = (name_key_tuple_t *)names->data;
      char *name = tuple->name;
      const int count = tuple->count;
      const int status = tuple->status;
      if(name == NULL) continue; // safeguard against degenerated db entries

      // this is just for tags
      gboolean uncategorized_found = FALSE;
      if(!no_uncategorized && strchr(name, '|') == NULL)
      {
        char *next_name = g_strdup(names->next ? ((name_key_tuple_t *)names->next->data)->name : "");
        if(strlen(next_name) >= strlen(name) + 1 && next_name[strlen(name)] == '|')
          next_name[strlen(name)] = '\0';

        if(g_strcmp0(next_name, name) && g_strcmp0(name, _("not tagged")))
        {
          /* add uncategorized root iter if not exists */
          if(!uncategorized.stamp)
          {
            gtk_tree_store_insert_with_values(GTK_TREE_STORE(model), &uncategorized, NULL, -1,
                                              DT_LIB_COLLECT_COL_TEXT, _(UNCATEGORIZED_TAG),
                                              DT_LIB_COLLECT_COL_PATH, "", DT_LIB_COLLECT_COL_VISIBLE, TRUE,
                                              DT_LIB_COLLECT_COL_INDEX, index, -1);
            index++;
          }

          /* adding an uncategorized tag */
          gtk_tree_store_insert_with_values(GTK_TREE_STORE(model), &temp, &uncategorized, 0,
                                            DT_LIB_COLLECT_COL_TEXT, name,
                                            DT_LIB_COLLECT_COL_PATH, name, DT_LIB_COLLECT_COL_VISIBLE, TRUE,
                                            DT_LIB_COLLECT_COL_COUNT, count, DT_LIB_COLLECT_COL_INDEX, index, -1);
          uncategorized_found = TRUE;
          index++;
        }
        g_free(next_name);
      }

      if(!uncategorized_found)
      {
        char **tokens;
        if(property == DT_COLLECTION_PROP_FOLDERS)
          tokens = split_path(name);
        else if(property == DT_COLLECTION_PROP_DAY)
          tokens = g_strsplit(name, ":", -1);
        else if(is_time_property(property))
          tokens = g_strsplit_set(name, ": ", 4);
        else
          tokens = g_strsplit(name, "|", -1);

        if(tokens != NULL)
        {
          // find the number of common parts at the beginning of tokens and last_tokens
          GtkTreeIter parent = last_parent;
          const int tokens_length = string_array_length(tokens);
          int common_length = 0;
          if(last_tokens)
          {
            while(tokens[common_length]
                  && last_tokens[common_length]
                  && !g_strcmp0(tokens[common_length], last_tokens[common_length]))
            {
              common_length++;
            }

            // point parent iter to where the entries should be added
            for(int i = common_length; i < last_tokens_length; i++)
            {
              gtk_tree_model_iter_parent(model, &parent, &last_parent);
              last_parent = parent;
            }
          }

          // insert everything from tokens past the common part

          char *pth = NULL;
#ifndef _WIN32
          if(property == DT_COLLECTION_PROP_FOLDERS) pth = g_strdup("/");
#endif
          for(int i = 0; i < common_length; i++)
            pth = dt_util_dstrcat(pth, format_separator, tokens[i]);

          for(char **token = &tokens[common_length]; *token; token++)
          {
            GtkTreeIter iter;

            pth = dt_util_dstrcat(pth, format_separator, *token);
            if(is_time_property(property) && !*(token + 1))
              pth[10] = ' ';

            gchar *pth2 = g_strdup(pth);
            pth2[strlen(pth2) - 1] = '\0';
            gtk_tree_store_insert_with_values(GTK_TREE_STORE(model), &iter, common_length > 0 ? &parent : NULL, 0,
                                              DT_LIB_COLLECT_COL_TEXT, *token,
                                              DT_LIB_COLLECT_COL_PATH, pth2, DT_LIB_COLLECT_COL_VISIBLE, TRUE,
                                              DT_LIB_COLLECT_COL_COUNT, (*(token + 1)?0:count),
                                              DT_LIB_COLLECT_COL_INDEX, index,
                                              DT_LIB_COLLECT_COL_UNREACHABLE, (*(token + 1) ? 0 : !status), -1);
            index++;
            // also add the item count to parents
            if((property == DT_COLLECTION_PROP_DAY
                ||  is_time_property(property))
               && !*(token + 1))
            {
              guint parentcount;
              GtkTreeIter parent2, child = iter;

              while(gtk_tree_model_iter_parent(model, &parent2, &child))
              {
                gtk_tree_model_get(model, &parent2, DT_LIB_COLLECT_COL_COUNT, &parentcount, -1);
                gtk_tree_store_set(GTK_TREE_STORE(model), &parent2, DT_LIB_COLLECT_COL_COUNT, count + parentcount, -1);
                child = parent2;
              }
            }

            common_length++;
            parent = iter;
            g_free(pth2);
          }

          g_free(pth);

          // remember things for the next round
          if(last_tokens) g_strfreev(last_tokens);
          last_tokens = tokens;
          last_parent = parent;
          last_tokens_length = tokens_length;
        }
      }
    }
    g_list_free_full(sorted_names, free_tuple);

    gtk_tree_view_set_tooltip_column(GTK_TREE_VIEW(d->view), DT_LIB_COLLECT_COL_TOOLTIP);

    d->treefilter = _create_filtered_model(model, dr);

    GtkTreeSelection *selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(d->view));
    if(property == DT_COLLECTION_PROP_DAY || is_time_property(property))
    {
      gtk_tree_selection_set_mode(selection, GTK_SELECTION_MULTIPLE);
    }
    else
    {
      gtk_tree_selection_set_mode(selection, GTK_SELECTION_SINGLE);
    }

    gtk_tree_view_set_model(GTK_TREE_VIEW(d->view), d->treefilter);
    gtk_widget_set_no_show_all(GTK_WIDGET(d->view), FALSE);
    gtk_widget_show_all(GTK_WIDGET(d->view));

    g_object_unref(model);
    g_strfreev(last_tokens);
    d->view_rule = property;
  }

  // if needed, we restrict the tree to matching entries
  if(dr->typing) tree_set_visibility(model, dr);
  // we update tree expansion and selection
  gtk_tree_selection_unselect_all(gtk_tree_view_get_selection(d->view));
  gtk_tree_view_collapse_all(d->view);

  if(item_is_numeric_collection(property))
  {
    // test selection range [xxx;xxx]
    GRegex *regex;
    GMatchInfo *match_info;
    int match_count;

    regex = g_regex_new("^\\s*\\[\\s*(.*)\\s*;\\s*(.*)\\s*\\]\\s*$", 0, 0, NULL);
    g_regex_match_full(regex, gtk_entry_get_text(GTK_ENTRY(dr->text)), -1, 0, 0, &match_info, NULL);
    match_count = g_match_info_get_match_count(match_info);

    if(match_count == 3)
    {
      _range_t *range = (_range_t *)calloc(1, sizeof(_range_t));
      /* inversed as dates are in reverse order */
      range->start = g_match_info_fetch(match_info, 2);
      range->stop = g_match_info_fetch(match_info, 1);

      gtk_tree_model_foreach(d->treefilter, (GtkTreeModelForeachFunc)range_select, range);
      if(range->path1 && range->path2)
      {
        gtk_tree_selection_select_range(gtk_tree_view_get_selection(d->view), range->path1, range->path2);
      }
      g_free(range->start);
      g_free(range->stop);
      gtk_tree_path_free(range->path1);
      gtk_tree_path_free(range->path2);
      free(range);
    }
    else
      gtk_tree_model_foreach(d->treefilter, (GtkTreeModelForeachFunc)tree_expand, dr);

    g_match_info_free(match_info);
    g_regex_unref(regex);
  }
  else
    gtk_tree_model_foreach(d->treefilter, (GtkTreeModelForeachFunc)tree_expand, dr);
}

static void list_view(dt_lib_collect_rule_t *dr)
{
  // update related list
  dt_lib_collect_t *d = get_collect(dr);
  const int property = _combo_get_active_collection(dr->combo);

  set_properties(dr);

  GtkTreeModel *model = gtk_tree_model_filter_get_model(GTK_TREE_MODEL_FILTER(d->listfilter));
  if(d->view_rule != property)
  {
    sqlite3_stmt *stmt;
    GtkTreeIter iter;
    g_object_unref(d->listfilter);
    g_object_ref(model);
    gtk_tree_view_set_model(GTK_TREE_VIEW(d->view), NULL);
    gtk_list_store_clear(GTK_LIST_STORE(model));
    gtk_widget_hide(GTK_WIDGET(d->view));
    gchar *where_ext = dt_collection_get_extended_where(darktable.collection, dr->num);

    char query[1024] = { 0 };

    switch(property)
    {
      case DT_COLLECTION_PROP_CAMERA:; // camera
        int index = 0;
        // clang-format off
        gchar *makermodel_query = g_strdup_printf("SELECT maker, model, COUNT(*) AS count "
                                                  "FROM main.images AS mi WHERE %s GROUP BY maker, model", where_ext);
        // clang-format on

        DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                                makermodel_query,
                                -1, &stmt, NULL);

        while(sqlite3_step(stmt) == SQLITE_ROW)
        {
          const char *exif_maker = (char *)sqlite3_column_text(stmt, 0);
          const char *exif_model = (char *)sqlite3_column_text(stmt, 1);
          const int count = sqlite3_column_int(stmt, 2);

          gchar *value =  dt_collection_get_makermodel(exif_maker, exif_model);

          gtk_list_store_append(GTK_LIST_STORE(model), &iter);
          gtk_list_store_set(GTK_LIST_STORE(model), &iter, DT_LIB_COLLECT_COL_TEXT, value,
                             DT_LIB_COLLECT_COL_ID, index, DT_LIB_COLLECT_COL_TOOLTIP, value,
                             DT_LIB_COLLECT_COL_PATH, value, DT_LIB_COLLECT_COL_VISIBLE, TRUE,
                             DT_LIB_COLLECT_COL_COUNT, count,
                             -1);

          g_free(value);
          index++;
        }
        g_free(makermodel_query);
        break;

      case DT_COLLECTION_PROP_HISTORY: // History
        // images without history are counted as if they were basic
        // clang-format off
        g_snprintf(query, sizeof(query),
                   "SELECT CASE"
                   "       WHEN basic_hash == current_hash THEN '%s'"
                   "       WHEN auto_hash == current_hash THEN '%s'"
                   "       WHEN current_hash IS NOT NULL THEN '%s'"
                   "       ELSE '%s'"
                   "     END as altered, 1, COUNT(*) AS count"
                   " FROM main.images AS mi"
                   " LEFT JOIN (SELECT DISTINCT imgid, basic_hash, auto_hash, current_hash"
                   "            FROM main.history_hash) ON id = imgid"
                   " WHERE %s"
                   " GROUP BY altered"
                   " ORDER BY altered ASC",
                   _("basic"), _("auto applied"), _("altered"), _("basic"), where_ext);
        // clang-format on
        break;

      case DT_COLLECTION_PROP_LOCAL_COPY: // local copy, 2 hardcoded alternatives
        // clang-format off
        g_snprintf(query, sizeof(query),
                   "SELECT CASE "
                   "         WHEN (flags & %d) THEN '%s'"
                   "         ELSE '%s'"
                   "       END as lcp, 1, COUNT(*) AS count"
                   " FROM main.images AS mi "
                   " WHERE %s"
                   " GROUP BY lcp ORDER BY lcp ASC",
                   DT_IMAGE_LOCAL_COPY, _("copied locally"),  _("not copied locally"), where_ext);
        // clang-format on
        break;

      case DT_COLLECTION_PROP_COLORLABEL: // colorlabels
        // clang-format off
        g_snprintf(query, sizeof(query),
                   "SELECT CASE color"
                   "         WHEN 0 THEN '%s'"
                   "         WHEN 1 THEN '%s'"
                   "         WHEN 2 THEN '%s'"
                   "         WHEN 3 THEN '%s'"
                   "         WHEN 4 THEN '%s' "
                   "         ELSE ''"
                   "       END, color, COUNT(*) AS count"
                   " FROM main.images AS mi"
                   " JOIN "
                   "   (SELECT imgid AS color_labels_id, color FROM main.color_labels)"
                   " ON id = color_labels_id "
                   " WHERE %s"
                   " GROUP BY color"
                   " ORDER BY color DESC",
                   _("red"), _("yellow"), _("green"), _("blue"), _("purple"), where_ext);
        // clang-format on
        break;

      case DT_COLLECTION_PROP_LENS: // lens
        // clang-format off
        g_snprintf(query, sizeof(query),
                   "SELECT lens, 1, COUNT(*) AS count"
                   " FROM main.images AS mi"
                   " WHERE %s"
                   " GROUP BY lens"
                   " ORDER BY lens", where_ext);
        // clang-format on
        break;

      case DT_COLLECTION_PROP_FOCAL_LENGTH: // focal length
        // clang-format off
        g_snprintf(query, sizeof(query),
                   "SELECT CAST(focal_length AS INTEGER) AS focal_length, 1, COUNT(*) AS count"
                   " FROM main.images AS mi"
                   " WHERE %s"
                   " GROUP BY CAST(focal_length AS INTEGER)"
                   " ORDER BY CAST(focal_length AS INTEGER)",
                   where_ext);
        // clang-format on
        break;

      case DT_COLLECTION_PROP_ISO: // iso
        // clang-format off
        g_snprintf(query, sizeof(query),
                   "SELECT CAST(iso AS INTEGER) AS iso, 1, COUNT(*) AS count"
                   " FROM main.images AS mi"
                   " WHERE %s"
                   " GROUP BY iso"
                   " ORDER BY iso",
                   where_ext);
        // clang-format on
        break;

      case DT_COLLECTION_PROP_APERTURE: // aperture
        // clang-format off
        g_snprintf(query, sizeof(query),
                   "SELECT ROUND(aperture,1) AS aperture, 1, COUNT(*) AS count"
                   " FROM main.images AS mi"
                   " WHERE %s"
                   " GROUP BY aperture"
                   " ORDER BY aperture",
                   where_ext);
        // clang-format on
        break;

      case DT_COLLECTION_PROP_EXPOSURE: // exposure
        // clang-format off
        g_snprintf(query, sizeof(query),
                   "SELECT CASE"
                   "         WHEN (exposure < 0.4) THEN '1/' || CAST(1/exposure + 0.9 AS INTEGER) "
                   "         ELSE ROUND(exposure,2) || '\"'"
                   "       END as _exposure, 1, COUNT(*) AS count"
                   " FROM main.images AS mi"
                   " WHERE %s"
                   " GROUP BY _exposure"
                   " ORDER BY exposure",
                  where_ext);
        // clang-format on
        break;

      case DT_COLLECTION_PROP_FILENAME: // filename
        // clang-format off
        g_snprintf(query, sizeof(query),
                   "SELECT filename, 1, COUNT(*) AS count"
                   " FROM main.images AS mi"
                   " WHERE %s"
                   " GROUP BY filename"
                   " ORDER BY filename", where_ext);
        // clang-format on
        break;

      case DT_COLLECTION_PROP_GROUPING: // Grouping, 2 hardcoded alternatives
        // clang-format off
        g_snprintf(query, sizeof(query),
                   "SELECT CASE"
                   "         WHEN id = group_id THEN '%s'"
                   "         ELSE '%s'"
                   "       END as group_leader, 1, COUNT(*) AS count"
                   " FROM main.images AS mi"
                   " WHERE %s"
                   " GROUP BY group_leader"
                   " ORDER BY group_leader ASC",
                   _("group leaders"),  _("group followers"), where_ext);
        // clang-format on
        break;

      case DT_COLLECTION_PROP_MODULE: // module
        // clang-format off
        snprintf(query, sizeof(query),
                 "SELECT m.name AS module_name, 1, COUNT(*) AS count"
                 " FROM main.images AS mi"
                 " JOIN (SELECT DISTINCT imgid, operation FROM main.history WHERE enabled = 1) AS h"
                 "  ON h.imgid = mi.id"
                 " JOIN memory.darktable_iop_names AS m"
                 "  ON m.operation = h.operation"
                 " WHERE %s"
                 " GROUP BY module_name"
                 " ORDER BY module_name",
                 where_ext);
        // clang-format on
        break;

      case DT_COLLECTION_PROP_ORDER: // modules order
        {
          char *orders = NULL;
          for(int i = 0; i < DT_IOP_ORDER_LAST; i++)
          {
            orders = dt_util_dstrcat(orders, "WHEN mo.version = %d THEN '%s' ",
                                     i, _(dt_iop_order_string(i)));
          }
          orders = dt_util_dstrcat(orders, "ELSE '%s' ", _("none"));
          // clang-format off
          snprintf(query, sizeof(query),
                   "SELECT CASE %s END as ver, 1, COUNT(*) AS count"
                   " FROM main.images AS mi"
                   " LEFT JOIN (SELECT imgid, version FROM main.module_order) mo"
                   "  ON mo.imgid = mi.id"
                   " WHERE %s"
                   " GROUP BY ver"
                   " ORDER BY ver",
                   orders, where_ext);
          // clang-format on
          g_free(orders);
        }
        break;

      case DT_COLLECTION_PROP_RATING: // image rating
        {
        // clang-format off
          g_snprintf(query, sizeof(query),
                     "SELECT CASE WHEN (flags & 8) == 8 THEN -1 ELSE (flags & 7) END AS rating, 1,"
                     " COUNT(*) AS count"
                     " FROM main.images AS mi"
                     " WHERE %s"
                     " GROUP BY rating"
                     " ORDER BY rating", where_ext);
          // clang-format on
        }
        break;

      default:
        if(property >= DT_COLLECTION_PROP_METADATA
           && property < DT_COLLECTION_PROP_METADATA + DT_METADATA_NUMBER)
        {
          const int keyid = dt_metadata_get_keyid_by_display_order(property - DT_COLLECTION_PROP_METADATA);
          const char *name = (gchar *)dt_metadata_get_name(keyid);
          char *setting = g_strdup_printf("plugins/lighttable/metadata/%s_flag", name);
          const gboolean hidden = dt_conf_get_int(setting) & DT_METADATA_FLAG_HIDDEN;
          g_free(setting);
          if(!hidden)
          {
            // clang-format off
            snprintf(query, sizeof(query),
                     "SELECT"
                     " CASE WHEN value IS NULL THEN '%s' ELSE value END AS value,"
                     " 1, COUNT(*) AS count,"
                     " CASE WHEN value IS NULL THEN 0 ELSE 1 END AS force_order"
                     " FROM main.images AS mi"
                     " LEFT JOIN (SELECT id AS meta_data_id, value FROM main.meta_data WHERE key = %d)"
                     "  ON id = meta_data_id"
                     " WHERE %s"
                     " GROUP BY value"
                     " ORDER BY force_order, value",
                     _("not defined"), keyid, where_ext);
            // clang-format on
          }
        }
        else
        // filmroll
        {
          gchar *order_by = NULL;
          const char *filmroll_sort = dt_conf_get_string_const("plugins/collect/filmroll_sort");
          if(strcmp(filmroll_sort, "id") == 0)
            order_by = g_strdup("film_rolls_id DESC");
          else
            if(dt_conf_get_bool("plugins/collect/descending"))
              order_by = g_strdup("folder DESC");
            else
              order_by = g_strdup("folder");

          // clang-format off
          g_snprintf(query, sizeof(query),
                     "SELECT folder, film_rolls_id, COUNT(*) AS count, status"
                     " FROM main.images AS mi"
                     " JOIN (SELECT fr.id AS film_rolls_id, folder, status"
                     "       FROM main.film_rolls AS fr"
                     "        JOIN memory.film_folder AS ff"
                     "        ON ff.id = fr.id)"
                     "   ON film_id = film_rolls_id "
                     " WHERE %s"
                     " GROUP BY folder"
                     " ORDER BY %s", where_ext, order_by);
          // clang-format on

          g_free(order_by);
        }
        break;
    }

    g_free(where_ext);

    if(strlen(query) > 0)
    {
      DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db), query, -1, &stmt, NULL);
      while(sqlite3_step(stmt) == SQLITE_ROW)
      {
        const char *folder = (const char *)sqlite3_column_text(stmt, 0);
        if(folder == NULL) continue; // safeguard against degenerated db entries

        gtk_list_store_append(GTK_LIST_STORE(model), &iter);
        int status = 0;
        if(property == DT_COLLECTION_PROP_FILMROLL)
        {
          folder = dt_image_film_roll_name(folder);
          status = !sqlite3_column_int(stmt, 3);
        }
        const gchar *value = (gchar *)sqlite3_column_text(stmt, 0);
        const int count = sqlite3_column_int(stmt, 2);

        // replace invalid utf8 characters if any
        gchar *text = g_strdup(value);
        gchar *ptr = text;
        while(!g_utf8_validate(ptr, -1, (const gchar **)&ptr)) ptr[0] = '?';

        gchar *escaped_text = g_markup_escape_text(text, -1);

        gtk_list_store_set(GTK_LIST_STORE(model), &iter, DT_LIB_COLLECT_COL_TEXT, folder,
                           DT_LIB_COLLECT_COL_ID, sqlite3_column_int(stmt, 1), DT_LIB_COLLECT_COL_TOOLTIP,
                           escaped_text, DT_LIB_COLLECT_COL_PATH, value, DT_LIB_COLLECT_COL_VISIBLE, TRUE,
                           DT_LIB_COLLECT_COL_COUNT, count, DT_LIB_COLLECT_COL_UNREACHABLE, status,
                           -1);

        g_free(text);
        g_free(escaped_text);
      }
      sqlite3_finalize(stmt);
    }

    gtk_tree_view_set_tooltip_column(GTK_TREE_VIEW(d->view), DT_LIB_COLLECT_COL_TOOLTIP);

    d->listfilter = _create_filtered_model(model, dr);

    GtkTreeSelection *selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(d->view));
    if(property == DT_COLLECTION_PROP_APERTURE
       || property == DT_COLLECTION_PROP_FOCAL_LENGTH
       || property == DT_COLLECTION_PROP_ISO
       || property == DT_COLLECTION_PROP_EXPOSURE
       || property == DT_COLLECTION_PROP_RATING)
    {
      gtk_tree_selection_set_mode(selection, GTK_SELECTION_MULTIPLE);
    }
    else
    {
      gtk_tree_selection_set_mode(selection, GTK_SELECTION_SINGLE);
    }

    gtk_tree_view_set_model(GTK_TREE_VIEW(d->view), d->listfilter);
    gtk_widget_set_no_show_all(GTK_WIDGET(d->view), FALSE);
    gtk_widget_show_all(GTK_WIDGET(d->view));

    g_object_unref(model);

    d->view_rule = property;
  }

  // if needed, we restrict the tree to matching entries
  if(dr->typing && (property == DT_COLLECTION_PROP_CAMERA
                    || property == DT_COLLECTION_PROP_FILENAME
                    || property == DT_COLLECTION_PROP_FILMROLL
                    || property == DT_COLLECTION_PROP_LENS
                    || property == DT_COLLECTION_PROP_APERTURE
                    || property == DT_COLLECTION_PROP_FOCAL_LENGTH
                    || property == DT_COLLECTION_PROP_ISO
                    || property == DT_COLLECTION_PROP_MODULE
                    || property == DT_COLLECTION_PROP_ORDER
                    || property == DT_COLLECTION_PROP_RATING
                    || (property >= DT_COLLECTION_PROP_METADATA
                        && property < DT_COLLECTION_PROP_METADATA + DT_METADATA_NUMBER)))
  {
    gchar *needle = g_utf8_strdown(gtk_entry_get_text(GTK_ENTRY(dr->text)), -1);
    if(g_str_has_suffix(needle, "%")) needle[strlen(needle) - 1] = '\0';
    dr->searchstring = needle;
    gtk_tree_model_foreach(model, (GtkTreeModelForeachFunc)list_match_string, dr);
    dr->searchstring = NULL;
    g_free(needle);
  }
  // we update list selection
  gtk_tree_selection_unselect_all(gtk_tree_view_get_selection(d->view));

  if(property == DT_COLLECTION_PROP_APERTURE
     || property == DT_COLLECTION_PROP_FOCAL_LENGTH
     || property == DT_COLLECTION_PROP_ISO
     || property == DT_COLLECTION_PROP_EXPOSURE
     || property == DT_COLLECTION_PROP_RATING)
  {
    // test selection range [xxx;xxx]
    GRegex *regex;
    GMatchInfo *match_info;

    regex = g_regex_new("^\\s*\\[\\s*(.*)\\s*;\\s*(.*)\\s*\\]\\s*$", 0, 0, NULL);
    g_regex_match_full(regex, gtk_entry_get_text(GTK_ENTRY(dr->text)), -1, 0, 0, &match_info, NULL);
    const int match_count = g_match_info_get_match_count(match_info);

    if(match_count == 3)
    {
      _range_t *range = (_range_t *)calloc(1, sizeof(_range_t));
      range->start = g_match_info_fetch(match_info, 1);
      range->stop = g_match_info_fetch(match_info, 2);

      gtk_tree_model_foreach(d->listfilter, (GtkTreeModelForeachFunc)range_select, range);
      if(range->path1 && range->path2)
      {
        gtk_tree_selection_select_range(gtk_tree_view_get_selection(d->view), range->path1, range->path2);
      }
      g_free(range->start);
      g_free(range->stop);
      gtk_tree_path_free(range->path1);
      gtk_tree_path_free(range->path2);
      free(range);
    }
    else
      gtk_tree_model_foreach(d->listfilter, (GtkTreeModelForeachFunc)list_select, dr);

    g_match_info_free(match_info);
    g_regex_unref(regex);
  }
  else
    gtk_tree_model_foreach(d->listfilter, (GtkTreeModelForeachFunc)list_select, dr);
}


static void _clean_wildcards(dt_lib_collect_rule_t *dr)
{
  char *pref = g_strdup_printf("plugins/lighttable/collect/string%1d", dr->num);
  char *filmroll = dt_conf_get_string(pref);
  if(filmroll && pref)
  {
    // cleanup conf
    gchar *no_star_filmroll = dt_string_replace(filmroll, "*");
    gchar *no_percent_filmroll = dt_string_replace(no_star_filmroll, "%");

    dt_conf_set_string(pref, no_percent_filmroll);
    g_free(no_percent_filmroll);
    g_free(no_star_filmroll);
    g_free(filmroll);

    // reload GUI from conf
    get_properties(dr);
  }
  g_free(pref);
}


static void update_view(dt_lib_collect_rule_t *dr)
{
  const int property = _combo_get_active_collection(dr->combo);
  // wildcards are not supported in pathes for filmrolls
  if(property == DT_COLLECTION_PROP_FILMROLL)
    _clean_wildcards(dr);

  if(property == DT_COLLECTION_PROP_FOLDERS
     || property == DT_COLLECTION_PROP_TAG
     || property == DT_COLLECTION_PROP_GEOTAGGING
     || property == DT_COLLECTION_PROP_DAY
     || is_time_property(property)
    )
    tree_view(dr);
  else
  {
    list_view(dr);
  }
}

static void _set_tooltip(dt_lib_collect_rule_t *d)
{
  const int property = _combo_get_active_collection(d->combo);

  if(property == DT_COLLECTION_PROP_APERTURE
     || property == DT_COLLECTION_PROP_FOCAL_LENGTH
     || property == DT_COLLECTION_PROP_ISO
     || property == DT_COLLECTION_PROP_EXPOSURE)
  {
    gtk_widget_set_tooltip_text(d->text, _("use <, <=, >, >=, <>, =, [;] as operators"));
  }
  else if(property == DT_COLLECTION_PROP_RATING)
  {
    gtk_widget_set_tooltip_text(d->text, _("use <, <=, >, >=, <>, =, [;] as operators\n"
                                           "star rating: 0-5\n"
                                           "rejected images: -1"));
  }
  else if(property == DT_COLLECTION_PROP_DAY || is_time_property(property))
  {
    gtk_widget_set_tooltip_text(d->text,
                                _("use <, <=, >, >=, <>, =, [;] as operators\n"
                                  "type dates in the form: YYYY:MM:DD hh:mm:ss.sss (only the year is mandatory)"));
  }
  else if(property == DT_COLLECTION_PROP_FILENAME)
  {
    gtk_widget_set_tooltip_text(d->text,
    /* xgettext:no-c-format */
                                _("use `%' as wildcard and `,' to separate values"));
  }
  else if(property == DT_COLLECTION_PROP_TAG)
  {
    gtk_widget_set_tooltip_text(d->text,
    /* xgettext:no-c-format */
                                _("use `%' as wildcard\n"
    /* xgettext:no-c-format */
                                  "click to include hierarchy + sub-hierarchies (suffix `*')\n"
    /* xgettext:no-c-format */
                                  "shift+click to include only the current hierarchy (no suffix)\n"
    /* xgettext:no-c-format */
                                  "ctrl+click to include only sub-hierarchies (suffix `|%')"));
  }
  else if(property == DT_COLLECTION_PROP_GEOTAGGING)
  {
    gtk_widget_set_tooltip_text(d->text,
    /* xgettext:no-c-format */
                                _("use `%' as wildcard\n"
    /* xgettext:no-c-format */
                                  "click to include location + sub-locations (suffix `*')\n"
    /* xgettext:no-c-format */
                                  "shift+click to include only the current location (no suffix)\n"
    /* xgettext:no-c-format */
                                  "ctrl+click to include only sub-locations (suffix `|%')"));
  }
  else if(property == DT_COLLECTION_PROP_FOLDERS)
  {
    gtk_widget_set_tooltip_text(d->text,
    /* xgettext:no-c-format */
                                _("use `%' as wildcard\n"
    /* xgettext:no-c-format */
                                  "click to include current + sub-folders (suffix `*')\n"
    /* xgettext:no-c-format */
                                  "shift+click to include only the current folder (no suffix)\n"
    /* xgettext:no-c-format */
                                  "ctrl+click to include only sub-folders (suffix `|%')"));
  }
  else
  {
    /* xgettext:no-c-format */
    gtk_widget_set_tooltip_text(d->text, _("use `%' as wildcard"));
  }

  //set the combobox tooltip as well
  gchar *tip = gtk_widget_get_tooltip_text(d->text);
  gtk_widget_set_tooltip_text(GTK_WIDGET(d->combo), tip);
  g_free(tip);
}

static void _set_rules_visibility(dt_lib_collect_rule_t *dr, gboolean last, gboolean active)
{
  gtk_widget_set_no_show_all(dr->hbox, FALSE);
  gtk_widget_set_visible(dr->hbox, TRUE);
  gtk_widget_show_all(dr->hbox);

  get_properties(dr);

  char confname[200] = { 0 };

  if(last)
  {
    // only clear
    gtk_button_set_label(GTK_BUTTON(dr->button), "-");
    gtk_widget_set_tooltip_text(GTK_WIDGET(dr->button), _("clear this rule"));
  }
  else if(active)
  {
    gtk_button_set_label(GTK_BUTTON(dr->button), "+");
    gtk_widget_set_tooltip_text(GTK_WIDGET(dr->button), _("clear this rule or add new rules"));
  }
  else
  {
    snprintf(confname, sizeof(confname), "plugins/lighttable/collect/mode%1d", dr->num + 1);
    const int mode = dt_conf_get_int(confname);
    if(mode == DT_LIB_COLLECT_MODE_AND) gtk_button_set_label(GTK_BUTTON(dr->button), _("AND"));
    if(mode == DT_LIB_COLLECT_MODE_OR) gtk_button_set_label(GTK_BUTTON(dr->button), _("OR"));
    if(mode == DT_LIB_COLLECT_MODE_AND_NOT) gtk_button_set_label(GTK_BUTTON(dr->button), _("AND NOT"));
    gtk_widget_set_tooltip_text(GTK_WIDGET(dr->button), _("clear this rule"));
  }

  _set_tooltip(dr);
}

static gboolean _is_folder_collection(int i)
{
  /*
  * The historical design of collections sucks. The thing is wired as a general
  * but complex SQL querying tool. Most users just want the content of a folder and find
  * the SQL GUI overwhelming and cryptic. Many complain that dt doesn't even have a filemanager
  * because they didn't understand that querying folders is only one option of a more
  * general feature.
  *
  * But it turns out that folders are called "folders" only when they are in hierarchical
  * tree view, otherwise they are "filmrolls" in list view, but it's the same content. This is highly misleading
  * but it doesn't stop there. Indeed, the name of the filmroll is actually the name of the
  * filesystem folder, and cannot be changed, while people may think of it as an abitrary collection
  * with a customizable name à la Lightroom. This is all a problem of GUI and calling folders
  * something else while still having something actually called folder.
  *
  * We re-interface here the simple case of querying a filesystem folder, with no additional constrain,
  * with a lite GUI calling folder folder, and having a "view" option for tree or list.
  */

  char confname[200] = { 0 };
  snprintf(confname, sizeof(confname), "plugins/lighttable/collect/item%1d", i);
  const int result = dt_conf_get_int(confname);
  return result == DT_COLLECTION_PROP_FILMROLL || result == DT_COLLECTION_PROP_FOLDERS;
}

static gboolean _is_tag_collection(int i)
{
  char confname[200] = { 0 };
  snprintf(confname, sizeof(confname), "plugins/lighttable/collect/item%1d", i);
  const int result = dt_conf_get_int(confname);
  return result == DT_COLLECTION_PROP_TAG;
}

static void combo_changed(GtkWidget *combo, dt_lib_collect_rule_t *d);
static gboolean _clear_text_entry(dt_collection_properties_t next_property, dt_collection_properties_t prev_property,
                                  dt_lib_collect_rule_t *d);

static void _depopulate_combo(GtkWidget *combo)
{
  // De-populate the combobox and restrict to folder/filmroll options
  dt_bauhaus_combobox_clear(combo);
  dt_bauhaus_combobox_remove_at(combo, 0); // remove section title
  dt_bauhaus_widget_set_label(combo, _("View"));
  dt_bauhaus_combobox_set_selected_text_align(combo, DT_BAUHAUS_COMBOBOX_ALIGN_RIGHT);
  dt_bauhaus_combobox_add_full(combo, _("List"), DT_BAUHAUS_COMBOBOX_ALIGN_RIGHT,
                                GUINT_TO_POINTER(DT_COLLECTION_PROP_FILMROLL + 1), NULL, TRUE);
  dt_bauhaus_combobox_add_full(combo, _("Tree"), DT_BAUHAUS_COMBOBOX_ALIGN_RIGHT,
                                  GUINT_TO_POINTER(DT_COLLECTION_PROP_FOLDERS + 1), NULL, TRUE);
}

static void _repopulate_combo(GtkWidget *combo)
{
  // Re-populate the comboboxes with the full set of options
  dt_bauhaus_combobox_clear(combo);
  dt_bauhaus_widget_set_label(combo, NULL);
  dt_bauhaus_combobox_set_selected_text_align(combo, DT_BAUHAUS_COMBOBOX_ALIGN_RIGHT);
  _populate_collect_combo(combo);
}

static void _hide_all_rules(dt_lib_collect_t *d)
{
  // Reset everything to hidden
  gtk_widget_set_no_show_all(GTK_WIDGET(d->view), TRUE);
  for(int i = 0; i < MAX_RULES; i++)
  {
    gtk_widget_set_no_show_all(d->rule[i].hbox, TRUE);
    gtk_widget_set_visible(d->rule[i].hbox, FALSE);
  }
}

static void _refresh_rules_visibility(dt_lib_collect_t *d)
{
  // Show active rule(s)
  get_number_of_rules(d); // refresh d->active_rule
  for(int i = 0; i <= d->active_rule; i++)
    _set_rules_visibility(d->rule + i, i == MAX_RULES - 1, i == d->active_rule);
}

static void _update_collect_modes(dt_lib_collect_t *d, int variant)
{
  switch(variant)
  {
    case 0:
    {
      // Ensure the property is either folder or filmstrip
      dt_collection_properties_t prev_prop = dt_conf_get_int("plugins/lighttable/collect/item0");
      dt_collection_properties_t prop = (_is_folder_collection(0)) ? prev_prop : DT_COLLECTION_PROP_FOLDERS;
      dt_conf_set_int("plugins/lighttable/collect/item0", prop);

      // Rebuild the combobox list and remap the property to the new entry
      _depopulate_combo(d->rule[0].combo);
      _combo_set_active_collection(d->rule[0].combo, prop);

      if(_clear_text_entry(prev_prop, prop, d->rule))
        gtk_entry_set_placeholder_text(GTK_ENTRY(d->rule[0].text), _("Search a folder..."));

      break;
    }
    case 1:
    {
      dt_collection_properties_t prev_prop = dt_conf_get_int("plugins/lighttable/collect/item0");
      dt_collection_properties_t prop = DT_COLLECTION_PROP_TAG;
      dt_conf_set_int("plugins/lighttable/collect/item0", DT_COLLECTION_PROP_TAG);

      // Depopulate combo
      dt_bauhaus_combobox_clear(d->rule[0].combo);
      dt_bauhaus_combobox_remove_at(d->rule[0].combo, 0); // remove section title
      dt_bauhaus_widget_set_label(d->rule[0].combo, _("View"));
      dt_bauhaus_combobox_set_selected_text_align(d->rule[0].combo, DT_BAUHAUS_COMBOBOX_ALIGN_RIGHT);
      dt_bauhaus_combobox_add_full(d->rule[0].combo, _("Collections"), DT_BAUHAUS_COMBOBOX_ALIGN_RIGHT,
                                  GUINT_TO_POINTER(DT_COLLECTION_PROP_TAG + 1), NULL, TRUE);

      // Remap the property to the new entry. Only one remaining.
      _combo_set_active_collection(d->rule[0].combo, prop);

      if(_clear_text_entry(prev_prop, prop, d->rule))
        gtk_entry_set_placeholder_text(GTK_ENTRY(d->rule[0].text), _("Search a collection..."));

      break;
    }
    case 2:
    default:
    {
      // Rebuild the combobox list and remap the property to the new entry
      dt_collection_properties_t prop = dt_conf_get_int("plugins/lighttable/collect/item0");
      _repopulate_combo(d->rule[0].combo);
      _combo_set_active_collection(d->rule[0].combo, prop);

      gtk_entry_set_placeholder_text(GTK_ENTRY(d->rule[0].text), _("Search..."));
    }
  }
}

static void _lib_collect_mode(GtkNotebook *notebook, GtkWidget *page, guint page_num, gpointer user_data)
{
  dt_lib_module_t *self = (dt_lib_module_t *)user_data;
  dt_lib_collect_t *d = (dt_lib_collect_t *)self->data;

  if(page_num == 0 || page_num == 1) dt_conf_set_int("plugins/lighttable/collect/num_rules", 1);
  dt_conf_set_int("plugins/lighttable/collect/tab", page_num);

  ++darktable.gui->reset;
  _update_collect_modes(d, page_num);

  _hide_all_rules(d);
  _refresh_rules_visibility(d);

  if(page_num == 1)
    gtk_widget_hide(d->rule[0].combo); // only 1 option to show, so hide it
  else
    gtk_widget_show(d->rule[0].combo);

  update_view(get_active_rule(d));
  --darktable.gui->reset;
}

static void _lib_collect_gui_update(dt_lib_module_t *self)
{
  dt_lib_collect_t *d = (dt_lib_collect_t *)self->data;

  // we check if something has changed since last call
  if(d->view_rule != -1) return;

  ++darktable.gui->reset;
  _hide_all_rules(d);
  _refresh_rules_visibility(d);

  // Enable simplified tab view if using one single folder or tag rule
  //const int num_page = dt_conf_get_int("plugins/lighttable/collect/tab");
  int page_num;
  if(_is_folder_collection(0) && d->nb_rules == 1)
  {
    page_num = 0;
    // fprintf(stdout, "folder case\n");
  }
  else if(_is_tag_collection(0) && d->nb_rules == 1)
  {
    page_num = 1;
    // fprintf(stdout, "tag case\n");
  }
  else
  {
    page_num = 2;
    // fprintf(stdout, "general case\n");
  }

  // Set notebook page without triggering its callback
  g_signal_handlers_block_matched(d->notebook, G_SIGNAL_MATCH_FUNC, 0, 0, NULL, _lib_collect_mode, NULL);
  gtk_notebook_set_current_page(GTK_NOTEBOOK(d->notebook), page_num);
  g_signal_handlers_unblock_matched(d->notebook, G_SIGNAL_MATCH_FUNC, 0, 0, NULL, _lib_collect_mode, NULL);

  _update_collect_modes(d, page_num);

  if(page_num == 1)
    gtk_widget_hide(d->rule[0].combo); // only 1 option to show, so hide it
  else
    gtk_widget_show(d->rule[0].combo);

  // Update the treeviews
  update_view(get_active_rule(d));
  --darktable.gui->reset;
}

void gui_reset(dt_lib_module_t *self)
{
  dt_conf_set_int("plugins/lighttable/collect/num_rules", 1);
  dt_conf_set_int("plugins/lighttable/collect/item0", DT_COLLECTION_PROP_FILMROLL);
  dt_conf_set_int("plugins/lighttable/collect/mode0", 0);
  dt_conf_set_string("plugins/lighttable/collect/string0", "");
  dt_lib_collect_t *d = (dt_lib_collect_t *)self->data;
  d->active_rule = 0;
  d->view_rule = -1;
  dt_collection_set_query_flags(darktable.collection, COLLECTION_QUERY_FULL);
  dt_collection_update_query(darktable.collection, DT_COLLECTION_CHANGE_NEW_QUERY, DT_COLLECTION_PROP_UNDEF, NULL);
}

static gboolean _is_folder_property(const int prop)
{
  // Filmrolls and folders are ultimately the same content displayed differently.
  return prop == DT_COLLECTION_PROP_FILMROLL || prop == DT_COLLECTION_PROP_FOLDERS;
}

static gboolean _is_tag_property(const int prop)
{
  return prop == DT_COLLECTION_PROP_TAG;
}

static gboolean _clear_text_entry(dt_collection_properties_t next_property,
                                  dt_collection_properties_t prev_property,
                                  dt_lib_collect_rule_t *d)
{
  // Clear the rule text entry only if needed
  gboolean cleared = FALSE;

  if((_is_folder_property(next_property) && _is_folder_property(prev_property)) ||
     (_is_tag_property(next_property) && _is_tag_property(prev_property)))
  {
    // Don't clear the text entry because the search string is transferable.
  }
  else
  {
    g_signal_handlers_block_matched(d->text, G_SIGNAL_MATCH_FUNC, 0, 0, NULL, entry_changed, NULL);
    gtk_entry_set_text(GTK_ENTRY(d->text), "");
    g_signal_handlers_unblock_matched(d->text, G_SIGNAL_MATCH_FUNC, 0, 0, NULL, entry_changed, NULL);
    cleared = TRUE;
  }

  return cleared;
}

static void combo_changed(GtkWidget *combo, dt_lib_collect_rule_t *d)
{
  if(darktable.gui->reset) return;

  dt_lib_collect_t *c = get_collect(d);
  const dt_collection_properties_t property = _combo_get_active_collection(d->combo);
  char confname[200] = { 0 };
  snprintf(confname, sizeof(confname), "plugins/lighttable/collect/item%1d", d->num);
  const dt_collection_properties_t previous_property = dt_conf_get_int(confname);
  _clear_text_entry(property, previous_property, d);

  c->active_rule = d->num;

  if(_is_folder_property(property)
     || property == DT_COLLECTION_PROP_TAG
     || property == DT_COLLECTION_PROP_GEOTAGGING
     || property == DT_COLLECTION_PROP_DAY
     || is_time_property(property)
    )
  {
    d->typing = FALSE;
  }

  _set_tooltip(d);
  set_properties(d);
  update_view(d);
  c->view_rule = -1;
  dt_collection_update_query(darktable.collection, DT_COLLECTION_CHANGE_NEW_QUERY, DT_COLLECTION_PROP_UNDEF, NULL);
}

static void row_activated_with_event(GtkTreeView *view, GtkTreePath *path, GtkTreeViewColumn *col, GdkEventButton *event, dt_lib_collect_t *d)
{
  GtkTreeIter iter;
  GtkTreeModel *model = NULL;

  GtkTreeSelection *selection = gtk_tree_view_get_selection(GTK_TREE_VIEW(view));
  int n_selected = gtk_tree_selection_count_selected_rows(selection);
  if(n_selected < 1) return;
  GList *sels = gtk_tree_selection_get_selected_rows(selection, &model);
  GtkTreePath *path1 = (GtkTreePath *)sels->data;
  if(!gtk_tree_model_get_iter(model, &iter, path1))
  {
    g_list_free_full(sels, (GDestroyNotify)gtk_tree_path_free);
    return;
  }

  gchar *text;
  gboolean order_request = FALSE;
  int order;

  get_number_of_rules(d);
  dt_lib_collect_rule_t *active_rule = get_active_rule(d);
  active_rule->typing = FALSE;

  const int item = d->view_rule;
  gtk_tree_model_get(model, &iter, DT_LIB_COLLECT_COL_PATH, &text, -1);

  if(text && strlen(text) > 0)
  {
    if(n_selected > 1 && item_is_numeric_collection(item))
    {
      /* this is a range selection */
      GtkTreeIter iter2;
      GtkTreePath *path2 = (GtkTreePath *)g_list_last(sels)->data;
      if(!gtk_tree_model_get_iter(model, &iter2, path2)) return;

      gchar *text2;
      gtk_tree_model_get(model, &iter2, DT_LIB_COLLECT_COL_PATH, &text2, -1);

      gchar *n_text;
      if(item == DT_COLLECTION_PROP_DAY || is_time_property(item))
        n_text = g_strdup_printf("[%s;%s]", text2, text); /* dates are in reverse order */
      else
        n_text = g_strdup_printf("[%s;%s]", text, text2);

      g_free(text);
      g_free(text2);
      text = n_text;
    }
    else if(item == DT_COLLECTION_PROP_TAG
            || item == DT_COLLECTION_PROP_GEOTAGGING
            || item == DT_COLLECTION_PROP_FOLDERS)
    {
      if(gtk_tree_model_iter_has_child(model, &iter))
      {
        /* if a tag has children, ctrl-clicking on a parent node should display all images under this hierarchy. */
        if(dt_modifier_is(event->state, GDK_CONTROL_MASK))
        {
          gchar *n_text = g_strconcat(text, "|%", NULL);
          g_free(text);
          text = n_text;
        }
        /* if a tag has children, left-clicking on a parent node should display all images in and under this
         * hierarchy. */
        else if(!dt_modifier_is(event->state, GDK_SHIFT_MASK))
        {
          gchar *n_text = g_strconcat(text, "*", NULL);
          g_free(text);
          text = n_text;
        }
      }
      else if(active_rule == d->rule && g_strcmp0(text, _("not tagged")))
      {
        // first filter is tag and the row is a leave
        uint32_t sort = DT_COLLECTION_SORT_NONE;
        gboolean descending = FALSE;
        const uint32_t tagid = dt_tag_get_tag_id_by_name(text);
        if(tagid)
        {
          order_request = TRUE;
          if(dt_tag_get_tag_order_by_id(tagid, &sort, &descending))
          {
            order = sort | (descending  ? DT_COLLECTION_ORDER_FLAG : 0);
          }
          else
          {
            // the tag order is not set yet - default order (filename)
            order = DT_COLLECTION_SORT_FILENAME;
            dt_tag_set_tag_order_by_id(tagid, order & ~DT_COLLECTION_ORDER_FLAG,
                                       order & DT_COLLECTION_ORDER_FLAG);
          }
          dt_collection_set_tag_id((dt_collection_t *)darktable.collection, tagid);
        }
        else dt_collection_set_tag_id((dt_collection_t *)darktable.collection, 0);
      }
    }
    else
      _combo_set_active_collection(active_rule->combo, item);
  }
  g_list_free_full(sels, (GDestroyNotify)gtk_tree_path_free);

  g_signal_handlers_block_matched(active_rule->text, G_SIGNAL_MATCH_FUNC, 0, 0, NULL, entry_changed, NULL);
  gtk_entry_set_text(GTK_ENTRY(active_rule->text), text);
  gtk_editable_set_position(GTK_EDITABLE(active_rule->text), -1);
  g_signal_handlers_unblock_matched(active_rule->text, G_SIGNAL_MATCH_FUNC, 0, 0, NULL, entry_changed, NULL);
  g_free(text);

  if(item == DT_COLLECTION_PROP_TAG
     || item == DT_COLLECTION_PROP_FILMROLL
     || item == DT_COLLECTION_PROP_DAY
     || is_time_property(item)
     || item == DT_COLLECTION_PROP_COLORLABEL
     || item == DT_COLLECTION_PROP_GEOTAGGING
     || item == DT_COLLECTION_PROP_HISTORY
     || item == DT_COLLECTION_PROP_LOCAL_COPY
     || item == DT_COLLECTION_PROP_GROUPING)
  {
    set_properties(active_rule); // we just have to set the selection
  }
  else
    update_view(active_rule); // we have to update visible items too

  dt_control_signal_block_by_func(darktable.signals, G_CALLBACK(collection_updated),
                                  darktable.view_manager->proxy.module_collect.module);
  if(order_request)
    DT_DEBUG_CONTROL_SIGNAL_RAISE(darktable.signals, DT_SIGNAL_IMAGES_ORDER_CHANGE, order);
  dt_collection_update_query(darktable.collection, DT_COLLECTION_CHANGE_NEW_QUERY, DT_COLLECTION_PROP_UNDEF, NULL);
  dt_control_signal_unblock_by_func(darktable.signals, G_CALLBACK(collection_updated),
                                    darktable.view_manager->proxy.module_collect.module);
  dt_gui_refocus_center();
  dt_control_queue_redraw_center();
}

static void entry_activated(GtkWidget *entry, dt_lib_collect_rule_t *d)
{
  GtkTreeView *view;
  GtkTreeModel *model;
  int rows;

  update_view(d);
  dt_lib_collect_t *c = get_collect(d);

  const int property = _combo_get_active_collection(d->combo);

  if(property != DT_COLLECTION_PROP_FOLDERS
     && property != DT_COLLECTION_PROP_TAG
     && property != DT_COLLECTION_PROP_GEOTAGGING
     && property != DT_COLLECTION_PROP_DAY
     && !is_time_property(property))
  {
    view = c->view;
    model = gtk_tree_view_get_model(GTK_TREE_VIEW(view));

    rows = gtk_tree_model_iter_n_children(model, NULL);

    // if only one row, press enter in entry box to fill it with the row value
    if(rows == 1)
    {
      GtkTreeIter iter;
      if(gtk_tree_model_get_iter_first(model, &iter))
      {
        gchar *text;
        gtk_tree_model_get(model, &iter, DT_LIB_COLLECT_COL_PATH, &text, -1);

        g_signal_handlers_block_matched(d->text, G_SIGNAL_MATCH_FUNC, 0, 0, NULL, entry_changed, NULL);
        gtk_entry_set_text(GTK_ENTRY(d->text), text);
        gtk_editable_set_position(GTK_EDITABLE(d->text), -1);
        g_signal_handlers_unblock_matched(d->text, G_SIGNAL_MATCH_FUNC, 0, 0, NULL, entry_changed, NULL);
        g_free(text);
        update_view(d);
      }
    }
  }
  dt_control_signal_block_by_func(darktable.signals, G_CALLBACK(collection_updated),
                                  darktable.view_manager->proxy.module_collect.module);
  dt_collection_update_query(darktable.collection, DT_COLLECTION_CHANGE_NEW_QUERY, DT_COLLECTION_PROP_UNDEF, NULL);
  dt_control_signal_unblock_by_func(darktable.signals, G_CALLBACK(collection_updated),
                                    darktable.view_manager->proxy.module_collect.module);
  d->typing = FALSE;

  dt_control_queue_redraw_center();
}

static void entry_changed(GtkEntry *entry, dt_lib_collect_rule_t *dr)
{
  dr->typing = TRUE;
  update_view(dr);
}

int position()
{
  return 400;
}

static gboolean entry_focus_in_callback(GtkWidget *w, GdkEventFocus *event, dt_lib_collect_rule_t *d)
{
  dt_lib_collect_t *c = get_collect(d);
  dt_lib_collect_rule_t *active_rule = get_active_rule(c);
  update_view(active_rule);
  return FALSE;
}

/*
static void _focus_first_search_entry(dt_action_t *action)
{
  entry_focus_in_callback(action->owner, NULL,
}
*/

static void menuitem_mode(GtkMenuItem *menuitem, dt_lib_collect_rule_t *d)
{
  // add next row with and operator
  const int _a = dt_conf_get_int("plugins/lighttable/collect/num_rules");
  const int active = CLAMP(_a, 1, MAX_RULES);
  if(active < MAX_RULES)
  {
    char confname[200] = { 0 };
    snprintf(confname, sizeof(confname), "plugins/lighttable/collect/mode%1d", active);
    const dt_lib_collect_mode_t mode = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(menuitem), "menuitem_mode"));
    dt_conf_set_int(confname, mode);
    snprintf(confname, sizeof(confname), "plugins/lighttable/collect/string%1d", active);
    dt_conf_set_string(confname, "");
    dt_conf_set_int("plugins/lighttable/collect/num_rules", active + 1);
    dt_lib_collect_t *c = get_collect(d);
    c->active_rule = active;
    c->view_rule = -1;
  }
  dt_collection_update_query(darktable.collection, DT_COLLECTION_CHANGE_NEW_QUERY, DT_COLLECTION_PROP_UNDEF, NULL);
}

static void menuitem_mode_change(GtkMenuItem *menuitem, dt_lib_collect_rule_t *d)
{
  // add next row with and operator
  const int num = d->num + 1;
  if(num < MAX_RULES && num > 0)
  {
    char confname[200] = { 0 };
    snprintf(confname, sizeof(confname), "plugins/lighttable/collect/mode%1d", num);
    const dt_lib_collect_mode_t mode = GPOINTER_TO_INT(g_object_get_data(G_OBJECT(menuitem), "menuitem_mode"));
    dt_conf_set_int(confname, mode);
  }
  dt_lib_collect_t *c = get_collect(d);
  c->view_rule = -1;
  dt_collection_update_query(darktable.collection, DT_COLLECTION_CHANGE_NEW_QUERY, DT_COLLECTION_PROP_UNDEF, NULL);
}

static void collection_updated(gpointer instance, dt_collection_change_t query_change,
                               dt_collection_properties_t changed_property, gpointer imgs, int next, gpointer self)
{
  dt_lib_module_t *dm = (dt_lib_module_t *)self;
  dt_lib_collect_t *d = (dt_lib_collect_t *)dm->data;

  // update tree
  d->view_rule = -1;
  dt_lib_collect_rule_t *active_rule = get_active_rule(d);
  active_rule->typing = FALSE;

  // determine if we want to refresh the tree or not
  gboolean refresh = TRUE;
  if(query_change == DT_COLLECTION_CHANGE_RELOAD && changed_property != DT_COLLECTION_PROP_UNDEF)
  {
    // if we only reload the collection, that means that we don't change the query itself
    // so we only rebuild the treeview if a used property has changed
    refresh = FALSE;
    for(int i = 0; i <= d->active_rule; i++)
    {
      const int item = _combo_get_active_collection(d->rule[i].combo);
      if(item == changed_property)
      {
        refresh = TRUE;
        break;
      }
    }
  }

  if(refresh) _lib_collect_gui_update(self);
}


static void filmrolls_updated(gpointer instance, gpointer self)
{
  // TODO: We should update the count of images here
  dt_lib_module_t *dm = (dt_lib_module_t *)self;
  dt_lib_collect_t *d = (dt_lib_collect_t *)dm->data;
  d->view_rule = -1;
  _lib_collect_gui_update(self);
}

static void filmrolls_imported(gpointer instance, int film_id, gpointer self)
{
  dt_lib_module_t *dm = (dt_lib_module_t *)self;
  dt_lib_collect_t *d = (dt_lib_collect_t *)dm->data;

  // update tree
  d->view_rule = -1;
  dt_lib_collect_rule_t *active_rule = get_active_rule(d);
  active_rule->typing = FALSE;
  _lib_collect_gui_update(self);
}

static void preferences_changed(gpointer instance, gpointer self)
{
  dt_collection_update_query(darktable.collection, DT_COLLECTION_CHANGE_RELOAD, DT_COLLECTION_PROP_UNDEF, NULL);
}

static void filmrolls_removed(gpointer instance, gpointer self)
{
  dt_lib_module_t *dm = (dt_lib_module_t *)self;
  dt_lib_collect_t *d = (dt_lib_collect_t *)dm->data;

  // update tree
  if (d->view_rule != DT_COLLECTION_PROP_FOLDERS)
  {
    d->view_rule = -1;
  }
  dt_lib_collect_rule_t *active_rule = get_active_rule(d);
  active_rule->typing = FALSE;
  _lib_collect_gui_update(self);
}

static void tag_changed(gpointer instance, gpointer self)
{
  dt_lib_module_t *dm = (dt_lib_module_t *)self;
  dt_lib_collect_t *d = (dt_lib_collect_t *)dm->data;
  dt_lib_collect_rule_t *active_rule = get_active_rule(d);
  // update tree
  if(_combo_get_active_collection(active_rule->combo) == DT_COLLECTION_PROP_TAG)
  {
    d->view_rule = -1;
    active_rule->typing = FALSE;

    //need to reload collection since we have tags as active collection filter
    dt_control_signal_block_by_func(darktable.signals, G_CALLBACK(collection_updated),
                                    darktable.view_manager->proxy.module_collect.module);
    dt_collection_update_query(darktable.collection, DT_COLLECTION_CHANGE_RELOAD, DT_COLLECTION_PROP_TAG, NULL);
    dt_control_signal_unblock_by_func(darktable.signals, G_CALLBACK(collection_updated),
                                      darktable.view_manager->proxy.module_collect.module);
  }
  else
  {
    // currently tag filter isn't the one selected but it might be in one of rules. needs check
    gboolean needs_update = FALSE;
    for(int i = 0; i < d->nb_rules && !needs_update; i++)
    {
      needs_update = needs_update
        || _combo_get_active_collection(d->rule[i].combo) == DT_COLLECTION_PROP_TAG;
    }
    if(needs_update){
      // we have tags as one of rules, needs reload.
      dt_control_signal_block_by_func(darktable.signals, G_CALLBACK(collection_updated),
                                      darktable.view_manager->proxy.module_collect.module);
      dt_collection_update_query(darktable.collection, DT_COLLECTION_CHANGE_RELOAD, DT_COLLECTION_PROP_TAG, NULL);
      dt_control_signal_unblock_by_func(darktable.signals, G_CALLBACK(collection_updated),
                                        darktable.view_manager->proxy.module_collect.module);
    }
  }
}

static void geotag_changed(gpointer instance, GList *imgs, const int locid, gpointer self)
{
  // if locid <> NULL this event doesn't concern collect module
  if(!locid)
  {
    dt_lib_module_t *dm = (dt_lib_module_t *)self;
    dt_lib_collect_t *d = (dt_lib_collect_t *)dm->data;
    dt_lib_collect_rule_t *active_rule = get_active_rule(d);

    // update tree
    if(_combo_get_active_collection(active_rule->combo) == DT_COLLECTION_PROP_GEOTAGGING)
    {
      d->view_rule = -1;
      active_rule->typing = FALSE;
      _lib_collect_gui_update(self);

      //need to reload collection since we have geotags as active collection filter
      dt_control_signal_block_by_func(darktable.signals, G_CALLBACK(collection_updated),
                                      darktable.view_manager->proxy.module_collect.module);
      dt_collection_update_query(darktable.collection, DT_COLLECTION_CHANGE_RELOAD, DT_COLLECTION_PROP_GEOTAGGING,
                                 NULL);
      dt_control_signal_unblock_by_func(darktable.signals, G_CALLBACK(collection_updated),
                                        darktable.view_manager->proxy.module_collect.module);
    }
  }
}

static void metadata_changed(gpointer instance, int type, gpointer self)
{
  dt_lib_module_t *dm = (dt_lib_module_t *)self;
  dt_lib_collect_t *d = (dt_lib_collect_t *)dm->data;
  if(type == DT_METADATA_SIGNAL_HIDDEN
     || type == DT_METADATA_SIGNAL_SHOWN)
  {
    // hidden/shown metadata have changed - update the collection list
    for(int i = 0; i < MAX_RULES; i++)
    {
      g_signal_handlers_block_matched(d->rule[i].combo, G_SIGNAL_MATCH_FUNC, 0, 0, NULL, combo_changed, NULL);
      const int property = _combo_get_active_collection(d->rule[i].combo);
      dt_bauhaus_combobox_clear(d->rule[i].combo);
      _populate_collect_combo(d->rule[i].combo);
      if(property != -1 && !_combo_set_active_collection(d->rule[i].combo, property))
      {
        // this one has been hidden - remove entry
        g_signal_handlers_block_matched(d->rule[i].text, G_SIGNAL_MATCH_FUNC, 0, 0, NULL, entry_changed, NULL);
        gtk_entry_set_text(GTK_ENTRY(d->rule[i].text), "");
        g_signal_handlers_unblock_matched(d->rule[i].text, G_SIGNAL_MATCH_FUNC, 0, 0, NULL, entry_changed, NULL);
        d->rule[i].typing = FALSE;
        set_properties(&d->rule[i]);
      }
      g_signal_handlers_unblock_matched(d->rule[i].combo, G_SIGNAL_MATCH_FUNC, 0, 0, NULL, combo_changed, NULL);
    }
  }

  dt_lib_collect_rule_t *active_rule = get_active_rule(d);

  // update collection if metadata have been hidden or a metadata collection is active
  const int prop = _combo_get_active_collection(active_rule->combo);
  if(type == DT_METADATA_SIGNAL_HIDDEN
     || (prop >= DT_COLLECTION_PROP_METADATA
         && prop < DT_COLLECTION_PROP_METADATA + DT_METADATA_NUMBER))
  {
    dt_collection_update_query(darktable.collection, DT_COLLECTION_CHANGE_RELOAD, DT_COLLECTION_PROP_METADATA,
                               NULL);
  }
}

static void menuitem_clear(GtkMenuItem *menuitem, dt_lib_collect_rule_t *d)
{
  // remove this row, or if 1st, clear text entry box
  const int _a = dt_conf_get_int("plugins/lighttable/collect/num_rules");
  const int active = CLAMP(_a, 1, MAX_RULES);
  dt_lib_collect_t *c = get_collect(d);
  if(active > 1)
  {
    dt_conf_set_int("plugins/lighttable/collect/num_rules", active - 1);
    if(c->active_rule >= active - 1) c->active_rule = active - 2;
  }
  else
  {
    dt_conf_set_int("plugins/lighttable/collect/mode0", DT_LIB_COLLECT_MODE_AND);
    dt_conf_set_int("plugins/lighttable/collect/item0", 0);
    dt_conf_set_string("plugins/lighttable/collect/string0", "");
    d->typing = FALSE;
  }
  // move up all still active rules by one.
  for(int i = d->num; i < MAX_RULES - 1; i++)
  {
    char confname[200] = { 0 };
    snprintf(confname, sizeof(confname), "plugins/lighttable/collect/mode%1d", i + 1);
    const int mode = dt_conf_get_int(confname);
    snprintf(confname, sizeof(confname), "plugins/lighttable/collect/item%1d", i + 1);
    const int item = dt_conf_get_int(confname);
    snprintf(confname, sizeof(confname), "plugins/lighttable/collect/string%1d", i + 1);
    gchar *string = dt_conf_get_string(confname);
    if(string)
    {
      snprintf(confname, sizeof(confname), "plugins/lighttable/collect/mode%1d", i);
      dt_conf_set_int(confname, mode);
      snprintf(confname, sizeof(confname), "plugins/lighttable/collect/item%1d", i);
      dt_conf_set_int(confname, item);
      snprintf(confname, sizeof(confname), "plugins/lighttable/collect/string%1d", i);
      dt_conf_set_string(confname, string);
      g_free(string);
    }
  }

  c->view_rule = -1;
  dt_collection_update_query(darktable.collection, DT_COLLECTION_CHANGE_NEW_QUERY, DT_COLLECTION_PROP_UNDEF, NULL);
}

static gboolean popup_button_callback(GtkWidget *widget, GdkEventButton *event, dt_lib_collect_rule_t *d)
{
  if(event->button != 1) return FALSE;

  GtkWidget *menu = gtk_menu_new();
  GtkWidget *mi;
  const int _a = dt_conf_get_int("plugins/lighttable/collect/num_rules");
  const int active = CLAMP(_a, 1, MAX_RULES);

  mi = gtk_menu_item_new_with_label(_("clear this rule"));
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);
  g_signal_connect(G_OBJECT(mi), "activate", G_CALLBACK(menuitem_clear), d);

  if(d->num == active - 1)
  {
    mi = gtk_menu_item_new_with_label(_("narrow down search"));
    g_object_set_data(G_OBJECT(mi), "menuitem_mode", GINT_TO_POINTER(DT_LIB_COLLECT_MODE_AND));
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);
    g_signal_connect(G_OBJECT(mi), "activate", G_CALLBACK(menuitem_mode), d);

    mi = gtk_menu_item_new_with_label(_("add more images"));
    g_object_set_data(G_OBJECT(mi), "menuitem_mode", GINT_TO_POINTER(DT_LIB_COLLECT_MODE_OR));
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);
    g_signal_connect(G_OBJECT(mi), "activate", G_CALLBACK(menuitem_mode), d);

    mi = gtk_menu_item_new_with_label(_("exclude images"));
    g_object_set_data(G_OBJECT(mi), "menuitem_mode", GINT_TO_POINTER(DT_LIB_COLLECT_MODE_AND_NOT));
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);
    g_signal_connect(G_OBJECT(mi), "activate", G_CALLBACK(menuitem_mode), d);
  }
  else if(d->num < active - 1)
  {
    mi = gtk_menu_item_new_with_label(_("change to: and"));
    g_object_set_data(G_OBJECT(mi), "menuitem_mode", GINT_TO_POINTER(DT_LIB_COLLECT_MODE_AND));
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);
    g_signal_connect(G_OBJECT(mi), "activate", G_CALLBACK(menuitem_mode_change), d);

    mi = gtk_menu_item_new_with_label(_("change to: or"));
    g_object_set_data(G_OBJECT(mi), "menuitem_mode", GINT_TO_POINTER(DT_LIB_COLLECT_MODE_OR));
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);
    g_signal_connect(G_OBJECT(mi), "activate", G_CALLBACK(menuitem_mode_change), d);

    mi = gtk_menu_item_new_with_label(_("change to: except"));
    g_object_set_data(G_OBJECT(mi), "menuitem_mode", GINT_TO_POINTER(DT_LIB_COLLECT_MODE_AND_NOT));
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);
    g_signal_connect(G_OBJECT(mi), "activate", G_CALLBACK(menuitem_mode_change), d);
  }

  gtk_widget_show_all(GTK_WIDGET(menu));

  gtk_menu_popup_at_pointer(GTK_MENU(menu), (GdkEvent *)event);

  return TRUE;
}

static void _populate_collect_combo(GtkWidget *w)
{
#define ADD_COLLECT_ENTRY(value)                                                              \
  dt_bauhaus_combobox_add_full(w, dt_collection_name(value), DT_BAUHAUS_COMBOBOX_ALIGN_RIGHT, \
                               GUINT_TO_POINTER(value + 1), NULL, TRUE)

    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_FILMROLL);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_FOLDERS);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_FILENAME);

    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_TAG);
    for(unsigned int i = 0; i < DT_METADATA_NUMBER; i++)
    {
      const uint32_t keyid = dt_metadata_get_keyid_by_display_order(i);
      const gchar *name = dt_metadata_get_name(keyid);
      gchar *setting = g_strdup_printf("plugins/lighttable/metadata/%s_flag", name);
      const gboolean hidden = dt_conf_get_int(setting) & DT_METADATA_FLAG_HIDDEN;
      g_free(setting);
      const int meta_type = dt_metadata_get_type(keyid);
      if(meta_type != DT_METADATA_TYPE_INTERNAL && !hidden)
      {
        ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_METADATA + i);
      }
    }
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_RATING);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_COLORLABEL);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_GEOTAGGING);

    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_DAY);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_TIME);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_IMPORT_TIMESTAMP);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_CHANGE_TIMESTAMP);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_EXPORT_TIMESTAMP);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_PRINT_TIMESTAMP);

    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_CAMERA);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_LENS);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_APERTURE);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_EXPOSURE);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_FOCAL_LENGTH);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_ISO);

    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_GROUPING);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_LOCAL_COPY);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_HISTORY);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_MODULE);
    ADD_COLLECT_ENTRY(DT_COLLECTION_PROP_ORDER);

#undef ADD_COLLECT_ENTRY
}

void _menuitem_preferences(GtkMenuItem *menuitem, dt_lib_module_t *self)
{
  GtkWidget *win = dt_ui_main_window(darktable.gui->ui);
  GtkWidget *dialog = gtk_dialog_new_with_buttons(_("collections settings"), GTK_WINDOW(win),
                                                  GTK_DIALOG_DESTROY_WITH_PARENT,
                                                 _("cancel"), GTK_RESPONSE_NONE,
                                                 _("save"), GTK_RESPONSE_ACCEPT, NULL);
  dt_prefs_init_dialog_collect(dialog);
  g_signal_connect(dialog, "key-press-event", G_CALLBACK(dt_handle_dialog_enter), NULL);

#ifdef GDK_WINDOWING_QUARTZ
  dt_osx_disallow_fullscreen(dialog);
#endif
  gtk_widget_show_all(dialog);
  gtk_dialog_run(GTK_DIALOG(dialog));
  gtk_widget_destroy(dialog);
  dt_collection_update_query(darktable.collection, DT_COLLECTION_CHANGE_NEW_QUERY, DT_COLLECTION_PROP_UNDEF, NULL);
}

void set_preferences(void *menu, dt_lib_module_t *self)
{
  GtkWidget *mi = gtk_menu_item_new_with_label(_("preferences..."));
  g_signal_connect(G_OBJECT(mi), "activate", G_CALLBACK(_menuitem_preferences), self);
  gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);
}

static gint _sort_model_func(GtkTreeModel *model, GtkTreeIter *a, GtkTreeIter *b, dt_lib_module_t *self)
{
  gint ia, ib;
  gtk_tree_model_get(model, a, DT_LIB_COLLECT_COL_INDEX, &ia, -1);
  gtk_tree_model_get(model, b, DT_LIB_COLLECT_COL_INDEX, &ib, -1);
  return ib - ia;
}

#ifdef _WIN32
void _mount_changed(GVolumeMonitor *volume_monitor, GMount *mount, dt_lib_module_t *self)
#else
void _mount_changed(GUnixMountMonitor *monitor, dt_lib_module_t *self)
#endif
{
  dt_lib_collect_t *d = (dt_lib_collect_t *)self->data;
  dt_film_set_folder_status();
  // very rough update (rebuild the view). As these events are not too many that remains acceptable
  // adding film_id to treeview and listview would be cleaner to update just the parameter "reachable"
  dt_lib_collect_rule_t *dr = get_active_rule(d);
  const int property = _combo_get_active_collection(dr->combo);
  if(property == DT_COLLECTION_PROP_FOLDERS)
  {
    dr->typing = FALSE;
    d->view_rule = -1;
    tree_view(dr);
  }
  else if(property == DT_COLLECTION_PROP_FILMROLL)
  {
    dr->typing = FALSE;
    d->view_rule = -1;
    list_view(dr);
  }
}

void gui_init(dt_lib_module_t *self)
{
  dt_lib_collect_t *d = (dt_lib_collect_t *)calloc(1, sizeof(dt_lib_collect_t));

  self->data = (void *)d;
  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);

  d->active_rule = 0;
  d->nb_rules = 0;
  d->params = (dt_lib_collect_params_t *)malloc(sizeof(dt_lib_collect_params_t));

  GtkBox *box = NULL;
  GtkWidget *w = NULL;

  // Main notebook
  d->notebook = GTK_WIDGET(dt_ui_notebook_new());

  dt_ui_notebook_page(GTK_NOTEBOOK(d->notebook), _("Folders"), _("Browse the content of folders known to Ansel"));
  dt_ui_notebook_page(GTK_NOTEBOOK(d->notebook), _("Collections"), _("Query files by hierarchical tags"));
  dt_ui_notebook_page(GTK_NOTEBOOK(d->notebook), _("Queries"), _("Perform advanced queries into Ansel library"));
  gtk_widget_show_all(d->notebook); // needed to be able to switch pages programmatically

  gtk_box_pack_start(GTK_BOX(self->widget), d->notebook, TRUE, TRUE, 0);
  gtk_notebook_set_scrollable(GTK_NOTEBOOK(d->notebook), TRUE);
  g_signal_connect(G_OBJECT(d->notebook), "switch_page", G_CALLBACK(_lib_collect_mode), self);

  for(int i = 0; i < MAX_RULES; i++)
  {
    d->rule[i].num = i;
    d->rule[i].typing = FALSE;
    d->rule[i].lib_collect = (void *)d;

    box = GTK_BOX(gtk_box_new(GTK_ORIENTATION_VERTICAL, 0));
    d->rule[i].hbox = GTK_WIDGET(box);
    gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(box), TRUE, TRUE, 0);
    gtk_widget_set_name(GTK_WIDGET(box), "lib-dtbutton");

    d->rule[i].combo = dt_bauhaus_combobox_new(darktable.bauhaus, DT_GUI_MODULE(NULL));
    dt_bauhaus_combobox_set_selected_text_align(d->rule[i].combo, DT_BAUHAUS_COMBOBOX_ALIGN_RIGHT);
    _populate_collect_combo(d->rule[i].combo);

    g_signal_connect(G_OBJECT(d->rule[i].combo), "value-changed", G_CALLBACK(combo_changed), d->rule + i);
    gtk_box_pack_start(box, d->rule[i].combo, FALSE, FALSE, 0);

    GtkBox *hbox = GTK_BOX(gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0));
    gtk_box_pack_start(box, GTK_WIDGET(hbox), FALSE, FALSE, 0);

    w = gtk_search_entry_new();
    dt_accels_disconnect_on_text_input(w);
    d->rule[i].text = w;
    gtk_widget_add_events(w, GDK_FOCUS_CHANGE_MASK);
    gtk_entry_set_placeholder_text(GTK_ENTRY(d->rule[i].text), _("Search..."));
    g_signal_connect(G_OBJECT(w), "focus-in-event", G_CALLBACK(entry_focus_in_callback), d->rule + i);

    /* xgettext:no-c-format */
    gtk_widget_add_events(w, GDK_KEY_PRESS_MASK);
    g_signal_connect(G_OBJECT(w), "changed", G_CALLBACK(entry_changed), d->rule + i);
    g_signal_connect(G_OBJECT(w), "activate", G_CALLBACK(entry_activated), d->rule + i);
    gtk_widget_set_name(GTK_WIDGET(w), "lib-collect-entry");
    gtk_box_pack_start(hbox, w, TRUE, TRUE, 0);
    gtk_entry_set_width_chars(GTK_ENTRY(w), 5);

    d->rule[i].button = gtk_button_new();
    gtk_widget_set_events(d->rule[i].button, GDK_BUTTON_PRESS_MASK);
    g_signal_connect(G_OBJECT(d->rule[i].button), "button-press-event", G_CALLBACK(popup_button_callback), d->rule + i);
    gtk_box_pack_start(hbox, d->rule[i].button, FALSE, FALSE, 0);
  }

  GtkTreeView *view = GTK_TREE_VIEW(gtk_tree_view_new());
  d->view_rule = -1;
  d->view = view;
  gtk_tree_view_set_headers_visible(view, FALSE);
  gtk_tree_view_set_activate_on_single_click(view, TRUE);
  gtk_widget_set_can_focus(GTK_WIDGET(view), TRUE);
  g_signal_connect(G_OBJECT(view), "button-press-event", G_CALLBACK(view_onButtonPressed), d);
  g_signal_connect(G_OBJECT(view), "popup-menu", G_CALLBACK(view_onPopupMenu), d);

  GtkTreeViewColumn *col = gtk_tree_view_column_new();
  gtk_tree_view_append_column(view, col);
  GtkCellRenderer *renderer = gtk_cell_renderer_text_new();
  gtk_tree_view_column_pack_start(col, renderer, TRUE);
  gtk_tree_view_column_set_cell_data_func(col, renderer, tree_count_show, NULL, NULL);
  g_object_set(renderer, "strikethrough", TRUE, "ellipsize", PANGO_ELLIPSIZE_MIDDLE, (gchar *)0);
  gtk_tree_view_column_add_attribute(col, renderer, "strikethrough-set", DT_LIB_COLLECT_COL_UNREACHABLE);

  GtkTreeModel *listmodel
      = GTK_TREE_MODEL(gtk_list_store_new(DT_LIB_COLLECT_NUM_COLS, G_TYPE_STRING, G_TYPE_UINT, G_TYPE_STRING,
                                          G_TYPE_STRING, G_TYPE_BOOLEAN, G_TYPE_BOOLEAN, G_TYPE_UINT, G_TYPE_UINT));
  gtk_tree_sortable_set_sort_func(GTK_TREE_SORTABLE(listmodel), DT_LIB_COLLECT_COL_INDEX,
                  (GtkTreeIterCompareFunc)_sort_model_func, self, NULL);
  d->listfilter = gtk_tree_model_filter_new(listmodel, NULL);
  gtk_tree_model_filter_set_visible_column(GTK_TREE_MODEL_FILTER(d->listfilter), DT_LIB_COLLECT_COL_VISIBLE);

  GtkTreeModel *treemodel
      = GTK_TREE_MODEL(gtk_tree_store_new(DT_LIB_COLLECT_NUM_COLS, G_TYPE_STRING, G_TYPE_UINT, G_TYPE_STRING,
                                          G_TYPE_STRING, G_TYPE_BOOLEAN, G_TYPE_BOOLEAN, G_TYPE_UINT, G_TYPE_UINT));
  d->treefilter = gtk_tree_model_filter_new(treemodel, NULL);
  gtk_tree_model_filter_set_visible_column(GTK_TREE_MODEL_FILTER(d->treefilter), DT_LIB_COLLECT_COL_VISIBLE);
  g_object_unref(treemodel);

  gtk_box_pack_start(GTK_BOX(self->widget),
                     dt_ui_scroll_wrap(GTK_WIDGET(view), 400, "plugins/lighttable/collect/windowheight"), TRUE, TRUE, 0);

  /* setup proxy */
  darktable.view_manager->proxy.module_collect.module = self;
  darktable.view_manager->proxy.module_collect.update = _lib_collect_gui_update;

  _lib_collect_gui_update(self);

#ifdef _WIN32
  d->vmonitor = g_volume_monitor_get();
  g_signal_connect(G_OBJECT(d->vmonitor), "mount-changed", G_CALLBACK(_mount_changed), self);
  g_signal_connect(G_OBJECT(d->vmonitor), "mount-added", G_CALLBACK(_mount_changed), self);
#else
  d->vmonitor = g_unix_mount_monitor_get();
  g_signal_connect(G_OBJECT(d->vmonitor), "mounts-changed", G_CALLBACK(_mount_changed), self);
#endif

  DT_DEBUG_CONTROL_SIGNAL_CONNECT(darktable.signals, DT_SIGNAL_COLLECTION_CHANGED, G_CALLBACK(collection_updated),
                            self);

  DT_DEBUG_CONTROL_SIGNAL_CONNECT(darktable.signals, DT_SIGNAL_FILMROLLS_CHANGED, G_CALLBACK(filmrolls_updated),
                            self);

  DT_DEBUG_CONTROL_SIGNAL_CONNECT(darktable.signals, DT_SIGNAL_PREFERENCES_CHANGE, G_CALLBACK(preferences_changed),
                            self);

  DT_DEBUG_CONTROL_SIGNAL_CONNECT(darktable.signals, DT_SIGNAL_FILMROLLS_REMOVED, G_CALLBACK(filmrolls_removed),
                            self);

  DT_DEBUG_CONTROL_SIGNAL_CONNECT(darktable.signals, DT_SIGNAL_TAG_CHANGED, G_CALLBACK(tag_changed),
                            self);

  DT_DEBUG_CONTROL_SIGNAL_CONNECT(darktable.signals, DT_SIGNAL_GEOTAG_CHANGED, G_CALLBACK(geotag_changed),
                            self);

  DT_DEBUG_CONTROL_SIGNAL_CONNECT(darktable.signals, DT_SIGNAL_METADATA_CHANGED, G_CALLBACK(metadata_changed), self);

}

void gui_cleanup(dt_lib_module_t *self)
{
  dt_lib_collect_t *d = (dt_lib_collect_t *)self->data;

  DT_DEBUG_CONTROL_SIGNAL_DISCONNECT(darktable.signals, G_CALLBACK(collection_updated), self);
  DT_DEBUG_CONTROL_SIGNAL_DISCONNECT(darktable.signals, G_CALLBACK(filmrolls_updated), self);
  DT_DEBUG_CONTROL_SIGNAL_DISCONNECT(darktable.signals, G_CALLBACK(filmrolls_imported), self);
  DT_DEBUG_CONTROL_SIGNAL_DISCONNECT(darktable.signals, G_CALLBACK(preferences_changed), self);
  DT_DEBUG_CONTROL_SIGNAL_DISCONNECT(darktable.signals, G_CALLBACK(filmrolls_removed), self);
  DT_DEBUG_CONTROL_SIGNAL_DISCONNECT(darktable.signals, G_CALLBACK(tag_changed), self);
  DT_DEBUG_CONTROL_SIGNAL_DISCONNECT(darktable.signals, G_CALLBACK(geotag_changed), self);
  darktable.view_manager->proxy.module_collect.module = NULL;
  free(d->params);

  /* cleanup mem */

  g_object_unref(d->treefilter);
  g_object_unref(d->listfilter);
  g_object_unref(d->vmonitor);

  /* TODO: Make sure we are cleaning up all allocations */

  free(self->data);
  self->data = NULL;
}

static int is_time_property(int property)
{
  return (property == DT_COLLECTION_PROP_TIME
          || property == DT_COLLECTION_PROP_IMPORT_TIMESTAMP
          || property == DT_COLLECTION_PROP_CHANGE_TIMESTAMP
          || property == DT_COLLECTION_PROP_EXPORT_TIMESTAMP
          || property == DT_COLLECTION_PROP_PRINT_TIMESTAMP);
}

#ifdef USE_LUA
static int new_rule_cb(lua_State*L)
{
  dt_lib_collect_params_rule_t rule;
  memset(&rule, 0, sizeof(dt_lib_collect_params_rule_t));
  luaA_push(L, dt_lib_collect_params_rule_t, &rule);
  return 1;
}

static int filter_cb(lua_State *L)
{
  dt_lib_module_t *self = lua_touserdata(L, lua_upvalueindex(1));

  int size;
  dt_lib_collect_params_t *p = get_params(self, &size);
  // put it in stack so memory is not lost if a lua exception is raised

  if(lua_gettop(L) > 0)
  {
    luaL_checktype(L, 1, LUA_TTABLE);
    dt_lib_collect_params_t *new_p = get_params(self, &size);
    new_p->rules = 0;

    do
    {
      lua_pushinteger(L, new_p->rules + 1);
      lua_gettable(L, 1);
      if(lua_isnil(L, -1)) break;
      luaA_to(L, dt_lib_collect_params_rule_t, &new_p->rule[new_p->rules], -1);
      new_p->rules++;
    } while(new_p->rules < MAX_RULES);

    if(new_p->rules == MAX_RULES) {
      lua_pushinteger(L, new_p->rules + 1);
      lua_gettable(L, 1);
      if(!lua_isnil(L, -1)) {
        luaL_error(L, "Number of rules given exceeds max allowed (%d)", MAX_RULES);
      }
    }
    set_params(self, new_p, size);
    free(new_p);

  }

  lua_newtable(L);
  for(int i = 0; i < p->rules; i++)
  {
    luaA_push(L, dt_lib_collect_params_rule_t, &p->rule[i]);
    lua_seti(L, -2, i + 1);  // lua tables are 1 based
  }
  free(p);
  return 1;
}

static int mode_member(lua_State *L)
{
  dt_lib_collect_params_rule_t *rule = luaL_checkudata(L, 1, "dt_lib_collect_params_rule_t");

  if(lua_gettop(L) > 2)
  {
    dt_lib_collect_mode_t value;
    luaA_to(L, dt_lib_collect_mode_t, &value, 3);
    rule->mode = value;
    return 0;
  }

  const dt_lib_collect_mode_t tmp = rule->mode; // temp buffer because of bitfield in the original struct
  luaA_push(L, dt_lib_collect_mode_t, &tmp);
  return 1;
}

static int item_member(lua_State *L)
{
  dt_lib_collect_params_rule_t *rule = luaL_checkudata(L, 1, "dt_lib_collect_params_rule_t");

  if(lua_gettop(L) > 2)
  {
    dt_collection_properties_t value;
    luaA_to(L, dt_collection_properties_t, &value, 3);
    rule->item = value;
    return 0;
  }

  const dt_collection_properties_t tmp = rule->item; // temp buffer because of bitfield in the original struct
  luaA_push(L, dt_collection_properties_t, &tmp);
  return 1;
}

static int data_member(lua_State *L)
{
  dt_lib_collect_params_rule_t *rule = luaL_checkudata(L, 1, "dt_lib_collect_params_rule_t");

  if(lua_gettop(L) > 2)
  {
    size_t tgt_size;
    const char*data = luaL_checklstring(L, 3, &tgt_size);
    if(tgt_size > PARAM_STRING_SIZE)
    {
      return luaL_error(L, "string '%s' too long (max is %d)", data, PARAM_STRING_SIZE);
    }
    g_strlcpy(rule->string, data, sizeof(rule->string));
    return 0;
  }

  lua_pushstring(L, rule->string);
  return 1;
}

void init(struct dt_lib_module_t *self)
{
  lua_State *L = darktable.lua_state.state;
  int my_type = dt_lua_module_entry_get_type(L, "lib", self->plugin_name);
  lua_pushlightuserdata(L, self);
  lua_pushcclosure(L, filter_cb, 1);
  dt_lua_gtk_wrap(L);
  lua_pushcclosure(L, dt_lua_type_member_common, 1);
  dt_lua_type_register_const_type(L, my_type, "filter");
  lua_pushcfunction(L, new_rule_cb);
  lua_pushcclosure(L, dt_lua_type_member_common, 1);
  dt_lua_type_register_const_type(L, my_type, "new_rule");

  dt_lua_init_type(L, dt_lib_collect_params_rule_t);
  lua_pushcfunction(L, mode_member);
  dt_lua_type_register(L, dt_lib_collect_params_rule_t, "mode");
  lua_pushcfunction(L, item_member);
  dt_lua_type_register(L, dt_lib_collect_params_rule_t, "item");
  lua_pushcfunction(L, data_member);
  dt_lua_type_register(L, dt_lib_collect_params_rule_t, "data");


  luaA_enum(L, dt_lib_collect_mode_t);
  luaA_enum_value(L, dt_lib_collect_mode_t, DT_LIB_COLLECT_MODE_AND);
  luaA_enum_value(L, dt_lib_collect_mode_t, DT_LIB_COLLECT_MODE_OR);
  luaA_enum_value(L, dt_lib_collect_mode_t, DT_LIB_COLLECT_MODE_AND_NOT);

  luaA_enum(L, dt_collection_properties_t);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_FILMROLL);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_FOLDERS);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_CAMERA);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_TAG);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_DAY);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_TIME);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_IMPORT_TIMESTAMP);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_CHANGE_TIMESTAMP);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_EXPORT_TIMESTAMP);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_PRINT_TIMESTAMP);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_HISTORY);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_RATING);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_COLORLABEL);

  for(unsigned int i = 0; i < DT_METADATA_NUMBER; i++)
  {
    if(dt_metadata_get_type(i) != DT_METADATA_TYPE_INTERNAL)
    {
      const char *name = dt_metadata_get_name(i);
      gchar *setting = g_strdup_printf("plugins/lighttable/metadata/%s_flag", name);
      const gboolean hidden = dt_conf_get_int(setting) & DT_METADATA_FLAG_HIDDEN;
      g_free(setting);

      if(!hidden)
        luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_METADATA + i);
    }
  }

  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_LENS);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_FOCAL_LENGTH);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_ISO);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_APERTURE);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_EXPOSURE);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_FILENAME);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_GEOTAGGING);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_LOCAL_COPY);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_GROUPING);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_MODULE);
  luaA_enum_value(L, dt_collection_properties_t, DT_COLLECTION_PROP_ORDER);
}
#endif
#undef MAX_RULES
// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
