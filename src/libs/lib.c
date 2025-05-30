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
#include "libs/lib.h"
#include "common/debug.h"
#include "common/module.h"
#include "control/conf.h"
#include "control/control.h"
#include "dtgtk/button.h"
#include "dtgtk/expander.h"
#include "dtgtk/icon.h"

#include "gui/gtk.h"
#include "gui/presets.h"
#ifdef GDK_WINDOWING_QUARTZ
#include "osx/osx.h"
#endif
#include <stdbool.h>
#include <stdlib.h>

typedef enum dt_action_element_lib_t
{
  DT_ACTION_ELEMENT_SHOW = 0,
  DT_ACTION_ELEMENT_RESET = 1,
  DT_ACTION_ELEMENT_PRESETS = 2,
} dt_action_element_lib_t;

typedef struct dt_lib_module_info_t
{
  char *plugin_name;
  int32_t version;
  char *params;
  int params_size;
  dt_lib_module_t *module;
} dt_lib_module_info_t;

typedef struct dt_lib_presets_edit_dialog_t
{
  GtkEntry *name, *description;
  char plugin_name[128];
  int32_t version;
  void *params;
  int32_t params_size;
  gchar *original_name;
  dt_lib_module_t *module;
  gint old_id;
} dt_lib_presets_edit_dialog_t;

gboolean dt_lib_is_visible_in_view(dt_lib_module_t *module, const dt_view_t *view)
{
  if(!module->views)
  {
    fprintf(stderr, "module %s doesn't have views flags\n", module->name(module));
    return FALSE;
  }

  const char **views = module->views(module);
  for(const char **iter = views; *iter; iter++)
  {
    if(!strcmp(*iter, "*") || !strcmp(*iter, view->module_name)) return TRUE;
  }
  return FALSE;
}

/** calls module->cleanup and closes the dl connection. */
static void dt_lib_unload_module(dt_lib_module_t *module);

static gchar *get_active_preset_name(dt_lib_module_info_t *minfo)
{
  sqlite3_stmt *stmt;
  // clang-format off
  DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT name, op_params, writeprotect"
      " FROM data.presets"
      " WHERE operation=?1 AND op_version=?2",
      -1, &stmt, NULL);
  // clang-format on
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, minfo->plugin_name, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, minfo->version);
  gchar *name = NULL;
  // collect all presets for op from db
  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    void *op_params = (void *)sqlite3_column_blob(stmt, 1);
    int32_t op_params_size = sqlite3_column_bytes(stmt, 1);
    if(op_params_size == minfo->params_size && !memcmp(minfo->params, op_params, op_params_size))
    {
      name = g_strdup((char *)sqlite3_column_text(stmt, 0));
      break;
    }
  }
  sqlite3_finalize(stmt);
  return name;
}

static void edit_preset(const char *name_in, dt_lib_module_info_t *minfo)
{
  // get the original name of the preset
  gchar *name = NULL;
  if(name_in == NULL)
  {
    name = get_active_preset_name(minfo);
    if(name == NULL) return;
  }
  else
    name = g_strdup(name_in);

  // find the rowid of the preset
  int rowid = -1;
  sqlite3_stmt *stmt;
  // clang-format off
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                              "SELECT rowid"
                              " FROM data.presets"
                              " WHERE name = ?1 AND operation = ?2 AND op_version = ?3",
                              -1, &stmt, NULL);
  // clang-format on
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, name, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, minfo->plugin_name, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 3, minfo->version);
  if(sqlite3_step(stmt) == SQLITE_ROW)
  {
    rowid = sqlite3_column_int(stmt, 0);
  }
  sqlite3_finalize(stmt);

  // if we don't have a valid rowid, just exit, there's a problem !
  if(rowid < 0) return;

  GtkWidget *window = dt_ui_main_window(darktable.gui->ui);
  dt_gui_presets_show_edit_dialog(name, minfo->plugin_name, rowid, NULL, NULL, TRUE, TRUE, FALSE,
                                  GTK_WINDOW(window));
}

static void menuitem_update_preset(GtkMenuItem *menuitem, dt_lib_module_info_t *minfo)
{
  char *name = g_object_get_data(G_OBJECT(menuitem), "dt-preset-name");

  gint res = GTK_RESPONSE_YES;

  if(dt_conf_get_bool("plugins/lighttable/preset/ask_before_delete_preset"))
  {
    GtkWidget *window = dt_ui_main_window(darktable.gui->ui);
    GtkWidget *dialog
      = gtk_message_dialog_new(GTK_WINDOW(window), GTK_DIALOG_DESTROY_WITH_PARENT, GTK_MESSAGE_QUESTION,
                               GTK_BUTTONS_YES_NO, _("do you really want to update the preset `%s'?"), name);
#ifdef GDK_WINDOWING_QUARTZ
    dt_osx_disallow_fullscreen(dialog);
#endif
    gtk_window_set_title(GTK_WINDOW(dialog), _("update preset?"));
    res = gtk_dialog_run(GTK_DIALOG(dialog));
    gtk_widget_destroy(dialog);
  }

  if(res == GTK_RESPONSE_YES)
  {
    // commit all the module fields
    sqlite3_stmt *stmt;
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                                "UPDATE data.presets"
                                " SET op_version=?2, op_params=?3"
                                " WHERE name=?4 AND operation=?1",
                                -1, &stmt, NULL);
    // clang-format on

    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, minfo->plugin_name, -1, SQLITE_TRANSIENT);
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, minfo->version);
    DT_DEBUG_SQLITE3_BIND_BLOB(stmt, 3, minfo->params, minfo->params_size, SQLITE_TRANSIENT);
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 4, name, -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    DT_DEBUG_CONTROL_SIGNAL_RAISE(darktable.signals, DT_SIGNAL_PRESETS_CHANGED,
                                  g_strdup(minfo->plugin_name));
  }
}

static void menuitem_new_preset(GtkMenuItem *menuitem, dt_lib_module_info_t *minfo)
{
  dt_lib_presets_remove(_("new preset"), minfo->plugin_name, minfo->version);

  // add new preset
  sqlite3_stmt *stmt;
  // clang-format off
  DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "INSERT INTO data.presets (name, description, operation, op_version, op_params,"
      "  blendop_params, blendop_version, enabled, model, maker, lens,"
      "  iso_min, iso_max, exposure_min, exposure_max, aperture_min, aperture_max,"
      "  focal_length_min, focal_length_max, writeprotect, "
      "  autoapply, filter, def, format)"
      " VALUES (?1, '', ?2, ?3, ?4, NULL, 0, 1, '%', "
      "         '%', '%', 0, 340282346638528859812000000000000000000, 0, 100000000, 0, 100000000,"
      "          0, 1000, 0, 0, 0, 0, 0)",
      -1, &stmt, NULL);
  // clang-format on
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, _("new preset"), -1, SQLITE_STATIC);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, minfo->plugin_name, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 3, minfo->version);
  DT_DEBUG_SQLITE3_BIND_BLOB(stmt, 4, minfo->params, minfo->params_size, SQLITE_TRANSIENT);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  // create a shortcut for the new entry

  // then show edit dialog
  edit_preset(_("new preset"), minfo);
}

static void menuitem_edit_preset(GtkMenuItem *menuitem, dt_lib_module_info_t *minfo)
{
  edit_preset(NULL, minfo);
}

static void menuitem_manage_presets(GtkMenuItem *menuitem, dt_lib_module_info_t *minfo)
{
  if(minfo->module->manage_presets) minfo->module->manage_presets(minfo->module);
}

static void menuitem_delete_preset(GtkMenuItem *menuitem, dt_lib_module_info_t *minfo)
{
  gchar *name = get_active_preset_name(minfo);
  if(name == NULL) return;

  gint res = GTK_RESPONSE_YES;

  if(dt_conf_get_bool("plugins/lighttable/preset/ask_before_delete_preset"))
  {
    GtkWidget *window = dt_ui_main_window(darktable.gui->ui);
    GtkWidget *dialog
      = gtk_message_dialog_new(GTK_WINDOW(window), GTK_DIALOG_DESTROY_WITH_PARENT, GTK_MESSAGE_QUESTION,
                               GTK_BUTTONS_YES_NO, _("do you really want to delete the preset `%s'?"), name);
#ifdef GDK_WINDOWING_QUARTZ
    dt_osx_disallow_fullscreen(dialog);
#endif
    gtk_window_set_title(GTK_WINDOW(dialog), _("delete preset?"));
    res = gtk_dialog_run(GTK_DIALOG(dialog));
    gtk_widget_destroy(dialog);
  }

  if(res == GTK_RESPONSE_YES)
  {
    dt_lib_presets_remove(name, minfo->plugin_name, minfo->version);

    DT_DEBUG_CONTROL_SIGNAL_RAISE(darktable.signals, DT_SIGNAL_PRESETS_CHANGED,
                                  g_strdup(minfo->plugin_name));
  }
  g_free(name);
}

gchar *dt_lib_presets_duplicate(const gchar *preset, const gchar *module_name, int module_version)
{
  sqlite3_stmt *stmt;

  // find the new name
  int i = 0;
  gboolean ko = TRUE;
  while(ko)
  {
    i++;
    gchar *tx = g_strdup_printf("%s_%d", preset, i);
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(
        dt_database_get(darktable.db),
        "SELECT name"
        " FROM data.presets"
        " WHERE operation = ?1 AND op_version = ?2 AND name = ?3", -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, module_name, -1, SQLITE_TRANSIENT);
    DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, module_version);
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 3, tx, -1, SQLITE_TRANSIENT);
    if(sqlite3_step(stmt) != SQLITE_ROW) ko = FALSE;
    sqlite3_finalize(stmt);
    g_free(tx);
  }
  gchar *nname = g_strdup_printf("%s_%d", preset, i);

  // and we duplicate the entry
  // clang-format off
  DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "INSERT INTO data.presets"
      " (name, description, operation, op_version, op_params, "
      "  blendop_params, blendop_version, enabled, model, maker, lens, "
      "  iso_min, iso_max, exposure_min, exposure_max, aperture_min, aperture_max, "
      "  focal_length_min, focal_length_max, writeprotect, "
      "  autoapply, filter, def, format) "
      "SELECT ?1, description, operation, op_version, op_params, "
      "  blendop_params, blendop_version, enabled, model, maker, lens, "
      "  iso_min, iso_max, exposure_min, exposure_max, aperture_min, aperture_max, "
      "  focal_length_min, focal_length_max, 0, "
      "  autoapply, filter, def, format"
      " FROM data.presets"
      " WHERE operation = ?2 AND op_version = ?3 AND name = ?4",
      -1, &stmt, NULL);
  // clang-format on
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, nname, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, module_name, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 3, module_version);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 4, preset, -1, SQLITE_TRANSIENT);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);

  return nname;
}

void dt_lib_presets_remove(const gchar *preset, const gchar *module_name, int module_version)
{
  sqlite3_stmt *stmt;
  // clang-format off
  DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "DELETE FROM data.presets"
      " WHERE name=?1 AND operation=?2 AND op_version=?3 AND writeprotect=0", -1, &stmt,
      NULL);
  // clang-format on
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, preset, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, module_name, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 3, module_version);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}

gboolean dt_lib_presets_apply(const gchar *preset, const gchar *module_name, int module_version)
{
  gboolean ret = TRUE;
  sqlite3_stmt *stmt;
  // clang-format off
  DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "SELECT op_params, writeprotect"
      " FROM data.presets"
      " WHERE operation = ?1 AND op_version = ?2 AND name = ?3",
      -1, &stmt, NULL);
  // clang-format on
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, module_name, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, module_version);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 3, preset, -1, SQLITE_TRANSIENT);

  int res = 0;
  if(sqlite3_step(stmt) == SQLITE_ROW)
  {
    const void *blob = sqlite3_column_blob(stmt, 0);
    int length = sqlite3_column_bytes(stmt, 0);
    int writeprotect = sqlite3_column_int(stmt, 1);
    if(blob)
    {
      for(const GList *it = darktable.lib->plugins; it; it = g_list_next(it))
      {
        dt_lib_module_t *module = (dt_lib_module_t *)it->data;
        if(!strncmp(module->plugin_name, module_name, 128))
        {
          gchar *tx = g_strdup_printf("plugins/darkroom/%s/last_preset", module_name);
          dt_conf_set_string(tx, preset);
          g_free(tx);
          res = module->set_params(module, blob, length);
          break;
        }
      }
    }

    if(!writeprotect) dt_gui_store_last_preset(preset);
  }
  else
    ret = FALSE;
  sqlite3_finalize(stmt);
  if(res)
  {
    dt_control_log(_("deleting preset for obsolete module"));
    dt_lib_presets_remove(preset, module_name, module_version);
  }
  return ret;
}

void dt_lib_presets_update(const gchar *preset, const gchar *module_name, int module_version, const gchar *newname,
                           const gchar *desc, const void *params, const int32_t params_size)
{
  sqlite3_stmt *stmt;
  // clang-format off
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                              "UPDATE data.presets"
                              " SET name = ?1, description = ?2, op_params = ?3"
                              " WHERE operation = ?4 AND op_version = ?5 AND name = ?6",
                              -1, &stmt, NULL);
  // clang-format on
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, newname, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, desc, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_BLOB(stmt, 3, params, params_size, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 4, module_name, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 5, module_version);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 6, preset, -1, SQLITE_TRANSIENT);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}

static void pick_callback(GtkMenuItem *menuitem, dt_lib_module_info_t *minfo)
{
  // apply preset via set_params
  const char *pn = g_object_get_data(G_OBJECT(menuitem), "dt-preset-name");
  dt_lib_presets_apply(pn, minfo->plugin_name, minfo->version);
}

static void free_module_info(GtkWidget *widget, gpointer user_data)
{
  dt_lib_module_info_t *minfo = (dt_lib_module_info_t *)user_data;
  g_free(minfo->plugin_name);
  free(minfo->params);
  free(minfo);
}

static void dt_lib_presets_popup_menu_show(dt_lib_module_info_t *minfo)
{
  GtkMenu *menu = darktable.gui->presets_popup_menu;
  if(menu) gtk_widget_destroy(GTK_WIDGET(menu));
  darktable.gui->presets_popup_menu = GTK_MENU(gtk_menu_new());
  menu = darktable.gui->presets_popup_menu;

  const gboolean hide_default = dt_conf_get_bool("plugins/lighttable/hide_default_presets");
  const gboolean default_first = dt_conf_get_bool("modules/default_presets_first");

  g_signal_connect(G_OBJECT(menu), "destroy", G_CALLBACK(free_module_info), minfo);

  GtkWidget *mi;
  int active_preset = -1, cnt = 0;
  gboolean selected_writeprotect = FALSE;
  sqlite3_stmt *stmt;
  // order like the pref value
  // clang-format off
  gchar *query = g_strdup_printf("SELECT name, op_params, writeprotect, description"
                                 " FROM data.presets"
                                 " WHERE operation=?1 AND op_version=?2"
                                 " ORDER BY writeprotect %s, LOWER(name), rowid",
                                 default_first ? "DESC" : "ASC");
  // clang-format on
  DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db), query, -1, &stmt, NULL);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, minfo->plugin_name, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 2, minfo->version);
  g_free(query);

  // collect all presets for op from db
  int found = 0;
  int last_wp = -1;
  while(sqlite3_step(stmt) == SQLITE_ROW)
  {
    // default vs built-in stuff
    const gboolean writeprotect = sqlite3_column_int(stmt, 2);
    if(hide_default && writeprotect)
    {
      // skip default module if set to hide them.
      continue;
    }
    if(last_wp == -1)
    {
      last_wp = writeprotect;
    }
    else if(last_wp != writeprotect)
    {
      last_wp = writeprotect;
      gtk_menu_shell_append(GTK_MENU_SHELL(menu), gtk_separator_menu_item_new());
    }

    void *op_params = (void *)sqlite3_column_blob(stmt, 1);
    int32_t op_params_size = sqlite3_column_bytes(stmt, 1);
    const char *name = (char *)sqlite3_column_text(stmt, 0);

    if(darktable.gui->last_preset && strcmp(darktable.gui->last_preset, name) == 0) found = 1;

    // selected in bold:
    // printf("comparing %d bytes to %d\n", op_params_size, minfo->params_size);
    // for(int k=0;k<op_params_size && !memcmp(minfo->params, op_params, k);k++) printf("compare [%c %c] %d:
    // %d\n",
    // ((const char*)(minfo->params))[k],
    // ((const char*)(op_params))[k],
    // k, memcmp(minfo->params, op_params, k));
    if(op_params_size == minfo->params_size && !memcmp(minfo->params, op_params, op_params_size))
    {
      active_preset = cnt;
      selected_writeprotect = writeprotect;
      mi = gtk_check_menu_item_new_with_label(name);
      dt_gui_add_class(mi, "dt_transparent_background");
      gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(mi), TRUE);
      dt_gui_add_class(mi, "active_menu_item");
    }
    else
    {
      mi = gtk_menu_item_new_with_label((const char *)name);
    }
    g_object_set_data_full(G_OBJECT(mi), "dt-preset-name", g_strdup(name), g_free);
    g_signal_connect(G_OBJECT(mi), "activate", G_CALLBACK(pick_callback), minfo);
    gtk_widget_set_tooltip_text(mi, (const char *)sqlite3_column_text(stmt, 3));
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);
    cnt++;
  }
  sqlite3_finalize(stmt);

  if(cnt > 0)
  {
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), gtk_separator_menu_item_new());
    cnt = 0;
  }

  if(minfo->module->manage_presets)
  {
    mi = gtk_menu_item_new_with_label(_("manage presets..."));
    g_signal_connect(G_OBJECT(mi), "activate", G_CALLBACK(menuitem_manage_presets), minfo);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);
    cnt++;
  }
  else if(active_preset >= 0) // FIXME: this doesn't seem to work.
  {
    if(!selected_writeprotect)
    {
      mi = gtk_menu_item_new_with_label(_("edit this preset.."));
      g_signal_connect(G_OBJECT(mi), "activate", G_CALLBACK(menuitem_edit_preset), minfo);
      gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);

      mi = gtk_menu_item_new_with_label(_("delete this preset"));
      g_signal_connect(G_OBJECT(mi), "activate", G_CALLBACK(menuitem_delete_preset), minfo);
      gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);
      cnt++;
    }
  }
  else
  {
    mi = gtk_menu_item_new_with_label(_("store new preset.."));
    if(minfo->params_size == 0)
    {
      gtk_widget_set_sensitive(GTK_WIDGET(mi), FALSE);
      gtk_widget_set_tooltip_text(mi, _("nothing to save"));
    }
    else
      g_signal_connect(G_OBJECT(mi), "activate", G_CALLBACK(menuitem_new_preset), minfo);
    gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);

    if(darktable.gui->last_preset && found)
    {
      char *markup = g_markup_printf_escaped("%s <span weight=\"bold\">%s</span>", _("update preset"),
                                             darktable.gui->last_preset);
      mi = gtk_menu_item_new_with_label("");
      gtk_widget_set_sensitive(GTK_WIDGET(mi), minfo->params_size > 0);
      gtk_label_set_markup(GTK_LABEL(gtk_bin_get_child(GTK_BIN(mi))), markup);
      g_object_set_data_full(G_OBJECT(mi), "dt-preset-name", g_strdup(darktable.gui->last_preset), g_free);
      g_signal_connect(G_OBJECT(mi), "activate", G_CALLBACK(menuitem_update_preset), minfo);
      gtk_menu_shell_append(GTK_MENU_SHELL(menu), mi);
      g_free(markup);
    }
    cnt++;
  }

  if(minfo->module->set_preferences)
  {
    if(cnt>0)
    {
      gtk_menu_shell_append(GTK_MENU_SHELL(menu), gtk_separator_menu_item_new());
    }
    minfo->module->set_preferences(GTK_MENU_SHELL(menu), minfo->module);
  }
}

gint dt_lib_sort_plugins(gconstpointer a, gconstpointer b)
{
  const dt_lib_module_t *am = (const dt_lib_module_t *)a;
  const dt_lib_module_t *bm = (const dt_lib_module_t *)b;
  const int apos = (am && am->position) ? am->position() : 0;
  const int bpos = (bm && bm->position) ? bm->position() : 0;
  return apos - bpos;
}

/* default expandable implementation */
static gboolean default_expandable(dt_lib_module_t *self)
{
  return TRUE;
}

/* default autoapply implementation */
static gboolean default_preset_autoapply(dt_lib_module_t *self)
{
  return FALSE;
}

int default_lib_focus(dt_gui_module_t *m, gboolean toogle)
{
  dt_lib_module_t *module = (dt_lib_module_t *) m;
  dt_lib_gui_set_expanded(module, !dt_lib_gui_get_expanded(module));
  return 1;
}

static int _lib_plugin_body_button_press(GtkWidget *w, GdkEventButton *e, gpointer user_data)
{
  /* Reset the scrolling focus. If the click happened on any bauhaus element,
   * its internal button_press method will set it for itself */
  darktable.gui->has_scroll_focus = NULL;
  int handled = FALSE;
  return handled;
}

static int dt_lib_load_module(void *m, const char *libname, const char *module_name)
{
  dt_lib_module_t *module = (dt_lib_module_t *)m;

#define INCLUDE_API_FROM_MODULE_LOAD "lib_load_module"
#include "libs/lib_api.h"

  // Load modules only if they belong to a loaded view.
  // Designed for print settings, which loads all installed CUPS printers at gui_init() time,
  // which can take some time.
  gboolean load = FALSE;
  for(const char **view_m = module->views(module); *view_m; ++view_m)
  {
    for(GList *iter = darktable.view_manager->views; iter; iter = g_list_next(iter))
    {
      dt_view_t *view = (dt_view_t *)iter->data;
      if(!g_strcmp0(view->module_name, *view_m) || !g_strcmp0("*", *view_m) || !g_strcmp0("special", *view_m))
      {
        load = TRUE;
        break;
      }
    }
    if(load) break;
  }

  if(!load) return 1;

  g_strlcpy(module->plugin_name, module_name, sizeof(module->plugin_name));

  if(((!module->get_params || !module->set_params)
      && (module->legacy_params || module->set_params || module->get_params))
     || (!module->init_presets && module->manage_presets))
  {
    fprintf(stderr,"[dt_lib_load_module] illegal method combination in '%s'\n", module->plugin_name);
  }

  if(!module->get_params || !module->set_params)
  {
    // need all at the same time, or none, note that in this case
    // all the presets for the corresponding module will be deleted.
    // see: dt_lib_init_presets.
    module->legacy_params = NULL;
    module->set_params = NULL;
    module->get_params = NULL;
    module->manage_presets = NULL;
  }

  module->widget = NULL;
  module->expander = NULL;
  module->arrow = NULL;
  module->reset_button = NULL;
  module->presets_button = NULL;

#ifdef USE_LUA
  dt_lua_lib_register(darktable.lua_state.state, module);
#endif
  if(module->init) module->init(module);

  /* pass on the dt_gui_module_t args for bauhaus widgets */
  module->common_fields.name = g_strdup(module->name(module));
  module->common_fields.view = NULL; // view is set at gui_init time
  module->common_fields.widget_list = NULL;
  module->common_fields.widget_list_bh = NULL;
  module->common_fields.focus = module->lib_focus;
  module->common_fields.deprecated = FALSE;

  return 0;
}

static void *_update_params(dt_lib_module_t *module,
                            const void *const old_params, size_t old_params_size, int old_version,
                            int target_version, size_t *new_size)
{
  // make a copy of the old params so we can free it in the loop
  void *params = malloc(old_params_size);
  if(params == NULL) return NULL;
  memcpy(params, old_params, old_params_size);
  while(old_version < target_version)
  {
    size_t size;
    int version;
    void *new_params = module->legacy_params(module, params, old_params_size, old_version, &version, &size);
    free(params);
    if(new_params == NULL) return NULL;
    params = new_params;
    old_version = version;
    old_params_size = size;
  }
  *new_size = old_params_size;
  return params;
}

void dt_lib_init_presets(dt_lib_module_t *module)
{
  // since lighttable presets can't end up in styles or any other place outside of the presets table it is
  // sufficient
  // to update that very table here and assume that everything is up to date elsewhere.
  // the intended logic is as follows:
  // - no set_params -> delete all presets
  // - op_version >= module_version -> done
  // - op_version < module_version ->
  //   - module has legacy_params -> try to update
  //   - module doesn't have legacy_params -> delete it

  if(module->set_params == NULL)
  {
    sqlite3_stmt *stmt;
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                                "DELETE FROM data.presets"
                                " WHERE operation=?1", -1,
                                &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, module->plugin_name, -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
  }
  else
  {
    sqlite3_stmt *stmt;
    // clang-format off
    DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                                "SELECT rowid, op_version, op_params, name"
                                " FROM data.presets"
                                " WHERE operation=?1",
                                -1, &stmt, NULL);
    // clang-format on
    DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, module->plugin_name, -1, SQLITE_TRANSIENT);
    while(sqlite3_step(stmt) == SQLITE_ROW)
    {
      int rowid = sqlite3_column_int(stmt, 0);
      int op_version = sqlite3_column_int(stmt, 1);
      void *op_params = (void *)sqlite3_column_blob(stmt, 2);
      size_t op_params_size = sqlite3_column_bytes(stmt, 2);
      const char *name = (char *)sqlite3_column_text(stmt, 3);

      int version = module->version();

      if(op_version < version)
      {
        size_t new_params_size = 0;
        void *new_params = NULL;

        if(module->legacy_params
          && (new_params = _update_params(module, op_params, op_params_size, op_version, version, &new_params_size)))
        {
          // write the updated preset back to db
          fprintf(stderr,
                  "[lighttable_init_presets] updating '%s' preset '%s' from version %d to version %d\n",
                  module->plugin_name, name, op_version, version);
          sqlite3_stmt *innerstmt;
          // clang-format off
          DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                                      "UPDATE data.presets"
                                      " SET op_version=?1, op_params=?2"
                                      " WHERE rowid=?3", -1,
                                      &innerstmt, NULL);
          // clang-format on
          DT_DEBUG_SQLITE3_BIND_INT(innerstmt, 1, version);
          DT_DEBUG_SQLITE3_BIND_BLOB(innerstmt, 2, new_params, new_params_size, SQLITE_TRANSIENT);
          DT_DEBUG_SQLITE3_BIND_INT(innerstmt, 3, rowid);
          sqlite3_step(innerstmt);
          sqlite3_finalize(innerstmt);
        }
        else
        {
          // delete the preset
          fprintf(stderr, "[lighttable_init_presets] Can't upgrade '%s' preset '%s' from version %d to %d, "
                          "no legacy_params() implemented or unable to update\n",
                  module->plugin_name, name, op_version, version);
          sqlite3_stmt *innerstmt;
          // clang-format off
          DT_DEBUG_SQLITE3_PREPARE_V2(dt_database_get(darktable.db),
                                      "DELETE FROM data.presets"
                                      " WHERE rowid=?1", -1,
                                      &innerstmt, NULL);
          // clang-format on
          DT_DEBUG_SQLITE3_BIND_INT(innerstmt, 1, rowid);
          sqlite3_step(innerstmt);
          sqlite3_finalize(innerstmt);
        }
        free(new_params);
      }
    }
    sqlite3_finalize(stmt);
  }

  if(module->init_presets)
    module->init_presets(module);

  DT_DEBUG_CONTROL_SIGNAL_RAISE(darktable.signals, DT_SIGNAL_PRESETS_CHANGED,
                                g_strdup(module->plugin_name));
}


static gboolean _lib_plugin_focus_accel(GtkAccelGroup *accel_group, GObject *accelerable, guint keyval,
                                        GdkModifierType modifier, gpointer data)
{
  dt_gui_module_t *module = (dt_gui_module_t *)data;
  if(!module || !module->focus) return FALSE;
  return module->focus(module, FALSE);
}


static void dt_lib_init_module(void *m)
{
  dt_lib_module_t *module = (dt_lib_module_t *)m;
  dt_lib_init_presets(module);

  if(darktable.gui)
  {
    module->gui_init(module);
    if(module->widget) g_object_ref_sink(module->widget);

    // TODO: look for active view. Do we know it at init time ?
    module->common_fields.view = g_strdup(_("Lighttable"));

    if(!module->views || (module->expandable && !module->expandable(module))) return; // We are done
    // Else: add accel pathes

    gchar *clean_name = delete_underscore(module->name(module));
    dt_capitalize_label(clean_name);

    // slash is not allowed in module names because that makes accel pathes fail
    assert(g_strrstr(clean_name, "/") == NULL);

    const char **views = module->views(module);

    // We add one accel path per view
    for(const char **view = views; *view; ++view)
    {
      GtkAccelGroup *accel_group = NULL;
      gchar *label = NULL;
      if(!g_strcmp0(*view, "darkroom"))
      {
        accel_group = darktable.gui->accels->darkroom_accels;
        label = g_strdup("Darkroom/Toolboxes");
      }
      else if(!g_strcmp0(*view, "lighttable"))
      {
        accel_group = darktable.gui->accels->lighttable_accels;
        label = g_strdup("Lighttable/Toolboxes");
      }
      // TODO: handle the other views

      if(accel_group && label)
        dt_accels_new_action_shortcut(darktable.gui->accels, _lib_plugin_focus_accel, m,
                                      accel_group, label, clean_name, 0, 0, FALSE,
                                      _("Focuses the module"));
      g_free(label);
    }

    g_free(clean_name);
  }
}

void dt_lib_unload_module(dt_lib_module_t *module)
{
  dt_gui_module_t *m = DT_GUI_MODULE(module);
  g_list_free(m->widget_list);
  g_list_free(m->widget_list_bh);
  g_free(m->name);
  g_free(m->view);

  if(module->module) g_module_close(module->module);
}

static void dt_lib_gui_reset_callback(GtkButton *button, gpointer user_data)
{
  dt_lib_module_t *module = (dt_lib_module_t *)user_data;
  module->gui_reset(module);
}

static void presets_popup_callback(GtkButton *button, dt_lib_module_t *module)
{
  dt_lib_module_info_t *mi = (dt_lib_module_info_t *)calloc(1, sizeof(dt_lib_module_info_t));

  mi->plugin_name = g_strdup(module->plugin_name);
  mi->version = module->version();
  mi->module = module;
  mi->params = module->get_params ? module->get_params(module, &mi->params_size) : NULL;
  if(!mi->params)
  {
    // this is a valid case, for example in location.c when nothing got selected
    // fprintf(stderr, "something went wrong: &params=%p, size=%i\n", mi->params, mi->params_size);
    mi->params_size = 0;
  }
  dt_lib_presets_popup_menu_show(mi);

  dt_gui_menu_popup(darktable.gui->presets_popup_menu, GTK_WIDGET(button), GDK_GRAVITY_SOUTH_EAST, GDK_GRAVITY_NORTH_EAST);

  if(button) dtgtk_button_set_active(DTGTK_BUTTON(button), FALSE);
}


void dt_lib_gui_set_expanded(dt_lib_module_t *module, gboolean expanded)
{
  if(!module->expander || !module->arrow) return;

  dtgtk_expander_set_expanded(DTGTK_EXPANDER(module->expander), expanded);

  /* update expander arrow state */
  gint flags = (expanded ? CPF_DIRECTION_DOWN : CPF_DIRECTION_RIGHT);
  dtgtk_button_set_paint(DTGTK_BUTTON(module->arrow), dtgtk_cairo_paint_solid_arrow, flags, NULL);

  /* show / hide plugin widget */
  if(expanded)
  {
    /* register to receive draw events */
    darktable.lib->gui_module = module;
    darktable.gui->scroll_to[1] = module->expander;
    gtk_widget_grab_focus(GTK_WIDGET(module->expander));
  }
  else
  {
    if(darktable.lib->gui_module == module)
    {
      darktable.lib->gui_module = NULL;
      dt_control_queue_redraw();
    }
    dt_gui_refocus_center();
  }

  /* store expanded state of module */
  char var[1024];
  const dt_view_t *current_view = dt_view_manager_get_current_view(darktable.view_manager);
  snprintf(var, sizeof(var), "plugins/%s/%s/expanded", current_view->module_name, module->plugin_name);
  dt_conf_set_bool(var, expanded);
}

gboolean dt_lib_gui_get_expanded(dt_lib_module_t *module)
{
  if(!module->expandable(module)) return true;
  if(!module->expander) return true;
  if(!module->widget)
  {
    char var[1024];
    const dt_view_t *current_view = dt_view_manager_get_current_view(darktable.view_manager);
    snprintf(var, sizeof(var), "plugins/%s/%s/expanded", current_view->module_name, module->plugin_name);
    return dt_conf_get_bool(var);
  }
  return dtgtk_expander_get_expanded(DTGTK_EXPANDER(module->expander));
}

static gboolean _lib_plugin_header_button_press(GtkWidget *w, GdkEventButton *e, gpointer user_data)
{
  if(e->type == GDK_2BUTTON_PRESS || e->type == GDK_3BUTTON_PRESS) return TRUE;

  dt_lib_module_t *module = (dt_lib_module_t *)user_data;

  /* Reset the scrolling focus. If the click happened on any bauhaus element,
   * its internal button_press method will set it for itself */
  darktable.gui->has_scroll_focus = NULL;

  if(e->button == 1)
  {
    /* bail out if module is static */
    if(!module->expandable(module)) return FALSE;

    // make gtk scroll to the module once it updated its allocation size
    uint32_t container = module->container(module);
    if(container == DT_UI_CONTAINER_PANEL_LEFT_CENTER)
      darktable.gui->scroll_to[0] = module->expander;
    else if(container == DT_UI_CONTAINER_PANEL_RIGHT_CENTER)
      darktable.gui->scroll_to[1] = module->expander;

    gtk_widget_grab_focus(GTK_WIDGET(module->expander));

    /* handle shiftclick on expander, hide all except this */
    if(dt_modifier_is(e->state, GDK_SHIFT_MASK))
    {
      const dt_view_t *v = dt_view_manager_get_current_view(darktable.view_manager);
      gboolean all_other_closed = TRUE;
      for(const GList *it = darktable.lib->plugins; it; it = g_list_next(it))
      {
        dt_lib_module_t *m = (dt_lib_module_t *)it->data;

        if(m != module && container == m->container(m) && m->expandable(m) && dt_lib_is_visible_in_view(m, v))
        {
          all_other_closed = all_other_closed && !dtgtk_expander_get_expanded(DTGTK_EXPANDER(m->expander));
          dt_lib_gui_set_expanded(m, FALSE);
        }
      }
      if(all_other_closed)
        dt_lib_gui_set_expanded(module, !dtgtk_expander_get_expanded(DTGTK_EXPANDER(module->expander)));
      else
        dt_lib_gui_set_expanded(module, TRUE);
    }
    else
    {
      /* else just toggle */
      dt_lib_gui_set_expanded(module, !dtgtk_expander_get_expanded(DTGTK_EXPANDER(module->expander)));
    }

    return TRUE;
  }
  else if(e->button == 3)
  {
    if(gtk_widget_get_sensitive(module->presets_button))
      presets_popup_callback(NULL, module);

    return TRUE;
  }
  return FALSE;
}

#if 0
static void show_module_callback(dt_lib_module_t *module)
{
  /* bail out if module is static */
  if(!module->expandable(module)) return;

  // make gtk scroll to the module once it updated its allocation size
  uint32_t container = module->container(module);
  if(container == DT_UI_CONTAINER_PANEL_LEFT_CENTER)
    darktable.gui->scroll_to[0] = module->expander;
  else if(container == DT_UI_CONTAINER_PANEL_RIGHT_CENTER)
    darktable.gui->scroll_to[1] = module->expander;

  dt_lib_gui_set_expanded(module, !dtgtk_expander_get_expanded(DTGTK_EXPANDER(module->expander)));
}

#endif

GtkWidget *dt_lib_gui_get_expander(dt_lib_module_t *module)
{
  /* check if module is expandable */
  if(!module->expandable(module))
  {
    if(module->presets_button)
    {
      // FIXME separately define as darkroom widget shortcut/action, because not automatically registered via lib
      // if presets btn has been loaded to be shown outside expander
      g_signal_connect(G_OBJECT(module->presets_button), "clicked", G_CALLBACK(presets_popup_callback), module);
    }
    module->expander = NULL;
    return NULL;
  }

  GtkWidget *header = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 0);
  gtk_widget_set_name(GTK_WIDGET(header), "module-header");

  GtkWidget *expander = dtgtk_expander_new(header, module->widget);
  dt_gui_add_class(expander, "dt_module_frame");

  GtkWidget *header_evb = dtgtk_expander_get_header_event_box(DTGTK_EXPANDER(expander));
  GtkWidget *body_evb = dtgtk_expander_get_body_event_box(DTGTK_EXPANDER(expander));
  GtkWidget *pluginui_frame = dtgtk_expander_get_frame(DTGTK_EXPANDER(expander));
  dt_gui_add_class(pluginui_frame, "dt_plugin_ui");

  /* setup the header box */
  g_signal_connect(G_OBJECT(header_evb), "button-press-event", G_CALLBACK(_lib_plugin_header_button_press),
                   module);

  /* connect mouse button callbacks for focus and presets */
  g_signal_connect(G_OBJECT(body_evb), "button-press-event", G_CALLBACK(_lib_plugin_body_button_press), module);
  gtk_widget_add_events(body_evb, GDK_POINTER_MOTION_MASK);

  /*
   * initialize the header widgets
   */
  /* add the expand indicator icon */
  module->arrow = dtgtk_button_new(dtgtk_cairo_paint_solid_arrow, 0, NULL);
  gtk_widget_set_tooltip_text(module->arrow, _("show module"));
  g_signal_connect(G_OBJECT(module->arrow), "button-press-event", G_CALLBACK(_lib_plugin_header_button_press), module);
  gtk_box_pack_start(GTK_BOX(header), module->arrow, FALSE, FALSE, 0);

  /* add module label */
  GtkWidget *label = gtk_label_new("");
  GtkWidget *label_evb = gtk_event_box_new();
  gtk_container_add(GTK_CONTAINER(label_evb), label);
  gchar *mname = g_markup_escape_text(module->name(module), -1);
  dt_capitalize_label(mname);
  gtk_label_set_markup(GTK_LABEL(label), mname);
  gtk_widget_set_tooltip_text(label_evb, mname);
  g_free(mname);
  gtk_label_set_ellipsize(GTK_LABEL(label), PANGO_ELLIPSIZE_END);
  g_object_set(G_OBJECT(label), "halign", GTK_ALIGN_START, "xalign", 0.0, (gchar *)0);
  gtk_widget_set_name(label, "lib-panel-label");
  gtk_box_pack_start(GTK_BOX(header), label_evb, FALSE, FALSE, 0);

  /* add preset button if module has implementation */
  module->presets_button = dtgtk_button_new(dtgtk_cairo_paint_presets, 0, NULL);
  g_signal_connect(G_OBJECT(module->presets_button), "clicked", G_CALLBACK(presets_popup_callback), module);
  if(!module->get_params && !module->set_preferences) gtk_widget_set_sensitive(GTK_WIDGET(module->presets_button), FALSE);
  gtk_box_pack_end(GTK_BOX(header), module->presets_button, FALSE, FALSE, 0);

  /* add reset button if module has implementation */
  module->reset_button = dtgtk_button_new(dtgtk_cairo_paint_reset, 0, NULL);
  g_signal_connect(G_OBJECT(module->reset_button), "clicked", G_CALLBACK(dt_lib_gui_reset_callback), module);
  if(!module->gui_reset) gtk_widget_set_sensitive(GTK_WIDGET(module->reset_button), FALSE);
  gtk_box_pack_end(GTK_BOX(header), module->reset_button, FALSE, FALSE, 0);

  gtk_widget_show_all(GTK_WIDGET(module->widget));
  dt_gui_add_class(module->widget, "dt_plugin_ui_main");
  module->expander = expander;

  gtk_widget_set_hexpand(module->widget, FALSE);
  gtk_widget_set_vexpand(module->widget, FALSE);

  return module->expander;
}

void dt_lib_init(dt_lib_t *lib)
{
  // Setting everything to null initially
  memset(lib, 0, sizeof(dt_lib_t));
  darktable.lib->plugins = dt_module_load_modules("/plugins/lighttable", sizeof(dt_lib_module_t),
                                                  dt_lib_load_module, dt_lib_init_module, dt_lib_sort_plugins);
}

void dt_lib_cleanup(dt_lib_t *lib)
{
  while(lib->plugins)
  {
    dt_lib_module_t *module = (dt_lib_module_t *)(lib->plugins->data);
    if(module)
    {
      if(module->data != NULL)
      {
        module->gui_cleanup(module);
        module->data = NULL;
      }
      dt_lib_unload_module(module);
      free(module);
    }
    lib->plugins = g_list_delete_link(lib->plugins, lib->plugins);
  }
}

void dt_lib_presets_add(const char *name, const char *plugin_name, const int32_t version, const void *params,
                        const int32_t params_size, gboolean readonly)
{
  dt_lib_presets_remove(name, plugin_name, version);

  sqlite3_stmt *stmt;
  // clang-format off
  DT_DEBUG_SQLITE3_PREPARE_V2(
      dt_database_get(darktable.db),
      "INSERT INTO data.presets"
      " (name, description, operation, op_version, op_params, "
      "  blendop_params, blendop_version, enabled, model, maker, lens, "
      "  iso_min, iso_max, exposure_min, exposure_max, aperture_min, aperture_max, "
      "  focal_length_min, focal_length_max, writeprotect, "
      "  autoapply, filter, def, format)"
      " VALUES "
      "  (?1, '', ?2, ?3, ?4, NULL, 0, 1, '%', "
      "   '%', '%', 0, 340282346638528859812000000000000000000, 0, 10000000, 0, 100000000, 0,"
      "   1000, ?5, 0, 0, 0, 0)",
      -1, &stmt, NULL);
  // clang-format on
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 1, name, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_TEXT(stmt, 2, plugin_name, -1, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 3, version);
  DT_DEBUG_SQLITE3_BIND_BLOB(stmt, 4, params, params_size, SQLITE_TRANSIENT);
  DT_DEBUG_SQLITE3_BIND_INT(stmt, 5, readonly);
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}

static gchar *_get_lib_view_path(dt_lib_module_t *module, char *suffix)
{
  if(!darktable.view_manager) return NULL;
  const dt_view_t *cv = dt_view_manager_get_current_view(darktable.view_manager);
  // in lighttable, we store panels states per layout
  char lay[32] = "";
  if(g_strcmp0(cv->module_name, "lighttable") == 0)
  {
    g_snprintf(lay, sizeof(lay), "%d/", 0);
  }
  else if(g_strcmp0(cv->module_name, "darkroom") == 0)
  {
    g_snprintf(lay, sizeof(lay), "%d/", dt_view_darkroom_get_layout(darktable.view_manager));
  }

  return g_strdup_printf("plugins/%s/%s%s%s", cv->module_name, lay, module->plugin_name, suffix);
}

gboolean dt_lib_is_visible(dt_lib_module_t *module)
{
  gchar *key = _get_lib_view_path(module, "_visible");
  gboolean ret = TRUE; /* if not key found, always make module visible */
  if(key && dt_conf_key_exists(key)) ret = dt_conf_get_bool(key);
  g_free(key);

  return ret;
}

void dt_lib_set_visible(dt_lib_module_t *module, gboolean visible)
{
  gchar *key = _get_lib_view_path(module, "_visible");
  GtkWidget *widget;
  dt_conf_set_bool(key, visible);
  g_free(key);
  if(module->widget)
  {
    if(module->expander)
      widget = module->expander;
    else
      widget = module->widget;


    if(visible)
      gtk_widget_show(GTK_WIDGET(widget));
    else
      gtk_widget_hide(GTK_WIDGET(widget));
  }
}

gchar *dt_lib_get_localized_name(const gchar *plugin_name)
{
  // Prepare mapping op -> localized name
  static GHashTable *module_names = NULL;
  if(module_names == NULL)
  {
    module_names = g_hash_table_new(g_str_hash, g_str_equal);
    for(const GList *lib = darktable.lib->plugins; lib; lib = g_list_next(lib))
    {
      dt_lib_module_t *module = (dt_lib_module_t *)lib->data;
      g_hash_table_insert(module_names, module->plugin_name, g_strdup(module->name(module)));
    }
  }

  return (gchar *)g_hash_table_lookup(module_names, plugin_name);
}

void dt_lib_colorpicker_set_box_area(dt_lib_t *lib, const dt_boundingbox_t box)
{
  if(!lib->proxy.colorpicker.module || !lib->proxy.colorpicker.set_sample_box_area) return;
  lib->proxy.colorpicker.set_sample_box_area(lib->proxy.colorpicker.module, box);
  dt_gui_refocus_center();
}

void dt_lib_colorpicker_set_point(dt_lib_t *lib, const float pos[2])
{
  if(!lib->proxy.colorpicker.module || !lib->proxy.colorpicker.set_sample_point) return;
  lib->proxy.colorpicker.set_sample_point(lib->proxy.colorpicker.module, pos);
  dt_gui_refocus_center();
}

dt_lib_module_t *dt_lib_get_module(const char *name)
{
  /* hide/show modules as last config */
  for(GList *iter = darktable.lib->plugins; iter; iter = g_list_next(iter))
  {
    dt_lib_module_t *plugin = (dt_lib_module_t *)(iter->data);
    if(strcmp(plugin->plugin_name, name) == 0)
      return plugin;
  }

  return NULL;
}

/* callback function for delayed update after user interaction */
static gboolean _postponed_update(gpointer data)
{
  dt_lib_module_t *self = (dt_lib_module_t *)data;
  self->timeout_handle = 0;
  if (self->_postponed_update)
    self->_postponed_update(self);

  return FALSE; // cancel the timer
}

/** queue a delayed call of update function after user interaction */
void dt_lib_queue_postponed_update(dt_lib_module_t *mod, void (*update_fn)(dt_lib_module_t *self))
{
  if(mod->timeout_handle)
  {
    // here we're making sure the event fires at last hover
    // and we won't have avalanche of events in the mean time.
    g_source_remove(mod->timeout_handle);
  }
  const int delay = CLAMP(darktable.develop->average_delay / 2, 10, 250);
  mod->_postponed_update = update_fn;
  mod->timeout_handle = g_timeout_add(delay, _postponed_update, mod);
}

void dt_lib_cancel_postponed_update(dt_lib_module_t *mod)
{
  mod->_postponed_update = NULL;
  if (mod->timeout_handle)
  {
    g_source_remove(mod->timeout_handle);
    mod->timeout_handle = 0;
  }
}

gboolean dt_lib_presets_can_autoapply(dt_lib_module_t *mod)
{
  return mod->preset_autoapply(mod);
}

gboolean dt_handle_dialog_enter(GtkWidget *widget, GdkEventKey *event, gpointer data)
{
  if(event->keyval == GDK_KEY_Return || event->keyval == GDK_KEY_KP_Enter)
  {
    gtk_dialog_response(GTK_DIALOG(widget), GTK_RESPONSE_ACCEPT);
    return TRUE;
  }
  return FALSE;
}

GtkWidget *dt_action_button_new(dt_lib_module_t *self, const gchar *label, gpointer callback, gpointer data, const gchar *tooltip, guint accel_key, GdkModifierType mods)
{
  gchar *label_copy = g_strdup(_(label));
  dt_capitalize_label(label_copy);
  GtkWidget *button = gtk_button_new_with_label(label_copy);
  g_free(label_copy);
  gtk_label_set_ellipsize(GTK_LABEL(gtk_bin_get_child(GTK_BIN(button))), PANGO_ELLIPSIZE_END);
  if(tooltip) gtk_widget_set_tooltip_text(button, tooltip);
  g_signal_connect(G_OBJECT(button), "clicked", G_CALLBACK(callback), data);
  return button;
}



// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
