/*
    This file is part of the Ansel project.
    Copyright (C) 2026 Guillaume STUTIN.

    Ansel is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Ansel is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.
*/

/** Studio Capture: import module. The scan frequency and session status sit
    at the top, above two tabs holding the folder survey settings: "Source"
    (monitored folder) and "Destination" (copy mode, naming patterns,
    conflict policy...); the Start/Stop button is the notebook's action
    widget, on the tab row itself, to the right of the tab labels. Every
    widget persists its conf key immediately; the survey engine reads conf on
    each scan, so destination changes take effect on the next pass, while the
    source folder and the scan frequency are locked during a session to
    protect the engine's baseline. */

#include "common/darktable.h"
#include "common/datetime.h"
#include "common/debug.h"
#include "common/folder_survey.h"
#include "control/conf.h"
#include "control/control.h"
#include "control/jobs/import_jobs.h"
#include "gui/gtk.h"
#include "gui/gtkentry.h"
#include "libs/lib.h"
#include "libs/lib_api.h"

DT_MODULE(1)

typedef struct dt_lib_studio_import_t
{
  // Source tab
  GtkWidget *source_folder;

  // Destination tab
  GtkWidget *copy;
  // Container of every copy-only setting, hidden when "Add to library".
  GtkWidget *copy_options;
  GtkWidget *delete_source;
  GtkWidget *on_conflict;
  GtkWidget *datetime;
  GtkWidget *jobcode;
  GtkWidget *base_folder;
  GtkWidget *subfolder_pattern;
  GtkWidget *file_pattern;
  GtkWidget *preview;

  // Session controls: interval/status at the top, toggle as the notebook's action widget
  GtkWidget *interval;
  GtkWidget *status;
  GtkWidget *toggle;
} dt_lib_studio_import_t;

const char *name(dt_lib_module_t *self)
{
  return _("Auto import");
}

const char **views(dt_lib_module_t *self)
{
  static const char *v[] = { "studio_capture", NULL };
  return v;
}

uint32_t container(dt_lib_module_t *self)
{
  return DT_UI_CONTAINER_PANEL_LEFT_CENTER;
}

int position()
{
  return 990;
}

/**
 * @brief Sync the session controls and locks with the engine state.
 */
static void _studio_import_update_state(dt_lib_studio_import_t *d)
{
  const gboolean active = dt_folder_survey_is_active();

  // The engine compares scans against a baseline keyed on the source folder:
  // both stay locked while monitoring runs.
  gtk_widget_set_sensitive(d->source_folder, !active);
  gtk_widget_set_sensitive(d->interval, !active);

  g_signal_handlers_block_matched(d->toggle, G_SIGNAL_MATCH_DATA, 0, 0, NULL, NULL, d);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(d->toggle), active);
  g_signal_handlers_unblock_matched(d->toggle, G_SIGNAL_MATCH_DATA, 0, 0, NULL, NULL, d);
  gtk_button_set_label(GTK_BUTTON(d->toggle), active ? _("Stop the session") : _("Start the session"));

  if(active)
  {
    gtk_widget_set_sensitive(d->toggle, TRUE);
    char *folder = dt_conf_get_string("studio_capture/folder");
    gchar *status = g_strdup_printf(_("Status: Monitoring `%s`"), folder ? folder : "");
    gtk_label_set_text(GTK_LABEL(d->status), status);
    dt_free(status);
    dt_free(folder);
    return;
  }

  // Not running: gray out Start until the configuration has the minimum it
  // needs to succeed, and say why (in orange) in the status label.
  const char *message = NULL;
  const gboolean ready = dt_folder_survey_can_start(&message);
  gtk_widget_set_sensitive(d->toggle, ready);
  if(ready)
    gtk_label_set_text(GTK_LABEL(d->status), _("Ready to start the session."));
  else
  {
    gchar *markup = g_markup_printf_escaped("<span foreground='orange'>%s</span>", message);
    gtk_label_set_markup(GTK_LABEL(d->status), markup);
    dt_free(markup);
  }
}

static void _studio_import_update_preview(dt_lib_studio_import_t *d)
{
  const gboolean copy = dt_conf_get_bool("studio_capture/copy");

  // The copy-only settings are meaningless when images are added in place.
  // copy_options carries no-show-all so the framework's blanket show_all on
  // the module widget cannot re-show it behind our back.
  if(copy)
  {
    gtk_widget_set_no_show_all(d->copy_options, FALSE);
    gtk_widget_show_all(d->copy_options);
    gtk_widget_set_no_show_all(d->copy_options, TRUE);
  }
  else
    gtk_widget_hide(d->copy_options);

  if(!copy)
  {
    gtk_label_set_text(GTK_LABEL(d->preview), _("No copy. Images will be added from the surveyed folder."));
    return;
  }

  char *preview = dt_folder_survey_destination_preview();
  gtk_label_set_text(GTK_LABEL(d->preview),
                     preview ? preview : _("Can't build a valid destination path. Check the settings above."));
  dt_free(preview);
}

static void _studio_import_source_folder_callback(GtkWidget *widget, gpointer user_data)
{
  char *folder = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(widget));
  if(!IS_NULL_PTR(folder))
  {
    dt_conf_set_string("studio_capture/folder", folder);
    dt_free(folder);
  }
  _studio_import_update_state((dt_lib_studio_import_t *)user_data);
}

static void _studio_import_interval_callback(GtkWidget *widget, gpointer user_data)
{
  dt_conf_set_int("studio_capture/interval", gtk_spin_button_get_value_as_int(GTK_SPIN_BUTTON(widget)));
}

static void _studio_import_toggle_callback(GtkWidget *widget, gpointer user_data)
{
  dt_lib_studio_import_t *d = (dt_lib_studio_import_t *)user_data;

  if(gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget)))
    dt_folder_survey_start();
  else
    dt_folder_survey_halt();

  // On start failure the engine stays inactive: reflect the real state.
  _studio_import_update_state(d);
}

static void _studio_import_survey_changed_callback(gpointer instance, gpointer user_data)
{
  _studio_import_update_state((dt_lib_studio_import_t *)user_data);
}

static void _studio_import_copy_callback(GtkWidget *widget, gpointer user_data)
{
  dt_conf_set_bool("studio_capture/copy", gtk_combo_box_get_active(GTK_COMBO_BOX(widget)) == 1);
  _studio_import_update_preview((dt_lib_studio_import_t *)user_data);
  _studio_import_update_state((dt_lib_studio_import_t *)user_data);
}

static void _studio_import_delete_callback(GtkWidget *widget, gpointer user_data)
{
  dt_conf_set_bool("studio_capture/delete_source",
                   gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget)));
}

static void _studio_import_conflict_callback(GtkWidget *widget, gpointer user_data)
{
  dt_conf_set_int("studio_capture/on_conflict", gtk_combo_box_get_active(GTK_COMBO_BOX(widget)));
}

static void _studio_import_datetime_callback(GtkWidget *widget, gpointer user_data)
{
  dt_conf_set_string("studio_capture/datetime", gtk_entry_get_text(GTK_ENTRY(widget)));
  _studio_import_update_preview((dt_lib_studio_import_t *)user_data);
  _studio_import_update_state((dt_lib_studio_import_t *)user_data);
}

static void _studio_import_jobcode_callback(GtkWidget *widget, gpointer user_data)
{
  dt_conf_set_string("studio_capture/jobcode", gtk_entry_get_text(GTK_ENTRY(widget)));
  _studio_import_update_preview((dt_lib_studio_import_t *)user_data);
  _studio_import_update_state((dt_lib_studio_import_t *)user_data);
}

static void _studio_import_base_folder_callback(GtkWidget *widget, gpointer user_data)
{
  char *folder = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(widget));
  if(!IS_NULL_PTR(folder))
  {
    dt_conf_set_string("studio_capture/base_directory_pattern", folder);
    dt_free(folder);
  }
  _studio_import_update_preview((dt_lib_studio_import_t *)user_data);
  _studio_import_update_state((dt_lib_studio_import_t *)user_data);
}

static void _studio_import_subfolder_callback(GtkWidget *widget, gpointer user_data)
{
  dt_conf_set_string("studio_capture/sub_directory_pattern", gtk_entry_get_text(GTK_ENTRY(widget)));
  _studio_import_update_preview((dt_lib_studio_import_t *)user_data);
  _studio_import_update_state((dt_lib_studio_import_t *)user_data);
}

static void _studio_import_file_pattern_callback(GtkWidget *widget, gpointer user_data)
{
  dt_conf_set_string("studio_capture/filename_pattern", gtk_entry_get_text(GTK_ENTRY(widget)));
  _studio_import_update_preview((dt_lib_studio_import_t *)user_data);
  _studio_import_update_state((dt_lib_studio_import_t *)user_data);
}

/**
 * @brief Pack one field as a start-aligned label above its input widget.
 */
static void _studio_import_pack_row(GtkBox *parent, const char *label_text, GtkWidget *widget)
{
  GtkWidget *row = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
  GtkWidget *label = gtk_label_new(label_text);
  gtk_widget_set_halign(label, GTK_ALIGN_START);
  gtk_box_pack_start(GTK_BOX(row), label, FALSE, FALSE, 0);
  gtk_box_pack_start(GTK_BOX(row), widget, FALSE, FALSE, 0);
  gtk_box_pack_start(parent, row, FALSE, FALSE, 0);
}

void gui_init(dt_lib_module_t *self)
{
  dt_lib_studio_import_t *d = (dt_lib_studio_import_t *)g_malloc0(sizeof(dt_lib_studio_import_t));
  self->data = (void *)d;

  self->widget = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_GUI_BOX_SPACING);

  /* Session controls: scan frequency + status, at the top of the module */
  
  d->datetime = gtk_entry_new();
  gtk_entry_set_text(GTK_ENTRY(d->datetime), dt_conf_get_string_const("studio_capture/datetime"));
  gtk_entry_set_placeholder_text(GTK_ENTRY(d->datetime), _("Current date at scan time"));
  // Unlike the regular Import dialog's own datetime field, this one has no calendar picker to
  // fall back on, so the expected format needs to be explicit. dt_string_to_datetime()
  // (common/datetime.c) overlays the typed text onto the "0001-01-01 00:00:00.000" template
  // byte-for-byte: a partial value is valid and gets the untyped tail filled from the template
  // (e.g. "2026" -> 2026-01-01 00:00:00.000), but the format itself (dashes/space/colons/dot at
  // these exact positions) must match.
  gtk_widget_set_tooltip_text(d->datetime,
                              _("Format: YYYY-MM-DD, optionally followed by HH:MM:SS.mmm\n"
                                "Partial values are completed with defaults, e.g. \"2026\" becomes "
                                "2026-01-01.\n"
                                "Leave empty to use the current date at scan time."));
  g_signal_connect(G_OBJECT(d->datetime), "changed", G_CALLBACK(_studio_import_datetime_callback), d);
  _studio_import_pack_row(GTK_BOX(self->widget), _("Project date"), d->datetime);

  d->jobcode = gtk_entry_new();
  gtk_entry_set_text(GTK_ENTRY(d->jobcode), dt_conf_get_string_const("studio_capture/jobcode"));
  g_signal_connect(G_OBJECT(d->jobcode), "changed", G_CALLBACK(_studio_import_jobcode_callback), d);
  _studio_import_pack_row(GTK_BOX(self->widget), _("Jobcode"), d->jobcode);
  
  d->interval = gtk_spin_button_new_with_range(2, 3600, 1);
  gtk_spin_button_set_value(GTK_SPIN_BUTTON(d->interval),
                            CLAMP(dt_conf_get_int("studio_capture/interval"), 2, 3600));
  gtk_widget_set_tooltip_text(d->interval, _("Applied the next time the session is started"));
  g_signal_connect(G_OBJECT(d->interval), "value-changed", G_CALLBACK(_studio_import_interval_callback), d);
  _studio_import_pack_row(GTK_BOX(self->widget), _("Scan frequency (seconds)"), d->interval);

  d->status = gtk_label_new("");
  gtk_widget_set_halign(d->status, GTK_ALIGN_START);
  gtk_label_set_line_wrap(GTK_LABEL(d->status), TRUE);
  gtk_label_set_ellipsize(GTK_LABEL(d->status), PANGO_ELLIPSIZE_MIDDLE);
  gtk_box_pack_start(GTK_BOX(self->widget), d->status, FALSE, FALSE, 0);





  GtkNotebook *notebook = dt_ui_notebook_new();
  GtkWidget *source_page
      = dt_ui_notebook_page(notebook, _("Source"), _("Folder monitored during the session"));
  GtkWidget *destination_page
      = dt_ui_notebook_page(notebook, _("Destination"), _("Where and how the captured images are imported"));
  gtk_box_pack_start(GTK_BOX(self->widget), GTK_WIDGET(notebook), FALSE, FALSE, 0);

  /* Start/Stop sits on the tab row itself, to the right of the tab labels. */
  d->toggle = gtk_toggle_button_new_with_label(_("Start the session"));
  gtk_widget_set_tooltip_text(d->toggle,
                              _("Monitor the source folder and import new images as they appear.\n"
                                "The first scan records the existing images as a baseline: only images "
                                "arriving afterwards are imported."));
  g_signal_connect(G_OBJECT(d->toggle), "toggled", G_CALLBACK(_studio_import_toggle_callback), d);
  gtk_notebook_set_action_widget(notebook, d->toggle, GTK_PACK_END);
  gtk_widget_show(d->toggle);

  /* Source tab */
  d->source_folder
      = gtk_file_chooser_button_new(_("Select a folder to survey"), GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER);
  const char *folder = dt_conf_get_string_const("studio_capture/folder");
  if(!IS_NULL_PTR(folder) && folder[0])
    gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(d->source_folder), folder);
  gtk_widget_set_tooltip_text(d->source_folder, _("Folder receiving the captured images to monitor"));
  g_signal_connect(G_OBJECT(d->source_folder), "file-set",
                   G_CALLBACK(_studio_import_source_folder_callback), d);
  _studio_import_pack_row(GTK_BOX(source_page), _("Folder to survey"), d->source_folder);

  d->delete_source = gtk_check_button_new_with_label(_("Delete original file"));
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(d->delete_source),
                               dt_conf_get_bool("studio_capture/delete_source"));
  gtk_widget_set_tooltip_text(d->delete_source,
                              _("Delete the original image file after verifying the complete copy."));
  g_signal_connect(G_OBJECT(d->delete_source), "toggled", G_CALLBACK(_studio_import_delete_callback), d);
  gtk_box_pack_start(GTK_BOX(source_page), d->delete_source, FALSE, FALSE, 0);

  /* Destination tab */
  d->copy = gtk_combo_box_text_new();
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(d->copy), _("Add to library"));
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(d->copy), _("Copy to disk"));
  gtk_combo_box_set_active(GTK_COMBO_BOX(d->copy), dt_conf_get_bool("studio_capture/copy"));
  g_signal_connect(G_OBJECT(d->copy), "changed", G_CALLBACK(_studio_import_copy_callback), d);
  _studio_import_pack_row(GTK_BOX(destination_page), _("File handling"), d->copy);

  d->copy_options = gtk_box_new(GTK_ORIENTATION_VERTICAL, DT_GUI_BOX_SPACING);
  gtk_widget_set_no_show_all(d->copy_options, TRUE);
  gtk_box_pack_start(GTK_BOX(destination_page), d->copy_options, FALSE, FALSE, 0);

  d->on_conflict = gtk_combo_box_text_new();
  // Order matches dt_import_onconflict_t
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(d->on_conflict), _("Skip"));
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(d->on_conflict), _("Overwrite"));
  gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(d->on_conflict), _("Create unique filename"));
  gtk_combo_box_set_active(GTK_COMBO_BOX(d->on_conflict),
                           CLAMP(dt_conf_get_int("studio_capture/on_conflict"), DT_IMPORT_ONCONFLICT_SKIP,
                                 DT_IMPORT_ONCONFLICT_UNIQUE));
  gtk_widget_set_tooltip_text(d->on_conflict,
                              _("Expected behaviour when the naming pattern produces a destination file "
                                "that already exists"));
  g_signal_connect(G_OBJECT(d->on_conflict), "changed", G_CALLBACK(_studio_import_conflict_callback), d);
  _studio_import_pack_row(GTK_BOX(d->copy_options), _("On conflict"), d->on_conflict);

  d->base_folder
      = gtk_file_chooser_button_new(_("Select a base directory"), GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER);
  const char *base_folder = dt_conf_get_string_const("studio_capture/base_directory_pattern");
  if(!IS_NULL_PTR(base_folder) && base_folder[0])
    gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(d->base_folder), base_folder);
  g_signal_connect(G_OBJECT(d->base_folder), "file-set", G_CALLBACK(_studio_import_base_folder_callback), d);
  _studio_import_pack_row(GTK_BOX(d->copy_options), _("Base directory"), d->base_folder);

  d->subfolder_pattern = gtk_entry_new();
  gtk_entry_set_text(GTK_ENTRY(d->subfolder_pattern),
                     dt_conf_get_string_const("studio_capture/sub_directory_pattern"));
  dt_gtkentry_setup_completion(GTK_ENTRY(d->subfolder_pattern), dt_gtkentry_get_default_path_compl_list(),
                               "$(");
  gtk_widget_set_tooltip_text(d->subfolder_pattern,
                              _("Start typing `$(` to see available variables through auto-completion"));
  g_signal_connect(G_OBJECT(d->subfolder_pattern), "changed",
                   G_CALLBACK(_studio_import_subfolder_callback), d);
  _studio_import_pack_row(GTK_BOX(d->copy_options), _("Project directory pattern"), d->subfolder_pattern);

  d->file_pattern = gtk_entry_new();
  gtk_entry_set_text(GTK_ENTRY(d->file_pattern), dt_conf_get_string_const("studio_capture/filename_pattern"));
  dt_gtkentry_setup_completion(GTK_ENTRY(d->file_pattern), dt_gtkentry_get_default_path_compl_list(), "$(");
  gtk_widget_set_tooltip_text(d->file_pattern,
                              _("Start typing `$(` to see available variables through auto-completion"));
  g_signal_connect(G_OBJECT(d->file_pattern), "changed",
                   G_CALLBACK(_studio_import_file_pattern_callback), d);
  _studio_import_pack_row(GTK_BOX(d->copy_options), _("File naming pattern"), d->file_pattern);

  d->preview = gtk_label_new("");
  gtk_widget_set_halign(d->preview, GTK_ALIGN_START);
  gtk_label_set_ellipsize(GTK_LABEL(d->preview), PANGO_ELLIPSIZE_MIDDLE);
  gtk_label_set_line_wrap(GTK_LABEL(d->preview), TRUE);
  gtk_box_pack_start(GTK_BOX(destination_page), d->preview, FALSE, FALSE, 0);

  _studio_import_update_preview(d);
  _studio_import_update_state(d);

  DT_DEBUG_CONTROL_SIGNAL_CONNECT(darktable.signals, DT_SIGNAL_FOLDER_SURVEY_CHANGED,
                                  G_CALLBACK(_studio_import_survey_changed_callback), d);
}

void view_enter(struct dt_lib_module_t *self, struct dt_view_t *old_view, struct dt_view_t *new_view)
{
  // The session may have been resumed or halted outside this module.
  _studio_import_update_state((dt_lib_studio_import_t *)self->data);
}

void gui_cleanup(dt_lib_module_t *self)
{
  DT_DEBUG_CONTROL_SIGNAL_DISCONNECT(darktable.signals, G_CALLBACK(_studio_import_survey_changed_callback),
                                     self->data);
  g_free(self->data);
  self->data = NULL;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
