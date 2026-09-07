/*
 *    This file is part of Ansel,
 *    Copyright (C) 2026 Aurélien PIERRE.
 *
 *    Ansel is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    Ansel is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with Ansel.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "widgets/container.h"

#include "system/macros.h"   // IS_NULL_PTR

gboolean dt_gui_container_has_children(GtkContainer *container)
{
  g_return_val_if_fail(GTK_IS_CONTAINER(container), FALSE);
  GList *children = gtk_container_get_children(container);
  gboolean has_children = !IS_NULL_PTR(children);
  g_list_free(children);
  children = NULL;
  return has_children;
}

int dt_gui_container_num_children(GtkContainer *container)
{
  g_return_val_if_fail(GTK_IS_CONTAINER(container), FALSE);
  GList *children = gtk_container_get_children(container);
  int num_children = g_list_length(children);
  g_list_free(children);
  children = NULL;
  return num_children;
}

GtkWidget *dt_gui_container_first_child(GtkContainer *container)
{
  g_return_val_if_fail(GTK_IS_CONTAINER(container), NULL);
  GList *children = gtk_container_get_children(container);
  GtkWidget *child = children ? (GtkWidget*)children->data : NULL;
  g_list_free(children);
  children = NULL;
  return child;
}

GtkWidget *dt_gui_container_nth_child(GtkContainer *container, int which)
{
  g_return_val_if_fail(GTK_IS_CONTAINER(container), NULL);
  GList *children = gtk_container_get_children(container);
  GtkWidget *child = (GtkWidget*)g_list_nth_data(children, which);
  g_list_free(children);
  children = NULL;
  return child;
}

static void _remove_child(GtkWidget *widget, gpointer data)
{
  gtk_container_remove((GtkContainer*)data, widget);
}

void dt_gui_container_remove_children(GtkContainer *container)
{
  g_return_if_fail(GTK_IS_CONTAINER(container));
  gtk_container_foreach(container, _remove_child, container);
}

static void _delete_child(GtkWidget *widget, gpointer data)
{
  (void)data;  // avoid unreferenced-parameter warning
  gtk_widget_destroy(widget);
}

void dt_gui_container_destroy_children(GtkContainer *container)
{
  g_return_if_fail(GTK_IS_CONTAINER(container));
  gtk_container_foreach(container, _delete_child, NULL);
}

/* Typed, deliberately. The idiom this replaces was
 *   gtk_container_foreach(box, (GtkCallback)gtk_widget_set_can_focus, GINT_TO_POINTER(FALSE))
 * which calls a function through an incompatible pointer type: gtk_widget_set_can_focus() takes
 * a gboolean, GtkCallback passes a gpointer. It happens to work where the ABI leaves the int in
 * the low half of the register the pointer arrived in, and this binary is built with -flto, so
 * the two declarations meet at link time. */
static void _child_no_focus(GtkWidget *child, gpointer data)
{
  (void)data;  // avoid unreferenced-parameter warning
  gtk_widget_set_can_focus(child, FALSE);
}

void dt_gui_flow_box_as_layout(GtkFlowBox *box)
{
  g_return_if_fail(GTK_IS_FLOW_BOX(box));
  gtk_flow_box_set_selection_mode(box, GTK_SELECTION_NONE);
  gtk_container_foreach(GTK_CONTAINER(box), _child_no_focus, NULL);
}

void dt_gui_flow_box_child_as_layout(GtkWidget *child)
{
  g_return_if_fail(GTK_IS_WIDGET(child));
  // gtk_flow_box_insert() wraps the child; clearing can-focus on the CHILD would only stop the
  // widget the user came for from being reachable at all.
  GtkWidget *wrapper = gtk_widget_get_parent(child);
  if(GTK_IS_FLOW_BOX_CHILD(wrapper)) gtk_widget_set_can_focus(wrapper, FALSE);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
