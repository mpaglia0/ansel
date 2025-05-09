#pragma once
#include <gtk/gtk.h>

G_BEGIN_DECLS

#define DTGTK_TYPE_CELL_RENDERER_BUTTON (dtgtk_cell_renderer_button_get_type())
G_DECLARE_FINAL_TYPE(DtGtkCellRendererButton, dtgtk_cell_renderer_button, DTGTK, CELL_RENDERER_BUTTON,
                     GtkCellRendererPixbuf)

/* convenience constructor */
GtkCellRenderer *dtgtk_cell_renderer_button_new(void);

G_END_DECLS
