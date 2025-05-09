#include <gdk/gdkkeysyms.h>
#include <gtk/gtk.h>
#ifdef GDK_WINDOWING_WAYLAND
#include <gdk/gdkwayland.h>
#endif

/**
 * @file accelerators.h
 * @brief Handle default and user-set shortcuts (accelerators)
 *
 * Gtk is a bit weird here, so we need to :
 *
 *  1. have each acceleratable widget declare its accel path and default shortcut (if any),
 *  2. read the accels map file, which only import accels for already-known pathes (from previous step),
 *     and may update the default shortcut with user-defined one,
 *  3. connect widget relevant signal and shortcut through the active GtkAccelGroup. But the actual
 *     signal depends on the widget type.
 *
 * Because of that, we need to prepare the list of acceleratable widgets first, populate the accel maps
 * with their pathes, and only after fetching user-defined accels, we can connect actual actions.
 * So it's not straight-forward.
 *
 * That is:
 *
 * ```C
 * // 1. Init accels handlers
 * dt_accels_init(char *config_file, GtkAccelFlags flags);
 *
 * // 2. Init GUI widgets where each acceleratable declares its slot using either:
 * dt_accels_new_widget_shortcut(dt_accels_t *accels, GtkWidget *widget, const gchar *signal,
 *                               GtkAccelGroup *accel_group, const gchar *accel_path, guint key_val,
 *                               GdkModifierType accel_mods, const gboolean lock);
 * // or:
 * const dt_shortcut_t *dt_accels_new_action_shortcut(dt_accels_t *accels, void(*action_callback), gpointer data,
 *                                                    GtkAccelGroup *accel_group, const gchar *action_scope, const gchar *action_name,
 *                                                    guint key_val, GdkModifierType accel_mods, const gboolean lock);
 *
 * // 3. Read shortcutsrc.lang file to assign user-defined shortcuts to those slots:
 * void dt_accels_load_user_config(dt_accels_t *accels);
 *
 * // 4. Actually enable those shortcuts:
 * void dt_accels_connect_accels(dt_accels_t *accels);
 *
 * // 5. Connect/Disconnect contextual accel groups when entering/exiting some view:
 * // NOTE: contextual accel groups may redefine/overwrite some global keys combinations temporarily.
 * void dt_accels_disconnect_active_group(dt_accels_t *accels);
 * void dt_accels_connect_active_group(dt_accels_t *accels, const gchar *group);
 *
 * // 6. Install our own shortcuts listener. This is used within an event handler callback.
 * gboolean dt_accels_dispatch(GtkWidget *w, GdkEvent *event, gpointer user_data);
 * ```
 *
 * Some keys appear reserved to Gtk/System, like Tab or Enter, depending on OS. So we have to use our own, custom,
 * shortcut handler, which is mostly a thin wrapper over Gtk native features.
 *
 * This allows us to decide in which order we process the several sets of shortcuts we maintain
 * (global, lighttable, darkroom). Global shortcuts are processed last, for all views.
 * Lighttable and darkroom shortcuts are processed first, for the relevant view.
 * This lets user have different actions mapped to the same shortcuts, depending on view
 * but also have the view-centric shortcuts overwrite global ones if needed.
 */

#pragma once

typedef struct dt_accels_t
{
  char *config_file;
  GtkAccelGroup *global_accels;     // used, aka init it and free it
  GtkAccelGroup *darkroom_accels;   // darkroom-specific accels
  GtkAccelGroup *lighttable_accels; // lighttable-specific accels

  // reference to the above group currently loaded in the main window. don't init,
  // don't free, only update
  GtkAccelGroup *active_group;

  GHashTable *acceleratables;       // Key/value list of path/dt_shortcut_t
  gint reset;                       // ref counter of how many parts disconnected accels
  GdkKeymap *keymap;                // default screen keymap to decode key values
  GdkModifierType default_mod_mask; // set of modifier masks relevant only to key strokes

  // TRUE if we didn't find a keyboardrc config file at startup and we need to init a new one
  gboolean init;

  // between key_pressed and key_release events, this stores the active key strokes
  GtkAccelKey active_key;

  // Temporarily disable accelerators
  gboolean disable_accels;

  GtkAccelFlags flags;

  // Views can register a global callback to handle scroll events
  // for example while keystrokes are on.
  struct scroll
  {
    gboolean (*callback)(GdkEventScroll event, void *data);
    void *data;
  } scroll;
} dt_accels_t;

typedef enum dt_shortcut_type_t
{
  DT_SHORTCUT_UNSET = 0,   // shortcut non-inited
  DT_SHORTCUT_DEFAULT = 1, // shortcut inited with compile-time defaults
  DT_SHORTCUT_USER = 2     // shortcut changed by user config
} dt_shortcut_type_t;

typedef struct dt_shortcut_t
{
  GtkWidget *widget;          // link to the widget being accelerated. Can be NULL.
  GClosure *closure;          // callback + data being accelerated. Has to be non-NULL if widget is NULL.
  char *path;                 // global path for that accel
  const char *signal;         // widget signal to be wired to that accel
  GtkAccelGroup *accel_group; // the accel_group to which this shortcut belongs
  guint key;                  // default key
  GdkModifierType mods;       // default modifier
  dt_shortcut_type_t type;
  gboolean locked;            // if shortcut can't be changed by user
  gboolean virtual_shortcut;  // if shortcut is mapped to a key-pressed event handler instead of a global action callback
  const char *description;    // user-legible description of the action
  dt_accels_t *accels;        // back-reference for convenience
} dt_shortcut_t;


dt_accels_t *dt_accels_init(char *config_file, GtkAccelFlags flags);
void dt_accels_cleanup(dt_accels_t *accels);


gchar *dt_accels_build_path(const gchar *scope, const gchar *feature);

/**
 * @brief Loads keyboardrc.lang from config dir. This needs to run after we inited the accel map from widgets
 * creation.
 *
 * @param accels
 */
void dt_accels_load_user_config(dt_accels_t *accels);


/**
 * @brief Actually enable accelerators after having loaded user config
 *
 * @param accels
 */
void dt_accels_connect_accels(dt_accels_t *accels);


/**
 * @brief Connect the contextual active accels group to the window.
 * Views can declare their own set of contextual accels, which can
 * override the global accels, in case they use the same keys.
 *
 * @param accels
 * @param group any of the following: "darkroom", "lighttable".
 */
void dt_accels_connect_active_group(dt_accels_t *accels, const gchar *group);

/**
 * @brief Disconnect the contextual active accels group from the window
 *
 * @param accels
 */
void dt_accels_disconnect_active_group(dt_accels_t *accels);


/**
 * @brief Add a new virtual shortcut. Virtual shortcuts are immutable, read-only
 * and don't trigger any action. They are meant to serve as placeholders, in a purely
 * declarative way, for key combinations hardcoded in the key-pressed events handlers
 * of widgets able to capture focus. Once declared here, they will prevent users from declaring
 * their own shortcuts using hardcoded combinations for the corresponding accel_group.
 *
 * @param accel_group
 * @param accel_path
 * @param key_val
 * @param accel_mods
 */
void dt_accels_new_virtual_shortcut(dt_accels_t *accels, GtkAccelGroup *accel_group, const gchar *accel_path,
                                    GtkWidget *widget,
                                    guint key_val, GdkModifierType accel_mods);

/**
 * @brief Register a new shortcut for a widget, setting up its path, default keys and accel group.
 * This does everything but connecting it, so exists only as a defined slot to be connected later.
 *
 * @param accels
 * @param widget
 * @param signal
 * @param accel_group
 * @param accel_path
 * @param key_val
 * @param accel_mods
 * @param lock prevent user edition
 */
void dt_accels_new_widget_shortcut(dt_accels_t *accels, GtkWidget *widget, const gchar *signal,
                                   GtkAccelGroup *accel_group, const gchar *accel_path, guint key_val,
                                   GdkModifierType accel_mods, const gboolean lock);


/**
 * @brief Register a new shortcut for a generic action, setting up its path, default keys and accel group.
 * This does everything but connecting it, so exists only as a defined slot to be connected later.
 *
 * The callback should have the following signature:
 *
 * ```C
 *
 * gboolean action_callback(GtkAccelGroup *accel_group, GObject *accelerable, guint keyval, GdkModifierType
 * modifier, gpointer data)
 *
 * ```
 *
 * @param accels
 * @param data
 * @param accel_group
 * @param action_scope Human-readable, translated, category or scope of the action. Will be turned into path
 * internally
 * @param action_name Human-readable, translated, name or description of the action. Will be turned into path
 * internally
 * @param key_val
 * @param accel_mods
 * @param lock prevent user edition
 */
void dt_accels_new_action_shortcut(dt_accels_t *accels, void(*action_callback), gpointer data,
                                   GtkAccelGroup *accel_group, const gchar *action_scope, const gchar *action_name,
                                   guint key_val, GdkModifierType accel_mods, const gboolean lock, const char *description);


/**
 * @brief Force our listener for all key strokes to bypass reserved Gtk keys.
 *
 * @param w
 * @param event
 * @param user_data
 * @return gboolean
 */
gboolean dt_accels_dispatch(GtkWidget *w, GdkEvent *event, gpointer user_data);

/**
 * @brief Attach a new global scroll event callback. So far this is used in darkroom
 * to redirect scroll events to a Bauhaus widget when the focusing shortcut of that widget
 * is held down on keyboard.
 *
 * @param callback
 * @param data
 */
void dt_accels_attach_scroll_handler(dt_accels_t *accels, gboolean (*callback)(GdkEventScroll event, void *data),
                                     void *data);

void dt_accels_detach_scroll_handler(dt_accels_t *accels);


// Temporarily enable/disable keyboard accels, for example during GtkEntry typing.
// Connect it from Gtk focus in/out event handlers
static inline void dt_accels_disable(dt_accels_t *accels, gboolean state)
{
  accels->disable_accels = state;
}

/**
 * @brief Show the modal dialog listing all available keyboard shortcuts and letting user to set them.
 *
 * @param accels
 * @param main_window The main Ansel application window (for modal/transient)
 */
void dt_accels_window(dt_accels_t *accels, GtkWindow *main_window);
