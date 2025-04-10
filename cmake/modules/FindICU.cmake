#
# Find the ICU includes and library
#

# This module defines
# ICU_INCLUDE_DIRS, where to find *.h, etc
# ICU_LIBRARIES, the libraries
# ICU_FOUND, If false, do not try to use ICU.

include(LibFindMacros)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(ICU_PKGCONF icu-i18n icu-uc)

# Find include dir
find_path(ICU_INCLUDE_DIR
  NAMES unicode/utypes.h
  HINTS ${IUC_PKGCONF_INCLUDE_DIRS}
)

# Find the libraries
foreach(i ${ICU_PKGCONF_LIBRARIES})
  find_library(_ICU_LIBRARY NAMES ${i} HINTS ${ICU_PKGCONF_LIBRARY_DIRS})
  LIST(APPEND ICU_LIBRARY ${_ICU_LIBRARY})
  unset(_ICU_LIBRARY CACHE)
endforeach(i)

if(ICU_LIBRARY AND ICU_INCLUDE_DIR)
  set(ICU_PROCESS_INCLUDES ${ICU_INCLUDE_DIR})
  set(ICU_PROCESS_LIBS ${ICU_LIBRARY})
  libfind_process(ICU)
endif(ICU_LIBRARY AND ICU_INCLUDE_DIR)

if(ICU_FOUND)
  set(ICU_INCLUDE_DIRS ${ICU_INCLUDE_DIR})
  set(ICU_LIBRARIES ${ICU_LIBRARY})
else()
  message(STATUS "ICU not found")
endif(ICU_FOUND)
