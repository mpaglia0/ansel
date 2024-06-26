include_directories(${CMAKE_CURRENT_BINARY_DIR}/../..)

include_directories(SYSTEM "${PNG_PNG_INCLUDE_DIR}")

include(manage-symbol-visibility)

add_definitions(-include common/module_api.h)
add_definitions(-include imageio/format/imageio_format_api.h)

set(MODULES copy jpeg pdf png ppm pfm tiff)

add_library(copy MODULE "copy.c")
add_library(jpeg MODULE "jpeg.c")
add_library(pdf MODULE "pdf.c")
add_library(png MODULE "png.c")
add_library(ppm MODULE "ppm.c")
add_library(pfm MODULE "pfm.c")
add_library(tiff MODULE "tiff.c")

if(WebP_FOUND)
  list(APPEND MODULES "webp")
  add_library(webp MODULE "webp.c")
endif(WebP_FOUND)

if(OpenEXR_FOUND)
  list(APPEND MODULES "exr")
  add_library(exr MODULE "exr.cc")
endif(OpenEXR_FOUND)

if(OpenJPEG_FOUND)
  list(APPEND MODULES "j2k")
  add_library(j2k MODULE "j2k.c")
endif(OpenJPEG_FOUND)

if(libavif_FOUND)
  list(APPEND MODULES "avif_format")
  add_library(avif_format MODULE "avif.c")
  set_target_properties(avif_format PROPERTIES OUTPUT_NAME avif)
endif()

if(TARGET xcf)
  list(APPEND MODULES "gimp_xcf")
  add_library(gimp_xcf MODULE "xcf.c")
  target_link_libraries(gimp_xcf PRIVATE xcf)
  set_target_properties(gimp_xcf PROPERTIES OUTPUT_NAME xcf)
endif()

foreach(module ${MODULES})
  target_link_libraries(${module} PUBLIC lib_ansel)
  if(WIN32)
    # _detach_debuginfo(${module} ${CMAKE_INSTALL_LIBDIR}/ansel/plugins/imageio/format)
  else()
    set_target_properties(${module}
                          PROPERTIES
                          INSTALL_RPATH ${RPATH_ORIGIN}/../../../..)
    endif()
    install(TARGETS ${module} DESTINATION ${CMAKE_INSTALL_LIBDIR}/ansel/plugins/imageio/format COMPONENT DTApplication)
endforeach(module)
