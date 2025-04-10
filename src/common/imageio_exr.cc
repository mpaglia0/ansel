/*
    This file is part of ansel,
    Copyright (C) 2010-2020 darktable developers.
    Copyright (C) 2023 ansel developers.

    ansel is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ansel is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ansel.  If not, see <http://www.gnu.org/licenses/>.
*/

#define __STDC_FORMAT_MACROS

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <assert.h>
#include <inttypes.h>
#include <memory>
#include <stdio.h>
#include <string.h>

#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfStandardAttributes.h>
#include <OpenEXR/ImfTestFile.h>
#include <OpenEXR/ImfThreading.h>
#include <OpenEXR/ImfTiledInputFile.h>

#include "glib.h"

#include "common/colorspaces.h"
#include "common/darktable.h"
#include "common/exif.h"
#include "common/imageio.h"
#include "common/imageio_exr.h"
#include "control/conf.h"
#include "develop/develop.h"

#include "common/imageio_exr.hh"

dt_imageio_retval_t dt_imageio_open_exr(dt_image_t *img, const char *filename, dt_mipmap_buffer_t *mbuf)
{
  bool isTiled = false;

  Imf::setGlobalThreadCount(darktable.num_openmp_threads);

  std::unique_ptr<Imf::TiledInputFile> fileTiled;
  std::unique_ptr<Imf::InputFile> file;

  Imath::Box2i dw;
  Imf::FrameBuffer frameBuffer;
  uint32_t xstride, ystride;


  /* verify openexr image */
  if(!Imf::isOpenExrFile((const char *)filename, isTiled)) return DT_IMAGEIO_FILE_CORRUPTED;

  /* open exr file */
  try
  {
    if(isTiled)
    {
      std::unique_ptr<Imf::TiledInputFile> temp(new Imf::TiledInputFile(filename));
      fileTiled = std::move(temp);
    }
    else
    {
      std::unique_ptr<Imf::InputFile> temp(new Imf::InputFile(filename));
      file = std::move(temp);
    }
  }
  catch(const std::exception &e)
  {
    return DT_IMAGEIO_FILE_CORRUPTED;
  }

  const Imf::Header &header = isTiled ? fileTiled->header() : file->header();

  /* check that channels available is any of supported RGB(a) */
  bool hasR = false, hasG = false, hasB = false;
  for(Imf::ChannelList::ConstIterator i = header.channels().begin(); i != header.channels().end(); ++i)
  {
    std::string name(i.name());
    if(name == "R") hasR = true;
    if(name == "G") hasG = true;
    if(name == "B") hasB = true;
  }
  if(!(hasR && hasG && hasB))
  {
    fprintf(stderr, "[exr_read] Warning, only files with RGB(A) channels are supported.\n");
    return DT_IMAGEIO_FILE_CORRUPTED;
  }

  if(!img->exif_inited)
  {
    // read back exif data
    // if another software is able to update these exif data, the former test
    // should be removed to take the potential changes in account (not done
    // by normal import image flow)
    const Imf::BlobAttribute *exif = header.findTypedAttribute<Imf::BlobAttribute>("exif");
    // we append a jpg-compatible exif00 string, so get rid of that again:
    if(exif && exif->value().size > 6)
      dt_exif_read_from_blob(img, ((uint8_t *)(exif->value().data.get())) + 6, exif->value().size - 6);
  }

  /* Get image width and height from displayWindow */
  dw = header.displayWindow();
  img->width = dw.max.x - dw.min.x + 1;
  img->height = dw.max.y - dw.min.y + 1;

  // Try to allocate image data
  img->buf_dsc.channels = 4;
  img->buf_dsc.datatype = TYPE_FLOAT;
  float *buf = (float *)dt_mipmap_cache_alloc(mbuf, img);
  if(!buf)
  {
    fprintf(stderr, "[exr_read] could not alloc full buffer for image `%s'\n", img->filename);
    /// \todo open exr cleanup...
    return DT_IMAGEIO_CACHE_FULL;
  }

  // FIXME: is this really needed?
  memset(buf, 0, sizeof(float) * 4 * img->width * img->height);

  /* setup framebuffer */
  xstride = sizeof(float) * 4;
  ystride = sizeof(float) * img->width * 4;
  frameBuffer.insert("R", Imf::Slice(Imf::FLOAT, (char *)(buf + 0), xstride, ystride, 1, 1, 0.0));
  frameBuffer.insert("G", Imf::Slice(Imf::FLOAT, (char *)(buf + 1), xstride, ystride, 1, 1, 0.0));
  frameBuffer.insert("B", Imf::Slice(Imf::FLOAT, (char *)(buf + 2), xstride, ystride, 1, 1, 0.0));
  frameBuffer.insert("A", Imf::Slice(Imf::FLOAT, (char *)(buf + 3), xstride, ystride, 1, 1, 0.0));

  if(isTiled)
  {
    fileTiled->setFrameBuffer(frameBuffer);
    fileTiled->readTiles(0, fileTiled->numXTiles() - 1, 0, fileTiled->numYTiles() - 1);
  }
  else
  {
    /* read pixels from dataWindow */
    dw = header.dataWindow();
    file->setFrameBuffer(frameBuffer);
    file->readPixels(dw.min.y, dw.max.y);
  }

  /* try to get the chromaticities and whitepoint. this will add the default linear rec709 profile when nothing
   * was embedded and look as if it was embedded in colorin. better than defaulting to something wrong there. */
  Imf::Chromaticities chromaticities;
  float whiteLuminance = 1.0;

  if(Imf::hasChromaticities(header))
  {
    chromaticities = Imf::chromaticities(header);

    /* adapt chromaticities to D65 expected by colorin */
    cmsCIExyY red_xy = { chromaticities.red[0], chromaticities.red[1], 1.0 };
    cmsCIEXYZ srcRed;
    cmsxyY2XYZ(&srcRed, &red_xy);

    cmsCIExyY green_xy = { chromaticities.green[0], chromaticities.green[1], 1.0 };
    cmsCIEXYZ srcGreen;
    cmsxyY2XYZ(&srcGreen, &green_xy);

    cmsCIExyY blue_xy = { chromaticities.blue[0], chromaticities.blue[1], 1.0 };
    cmsCIEXYZ srcBlue;
    cmsxyY2XYZ(&srcBlue, &blue_xy);

    const cmsCIExyY srcWhite_xy = { chromaticities.white[0], chromaticities.white[1], 1.0 };
    cmsCIEXYZ srcWhite;
    cmsxyY2XYZ(&srcWhite, &srcWhite_xy);

    /* use Imf::Chromaticities definition */
    const cmsCIExyY d65_xy = {0.3127f, 0.3290f, 1.0};
    cmsCIEXYZ d65;
    cmsxyY2XYZ(&d65, &d65_xy);

    cmsCIEXYZ dstRed;
    cmsAdaptToIlluminant(&dstRed, &srcWhite, &d65, &srcRed);

    cmsCIEXYZ dstGreen;
    cmsAdaptToIlluminant(&dstGreen, &srcWhite, &d65, &srcGreen);

    cmsCIEXYZ dstBlue;
    cmsAdaptToIlluminant(&dstBlue, &srcWhite, &d65, &srcBlue);

    cmsXYZ2xyY(&red_xy, &dstRed);
    chromaticities.red[0] = (float)red_xy.x;
    chromaticities.red[1] = (float)red_xy.y;

    cmsXYZ2xyY(&green_xy, &dstGreen);
    chromaticities.green[0] = (float)green_xy.x;
    chromaticities.green[1] = (float)green_xy.y;

    cmsXYZ2xyY(&blue_xy, &dstBlue);
    chromaticities.blue[0] = (float)blue_xy.x;
    chromaticities.blue[1] = (float)blue_xy.y;

    chromaticities.white[0] = 0.3127f;
    chromaticities.white[1] = 0.3290f;
  }

  if(Imf::hasWhiteLuminance(header))
    whiteLuminance = Imf::whiteLuminance(header);

//   printf("hasChromaticities: %d\n", Imf::hasChromaticities(header));
//   printf("hasWhiteLuminance: %d\n", Imf::hasWhiteLuminance(header));
//   std::cout << chromaticities.red << std::endl;
//   std::cout << chromaticities.green << std::endl;
//   std::cout << chromaticities.blue << std::endl;
//   std::cout << chromaticities.white << std::endl;

  Imath::M44f m = Imf::XYZtoRGB(chromaticities, whiteLuminance);

  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
    {
      img->d65_color_matrix[3 * i + j] = m[j][i];
    }


  img->buf_dsc.cst = IOP_CS_RGB;
  img->buf_dsc.filters = 0u;
  img->flags &= ~DT_IMAGE_RAW;
  img->flags &= ~DT_IMAGE_S_RAW;
  img->flags &= ~DT_IMAGE_LDR;
  img->flags |= DT_IMAGE_HDR;
  img->loader = LOADER_EXR;

  return DT_IMAGEIO_OK;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
