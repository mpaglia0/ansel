# Picture resizing and scaling

The pipeline has 2 ways of defining the final size of the output buffer :

- from input to output, by chaining the calls to the `modify_roi_out()` method of each module, through the higher-level method `dt_dev_pixelpipe_get_roi_out()`,
- from output to input, by chaining the calls to the `modify_roi_in()` method of each module, through the higher-level method `dt_dev_pixelpipe_get_roi_in()`.

Export, thumbnail and darkroom main preview use the output to input, because the constraint lies on the output. The darkroom thumbnail preview, uses the input to output. The modules automatically affecting sizes are :

- `iop/rawprepare.c` : trim the sensor edges, as defined by rawspeed,
- `iop/demosaic.c` : performs early rescaling to final size for thumbnail pipeline only,
- `iop/lens.cc` : crops to fit the rectangular frame when heavy distortions are corrected,
- `iop/finalsize.c` : performs late rescaling to final size for export pipeline only (so early pipe stages run at full resolution).

Other modules affect size in more unpredictable ways :
- `iop/crop.c`/`iop/ashift.c`/`iop/clipping.c` : user-defined cropping and perspective correction,
- `iop/borders.c` : add a frame.

Note that the darkroom main preview processes an image restricted to the visible area (ROI, or _Region of Interest_). The full RAW image is therefore clipped and zoomed at the step 0 of the pipeline, directly when fetching the base buffer (aka out of any module).

Now, because `iop/demosaic.c` rescales to final size only for the thumbnail pipe, which defines sizes from output to input, its `modify_roi_in()` implements this rescaling, but its `modify_roi_out()` method is essentially a no-operation. Similarly, because `iop/finalscale.c` is used only for export pipelines, defining sizes from output to input, it has a `modify_roi_in()` but no `modify_roi_out()` at all.

For this reason, we always call `dt_dev_pixelpipe_get_roi_in()` to get the in/out sizes from the end, before computing the pipeline global hashes, which use both input and output sizes for each module (plus the `(x, y)` coordinates of the top-left corner of the _Region Of Interest_). `dt_dev_pixelpipe_get_roi_out()` is only used on the full-resolution input to compute the final image ratio of the full-resolution output.

However, since we apply real scaling coefficients on integers pixel dimensions, rounding errors are bound to happen and it is an ordeal to ensure that modules having both `modify_roi_in()` and `modify_roi_out()` methods are pixel-accurate when doing a roundtrip (for the ones even designed to roundtrip). Pixel-accuracy in this context is important because the input/output buffer sizes are both used in the hashes identifying module cache lines. This is what allows the thumbnail pipeline to reuse the cache from the darkroom preview, and they don't compute their sizes in the same direction.

Some care has been taken to try and correct pixel rounding errors, in the ROI-modifying methods of modules using them. There is no guaranty that these methods work all the time and they will most likely be a burden to maintain. In an ideal world, `iop/demosaic.c` would do just that, and early downsampling would have a dedicated method, or even be done at the base buffer initialisation, like in the darkroom main preview.
