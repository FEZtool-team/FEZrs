# Mosaic
## Overview

The `mosaic` module provides scalable, coordinate-aligned image-merging architectures designed to consolidate multiple discrete satellite imagery tiles into a single georeferenced master mosaic. In regional or country-scale remote sensing, a single satellite pass rarely covers the entire target study area. Analyses instead require combining multiple overlapping image swaths or structural tiles.

This module resolves these spatial divisions by calculating structural bounding extents, establishing unified coordinate spaces, and mapping independent pixel grids into a seamless, unified mosaic tensor. The resulting file preserves full metadata transparency, enabling immediate downstream feature engineering across the expanded study area.

```
                  [Input Raster Files Path List]
                  (e.g., tile_01.tif, tile_02.tif)
                                 │
                                 ▼
                     ┌──────────────────────┐
                     │   MosaicCalculator   │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌──────────────────────┐
                     │  rasterio.merge Core │
                     └───────────┬───────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
 [Spatial Extent Union]                        [Grid Overlap Resolution]
 Calculates absolute geographic bounding box     Evaluates pixel decision rule:
 via coordinate transform tracking.             $output(x, y) = I_{k^*}(x, y)$
                                 │                               │
                                 └───────────────┬───────────────┘
                                                 │
                                                 ▼
                                     ┌───────────────────────┐
                                     │ File Output Lifecycle │
                                     └───────────┬───────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
             [Fully Georeferenced GeoTIFF]                        [Visual PNG Preview]
             Saved to `self._output` for chaining                Auto-named via `Mosaic_` + UUID
```

## Comprehensive Class Specification: `MosaicCalculator`

### Scientific and Mathematical Objective

The objective of `MosaicCalculator` is to combine multiple georeferenced raster layers that share a common Coordinate Reference System (CRS) into a single, continuous, spatial data array. This pipeline standardizes structural pixel boundaries across different satellite tracks and scene footprints, eliminating artificial layout boundaries across large-scale geographic regions.

### Algorithmic Processing Mechanics and Coordinate Transformations

Mosaicking multi-spectral imagery requires matching different pixel spaces to an identical spatial grid. The module uses `rasterio.merge.merge` to perform these non-linear coordinate and tensor alignments.

#### Affine Georeferencing Framework

Every input GeoTIFF file includes a 2D affine transformation matrix that maps internal pixel coordinates (expressed as a row $r$ and column $c$) to continuous real-world map projection coordinates (such as UTM easting $x$ and northing $y$):

$$\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} a & b & c \\ d & e & f \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} c \\ r \\ 1 \end{bmatrix}$$

Where:

- $a = \Delta x = \text{pixel width}$ (spatial resolution in the horizontal direction).
    
- $e = \Delta y = \text{pixel height}$ (typically negative for standard north-up geographic configurations, where $e < 0$).
    
- $b, d = \text{rotation and shear coefficients}$ (evaluate to exactly $0.0$ for traditional north-up alignment).
    
- $c = x_{\text{origin}}$ (absolute spatial $x$-coordinate matching the center of the upper-left pixel).
    
- $f = y_{\text{origin}}$ (absolute spatial $y$-coordinate matching the center of the upper-left pixel).

This matrix lets the transformation engine calculate the exact geographic footprint of every pixel in each input image.

#### Determining Output Bounds and Pixel Grids

The spatial boundaries of the final mosaic are computed by calculating the geographic union of all input image extents. For a collection of input images where each image $I_k$ defines a bounding box $(x_{\min}^k, y_{\min}^k, x_{\max}^k, y_{\max}^k)$, the global bounding parameters for the output mosaic are calculated as:

$$x_{\text{min}}^{\text{out}} = \min_k\left(x_{\text{min}}^k\right), \quad y_{\text{min}}^{\text{out}} = \min_k\left(y_{\text{min}}^k\right)$$

$$x_{\text{max}}^{\text{out}} = \max_k\left(x_{\text{max}}^k\right), \quad y_{\text{max}}^{\text{out}} = \max_k\left(y_{\text{max}}^k\right)$$

By default, the pixel resolution and orientation parameters are copied from the first image in the input file list ($I_1$). The output affine transform matrix is then constructed using these baseline parameters:

$$a^{\text{out}} = a^1, \quad e^{\text{out}} = e^1, \quad b^{\text{out}} = b^1, \quad d^{\text{out}} = d^1$$

$$c^{\text{out}} = x_{\text{min}}^{\text{out}} + \frac{a^{\text{out}}}{2}, \quad f^{\text{out}} = y_{\text{max}}^{\text{out}} - \frac{|e^{\text{out}}|}{2}$$

This calculation handles upper-left grid corner positioning in compliance with standard GDAL geographic conventions. The final grid dimensions of the output raster (Width $W^{\text{out}}$, Height $H^{\text{out}}$) are calculated as:

$$W^{\text{out}} = \left| \frac{x_{\text{max}}^{\text{out}} - x_{\text{min}}^{\text{out}}}{a^{\text{out}}} \right|, \quad H^{\text{out}} = \left| \frac{y_{\text{max}}^{\text{out}} - y_{\text{min}}^{\text{out}}}{e^{\text{out}}} \right|$$

#### Mapping Pixels onto the Output Grid

For each coordinate cell $(c^{\text{out}}, r^{\text{out}})$ in the destination array, the merging engine projects the pixel's location back into the coordinate space of the source input image. This inverse projection applies the input image's inverse affine matrix ($\text{Affine}_{\text{in}}^{-1}$) to the output geographic transform coordinates:

$$\begin{bmatrix} c_{\text{in}} \\ r_{\text{in}} \end{bmatrix} = \text{Affine}_{\text{in}}^{-1} \left( \text{Affine}_{\text{out}} \begin{bmatrix} c_{\text{out}} \\ r_{\text{out}} \\ 1 \end{bmatrix} \right)$$

If the calculated source coordinate $(c_{\text{in}}, r_{\text{in}})$ falls within the valid boundaries of the input image, the engine samples the corresponding pixel value (using nearest-neighbor interpolation by default) and writes it to the destination array.

#### Handling Overlapping Regions

When multiple input images overlap the same geographic area, the module applies a strict priority rule to determine the final pixel value. Because the core `merge` execution statement does not specify an alternative composition method, it defaults to the **"first"** strategy.

```
                          Overlap Region Profile
                    ┌──────────────────────────────┐
                    │ Image Tile 1 (Priority Index)│
                    │   ┌──────────────────────────┼──────────────┐
                    │   │  Resolved Output Pixel   │              │
                    │   │  Matches Tile 1 Value    │              │
                    └───┼──────────────────────────┘              │
                        │ Image Tile 2 (Ignored Overlap Overrides)│
                        └─────────────────────────────────────────┘
```

The engine evaluates input files in the order they appear in the provided file list. For each output pixel location, it selects the value from the first image in the list that contains valid data (non-nodata values) at those coordinates:

$$\text{Output}(x, y) = I_{k^*}(x, y) \quad \text{where } k^* = \min\left\{k \mid I_k \text{ is valid at } (x, y)\right\}$$

Any data from subsequent overlapping layers is ignored at that specific location.

#### Modifying Metadata Configurations

Once pixel value mapping is complete, the engine updates the metadata dictionary to reflect the properties of the new dataset:

- `driver`: Enforces `"GTiff"` (GeoTIFF Specification Standard).
    
- `height`, `width`: Set to the calculated global grid dimensions ($H^{\text{out}}, W^{\text{out}}$).
    
- `transform`: Updates to the newly calculated output affine transform matrix.
    
- `crs`: Copied from the first input image in the file list. All input datasets should share this identical CRS to prevent reprojection errors or alignment offsets during processing.

### Interface Specifications

#### Constructor Method Input Arguments (`__init__`)

- `tif_paths` (`list[str | Path]`): A list of absolute file paths to the source GeoTIFF files to be combined.

#### Core Processing Interface (`process()`)

- Executes the `rasterio.merge.merge` function across the input file list.
    
- Extracts and builds the updated spatial metadata dictionary.
    
- Stores the final combined multi-band pixel array in `self.mosaic_mimg` and its metadata configurations in `self.mosaic_meta`.

#### Custom File Serialization Method (`_export_file`)

This module overrides the default file export lifecycle. Instead of exporting only a standard display image, it handles two separate outputs:

1. **Fully Georeferenced GeoTIFF:** Serializes the complete multi-spectral array to disk, updating its internal header to include the new coordinate reference system and transform parameters.
    
2. **Visual PNG Preview:** Generates a lightweight browse image to support rapid quality control inspections.

Both output files are automatically named using the class prefix `Mosaic_` combined with a unique Universally Unique Identifier (`UUID`).

#### Return State (`execute()`)

Returns a `Path` object pointing to the newly written GeoTIFF file location on disk. This file path is stored directly in `self._output`, allowing it to be passed directly to downstream calculators for chained processing.

### Operational Implementation

```Python
from pathlib import Path
from fezrs.tools.mosaic import MosaicCalculator

# Define the list of adjacent, overlapping input imagery tiles
target_tiles = [
    Path("./tiles/UTM32N_Scene_01.tif"),
    Path("./tiles/UTM32N_Scene_02.tif"),
    Path("./tiles/UTM32N_Scene_03.tif")
]

# Initialize the spatial merging engine
mosaic_engine = MosaicCalculator(tif_paths=target_tiles)

# Run the mosaicking pipeline and serialize outputs
georeferenced_mosaic_path = mosaic_engine.execute(
    output_path="./exports/regional_mosaics/"
)

print(f"Operational master mosaic successfully saved at: {georeferenced_mosaic_path}")
```

## Operational Reference: Advanced Overlap Resolution Methods

While the calculator defaults to the `"first"` pixel selection strategy, alternative merging methods can be passed directly to the underlying `rasterio.merge.merge` library to handle overlapping areas. The table below profiles these alternative strategies:

| **Strategy**              | **Functional Behavior Profile**                                                                              | **Use Cases**                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| **`"first"`** _(Default)_ | Retains the pixel value from the first image in the file list that contains valid data at those coordinates. | Standard data combination when tiles share a consistent calibration profile. |
| **`"last"`**              | Overwrites previous layers, using the pixel value from the last valid image in the file list.                | Updating an existing base map with newer imagery data.                       |
| **`"min"`**               | Evaluates overlapping pixels across all layers and keeps the lowest value.                                   | Minimizing transient bright anomalies like cloud cover or glint.             |
| **`"max"`**               | Evaluates overlapping pixels across all layers and keeps the highest value.                                  | Highlighting maximum surface extents or urban thermal signatures.            |
| **`"mean"`**              | Computes the mathematical average value across all valid overlapping pixels.                                 | Smoothing out temporal differences and reducing random sensor noise.         |
