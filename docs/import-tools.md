# Import Tools
## Overview

The `import_tools` module acts as the definitive data ingestion gateway and multi-spectral parsing core for the FEZrs processing engine. High-resolution commercial sensors (e.g., GeoEye, WorldView) and public science constellations (e.g., Landsat 8/9, Sentinel-2) employ fundamentally different data distribution architectures: commercial payloads typically bundle overlapping spectral channels into unified, multi-band raster arrays, whereas public science programs distribute independent spectral bands as discrete, single-channel raster products.

This module unifies these disparate payload formats, rescaling values to a common $[0.0, 1.0]$ range and assembling structurally synchronized tensor cubes. It transforms raw orbital observations into optimized 2D and 3D spatial grids calibrated for immediate downstream geometric partitioning, advanced feature engineering, or direct visual observation.

```
       [Raw Commercial Imagery]                      [Raw Public Science Imagery]
      Unified Multi-Band GeoTIFF                    Discrete Single-Band Files
                   │                                             │
                   ▼                                             ▼
          Geoeye_Calculator                              Landsat8_Calculator
                   │                                             │
                   ▼                                             ▼
       [Radiometric Normalization]                   [Dynamic Spectral Compositing]
      $I_{\text{norm}} \in [0.0, 1.0]$              $\vec{P}(x, y) = [R, G, B]^T$
                   │                                             │
                   └──────────────────────┬──────────────────────┘
                                          │
                                          ▼
                         [Downstream FEZrs Core Compute]
                          (glcm, hsv, enhancement, etc.)
```

## Comprehensive Class Specifications

### `Geoeye_Calculator` (Unified Multi-Band Tensor Slicing)

#### Scientific and Engineering Objective

`Geoeye_Calculator` ingests high-resolution satellite scenes distributed as a single, multi-band file structure. Its objective is to linearly normalize the entire spatial data cube and extract an isolated, 2D spectral band based on a zero-indexed channel allocation. This provides an unwarped spatial array on a $[0.0, 1.0]$ scale, ready for localized texture profiling or edge-detection filters.

#### Technical Foundation & Mathematical Formulations

##### Linear Radiometric Normalization

The constructor calls `self.files_handler.get_normalized_bands(requested_bands=["tif"])`. This loads the stack as `np.float64` with the values stored in the file (the raw digital numbers $I_{\text{raw}}$), then applies **one** linear min-max stretch to the **entire cube**, all bands together:

$$I_{\text{norm}}(x, y, c) = \frac{I_{\text{raw}}(x, y, c) - \min I_{\text{raw}}}{\max I_{\text{raw}} - \min I_{\text{raw}}}$$

Where $c$ represents the internal channel dimension, and the minimum and maximum are taken over every valid pixel of every band. No sensor bit-depth is read or assumed: the stretch depends only on the values present in the file. Pixels covered by a declared nodata value are NaN, are left out of the minimum and maximum, and stay NaN in the result. If every valid pixel holds the same value, they all become $0.0$.

The cube as a whole therefore spans the normalized floating-point range:

$$I_{\text{norm}} \in [0.0, 1.0] \subset \mathbb{R}$$

A single band generally does not. Only the band holding the cube's minimum reaches $0.0$, and only the band holding its maximum reaches $1.0$. Because every band shares the same stretch, the ordering of the bands at each pixel and the relative size of their differences are preserved, which a separate stretch per band would destroy. The trade-off is that a band's normalized values depend on the other bands in the file: the same band extracted from a different stack can come out on a different scale.

##### Vectorized Tensor Slicing

The normalized data cube is formatted as a 3D tensor with spatial and spectral dimensions:

$$\text{Shape} = (H, W, N_{\text{bands}})$$

This holds whether the file is pixel- or band-interleaved: the loader reads the band count and grid from the raster profile and puts the band axis last. `Geoeye_Calculator` is the only tool that accepts a stacked `tif_path`; the filters require a single band and raise `ValueError` when given a stack.

To isolate a target band without introducing processing overhead or memory allocation leaks, the class applies a zero-indexed slicing operation along the third axis at the specified layer value ($L$):

$$\text{Output Array} = I_{\text{norm}}[:, :, L]$$

This operation isolates a 2D spatial array of dimensions $(H, W)$ that contains the normalized reflectance values measured by the sensor.

#### Interface Architecture

- **Constructor Method (`__init__`) Input Arguments:**
    
    - `tif_path` (`str` | `Path`): Absolute system path pointing to the source multi-band GeoTIFF file.
        
    - `level` (`int`): The zero-based integer index of the target spectral band to extract (e.g., $0$ identifies the first spectral band). Default setting is `0`.
    
- **Operational Validation (`_validate`):** Reads the tensor shape along the spectral dimension to determine $N_{\text{bands}}$. It verifies that the requested band index falls within valid bounds:

$$0 \le L < N_{\text{bands}}$$

If the requested index is out of bounds, the method raises a `ValueError` to prevent invalid memory access or segmentation faults. A single-band raster also raises a `ValueError`, since there is no band axis to select from. Validation runs on both `execute()` and `process()`.

- **Return State (`process()`):** Returns a single-channel 2D `numpy.ndarray` (`np.float64`) of shape $(H, W)$: band $L$ of the normalized cube. Its values lie within $[0.0, 1.0]$ but usually span only part of that range, as described above.

```Python
from pathlib import Path
from fezrs.tools.import_tools import Geoeye_Calculator

# Instantiate the multi-band extraction engine
# Extracting index 3 isolates the Near-Infrared (NIR) channel in standard 4-band payloads
sensor_extractor = Geoeye_Calculator(
    tif_path=Path("./data/geoeye_subscene_4b.tif"),
    level=3
)

# Execute the tensor slicing pipeline
normalized_channel = sensor_extractor.process()
```

### `Landsat8_Calculator` (Discrete Multi-Spectral Composition)

#### Scientific and Engineering Objective

`Landsat8_Calculator` ingests separate single-band raster files and coordinates their spatial alignment into custom multi-band color composites. It maps distinct spectral wavelengths to the standard red, green, and blue ($R, G, B$) display channels, helping human analysts identify specific surface features or preparing consistent data stacks for machine learning workflows.

#### Technical Foundation & Mathematical Formulations

##### Vectorized Layer Stacking

The engine ingests separate spatial bands of identical resolution and uses a vectorized stacking operation to generate a combined multi-channel array. It aligns the three assigned 2D channel arrays ($R(x,y)$, $G(x,y)$, and $B(x,y)$) along the third tensor dimension using NumPy's optimized `np.stack` execution layer:

$$\text{Output Tensor} = \text{np.stack}([R, G, B], \text{axis}=2)$$

This transformation packs the independent 2D arrays into a continuous 3D coordinate mesh of shape $(H, W, 3)$. For every coordinate pair $(x, y)$, the pixel values are represented as a localized multi-spectral vector:

$$\vec{P}(x, y) = \begin{bmatrix} R(x, y) \\ G(x, y) \\ B(x, y) \end{bmatrix}$$

##### Analytical Compositing Modes

The internal mapping architecture supports three compositing profiles via the `exportType` parameter. Any value other than `"rgb"` or `"infrared"` falls back to the raw stack of profile A.

```
                  ┌──────────────────────────────┐
                  │      exportType Match        │
                  └──────────────┬───────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
      [None]                  ["rgb"]               ["infrared"]
  Raw DN Stack          Normalized True-Color   Short-Wave IR Composite
  R: Band 4 (Red)         R: Normalized B4        R: Normalized B7 (SWIR2)
  G: Band 3 (Green)       G: Normalized B3        G: Normalized B6 (SWIR1)
  B: Band 2 (Blue)        B: Normalized B2        B: Normalized B5 (NIR)
```

###### Profile A: Raw Digital Number Stack (`exportType=None`)

Combines the raw visible bands directly without applying any scaling or radiometric correction. The values are the digital numbers ($DN$) stored in the files, typically `uint16`, converted to `np.float64` on load. Pixels covered by a declared nodata value are NaN.

$$\begin{bmatrix} R \\ G \\ B \end{bmatrix} \leftarrow \begin{bmatrix} \text{Band 4 (Red)} \\ \text{Band 3 (Green)} \\ \text{Band 2 (Blue)} \end{bmatrix}$$

###### Profile B: Normalized True-Color (`exportType="rgb"`)

Applies a linear min-max stretch to **each band independently** before compositing:

$$X_{\text{norm}}(x, y) = \frac{X(x, y) - \min X}{\max X - \min X}$$

Here the minimum and maximum are taken over the valid pixels of that band alone, with nodata left out and kept as NaN. Every channel therefore spans the full floating-point range $[0.0, 1.0]$, which maximizes contrast for visual interpretation. The cost is that the relative brightness *between* bands is discarded: a pixel's channel values are not radiometrically comparable to one another, or to the same composite of another scene. Use profile A when those relationships matter.

$$\begin{bmatrix} R \\ G \\ B \end{bmatrix} \leftarrow \begin{bmatrix} \text{Normalized Band 4} \\ \text{Normalized Band 3} \\ \text{Normalized Band 2} \end{bmatrix}$$

###### Profile C: Short-Wave False-Color Infrared (`exportType="infrared"`)

Constructs an advanced infrared composite by shifting invisible short-wave and near-infrared wavelengths into the visible color space. This combination is highly effective for identifying changes in vegetation health, mapping burn scars, and tracking soil moisture anomalies. Each band receives the same independent min-max stretch as in profile B, with the same trade-off.

$$\begin{bmatrix} R \\ G \\ B \end{bmatrix} \leftarrow \begin{bmatrix} \text{Normalized Band 7 (SWIR2)} \\ \text{Normalized Band 6 (SWIR1)} \\ \text{Normalized Band 5 (NIR)} \end{bmatrix}$$

##### Spectral Rationale for Infrared Compositing

The mapping logic used in `"infrared"` mode is specifically tailored to the physical interaction of light with different surface materials:

- **SWIR2 ($\approx 2.2\,\mu\text{m}$) $\to$ Red Channel:** Water molecules strongly absorb short-wave infrared energy, whereas dry soils, exposed rock, and active burn scars reflect it highly. This ensures that dry or fire-damaged surfaces show up clearly in the red spectrum.
    
- **SWIR1 ($\approx 1.6\,\mu\text{m}$) $\to$ Green Channel:** This band is highly sensitive to inner leaf moisture and cellular gaps, tracking structural variations in the landscape.
    
- **NIR ($\approx 0.85\,\mu\text{m}$) $\to$ Blue Channel:** The spongy mesophyll tissue in healthy plant leaves strongly reflects near-infrared light. Consequently, areas with dense, active vegetation stand out vividly as cyan and blue features.

## Analytical Compositing Reference Matrix

The table below summarizes the technical specifications and application profiles for the different compositing modes:

|**Composite Mode Selection**|**Internal Band Routing Diagram**|**Data Type**|**Radiometric Scale Bounds**|**Primary Remote Sensing Use Cases**|
|---|---|---|---|---|
|**`None`** (Raw Stack)|$R \leftarrow B4$<br><br>  <br><br>$G \leftarrow B3$<br><br>  <br><br>$B \leftarrow B2$|`float64`|As stored, e.g. $[0, \,\, 65535]$ for `uint16` input|Preserves raw unscaled calibration data for basic archive ingestion.|
|**`"rgb"`** (True Color)|$R \leftarrow B4_{\text{norm}}$<br><br>  <br><br>$G \leftarrow B3_{\text{norm}}$<br><br>  <br><br>$B \leftarrow B2_{\text{norm}}$|`float64`|$[0.0, \,\, 1.0]$ per channel|General visual mapping, urban planning, and baseline land-surface profiling.|
|**`"infrared"`** (False Color)|$R \leftarrow B7_{\text{norm}}$<br><br>  <br><br>$G \leftarrow B6_{\text{norm}}$<br><br>  <br><br>$B \leftarrow B5_{\text{norm}}$|`float64`|$[0.0, \,\, 1.0]$ per channel|Mapping burn boundaries, tracking forest fire severity, and monitoring agricultural crop stress.|
