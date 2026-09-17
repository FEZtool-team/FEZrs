# Image Enhancement
## Overview

The `enhancement` module delivers a comprehensive suite of linear and non-linear radiometric transformation pipelines designed to optimize the visual quality, structural contrast, and dynamic range of satellite imagery. These processing steps adjust the distribution of digital numbers ($DN$) across the available bit-depth spectrum, maximizing feature interpretability for human analysts or standardizing input spaces for downstream machine learning and deep learning computer vision architectures.

```
                    fezrs.base.BaseTool        HistogramExportMixin
                            │                           │
                            └─────────────┬─────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    fezrs.tools.image_enhancement Module     │
                      └───────────────────┬───────────────────┘
                                          │
         ┌────────────────────────────────┴────────────────────────────────┐
         ▼                                                                 ▼
 [Single-Band Processing Matrix]                               [Multi-Band RGB Processing Stacks]
 ├─ OriginalCalculator / FloatCalculator                       ├─ OriginalRGBCalculator
 ├─ EqualizeCalculator                                         ├─ EqualizeRGBCalculator
 ├─ AdaptiveCalculator (CLAHE)                                 ├─ AdaptiveRGBCalculator
 ├─ GammaCalculator                                            └─ GammaRGBCalculator
 ├─ LogAdjustCalculator                                        
 └─ SigmoidAdjustCalculator                                    
```

## Global Radiometric Range Standardization

Before applying non-linear contrast alterations, input matrices must be converted from raw integer digital numbers ($DN$) into a normalized, unitless floating-point domain.

### Mathematical Quantization Normalization

Given an input integer raster channel $I_{\text{int}}$ exhibiting an arbitrary hardware bit-depth $b$ (e.g., 8-bit, 12-bit, or 16-bit engineering configurations), the linear standardization mapping resolves to:

$$I_{\text{float}}(x, y) = \frac{I_{\text{int}}(x, y)}{2^b - 1}$$

This conversion maps raw raster matrices directly into a normalized floating-point range:

$$I_{\text{float}} \in [0.0, 1.0] \subset \mathbb{R}$$

This transformation preserves the exact profile shape of the original data distribution while protecting downstream calculations from numerical overflow or underflow vulnerabilities.

### Core Data Staging Interfaces

#### `FloatCalculator`

- **Programmatic Purpose:** Converts a single-band integer array to a normalized float format ($[0.0, 1.0]$) of type `np.float64` to establish a clean processing baseline.

#### `OriginalCalculator` & `OriginalRGBCalculator`

- **Programmatic Purpose:** Serve as unchanged reference baselines to evaluate the radiometric impact of active enhancement pipelines. For multi-spectral configurations, `OriginalRGBCalculator` ingests individual visible channels and combines them along the third tensor dimension to yield a standard multi-band array layout:

$$\text{Shape}_{\text{RGB}} = (H, W, 3)$$

## Comprehensive Class Specifications

### `EqualizeCalculator` & `EqualizeRGBCalculator` (Global Histogram Equalization)

#### Scientific and Physical Objective

Global histogram equalization maximizes global contrast across low-variance datasets (e.g., scenes flattened by heavy atmospheric haze or poor sensor exposure). It spreads out the most frequent intensity values, remapping the raw data profile into an idealized uniform probability distribution.

#### Theoretical Foundation & Mathematical Formulations

The mathematical transformation uses the continuous **Cumulative Distribution Function (CDF)** of the target image's grayscale values.

Let $p(r_k)$ represent the normalized probability of an intensity level $r_k$ occurring within a discrete domain containing $L$ discrete levels (where $L=256$ for standard 8-bit distributions):

$$p(r_k) = \frac{n_k}{N}$$

Where $n_k$ is the absolute count of pixels expressing intensity $r_k$, and $N$ represents the global pixel count. The localized discrete cumulative probability function $C(r_k)$ maps as:

$$C(r_k) = \sum_{i=0}^{k} p(r_i)$$

The equalized intensity value $s_k$ is scaled directly to the normalized interval $[0.0, 1.0]$ via:

$$s_k = C(r_k)$$

In the underlying vectorized implementation handled by `skimage.exposure.equalize_hist`, this process evaluates sorted absolute ranks, replacing raw values with their normalized cumulative frequency:

$$I_{\text{eq}}(x, y) = \frac{1}{N} \sum_{j=1}^{\text{rank}(I(x, y))} \text{count}(j)$$

#### Multi-Band Radiometric Behavior

`EqualizeRGBCalculator` runs this transformation across each channel independently. While this maximizes structural detail within each channel, it can distort the relative balance between channels, often causing unintended **chromatic color-shifting** artifacts across natural-color composites.

#### Interface Architecture

- **Constructor Inputs:**
    
    - `EqualizeCalculator`: `nir_path` (`str` | `Path`)
        
    - `EqualizeRGBCalculator`: `red_path`, `green_path`, `blue_path` (`str` | `Path`)
    
- **Return State:** Returns a transformed 2D or 3D `numpy.ndarray` (`np.float64`) mapped to a uniform output distribution.

```Python
from pathlib import Path
from fezrs.tools.image_enhancement import EqualizeCalculator

# Initialize global histogram equalization pipeline
equalizer = EqualizeCalculator(nir_path=Path("./data/Hazy_NIR.tif"))
equalizer.execute(output_path="./exports/enhanced/", title="Global Histogram Equalized")
```

### `AdaptiveCalculator` & `AdaptiveRGBCalculator` (CLAHE)

#### Scientific and Physical Objective

**Contrast Limited Adaptive Histogram Equalisation (CLAHE)** enhances local contrast within localized window tiles rather than modifying the global image histogram. This approach reveals fine structural details in both shadow and highlight regions simultaneously without amplifying background sensor noise.

#### Theoretical Foundation & Mathematical Formulations

The processing pipeline divides the 2D image matrix into a grid of non-overlapping contextual regions called **tiles** (typically matching an $8 \times 8$ pixel block configuration).

For an arbitrary tile containing a spatial area of $M \times M$ pixels across $N_{\text{bins}} = 256$, the baseline uniform bin height evaluates to:

$$\bar{n} = \frac{M^2}{N_{\text{bins}}}$$

To prevent the amplification of high-frequency background noise in uniform areas, a user-defined clipping constraint ($C$) limits the height of any single bin:

$$\text{Actual Clip Height} = C \cdot \bar{n}$$

Any structural data points that exceed this clipping threshold are clipped and redistributed evenly across all available histogram bins. The system then computes a localized cumulative distribution function ($T_t$) for that specific tile ($t$):

$$T_t(r_k) = \frac{1}{N_t} \sum_{i=0}^{k} h_t(i)$$

Where $h_t$ is the clipped local histogram profile, and $N_t$ represents the total number of pixels in the tile following redistribution.

To eliminate blocking artifacts along tile boundaries, the final value for any pixel coordinate $(x, y)$ is computed using bilinear interpolation from the mapping functions ($T_{t_1}, T_{t_2}, T_{t_3}, T_{t_4}$) of the four nearest neighboring tiles:

$$I_{\text{CLAHE}}(x, y) = \text{BilinearInterpolate}\left(T_{t_1}, T_{t_2}, T_{t_3}, T_{t_4}\right)$$

#### Interface Architecture

- **Constructor Arguments (`AdaptiveCalculator`):**
    
    - `nir_path` (`str` | `Path`): Target raster file location.
        
    - `clip_limit` (`float`): Local contrast threshold limiter. Bounded between $[0.01, 0.1]$.
    
- **Constructor Arguments (`AdaptiveRGBCalculator`):**
    
    - Ingests individual `red_path`, `green_path`, and `blue_path` layers while enforcing a hardcoded clip constraint of $0.08$.

```Python
from pathlib import Path
from fezrs.tools.image_enhancement import AdaptiveCalculator

# Execute local adaptive contrast optimization via CLAHE
clahe_engine = AdaptiveCalculator(
    nir_path=Path("./data/Variable_Lighting_NIR.tif"),
    clip_limit=0.03
)
clahe_engine.execute(output_path="./exports/enhanced/", title="CLAHE Local Optimization")
```

### `GammaCalculator` & `GammaRGBCalculator` (Power-Law / Gamma Correction)

#### Scientific and Physical Objective

Gamma correction applies a non-linear power-law transformation to adjust image brightness. It can brighten shadow regions ($\gamma < 1.0$) or compress high-reflectance highlights ($\gamma > 1.0$) while anchoring the absolute dark and light endpoints ($0.0$ and $1.0$).

#### Theoretical Foundation & Mathematical Formulations

The transformation scales pixel values using a non-linear exponential function:

$$I_{\text{out}} = A \cdot I_{\text{in}}^{\gamma}$$

Where:

- $I_{\text{in}}$ represents the standardized input floating-point value in the range $[0.0, 1.0]$.
    
- $A$ is an optional linear scaling factor (configured via `gain`, defaulting to $1.0$).
    
- $\gamma$ is the exponent that defines the shape of the correction curve.

#### Algorithmic Dynamics

- **$\gamma = 1.0$:** Maps as a strict linear identity function, leaving the data unchanged.
    
- **$0.0 < \gamma < 1.0$:** Bends the response curve upward. This lifts mid-tones and stretches shadow details, making dark regions brighter while compressing highlights.
    
- **$\gamma > 1.0$:** Bends the curve downward. This darkens mid-tones, stretching highlight details while compressing shadows.

#### Interface Architecture

- **Constructor Arguments (`GammaCalculator`):**
    
    - `nir_path` (`str` | `Path`)
        
    - `gamma` (`float`): Default value set to $0.2$.
        
    - `gain` (`int` | `float`): Linear scaling factor, defaulting to $1.0$.

```Python
from pathlib import Path
from fezrs.tools.image_enhancement import GammaCalculator

# Initialize power-law gamma transformation engine
gamma_corrector = GammaCalculator(
    nir_path=Path("./data/Shadowed_Terrain.tif"),
    gamma=0.45,
    gain=1.0
)
gamma_corrector.execute(output_path="./exports/enhanced/", title="Power-Law Gamma Correction")
```

### `LogAdjustCalculator` (Logarithmic Compression)

#### Scientific and Physical Objective

The logarithmic transformation enhances structural detail within highly compressed shadow regions. It expands dark pixel values while compressing bright, high-reflectance highlights.

#### Theoretical Foundation & Mathematical Formulations

The forward transformation maps input intensities to a logarithmic curve via:

$$I_{\text{out}} = A \cdot \log_e(1 + I_{\text{in}})$$

Where $A$ represents the user-defined linear `gain` factor. Because $\log_e(1 + 0) = 0$ and $\log_e(1 + 1) = \log_e(2) \approx 0.693$, using a default gain of $1.0$ scales the output range to $[0.0, 0.693]$. To restore the full dynamic range to $[0.0, 1.0]$, the scaling factor must be rebalanced using a gain of $A = \frac{1}{\log_e(2)} \approx 1.4427$.

When the `inverse` flag is set to `True`, the engine reverses this behavior by applying an exponential transformation:

$$I_{\text{out}} = A \cdot \left(e^{I_{\text{in}}} - 1\right)$$

This inverse mapping compresses shadow detail and expands variations in bright regions, making it ideal for highlighting features like clouds, snowcaps, or highly reflective mineral sands.

#### Interface Architecture

- **Constructor Arguments:**
    
    - `nir_path` (`str` | `Path`)
        
    - `gain` (`float`): Multiplicative scaler, defaulting to $1.0$.
        
    - `inverse` (`bool`): Determines whether to apply the forward log or inverse exponential curve.

### `SigmoidAdjustCalculator` (Sigmoid Contrast Optimization)

#### Scientific and Physical Objective

`SigmoidAdjustCalculator` applies a logistic function to enhance contrast within a specific mid-range intensity band. It stretches the contrast of mid-tones while smoothly compressing extreme highlights and shadows, preserving subtle details at both ends of the spectrum.

#### Theoretical Foundation & Mathematical Formulations

The forward logistic transformation is defined mathematically as:

$$I_{\text{out}} = \frac{1}{1 + \exp\left(g \cdot (\phi - I_{\text{in}})\right)}$$

Where:

- $\phi$ represents the `cutoff` parameter, which determines the inflection point of the sigmoid curve along the intensity axis (set to $0.5$ by default). Pixels matching this intensity map exactly to the center of the output range ($0.5$).
    
- $g$ represents the `gain` parameter, controlling the slope and steepness of the curve. Higher gain values create a steeper transition, increasing contrast near the inflection point.

When the `inverse` flag is set to `True`, the transformation is inverted using the following function:

$$I_{\text{out}} = \phi - \frac{1}{g} \cdot \ln\left(\frac{1}{I_{\text{in}}} - 1\right)$$

This inverse configuration stretches the extremes (highlights and shadows) while compressing contrast across mid-tone values.

#### Interface Architecture

- **Constructor Arguments:**
    
    - `nir_path` (`str` | `Path`)
        
    - `gain` (`float`): Sloping multiplier value (default: $1.0$).
        
    - `cutoff` (`float`): Inflection point coordinate locator (default: $0.5$).
        
    - `inverse` (`bool`): Toggle flag for inverse logistic transformation.

## Telemetry Integration: The Histogram Export Subsystem

Except for `LogAdjustCalculator`, all classes within this enhancement module inherit from **`HistogramExportMixin`**. This component tracks radiometric changes across the processing pipeline by generating, formatting, and saving high-fidelity diagnostic histogram plots.

### Telemetry Pipeline Features

- **Probability Density Profiling:** Plots pixel intensity frequencies across $256$ bins to monitor radiometric shifts.
    
- **Automated Publication Watermarking:** The mixin's internal `_add_watermark()` method injects a professional **FEZrs** logo onto the plot layout, ensuring all exported figures are ready for inclusion in official technical reports or academic publications.

```Python
# Direct export example for radiometric distribution profiling
equalizer.histogram_export(
    output_path="./exports/diagnostics/",
    title="Radiometric Profile: Post Global Equalization",
    figsize=(10, 6),
    filename_prefix="equalized_band_5",
    dpi=400
)
```

## Architectural Reference Summary

|**Class Name**|**Optimization Profile**|**Mathematical Core**|**Key Downstream Use Case**|
|---|---|---|---|
|**`FloatCalculator`**|Linear Range Normalization|$I / (2^b - 1)$|Standardizing integer imagery to float arrays before feature engineering.|
|**`EqualizeCalculator`**|Global Distribution Flattening|$\frac{1}{N} \sum \text{count}(j)$|Enhancing contrast in low-variance images flattened by heavy atmospheric haze.|
|**`AdaptiveCalculator`**|Localized Adaptive Contrast (CLAHE)|Bilinear Interpolation over Grid Tiles|Enhancing structural details across scenes with high local variation or uneven lighting.|
|**`GammaCalculator`**|Non-Linear Power-Law Adjustment|$A \cdot I^{\gamma}$|Brightening dark shadow regions ($\gamma < 1$) without blowing out pure highlights.|
|**`LogAdjustCalculator`**|Shadow Expansion / Highlight Compression|$A \cdot \log_e(1 + I)$|Mapping wide-dynamic-range sensor data to reveal low-intensity shadow details.|
|**`SigmoidAdjustCalculator`**|Mid-Tone Logistic Contrast Stretching|$\left[1 + e^{g(\phi - I)}\right]^{-1}$|Enhancing local features by stretching contrast across mid-range intensities.|
