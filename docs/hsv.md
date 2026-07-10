# HSV
## Overview

The `hsv` module provides advanced color-space transformation tools that map multi-spectral satellite band configurations from the standard **RGB (Red, Green, Blue)** additive color model to the **HSV (Hue, Saturation, Value)** cylindrical coordinate system.

While RGB representations are ideal for electronic display hardware, they mix chromatic information (dominant color tones) with illumination intensity. This coupling makes automated pixel classification and feature extraction highly sensitive to shadows, cloud cover, and changing terrain illumination.

Converting imagery to the HSV color space resolves this issue by isolating the pure chromatic signature (Hue and Saturation) from the scene's structural brightness (Value). This decoupling allows analysts to isolate changes in land cover independently of lighting conditions.

```
       [Raw Satellite Bands]
        (e.g., NIR, SWIR, Red)
                 │
                 ▼
     ┌───────────────────────┐
     │  Normalized RGB Stack │ (Values bounded in [0.0, 1.0])
     └───────────┬───────────┘
                 │
                 ▼
     ┌───────────────────────┐
     │ skimage.color.rgb2hsv │ (Non-linear Cylindrical Projection)
     └───────────┬───────────┘
                 │
        ┌────────┴────────┬────────────────┐
        ▼                 ▼                ▼
    [Hue (H)]     [Saturation (S)]   [Value (V)]
  Dominant Tone    Spectral Purity    Illumination
  (0.0 to 1.0)      (0.0 to 1.0)     (0.0 to 1.0)
```

## Mathematical Foundations of the RGB → HSV Transformation

The non-linear projection from a Cartesian RGB cube to a cylindrical HSV coordinate system assumes that all input channels are normalized to the floating-point range $[0.0, 1.0]$.

Given an input pixel triplet $(R, G, B)$, let $V$ represent the maximum channel intensity and $C$ represent the total chroma (dynamic range):

$$V = \max(R, G, B)$$

$$C = V - \min(R, G, B)$$

### Value ($V$)

The Value component represents the overall brightness of a pixel, extracted as the maximum value among the three color channels. In remote sensing, this component acts as a shadow-insensitive index of maximum surface reflectance.

$$V = \max(R, G, B)$$

### Saturation ($S$)

Saturation quantifies the purity or vividness of a color tone. It measures how far a spectral signature deviates from a grayscale value (where $R=G=B$).

$$S = \begin{cases} 0, & \text{if } V = 0 \\ \frac{C}{V}, & \text{if } V > 0 \end{cases}$$

- **Low Saturation ($S \to 0$):** Indicates balanced reflectance across all bands, typical of gray or white features like concrete, clouds, or highly reflective bare soils.
    
- **High Saturation ($S \to 1$):** Indicates that one or two bands strongly dominate the spectral signature, pointing to distinct target features (such as healthy vegetation or clear deep water).

### Hue ($H$)

Hue determines the dominant color tone expressed as an angular coordinate. In `scikit-image`, this angle ($0^\circ \text{ to } 360^\circ$) is mapped to the normalized continuous range $[0.0, 1.0]$. The value is calculated using a piecewise function determined by which channel matches the maximum intensity $V$:

$$H = \frac{60^\circ}{360^\circ} \cdot H_{\text{deg}}$$

Where the angular component $H_{\text{deg}}$ is defined as:

$$H_{\text{deg}} = \begin{cases} 0, & \text{if } C = 0 \\ \left( \frac{G - B}{C} \right) \bmod 6, & \text{if } V = R \\ \left( \frac{B - R}{C} \right) + 2, & \text{if } V = G \\ \left( \frac{R - G}{C} \right) + 4, & \text{if } V = B \end{cases}$$

If the computed value is negative, it is wrapped back into the valid range by adding $1.0$ ($H = H + 1.0$), ensuring a seamless cyclic boundary where $0.0$ and $1.0$ both map to pure red.

## Detailed Class Specifications

### `HSVCalculator` (Standard False-Color Vegetation Space)

#### Scientific and Physical Objective

`HSVCalculator` separates chromatic and illumination components from a standard $NIR\text{--}Green\text{--}Blue$ false-color composite. Mapping the highly reflective Near-Infrared ($NIR$) band to the Red channel emphasizes variations in leaf cellular structure and canopy density, making this tool ideal for analyzing vegetation health and spatial biomass patterns.

#### Channel Mapping & Remote Sensing Interpretation

The class routes normalized single-band inputs into a target three-channel matrix:

$$\begin{bmatrix} R \\ G \\ B \end{bmatrix} \leftarrow \begin{bmatrix} \text{NIR} \\ \text{Green} \\ \text{Blue} \end{bmatrix}$$

- **`hue` ($H$):** Pinpoints the dominant color tone. Healthy vegetation exhibits high $NIR$ reflectance paired with low visible light absorption, concentrating its signature near pure red ($H \approx 0.0$ or $1.0$). As vegetation undergoes stress or thins out, the visible bands contribute more to the composite, shifting the hue toward cyan and blue tones.
    
- **`saturation` ($S$):** Measures the contrast between the $NIR$ plateau and visible bands. High saturation values indicate high chlorophyll activity and dense canopies.
    
- **`value` ($V$):** Tracks maximum surface albedo. This provides a clear structural view of the landscape that helps identify topography and terrain boundaries while minimizing the impact of cloud shadows.

#### Interface Architecture

- **Constructor Method (`__init__`) Input Arguments:**
    
    - `nir_path` (`str` | `Path`): File path to the Near-Infrared raster layer.
        
    - `green_path` (`str` | `Path`): File path to the visible Green raster layer.
        
    - `blue_path` (`str` | `Path`): File path to the visible Blue raster layer.
        
    - `channel` (`Literal["hsv", "hue", "saturation", "value"]`): Specifies the output format. Selecting `"hsv"` exports a 3D multi-band cube `(Height, Width, 3)`, while selecting a single channel name returns a 2D spatial array.
    
- **Return State (`process()`):** Returns a 2D or 3D floating-point `numpy.ndarray` with values scaled between $[0.0, 1.0]$.

#### Operational Implementation

```Python
from pathlib import Path
from fezrs.tools.hsv import HSVCalculator

# Initialize standard vegetation HSV calculator
veg_engine = HSVCalculator(
    nir_path=Path("./data/S2_B08_NIR.tif"),
    green_path=Path("./data/S2_B03_Green.tif"),
    blue_path=Path("./data/S2_B02_Blue.tif"),
    channel="hue"
)

# Execute transformation and save output
# Note: Cyclic colormaps like 'hsv' or 'twilight' match the 
# circular properties of Hue, preventing edge artifacts at the 0.0/1.0 boundary.
veg_engine.execute(
    output_path="./exports/color_space/",
    title="Normalized False-Color Vegetation Hue Map",
    colormap="hsv",
    show_colorbar=True,
    dpi=500
)
```

### `IRHSVCalculator` (Infrared Moisture & Burn Space)

#### Scientific and Physical Objective

`IRHSVCalculator` maps short-wave infrared and visible bands to capture surface moisture anomalies, structural vegetation damage, and fire boundaries. It processes a $SWIR2\text{--}SWIR1\text{--}Red$ false-color composite, taking advantage of the fact that liquid water and high-moisture canopies strongly absorb short-wave infrared energy, whereas dry soil, exposed rock, and active burn scars reflect it highly.

#### Channel Mapping & Remote Sensing Interpretation

The input layers are mapped to the core color channels as follows:

$$\begin{bmatrix} R \\ G \\ B \end{bmatrix} \leftarrow \begin{bmatrix} \text{SWIR2} \\ \text{SWIR1} \\ \text{Red} \end{bmatrix}$$

- **`irhue` ($H_{\text{IR}}$):** Identifies specific land-surface modifications. Freshly burned areas show high $SWIR2$ reflectance from dry ash combined with low visible reflectance from charred surfaces. This isolates their signature within a narrow, predictable hue range ($H_{\text{IR}} \in [0.0, 0.15]$), separating fire scars from living vegetation.
    
- **`irsaturation` ($S_{\text{IR}}$):** Highlights areas with highly contrastive spectral profiles, such as mineral outcrops or intense fire impacts where $SWIR2$ values dominate over the other channels.
    
- **`irvalue` ($V_{\text{IR}}$):** Acts as an index of absolute shortwave reflectance, making it useful for separating high-reflectance features like clouds and ice from high-absorption features like open water.

#### Interface Architecture

- **Constructor Method (`__init__`) Input Arguments:**
    
    - `swir2_path` (`str` | `Path`): File path to the Short-Wave Infrared 2 raster layer (e.g., Landsat Band 7).
        
    - `swir1_path` (`str` | `Path`): File path to the Short-Wave Infrared 1 raster layer (e.g., Landsat Band 6).
        
    - `red_path` (`str` | `Path`): File path to the visible Red raster layer.
        
    - `channel` (`Literal["irhsv", "irhue", "irsaturation", "irvalue"]`): Specifies the output format. Defaults to `"irhsv"`.
    
- **Return State (`process()`):** Returns a 2D or 3D floating-point `numpy.ndarray` array capturing infrared texture indices scaled between $[0.0, 1.0]$.

#### Operational Implementation

```Python
from pathlib import Path
from fezrs.tools.hsv import IRHSVCalculator

# Initialize shortwave infrared HSV calculator for burn scar mapping
burn_engine = IRHSVCalculator(
    swir2_path=Path("./data/L8_B07_SWIR2.tif"),
    swir1_path=Path("./data/L8_B06_SWIR1.tif"),
    red_path=Path("./data/L8_B04_Red.tif"),
    channel="irhue"
)

# Execute transformation and save output
burn_engine.execute(
    output_path="./exports/color_space/",
    title="Infrared Hue Map for Burn Scar Analysis",
    colormap="twilight",
    show_colorbar=True,
    dpi=500
)
```

## Analytical Reference: Component Profiles

The table below summarizes how specific surface types behave across the different color-space components, providing a reference for setting up rule-based classification models:

|**Target Surface Feature**|**Composite Type**|**Hue Range Profile (H)**|**Saturation Profile (S)**|**Value Profile (V)**|**Analytical Application**|
|---|---|---|---|---|---|
|**Healthy Vegetation Canopy**|$NIR\text{--}Green\text{--}Blue$|$0.0 \le H \le 0.08$<br><br>  <br><br>(Pure Red Region)|High ($S \ge 0.75$)|Moderate to Low|Biomass monitoring, canopy tracking, and forest health assessments.|
|**Sparsely Vegetated / Bare Soil**|$NIR\text{--}Green\text{--}Blue$|$0.45 \le H \le 0.65$<br><br>  <br><br>(Cyan/Blue Shift)|Low ($S \le 0.25$)|High to Moderate|Desertification mapping and urban sprawl monitoring.|
|**Fresh Burn Scar / Charcoal**|$SWIR2\text{--}SWIR1\text{--}Red$|$0.0 \le H_{\text{IR}} \le 0.12$<br><br>  <br><br>(Deep Infrared Red)|Moderate ($0.4 \le S_{\text{IR}} \le 0.6$)|Moderate|Delineation of active fire perimeters and burn severity assessment.|
|**High Moisture / Water Saturated**|$SWIR2\text{--}SWIR1\text{--}Red$|Highly Variable|Low ($S_{\text{IR}} \le 0.15$)|Low ($V_{\text{IR}} \le 0.10$)|Flood boundary mapping and wetland delineation.|
