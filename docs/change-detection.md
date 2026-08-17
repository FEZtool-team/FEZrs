# Change Detection
## Overview

The `change_detection` module provides a comprehensive suite of remote sensing tools designed to isolate, classify, and quantify temporal surface dynamics using bi-temporal satellite imagery. By analyzing multi-spectral imagery captured across two distinct temporal windows—typically classified as **Pre-Event ($t_0$, Before)** and **Post-Event ($t_1$, After)**—these tools facilitate automated monitoring of ecological disturbances such as wildfire burn severity, flood inundation, land-cover conversions, and vegetation degradation.

```
       fezrs.base.BaseTool [Base Architecture]
                 │
                 ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                   fezrs.tools.change_detection Module                  │
 ├───────────────────┬─────────────────────┬──────────────────────────────┤
 │                   │                     │                              │
 ▼                   ▼                     ▼                              ▼
BurnCalculator   IndicesCalculator   MagDirCalculator    SubDivCalculator & TimeCalculator
```

## Common Dependencies & Inheritance Model

All tools share a strict, predictable execution lifecycle managed by the parent class's workflow.

- **File Tracking:** Inputs are passed as standardized path types (`str` or `pathlib.Path`), resolved by the file management handler, and converted into multi-dimensional NumPy arrays (`numpy.ndarray`).
    
- **Processing Lifecycle:** Every tool overrides the private engineering core `_validate()` and the main operational gateway `process()`.
    
- **Output Pipe:** The `execute()` method serializes the internal computed state (`self._output`) to a PNG figure via Matplotlib. Call `export_raster()` to write a georeferenced GeoTIFF that copies CRS and transform from the first input band.

## Comprehensive Class Specifications

### `BurnCalculator` — Wildfire Severity Mapping via Differential Burn Ratios

#### Scientific & Physical Objective

The primary operational goal of `BurnCalculator` is to compute the **Differenced Normalized Burn Ratio ($dNBR$)**, which isolates the physical destruction of vegetative biomass caused by fire events, delineating burn scar perimeters and classifying relative ecological severity.

#### Theoretical Foundation & Mathematical Formulations

Healthy vegetative canopies exhibit high cellular reflectance in the Near-Infrared ($NIR$) spectrum due to the structural scattering properties of leaf mesophyll tissue, alongside low reflectance in the Short-Wave Infrared ($SWIR2$) band due to strong absorption by liquid water stored within the plant tissue.

During an intense fire event, photosynthetic vegetation is consumed, destroying canopy architecture and liquid water reservoirs. The surface is converted into ash, charcoal, and exposed mineral soil. This radical physical shift triggers a steep drop in $NIR$ reflectance coupled with a profound increase in $SWIR2$ reflectance.

```
[Healthy Canopy] ──────► High NIR Reflection  + High SWIR2 Absorption ──► High Positive NBR
[Burned Canopy]  ──────► Low NIR Reflection   + Low SWIR2 Absorption  ──► Negative/Low NBR
```

The **Normalized Burn Ratio ($NBR$)** mathematically scales this relationship within normalized limits of $[-1.0, +1.0]$:

$$NBR = \frac{NIR - SWIR2}{NIR + SWIR2}$$

To eliminate background physical constants, topographical lighting discrepancies, and solar zenith illumination geometry, the **Differenced Normalized Burn Ratio ($dNBR$)** evaluates the absolute delta across the temporal baseline:

$$dNBR = NBR_{\text{before}} - NBR_{\text{after}}$$

In code execution, the delta is structurally represented as:

$$\Delta NBR = NBR_{t_0} - NBR_{t_1}$$

An elevated positive value directly registers vegetative loss and severe canopy charring. The module applies a strict binary thresholding condition to classify highly affected fire zones:

$$\text{Burn Mask} = \begin{cases} \text{True (1),} & \text{if } dNBR > 0.7 \\ \text{False (0),} & \text{if } dNBR \le 0.7 \end{cases}$$

#### Standard Interpretation Scale

The operational cutoff of $0.7$ targeting high-severity burn scars conforms strictly to the environmental monitoring metrics established by Key & Benson (2006) and the United States Geological Survey (USGS):

|**dNBR Interval Range**|**Biophysical Classification Severity Status**|
|---|---|
|$< 0.10$|Unburned / Control Zone|
|$0.10 \le dNBR \le 0.27$|Low-Severity Burn|
|$0.27 < dNBR \le 0.66$|Moderate-Severity Burn|
|$> 0.66$|High-Severity Burn Scar ($> 0.70$ application cutoff)|

#### Edge-Case Computational Handling

In rare conditions where absolute dark pixels or shadow voids generate a total zero denominator ($NIR + SWIR2 = 0$), the resulting $\text{NaN}$ floating-point evaluation is trapped by the conditional evaluator ($\text{NaN} > 0.7$), naturally falling back safely to a `False` assignment.

#### Interface Architecture

- **Constructor Method (`__init__`) Input Arguments:**
    
    - `nir_path` (`str` | `Path`): Post-event ($t_1$) Near-Infrared band file location.
        
    - `swir2_path` (`str` | `Path`): Post-event ($t_1$) Short-Wave Infrared (Band 7 equivalent) file location.
        
    - `before_nir_path` (`str` | `Path`): Pre-event ($t_0$) Near-Infrared reference band file location.
        
    - `before_swir2_path` (`str` | `Path`): Pre-event ($t_0$) Short-Wave Infrared reference band file location.
    
- **Return State (`process()`):** Returns a boolean `numpy.ndarray` acting as a strict spatial mask for severe burn perimeters.

#### Execution Implementation

```Python
from pathlib import Path
from fezrs.tools.change_detection import BurnCalculator

# Initialize burn severity calculation pipeline
burn_analyzer = BurnCalculator(
    nir_path=Path("./data/post_event_B5.tif"),
    swir2_path=Path("./data/post_event_B7.tif"),
    before_nir_path=Path("./data/pre_event_B5.tif"),
    before_swir2_path=Path("./data/pre_event_B7.tif")
)

# Execute core processing and save high-resolution binary burn mask
burn_analyzer.execute(
    output_path="./exports/burn_mapping/",
    title="High-Severity Wildfire Scar Mask",
    colormap="Reds",
    dpi=500
)
```

### `IndicesCalculator` — Single-Date Baseline Radiometric Evaluation

#### Scientific & Physical Objective

This calculator extracts the absolute, standalone Normalized Burn Ratio ($NBR$) array for a singular, targeted date instance ($t_0$ or $t_1$). It provides structural diagnostic controls for baseline canopy vigor, pre-fire fuel-load mapping, or intermediate analytical steps.

#### Theoretical Foundation & Mathematical Formulations

The analytical equation executes the non-linear difference calculation isolated to the user's temporal selection:

$$NBR = \frac{NIR - SWIR2}{NIR + SWIR2}$$

The temporal trajectory is governed explicitly via the `time` parameter flag:

- When `time="before"`, calculations map the initial environmental status ($t_0$). This helps diagnose pre-fire desiccation or moisture stress across vulnerable canopies.
    
- When `time="after"`, calculations map the structural landscape distribution ($t_1$) after the occurrence of the physical disturbance event.

The array maintains continuous floating-point values constrained to a $[-1.0, 1.0]$ range.

#### Interface Architecture

- **Constructor Method (`__init__`) Input Arguments:**
    
    - `nir_path` (`str` | `Path`): Post-event ($t_1$) $NIR$ band file path.
        
    - `swir2_path` (`str` | `Path`): Post-event ($t_1$) $SWIR2$ band file path.
        
    - `before_nir_path` (`str` | `Path`): Pre-event ($t_0$) $NIR$ band file path.
        
    - `before_swir2_path` (`str` | `Path`): Pre-event ($t_0$) $SWIR2$ band file path.
        
    - `time` (`Literal["before", "after"]`): Directing pointer selecting the active target layer matrix.
    
- **Return State (`process()`):** Returns a floating-point `numpy.ndarray` array representing absolute $NBR$ spectral coordinates.

#### Execution Implementation

```Python
from pathlib import Path
from fezrs.tools.change_detection import IndicesCalculator

# Instantiate standalone single-date baseline engine
nbr_baseline = IndicesCalculator(
    nir_path=Path("./data/t1_B5.tif"),
    swir2_path=Path("./data/t1_B7.tif"),
    before_nir_path=Path("./data/t0_B5.tif"),
    before_swir2_path=Path("./data/t0_B7.tif"),
    time="before"
)

# Run process to isolate baseline pre-fire fuel stress
nbr_baseline.execute(
    output_path="./exports/indices/",
    title="Pre-Fire Baseline NBR",
    colormap="YlGn",
    dpi=500
)
```

### `MagDirCalculator` — Multi-Spectral Change Vector Analysis (CVA)

#### Scientific & Physical Objective

`MagDirCalculator` processes spatial dynamics through Change Vector Analysis (CVA) across a dual-dimensional spectral feature space ($NIR \times SWIR1$). This technique quantifies the exact magnitude of spectral pixel shifts and categorizes the physical direction of land-cover transitions.

#### Theoretical Foundation & Mathematical Formulations

A pixel's bi-temporal spectral state is modeled as a mathematical coordinate trajectory shifting through space. By establishing the primary axes via Near-Infrared ($NIR$, indicating vegetative density) and Short-Wave Infrared ($SWIR1$, indicating canopy and soil moisture metrics), bi-temporal coordinates are plotted as:

$$P_{\text{before}} = (NIR_{t_0}, SWIR1_{t_0}) \quad \text{and} \quad P_{\text{after}} = (NIR_{t_1}, SWIR1_{t_1})$$

The corresponding **Change Vector ($\vec{V}$)** is defined as:

$$\vec{V} = \begin{bmatrix} \Delta NIR \\ \Delta SWIR1 \end{bmatrix} = \begin{bmatrix} NIR_{t_1} - NIR_{t_0} \\ SWIR1_{t_1} - SWIR1_{t_0} \end{bmatrix}$$

##### Change Vector Magnitude ($|\vec{V}|$)

The total geometric shift across spectral feature space is calculated using the Euclidean distance equation:

$$|\vec{V}| = \sqrt{(NIR_{t_1} - NIR_{t_0})^2 + (SWIR1_{t_1} - SWIR1_{t_0})^2}$$

Higher values indicate intense land-cover modifications (e.g., rapid deforestation, urban clearings, or severe wildfire devastation), independent of the qualitative nature of the change.

##### Change Vector Directional Coding

The discrete directional quadrant ($1$ through $4$) indicates the path taken by the pixel trajectory across the spectral domain. These paths map directly to distinct qualitative environmental transformations:

```
                  ▲ ΔSWIR1 (Moisture Decrease)
                  │
     Quadrant 3   │   Quadrant 4
     [ΔNIR < 0]   │   [ΔNIR > 0]
     [ΔSWIR1 > 0] │   [ΔSWIR1 > 0]
                  │
──────────────────┼──────────────────► ΔNIR
                  │  (Biomass Increase)
     Quadrant 1   │   Quadrant 2
     [ΔNIR < 0]   │   [ΔNIR > 0]
     [ΔSWIR1 < 0] │   [ΔSWIR1 < 0]
                  │
```

- **Quadrant Code 1 ($\Delta NIR < 0 \text{ and } \Delta SWIR1 < 0$):** Simultaneous drop in structural biomass and liquid moisture absorption. This signature often represents intense fire events, vegetation clearings, or canopy die-back.
    
- **Quadrant Code 2 ($\Delta NIR > 0 \text{ and } \Delta SWIR1 < 0$):** Increasing cell structures alongside active water-retention scaling. This confirms vegetation vigor enhancement, characteristic of reforestation, crop maturity, or canopy recovery.
    
- **Quadrant Code 3 ($\Delta NIR < 0 \text{ and } \Delta SWIR1 > 0$):** Biomass loss with rising shortwave reflection. This indicates clear canopy stripping accompanied by soil water saturation, revealing events like structural flooding, wetland expansion, or extensive irrigation.
    
- **Quadrant Code 4 ($\Delta NIR > 0 \text{ and } \Delta SWIR1 > 0$):** Co-registered elevation across both spectrum tracks. This typically points to complex land conversions, new urban concrete structures, or sensor illumination artifacts.

#### Interface Architecture

- **Constructor Method (`__init__`) Input Arguments:**
    
    - `nir_path` (`Path`): Post-event ($t_1$) $NIR$ band file path.
        
    - `swir1_path` (`Path`): Post-event ($t_1$) $SWIR1$ band file path.
        
    - `before_nir_path` (`Path`): Pre-event ($t_0$) $NIR$ band file path.
        
    - `before_swir1_path` (`Path`): Pre-event ($t_0$) $SWIR1$ band file path.
        
    - `selecte` (`Literal["magnitude", "direction"]`): Selector pointing target execution to compute numerical scalar distances or integer category labels.
    
- **Return State (`process()`):** Returns a `numpy.ndarray` containing floating-point Euclidean distances or integer categorical maps. Direction codes are $1$–$4$ as above; **$0$ means no change** on at least one axis (a difference of exactly $0$).

#### Execution Implementation

```Python
from pathlib import Path
from fezrs.tools.change_detection import MagDirCalculator

# Instantiate change vector analysis workflow
cva_engine = MagDirCalculator(
    nir_path=Path("./data/after_NIR.tif"),
    swir1_path=Path("./data/after_SWIR1.tif"),
    before_nir_path=Path("./data/before_NIR.tif"),
    before_swir1_path=Path("./data/before_SWIR1.tif"),
    selecte="magnitude"
)

# Render change tracking magnitude map
cva_engine.execute(
    output_path="./exports/cva/",
    title="Change Vector Euclidean Magnitude",
    colormap="viridis",
    dpi=500
)
```

### `SubDivCalculator` — Direct Linear Differencing and Ratio Computations

#### Scientific & Physical Objective

`SubDivCalculator` provides quick, intuitive structural diagnostics by performing cell-by-cell algebraic subtraction or division across a targeted band array. This delivers immediate localized assessments of surface reflectance changes without index scaling overhead.

#### Theoretical Foundation & Mathematical Formulations

The system processes change tracking across two structural arithmetic options:

##### Subtraction Mode (`operation="subtract"`)

$$\Delta_{\text{Reflectance}} = Band_{t_0} - Band_{t_1}$$

This calculation preserves the original radiometric values of the scene. Positive results indicate a drop in surface reflection at $t_1$ (e.g., vegetative harvesting), while negative results capture localized reflection gains (e.g., sediment deposits or building developments).

##### Division Mode (`operation="divide"`)

$$R_{\text{Reflectance}} = \frac{Band_{t_0}}{Band_{t_1}}$$

The ratio transformation helps minimize terrain shadow effects and illumination differences caused by topography.

- A ratio value of $1.0$ indicates perfect stability across the bi-temporal window.
    
- Ratios $> 1.0$ reveal structural reflection losses at $t_1$.
    
- Ratios $< 1.0$ show absolute gains in reflection over the same period.

#### Interface Architecture

- **Constructor Method (`__init__`) Input Arguments:**
    
    - `nir_path` (`Path`): Post-event ($t_1$) band file target location.
        
    - `before_nir_path` (`Path`): Pre-event ($t_0$) band file reference location.
        
    - `operation` (`Literal["subtract", "divide"]`): Algebraic operator selection.
    
- **Return State (`process()`):** Returns a `numpy.ndarray` numerical array capturing pixel-by-pixel change deltas or ratios.

#### Execution Implementation

```Python
from pathlib import Path
from fezrs.tools.change_detection import SubDivCalculator

# Setup quick linear band subtract pipeline
linear_diff = SubDivCalculator(
    nir_path=Path("./data/t1_NIR.tif"),
    before_nir_path=Path("./data/t0_NIR.tif"),
    operation="subtract"
)

# Export structural linear difference array
linear_diff.execute(
    output_path="./exports/linear/",
    title="NIR Band Absolute Subtraction (t0 - t1)",
    colormap="bwr",
    dpi=500
)
```

### `TimeCalculator` — Bi-Temporal Extraction and Diagnostic Baseline Isolation

#### Scientific & Physical Objective

`TimeCalculator` provides low-level debugging isolation and user-interface baseline extraction. It bypasses all change-detection operations to directly stream standardized, core array matrices from either $t_0$ or $t_1$.

#### Interface Architecture

- **Constructor Method (`__init__`) Input Arguments:**
    
    - `nir_path` (`Path`): Post-event ($t_1$) $NIR$ band file path reference.
        
    - `before_nir_path` (`Path`): Pre-event ($t_0$) $NIR$ band file path reference.
        
    - `time` (`Literal["before", "after"]`): Directing pointer selecting the temporal layer target.
    
- **Return State (`process()`):** Returns a standardized floating-point `numpy.ndarray` array containing raw reflectance values from the selected timestamp.

#### Execution Implementation

```Python
from pathlib import Path
from fezrs.tools.change_detection import TimeCalculator

# Isolate pristine baseline array data for validation
raw_viewer = TimeCalculator(
    nir_path=Path("./data/t1_NIR.tif"),
    before_nir_path=Path("./data/t0_NIR.tif"),
    time="before"
)

# Output raw baseline array map
raw_viewer.execute(
    output_path="./exports/diagnostic/",
    title="Pristine Baseline Pre-Event NIR Array",
    colormap="gray",
    dpi=500
)
```
