# Spectral Indices
## Overview

The `indices` module provides algebraic raster calculation pipelines designed to extract quantitative geophysical properties from multi-spectral satellite imagery. Every target surface feature exhibits a unique **spectral signature**—a characteristic pattern of electromagnetic radiation reflection and absorption across different wavelengths.

By calculating normalized differences, empirical scaling offsets, and non-linear band ratios, these calculators isolate specific surface materials, such as chlorophyll-heavy plant canopies, exposed mineral soils, open water bodies, and artificial urban structures.

```
                    ┌────────────────────────────────┐
                    │    Input Spectral TIF Bands    │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │ self.files_handler Ingestion   │
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │ Radiometric Scaling (optional) │ ──► $\rho = \text{DN} \times s + o$
                    └───────────────┬────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
 [Single-Band Processing Arrays]                       [Multi-Band Raster Math Engine]
 ├─ NDVICalculator (NIR, Red)                           ├─ BICalculator / BSI
 ├─ NDWICalculator (Green, NIR)                         │    (SWIR1, Red, NIR, Blue)
 ├─ UICalculator (SWIR2, NIR)                          └─ SAVICalculator (NIR, Red, $L=0.5$)
 └─ AFRICalculator (NIR, SWIR1 | SWIR2)                
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │    Dimensionless Float Output  │ ──► Dynamic range: $[-1.0, 1.0]$ or $[0.0, 1.0]$
                    └────────────────────────────────┘
```

## Radiometric Input Requirements

**Indices are computed on the band values as read.** No rescaling is applied unless you ask for it.

> **Changed in 1.4.0.** Earlier releases routed every index through a per-band min–max rescale, $(x - x_{\min}) / (x_{\max} - x_{\min})$, applied *independently to each band*. Because each band received a different affine transform, this altered the relationships **between** bands — and those relationships are the entire physical content of a spectral index. Three consequences: published thresholds did not apply, values were not comparable across scenes or dates, and **a pixel's value depended on how much of the image you loaded**, since the rescale used the loaded extent's own extrema. Index values from earlier versions are not comparable with current output.
>
> Measured on the bundled example, the rescale inverted inter-band relationships outright: AFRI's correlation with NDVI ran **+0.71 on the values as stored and −0.69 after rescaling**, against a source paper that reports the two as nearly identical. Min–max normalization remains in use for the enhancement, HSV, PCA and SVM modules, where rescaling is appropriate — and for SVM it is necessary, since an RBF kernel needs comparable feature scales.

### Which indices need reflectance, and which do not

| Index | Constant in reflectance units? | Safe on raw DN? |
|---|---|---|
| NDVI, NDWI, UI, BSI | none | **Yes** — a normalized difference is invariant to a gain applied to all bands equally |
| SAVI | soil adjustment $L = 0.5$ | **No** |
| AFRI | coefficients $0.66$ / $0.50$ on SWIR | **No** |

Adding $L = 0.5$ to a digital number in the thousands contributes nothing, so SAVI on unscaled input is silently not SAVI. The same applies to AFRI's coefficients. Both emit a `UserWarning` when handed values far outside the reflectance range.

### Supplying the scaling

Every index accepts `scale_factor` and `offset`, applied as $\rho = \text{DN} \times s + o$. Published values for the common analysis-ready products ship as `RADIOMETRIC_PRESETS`:

```Python
from fezrs import SAVICalculator
from fezrs.utils.radiometry_handler import RADIOMETRIC_PRESETS

SAVICalculator(
    nir_path="LC09_..._SR_B5.TIF",
    red_path="LC09_..._SR_B4.TIF",
    **RADIOMETRIC_PRESETS["landsat-c2-l2"],       # scale 2.75e-5, offset -0.2
).execute(output_path="./exports/")
```

| Preset | Scale | Offset |
|---|---|---|
| `landsat-c2-l2` | $2.75 \times 10^{-5}$ | $-0.2$ |
| `sentinel2-l2a` | $10^{-4}$ | $0.0$ |
| `sentinel2-l2a-baseline4` | $10^{-4}$ | $-0.1$ |
| `reflectance` | $1.0$ | $0.0$ |

Sentinel-2 processing baseline 04.00 and later carries a `BOA_ADD_OFFSET` of $-1000$, hence the separate preset. If your product is already reflectance, the defaults are correct and nothing needs passing.

## Mathematical & Scientific Formulations

### `NDVICalculator` (Normalized Difference Vegetation Index)

#### Scientific and Physical Objective

The NDVI is the standard index used to evaluate the presence, structural density, and photosynthetic health of green vegetation. It leverages the sharp transition—known as the **red edge**—between the strong visible light absorption of chlorophyll and the high near-infrared scattering of leaf tissues.

#### Mathematical Formulation

The mathematical index uses a normalized difference ratio:

$$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$

#### Biophysical Interaction Properties

- **Healthy Photosynthetic Canopies:** Chlorophyll pigments in functional leaves absorb up to **90%** of incident visible blue and red light to power photosynthesis, resulting in low reflectance in the $\text{Red}$ spectrum ($0.02 - 0.08$). Concurrently, the internal spongy mesophyll tissue strongly scatters near-infrared radiation back into space to prevent cellular overheating, driving high $\text{NIR}$ reflectance ($0.40 - 0.70$). This divergence causes the NDVI to approach its upper limit:
    

$$\text{NDVI}_{\text{veg}} \longrightarrow +1.0$$

- **Exposed Soils and Bedrock:** Lacking chlorophyll structures or complex cellular scattering cavities, bare earth exhibits a steady, linear increase in reflectance across both the red and near-infrared bands. Because $\text{NIR} \approx \text{Red}$, the numerator shrinks toward zero, yielding baseline values between **0.1** and **0.2**.
    
- **Open Water Bodies:** Clear water strongly absorbs electromagnetic energy across the reflective infrared spectrum, dropping $\text{NIR}$ values to near zero while maintaining a slight reflectance in visible wavelengths. This results in a negative index value.

###  `SAVICalculator` (Soil-Adjusted Vegetation Index)

#### Scientific and Physical Objective

The Soil-Adjusted Vegetation Index modifies the standard NDVI calculation for areas with sparse vegetation, such as arid shrublands, semi-arid agricultural fields, or early-stage crops. In these environments, the underlying soil brightness introduces significant background variation that can distort standard vegetation index values.

#### Mathematical Formulation

Originally derived by Huete (1988), the index adds an empirical soil calibration factor ($L$):

$$\text{SAVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red} + L} \times (1 + L)$$

To optimize the index for intermediate or variable canopy covers, the module hardcodes the scaling factor to $L = 0.5$:

$$\text{SAVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red} + 0.5} \times 1.5$$

#### Biophysical Interaction Properties

In sparse landscapes, variations in soil moisture, organic matter, and roughness can shift the baseline "soil line" in spectral feature space, artificially inflating or depressing standard NDVI values.

The $L = 0.5$ adjustment factor shifts the intersection of the soil line back to the coordinate origin, minimizing the effect of background soil brightness. The multiplicative scaler $(1 + L) = 1.5$ ensures the final output remains comparable to the standard $[-1.0, 1.0]$ range.

### `AFRICalculator` (Aerosol Free Vegetation Index)

#### Scientific and Physical Objective

The AFRI is designed to map dense forest canopies and high-biomass woody vegetation while providing a path to bypass atmospheric aerosol scattering (like smoke, haze, or dust). It enhances structural forest signatures while reducing sensitivity to variations in solar illumination, terrain shadowing, and background soil signatures by utilizing the short-wave infrared band instead of visible red.

#### Mathematical Formulation

Karnieli et al. (2001) define two formulations, selected with the `variant` parameter:

$$\text{AFRI}_{1.6} = \frac{\text{NIR} - 0.66 \times \text{SWIR1.6}}{\text{NIR} + 0.66 \times \text{SWIR1.6}} \qquad \text{AFRI}_{2.1} = \frac{\text{NIR} - 0.50 \times \text{SWIR2.1}}{\text{NIR} + 0.50 \times \text{SWIR2.1}}$$

#### Biophysical Interaction Properties

- **SWIR Substitution:** AFRI is structurally NDVI with a SWIR band substituted for the visible red. Aerosol scattering is strongly wavelength dependent and falls off toward longer wavelengths, so a SWIR-based index sees through smoke, haze and dust that would depress a red-based index. This is what makes AFRI usable over biomass-burning plumes and dust-laden atmospheres where NDVI fails.

- **The Coefficients ($0.66$ and $0.50$):** These are **scaling factors applied to the SWIR band**, not thresholds. They come from the empirical reflectance relationships $\rho_{0.645} \approx 0.66 \times \rho_{1.6}$ and $\rho_{0.469} \approx 0.50 \times \rho_{2.1}$ reported in the source paper, and their role is to place the SWIR reflectance on the scale of the visible band it replaces. Applying the coefficient anywhere other than to the SWIR reflectance makes the expression dimensionally incoherent.

- **Consistency With NDVI:** Under clear-sky conditions Karnieli et al. report that AFRI and NDVI values are almost identical. That equivalence is a practical validation check: on a haze-free scene, an AFRI implementation that does **not** correlate strongly and positively with NDVI is computing something else.

> **Changed in 1.4.0.** Earlier releases computed $(\text{NIR} - 0.66) \times \text{SWIR1} / (\text{NIR} + 0.66 \times \text{SWIR1})$, which subtracts a bare dimensionless constant from a reflectance and then multiplies by a band ratio. It is a different quantity from Karnieli's index, correlating with it at only 0.17, and it produced negative values over 96% of the bundled example while this page claimed a $[0, +1]$ range. AFRI results from earlier versions are not comparable with current output.

### `NDWICalculator` (Normalized Difference Water Index)

#### Scientific and Physical Objective

The NDWI outlines open water bodies (such as lakes, reservoirs, river channels, and wetlands) and separates them from surrounding land features based on the McFeeters (1996) specification.

#### Mathematical Formulation

The index calculates the normalized difference between the visible green and near-infrared bands:

$$\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$$

#### Biophysical Interaction Properties

Clear surface water reflects slightly in the visible spectrum—peaking near the $\text{Green}$ wavelength—but absorbs almost all incident light in the near-infrared ($\text{NIR}$) band. This characteristic behavior yields a positive index value for water bodies:

$$\text{NDWI}_{\text{water}} \longrightarrow +1.0$$

In contrast, land features like healthy vegetation or dry soils reflect much more strongly in the $\text{NIR}$ than in the $\text{Green}$ band, producing negative values that clearly separate terrestrial features from open water.

### `BICalculator` (Bare Soil Index)

#### Scientific and Physical Objective

The Bare Soil Index isolates exposed soil surfaces, agricultural fallow fields, mining areas, and bare rock outcroppings by contrasting short-wave infrared and red brightness against near-infrared and blue reflectance.

#### Mathematical Formulation

The default formulation is the Bare Soil Index (BSI), selected automatically when `swir1_path` and `blue_path` are supplied:

$$\text{BSI} = \frac{(\text{SWIR1} + \text{Red}) - (\text{NIR} + \text{Blue})}{(\text{SWIR1} + \text{Red}) + (\text{NIR} + \text{Blue})}$$

#### Biophysical Interaction Properties

- **Four-Band Contrast:** Bare soil and exposed rock are bright in $\text{SWIR1}$ and $\text{Red}$ and comparatively dark in $\text{NIR}$ and $\text{Blue}$. Vegetation is the exact inverse — a strong $\text{NIR}$ plateau against low visible reflectance. Pairing the bands this way makes the numerator change sign between the two surface types, so **BSI is positive over exposed ground and negative over canopy**, which is what allows lithological exposure to be separated from vegetation cover.

- **Why SWIR Is Required:** The $\text{SWIR1}$ term carries the soil-moisture and clay-mineral response that no visible-band combination reproduces. An index built only from visible and NIR bands cannot distinguish dry bare soil from a sparsely vegetated surface of similar visible brightness.

#### Legacy Formulation

Previous releases computed a different expression, which remains reachable by passing `green_path` instead of `swir1_path`/`blue_path`:

$$\text{BI}_{\text{legacy}} = \frac{\text{NIR} - \text{Green} - \text{Red}}{\text{NIR} + \text{Green} + \text{Red}}$$

This is **not a published bare-soil index**. It subtracts two visible bands from NIR, so it responds primarily to vegetation brightness — dense vegetation yields positive values and bare soils cluster near zero or negative, the opposite of what the name implies. It is retained so existing results stay reproducible, and it emits a `DeprecationWarning`. Use the BSI formulation for new work.

> **Citation note.** Earlier documentation attributed this module to As-syakur et al. (2012), which defines EBBI, $(\text{SWIR} - \text{NIR}) / (10\sqrt{\text{SWIR} + \text{TIR}})$ — a different formula requiring a thermal band this tool does not accept. That citation has been removed. The legacy expression above has no published source and should be treated as an original formulation.

### `UICalculator` (Urban Index)

#### Scientific and Physical Objective

The Urban Index detects and maps built-up areas, impervious surfaces, and artificial infrastructure (such as concrete, asphalt, and roofing materials) by leveraging the contrast between short-wave infrared and near-infrared reflectance.

#### Mathematical Formulation

The index is computed as:

$$\text{UI} = \frac{\text{SWIR2} - \text{NIR}}{\text{SWIR2} + \text{NIR}}$$

#### Biophysical Interaction Properties

Man-made materials like concrete, asphalt, and stone maintain high reflectance in the longer short-wave infrared band ($\text{SWIR2}$) but lack the internal cell structures that cause high near-infrared ($\text{NIR}$) scattering. This gives urban features a positive index value:

$$\text{UI}_{\text{urban}} \longrightarrow +1.0$$

Vegetated regions show the opposite response, strongly absorbing $\text{SWIR2}$ energy via leaf moisture while scattering $\text{NIR}$ light, resulting in highly negative values that separate urban areas from surrounding natural land covers.

## Reference Matrix & Sensor Configurations

The following matrix cross-references the required sensor channels, target ranges, and optimal styling palettes across different satellite platforms:

|**Index Class**|**Target Mapping Feature**|**Mathematical Equation Matrix**|**Operational Value Range**|**Landsat 8/9 Channels**|**Sentinel-2 Channels**|**Optimal Colormap**|
|---|---|---|---|---|---|---|
|**`NDVICalculator`**|Canopy Health & Density|$\frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$|$[-1.0, \,\, +1.0]$|B5, B4|B8, B4|`'RdYlGn'` / `'YlGn'`|
|**`SAVICalculator`**|Sparse / Arid Shrublands|$\frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red} + 0.5} \times 1.5$|$[-1.0, \,\, +1.0]$|B5, B4|B8, B4|`'RdYlGn'` / `'YlGn'`|
|**`AFRICalculator`**|Dense / Woody Forests, Hazy Scenes|$\frac{\text{NIR} - 0.66 \cdot \text{SWIR1}}{\text{NIR} + 0.66 \cdot \text{SWIR1}}$|$[-1.0, \,\, +1.0]$|B5, B6|B8, B11|`'YlGn'`|
|**`NDWICalculator`**|Water Bodies & Hydrology|$\frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$|$[-1.0, \,\, +1.0]$|B3, B5|B3, B8|`'Blues'`|
|**`BICalculator`**|Bare Soil & Exposed Rock|$\frac{(\text{SWIR1} + \text{Red}) - (\text{NIR} + \text{Blue})}{(\text{SWIR1} + \text{Red}) + (\text{NIR} + \text{Blue})}$|$[-1.0, \,\, +1.0]$|B6, B4, B5, B2|B11, B4, B8, B2|`'inferno'` / `'hot'`|
|**`UICalculator`**|Built-up Urban Areas|$\frac{\text{SWIR2} - \text{NIR}}{\text{SWIR2} + \text{NIR}}$|$[-1.0, \,\, +1.0]$|B7, B5|B12, B8|`'coolwarm'`|

## Operational Implementation Examples

### Visualizing Vegetation Density (NDVI)

```Python
from pathlib import Path
from fezrs.tools.indices import NDVICalculator

# Instantiate the NDVI processing engine using Landsat 8 paths
ndvi_engine = NDVICalculator(
    nir_path=Path("./data/LC08_B5_NIR.tif"),
    red_path=Path("./data/LC08_B4_Red.tif")
)

# Run calculations and export a stylized, color-mapped quick-look array
ndvi_engine.execute(
    output_path="./exports/indices/",
    title="Normalized Difference Vegetation Index (NDVI)",
    colormap="RdYlGn",
    show_colorbar=True
)
```

### Surface Hydrology Mapping (NDWI)

```Python
from pathlib import Path
from fezrs.tools.indices import NDWICalculator

# Instantiate McFeeters NDWI calculator using Sentinel-2 paths
ndwi_engine = NDWICalculator(
    green_path=Path("./data/S2_B03_Green.tif"),
    nir_path=Path("./data/S2_B08_NIR.tif")
)

# Extract water body boundaries
water_mask_raster = ndwi_engine.process()
```
