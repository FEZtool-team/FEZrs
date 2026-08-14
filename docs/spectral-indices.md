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
                    │ Linear Quantization Scaling    │ ──► $I_{\text{norm}} \in [0.0, 1.0]$
                    └───────────────┬────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
 [Single-Band Processing Arrays]                       [Multi-Band Raster Math Engine]
 ├─ NDVICalculator (NIR, Red)                           ├─ BICalculator (NIR, Red, Green)
 ├─ NDWICalculator (Green, NIR)                         └─ SAVICalculator (NIR, Red, $L=0.5$)
 ├─ UICalculator (SWIR2, NIR)                          
 └─ AFRICalculator (NIR, SWIR1)                        
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │    Dimensionless Float Output  │ ──► Dynamic range: $[-1.0, 1.0]$ or $[0.0, 1.0]$
                    └────────────────────────────────┘
```

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

### AFRICalculator` (Aerosol Free Vegetation Index)

#### Scientific and Physical Objective

The AFRI is designed to map dense forest canopies and high-biomass woody vegetation while providing a path to bypass atmospheric aerosol scattering (like smoke, haze, or dust). It enhances structural forest signatures while reducing sensitivity to variations in solar illumination, terrain shadowing, and background soil signatures by utilizing the short-wave infrared band instead of visible red.

#### Mathematical Formulation

The calculation separates the input into two distinct, interacting factors:

$$\text{AFRI} = (\text{NIR} - 0.66) \times \left( \frac{\text{SWIR1}}{\text{NIR} + 0.66 \times \text{SWIR1}} \right)$$

#### Biophysical Interaction Properties

- **NIR Offset Constant ($\text{NIR} - 0.66$):** Dense, healthy forest canopies consistently exhibit high near-infrared reflectance. The empirical constant **0.66** serves as a structural threshold; pixels with low near-infrared values (such as open water, shadows, or asphalt) produce negative or near-zero results, effectively suppressing non-vegetated features.
    
- **Non-Linear Modulation Ratio:** The second term uses the short-wave infrared band ($\text{SWIR1}$) to modulate the index response. Because moisture-rich forest leaf canopies absorb $\text{SWIR1}$ energy while reflecting $\text{NIR}$, this ratio stays small but positive for healthy forests. This dampens variations caused by topographic shadows, ensuring consistent canopy mapping across rugged terrain.

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

The Bare Soil Index isolates exposed soil surfaces, agricultural fallow fields, mining areas, and bare rock outcroppings by contrasting visible light combinations with near-infrared reflectance.

#### Mathematical Formulation

The index is calculated using a specialized normalized difference layout:

$$\text{BI} = \frac{(\text{NIR} - \text{Green}) - \text{Red}}{(\text{NIR} + \text{Green} + \text{Red})} = \frac{\text{NIR} - \text{Green} - \text{Red}}{\text{NIR} + \text{Green} + \text{Red}}$$

#### Biophysical Interaction Properties

 **Functional Inversion Reference:** Many published geological studies invert this formula to produce positive values for exposed soils. In this specific implementation, the layout uses $\text{NIR} - (\text{Green} + \text{Red})$ in the numerator. As a result, dense vegetation yields positive values, while bare soils—which show similar reflectance values across the visible green, red, and near-infrared bands—cluster near zero or drop into negative values. Concrete and asphalt structures typically produce strong negative values due to low near-infrared reflectance relative to visible wavelengths.

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
|**`AFRICalculator`**|Dense / Woody Forests|$(\text{NIR} - 0.66) \times \frac{\text{SWIR1}}{\text{NIR} + 0.66 \cdot \text{SWIR1}}$|$[0.0, \,\, +1.0]$|B5, B6|B8, B11|`'YlGn'`|
|**`NDWICalculator`**|Water Bodies & Hydrology|$\frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$|$[-1.0, \,\, +1.0]$|B3, B5|B3, B8|`'Blues'`|
|**`BICalculator`**|Bare Soil & Exposed Rock|$\frac{\text{NIR} - \text{Green} - \text{Red}}{\text{NIR} + \text{Green} + \text{Red}}$|$[-1.0, \,\, +1.0]$|B5, B4, B3|B8, B4, B3|`'inferno'` / `'hot'`|
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
