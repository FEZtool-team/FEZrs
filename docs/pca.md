# PCA (Principal Component Analysis)
## Overview

The `pca` module implements a Principal Component Analysis (PCA) workflow designed for dimensionality reduction, spectral-spatial information compression, and uncorrelated feature extraction from multi-spectral satellite imagery. Satellite sensors capture surface reflectance across multiple overlapping spectral bands, which often results in high data redundancy and strong correlation between adjacent channels.

This module unifies a six-band multi-spectral stack—typically comprising the visible spectrum (**Red, Green, Blue**), Near-Infrared (**NIR**), and Short-Wave Infrared (**SWIR1, SWIR2**)—and projects it into a new coordinate space. The resulting orthogonal axes, or **Principal Components (PCs)**, are ordered by the amount of total variance they explain, isolating dominant spatial patterns and suppressing high-frequency sensor noise.

This projection identifies the dominant spatial patterns that account for the most variability across the six spectral bands at each pixel, generating coherent **eigenimages** that capture unique landscape characteristics.

```
       [6 Native Spatial Bands] (Red, Green, Blue, NIR, SWIR1, SWIR2)
                  │
                  ▼
   ┌─────────────────────────────┐
   │ Stack + Flatten Block       │ ──► Each pixel is one sample; each band is one feature
   └──────────────┬──────────────┘
                  │
                  ▼
   ┌─────────────────────────────┐
   │ Matrix Construction ($X$)    │ ──► Dimensions: $\mathbb{R}^{N \times 6}$ ($N = H \times W$ pixels, 6 bands)
   └──────────────┬──────────────┘
                  │
                  ▼
   ┌─────────────────────────────┐
   │ Feature Centering           │ ──► Subtract each band's mean: $X_c = X - \bar{x}$
   └──────────────┬──────────────┘
                  │
                  ▼
   ┌─────────────────────────────┐
   │  Singular Value Decomposition│ ──► $X_c = U \Sigma V^T$ (covariance is $6 \times 6$)
   └──────────────┬──────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   [Right Singular Vectors $V$]  [Scores $T = X_c V$]
    Spectral loadings            6 spatial eigenimages (PC1 - PC6)
   ($V \in \mathbb{R}^{6 \times 6}$)     Shape: $(N, 6) \to$ reshaped to $(6, H, W)$
```

## Comprehensive Mathematical Foundations

### Spatial Data Matrix Construction

Let each input band $b$ (where $b \in \{1, 2, \dots, 6\}$) represent a discrete 2D image matrix of height $H$ and width $W$. Each pixel is treated as one statistical sample, and each spectral band is treated as one feature. After stacking the six bands and flattening the spatial axes, the multi-spectral data matrix is:

$$X \in \mathbb{R}^{N \times 6}, \qquad N = H \times W$$

Row $i$ of $X$ is the six-band reflectance vector of a single pixel. This is the orientation expected by `sklearn.decomposition.PCA`, which treats rows as samples and columns as features.

### Feature Centering

Each spectral band is centered by subtracting that band's mean across all pixels:

$$\bar{x}_b = \frac{1}{N} \sum_{i=1}^{N} X_{ib}, \qquad b \in \{1, \dots, 6\}$$

$$X_c = X - \bar{x}$$

This removes the global brightness offset of each band before decomposition.

### Covariance and SVD

With six features, the sample covariance is a $6 \times 6$ matrix:

$$C = \frac{1}{N - 1} X_c^T X_c \in \mathbb{R}^{6 \times 6}$$

`PCACalculator` uses `sklearn.decomposition.PCA`, which centers $X$ and computes the SVD:

$$X_c = U \Sigma V^T$$

Where:

- $U \in \mathbb{R}^{N \times 6}$ contains the left singular vectors.
    
- $\Sigma \in \mathbb{R}^{6 \times 6}$ is a diagonal matrix of singular values ($\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_6 \ge 0$).
    
- $V \in \mathbb{R}^{6 \times 6}$ contains the right singular vectors (spectral loadings).

The spatial principal-component images are the transformed sample scores:

$$T = X_c V = U \Sigma \in \mathbb{R}^{N \times 6}$$

Each column of $T$ is then reshaped back to $(H, W)$. The full output stack has shape $(6, H, W)$.

The eigenvalues $\lambda_k$ of $C$ are related to the singular values by:

$$\lambda_k = \frac{\sigma_k^2}{N - 1}$$

### Quantifying Explained Variance

The proportion of total spectral variance captured by component $k$ is:

$$VE_k = \frac{\lambda_k}{\sum_{j=1}^{6} \lambda_j} = \frac{\sigma_k^2}{\sum_{j=1}^{6} \sigma_j^2}$$

## Class Specification: `PCACalculator`

### Scientific and Interpretation Profiles

When a principal component vector of length $N$ is reshaped back to its original dimensions $(H, W)$, it forms a spatial map known as an **eigenimage**. These components separate different types of information based on how variation is distributed across the scene:

#### Principal Component 1 (PC1) — Albedo and Illumination Map

PC1 captures the most dominant spatial variation common to all six input bands, typically accounting for **more than 90% of the total scene variance**. Because land cover features generally reflect light with similar broad trends under uniform lighting, the PC1 image behaves like a panchromatic brightness map. It highlights overall surface albedo, topographic shading, and solar illumination while minimizing compositional differences.

#### Principal Component 2 (PC2) — Compositional and Vegetation Contrast

PC2 highlights the second most dominant axis of variation, focusing on strong contrasts between different wavelengths. In landscapes with active vegetation, PC2 typically captures the sharp divergence between high near-infrared ($NIR$) reflectance and the strong visible light absorption of chlorophyll. This makes it an effective index for mapping biomass distribution and separating vegetative cover from urban surfaces or open water.

#### Principal Components 3 to 6 (PC3–PC6) — Residuals and Sensor Noise

These higher-order components capture progressively smaller variations in the data. While PC3 often highlights subtle moisture or mineral variations across the short-wave infrared spectrum ($SWIR$), components PC4 through PC6 are typically dominated by high-frequency sensor noise, atmospheric striping, and random background variations. Consequently, these late-stage components can generally be discarded during data compression workflows without losing meaningful information.

### Interface Architecture

#### Constructor Method Signature (`__init__`)

- **Input Arguments:**
    
    - `red_path` (`str` | `Path`): File path to the visible Red band raster layer.
        
    - `green_path` (`str` | `Path`): File path to the visible Green band raster layer.
        
    - `blue_path` (`str` | `Path`): File path to the visible Blue band raster layer.
        
    - `nir_path` (`str` | `Path`): File path to the Near-Infrared band raster layer.
        
    - `swir1_path` (`str` | `Path`): File path to the Short-Wave Infrared 1 band raster layer.
        
    - `swir2_path` (`str` | `Path`): File path to the Short-Wave Infrared 2 band raster layer.
        
    - `selectBand` (`Literal["red","green","blue","nir","swir1","swir2", None]`): Optional parameter. Selects a specific input band to map against the component outputs during specialized diagnostic profiling.

#### Processing Pipeline Lifecycle (`process()`)

1. Ingests the six target spectral bands, each with shape `(height, width)`.
    
2. Stacks the bands into a matrix of shape `(N_pixels, 6)`, where `N_pixels = height * width`. Each row is one pixel; each column is one spectral band.
    
3. Passes this matrix to `sklearn.decomposition.PCA(n_components=6).fit_transform()`. The model centers each band and returns the six principal-component scores for every pixel.
    
4. Reshapes the `(N_pixels, 6)` score matrix to `(6, height, width)` and stores it in `self._output`.

#### Visualization Lifecycle (`_export_file`)

Generates a structured $6 \times 2$ grid layout to support visual data analysis:

- **Left Column Panels:** Displays the six individual principal component vectors, reshaped back to the original image dimensions `(Height, Width)`.
    
- **Right Column Panels:** Plots the corresponding frequency histograms for each component, helping analysts inspect the distribution of variance across the different axes.

#### Return Value

Returns a floating-point `numpy.ndarray` of shape `(6, height, width)`. Each slice along the first axis is one principal-component image.

### Operational Implementation

```Python
from pathlib import Path
from fezrs.tools.pca import PCACalculator

# Initialize the spatial PCA transformation engine
pca_transformer = PCACalculator(
    red_path=Path("./landsat/LC08_B04_Red.tif"),
    green_path=Path("./landsat/LC08_B03_Green.tif"),
    blue_path=Path("./landsat/LC08_B02_Blue.tif"),
    nir_path=Path("./landsat/LC08_B05_NIR.tif"),
    swir1_path=Path("./landsat/LC08_B06_SWIR1.tif"),
    swir2_path=Path("./landsat/LC08_B07_SWIR2.tif")
)

# Run the SVD engine, construct diagnostic histograms, and save the 6-panel visualization layout
pca_transformer.execute(
    output_path="./exports/pca_analytics/",
    title="Spatial PCA Matrix Decomposition",
    dpi=500
)
```

## Analytical Performance Reference

The table below outlines the general characteristics and typical interpretation profiles of the resulting principal components:

|**Component**|**Target Variance Share**|**Spatial Contrast Profile**|**Primary Analytical Applications**|
|---|---|---|---|
|**PC1**|Typical $\ge 90\%$|High structural detail; behaves like a panchromatic brightness map.|Topographic mapping, shadow analysis, and baseline albedo feature extraction.|
|**PC2**|Typical $5\% - 8\%$|High contrast between visible light absorption and near-infrared ($NIR$) plateau.|Biomass delineation, vegetation health mapping, and land-cover classification.|
|**PC3**|Typical $1\% - 3\%$|Captures variations across infrared bands ($SWIR1 / SWIR2$).|Soil moisture profiling, surface water mapping, and mineral identification.|
|**PC4 – PC6**|Distributive $\le 1\%$|Low structural coherence; dominated by random high-frequency sensor patterns.|Noise filtering, data compression filtering, and system calibration diagnostics.|
