# GLCM (Gray Level Co-occurrence Matrix)
## Overview

The `glcm` module implements second-order spatial statistical architectures designed to quantify, analyze, and map land-surface textures using the **Gray-Level Co-occurrence Matrix (GLCM)** and Haralick texture features. In multi-spectral remote sensing, relying solely on spectral reflectance parameters often fails to differentiate structurally distinct land covers that share overlapping spectral signatures—such as separating complex urban environments from highly reflective bare soils, or uniform natural grasslands from industrial row-crop agriculture.

This module addresses this limitation by processing the spatial arrangement and frequency of localized tone distributions. It extracts structural indices that capture surface roughness, homogeneity, and directional lineaments, producing continuous-valued thematic maps optimized for advanced image segmentation and land-cover classification.

```
                  fezrs.base.BaseTool [Base Architecture]
                            │
                            ▼
          ┌───────────────────────────────────┐
          │     fezrs.tools.glcm Module       │
          └─────────────────┬─────────────────┘
                            │
                            ▼
                     GLCMCalculator
                            │
     ┌──────────────────────┴──────────────────────┐
     ▼                                             ▼
[Spatial Sliding Window Engine]          [scikit-image Backend]
  ├─ 8-Bit Quantization Matrix Mapping     ├─ skimage.feature.graycomatrix
  └─ Output Grid Assembly Engine           └─ skimage.feature.graycoprops
```

## Comprehensive Class Specification: `GLCMCalculator`

### Scientific and Mathematical Objective

The objective of `GLCMCalculator` is to compute a local second-order joint probability distribution matrix within a spatial sliding window across a single raster band (such as Near-Infrared). It then extracts specific scalar Haralick texture metrics for each window position, mapping the structural characteristics of the surface to a continuous-valued spatial array.

### Mathematical Foundations of the GLCM

A standard first-order histogram describes the global or local frequency of independent gray levels but discards all spatial relationships. The GLCM, by contrast, is a second-order statistical matrix that tracks the frequency with which pairs of pixels with specific gray levels occur at a defined spatial displacement vector ($\mathbf{d}$).

#### Unnormalized Co-occurrence Formulation

Let an input image array $I$ contain quantized integer gray levels bounded within the range:

$$\mathcal{G} = \{0, 1, 2, \dots, G-1\}$$

Where $G$ represents the total number of gray levels (for standard 8-bit unsigned integer data, $G = 256$). A spatial offset vector is defined by its coordinate displacements:

$$\mathbf{d} = (d_x, d_y)$$

The unnormalized GLCM matrix $P(i, j \mid \mathbf{d})$ is a square matrix of dimensions $G \times G$. Each cell element $(i, j)$ stores the absolute frequency of pixel pairs that match gray levels $i$ and $j$ while separated by the displacement vector $\mathbf{d}$:

$$P(i, j \mid \mathbf{d}) = \sum_{x} \sum_{y} \begin{cases} 1, & \text{if } I(x, y) = i \text{ and } I(x + d_x, y + d_y) = j \\ 0, & \text{otherwise} \end{cases}$$

In this calculator's implementation, the displacement parameter is hardcoded to an absolute spatial distance of $1$ pixel along a strictly horizontal trajectory ($\theta = 0^\circ$). The operational displacement vector reduces to:

$$\mathbf{d} = (0, 1) \quad \text{(one pixel immediately to the right)}$$

#### Symmetry Stabilization

To treat pixel pair relationships as undirected spatial interactions, the module applies a symmetry transformation (`symmetric=True`). This incorporates the reverse spatial relationship by summing the unnormalized matrix with its transpose:

$$P_{\text{sym}}(i, j \mid \mathbf{d}) = P(i, j \mid \mathbf{d}) + P^T(i, j \mid \mathbf{d})$$

This operation ensures that the structural frequency count for the pair $(i, j)$ is identical to $(j, i)$, establishing an undirected analytical baseline.

#### Joint Probability Density Normalization

To transform absolute frequency counts into scale-invariant joint probabilities, the symmetric matrix is normalized by the sum of all internal elements (`normed=True`):

$$p(i, j) = \frac{P_{\text{sym}}(i, j)}{\sum_{a=0}^{G-1} \sum_{b=0}^{G-1} P_{\text{sym}}(a, b)}$$

Each element $p(i, j)$ represents the statistical joint probability that a pixel pair separated by vector $\mathbf{d}$ within the local sliding window contains the gray-level values $i$ and $j$.

### Mathematical Formulations of Haralick Texture Features

The module extracts six distinct scalar indices from the normalized symmetric joint probability matrix $p(i, j)$:

#### Contrast

$$\text{Contrast} = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} (i - j)^2 \cdot p(i, j)$$

- **Physical Interpretation:** Measures the local intensity variance and structural sharpness. The squared difference term $(i - j)^2$ acts as a quadratic weight that penalizes gray-level divergence away from the main diagonal of the matrix. Elevated outputs reveal sharp intensity transitions, indicating rough textures, complex geological lineaments, or dense urban structures. Smooth, uniform features (such as calm water or uniform sand) yield values near zero.

#### Dissimilarity

$$\text{Dissimilarity} = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} |i - j| \cdot p(i, j)$$

- **Physical Interpretation:** Similar to Contrast, Dissimilarity tracks localized surface roughness. However, it applies a linear absolute weight $|i - j|$ rather than a quadratic penalty. This makes it less sensitive to extreme, isolated radiometric outliers, providing a balanced measure of macro-texture roughness over highly dynamic landscapes.

#### Homogeneity (Inverse Difference Moment)

$$\text{Homogeneity} = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} \frac{p(i, j)}{1 + (i - j)^2}$$

- **Physical Interpretation:** Quantifies how closely pixel pairs concentrate along the main diagonal of the GLCM. The inverse weighting function $\frac{1}{1 + (i - j)^2}$ approaches its maximum value ($1.0$) when $i \approx j$, which occurs in regions with minimal spatial variance. High outputs indicate uniform, smooth surfaces (such as water bodies, continuous bare soil, or uniform crop leaves), while highly textured urban or forest canopies yield low values.

#### Angular Second Moment (ASM / Uniformity)

$$\text{ASM} = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} \big(p(i, j)\big)^2$$

- **Physical Interpretation:** Measures the structural orderliness and textural uniformity of the local neighborhood. When a spatial window contains highly repetitive, uniform patterns, the joint probabilities concentrate within a few specific gray-level pairs, producing high ASM values. If the texture is random or structurally complex, the probabilities distribute widely across the matrix, driving the sum of squares down.

#### Energy

$$\text{Energy} = \sqrt{\text{ASM}} = \sqrt{\sum_{i=0}^{G-1} \sum_{j=0}^{G-1} \big(p(i, j)\big)^2}$$

- **Physical Interpretation:** Computes the square root of the Angular Second Moment, transforming the metric into a linear scale that is often preferred for data visualization. High Energy indicates highly organized spatial patterns, such as regular row crops, orchard configurations, or gridded urban networks.

#### Correlation

$$\text{Correlation} = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} \frac{(i - \mu_i)(j - \mu_j) \cdot p(i, j)}{\sigma_i \sigma_j}$$

Where the marginal means ($\mu_i, \mu_j$) and standard deviations ($\sigma_i, \sigma_j$) along rows and columns are defined as:

$$\mu_i = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} i \cdot p(i, j) \quad \text{and} \quad \mu_j = \sum_{i=0}^{G-1} \sum_{j=0}^{G-1} j \cdot p(i, j)$$

$$\sigma_i = \sqrt{\sum_{i=0}^{G-1} \sum_{j=0}^{G-1} (i - \mu_i)^2 \cdot p(i, j)} \quad \text{and} \quad \sigma_j = \sqrt{\sum_{i=0}^{G-1} \sum_{j=0}^{G-1} (j - \mu_j)^2 \cdot p(i, j)}$$

- **Physical Interpretation:** Evaluates the linear dependence of gray levels between neighboring pixels separated by the displacement vector $\mathbf{d}$. Bounded within the range $[-1.0, +1.0]$, high positive values indicate strong linear predictability (e.g., bright pixels consistently adjacent to other bright pixels, typical of smooth terrain gradients). Low or negative values indicate complex or random textures. If the local window is completely flat ($\sigma_i \sigma_j = 0$), the denominator evaluates to zero; the implementation catches this edge case to prevent runtime exceptions.

### Processing Workflow and Spatial Layout Mechanics

```
  Input Image (H x W)                      Sliding Analysis Window             Output Raster Construction
┌──────────────────────────────┐          ┌───────────────────┐               ┌──────────────────────────────┐
│                              │          │ x ──► Stride = 1  │               │Full Feature Map (H x W)      │
│                              │          │ │                 │               │ ┌──────────────────────────┐ │
│                              │  ───►    │ ▼                 │       ───►    │ │ All pixels computed      │ │
│                              │          │ Window Size (W)   │               │ │ Right/bottom windows     │ │
│                              │          └───────────────────┘               │ │ are truncated to bounds  │ │
│                              │          Computes GLCM via                   │ └──────────────────────────┘ │
└──────────────────────────────┘          skimage per step                    └──────────────────────────────┘
```

1. **Quantization Baseline:** The pipeline extracts the single-band raster array (typically the Near-Infrared band) and casts it to an 8-bit unsigned integer array (`uint8`).
    
2. **Sliding Window Trajectory:** A square window of user-defined size $W \times W$ slides across the image grid with a horizontal and vertical stride of 1 pixel. The window is anchored at the current pixel $(i, j)$ and extends down and to the right.
    
3. **Array Extraction:** For each window position, the local GLCM is extracted, normalized, and evaluated across the selected Haralick property configuration.
    
4. **Output Matrix Offset:** The resulting scalar texture metric is written to the upper-left coordinate index $[i, j]$ of the active window position within the output array.
    
5. **Border Behavior:** Every output pixel is computed. Near the right and bottom edges the requested $W \times W$ window extends past the image, so NumPy slicing clips it to the remaining pixels (a truncated window). There is no padding, and border pixels are not left uninitialized. Top and left pixels still have a full window because the window extends inward from $(i, j)$.

### Interface Architecture

#### Constructor Method Input Arguments (`__init__`)

- `nir_path` (`str` | `Path`): File location pointing to the single-band target raster (Near-Infrared recommended).
    
- `window_size` (`int`): Dimension of the square local analysis window. Must be an odd integer satisfying:


$$\text{window\_size} \ge 3$$

- `propery` (`str`): Target Haralick feature name selection. Must match one of the following strings:
    
    - `"contrast"`, `"dissimilarity"`, `"homogeneity"`, `"ASM"`, `"energy"`, `"correlation"`.

#### Operational Validation (`_validate`)

The programmatic `_validate()` method enforces runtime constraints prior to code execution:

1. Verifies that `window_size` is an integer and an odd value greater than or equal to 3.
    
2. Checks that `propery` is a valid string matching one of the six supported Haralick feature types (`contrast`, `dissimilarity`, `homogeneity`, `ASM`, `energy`, `correlation`).
    
3. Confirms that the target single-band file path exists and is readable.

#### Return State (`process()`)

Returns a 2D `numpy.ndarray` with the same `(height, width)` shape as the input image. Every pixel is assigned a texture value. Pixels whose $W \times W$ window would extend past the right or bottom edge are computed from the truncated in-bounds window rather than from padded or missing data.

#### Operational Implementation

```Python
from pathlib import Path
from fezrs.tools.glcm import GLCMCalculator

# Initialize second-order statistical texture evaluation engine
texture_engine = GLCMCalculator(
    nir_path=Path("./data/Landsat8_NIR.tif"),
    window_size=5,
    propery="homogeneity"
)

# Run texture pipeline and save output map
texture_engine.execute(
    output_path="./exports/texture_mapping/",
    title="GLCM 5x5 Neighborhood Homogeneity Matrix",
    colormap="plasma",
    show_colorbar=True,
    dpi=500
)
```

## Reference Summary of Haralick Texture Features

|**Feature Metric**|**Core Mathematical Weight**|**Analytical Behavior Profile**|**Primary Target Applications**|
|---|---|---|---|
|**Contrast**|$(i - j)^2$|Escalates quadratically with local gray-level divergence; tracks high-frequency roughness.|Delineation of urban centers, structural fault lines, and highly fragmented edge networks.|
|**Dissimilarity**|$|i - j|$|
|**Homogeneity**|$\frac{1}{1 + (i - j)^2}$|Maximize output when neighboring pixel values match; tracks low-frequency smoothness.|Identification of calm water bodies, open desert soils, and uniform agricultural fields.|
|**ASM**|$\big(p(i, j)\big)^2$|Increases as joint probabilities concentrate within few cells; tracks organized order.|Detections of industrial row-crop configurations, commercial orchards, or gridded urban networks.|
|**Energy**|$\sqrt{\text{ASM}}$|Linear scaling of texture uniformity metrics; offers high contrast across linear visual scales.|High-fidelity visualization of regular, repetitive landscape geometries.|
|**Correlation**|$\frac{(i - \mu_i)(j - \mu_j)}{\sigma_i \sigma_j}$|Measures linear predictability along the displacement trajectory; scales from $-1.0$ to $+1.0$.|Mapping directional features such as linear sand dunes, continuous roads, and structural faults.|
