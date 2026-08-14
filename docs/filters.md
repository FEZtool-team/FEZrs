# Filters
## Overview

The `filters` module provides a comprehensive suite of digital image processing operators designed for radiometric enhancement, high-frequency noise suppression, and directional/isotropic structural feature extraction. These tools serve as critical pre-processing components within remote sensing workflows, stabilizing spatial matrices prior to executing higher-level analytical pipelines such as automated lineament mapping, land-cover classification, texture metrics estimation, and object-based image analysis (OBIA).

```
                    fezrs.base.BaseTool [Base Architecture]
                              │
                              ▼
        ┌─────────────────────┴─────────────────────┐
        │        fezrs.tools.filters Module         │
        └─────────────────────┬─────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
[Linear Shift-Invariant Filters]     [Non-Linear Statistical Filters]
├─ MeanCalculator                    └─ MedianCalculator
├─ GuassianCalculator
├─ SobelCalculator
└─ LaplacianCalculator
```

## Mathematical Foundations of Spatial Filtering

### The Discrete Convolution Operation

Linear filters operate across spatial domains via discrete 2D spatial convolution. Given a continuous discrete image matrix $I$ and a localized bounding window known as a kernel $K$ of odd-integer dimensional extents $(2a+1) \times (2a+1)$, the mathematically modified output coordinate $I'(x, y)$ is calculated via:

$$I'(x, y) = \sum_{i=-a}^{a} \sum_{j=-a}^{a} K(i, j) \cdot I(x + i, y + j)$$

The kernel matrix systematically slides across every coordinate index of the target raster. At each coordinate position, the scalar dot product of the kernel weights and the overlapping pixel values is computed, updating the center pixel's value.

```
       Overlapping Image Window                    Convolution Kernel
     ┌──────────┬──────────┬──────────┐       ┌──────────┬──────────┬──────────┐
     │I(x-1,y-1)│ I(x,y-1) │I(x+1,y-1)│       │  K(-1,-1)│  K(0,-1) │  K(1,-1) │
     ├──────────┼──────────┼──────────┤       ├──────────┼──────────┼──────────┤
     │ I(x-1,y) │  I(x,y)  │ I(x+1,y) │   X   │  K(-1,0) │  K(0,0)  │  K(1,0)  │
     ├──────────┼──────────┼──────────┤       ├──────────┼──────────┼──────────┤
     │I(x-1,y+1)│ I(x,y+1) │I(x+1,y+1)│       │  K(-1,1) │  K(0,1)  │  K(1,1)  │
     └──────────┴──────────┴──────────┘       └──────────┴──────────┴──────────┘
                                      │
                                      ▼
             Sum of Element-by-Element Products = Target Pixel I'(x,y)
```

### Boundary Condition Management

When the spatial kernel kernel overlaps the outer perimeter of an image, its dimensions extend beyond the raster bounds. The module delegates boundary extrapolation to OpenCV's optimized infrastructure using reflection padding (`BORDER_REFLECT_101` topology by default):

$$I(-x, y) = I(x, y) \quad \text{and} \quad I(Width + x, y) = I(Width - x, y)$$

This approach minimizes edge artifacts and prevents artificial gradient boundaries along the outer edges of the computed raster matrix.

## Comprehensive Class Specifications

### GuassianCalculator` — Isotropic Low-Pass Smoothing

#### Scientific & Physical Objective

The algorithmic goal of `GuassianCalculator` is to apply an isotropic low-pass filter to remove continuous high-frequency background noise (such as thermal electronic sensor noise or atmospheric scattering). This smoothing step helps maintain structural boundaries and edge locations more effectively than a standard unweighted box-average filter.

#### Theoretical Foundation & Mathematical Formulations

The spatial distribution weights of the filter are governed by the 2D isotropic Gaussian distribution equation:

$$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

Where:

- $x$ and $y$ represent the absolute spatial coordinate offsets from the center point of the kernel matrix.
    
- $\sigma$ represents the standard deviation of the Gaussian distribution, controlling the width of the bell curve and governing the level of spatial smoothing.

Within this calculator's initialization pipeline, the kernel window size is fixed at an odd boundary constraint of $13 \times 13$ pixels. By setting the explicit standard deviation parameter to zero ($\sigma = 0$), the underlying engine automatically calculates the optimal standard deviation using the following linear function of kernel width:

$$\sigma = 0.3 \cdot \left( \frac{\text{Kernel Size} - 1}{2} - 1 \right) + 0.8$$

For a hardcoded $13 \times 13$ window footprint, this optimization resolves to:

$$\sigma = 0.3 \cdot \left( \frac{13 - 1}{2} - 1 \right) + 0.8 = 0.3 \cdot (6 - 1) + 0.8 = 2.3$$

The weights within the resulting discrete kernel matrix are normalized to ensure their sum equals exactly one:

$$\sum_{i=-a}^{a} \sum_{j=-a}^{a} K(i, j) = 1.0$$

This normalization step preserves the global radiometric scale and average brightness of the input scene.

#### Frequency Domain Behavior

In the frequency domain, the Fourier transform of a Gaussian kernel is itself a Gaussian distribution. This ensures smooth attenuation of high-frequency components without introducing the phase-reversal or "ringing" artifacts typical of sharp cut-off box-car filters.

#### Interface Architecture

- **Constructor Method (`__init__`) Input Arguments:**
    
    - `tif_path` (`str` | `Path`): File location pointing to the single-band target raster.
    
- **Return State (`process()`):** Returns a smoothed 2D `numpy.ndarray` floating-point array with high-frequency noise suppressed.

#### Operational Implementation

```Python
from pathlib import Path
from fezrs.tools.filters import GuassianCalculator

# Initialize low-pass isotropic Gaussian engine
gaussian_blur = GuassianCalculator(
    tif_path=Path("./data/Landsat8_Band5.tif")
)

# Execute smoothing pipeline and export result
gaussian_blur.execute(
    output_path="./exports/filtered/",
    title="Isotropic Gaussian Low-Pass Blur",
    dpi=500
)
```

### `LaplacianCalculator` — Isotropic High-Pass Edge Detection

#### Scientific & Physical Objective

The primary objective of `LaplacianCalculator` is to compute the second-order spatial derivative of an image. This highlights high-frequency spatial transitions and isolates edge networks across all orientations.

#### Theoretical Foundation & Mathematical Formulations

For a continuous 2D intensity function $f(x, y)$, the Laplacian operator ($\nabla^2$) is defined as the sum of the unmixed second partial spatial derivatives:

$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

In the discrete pixel domain, this second-order derivative is approximated using finite central differences. For a standard $3 \times 3$ spatial window, the finite difference approximation resolves to the following default structural matrix kernel:

$$K_{L3} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

Or it can be configured as the more comprehensive eight-neighbor Laplacian variant:

$$K_{L8} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & -8 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$

Notice that the sum of all coefficients in these Laplacian kernels equals exactly zero:

$$\sum K_{L} = 0$$

Consequently, when the kernel processes homogeneous areas with uniform pixel intensities, the output evaluates to zero. In regions with sharp intensity shifts, the second-order derivative produces a strong response.

```
      Intensity Profile                        First Derivative                         Second Derivative (Laplacian)
      
          ┌─────────                           │      ▲                                      ▲    
          │                                    │      │                                 ─────┼─────
          │                                    │      │                                      │    ▼
   ───────┘                             ───────┴──────┴──────                         ───────┴──────
   [Step Edge Transition]                   [Gradient Peak]                              [Zero-Crossing Edge Point]
```

#### Analytical Target Interpretation

- **Zero-Crossings:** The exact location of a structural edge occurs where the Laplacian signal crosses zero.
    
- **Peak Magnitude:** The amplitude of the response is proportional to the sharpness and steepness of the spatial intensity transition.

To preserve negative gradient responses without truncation, the pipeline sets the destination data depth to match the input layer (`ddepth=-1`).

#### Interface Architecture

- **Constructor Method (`__init__`) Input Arguments:**
    
    - `tif_path` (`str` | `Path`): File location pointing to the single-band target raster.
        
    - `kernel_size` (`int`): Bounding odd integer dimension parameter (e.g., $3, 5, 7$).
    
- **Return State (`process()`):** Returns a 2D `numpy.ndarray` array capturing high-pass isotropic second-derivative edge structures.

#### Operational Implementation

```Python
from pathlib import Path
from fezrs.tools.filters import LaplacianCalculator

# Initialize second-derivative isotropic edge detector
laplacian_edge = LaplacianCalculator(
    tif_path=Path("./data/Geom_Features.tif"),
    kernel_size=5
)

# Run process and save sharp edge boundaries
laplacian_edge.execute(
    output_path="./exports/filtered/",
    title="Laplacian Isotropic Edge Detection",
    colormap="gray",
    dpi=500
)
```

### MeanCalculator` — Uniform Neighborhood Smoothing

#### Scientific & Physical Objective

`MeanCalculator` acts as a fast, localized low-pass filter that smooths spatial variance by averaging pixel intensities within a uniform neighborhood window.

#### Theoretical Foundation & Mathematical Formulations

The mean filter replaces the center pixel value with the unweighted arithmetic mean of all digital numbers enclosed within a localized window $W$ of dimensions $k \times k$. The corresponding convolution kernel assigns an equal, uniform weight to all coefficient positions:

$$K_{\text{mean}}(i, j) = \frac{1}{k^2} \quad \forall \quad i, j \in W$$

Within this module's implementation, the spatial footprint is fixed at a $9 \times 9$ pixel layout. This creates an unweighted box-filter kernel where each individual entry is defined as:

$$K_{\text{mean}} = \frac{1}{81}$$

The discrete spatial operation for each coordinate center point evaluates to:

$$I'(x, y) = \frac{1}{81} \sum_{(u, v) \in W} I(u, v)$$

#### Analytical Trade-Offs

While the mean filter effectively minimizes random Gaussian noise variance across uniform regions, its unweighted spatial averaging blurs valid structural lineaments and high-frequency boundaries. Additionally, it is highly sensitive to outlier values, meaning impulse noise (such as salt-and-pepper pixels) can significantly distort the local mean.

#### Interface Architecture

- **Constructor Method (`__init__`) Input Arguments:**
    
    - `tif_path` (`str` | `Path`): File location pointing to the single-band target raster.
    
- **Return State (`process()`):** Returns a uniformly smoothed low-pass 2D `numpy.ndarray`.


#### Operational Implementation

```Python
from pathlib import Path
from fezrs.tools.filters import MeanCalculator

# Initialize uniform box-car low-pass filter
mean_blur = MeanCalculator(
    tif_path=Path("./data/Radiometric_Input.tif")
)

# Export the averaged image matrix
mean_blur.execute(
    output_path="./exports/filtered/",
    title="Uniform Box-Car Mean Filter",
    dpi=500
)
```

### MedianCalculator` — Rank-Order Non-Linear Denoising

#### Scientific & Physical Objective

`MedianCalculator` is a non-linear, rank-order statistical filter designed to eliminate impulse noise (salt-and-pepper artifacts) while preserving sharp structural boundaries and edge transitions.

#### Theoretical Foundation & Mathematical Formulations

Unlike linear convolutional filters, the median filter does not use a weighted scalar dot product. Instead, it analyzes the neighborhood window $W$ of size $k \times k$ centered at coordinates $(x, y)$, extracts all raw pixel values, sorts them in ascending numerical order, and assigns the exact middle value to the target pixel:

$$I'(x, y) = \text{median} \left\{ I(u, v) \mid (u, v) \in W \right\}$$

For a user-defined kernel size of $5$ ($k=5$), the localized window contains 25 independent pixels. The values are ordered sequentially, and the 13th element is selected as the median output.

```
       1D Sorted Rank List
       ┌────┬────┬────┬──────────────────┬────┬────┬──── Dinoised Output
       │ 12 │ 14 │ 15 │ ...  [ 74 ] ...  │ 98 │ 99 │255 ──► Center Pixel = 74
       └────┴────┴────┴──────────────────┴────┴────┴────
                               ▲
                    Exact 50th Percentile Rank
```

#### Impulse Noise Mitigation Mechanics

Extreme anomalous outliers caused by transmission loss or sensor calibration errors manifest as maximum or minimum values (e.g., $0$ or $255$). Because the sorting operation places these anomalies at the extreme ends of the ranked array, they are excluded from the selection process.

As long as the noise artifacts occupy less than half of the total window area ($< 50\%$ spatial density), they are completely filtered out. Concurrently, valid high-contrast step edges are preserved without blurring because the median shift tracks the structural boundary once it covers the majority of pixels within the sliding window.

#### Interface Architecture

- **Constructor Method (`__init__`) Input Arguments:**
    
    - `tif_path` (`str` | `Path`): File location pointing to the single-band target raster.
        
    - `kernel_size` (`int`): Bounding odd integer dimension parameter (e.g., $3, 5, 7$).
    
- **Return State (`process()`):** Returns an impulse-denoised, edge-preserving 2D `numpy.ndarray`.

#### Operational Implementation

```Python
from pathlib import Path
from fezrs.tools.filters import MedianCalculator

# Initialize rank-order non-linear denoising engine
median_denoise = MedianCalculator(
    tif_path=Path("./data/Impulse_Noise_Band.tif"),
    kernel_size=5
)

# Execute denoising process and export result
median_denoise.execute(
    output_path="./exports/filtered/",
    title="Rank-Order Median Edge-Preserving Denoise",
    dpi=500
)
```

### `SobelCalculator` — First-Order Directional Gradient Estimation

#### Scientific & Physical Objective

`SobelCalculator` computes a first-order directional spatial derivative to approximate the intensity gradient across an image matrix, emphasizing structural lineaments and edge boundaries.

#### Theoretical Foundation & Mathematical Formulations

The spatial intensity gradient of a continuous image surface $I$ is defined as a 2D vector field pointing in the direction of maximum intensity change:

$$\nabla I = \left[ \frac{\partial I}{\partial x}, \frac{\partial I}{\partial y} \right]^T = [G_x, G_y]^T$$

To evaluate these directional changes, the Sobel operator applies two separate $3 \times 3$ convolutional kernels for horizontal ($G_x$) and vertical ($G_y$) derivative approximations:

$$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} \quad \text{and} \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}$$

The $G_x$ kernel isolates vertical edge structures by measuring intensity differences across columns, while the $G_y$ kernel captures horizontal features by measuring changes across rows.

In this implementation, the configuration uses concurrent directional tracking (`dx=1`, `dy=1`). The total edge magnitude approximation combines both directional gradient arrays:

$$\text{Output} \approx |G_x| + |G_y|$$

#### Noise Mitigation Layout

The Sobel kernel design incorporates a localized smoothing mechanism perpendicular to the derivative direction (for example, the center row/column weights are scaled by a factor of 2). This localized averaging makes the Sobel operator less sensitive to high-frequency pixel noise than simple central difference operators.

#### Interface Architecture

- **Constructor Method (`__init__`) Input Arguments:**
    
    - `tif_path` (`str` | `Path`): File location pointing to the single-band target raster.
        
    - `kernel_size` (`int`): Bounding odd integer dimension parameter (must be $3, 5, \text{or } 7$).
    
- **Return State (`process()`):** Returns a 2D `numpy.ndarray` gradient magnitude map highlighting structural edge ridges.

#### Operational Implementation

```Python
from pathlib import Path
from fezrs.tools.filters import SobelCalculator

# Initialize first-order directional gradient engine
sobel_gradient = SobelCalculator(
    tif_path=Path("./data/Topography_DEM.tif"),
    kernel_size=3
)

# Execute gradient extraction and save result
sobel_gradient.execute(
    output_path="./exports/filtered/",
    title="Sobel First-Order Gradient Magnitude Map",
    colormap="magma",
    dpi=500
)
```
