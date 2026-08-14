# SVM (Support Vector Machine)
## Overview

The `svm` module implements a Supervised Support Vector Machine (SVM) classification architecture designed for land-cover mapping and thematic feature extraction from multi-spectral satellite imagery. The module bridges machine learning workflows and interactive geospatial data engineering by providing an integrated graphical interface for manual training site selection.

Using an interactive OpenCV window, analysts select training coordinates directly on a live RGB preview of the image. The class extracts the underlying six-dimensional spectral profiles at those specific coordinates, trains an SVM model using a soft-margin Radial Basis Function (RBF) kernel, and classifies the remaining pixels across the entire spatial grid.

```
                   [6 Raster Bands Paths List]
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │     OpenCV Graphical UI Processing Frame     │ ──► Generates [0,1] normalized RGB canvas
         └──────────────────────┬───────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │  Interactive Coordinate Capture Subroutine   │ ──► Left-clicks sequentially register locations
         └──────────────────────┬───────────────────────┘     for $K$ classes $\times$ $N$ samples.
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │  Spectral Signature Matrix Ingestion ($X$)   │ ──► Pulls multi-spectral feature vectors:
         └──────────────────────┬───────────────────────┘     $\mathbf{x} \in \mathbb{R}^6$ per training pixel.
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │     Soft-Margin RBF Kernel Optimization      │ ──► Maps vectors implicitly into high-D space
         └──────────────────────┬───────────────────────┘     using adaptive scaling ($\gamma = \text{"scale"}$).
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │   One-vs-One Multi-Class Decision Matrix     │ ──► Resolves $K(K-1)/2$ pairwise tournaments 
         └──────────────────────┬───────────────────────┘     via majority voting logic.
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │        Global Spatial Transformation         │ ──► Classifies entire scene grid array;
         └──────────────────────────────────────────────┘     outputs categorized thematic map $(H, W)$.
```
## Comprehensive Mathematical Foundations

### Feature Representation Space

Each independent image pixel is treated as a distinct statistical sample in a six-dimensional spectral space. The feature vector $\mathbf{x}$ for a given coordinate is formed by stacking its normalized reflectance values from the six available bands:

$$\mathbf{x} = \begin{bmatrix} x_{\text{Red}} & x_{\text{Green}} & x_{\text{Blue}} & x_{\text{NIR}} & x_{\text{SWIR1}} & x_{\text{SWIR2}} \end{bmatrix}^T \in \mathbb{R}^6$$

The complete training array gathered via the user interface consists of $N_{\text{train}}$ examples:

$$\mathcal{D} = \left\{ (\mathbf{x}_i, y_i) \mid \mathbf{x}_i \in \mathbb{R}^6, \,\, y_i \in \{1, 2, \dots, K\} \right\}_{i=1}^{N_{\text{train}}}$$

Where $K$ represents the total number of target land-cover classes, and $N_{\text{train}} = K \times \text{sample\_number}$.

### The Binary Maximal Margin Classifier

For a simplified two-class scenario where labels are encoded as $y_i \in \{-1, +1\}$, the algorithm constructs a separating hyperplane defined by a weight vector $\mathbf{w}$ and a bias offset $b$:

$$\mathbf{w}^T \mathbf{x} + b = 0$$

The SVM maximizes the functional margin—the geometric distance from the splitting hyperplane to the closest training vectors (the **support vectors**)—by solving the following constrained quadratic optimization problem:

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1, \quad \forall i \in \{1, \dots, N_{\text{train}}\}$$

### Soft-Margin Formulations ($C$-SVM)

Real-world satellite observations are rarely perfectly separable in their raw spectral states due to mixed pixels, atmospheric variations, and overlapping land-cover signatures. To handle these non-separable distributions, the model introduces positive slack variables ($\xi_i \ge 0$) that allow controlled misclassifications during training:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N_{\text{train}}} \xi_i \quad \text{subject to} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

The regularization parameter $C > 0$ determines the balance between margin width and training error enforcement. A large value of $C$ penalizes misclassifications heavily, forcing a narrower margin focused on training accuracy, while a smaller $C$ allows more training errors to achieve a wider, more generalizable margin. The module uses a default setting of $C=1.0$.

### Non-Linear Mapping and the Radial Basis Function (RBF) Kernel

To resolve complex, non-linear boundaries between different land-cover types, the algorithm uses the **kernel trick**. This approach implicitly projects the raw six-dimensional feature vectors into an infinite-dimensional Hilbert space ($\Phi: \mathbb{R}^6 \to \mathcal{H}$), allowing the model to compute linear separations within this high-dimensional space. The resulting decision boundary is defined by:

$$f(\mathbf{x}) = \sum_{i \in \text{SV}} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b$$

Where $\alpha_i$ represents the calculated Lagrange multipliers. The system uses a non-linear **Radial Basis Function (RBF)** kernel:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

The kernel width parameter $\gamma$ controls the radius of influence for individual support vectors. The module configures this parameter using scikit-learn's adaptive `scale` heuristic:

$$\gamma = \frac{1}{n_{\text{features}} \times \text{Var}(X)} = \frac{1}{6 \times \text{Var}(X)}$$

Where $\text{Var}(X)$ is the total variance calculated across the training dataset matrix. This modification scales the kernel's distance sensitivity to match the overall spread of the training data.

### One-vs-One Multi-Class Strategy

Because Support Vector Machines are fundamentally binary classifiers, the system handles multi-class land-cover problems ($K \ge 3$) using a **One-vs-One (OvO)** multi-class reduction strategy.

The engine trains a total of $\frac{K(K-1)}{2}$ unique binary classifiers, where each model is optimized to separate a specific pair of classes. During the prediction step for an unlabelled pixel, all binary classifiers evaluate the feature vector, and each assigns its output to the winning class. A final majority-voting block counts these pairwise outcomes and assigns the pixel to the class with the most votes:

$$\hat{y} = \arg\max_{k \in \{1, \dots, K\}} \sum_{m=1}^{K(K-1)/2} \mathbb{I}\left(\text{Classifier}_m(\mathbf{x}) == k\right)$$

## Class Specification: `SVMCalculator`

### Operational Interface Parameters

#### Constructor Arguments (`__init__`)

- `red_path`, `green_path`, `blue_path` (`str` | `Path`): File paths to the visible bands, used to construct the interactive RGB selection canvas.
    
- `nir_path`, `swir1_path`, `swir2_path` (`str` | `Path`): File paths to the remaining infrared bands, providing the additional dimensions for the 6D feature space.
    
- `class_number` (`int`, default=`4`): Total number of discrete land-cover categories to classify ($K \ge 2$).
    
- `sample_number` (`int`, default=`10`): Number of training pixels to collect per class ($\ge 1$).

#### Interactive Sample Collection Workflow

1. Executing the module initializes an interactive OpenCV graphical canvas titled `"mouseClick"`, which displays a normalized RGB composite generated from the visible bands.
    
2. The user must click representative pixels for each target class in a strict, sequential order. The interface expects all samples for Class 1 first, followed by all samples for Class 2, and so on, continuing up to Class $K$.
    
3. Once the user records the total required number of clicks ($K \times \text{sample\_number}$), the interaction window closes automatically, and the pipeline starts training the SVM model.

#### Internal Data Validation Constraints (`_validate`)

- Verifies that `class_number >= 2` and `sample_number >= 1`.
    
- Confirms that the requested total number of training samples does not exceed the absolute pixel count of the input image.
    
- Checks if the training samples make up more than 5% of the total pixel population. If exceeded, it triggers an optimization warning to alert the user to the high manual workload.

#### Output State

Returns a 2D `numpy.ndarray` of shape `(Height, Width)` containing integer class labels ranging from `1` to `class_number`.

### Concrete Execution Example

```Python
from pathlib import Path
from fezrs.tools.svm import SVMCalculator

# Instantiate the interactive SVM classification engine
classifier = SVMCalculator(
    red_path=Path("./imagery/Landsat_B4.tif"),
    green_path=Path("./imagery/Landsat_B3.tif"),
    blue_path=Path("./imagery/Landsat_B2.tif"),
    nir_path=Path("./imagery/Landsat_B5.tif"),
    swir1_path=Path("./imagery/Landsat_B6.tif"),
    swir2_path=Path("./imagery/Landsat_B7.tif"),
    class_number=4,   # E.g., Class 1: Water, 2: Forest, 3: Urban, 4: Soil
    sample_number=12  # Collect 12 clicked pixel locations for each class
)

# Run the tool: this launches the GUI window, trains the model, and exports the final map
thematic_map = classifier.execute(
    output_path="./exports/classification/",
    title="SVM_Land_Cover_Map",
    colormap="tab10"
)
```

## Key Operational Considerations

- **Headless Display Dependencies:** Because the tool uses `cv2.imshow` for interactive pixel selection, it requires an active graphical windowing system. Running this tool on headless cloud instances, Docker containers, or Jupyter notebooks without configuring a virtual framebuffer (such as `Xvfb`) will cause a terminal application crash.
    
- **Strict Coordinate Input Order:** The matrix construction logic maps labels based on the exact time sequence of user clicks. The first block of clicks is assigned to Class 1, the second to Class 2, and so on. If the user clicks targets out of order, the training dataset will contain incorrect labels, leading to flawed classification results.
    
- **Feature Scaling Profiles:** While input bands are normalized to a standard $[0.0, 1.0]$ range, individual channels often retain significantly different underlying variances. Because SVM optimization is sensitive to scale variations across its input features, implementing an explicit standardization step can help improve overall classification accuracy.
