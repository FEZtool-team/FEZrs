# SVM (Support Vector Machine)
## Overview

The `svm` module implements a Supervised Support Vector Machine (SVM) classification architecture designed for land-cover mapping and thematic feature extraction from multi-spectral satellite imagery.

Training locations can be supplied programmatically as `(row, col, class_id)` triples (`training_samples`). That is the reproducible, scriptable path. When they are omitted, an OpenCV window collects clicks on a live RGB preview. Either way, the class reads the six **raw loaded band values** at those coordinates, trains `sklearn.svm.SVC` with a soft-margin Radial Basis Function (RBF) kernel (`gamma="scale"`, $C=1.0$), and classifies every pixel in the scene.

```
                   [6 Raster Bands Paths List]
                                │
              ┌─────────────────┴──────────────────┐
              ▼                                    ▼
 [training_samples triples]              [OpenCV RGB preview]
  (row, col, class_id)                   display-only [0,1] canvas;
  skips the GUI entirely                 left-clicks if no samples
              └─────────────────┬──────────────────┘
                                ▼
         ┌──────────────────────────────────────────────┐
         │  Spectral Signature Matrix Ingestion ($X$)   │ ──► Raw loaded band values:
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

Each independent image pixel is treated as a distinct statistical sample in a six-dimensional spectral space. The feature vector $\mathbf{x}$ is stacked from the **values as loaded from disk** (`skimage.io.imread(...).astype(float)` through `FileHandler.get_images_collection()`). **No per-band min–max rescale and no digital-number-to-reflectance conversion is applied to the classifier features.**

The six bands `SVMCalculator` always receives, in `FileHandler.band_paths` insertion order:

$$\mathbf{x} = \begin{bmatrix} x_{\text{Red}} & x_{\text{NIR}} & x_{\text{Blue}} & x_{\text{SWIR1}} & x_{\text{SWIR2}} & x_{\text{Green}} \end{bmatrix}^T \in \mathbb{R}^6$$

That is the column layout of `classifier_.support_vectors_`. The RBF kernel depends only on Euclidean distance, so permuting the axes does not change the predicted map.

The optional OpenCV window min–max normalizes **only the RGB preview** to $[0, 1]$ so the composite is displayable. That preview scaling is not reused for training or prediction. If the files already hold reflectance (or any other scaled product), those are the values the SVM sees; if they hold raw digital numbers, those are the values the SVM sees. FEZrs does not read scale/offset metadata.

The complete training array consists of $N_{\text{train}}$ examples:

$$\mathcal{D} = \left\{ (\mathbf{x}_i, y_i) \mid \mathbf{x}_i \in \mathbb{R}^6, \,\, y_i \in \{1, 2, \dots, K\} \right\}_{i=1}^{N_{\text{train}}}$$

Where $K$ is the number of land-cover classes. On the interactive path $N_{\text{train}} = K \times \text{sample\_number}$. On the programmatic path $N_{\text{train}}$ is the length of `training_samples`.

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
    
- `sample_number` (`int`, default=`10`): Number of training pixels to collect per class ($\ge 1$). Applies to the interactive path only.

- `training_samples` (`Sequence[tuple] | None`, default `None`): Training locations as `(row, col, class_id)` triples. **When supplied, the GUI is skipped entirely.** This is the reproducible, scriptable and testable path, and the one to use for anything that has to be repeated or published.

- `transform` (affine | `None`, default `None`): An affine transform — a `rasterio` `Affine`, or anything supporting `~transform * (x, y)` — used to interpret `training_samples` as **map coordinates** `(northing, easting)` in the raster's CRS rather than array indices. Training sites digitised over a basemap in a GIS carry coordinates, not row/column indices.

- `evaluate` (`bool`, default `False`), `test_size` (`float`, default `0.3`), `random_state` (`int`, default `0`): Hold out a stratified test split and report accuracy. See Accuracy Assessment below.

#### Reproducible Sample Collection

```Python
from fezrs import SVMCalculator

classifier = SVMCalculator(
    red_path="red.tif", green_path="green.tif", blue_path="blue.tif",
    nir_path="nir.tif", swir1_path="swir_1.tif", swir2_path="swir_2.tif",
    training_samples=[
        (120, 340, 1), (122, 345, 1),   # water
        (400, 210, 2), (405, 215, 2),   # forest
    ],
    evaluate=True,
)
classifier.execute(output_path="./exports/classification/")

print(classifier.accuracy_, classifier.kappa_)
print(classifier.confusion_matrix_)
```

With map coordinates from a GIS:

```Python
import rasterio

with rasterio.open("nir.tif") as source:
    transform = source.transform

SVMCalculator(
    ...,
    training_samples=[(4176615.0, 320325.0, 1), ...],   # (northing, easting, class)
    transform=transform,
)
```

#### Accuracy Assessment

Setting `evaluate=True` holds out a stratified `test_size` fraction, fits on the remainder, and populates:

- `accuracy_` — overall accuracy on the held-out split.
- `kappa_` — Cohen's $\kappa$, the standard agreement measure in the remote sensing literature, which corrects for agreement expected by chance.
- `confusion_matrix_` — the full class-by-class error matrix, from which producer's and user's accuracies can be derived.

`random_state` seeds the split, so a reported accuracy is reproducible. The final classifier is always fitted on the **full** training set; the split is used only for the estimate.

#### Interactive Sample Collection Workflow

Used only when `training_samples` is not supplied.

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

- **Reproducibility:** The interactive path cannot be reproduced. The training set depends on where the operator clicked, and there is no seed, no sample file and no record of which pixels were used, so two runs by two people produce different classifiers and neither can be repeated. **Pass `training_samples` for any result that has to be repeated, compared or published.**

- **Headless Display Dependencies:** Interactive selection uses `cv2.imshow` and needs an active windowing system. Without one, Qt **aborts the process** rather than raising, so a caller cannot catch it and degrade gracefully. FEZrs checks for a display first and raises a catchable `RuntimeError` naming `training_samples` as the fix. Supplying `training_samples` skips the GUI entirely, so cloud instances, containers and notebooks need no virtual framebuffer.

- **Interrupted Collection:** Closing the window with ESC before every sample is collected raises a `RuntimeError` stating how many samples were gathered. Previously `process()` returned `None`, leaving `_output` unset, and the failure surfaced later as `ValueError: Data not computed.` from the export step.

- **Strict Coordinate Input Order:** In the interactive path the matrix construction logic maps labels based on the exact time sequence of user clicks. The first block of clicks is assigned to Class 1, the second to Class 2, and so on. If the user clicks targets out of order, the training dataset will contain incorrect labels, leading to flawed classification results. `training_samples` carries an explicit `class_id` per location and is immune to this.
    
- **Feature scaling is the caller's responsibility.** Training and prediction use the raw loaded bands. An RBF kernel with $\gamma = \text{"scale"}$ is sensitive to the relative range of each axis: a 16-bit SWIR band sitting next to an 8-bit visible band will dominate the distance. Convert to reflectance, or otherwise scale the files, *before* passing them in if that is what the analysis requires. The RGB preview normalization is display-only and does not leak into the feature matrix.
