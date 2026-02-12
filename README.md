# **FEZrs**    

[![DOI](https://zenodo.org/badge/710286874.svg)](https://doi.org/10.5281/zenodo.14938038) ![Downloads](https://static.pepy.tech/badge/FEZrs) ![PyPI](https://img.shields.io/pypi/v/FEZrs?color=blue&label=PyPI&logo=pypi) [![Conda Version](https://img.shields.io/conda/vn/FEZtool/fezrs?label=Anaconda&color=orange&logo=anaconda)](https://anaconda.org/FEZtool/fezrs) ![License](https://img.shields.io/pypi/l/FEZrs) [![PyPI Downloads](https://static.pepy.tech/badge/fezrs)](https://pepy.tech/projects/fezrs) ![GitHub last commit](https://img.shields.io/github/last-commit/FEZtool-team/fezrs) [![Platform](https://img.shields.io/conda/pn/feztool/fezrs?color=blue&label=Platform&style=flat)](https://anaconda.org/feztool/fezrs) ![GitHub stars](https://img.shields.io/github/stars/FEZtool-team/FEZrs?style=social)

**FEZrs** is an advanced Python library developed by [**FEZtool**](https://feztool.com/) for Remote Sensing applications. It provides a set of powerful tools for image processing, feature extraction, and analysis of geospatial data.

## **Features**

✅ Apply various image filtering techniques (Gaussian, Laplacian, Sobel, Median, Mean)  
✅ Contrast enhancement and edge detection  
✅ Support for geospatial raster data (TIFF)  
✅ Designed for remote sensing and satellite imagery analysis  
✅ Easy integration with FastAPI for web-based processing

## **📦 Installation**

You can install **FEZrs** using your preferred Python package manager:

### Using `pip` (PyPI)

```bash
pip install fezrs
```

### Using `conda` (Anaconda)

```bash
conda install -c FEZtool fezrs
```

### Using `mamba` (optional, faster conda alternative)

```bash
mamba install FEZtool::fezrs
```

> **Note:** The `mamba` command requires [Mamba](https://github.com/mamba-org/mamba) to be installed. If it's not installed, use the `conda` command instead.

## **Usage**

Example of applying a Gaussian filter to an image:

```python
from fezrs import EqualizeRGBCalculator

equalize = EqualizeRGBCalculator(
    blue_path="path/to/your/image_band.tif",
    green_path="path/to/your/image_band.tif",
    red_path="path/to/your/image_band.tif",
)

equalize.chart_export(output_path="./your/export/path")
equalize.execute(output_path="./your/export/path")
```

## **Modules**

- `KMeansCalculator`
- `GuassianCalculator`
- `LaplacianCalculator`
- `MeanCalculator`
- `MedianCalculator`
- `SobelCalculator`
- `GLCMCalculator`
- `HSVCalculator`
- `IRHSVCalculator`
- `AdaptiveCalculator`
- `AdaptiveRGBCalculator`
- `EqualizeCalculator`
- `EqualizeRGBCalculator`
- `FloatCalculator`
- `GammaCalculator`
- `GammaRGBCalculator`
- `LogAdjustCalculator`
- `OriginalCalculator`
- `OriginalRGBCalculator`
- `SigmoidAdjustCalculator`
- `PCACalculator`
- `AFVICalculator`
- `BICalculator`
- `NDVICalculator`
- `NDWICalculator`
- `SAVICalculator`
- `UICalculator`
- `SpectralProfileCalculator`

## **Contributing**

We welcome contributions! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m "Add new feature"`)
4. Push to your branch (`git push origin feature-name`)
5. Open a Pull Request

## **Acknowledgment**

Special thanks to [**Chakad Cafe**](https://www.chakadcoffee.com/) for the coffee that kept us fueled during development! ☕

## **License**

This project is licensed under the [**Apache-2.0 license**.](https://github.com/FEZtool-team/FEZrs/edit/main/LICENSE)



