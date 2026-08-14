---
title: 'FEZrs: An Open-Source Python Package for Geospatial Multispectral Image Processing and Remote Sensing'
tags:
  - Python
  - remote sensing
  - geospatial
  - multispectral
  - image processing
  - spectral indices
authors:
  - name: Mahdi Farmahinifarahani
    orcid: 0009-0008-3800-8688
    affiliation: '1'
  - name: Parsa Elmi
    orcid: 0009-0005-2751-4161
    corresponding: true
    affiliation: '2'
  - name: Mehdi Nedaee
    orcid: 0009-0001-0357-6019
    affiliation: '3'
  - name: Mohammad Kiani faizabadi
    orcid: 0009-0001-3867-2107
    affiliation: '4'
  - name: Hooman Mirzaee
    orcid: 0009-0004-4289-5989
    affiliation: '4'
  - name: Parsa Moradi
    orcid: 0009-0000-2008-123X
    affiliation: '3'
  - name: Mohammadhossein Yazdanifar
    orcid: 0009-0009-3929-103X
    affiliation: '5'
  - name: Fariba Khosravani
    orcid: 0009-0007-3223-8506
    affiliation: '6'
  - name: Fatemeh Najafi
    orcid: 0009-0009-1541-2618
    affiliation: '6'
  - name: Mehdi Talkhablou
    orcid: 0009-0007-7657-7008
    affiliation: '6'
affiliations:
  - name: Department of Earth and Environmental Science, The Chinese University of Hong Kong, Hong Kong SAR
    index: 1
  - name: Faculty of Management, Islamic Azad University, Central Tehran Branch, Tehran, Iran
    index: 2
  - name: Department of Software Engineering, Shamsipour Technical College, Tehran, Iran
    index: 3
  - name: Faculty of Computer Science, Islamic Azad University, Central Tehran Branch, Tehran, Iran
    index: 4
  - name: Department of Industrial Engineering, Sharif University of Technology, Tehran, Iran
    index: 5
  - name: Faculty of Earth Sciences, Kharazmi University, Tehran, Iran
    index: 6
date: 11 July 2026
bibliography: paper.bib
---

# Summary

The history of remote sensing has evolved from early nineteenth-century balloon photography to modern multispectral and satellite-based earth observation systems, enabling large-scale environmental monitoring and geospatial analysis workflows [@Cracknell:2018]. Over time, remote sensing technologies have expanded from aerial observation platforms to advanced satellite [@Zhang:2022] and unmanned aerial vehicle [@SimicMilas:2018] systems that support scientific research, environmental assessment, and operational geospatial applications.

The increasing availability of multispectral remote sensing imagery from platforms such as Landsat [@Williams:2006] and Sentinel [@Malenovsky:2012] has accelerated the adoption of spectral analysis methods across a broad range of remote sensing tasks and applications [@Xue:2017; @Montero:2023]. These developments have significantly expanded the use of remote sensing across interdisciplinary domains including geoscience [@CampsValls:2021] and mineral resource assessment [@Sabins:1999], environmental engineering, Wild life monitoring [@Melesse:2007; @Shukla:2024], biology, marine science, [@Chassot:2011] and agriculture. [@Sishodia:2020]

As remote sensing datasets continue to grow in scale and complexity, researchers and practitioners increasingly rely on accessible computational tools for image analysis, feature extraction, and spectral interpretation. These workflows often require standardized implementations of widely used spectral indices [@Montero:2023] and consistent processing utilities to ensure reproducibility and efficiency in both research and operational applications. FEZrs is designed to simplify remote sensing data processing and interpretation by providing a consistent framework for a wide range of analytical workflows.

Although primarily developed for geoscience applications, FEZrs can also be applied more broadly to multispectral and hyperspectral spectral analysis in diverse fields beyond the geosciences.

# Statement of Need

FEZrs is an open-source project for remote sensing image processing and geospatial analysis that provides a Python package for spectral analysis, image enhancement, filtering, spectral index computation, and feature extraction. The package integrates established remote sensing workflows within a unified open-source framework, supporting both research and industrial applications.

![Overview of the FEZrs processing workflow. Multispectral remote sensing imagery is processed within the FEZrs framework to generate representative outputs, including (a) K-Means clustering, (b) gamma-corrected RGB enhancement, (c) RGB Image representation, and (d) Support Vector Machine (SVM) classification.\label{fig:workflow}](figures/figure1.png)

Numerous open-source software libraries provide relevant functionality; however, these capabilities are frequently distributed across separate packages and frameworks, requiring users to combine multiple tools to construct complete analysis workflows. FEZrs builds upon widely adopted libraries, including OpenCV [@Xie:2013], scikit-image [@vanderWalt:2014], scikit-learn [@Kramer:2016], NumPy [@Harris:2020], matplotlib [@Hunter:2007], and pandas [@McKinney:2010], while maintaining compatibility with other commonly used libraries and frameworks in the geospatial and image analysis ecosystem, such as Rasterio [@Gillies:2013], GDAL [@Warmerdam:2008], geopandas [@Jordahl:2020]. Nevertheless, constructing end-to-end analytical pipelines often demands considerable effort to integrate heterogeneous software components, methodologies, and data-processing workflows.

![Representative spectral index outputs generated by FEZrs, including (a) Soil Adjusted Vegetation Index (SAVI), (b) Urban Index (UI), (c) Normalized Difference Vegetation Index (NDVI), and (d) Normalized Difference Water Index (NDWI). All outputs were computed from the same multispectral dataset using the corresponding FEZrs implementations.\label{fig:indices}](figures/figure2.png)

Furthermore, modern remote sensing studies frequently rely on multiple analytical approaches. FEZrs implements a collection of widely used spectral indices [@Montero:2023], including the Normalized Difference Vegetation Index (NDVI) [@Huang:2021], Normalized Difference Water Index (NDWI) [@McFeeters:1996], Soil Adjusted Vegetation Index (SAVI) [@Huete:1988], aerosol free vegetation index (AFRI) [@Karnieli:2001], Bare Soil Index (BI) [@AsSyakur:2012], and Urban Index (UI) [@Deng:2012]. These indices are widely used for vegetation monitoring, water-resource assessment, environmental monitoring, land-surface characterization, and urban analysis.

The package also incorporates image enhancement and filtering techniques, including Gaussian smoothing [@Lindeberg:2024], median filtering [@Justusson:1981], Laplacian [@Merris:1994], and Sobel operators [@Chang:2023], which are widely used for noise reduction, edge detection, and image interpretation in remote sensing applications. Texture-based analysis is supported through the Gray-Level Co-occurrence Matrix (GLCM) [@Utaminingrum:2023], a widely adopted approach for characterizing spatial patterns in land-cover mapping and environmental studies. FEZrs further provides Principal Component Analysis (PCA) for dimensionality reduction and exploratory analysis of multispectral data [@Jolliffe:2016], as well as K-Means clustering for unsupervised image classification and pattern discovery [@Sinaga:2020]. In addition, Support Vector Machine (SVM) classification is included as a supervised machine-learning approach for land-cover classification and feature discrimination in remote sensing datasets [@Yue:2003]. spectral profile analysis enables the investigation of spectral responses across wavelengths and supports multispectral interpretation workflows.

| Change Detection | Filters   | Image Enhancement | Spectral Indices | Additional Tools |
|------------------|-----------|-------------------|------------------|------------------|
| Burn             | Gaussian  | Original          | AFRI             | KMeans           |
| Indices          | Laplacian | Float             | BI               | GLCM             |
| MagDir           | Mean      | Equalize          | NDVI             | HSV              |
| SubDiv           | Median    | Adaptive          | NDWI             | IRHSV            |
| Time             | Sobel     | Gamma             | SAVI             | Mosaic           |
|                  |           | LogAdjust         | UI               | PCA              |
|                  |           | SigmoidAdjust     |                  | SpectralProfile  |
|                  |           | OriginalRGB       |                  | SVM              |
|                  |           | EqualizeRGB       |                  |                  |
|                  |           | AdaptiveRGB       |                  |                  |
|                  |           | GammaRGB          |                  |                  |

Table 1: provides an overview of the implemented FEZrs modules, enabling users to perform common remote sensing processing tasks, spectral analysis, feature extraction, and image-based analysis workflows within a single Python package.

Although these methods are widely utilized, there is still considerable value in open-source tools that bring together essential remote sensing functionalities within a single, coherent analytical platform. While FEZrs was developed with geoscience applications in mind, its functionality is equally applicable to multispectral and hyperspectral data analysis across a variety of scientific and technical domains.

# State of the field

Numerous open-source software libraries provide relevant functionality for remote sensing and image analysis; however, these capabilities are frequently distributed across separate packages and frameworks, requiring users to combine multiple tools to construct complete analysis workflows. FEZrs builds upon widely adopted libraries, including OpenCV [@Xie:2013], scikit-image [@vanderWalt:2014], scikit-learn [@Kramer:2016], NumPy [@Harris:2020], matplotlib [@Hunter:2007], and pandas [@McKinney:2010], while maintaining compatibility with other commonly used libraries and frameworks in the geospatial and image analysis ecosystem, such as Rasterio [@Gillies:2013], GDAL [@Warmerdam:2008], and GeoPandas [@Jordahl:2020]. Nevertheless, constructing end-to-end analytical pipelines often demands considerable effort to integrate heterogeneous software components, methodologies, and data-processing workflows.

FEZrs was built to address this fragmentation by consolidating frequently used remote sensing operations-spectral index computation, filtering, enhancement, texture analysis, dimensionality reduction, and both unsupervised and supervised classification-behind a consistent, calculator-style API. Rather than replacing low-level geospatial I/O libraries or general-purpose machine-learning frameworks, FEZrs layers domain-oriented workflows on top of them so that researchers can move from multi-band imagery to interpretable outputs with fewer ad hoc integration steps. This design targets users who need reproducible, scriptable remote sensing analysis without assembling a bespoke toolchain for each study.

# Software design

FEZrs is organized around a shared abstract base class (`BaseTool`) that standardizes the lifecycle of every analysis module: input validation, processing, visualization customization, and export. Each calculator (for example, `NDVICalculator`, `GaussianCalculator`, or `KMeansCalculator`) inherits this interface, accepts band file paths through a common file-handling layer, and exposes a small set of methods such as `execute` and `chart_export`. This pattern keeps the public API uniform across spectral indices, filters, enhancement tools, change-detection utilities, and machine-learning modules, reducing cognitive overhead when composing multi-step workflows.

The package is modular by domain: (`spectral_indices`, `filters`, `image_enhancement`, `change_detection`, `clustering`, `glcm`, `pca`, `svm`, and others), while shared utilities handle band-path typing, file I/O, and histogram support. Raster inputs are typically multi-band geospatial imagery. Outputs are written as figures and processed products suitable for inspection, reporting, and further analysis. The design deliberately favors composition of independent calculators over a single monolithic pipeline object, so researchers can select only the methods required for a given study while still benefiting from consistent validation and export behavior.

# Research Impact Statement

Since its release on PyPI in February 2025, FEZrs has been downloaded more than 100,000 times and is openly available through GitHub, PyPI, and the Anaconda ecosystem as an open-source Python package.

FEZrs has been employed in research on remote sensing and geospatial analysis. It has been used for principal component analysis (PCA) of Landsat 9 multispectral imagery @farahani2024pca, comparison of K-means clustering and spectral indices for land use and land cover characterization @rezaei2024kmeans, and a comparative assessment of SVM, K-means, and spectral indices for land cover classification in engineering geology @talkhablou2025engineering. In addition, FEZrs received the Merit Award (Ideator Category) at the 9th Iran National Young Scientists Festival.

The project provides comprehensive documentation, practical examples, a test suite, and a modular architecture to support reproducible research and operational geospatial workflows. Detailed installation instructions for both pip (`pip install fezrs`) and conda (`conda install -c FEZtool fezrs`) are available in the FEZrs Installation Guide.

# Acknowledgements

The authors would like to express their sincere appreciation to the global open-source developer community. The collective efforts, shared knowledge, and freely available tools developed by this community have provided the essential foundation that made this project possible. Their commitment to collaboration, transparency, and innovation continues to advance technological progress and enables researchers to build upon and improve existing work. The authors gratefully acknowledge these invaluable contributions.

# AI Usage Disclosure

While the core software design, implementation, and development decisions of FEZrs were made by the authors, AI-assisted tools were used to assist with unit testing, validation workflows, and manuscript refinement. All AI-assisted outputs were independently reviewed, verified, and validated by the authors.

# Conflict of interest

The authors declare no conflict of interest.

# References
