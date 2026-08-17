# Spectral Profile
## Overview

The `spectral_profile` module computes and visualizes the average spectral signatures across multi-spectral satellite image collections. In remote sensing, a **spectral profile** (or spectral signature) charts how a target surface reflects electromagnetic radiation across different wavelengths. This signature serves as a diagnostic fingerprint for characterizing dominant surface materials and evaluating radiometric variations between land-cover classes.

This module unifies separate spectral bands—typically including the visible spectrum (**Red, Green, Blue**), Near-Infrared (**NIR**), and Short-Wave Infrared (**SWIR1, SWIR2**)—and aggregates their spatial grids. By extracting the mathematical mean of each band, the `SpectralProfileCalculator` produces a discrete line graph ($f(\text{band}) = \mu_{\text{band}}$) that captures the baseline radiometric identity of the entire scene. Call `histogram_export()` to write that line graph. `execute()` exports a log-stretched preview of one input band, not the profile itself.

```
       [6 Ingested Single-Band Layers] (Red, Green, Blue, NIR, SWIR1, SWIR2)
                      │
                      ▼
       ┌──────────────────────────────┐
       │     Valid Band Extraction    │ ──► Drops None values, unifies into 
       │      & Dictionary Staging    │     ordered list via insertion flags.
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │   Global Spatial Averaging   │ ──► Evaluates vector means via:
       │  $\mu = \frac{1}{HW}\sum I$  │     $\mu_{\text{band}} = \text{np.mean}(I_{\text{band}})$
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │ Data Vector Synchronization  │ ──► Maps structural axes arrays:
       │     (xaxis $\leftrightarrow$ yaxis)     │     $\text{xaxis} = \text{Bands}$, $\text{yaxis} = \text{Means}$
       └──────────────┬───────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │   Diagnostic Axis Plotting   │ ──► Draws line graph and attaches
       │   (Fixed Lifecycle Execution)│     the FEZrs system watermark.
       └──────────────────────────────┘
```

## 2. Mathematical Processing Framework

### 2.1. Spatial Band Aggregation

The calculator filters the incoming files to extract valid, non-null bands and stores them in an ordered layout:

$$\text{image\_columns} = \{ \text{band\_name} : I_{\text{band}}(x, y) \mid I_{\text{band}} \neq \text{None} \}$$

This collection is converted into a structurally indexed array where the dictionary keys determine the $X$-axis tracking names:

$$\text{image\_columns\_list\_of\_bands} = [b_1, b_2, \dots, b_m] \quad \text{where } m \le 6$$

### 2.2. Global Spatial Averaging

For each valid single-channel raster layer $I_{\text{band}}$ of height $H$ and width $W$, the engine calculates the overall radiometric mean ($\mu_{\text{band}}$). This scalar value represents the arithmetic average of the entire pixel population:

$$\mu_{\text{band}} = \frac{1}{H \times W} \sum_{x=1}^{H} \sum_{y=1}^{W} I_{\text{band}}(x, y)$$

This calculation reduces the 2D spatial array to a single statistical weight, balancing local anomalies to capture the broad thematic signature of the scene.

### 2.3. Vector Coordinate Mapping

The calculated data points are synchronized into two matching operational vectors that define the plot tracking coordinates:

$$\text{xaxis} = [b_1, b_2, \dots, b_m]$$

$$\text{yaxis} = [\mu_{b_1}, \mu_{b_2}, \dots, \mu_{b_m}]$$

This vector pair creates a discrete function $f(\text{band}) = \mu_{\text{band}}$ that visualizes variations in surface reflectance across the measured spectrum.

## 3. Remote Sensing Interpretation Profiles

The shape of the resulting curve reveals the dominant environmental features and land-cover types across the scene:

- **Vegetation Signature (NIR Peak & Red Dip):** Healthy green vegetation absorbs red light to power photosynthesis while strongly scattering near-infrared energy via leaf structures. This produces a distinct drop in the $\text{Red}$ band followed by a sharp increase ($\mu_{\text{NIR}} \gg \mu_{\text{Red}}$).
    
- **Open Water / Shadow Signature (Flat & Low):** Water bodies and deep shadows absorb most incident light across reflective infrared wavelengths. This results in a low, flat signature line that approaches zero in the $\text{NIR}$ and $\text{SWIR}$ regions.
    
- **Bare Soil Signature (Gradually Increasing):** Exposed soils, gravel fields, and bedrocks show a steady, linear increase in reflectance from the visible bands through the short-wave infrared spectrum.
    
- **Urban / Burn Scars Signature (SWIR Dominant):** Man-made materials (like concrete and asphalt) and burned areas show low near-infrared reflectance but reflect strongly in the short-wave infrared region ($\mu_{\text{SWIR2}} > \mu_{\text{NIR}}$).
