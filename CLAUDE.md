# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FEZrs — "Feature Extraction and Zoning for Remote Sensing". A Python library (Apache-2.0, `pip install fezrs`) of ~37 raster-analysis "Calculator" classes for multispectral satellite imagery. Published to PyPI and Anaconda (`FEZtool` channel), with a JOSS paper in `paper/`.

## Commands

```bash
python -m pip install -r requirements.txt   # runtime deps (lower bounds)
python -m pip install -r requirements-dev.txt  # pytest; CI does this too
```

`requirements.txt` uses lower bounds (what `setup.py` publishes). Exact pins for reproducible installs live in `requirements-lock.txt`. `requirements-dev.txt` is the test runner (`pytest`).

```bash
python -m pytest                            # full suite (what CI runs)
python -m pytest tests/tools/pca            # one directory
python -m pytest tests/tools/spectral_indices/ndvi_calculator_test.py
python -m pytest tests/tools/spectral_indices/ndvi_calculator_test.py::test_process_calculates_ndvi_correctly
python -m pytest -k "equalize and not rgb"
```

Run the end-to-end examples against the bundled TIFFs in `example/data/` (outputs land in `example/outputs/`):

```bash
python example/pca.py
```

Note `example/pca.py` builds paths from `Path.cwd() / "./example/..."` (run it from the repo root), while `example/filters.py`, `example/clustering.py`, and `example/hsv.py` use `./data/...` (run those from inside `example/`).

Version bumping (drives `setup.py`, the conda recipe, and `CITATION.cff`):

```bash
bump2version patch    # or minor / major — commits and tags per .bumpversion.cfg
```

## Architecture

### The `BaseTool` lifecycle

Every tool subclasses `BaseTool` ([fezrs/base.py](fezrs/base.py)) and follows the same contract:

1. `__init__(**bands_path)` — passes band paths up to `BaseTool`, which builds `self.files_handler` (a `FileHandler`) and loads the watermark logo. Subclasses then eagerly pull what they need: `self.normalized_bands = self.files_handler.get_normalized_bands(requested_bands=[...])` and/or `self.metadata_bands = self.files_handler.get_metadata_bands(...)`.
2. `_validate()` — subclass hook; most tools implement it as `pass`.
3. `process()` — computes and assigns `self._output` (a NumPy array, or a list/stack for multi-component tools like PCA) and returns it.
4. `_customize_export_file(ax)` — optional hook, used by the enhancement/PCA/spectral-profile tools to stamp the FEZtool watermark.
5. `_export_file(...)` — renders `self._output` with matplotlib and saves `{ToolName}_output_{uuid4}.png` under `output_path`. Raises `ValueError("Data not computed.")` if `_output` is `None`; creates the output directory.
6. `execute(output_path, ...)` — the public entry point: `_validate()` → `process()` → `_export_file()`, returns `self` so calls chain.

`filename_prefix` passed to `_export_file` is always overwritten with the class name minus the `Calculator` suffix (`NDVICalculator` → `NDVI`). `execute()` accepts `nrows`/`ncols` but does **not** forward them to `_export_file`; tools needing a grid (e.g. `PCACalculator`, 6×2) override `_export_file` entirely.

Nearly every calculator re-declares `execute()` with the same signature purely to change the matplotlib defaults (`colormap`, `dpi`, `show_colorbar`, `figsize`, `grid`) before delegating to `super().execute(...)` positionally. Follow that pattern rather than inventing new keyword plumbing.

### Band I/O layer

`FileHandler` ([fezrs/utils/file_handler.py](fezrs/utils/file_handler.py)) is the single point where pixels enter the library. It accepts a fixed keyword set — `red_path`, `green_path`, `blue_path`, `nir_path`, `swir1_path`, `swir2_path`, `tif_path`, `tif_paths` (list, for mosaic), plus `before_nir_path` / `before_swir1_path` / `before_swir2_path` for change detection — and eagerly loads every non-`None` path via `skimage.io.imread(...).astype(float)`. A missing path raises `FileNotFoundError` at construction time.

Four accessors, and which one a tool picks matters:

- `get_normalized_bands(requested_bands)` → min-max scaled to `[0, 1]`. Used by spectral indices, PCA-adjacent math, HSV.
- `get_metadata_bands(requested_bands)` → `{"image_plt", "image_skimage", "height", "width"}` per band, i.e. **raw, unnormalized** pixels. Used by the OpenCV filters and change detection.
- `get_images_collection()` → `skimage` `ImageCollection` over all non-`None` paths, ordered by the dict insertion order in `band_paths` (`tif, red, nir, blue, swir1, swir2, green, before_*`). `PCACalculator.bindTheBandsToNumber` is coupled to this order — changing `band_paths` ordering silently corrupts PCA band selection.
- `get_rasterio_tifs()` → `rasterio` datasets from `tif_paths`; raises if `tif_paths` is `None`.

Type aliases and the `Literal` enums for tool options live in [fezrs/utils/type_handler.py](fezrs/utils/type_handler.py) (`BandPathType`, `BandNameType`, `PropertyGLCMType`, `SubDivCDType`, `MagDirCDType`, `TimeCDType`, `Landsat8ExportType`, `BandNamePCAType`). Add new option enums there, not inline in a calculator.

### Histogram export

Tools that also produce a histogram inherit `HistogramExportMixin` ([fezrs/utils/histogram_handler.py](fezrs/utils/histogram_handler.py)) alongside `BaseTool` and expose a public `histogram_export(output_path, ...)`. The mixin supplies `_add_watermark(ax)` and `_save_histogram_figure(...)`. The public method is `histogram_export` (not `chart_export`).

### Tool categories

`fezrs/tools/<category>/<name>_calculator.py`, one class per file, mirrored 1:1 by `docs/<category>.md` and `tests/tools/<category>/<name>_calculator_test.py`:

`change_detection`, `clustering`, `filters`, `glcm`, `hsv`, `image_enhancement`, `import_tools`, `mosaic`, `pca`, `spectral_indices`, `spectral_profile`, `svm`.

`fezrs/__init__.py` is the canonical public surface: it imports every calculator and lists it in `__all__`. Per-category `__init__.py` files exist inconsistently — `change_detection/` and `svm/` have none, and `mosaic/__init__.py` re-exports `BaseTool` instead of `MosaicCalculator`. Prefer `from fezrs import XCalculator` in examples and docs.

Spectral indices divide through `divide_with_nan` ([fezrs/tools/spectral_indices/_division.py](fezrs/tools/spectral_indices/_division.py)), which yields `NaN` instead of raising on a zero denominator. Use it for any new ratio index.

## Adding a calculator

1. Create `fezrs/tools/<category>/<name>_calculator.py` with a `BaseTool` subclass named `<Name>Calculator`; add `HistogramExportMixin` only if it needs a histogram.
2. Register it in `fezrs/__init__.py` (import + `__all__`) and in the category `__init__.py` if one exists.
3. Add `tests/tools/<category>/<name>_calculator_test.py`. The established pattern patches `BaseTool.__init__` at the calculator's own module path and injects fake band dicts, so tests never touch disk or real TIFFs:

   ```python
   with patch("fezrs.tools.spectral_indices.ndvi_calculator.BaseTool.__init__", fake_init):
       calculator = NDVICalculator(nir_path="dummy_nir.tif", red_path="dummy_red.tif")
   ```

   `tests/conftest.py` forces the `Agg` matplotlib backend and puts the repo root on `sys.path`.
4. Add `docs/<category>.md` coverage if the category doc exists.
5. Any new module directory needs an `__init__.py` — `setup.py` uses `find_packages(include=["fezrs", "fezrs.*"])`, which skips directories without one.

## Conventions (from CONTRIBUTING.md)

- Commit subjects start with an uppercase verb: `ADD`, `FIX`, `UPDATE`, `REMOVE`. Keep commits focused.
- Every behavior change needs tests under `tests/`. Tests must not hit the network, and should avoid the filesystem unless the feature requires it — use mocks/fixtures.
- Do not reorganize the project structure, and do not add dependencies, without prior discussion.
- Do not modify `README.md` unless explicitly asked.
- Type-hint new public functions and methods; preserve array shapes, dtypes, and numerical stability in scientific code.
- Branch naming in `CONTRIBUTING.md` (`feature/...`) conflicts with this machine's enforced global policy — use `feat/`, `hotfix/`, or `refactor/`.

## Release plumbing

- The single source of version truth is `.bumpversion.cfg`; `setup.py` parses it, and the conda recipe (`recip/meta.yaml`) reads `FEZRS_VERSION` from the environment.
- `setup.py` reads `install_requires` straight from `requirements.txt`, and `tests/setup_test.py` asserts they stay in sync (including the `rasterio==1.5.0` pin and `python_requires=">=3.11"`). Editing `requirements.txt` therefore changes the published metadata.
- `CITATION.cff` is generated by `.github/scripts/update_citation.py` from `.github/citation.json` + `.bumpversion.cfg` + the `author` string in `setup.py`; the script validates that citation authors match `setup.py`. Do not hand-edit `CITATION.cff`. Covered by `tests/test_update_citation.py`.
- Workflows in `.github/workflows/`: tests (Python 3.12, Ubuntu 22.04), PyPI publish, conda publish, and citation update chained off a successful PyPI publish.
