"""
End-to-end smoke tests: every calculator, run for real, with no mocks.

Issue #46 found three public methods that crash on ordinary use, all of them in
code the suite already covered. Each slipped through for its own reason, but the
shape was the same -- the tests asserted that the plumbing got called, not that
the code ran:

* the spectral profile test passed a ``MagicMock()`` as ``ax``, which accepts
  ``ax.figure(...)`` and ``ax.xlabel(...)`` happily, so a method that could
  never execute still went green;
* the PCA test called ``process()`` on the line before the broken guard, so the
  guard was never exercised;
* the MagDir fixture chose values where no band difference is ever zero, so the
  one case that occurs in every real image pair was the one case not covered.

These tests are the counterweight. Each builds real GeoTIFFs on disk, constructs
the calculator through its public constructor, calls the public export methods,
and asserts a file actually appears. Nothing is patched. They are deliberately
shallow -- correctness of the numbers belongs in the per-tool tests -- but they
answer the one question mocked tests cannot: does this run at all?
"""

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

import fezrs  # noqa: E402


# Small enough to keep 37 calculators quick, large enough that windowed
# operations and clustering have something to work with.
SIZE = 24
# GLCM evaluates a co-occurrence matrix per pixel over four orientations, so it
# gets its own much smaller raster.
GLCM_SIZE = 8

CRS = "EPSG:32639"
TRANSFORM = from_origin(316725.0, 4176795.0, 30.0, 30.0)

# Low dpi keeps the matplotlib render cheap; the tests care that a file appears,
# not how pretty it is.
DPI = 60


def _band(seed, low, high, size=SIZE):
    """
    Deterministic pseudo-band with real spatial structure.

    A gradient plus noise, rather than pure noise: PCA needs the bands to be
    correlated but not collinear, GLCM needs actual texture, and the indices
    need ratios that are not constant.
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.indices((size, size), dtype=np.float64)
    gradient = (rows / size) * 0.6 + (cols / size) * 0.4
    values = low + (high - low) * (gradient * 0.7 + rng.random((size, size)) * 0.3)
    return values.astype(np.int16)


def _write(path, array):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=CRS,
        transform=TRANSFORM,
    ) as destination:
        destination.write(array, 1)
    return str(path)


@pytest.fixture(scope="session")
def bands(tmp_path_factory):
    """Synthetic 16-bit georeferenced bands, written once for the whole session."""
    directory = tmp_path_factory.mktemp("smoke_bands")

    # Reflectance-ordered DN ranges roughly mimicking a vegetated scene, so the
    # indices produce sensible signs rather than degenerate output.
    spec = {
        "blue": (1, 300, 900),
        "green": (2, 400, 1200),
        "red": (3, 350, 1400),
        "nir": (4, 1800, 5200),
        "swir1": (5, 900, 3000),
        "swir2": (6, 600, 2200),
        # Second date, shifted so change detection sees real differences.
        "before_nir": (7, 1500, 4700),
        "before_swir1": (8, 1000, 3300),
        "before_swir2": (9, 700, 2500),
        # Single-band panchromatic input for the filters and import tools.
        "tif": (10, 200, 6000),
    }

    paths = {
        name: _write(directory / f"{name}.tif", _band(seed, low, high))
        for name, (seed, low, high) in spec.items()
    }

    paths["glcm"] = _write(
        directory / "glcm.tif", _band(11, 100, 4000, size=GLCM_SIZE)
    )

    # GeoEye imagery is multispectral: the import tool indexes a band out of a
    # stacked raster, so it needs a real multi-band file rather than one of the
    # single-band products above.
    stack = np.stack([_band(30 + i, 200, 3000) for i in range(4)])
    multiband = directory / "multiband.tif"
    with rasterio.open(
        multiband,
        "w",
        driver="GTiff",
        height=SIZE,
        width=SIZE,
        count=4,
        dtype=stack.dtype,
        crs=CRS,
        transform=TRANSFORM,
    ) as destination:
        destination.write(stack)
    paths["multiband"] = str(multiband)

    # Two tiles for the mosaic tool. The second is offset east by half a tile so
    # they overlap rather than merely abut.
    for index, offset in enumerate((0.0, SIZE * 15.0)):
        tile = _band(20 + index, 300, 3000)
        path = directory / f"tile_{index}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=SIZE,
            width=SIZE,
            count=1,
            dtype=tile.dtype,
            crs=CRS,
            transform=from_origin(316725.0 + offset, 4176795.0, 30.0, 30.0),
        ) as destination:
            destination.write(tile, 1)
        paths[f"tile_{index}"] = str(path)

    return paths


def _cases(b):
    """
    (name, class, kwargs) for every calculator, using its real constructor.

    Kept as data so a new calculator that is never added here shows up as a
    failure of test_every_exported_calculator_has_a_smoke_case.
    """
    rgb = dict(red_path=b["red"], green_path=b["green"], blue_path=b["blue"])
    six = dict(
        **rgb,
        nir_path=b["nir"],
        swir1_path=b["swir1"],
        swir2_path=b["swir2"],
    )

    return [
        # --- spectral indices ---
        ("NDVICalculator", dict(nir_path=b["nir"], red_path=b["red"])),
        ("NDWICalculator", dict(nir_path=b["nir"], green_path=b["green"])),
        ("SAVICalculator", dict(nir_path=b["nir"], red_path=b["red"])),
        ("UICalculator", dict(nir_path=b["nir"], swir2_path=b["swir2"])),
        ("AFRICalculator", dict(nir_path=b["nir"], swir1_path=b["swir1"])),
        (
            "BICalculator",
            dict(
                nir_path=b["nir"],
                red_path=b["red"],
                swir1_path=b["swir1"],
                blue_path=b["blue"],
            ),
        ),
        # --- filters ---
        ("GaussianCalculator", dict(tif_path=b["tif"])),
        ("MeanCalculator", dict(tif_path=b["tif"])),
        ("LaplacianCalculator", dict(tif_path=b["tif"], kernel_size=3)),
        ("MedianCalculator", dict(tif_path=b["tif"], kernel_size=3)),
        ("SobelCalculator", dict(tif_path=b["tif"], kernel_size=3)),
        # --- enhancement ---
        ("OriginalCalculator", dict(nir_path=b["nir"])),
        ("OriginalRGBCalculator", rgb),
        ("EqualizeCalculator", dict(nir_path=b["nir"])),
        ("EqualizeRGBCalculator", rgb),
        ("AdaptiveCalculator", dict(nir_path=b["nir"], clip_limit=0.03)),
        ("AdaptiveRGBCalculator", rgb),
        ("GammaCalculator", dict(nir_path=b["nir"])),
        ("GammaRGBCalculator", rgb),
        ("LogAdjustCalculator", dict(nir_path=b["nir"])),
        ("SigmoidAdjustCalculator", dict(nir_path=b["nir"])),
        ("FloatCalculator", dict(nir_path=b["nir"])),
        # --- colour space ---
        (
            "HSVCalculator",
            dict(
                channel="hsv",
                nir_path=b["nir"],
                blue_path=b["blue"],
                green_path=b["green"],
            ),
        ),
        (
            "IRHSVCalculator",
            dict(
                red_path=b["red"], swir1_path=b["swir1"], swir2_path=b["swir2"]
            ),
        ),
        # --- texture, clustering, decomposition ---
        ("GLCMCalculator", dict(nir_path=b["glcm"], window_size=3, levels=16)),
        ("KMeansCalculator", dict(nir_path=b["nir"], n_clusters=3, random_state=0)),
        ("PCACalculator", dict(**six, component=1)),
        ("SpectralProfileCalculator", six),
        # --- change detection ---
        (
            "BurnCalculator",
            dict(
                nir_path=b["nir"],
                swir2_path=b["swir2"],
                before_nir_path=b["before_nir"],
                before_swir2_path=b["before_swir2"],
            ),
        ),
        (
            "IndicesCalculator",
            dict(
                nir_path=b["nir"],
                swir2_path=b["swir2"],
                before_nir_path=b["before_nir"],
                before_swir2_path=b["before_swir2"],
                time="after",
            ),
        ),
        (
            "MagDirCalculator",
            dict(
                nir_path=b["nir"],
                swir1_path=b["swir1"],
                before_nir_path=b["before_nir"],
                before_swir1_path=b["before_swir1"],
                selecte="magnitude",
            ),
        ),
        (
            "SubDivCalculator",
            dict(
                nir_path=b["nir"],
                before_nir_path=b["before_nir"],
                operation="subtract",
            ),
        ),
        (
            "TimeCalculator",
            dict(nir_path=b["nir"], before_nir_path=b["before_nir"], time="after"),
        ),
        # --- import, mosaic, classification ---
        ("Landsat8Calculator", dict(**six, exportType="rgb")),
        ("GeoeyeCalculator", dict(tif_path=b["multiband"], level=0)),
        ("MosaicCalculator", dict(tif_paths=[b["tile_0"], b["tile_1"]])),
        (
            "SVMCalculator",
            dict(
                **six,
                # Programmatic samples, so no GUI is involved. Before issue #39
                # this tool could not be smoke tested at all.
                training_samples=[
                    (2, 2, 1),
                    (3, 3, 1),
                    (2, 3, 1),
                    (SIZE - 3, SIZE - 3, 2),
                    (SIZE - 4, SIZE - 4, 2),
                    (SIZE - 3, SIZE - 4, 2),
                ],
            ),
        ),
    ]


def _case_ids(b):
    return [name for name, _ in _cases(b)]


def _build(name, kwargs):
    return getattr(fezrs, name)(**kwargs)


# pytest needs the parameter list at collection time, before fixtures exist, so
# the table is built from names here and resolved against the fixture inside
# each test.
CASE_NAMES = [
    name
    for name, _ in _cases(
        {
            key: f"{key}.tif"
            for key in (
                "red green blue nir swir1 swir2 before_nir before_swir1 "
                "before_swir2 tif glcm multiband tile_0 tile_1"
            ).split()
        }
    )
]


def _kwargs_for(name, b):
    for case_name, kwargs in _cases(b):
        if case_name == name:
            return kwargs
    raise AssertionError(f"no smoke case for {name}")


# --- The harness --------------------------------------------------------------


def test_every_exported_calculator_has_a_smoke_case():
    """
    A calculator added without a smoke case is the gap this whole module exists
    to close, so make that omission a failure rather than a silent absence.
    """
    exported = {
        name
        for name in fezrs.__all__
        if name.endswith("Calculator") and "_" not in name
    }

    assert exported == set(CASE_NAMES), (
        f"missing smoke cases: {sorted(exported - set(CASE_NAMES))}; "
        f"unknown cases: {sorted(set(CASE_NAMES) - exported)}"
    )


@pytest.mark.parametrize("name", CASE_NAMES)
def test_execute_writes_a_png(name, bands, tmp_path):
    """Every calculator must run end to end and produce an image."""
    calculator = _build(name, _kwargs_for(name, bands))

    output = tmp_path / name
    calculator.execute(output_path=output, dpi=DPI)

    written = list(output.glob("*.png"))
    assert len(written) == 1, f"{name} wrote {len(written)} PNGs"
    assert written[0].stat().st_size > 0


@pytest.mark.parametrize("name", CASE_NAMES)
def test_histogram_export_writes_a_png(name, bands, tmp_path):
    """
    For the 13 calculators that expose it. This is the check that would have
    caught the spectral-profile crash: a mocked ``ax`` accepts anything, a real
    one does not.
    """
    calculator_class = getattr(fezrs, name)
    if "histogram_export" not in vars(calculator_class):
        pytest.skip(f"{name} does not define histogram_export")

    calculator = _build(name, _kwargs_for(name, bands))

    output = tmp_path / name
    output.mkdir(parents=True, exist_ok=True)
    calculator.histogram_export(output_path=output, dpi=DPI)

    written = list(output.glob("*.png"))
    assert written, f"{name}.histogram_export() wrote nothing"
    assert written[0].stat().st_size > 0


@pytest.mark.parametrize("name", CASE_NAMES)
def test_to_raster_writes_a_georeferenced_tif(name, bands, tmp_path):
    """
    The data-product path from issue #41, exercised across every tool rather
    than the single one its own tests cover.
    """
    if name == "MosaicCalculator":
        pytest.skip(
            "MosaicCalculator.process() does not assign _output; the merged "
            "raster is written directly by _export_file, which sets _output to "
            "a file path rather than an array. Reported separately rather than "
            "changed here, since it alters public behaviour."
        )

    calculator = _build(name, _kwargs_for(name, bands))
    calculator.process()

    destination = tmp_path / f"{name}.tif"
    calculator.to_raster(destination)

    with rasterio.open(destination) as source:
        assert source.crs is not None
        assert source.count >= 1
        assert source.read(1).size > 0


# --- The three specific defects from issue #46 --------------------------------


def test_spectral_profile_histogram_export_runs(bands, tmp_path):
    """
    histogram_export() called pyplot functions on an Axes -- ax.figure(...),
    ax.xlabel(...), ax.ylabel(...) -- so it raised
    "TypeError: 'Figure' object is not callable" on every input. The existing
    test passed a MagicMock() as ax, which accepts all three without complaint.

    This is also the method that draws the per-band mean profile the module's
    documentation is about, so while it was broken there was no way to obtain
    the figure the docs describe.
    """
    calculator = fezrs.SpectralProfileCalculator(
        red_path=bands["red"],
        green_path=bands["green"],
        blue_path=bands["blue"],
        nir_path=bands["nir"],
        swir1_path=bands["swir1"],
        swir2_path=bands["swir2"],
    )

    output = tmp_path / "profile"
    output.mkdir()
    calculator.histogram_export(output_path=output, dpi=DPI)

    written = list(output.glob("*.png"))
    assert len(written) == 1
    assert written[0].stat().st_size > 0
    # The profile itself: one mean per band.
    assert len(calculator.xaxis) == 6
    assert len(calculator.yaxis) == 6


def test_pca_histogram_export_works_without_execute_first(bands, tmp_path):
    """
    The guard read `if not hasattr(self, "_output")`, but BaseTool.__init__
    assigns `self._output = None`, so hasattr was always True, process() never
    ran, and the next line indexed into None. It only worked when chained after
    execute(), which is exactly how example/pca.py calls it.
    """
    calculator = fezrs.PCACalculator(
        red_path=bands["red"],
        green_path=bands["green"],
        blue_path=bands["blue"],
        nir_path=bands["nir"],
        swir1_path=bands["swir1"],
        swir2_path=bands["swir2"],
        component=1,
    )

    assert calculator._output is None

    output = tmp_path / "pca"
    output.mkdir()
    calculator.histogram_export(output_path=output, dpi=DPI)

    assert calculator._output is not None
    assert len(list(output.glob("*.png"))) == 1


def test_magdir_handles_pixels_that_did_not_change(bands, tmp_path):
    """
    Every direction branch tested strictly < 0 or > 0, and change_direction was
    only assigned inside those branches. A difference of exactly zero therefore
    either raised UnboundLocalError (if it was the first pixel) or silently left
    the previous pixel's class in place -- a wrong map, with no error.

    Using one date as both inputs makes every difference zero, which is the
    most basic sanity check a change detector has.
    """
    unchanged = dict(
        nir_path=bands["nir"],
        before_nir_path=bands["nir"],
        swir1_path=bands["swir1"],
        before_swir1_path=bands["swir1"],
    )

    magnitude = fezrs.MagDirCalculator(**unchanged, selecte="magnitude").process()
    assert np.all(magnitude == 0), "no change must give magnitude zero everywhere"

    direction = fezrs.MagDirCalculator(**unchanged, selecte="direction").process()
    assert np.all(direction == 0), (
        "an unchanged pixel must be classified as no-change, not inherit a "
        f"neighbour's class; got classes {sorted(np.unique(direction).tolist())}"
    )

    # And it still runs end to end.
    output = tmp_path / "magdir"
    fezrs.MagDirCalculator(**unchanged, selecte="magnitude").execute(
        output_path=output, dpi=DPI
    )
    assert len(list(output.glob("*.png"))) == 1


def test_magdir_classifies_the_four_quadrants(bands, tmp_path):
    """The zero case must not have cost the four real direction classes."""
    calculator = fezrs.MagDirCalculator(
        nir_path=bands["nir"],
        before_nir_path=bands["before_nir"],
        swir1_path=bands["swir1"],
        before_swir1_path=bands["before_swir1"],
        selecte="direction",
    )

    classes = set(np.unique(calculator.process()).tolist())

    assert classes <= {0, 1, 2, 3, 4}
    assert classes - {0}, "no direction classes were assigned at all"
