import fezrs


def test_top_level_exports_include_all_calculators():
    exported = {name for name in dir(fezrs) if name.endswith("Calculator")}
    expected = {
        "KMeansCalculator",
        "GaussianCalculator",
        "LaplacianCalculator",
        "MeanCalculator",
        "MedianCalculator",
        "SobelCalculator",
        "GLCMCalculator",
        "HSVCalculator",
        "IRHSVCalculator",
        "AdaptiveCalculator",
        "AdaptiveRGBCalculator",
        "EqualizeCalculator",
        "EqualizeRGBCalculator",
        "FloatCalculator",
        "GammaCalculator",
        "GammaRGBCalculator",
        "LogAdjustCalculator",
        "OriginalCalculator",
        "OriginalRGBCalculator",
        "SigmoidAdjustCalculator",
        "PCACalculator",
        "AFRICalculator",
        "BICalculator",
        "NDVICalculator",
        "NDWICalculator",
        "SAVICalculator",
        "UICalculator",
        "SpectralProfileCalculator",
        "MosaicCalculator",
        "Geoeye_Calculator",
        "Landsat8_Calculator",
        "BurnCalculator",
        "IndicesCalculator",
        "MagDirCalculator",
        "SubDivCalculator",
        "TimeCalculator",
        "SVMCalculator",
    }
    assert expected.issubset(exported)


def test_subpackage_imports():
    from fezrs.tools.change_detection import BurnCalculator
    from fezrs.tools.svm import SVMCalculator
    from fezrs.tools.mosaic import MosaicCalculator
    from fezrs.tools.spectral_indices import NDVICalculator
    from fezrs.tools.image_enhancement import GammaCalculator
    from fezrs.tools.filters import GaussianCalculator

    assert BurnCalculator.__name__ == "BurnCalculator"
    assert SVMCalculator.__name__ == "SVMCalculator"
    assert MosaicCalculator.__name__ == "MosaicCalculator"
    assert NDVICalculator.__name__ == "NDVICalculator"
    assert GammaCalculator.__name__ == "GammaCalculator"
    assert GaussianCalculator.__name__ == "GaussianCalculator"
