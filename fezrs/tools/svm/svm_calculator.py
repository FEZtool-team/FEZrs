import itertools
import os
import sys
import warnings
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

from skimage import io
from sklearn import svm
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType


def display_is_available() -> bool:
    """
    Whether an OpenCV window can be opened.

    Without a display, ``cv2.imshow`` aborts the process at the Qt layer rather
    than raising, so a caller cannot wrap it in ``try``/``except``. Checking
    first lets the tool fail with a catchable, explanatory error.
    """
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True

    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class SVMCalculator(BaseTool):
    """
    Supervised SVM land-cover classification.

    Training samples can be supplied programmatically, which makes the
    classification reproducible, scriptable and testable. Without them the tool
    falls back to collecting samples through an OpenCV window, which requires a
    graphical session and cannot be reproduced: the training set depends on
    where the operator clicked, with no seed, no sample record and no
    coordinate list.
    """

    def __init__(
        self,
        red_path: BandPathType,
        green_path: BandPathType,
        blue_path: BandPathType,
        nir_path: BandPathType,
        swir1_path: BandPathType,
        swir2_path: BandPathType,
        class_number: int = 4,
        sample_number: int = 10,
        training_samples: Sequence[tuple] | None = None,
        transform=None,
        evaluate: bool = False,
        test_size: float = 0.3,
        random_state: int = 0,
    ):
        """
        Args:
            class_number: Number of land-cover classes.
            sample_number: Training pixels per class, for the interactive path.
            training_samples: Sequence of ``(row, col, class_id)`` triples. When
                given, the GUI is skipped entirely. Pass ``transform`` to supply
                map coordinates instead of array indices.
            transform: Optional affine transform (a ``rasterio`` ``Affine``, or
                any object accepting ``~transform * (x, y)``) used to convert
                ``training_samples`` given as map coordinates into pixel
                indices. A training set digitised in a GIS carries easting and
                northing, not array indices.
            evaluate: Hold out a stratified test split and report accuracy.
            test_size: Fraction held out when ``evaluate`` is set.
            random_state: Seed for the train/test split, so the reported
                accuracy is reproducible.
        """
        super().__init__(
            red_path=red_path,
            green_path=green_path,
            blue_path=blue_path,
            nir_path=nir_path,
            swir1_path=swir1_path,
            swir2_path=swir2_path,
        )
        # Feature scaling is appropriate here, unlike for the spectral indices:
        # an RBF kernel needs comparable ranges across features.
        self.normalized_bands = self.files_handler.get_normalized_bands(
            requested_bands=["red", "green", "blue"]
        )

        self.metadata_shape = self.files_handler.get_metadata_bands(["blue"])
        self.collection_bands = self.files_handler.get_images_collection()
        self.index_loop = 0

        self.is_finished_click_event = False

        self.class_number = class_number
        self.sample_number = sample_number
        self.training_samples = training_samples
        self.transform = transform
        self.evaluate = evaluate
        self.test_size = test_size
        self.random_state = random_state

        self.accuracy_ = None
        self.confusion_matrix_ = None
        self.kappa_ = None

    def _validate(self) -> None:
        # 1) class_number: must be an int ≥ 2 (at least binary classification)
        if not isinstance(self.class_number, int):
            raise ValueError("class_number must be an int.")
        if self.class_number < 2:
            raise ValueError("class_number must be at least 2.")

        # 2) sample_number: must be an int ≥ 1
        if not isinstance(self.sample_number, int):
            raise ValueError("sample_number must be an int.")
        if self.sample_number < 1:
            raise ValueError("sample_number must be at least 1.")

        # 3) Ensure image dimensions are available
        if not hasattr(self, "metadata_shape") or "blue" not in self.metadata_shape:
            self.metadata_shape = self.files_handler.get_metadata_bands(["blue"])

        height = self.metadata_shape["blue"]["height"]
        width = self.metadata_shape["blue"]["width"]
        total_pixels = height * width

        if self.training_samples is not None:
            if len(self.training_samples) == 0:
                raise ValueError("training_samples must not be empty.")

            labels = {sample[2] for sample in self.training_samples}
            if len(labels) < 2:
                raise ValueError(
                    "training_samples must cover at least two classes, "
                    f"received {sorted(labels)}."
                )
            return

        requested_samples = self.class_number * self.sample_number

        # Cannot request more samples than pixels in the image
        if requested_samples > total_pixels:
            raise ValueError(
                f"Requested {requested_samples} samples, "
                f"but the image only has {total_pixels} pixels."
            )

        # 4) Optional heads‑up when the manual workload might be huge
        max_reasonable = int(
            0.05 * total_pixels
        )  # arbitrary threshold: 5 % of the image
        if requested_samples > max_reasonable:
            warnings.warn(
                f"Selecting {requested_samples} pixels manually may be "
                "impractical. Consider passing training_samples instead.",
                UserWarning,
                stacklevel=2,
            )

    def _to_pixel_indices(self, row, col):
        """
        Resolve one training location to array indices.

        With a ``transform`` the pair is read as map coordinates ``(x, y)`` in
        the raster's CRS and inverted onto the pixel grid; without one it is
        already ``(row, col)``.
        """
        if self.transform is None:
            return int(row), int(col)

        # Map coordinates arrive as (x, y) = (easting, northing).
        column_float, row_float = ~self.transform * (col, row)
        return int(row_float), int(column_float)

    def _collect_features(self, samples):
        """Turn (row, col, class_id) triples into a feature matrix and labels."""
        height = self.metadata_shape["blue"]["height"]
        width = self.metadata_shape["blue"]["width"]

        features = []
        labels = []

        for row, col, class_id in samples:
            pixel_row, pixel_col = self._to_pixel_indices(row, col)

            if not (0 <= pixel_row < height and 0 <= pixel_col < width):
                raise ValueError(
                    f"Training sample ({row}, {col}) falls outside the raster "
                    f"extent ({height} x {width})."
                )

            features.append([band[pixel_row][pixel_col] for band in self.collection_bands])
            labels.append(int(class_id))

        return np.asarray(features, dtype=float), np.asarray(labels, dtype=int)

    def _fit_and_predict(self, features, labels):
        """
        Fit the classifier and predict the full scene.

        Shared by the interactive and programmatic paths so both run identical
        code.
        """
        height = self.metadata_shape["blue"]["height"]
        width = self.metadata_shape["blue"]["width"]

        all_images = io.concatenate_images(self.collection_bands).transpose(1, 2, 0)
        all_image_reshape = all_images.reshape(
            (height * width, len(self.collection_bands))
        )

        fit_features, fit_labels = features, labels

        if self.evaluate:
            # A classification reported without an accuracy statement is not a
            # result a reader can assess; kappa is the standard measure in the
            # remote sensing literature.
            (
                fit_features,
                test_features,
                fit_labels,
                test_labels,
            ) = train_test_split(
                features,
                labels,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=labels,
            )

            evaluator = svm.SVC(gamma="scale")
            evaluator.fit(fit_features, fit_labels)
            predicted = evaluator.predict(test_features)

            self.accuracy_ = float(np.mean(predicted == test_labels))
            self.confusion_matrix_ = confusion_matrix(test_labels, predicted)
            self.kappa_ = float(cohen_kappa_score(test_labels, predicted))

        classifier = svm.SVC(gamma="scale")
        classifier.fit(features, labels)

        prediction = classifier.predict(all_image_reshape)

        self.classifier_ = classifier
        self._output = prediction.reshape((height, width))
        return self._output

    def process(self):

        self._validate()

        if self.training_samples is not None:
            features, labels = self._collect_features(self.training_samples)
            return self._fit_and_predict(features, labels)

        return self._process_interactive()

    def _process_interactive(self):
        if not display_is_available():
            raise RuntimeError(
                "SVMCalculator needs training_samples when no display is "
                "available. Interactive sample collection opens an OpenCV "
                "window, which aborts the process on a headless host. Pass "
                "training_samples=[(row, col, class_id), ...] instead."
            )

        red_normalized = self.normalized_bands["red"]
        green_normalized = self.normalized_bands["green"]
        blue_normalized = self.normalized_bands["blue"]

        rgb = np.stack([red_normalized, green_normalized, blue_normalized], axis=2)

        class_num = self.class_number
        sample_num = self.sample_number

        columns = ["Band{}".format(i + 1) for i in range(len(self.collection_bands))]
        classes_df = pd.DataFrame(columns=columns)

        targets = [[i + 1] * sample_num for i in range(class_num)]
        merged = list(itertools.chain(*targets))
        classes_df["Target"] = merged

        def mouseclick(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.index_loop < class_num * sample_num:
                    mylist = []
                    for j in self.collection_bands:
                        mylist.append(j[y][x])
                    classes_df.iloc[self.index_loop, 0 : len(self.collection_bands)] = (
                        mylist
                    )
                    self.index_loop += 1
                    print(classes_df)
                else:
                    self.is_finished_click_event = True
                    array = classes_df.values
                    features = array[:, 0 : len(self.collection_bands)].astype(float)
                    labels = array[:, len(self.collection_bands)].astype("int")

                    self._fit_and_predict(features, labels)
                    return self._output

        cv2.namedWindow("mouseClick", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("mouseClick", mouseclick)

        while not self.is_finished_click_event:
            cv2.imshow("mouseClick", rgb)
            if cv2.waitKey(20) == 27:
                cv2.destroyAllWindows()
                raise RuntimeError(
                    "Sample collection was interrupted before all "
                    f"{class_num * sample_num} samples were collected "
                    f"({self.index_loop} so far), so no classifier was "
                    "trained. Collect every sample, or pass training_samples."
                )

        cv2.destroyAllWindows()
        return self._output

    def _export_file(
        self,
        output_path,
        title="SVM",
        figsize=[10, 10],
        show_axis=True,
        colormap=None,
        show_colorbar=False,
        filename_prefix=None,
        dpi=500,
        bbox_inches="tight",
        grid=False,
        nrows=1,
        ncols=1,
    ):
        return super()._export_file(
            output_path,
            title,
            figsize,
            show_axis,
            colormap,
            show_colorbar,
            filename_prefix,
            dpi,
            bbox_inches,
            grid,
            nrows,
            ncols,
        )

    def execute(
        self,
        output_path,
        title="SVM",
        figsize=[10, 10],
        show_axis=True,
        colormap=None,
        show_colorbar=False,
        filename_prefix=None,
        dpi=500,
        bbox_inches="tight",
        grid=False,
        nrows=None,
        ncols=None,
    ):
        return super().execute(
            output_path,
            title,
            figsize,
            show_axis,
            colormap,
            show_colorbar,
            filename_prefix,
            dpi,
            bbox_inches,
            grid,
            nrows,
            ncols,
        )
