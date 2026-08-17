import os
import sys
import cv2
import itertools
import numpy as np
import pandas as pd

from skimage import io
from sklearn import svm
from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType


def _display_available() -> bool:
    """Return True when an interactive display appears to be available."""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class SVMCalculator(BaseTool):
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
        training_samples=None,
    ):
        super().__init__(
            red_path=red_path,
            green_path=green_path,
            blue_path=blue_path,
            nir_path=nir_path,
            swir1_path=swir1_path,
            swir2_path=swir2_path,
        )
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

    def _validate(self) -> None:
        # 1) class_number: must be an int ≥ 2 (at least binary classification)
        if not isinstance(self.class_number, int):
            raise ValueError("class_number must be an int.")
        if self.class_number < 2:
            raise ValueError("class_number must be at least 2.")

        # 2) sample_number: must be an int ≥ 1
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
        )  # arbitrary threshold: 5 % of the image
        if requested_samples > max_reasonable:
            print(
                f"Warning: selecting {requested_samples} pixels manually may be impractical."
            )

    def _scene_features(self):
        width = self.metadata_shape["blue"]["width"]
        height = self.metadata_shape["blue"]["height"]
        all_images = io.concatenate_images(self.collection_bands).transpose(1, 2, 0)
        features = all_images.reshape((height * width, len(self.collection_bands)))
        return height, width, features

    def _fit_and_predict(self, samples, labels):
        height, width, features = self._scene_features()
        clf = svm.SVC(gamma="scale")
        clf.fit(samples, labels)
        pred = clf.predict(features)
        self._output = pred.reshape((height, width))
        return self._output

    def _process_from_samples(self):
        height = self.metadata_shape["blue"]["height"]
        width = self.metadata_shape["blue"]["width"]
        samples = list(self.training_samples)

        if len(samples) < 2:
            raise ValueError("training_samples must contain at least two samples.")

        rows, cols, labels = [], [], []
        for item in samples:
            if len(item) != 3:
                raise ValueError(
                    "Each training sample must be (row, col, class_id)."
                )
            row, col, class_id = item
            if not (0 <= int(row) < height and 0 <= int(col) < width):
                raise ValueError(
                    f"Sample ({row}, {col}) is outside the image bounds "
                    f"({height}, {width})."
                )
            rows.append(int(row))
            cols.append(int(col))
            labels.append(int(class_id))

        if len(set(labels)) < 2:
            raise ValueError("training_samples must include at least two classes.")

        training_x = np.array(
            [[band[row, col] for band in self.collection_bands] for row, col in zip(rows, cols)]
        )
        return self._fit_and_predict(training_x, np.asarray(labels, dtype=int))

    def process(self):

        self._validate()

        if getattr(self, "training_samples", None) is not None:
            return self._process_from_samples()

        if not _display_available():
            raise RuntimeError(
                "SVMCalculator requires a display for interactive sample "
                "collection, or pass training_samples=[(row, col, class_id), ...] "
                "for headless use."
            )

        red_normalized = self.normalized_bands["red"]
        green_normalized = self.normalized_bands["green"]
        blue_normalized = self.normalized_bands["blue"]

        rgb = np.stack([red_normalized, green_normalized, blue_normalized], axis=2)
        _height, _width, all_image_reshape = self._scene_features()

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
                    X = array[:, 0 : len(self.collection_bands)]
                    Y = array[:, len(self.collection_bands)].astype("int")
                    self._fit_and_predict(X, Y)
                    return self._output

        cv2.namedWindow("mouseClick", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("mouseClick", mouseclick)

        while not self.is_finished_click_event:
            cv2.imshow("mouseClick", rgb)
            if cv2.waitKey(20) == 27:
                break

        cv2.destroyAllWindows()

        if not self.is_finished_click_event:
            raise RuntimeError(
                "Sample collection was interrupted before all training "
                "samples were collected. Pass training_samples=... for a "
                "non-interactive run."
            )
        return self._output

    def _export_file(
        self,
        output_path,
        title="SVM",
        figsize=[10, 10],
        show_axis=True,
        colormap=None,
        show_colorbar=False,
        filename_prefix="Tool_output",
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
        filename_prefix="Tool_output",
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
