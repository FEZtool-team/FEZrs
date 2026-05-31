import cv2
import itertools
import numpy as np
import pandas as pd

from skimage import io
from sklearn import svm
from fezrs.base import BaseTool
from fezrs.utils.type_handler import BandPathType


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

    def process(self):

        self._validate()

        red_normalized = self.normalized_bands["red"]
        green_normalized = self.normalized_bands["green"]
        blue_normalized = self.normalized_bands["blue"]

        width = self.metadata_shape["blue"]["width"]
        height = self.metadata_shape["blue"]["height"]

        rgb = np.stack([red_normalized, green_normalized, blue_normalized], axis=2)

        all_images = io.concatenate_images(self.collection_bands).transpose()
        all_image_reshape = all_images.reshape(
            (height * width, len(self.collection_bands))
        )

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
                        mylist.append(j[x][y])
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

                    clf = svm.SVC(gamma="scale")
                    clf.fit(X, Y)
                    pred = clf.predict(all_image_reshape)
                    svm_output = pred.reshape((height, width)).transpose()
                    self._output = svm_output
                    return self._output

        cv2.namedWindow("mouseClick", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("mouseClick", mouseclick)

        while not self.is_finished_click_event:
            cv2.imshow("mouseClick", rgb)
            if cv2.waitKey(20) == 27:
                break

        cv2.destroyAllWindows()
        return

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
