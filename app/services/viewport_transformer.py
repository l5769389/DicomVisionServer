from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import affine_transform

from app.core import ZOOM_MAX, ZOOM_MIN
from app.models.viewer import ViewRecord


@dataclass(frozen=True)
class AffineTransform:
    matrix: np.ndarray

    def inverse_components(self) -> tuple[np.ndarray, np.ndarray]:
        inverse = np.linalg.inv(self.matrix)
        return inverse[:2, :2], inverse[:2, 2]

    def to_pil_coefficients(self) -> tuple[float, float, float, float, float, float]:
        inverse = np.linalg.inv(self.matrix)
        return (
            float(inverse[0, 0]),
            float(inverse[0, 1]),
            float(inverse[0, 2]),
            float(inverse[1, 0]),
            float(inverse[1, 1]),
            float(inverse[1, 2]),
        )


class ViewportTransformer:
    def build_image_to_canvas_transform(
        self,
        image_width: int,
        image_height: int,
        canvas_width: int,
        canvas_height: int,
        view: ViewRecord,
    ) -> AffineTransform:
        zoom = self.clamp_zoom(view.zoom)
        scale_x = -zoom if view.hor_flip else zoom
        scale_y = -zoom if view.ver_flip else zoom
        translate_x = canvas_width / 2.0 + view.offset_x - scale_x * image_width / 2.0
        translate_y = canvas_height / 2.0 + view.offset_y - scale_y * image_height / 2.0

        matrix = np.array(
            [
                [scale_x, 0.0, translate_x],
                [0.0, scale_y, translate_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return AffineTransform(matrix=matrix)

    @staticmethod
    def clamp_zoom(zoom: float) -> float:
        return min(max(float(zoom), ZOOM_MIN), ZOOM_MAX)

    def calculate_contain_zoom(
        self,
        image_width: int,
        image_height: int,
        canvas_width: int,
        canvas_height: int,
    ) -> float:
        if image_width <= 0 or image_height <= 0 or canvas_width <= 0 or canvas_height <= 0:
            return 1.0
        contain_zoom = min(canvas_width / image_width, canvas_height / image_height)
        return self.clamp_zoom(contain_zoom)

    def apply_affine_array(
        self,
        image_array: np.ndarray,
        canvas_width: int,
        canvas_height: int,
        transform: AffineTransform,
        *,
        order: int = 1,
        cval: float = 0.0,
    ) -> np.ndarray:
        affine_matrix, offset = transform.inverse_components()
        # scipy.ndimage.affine_transform indexes 2D arrays in row/col (y/x) order,
        # so convert the inverse transform from x/y into array coordinates.
        array_matrix = affine_matrix[[1, 0]][:, [1, 0]]
        array_offset = offset[[1, 0]]
        transformed = affine_transform(
            image_array,
            array_matrix,
            offset=array_offset,
            output_shape=(canvas_height, canvas_width),
            order=order,
            mode="constant",
            cval=cval,
        )
        if transformed.dtype != np.uint8:
            transformed = np.clip(transformed, 0, 255).astype(np.uint8)
        return transformed

    def apply_affine(
        self,
        image: Image.Image,
        canvas_width: int,
        canvas_height: int,
        transform: AffineTransform,
        *,
        resample: Image.Resampling,
    ) -> Image.Image:
        fillcolor = (0, 0, 0, 0) if image.mode == "RGBA" else 0
        return image.transform(
            (canvas_width, canvas_height),
            Image.Transform.AFFINE,
            transform.to_pil_coefficients(),
            resample=resample,
            fillcolor=fillcolor,
        )


viewport_transformer = ViewportTransformer()
