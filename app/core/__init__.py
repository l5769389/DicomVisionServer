from typing import Final

VIEW_OP_TYPE_SET_SIZE: Final = "setSize"
VIEW_OP_TYPE_SCROLL: Final = "scroll"
VIEW_OP_TYPE_CROSSHAIR: Final = "crosshair"
VIEW_OP_TYPE_PAN: Final = "pan"
VIEW_OP_TYPE_ZOOM: Final = "zoom"
VIEW_OP_TYPE_WINDOW: Final = "window"
VIEW_OP_TYPE_PSEUDOCOLOR: Final = "pseudocolor"
VIEW_OP_TYPE_TRANSFORM_2D: Final = "transform2d"
VIEW_OP_TYPE_ROTATE_3D: Final = "rotate3d"
VIEW_OP_TYPE_RESET: Final = "reset"
VIEW_OP_TYPE_VOLUME_PRESET: Final = "volumePreset"
VIEW_OP_TYPE_VOLUME_CONFIG: Final = "volumeConfig"
VIEW_OP_TYPE_MPR_MIP_CONFIG: Final = "mprMipConfig"
VIEW_OP_TYPE_MPR_OBLIQUE: Final = "mprOblique"
VIEW_OP_TYPE_MPR_STATE_SYNC: Final = "mprStateSync"
VIEW_OP_TYPE_MEASUREMENT: Final = "measurement"

DRAG_ACTION_START: Final = "start"
DRAG_ACTION_MOVE: Final = "move"
DRAG_ACTION_END: Final = "end"
DRAG_ACTION_TYPES: Final = {
    DRAG_ACTION_START,
    DRAG_ACTION_MOVE,
    DRAG_ACTION_END,
}

ZOOM_MIN: Final = 0.5
ZOOM_MAX: Final = 3.0
ZOOM_DRAG_SENSITIVITY: Final = 0.01
ZOOM_DRAG_SENSITIVITY_3D: Final = 0.0045
ZOOM_DRAG_FACTOR_MIN: Final = 0.05
ZOOM_MIN_3D: Final = 0.65
ZOOM_MAX_3D: Final = 2.35
WINDOW_WIDTH_MIN: Final = 1.0
WINDOW_DRAG_SENSITIVITY: Final = 2.0

MPR_VIEWPORT_AXIAL: Final = "mpr-ax"
MPR_VIEWPORT_CORONAL: Final = "mpr-cor"
MPR_VIEWPORT_SAGITTAL: Final = "mpr-sag"
MPR_VIEWPORT_TYPES: Final = {
    MPR_VIEWPORT_AXIAL,
    MPR_VIEWPORT_CORONAL,
    MPR_VIEWPORT_SAGITTAL,
}

