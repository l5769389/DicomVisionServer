from app.services.viewer.shared import *  # noqa: F403
from app.services.viewer.export import ViewerExportMixin
from app.services.viewer.fusion import ViewerFusionMixin
from app.services.viewer.interaction import ViewerInteractionMixin
from app.services.viewer.mpr import ViewerMprMixin
from app.services.viewer.operations import ViewerOperationsMixin
from app.services.viewer.presentation import ViewerPresentationMixin
from app.services.viewer.stack import ViewerStackMixin
from app.services.viewer.state import ViewerStateMixin
from app.services.viewer.volume import ViewerVolumeMixin


class ViewerService(
    ViewerExportMixin,
    ViewerInteractionMixin,
    ViewerStateMixin,
    ViewerVolumeMixin,
    ViewerStackMixin,
    ViewerFusionMixin,
    ViewerMprMixin,
    ViewerOperationsMixin,
    ViewerPresentationMixin,
):
    def __init__(self) -> None:
        self._series_patient_transform_cache: dict[str, VolumePatientTransform | None] = {}
        self._series_volume_geometry_cache: dict[str, VolumeGeometry] = {}
        self._series_representative_slice_cache: dict[str, tuple[int, int]] = {}
        self._series_volume_cache = SeriesVolumeCache(
            max_bytes=VOLUME_CACHE_MAX_BYTES,
            on_evict=self._handle_series_volume_cache_evict,
        )
        self._mpr_plane_cache: OrderedDict[tuple[object, ...], tuple[np.ndarray, int, int]] = OrderedDict()
        self._fast_base_pixels_cache: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
        self._fusion_registration_pet_layer_cache: OrderedDict[
            tuple[object, ...],
            FusionRegistrationPetLayerCacheEntry,
        ] = OrderedDict()
        self._fusion_registration_preview_drags: dict[str, FusionRegistrationPreviewDrag] = {}
        self._fusion_registration_overlay_frame_locks: dict[
            tuple[str, str],
            FusionRegistrationOverlayRenderFrame,
        ] = {}
        self._fusion_registration_transparent_primary_png = self._encode_image(
            Image.new("RGBA", (1, 1), (0, 0, 0, 0)),
            "png",
            fast_preview=True,
        )
        self._mtf_analysis_service = MtfAnalysisService(self)
        self._water_phantom_qa_service = WaterPhantomQaService(self)
        self._volume_render_preprocess_cache: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
        self._logger = logger

    @staticmethod
    def _is_mpr_view_type(view_type: str) -> bool:
        return view_type in {"MPR", "AX", "COR", "SAG"}

    @staticmethod
    def _is_3d_view_type(view_type: str) -> bool:
        return view_type == "3D"

    @staticmethod
    def _is_pet_view_type(view_type: str) -> bool:
        return view_type == "PET"

    @staticmethod
    def _is_fusion_view_type(view_type: str) -> bool:
        return view_type in FUSION_VIEW_TYPES

    def set_view_size(
        self,
        payload: ViewSetSizeRequest,
        workspace_id: str | None = None,
    ) -> OperationAcceptedResponse:
        if payload.op_type != VIEW_OP_TYPE_SET_SIZE:
            raise HTTPException(status_code=400, detail="opType must be setSize")

        view = view_registry.get(payload.view_id, workspace_id=workspace_id)
        previous_width = view.width
        previous_height = view.height
        size_changed = previous_width != payload.size.width or previous_height != payload.size.height
        should_refit_fusion = (
            self._is_fusion_view_type(view.view_type)
            and view.is_initialized
            and size_changed
            and self._is_fusion_view_at_auto_fit_size(
                view,
                canvas_width=previous_width,
                canvas_height=previous_height,
            )
        )
        view.width = payload.size.width
        view.height = payload.size.height
        if (
            self._is_fusion_view_type(view.view_type)
            and size_changed
        ):
            self._clear_fusion_registration_overlay_frame_locks(view.view_group)
        logger.info(
            "set view size view_id=%s width=%s height=%s",
            view.view_id,
            view.width,
            view.height,
        )

        if not view.is_initialized:
            if self._is_fusion_view_type(view.view_type):
                self._initialize_fusion_viewport(view)
            elif self._is_pet_view_type(view.view_type):
                self._initialize_pet_viewport(view)
                view.is_initialized = True
            elif not (self._is_mpr_view_type(view.view_type) or self._is_3d_view_type(view.view_type)):
                self._initialize_viewport(view)
                view.is_initialized = True
        elif should_refit_fusion:
            self._fit_initialized_fusion_view_to_source(view)

        return OperationAcceptedResponse(message="View size updated", viewId=view.view_id)

    def render_view_by_id(
        self,
        view_id: str,
        *,
        image_format: ImageFormat = "webp",
        fast_preview: bool = False,
        fast_preview_full_resolution: bool = False,
        metadata_mode: str = "full",
        progress_callback: ViewRenderProgressCallback | None = None,
        workspace_id: str | None = None,
    ) -> RenderedImageResult:
        view = view_registry.get(view_id, workspace_id=workspace_id)
        if self._is_mpr_view_type(view.view_type):
            view = self._snapshot_mpr_view_for_render(view)
        return self._render_by_view_type(
            view,
            image_format=image_format,
            fast_preview=fast_preview,
            fast_preview_full_resolution=fast_preview_full_resolution,
            metadata_mode=metadata_mode,
            progress_callback=progress_callback,
        )

    def _snapshot_mpr_view_for_render(self, view: ViewRecord) -> ViewRecord:
        ensure_view_size(view)
        if not view.is_initialized:
            self._initialize_mpr_viewport(view)
            view.is_initialized = True
        return deepcopy(view)

    def close_view_by_id(self, view_id: str, workspace_id: str | None = None) -> OperationAcceptedResponse:
        view = view_registry.delete(view_id, workspace_id=workspace_id)
        if self._is_3d_view_type(view.view_type):
            try:
                _get_vtk_volume_renderer().drop_session(view.view_id)
                _get_vtk_surface_renderer().drop_session(view.view_id)
            except Exception:
                logger.warning("failed to release 3D render session view_id=%s", view.view_id, exc_info=True)
        group = view.view_group
        if group is not None and not view_registry.list_view_group(group.group_id, workspace_id=workspace_id):
            view_group_registry.delete(group.group_id)
        return OperationAcceptedResponse(message="View closed", viewId=view.view_id)


viewer_service = ViewerService()
