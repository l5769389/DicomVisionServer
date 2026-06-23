# Changelog

## v3.0.0 - 2026-06-23

这是一次大型后端版本升级，为前端 v3.0.0 的 PET/CT Fusion、MPR 分割、右侧结果区和移动端工作流提供渲染、状态和 socket 支撑。

### 新增

- 新增 PET/CT Fusion 服务能力，覆盖 CT/PET/融合/MIP 多视口渲染、PET-only 渲染、手动配准预览、配准接受/保存和 socket 状态分发。
- 新增 MPR segmentation 后端能力，支持阈值分割、VOI、统计指标、overlay/sidecar 数据、preview metadata 和渲染 intent。
- 新增 PET 独立影像和 PET-only 视口的显示参数、伪彩/强度范围、白底渲染和 viewport metadata 支撑。
- 新增更完整的 viewer render dispatch、view group registry、view registry 和 socket runtime 路径，支持更复杂的多视口同步。

### 改进

- 优化 PACS DIMSE/WADO job 启动和查询路径，减少应用启动阶段不必要的阻塞。
- 优化 MPR oblique/reslice、viewport transform、annotation overlay 和 layered renderer 的状态传递。
- 扩展 view schema/model，统一前端多视图、PET/CT、分割、配准和渲染参数的数据结构。

### 修复

- 修复 PET-only fusion viewport 背景、显示范围和渲染结果不稳定的问题。
- 修复 PET/CT 手动配准预览、渲染 dispatch 和 socket 更新过程中的状态漂移问题。
- 修复 MPR 分割 overlay intent、threshold ghost projection、VOI overlay 和 preview metadata 稳定性问题。
- 修复 annotation/render intent 在多视图和分割场景下的兼容问题。

### 测试

- 新增和扩展 `test_viewer_fusion.py`、`test_viewer_pet_view.py`、`test_mpr_segmentation.py`、socket/render dispatch、annotation、pseudocolor 和 MPR oblique 相关测试。
