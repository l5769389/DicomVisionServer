# Changelog

## v3.1.0 - 2026-07-09

### 新增

- 新增 3D 自适应模板生成，覆盖 AAA、CT、CTA、MR、CBCT 等模板。
- 新增 Surface 渲染参数规范化和 modality/强度自适应 isoValue。
- 新增渲染时去床板 mask、自由形状视空间裁剪和 removeBed/clip 缓存 token。
- 新增 macOS 本地 demo 数据优先路径，部署环境继续使用默认 bundled sample。

### 改进

- 优化 VR/Surface preview/final 渲染状态复用，减少亮度、尺度和相机跳变。
- 优化 3D 旋转状态和 interactionId 抑制逻辑，避免旧 preview/final 覆盖新交互。
- 优化 AAA soft-detail 策略，增强低对比 CT 主体、圆形结构和高密度白色金属点显影。
- 优化 3D 初始相机 fit，按 volume bounds 和 viewport aspect 自动设置构图。

### 修复

- 修复 3D reset 固定回 Bone preset 的问题，改为恢复后端自动默认模板。
- 修复移动端/局域网开发时后端地址不随页面 host 变化导致连接失败的问题。
- 修复去床板算法误删中央主体结构的问题。

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
