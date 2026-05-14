# DicomVision Server 项目逻辑梳理

这份文档说明后端的整体架构、API、运行时状态和图像处理链路。后端承担 DICOM 扫描、像素解码、窗宽窗位、伪彩、2D 仿射重采样、MPR 重切片、3D 体渲染、测量换算、MTF 与水模 QA 等主要计算工作；前端主要负责展示后端生成的 PNG/JPEG 和绘制交互叠加层。

## 1. 技术栈与入口

| 内容 | 位置 | 说明 |
| --- | --- | --- |
| 应用入口 | `app/main.py` | 创建 FastAPI app、注册路由、创建 Socket.IO server，并用 `socketio.ASGIApp` 包装。 |
| HTTP 框架 | FastAPI | REST API 负责加载、创建视图、尺寸设置、导出和分析任务。 |
| 实时通信 | python-socketio | 高频交互、图像推送、hover、4D 播放。 |
| DICOM | pydicom | 读取 header 和 pixel array。 |
| 数值处理 | NumPy、SciPy、Pillow | 图像窗口、伪彩、仿射变换、MPR 重采样、图像编码。 |
| 3D 渲染 | VTK | 体渲染离屏输出 PNG/JPEG。 |

后端启动后挂载的主要路径：

```text
FastAPI
  /health
  /api/v1/dicom/*
  /api/v1/view/*

Socket.IO
  connect / disconnect
  bind_view
  view_operation
  view_hover
  four_d_playback_*
```

## 2. 目录结构阅读地图

| 目录 | 作用 |
| --- | --- |
| `app/api/routes` | HTTP 路由。 |
| `app/sockets` | Socket.IO 事件、view 绑定、渲染推送、4D 播放。 |
| `app/schemas` | API 请求和响应模型。 |
| `app/models` | 后端运行时领域模型，例如序列、视图、MPR group、测量。 |
| `app/services` | 主要业务服务和图像处理模块。 |
| `app/services/render_layers` | 2D 渲染层，包括窗口、伪彩和部分叠加层。 |
| `app/services/mpr` | MPR 几何、光标、切面、重采样。 |
| `app/services/volume_rendering` | VTK 体渲染。 |
| `app/core` | 配置与日志。 |

建议阅读顺序：

1. `app/main.py`
2. `app/api/routes/dicom.py`、`app/api/routes/view.py`
3. `app/sockets/handlers.py`
4. `app/services/viewer_service.py`
5. `app/services/viewer_operation_handlers.py`
6. 图像专项模块：`dicom_cache.py`、`viewport_transformer.py`、`mpr/*`、`volume_rendering/*`

## 3. 后端运行时对象

### 3.1 `SeriesRecord`

位置：`app/models/viewer.py`

一个 DICOM 序列的后端记录，包含序列元数据、实例列表、4D 信息等。由 `SeriesRegistry` 创建和管理。

常见字段：

| 字段 | 说明 |
| --- | --- |
| `series_id` | 后端内部序列 ID。 |
| `instances` | 排序后的 `InstanceRecord` 列表。 |
| `modality`、`description`、`patient_name` | DICOM header 摘要。 |
| `is_four_d`、`phase_index`、`phase_label` | 4D 相关信息。 |

### 3.2 `InstanceRecord`

单个 DICOM 实例记录，包含文件路径、SOPInstanceUID、InstanceNumber、ImagePositionPatient、PixelSpacing 等。扫描文件夹时只读取 header，不立即解码像素。

### 3.3 `ViewRecord`

后端 view 的运行时状态。Stack、MPR、3D 都通过 `ViewRecord` 表示。

重要字段：

| 字段 | 说明 |
| --- | --- |
| `view_id` | 后端 view ID，前端通过它绑定 Socket 和发操作。 |
| `series_id` | 所属序列。 |
| `view_type` | `STACK`、`AXIAL`、`CORONAL`、`SAGITTAL`、`VOLUME_3D` 等。 |
| `width`、`height` | 当前前端视口尺寸。 |
| `current_index` | 当前 slice index。 |
| `window` | 窗宽窗位。 |
| `transform` | 平移、缩放、旋转、翻转。 |
| `mpr_group_id` | MPR 视图共享状态 ID。 |
| `measurements` | 当前 view 的测量记录。 |

### 3.4 `ViewGroupRecord`

MPR 的共享状态对象。AX/COR/SAG 三个 view 通过同一个 group 同步光标、窗宽窗位、伪彩、体数据几何、MIP 配置、斜切面状态等。

这也是 MPR 操作通常需要广播三视口重渲染的原因。

## 4. 总体数据流

### 4.1 加载序列

```text
POST /api/v1/dicom/loadFolder
  -> SeriesRegistry.load_folder()
      -> 扫描目录
      -> pydicom.dcmread(stop_before_pixels=True)
      -> 按 SeriesInstanceUID / 相对目录 / phase 分组
      -> 创建 SeriesRecord 和 InstanceRecord
      -> 返回 SeriesSummary[]
```

扫描阶段只读取 DICOM header，不解码完整像素。这样大目录加载更快，也避免一开始占用大量内存。

### 4.2 创建视图并首帧渲染

```text
POST /api/v1/view/create
  -> ViewRegistry.create()
      -> 创建 ViewRecord
      -> MPR 视图进入共享 ViewGroupRecord

Socket bind_view
  -> ViewSocketHub 记录 sid <-> viewId

POST /api/v1/view/setSize
  -> ViewerService.set_view_size()
  -> 后台触发 render_view_by_id()
  -> ViewSocketHub.emit_image_update()
  -> 前端收到 image_update
```

### 4.3 交互操作

```text
Socket view_operation
  -> ViewOperationRequest 校验
  -> handle_view_operation()
      -> 根据 opType 修改 ViewRecord / ViewGroupRecord
      -> 判断是否需要单视图渲染、MPR 广播渲染或只返回草稿
  -> render_view_by_id()
  -> image_update
```

拖拽类交互会区分 `start`、`move`、`end`。`move` 阶段常使用 JPEG 和快速预览，降低高频操作延迟；`end` 阶段回到更高质量渲染。

## 5. 图像处理主链路

### 5.1 DICOM 像素读取与缓存

位置：`app/services/dicom_cache.py`

`DicomCache.get(instance_uid, path)` 是读取单张 DICOM 像素的关键入口：

1. 用 pydicom 读取完整文件。
2. 调用 `dataset.pixel_array` 解码像素。
3. 转为 `float32`。
4. 若是多帧数组，默认取第一帧。
5. 应用 `RescaleSlope` 和 `RescaleIntercept`，CT 通常得到 HU 值。
6. 对 `MONOCHROME1` 做反相处理。
7. 记录 min/max、WindowWidth、WindowCenter。
8. 放入 LRU cache。

缓存策略：

| 参数 | 说明 |
| --- | --- |
| 最大条目 | 128 张。 |
| 最大内存 | 约 512 MB。 |
| 淘汰策略 | LRU，超过条目数或内存上限时移除旧数据。 |

### 5.2 Stack 2D 渲染

主要入口：`ViewerService._render_view()`

流程：

```text
当前 InstanceRecord
  -> DicomCache.get()
  -> 取 pixel array
  -> 计算窗口和伪彩
  -> build_image_to_canvas_transform()
  -> apply_affine_array()
  -> Pillow 编码 PNG/JPEG
  -> ViewImageResponse meta + image bytes
```

关键点：

| 步骤 | 模块 | 说明 |
| --- | --- | --- |
| 窗宽窗位 | `render_layers/base_image_layer.py` | 优先使用 view 的窗口，其次 DICOM header，最后 min/max fallback。 |
| 伪彩 | `services/pseudocolor.py` | 支持 `bw`、`bwinverse`、`blackbody`、`cardiac`、`hotiron`、`pet`、`rainbow`。 |
| 仿射变换 | `services/viewport_transformer.py` | 处理 contain zoom、平移、旋转、翻转、像素宽高比。 |
| 元数据 | `ViewerService` | 返回 corner、orientation、scaleBar、transform、measurements 等。 |

### 5.3 窗宽窗位与伪彩

窗口公式在 `BaseImageLayer` 中完成，核心思想：

```text
low = center - width / 2
high = center + width / 2
clipped = clip(pixel, low, high)
gray = normalize(clipped, 0..255)
```

随后根据伪彩 preset 将灰度图映射为 RGB。黑白模式是普通灰度；其他 preset 使用 LUT 映射。

### 5.4 2D 仿射变换

位置：`app/services/viewport_transformer.py`

`build_image_to_canvas_transform()` 会把图像坐标映射到画布坐标，组合顺序可以理解为：

```text
图像中心移到原点
  -> 缩放：contain zoom * 用户 zoom * 翻转 * 像素宽高比
  -> 旋转：0 / 90 / 180 / 270
  -> 平移到 canvas 中心
  -> 加上用户 pan offset
```

`apply_affine_array()` 使用 SciPy 的 `affine_transform` 做重采样。注意 SciPy 使用 row/col 顺序，而业务层常以 x/y 或 col/row 理解坐标，所以该模块里有显式矩阵换算。

### 5.5 MPR 体数据构建

主要入口：`ViewerService._get_series_volume()`

流程：

```text
SeriesRecord.instances
  -> 逐张 DicomCache.get()
  -> 得到 slice 数组列表
  -> 读取 ImageOrientationPatient / ImagePositionPatient / PixelSpacing
  -> build_standardized_volume()
  -> 缓存标准化体数据
```

体数据缓存上限约 1 GB。MPR、3D 和部分分析任务都会复用这份 volume。

### 5.6 DICOM 几何标准化

位置：`app/services/dicom_geometry.py`

如果 DICOM header 中存在可靠的方向和位置：

1. 读取 `ImageOrientationPatient` 得到 row/column 方向。
2. 用 row × column 得到 slice normal。
3. 根据 `ImagePositionPatient` 沿 normal 排序切片。
4. 构造 voxel 到 patient world 的坐标映射。
5. 将数组转成后端统一使用的标准 patient axis 表达。

如果方向信息缺失或不可靠，后端退回简单 stack 逻辑，仍能渲染，但方向、间距和 MPR 精度会受影响。

### 5.7 MPR 切面提取

核心模块：

| 模块 | 作用 |
| --- | --- |
| `services/mpr/geometry.py` | 保存 volume 几何，提供 IJK 和 world 坐标互转。 |
| `services/mpr/cursor.py` | MPR 光标位置和方向矩阵，支持平移、旋转、夹紧到体数据。 |
| `services/mpr/planes.py` | 从光标和 viewport 推导 AX/COR/SAG/Oblique 切面姿态。 |
| `services/mpr/reslice.py` | 按切面姿态对体数据做重采样，支持 MIP slab。 |

`ViewerService._extract_mpr_plane()` 是 MPR 的关键汇合点：

```text
ViewRecord / ViewGroupRecord
  -> 获取标准化 volume 和 geometry
  -> 获取或初始化 MPR cursor
  -> derive_plane_pose()
  -> reslice_plane()
  -> 返回 2D plane array + spacing + orientation metadata
```

MPR 三视口共享同一个 `ViewGroupRecord`，所以十字线、窗宽窗位、MIP、斜切面调整通常会触发三视口广播渲染。

### 5.8 MIP slab

位置：`app/services/mpr/reslice.py`

启用 MIP 后，后端会沿切面 normal 方向采样多个平行切面，然后按算法合成：

| 算法 | 说明 |
| --- | --- |
| `maximum` | 最大密度投影，常用于血管/高密度结构观察。 |
| `minimum` | 最小密度投影。 |
| `average` | 平均投影。 |
| `sum` | 求和投影。 |

slab 厚度以毫米为单位，后端根据体数据 spacing 计算采样步长。

### 5.9 3D 体渲染

位置：`app/services/volume_rendering/vtk_volume_renderer.py`

3D 视图不走 MPR 切片，而是把 volume 送入 VTK：

```text
standardized volume
  -> vtkImageData
  -> vtkSmartVolumeMapper
  -> transfer functions
  -> offscreen render
  -> RGB image bytes
```

关键特性：

| 内容 | 说明 |
| --- | --- |
| 离屏渲染 | 不依赖前端 WebGL，后端生成 2D 图像。 |
| 会话缓存 | 每个 view 有 VTK session，避免每次重建管线。 |
| 传递函数 | 根据 volume preset/config 构造颜色和不透明度。 |
| 交互 | `rotate3d`、pan、zoom 修改 VTK 相机。 |
| 方向 | 后端返回 `volumeQuaternion`，前端用它绘制方向立方体。 |

VTK 渲染通过单线程 executor 串行执行，避免 VTK 线程安全问题。

### 5.10 测量、hover 与像素坐标

后端收到前端归一化点后，需要映射回图像像素或 MPR plane 像素：

| 功能 | 主要入口 | 说明 |
| --- | --- | --- |
| Hover | `ViewerService.get_hover_info()` | 根据归一化位置计算 row/col/value。 |
| 测量 | `ViewerService` 测量相关方法 | 将归一化点转换为图像点，结合 spacing 计算长度、角度、面积等。 |
| 可见测量 | `_build_visible_measurements()` | 只返回当前 slice 或当前 MPR 上应该显示的测量。 |

Stack 测量通常基于当前 2D slice。MPR 测量需要结合切面 spacing 和当前 plane 上下文。

### 5.11 MTF 分析

位置：`app/services/mtf_analysis_service.py`

HTTP 入口：`POST /api/v1/view/mtf/analyze`

流程：

1. 校验 view 是否是 2D 类型。
2. 将 ROI 归一化点映射到图像像素坐标。
3. 截取 ROI。
4. 调用 `MtfAnalyzer.analyze_roi()`。
5. 返回 MTF50、MTF10、FWHM、曲线和单位。

如果能获得像素间距，单位为 `lp/mm`；否则退回 `lp/pixel`。

### 5.12 水模 QA

位置：`app/services/water_phantom_qa_service.py`

HTTP 入口：`POST /api/v1/view/qa/water/analyze`

主要流程：

1. 读取当前 2D 图像。
2. 用阈值和连通域检测水模主体。
3. 构建中心 ROI、外围 ROI 和空气 ROI。
4. 计算 CT 值准确性、均匀性、噪声等指标。
5. 将 ROI 反算成前端可绘制的归一化坐标。

该功能依赖图像中水模边界可被自动检测到。如果图像裁切、窗位或内容不适合，可能返回检测失败。

## 6. HTTP API 注释

### 6.1 DICOM API

位置：`app/api/routes/dicom.py`

| 方法和路径 | 请求/响应模型 | 说明 |
| --- | --- | --- |
| `POST /api/v1/dicom/loadFolder` | `LoadFolderRequest` -> `LoadFolderResponse` | 扫描指定文件夹，注册序列，返回 `seriesList`。 |
| `POST /api/v1/dicom/loadSample` | `LoadSampleRequest` -> `LoadFolderResponse` | 加载配置中的样例数据目录。 |
| `POST /api/v1/dicom/cornerInfo` | `CornerInfoRequest` -> `CornerInfoResponse` | 生成当前序列/实例角标信息。 |
| `GET /api/v1/dicom/thumbnail` | query `seriesId` | 返回序列中间层缩略图。 |
| `POST /api/v1/dicom/fourD/phases` | `FourDPhasesRequest` -> `FourDPhasesResponse` | 解析并返回 4D phase 列表。 |
| `GET /api/v1/dicom/fourD/preview` | query `seriesId`、`phaseIndex` | 返回某个 phase 的预览图。 |
| `POST /api/v1/dicom/tags` | `DicomTagsRequest` -> `DicomTagsResponse` | 读取某个实例的 DICOM tags，用于 Tag 页。 |

### 6.2 View API

位置：`app/api/routes/view.py`

| 方法和路径 | 请求/响应模型 | 说明 |
| --- | --- | --- |
| `POST /api/v1/view/create` | `ViewCreateRequest` -> `ViewCreateResponse` | 创建后端 view。MPR 会加入或创建共享 group。 |
| `POST /api/v1/view/close` | `ViewCloseRequest` -> `ViewCloseResponse` | 关闭 view，释放 VTK session，清理空 MPR group。 |
| `POST /api/v1/view/setSize` | `ViewSizeRequest` -> `ViewSizeResponse` | 更新视口尺寸，并异步触发 render。 |
| `POST /api/v1/view/mtf/analyze` | `MtfAnalyzeRequest` -> `MtfAnalyzeResponse` | 对当前 view 的 ROI 做 MTF 分析。 |
| `POST /api/v1/view/qa/water/analyze` | `WaterPhantomQaRequest` -> `WaterPhantomQaResponse` | 自动检测水模 ROI 并计算 QA 指标。 |
| `POST /api/v1/view/export` | `ExportViewRequest` | 导出 PNG 或 DICOM Secondary Capture。 |

`setSize` 需要注意：路由函数会立即返回，同时用后台任务延迟触发渲染并通过 Socket 推送图像。

### 6.3 重点 API 使用说明

| API | 后端主要入口 | 核心职责 | 设计边界 |
| --- | --- | --- | --- |
| `POST /dicom/loadFolder` | `SeriesRegistry.load_folder()` | 扫描 DICOM header、分组序列、识别 4D 信息、注册 `SeriesRecord`。 | 不批量解码像素，避免加载大目录时占用过多内存。 |
| `POST /view/create` | `ViewRegistry.create()` | 创建 `ViewRecord`，MPR 视图加入共享 `ViewGroupRecord`。 | 只创建状态，不保证立即有图像；图像渲染通常由 `setSize` 或 Socket 触发。 |
| `POST /view/setSize` | `ViewerService.set_view_size()` + `ViewSocketHub.emit_render_for_view()` | 更新 canvas 尺寸并异步渲染。 | HTTP 响应是确认消息，图像通过 Socket `image_update` 返回。 |
| Socket `bind_view` | `ViewSocketHub.bind_view()` | 建立 `sid -> viewId` 订阅关系。 | 只有绑定后的连接会收到对应 view 的 `image_update`。 |
| Socket `view_operation` | `viewer_operation_handlers.handle_view_operation()` | 处理滚动、调窗、缩放、MPR 十字线、3D 旋转、测量等高频交互。 | handler 返回渲染决策；Socket 层再决定单视图、广播或延迟渲染。 |
| Socket `image_update` | `ViewSocketHub._emit_render_message()` | 推送 `ViewImageResponse` meta 和 PNG/JPEG bytes。 | JPEG 多用于拖拽快速预览；PNG 用于稳定最终帧。 |
| `POST /view/mtf/analyze` | `MtfAnalysisService.analyze()` | 将 ROI 映射到图像像素并计算 MTF50/MTF10/FWHM。 | 依赖 2D 图像和 ROI，有 spacing 时输出 `lp/mm`。 |
| `POST /view/qa/water/analyze` | `WaterPhantomQaService.analyze()` | 自动检测水模 ROI 并计算准确性、均匀性、噪声。 | 检测失败时返回 error 状态，前端应展示失败原因。 |
| `POST /view/export` | `ViewerService.export_view_by_id()` | 按当前 view 状态重渲染并打包 PNG/DICOM。 | 前端 overlay 默认不在后端图像中，导出时需通过 request 传入。 |

## 7. Socket API 注释

位置：`app/sockets/handlers.py`

### 7.1 客户端到服务端

| 事件 | 请求模型/字段 | 说明 |
| --- | --- | --- |
| `connect` | Socket 内建 | 建立连接，后端返回 `connected`。 |
| `disconnect` | Socket 内建 | 解绑该 sid 关联的 view，停止对应 4D 播放。 |
| `bind_view` | `viewId` | 将 sid 和 view 绑定，用于后续图像推送。 |
| `set_view_size` | `viewId`、`width`、`height` | Socket 版本的尺寸同步，功能类似 HTTP `setSize`。 |
| `view_hover` | `viewId`、`point` | 查询 hover 像素坐标和值。 |
| `view_operation` | `ViewOperationRequest` | 所有高频交互操作的统一入口。 |
| `image_operation` | 兼容事件 | 通常转成或兼容图像操作。 |
| `four_d_playback_start` | `FourDPlaybackStartRequest` | 启动某 tab 的 4D 播放循环。 |
| `four_d_playback_stop` | `FourDPlaybackStopRequest` | 停止播放。 |
| `four_d_playback_fps` | `FourDPlaybackFpsRequest` | 修改播放 FPS。 |

### 7.2 服务端到客户端

| 事件 | 说明 |
| --- | --- |
| `connected` | 连接确认。 |
| `view_bound` | view 绑定成功。 |
| `view_ack` | 部分操作确认。 |
| `image_update` | 最重要事件，发送 `ViewImageResponse` meta 和二进制图像 bytes。 |
| `image_error` | 单个 view 图像生成失败。 |
| `render_error` | 渲染任务异常。 |
| `hover_info` | hover 查询结果。 |
| `measurement_draft` | 测量过程中的草稿几何和指标。 |
| `four_d_phase_index` | 4D 播放当前 phase index。 |
| `four_d_playback_state` | 4D 播放状态、FPS、错误信息。 |

## 8. `view_operation` 处理注释

位置：`app/services/viewer_operation_handlers.py`

`handle_view_operation()` 是交互处理总入口。它会根据 `opType` 修改 view 或 group 状态，并返回渲染决策。

| opType | 主要影响 | 渲染范围 |
| --- | --- | --- |
| `scroll` | Stack 改 slice；MPR 改当前切面/光标。 | Stack 单视图；MPR 通常广播。 |
| `pan` | 修改 `transform.offset_x/y`。 | 单视图，拖拽 move 可快速预览。 |
| `zoom` | 修改 `transform.zoom`。 | 单视图。 |
| `window` | 修改窗宽窗位。 | Stack 单视图；MPR/3D 可能广播或重建传递函数。 |
| `pseudocolor` | 修改伪彩 preset。 | MPR 共享状态会广播。 |
| `transform2d` | 旋转 90 度、水平/垂直翻转。 | 单视图或 MPR 当前视口。 |
| `crosshair` | 更新 MPR cursor 世界坐标。 | MPR 三视口广播。 |
| `mprMipConfig` | 修改 MIP slab 配置。 | MPR 三视口广播。 |
| `mprOblique` | 修改斜切面 normal 或拖拽状态。 | MPR 三视口广播。 |
| `mprStateSync` | 同步 MPR 状态，常用于 4D phase 切换。 | MPR 三视口广播。 |
| `rotate3d` | 更新 VTK 相机 quaternion。 | 3D 单视图。 |
| `volumePreset` | 切换体渲染预设。 | 3D 单视图。 |
| `volumeConfig` | 更新体渲染 transfer function。 | 3D 单视图。 |
| `measurement` | 创建、移动、完成、删除、清空测量。 | 草稿可不渲染；提交/删除后渲染。 |
| `reset` | 恢复默认状态。 | 单视图或 MPR 广播。 |

渲染决策由 handler 返回，Socket 层再通过 `ViewSocketHub` 合并和调度实际渲染。

## 9. 渲染推送与并发控制

位置：`app/sockets/runtime.py`

`ViewSocketHub` 管理 view 和 Socket sid 的绑定关系，也负责渲染任务合并。

关键设计：

| 机制 | 说明 |
| --- | --- |
| view 绑定 | 一个 view 可以绑定到一个或多个 sid。 |
| render lock | 同一个 view 同一时间只跑一个 render。 |
| pending render | 如果渲染中又来了新请求，会记录 pending，并在当前 render 结束后再渲染最新状态。 |
| 质量合并 | 多个请求冲突时选择更高质量的图像格式；只有双方都是 fast preview 才保持快速预览。 |
| 二进制推送 | `image_update` 同时发送 meta 和 image bytes。 |

这个设计避免了拖拽时大量 render 并发堆积，也保证用户松手后的最终画面会使用最新状态。

## 10. 导出逻辑

入口：`POST /api/v1/view/export`

`ViewerService.export_view_by_id()` 会按当前 view 状态重新渲染图像，然后根据请求导出：

| 格式 | 说明 |
| --- | --- |
| PNG | 导出当前视口图像，可叠加测量和标注。 |
| DICOM | 生成 Secondary Capture DICOM。 |

`_apply_export_overlays()` 会把前端传入的部分叠加层绘制到导出图片上。注意日常浏览时，多数叠加层在前端显示；导出时才需要后端把它们烘焙到图片里。

## 11. 常见开发定位

| 想改的功能 | 优先查看 |
| --- | --- |
| 文件夹扫描、序列识别 | `app/services/series_registry.py` |
| DICOM 像素解码和缓存 | `app/services/dicom_cache.py` |
| Stack 渲染 | `ViewerService._render_view()`、`render_layers/base_image_layer.py` |
| 缩放/平移/旋转/翻转 | `app/services/viewport_transformer.py` |
| MPR 切面、十字线、斜切 | `ViewerService._extract_mpr_plane()`、`app/services/mpr/*` |
| 3D 体渲染 | `app/services/volume_rendering/vtk_volume_renderer.py` |
| Socket 交互 | `app/sockets/handlers.py`、`app/services/viewer_operation_handlers.py` |
| 4D phase 识别 | `app/services/four_d_service.py` |
| MTF 分析 | `app/services/mtf_analysis_service.py` |
| 水模 QA | `app/services/water_phantom_qa_service.py` |
| 导出 | `ViewerService.export_view_by_id()` |

## 12. 重要边界和注意事项

1. 文件夹扫描阶段只读 header；像素在渲染或分析时才按需读取。
2. Stack 图像是单张 DICOM slice 的窗口、伪彩和仿射结果。
3. MPR 图像来自标准化 volume 的任意切面重采样，不是简单取原始某张 slice。
4. 3D 图像来自 VTK 离屏体渲染，前端只显示结果图。
5. MPR 三视口共享 group 状态，很多操作必须广播渲染。
6. 高频交互通过 Socket 走 `view_operation`，低频任务和分析 API 通过 HTTP。
7. 前端显示的叠加层不一定存在于后端渲染图里；导出时需要显式传入 overlay 数据。
8. DICOM 几何信息缺失时仍能显示，但方向标记、真实间距和 MPR 结果会降级。

理解这几个边界后，再看 `viewer_service.py` 会容易很多：它是后端图像处理的中枢，负责把序列、view 状态、DICOM 像素、几何、渲染器和返回给前端的元数据连接起来。

