# DicomVision Server

[English](./README.en.md)

DicomVision Server 是 DicomVision 的 FastAPI + Socket.IO 后端，负责 DICOM 发现、PACS 查询与下载、2D/MPR/4D/3D 渲染、PET/CT 融合、分割、测量、QA、导出和桌面端内置后端 bundle。

## v3.1.0 后端更新

- **3D 渲染一致性**：VR/Surface 预览和最终帧复用同一视图状态，降低旋转结束后的亮度、尺度和姿态跳变。
- **3D 旋转与相机**：支持模型直接拖拽旋转、interactionId 防旧帧覆盖、移动端视口适配和根据体数据范围自动初始构图。
- **自适应 3D 模板**：AAA、CT、CTA、MR、CBCT 等模板使用 CT HU 锚点 + 前景百分位混合策略；非 HU 数据回退到百分位策略。
- **Surface 参数**：Surface 使用独立 isoValue、平滑、decimation、颜色和材质参数，并支持按 modality/强度分布生成合理默认值。
- **去床板与裁剪**：新增渲染时去床板 mask、自由形状贯穿裁剪、clip/removeBed 缓存 token、预处理进度和 timing log。
- **Web demo 数据**：macOS 本地开发优先使用 `/Users/jun/Documents/test_dicom/py_test_path/py_test_path2`，部署环境继续使用项目默认样例。
- **桌面 bundle**：继续支持 Windows/macOS Server bundle，并可被 Electron 桌面安装包内置启动。

## 仓库

- Server: [https://github.com/l5769389/DicomVisionServer](https://github.com/l5769389/DicomVisionServer)
- Client: [https://github.com/l5769389/DicomVisionClient](https://github.com/l5769389/DicomVisionClient)

## 主要能力

- DICOM 文件夹、单文件、Web 上传和示例数据加载。
- 缩略图、角标、DICOM Tag、序列、实例、4D phase 和视图 metadata 服务。
- PACS DICOMweb QIDO/WADO 与 DIMSE C-ECHO/C-FIND/C-GET。
- 2D、Compare、Layout、MPR、斜切 MPR、MIP、3D VR、3D Surface、4D phase 和 PET/CT Fusion。
- 3D 自适应模板、Surface 参数、去床板、自由裁剪、相机重置和移动端视口适配。
- 测量 ROI 指标、MTF/FWHM、水模 QA 和实时 hover/draft 交互。
- MPR 阈值分割、VOI、分割 overlay metadata 和导入/导出数据流。
- DICOM Tag 修改、批量任务、脱敏任务、DICOM SR/GSPS 和图像导出。
- Socket.IO 实时图像推送、view ack、进度事件、错误事件和播放状态同步。

## 产品截图

截图维护在 Client 仓库中，Server README 使用这些图片展示整体产品效果。

| PET/CT 融合 | PET/CT 手动配准 |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/pet_ct_fusion.png" alt="PET/CT Fusion" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/pet_ct_fusion_registration.png" alt="PET/CT manual registration" width="420"> |

| MPR / 斜切 | 分割与 VOI |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/mpr_rotate.png" alt="MPR oblique rotation" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/segmentation_voi.png" alt="Segmentation and VOI" width="420"> |

| 4D | MTF/FWHM |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/4D.png" alt="4D phase playback" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/mtf_fwhm_1.png" alt="MTF and FWHM" width="420"> |

| PACS Browser | 移动端 PET/CT |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/pacs_dicom_import_1.png" alt="PACS Browser" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/mobile_pet_ct_fusion.png" alt="Mobile PET/CT Fusion" width="260"> |

## 快速开始

```bash
uv sync
uv run python run.py
```

默认地址：

- HTTP: `http://127.0.0.1:8000`
- OpenAPI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Socket.IO: `http://127.0.0.1:8000/socket.io`
- 健康检查: `http://127.0.0.1:8000/health`

## 配置

常用环境变量：

- `APP_ENV`：运行环境，部署时通常为 `production`。
- `APP_HOST` / `APP_PORT`：监听地址和端口。
- `CORS_ORIGINS`：允许访问后端的前端来源，JSON 数组字符串，例如 `["http://localhost:5173"]`。
- `WEB_SAMPLE_DICOM_PATH`：Web demo 可加载的服务端示例 DICOM 路径。
- `WEB_UPLOAD_DICOM_ROOT`：浏览器上传 DICOM 的临时存储根目录。
- `DICOMVISION_PACS_CACHE_ROOT`：PACS 下载缓存目录。
- `DICOMVISION_PACS_CACHE_TTL_SECONDS`：PACS 缓存保留时间。

## 常用 API

基础路径：`/api/v1`

- `POST /dicom/loadFolder`
- `POST /dicom/upload`
- `POST /dicom/loadSample`
- `POST /dicom/tags`
- `POST /pacs/dicomweb/studies`
- `POST /pacs/dicomweb/series`
- `POST /pacs/dimse/studies`
- `POST /pacs/dimse/series`
- `POST /view/create`
- `POST /view/close`
- `POST /view/setSize`
- `POST /view/operation`
- `POST /view/export`
- `POST /view/mtf/analyze`
- `POST /view/qa/water/analyze`

精确请求和响应结构请查看 `/docs`。

## 桌面 bundle

Windows：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-desktop-bundle.ps1
```

跨平台 Python 脚本：

```bash
python3 scripts/build_desktop_bundle.py
```

默认输出：

```text
dist/
  DicomVisionServer/
    DicomVisionServer.exe  # Windows
    DicomVisionServer      # macOS
    ...
```

Client 的 `npm run release:win` 会构建 Server bundle，并把它打入 Windows 桌面安装包。

## 测试

```bash
uv run pytest
```
