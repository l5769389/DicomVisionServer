# DicomVision Server

[English](./README.en.md)

DicomVision Server 是 DicomVision 的 FastAPI + Socket.IO 后端，负责 DICOM 发现、PACS 查询与下载、2D/MPR/4D/3D 渲染、PET/CT 融合、分割、测量、QA、导出和桌面端内置后端 bundle。

## 架构

Server 是 DICOM 发现、渲染、视图状态、导出和计算密集型分析的权威执行层。它通过稳定的 REST + Socket.IO 接口服务桌面端、Web 端和移动端，既可独立部署，也可随桌面安装包内置运行。

- **渲染链路**：基于 VTK 的 2D/MPR/4D/3D 渲染，可选独立 GPU 进程，支持 WebRTC 交互预览和无损 WebP 最终帧。
- **安全导入**：接受 DICOM 文件、目录、ZIP、7z 与 RAR；在扫描前限制压缩包成员路径、条目数、解压体积和压缩比。
- **部署方式**：适用于本地、局域网、云服务、Docker 与桌面端内置后端。

## 仓库

- Server: [https://github.com/l5769389/DicomVisionServer](https://github.com/l5769389/DicomVisionServer)
- Client: [https://github.com/l5769389/DicomVisionClient](https://github.com/l5769389/DicomVisionClient)

## 主要能力

- DICOM 文件夹、单文件、Web 上传、ZIP/7z/RAR 压缩包和示例数据加载。
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
- `DICOMVISION_3D_TRANSPORT`：服务启动时固定 3D 帧传输方式，可选 `webp` 或 `webrtc`。
- `DICOMVISION_WEBRTC_VIDEO_CODEC` / `DICOMVISION_WEBRTC_VIDEO_BITRATE_BPS`：WebRTC 编码器和目标码率。
- `CORS_ORIGINS`：允许访问后端的前端来源，JSON 数组字符串，例如 `["http://localhost:5173"]`。
- `WEB_SAMPLE_DICOM_PATH`：Web demo 可加载的服务端示例 DICOM 路径。
- `WEB_UPLOAD_DICOM_ROOT`：浏览器上传 DICOM 的临时存储根目录。
- `WEB_UPLOAD_MAX_ARCHIVE_ENTRIES` / `WEB_UPLOAD_MAX_ARCHIVE_UNCOMPRESSED_BYTES` / `WEB_UPLOAD_MAX_ARCHIVE_COMPRESSION_RATIO`：压缩包导入安全限制。
- `VTK_RENDER_PROCESS_ENABLED`：是否启用独立 VTK GPU 渲染进程；macOS 默认开启。
- `VTK_SHARED_MEMORY_MAX_BYTES`：主进程与 GPU 进程之间的体数据共享内存上限，默认 1 GiB。
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

## VTK 渲染诊断与 Benchmark

macOS 默认在独立 GPU 进程中执行 3D VTK 渲染。服务启动日志会记录 OpenGL vendor、renderer、VTK mapper 模式，以及是否检测到软件渲染器。

使用合成体数据运行可重复基准：

```bash
uv run python scripts/benchmark_vtk_render.py --process --iterations 8
```

使用指定 DICOM 目录：

```bash
uv run python scripts/benchmark_vtk_render.py --process --dicom-path /path/to/dicom
```

报告分别统计 VTK render、GPU readback、WebP encode 和本地 Socket send。WebRTC 模式下，交互预览走视频轨道，操作稳定后使用无损 WebP 覆盖最终画面；传输基准会分别输出交互帧延迟和最终无损帧延迟。生产环境真实 Socket.IO 发送耗时记录在 `3d pipeline timing` 日志中。

## 测试

```bash
uv run pytest
```
