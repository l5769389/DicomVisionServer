# DicomVision Server

[English](./README.en.md)

DicomVision Server 是 DicomVision 的后端服务，负责 DICOM 序列发现、PACS DICOMweb/DIMSE 查询与序列取回、缩略图与标签读取、DICOM Tag 修改、脱敏导出、Stack/Compare 渲染、MPR/斜切 MPR 重建、4D 时相协同、VTK 3D 体渲染、测量计算、MTF/FWHM、水模 QA、图像导出，以及通过 Socket.IO 向前端实时推送图像帧和交互结果。

## 适合什么场景

DicomVision Server 的设计重点是把 DICOM 解码、体数据重建、PACS 缓存、MPR/4D/3D 等重计算放到后端，因此更适合这些情况：

- 阅片端设备是集显、低端独显或显存较小，前端直接做复杂渲染容易卡顿。
- 希望在服务器或桌面端内置后端中集中处理 DICOM 数据，并让前端保持相对轻量。
- 需要 PACS DICOMweb/DIMSE 查询、下载缓存、Tag 编辑、脱敏导出、DICOM SR/GSPS 导出等服务端能力。
- 需要 Electron 桌面端自动启动内置后端，而不是要求用户手动部署独立服务。

如果使用场景是高性能前端机器、较充足的显存、纯浏览器部署，并且希望体数据完全由前端持有和渲染，更建议优先考虑 Cornerstone3D（C3D）这类完全前端方案。DicomVision Server 的价值在于为低显存阅片端和需要集中式处理的场景提供后端支撑。

## 主要能力

- **DICOM 数据服务**：加载本地目录、单个 DICOM 文件或浏览器上传文件，发现序列、生成缩略图、读取实例级 DICOM 标签。
- **PACS 集成**：通过 DICOMweb 或 DIMSE 查询检查和序列，将 WADO 或 C-GET 取回的序列写入服务端缓存，并复用本地目录加载管线完成注册。
- **Stack 渲染**：根据视口尺寸、窗宽窗位、伪彩、旋转、翻转、缩放和平移状态生成 2D 图像。
- **Stack Compare**：为双序列对比维护独立 source/target Stack 视图，并接收前端同步发送的滚动、窗宽窗位、伪彩、缩放、平移和变换操作。
- **MPR / 斜切 MPR**：构建标准化体数据，支持轴位、冠状位、矢状位重建、十字线同步、斜切旋转和 MIP 配置。
- **4D 支持**：识别多时相序列，生成时相列表和预览图，并通过 Socket.IO 支持前端播放控制。
- **3D 体渲染**：基于 VTK 执行服务端体渲染，支持预设、传输函数、光照、插值、混合模式和图层配置。
- **测量和 QA**：计算线段、矩形、椭圆、角度、曲线、自由形状 ROI 指标，支持 MTF/FWHM 和水模 QA。
- **DICOM 标准对象**：支持测量结果导出为 DICOM SR，测量/标注导出为 DICOM GSPS，并支持导入 GSPS overlay。
- **非影像对象导入**：单独导入 DICOM SR 或未能附着到原始影像的 GSPS 时，会注册为非影像 DICOM 文档对象，供前端以 Tags 视图查看。
- **后台任务**：长耗时 Tag 修改和脱敏导出以后台任务执行，提供可轮询进度和可下载 artifact。
- **桌面 bundle**：可打包为 Windows/macOS 桌面端后端 bundle，供 Electron 客户端内置启动。

## 仓库

- 服务端：[https://github.com/l5769389/DicomVisionServer](https://github.com/l5769389/DicomVisionServer)
- 客户端：[https://github.com/l5769389/DicomVisionClient](https://github.com/l5769389/DicomVisionClient)

## 产品截图

截图由配套客户端仓库维护，后端 README 使用客户端仓库中的图片展示整体效果。

| Stack 阅片 | MPR 重建 |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/stack.png" alt="Stack 阅片" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/mpr.png" alt="MPR 重建" width="420"> |

| PACS 数据源 | PACS Browser 导入 |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/pacs_dicom_import.png" alt="PACS DICOMweb 和 DIMSE Profile 设置" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/pacs_dicom_import_1.png" alt="PACS Browser 查询并打开已下载序列" width="420"> |

| 斜切 MPR / 十字线旋转 | 4D 时相播放 |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/mpr_rotate.png" alt="斜切 MPR 与十字线旋转" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/4D.png" alt="4D 时相播放" width="420"> |

| DICOM 标签 | 脱敏导出 |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/dicomTags.png" alt="DICOM 标签" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/deIndentifyExport.png" alt="DICOM 脱敏导出" width="420"> |

## 架构

后端提供两层通信能力：

- HTTP API：处理加载目录、上传文件、读取标签、获取缩略图、创建/关闭视图、设置视图尺寸、分析和导出等请求。
- Socket.IO：处理低延迟交互命令，并向已绑定视口的前端会话推送实时渲染结果。

典型流程：

1. 前端调用 `POST /api/v1/dicom/loadFolder`、`POST /api/v1/dicom/upload` 或 `POST /api/v1/dicom/loadSample` 注册数据。
2. 后端扫描 DICOM 文件并维护序列、实例和时相元数据。
3. 前端调用 `POST /api/v1/view/create` 创建 Stack、MPR、3D 或其他视图。
4. 前端通过 `bind_view` 将 Socket.IO 会话绑定到视图。
5. 前端发送 `view_operation`、`image_operation`、`view_hover` 或 4D 播放事件。
6. 后端返回 `image_update`、`hover_info`、`measurement_draft`、`view_ack`、4D 播放状态或错误事件。

## 快速开始

```bash
uv sync
uv run python run.py
```

默认端点：

- HTTP：`http://127.0.0.1:8000`
- OpenAPI：`http://127.0.0.1:8000/docs`
- Socket.IO：`http://127.0.0.1:8000/socket.io`
- 健康检查：`http://127.0.0.1:8000/health`

## 常用 API

除健康检查外，常用业务路由默认挂载在 `/api/v1` 下：

- `GET /health`
- `POST /dicom/loadFolder`
- `POST /dicom/upload`
- `POST /dicom/loadSample`
- `POST /dicom/tags`
- `POST /dicom/modifyTag`
- `POST /dicom/modifyTag/jobs`
- `GET /dicom/modifyTag/jobs/{job_id}`
- `GET /dicom/modifyTag/jobs/{job_id}/artifact`
- `POST /view/create`
- `POST /view/resize`
- `POST /view/operation`
- `POST /view/export`
- `POST /analysis/mtf`
- `POST /analysis/waterPhantom`
- `GET /pacs/profiles`
- `POST /pacs/dicomweb/studies`
- `POST /pacs/dicomweb/series`
- `POST /pacs/dimse/studies`
- `POST /pacs/dimse/series`

## 配置

常用环境变量：

- `CORS_ORIGINS`：允许访问后端的前端来源，JSON 数组字符串，如 `["http://localhost:5173"]`。
- `WEB_SAMPLE_DICOM_PATH`：Web 演示模式下可加载的服务端示例 DICOM 路径。
- `DICOMVISION_PACS_CACHE_ROOT`：PACS 下载缓存目录。
- `DICOMVISION_PACS_CACHE_TTL_SECONDS`：PACS 缓存保留时间。

## 部署

后端可以作为 Web 服务部署，也可以作为桌面端内置 bundle。

Render / Fly.io / 自托管部署时需要保证：

- 服务端可以访问 DICOM 数据或 PACS 网络。
- 前端域名已加入 `CORS_ORIGINS`。
- 如果要使用 Web 上传，需要代理/平台允许上传足够大的请求体。
- 如果要使用 PACS DIMSE，需要部署环境允许对应端口和网络访问。

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

客户端 `npm run release:win` 和 `npm run release:mac` 会在打包桌面端时调用后端 bundle 构建流程。

## 测试

```bash
uv run pytest
```
