# DicomVision Server

[English](./README.en.md)

DicomVision Server 是 DicomVision 的 FastAPI + Socket.IO 后端，负责 DICOM 发现、PACS 查询与下载、2D/MPR/4D/3D 渲染、PET/CT 融合、分割、测量、QA、导出和桌面端内置后端 bundle。

## v3.0.0 后端更新

- **PET/CT Fusion**：支持 CT/PET/Fusion/PET MIP 多视口渲染、PET-only 显示、PET 强度范围、融合预览、手动配准、配准保存和 Socket.IO 交互。
- **MPR 分割与 VOI**：支持阈值分割、球形 VOI、overlay render intent、分割预览 metadata 和 sidecar 风格的数据流。
- **MPR/4D/播放**：增强 MPR、4D MPR、切片播放、时相同步和视口尺寸更新稳定性。
- **QA/MTF**：继续提供 MTF/FWHM、水模 QA、ROI 指标和报告数据，前端 v3.0.0 会以右侧报告区呈现。
- **PACS 与导出**：保留 DICOMweb/DIMSE 查询下载、Tag 修改、脱敏导出、DICOM SR/GSPS 和 PNG/DICOM 导出能力。
- **桌面 bundle**：可构建 Windows Server bundle，并被 Electron 桌面安装包内置启动。

## 仓库

- Server: [https://github.com/l5769389/DicomVisionServer](https://github.com/l5769389/DicomVisionServer)
- Client: [https://github.com/l5769389/DicomVisionClient](https://github.com/l5769389/DicomVisionClient)

## 主要能力

- DICOM 文件夹、单文件、Web 上传和示例数据加载。
- 缩略图、角标、DICOM Tag、序列、实例和 4D phase metadata 服务。
- PACS DICOMweb QIDO/WADO 与 DIMSE C-ECHO/C-FIND/C-GET。
- 2D、Compare、Layout、MPR、斜切 MPR、MIP、3D volume rendering、4D phase 和 PET/CT Fusion。
- 测量 ROI 指标、MTF/FWHM、水模 QA 和实时 hover/draft 交互。
- DICOM Tag 修改、批量任务、脱敏任务、DICOM SR/GSPS 和图像导出。
- Socket.IO 实时图像推送、view ack、错误事件和播放状态同步。

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
