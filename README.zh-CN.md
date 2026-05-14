# DicomVision Server

[English](./README.md)

DicomVision Server 是 DicomVision 的后端服务，提供 DICOM 序列发现、缩略图与标签读取、Stack 渲染、MPR/斜切 MPR 重建、4D 时相预览与播放协同、VTK 3D 体渲染、测量结果计算、MTF/FWHM 分析、水模 QA、图像导出，以及通过 Socket.IO 向前端实时推送图像帧和交互结果的能力。

## 仓库地址

- 服务端 Server：[https://github.com/l5769389/DicomVisionServer](https://github.com/l5769389/DicomVisionServer)
- 客户端 Client：[https://github.com/l5769389/DicomVisionClient](https://github.com/l5769389/DicomVisionClient)

## 功能总览

- **DICOM 数据服务**：加载本地目录或服务端示例目录，发现序列、生成缩略图、读取实例级 DICOM 标签。
- **Stack 渲染**：根据视口尺寸、窗宽窗位、伪彩、旋转、翻转、缩放和平移状态生成 2D 图像。
- **MPR / 斜切 MPR**：构建标准化体数据，支持轴位、冠状位、矢状位重建，十字线同步，斜切旋转和 MIP 配置。
- **4D 支持**：识别多时相序列，生成时相列表和预览图，并通过 Socket.IO 支持前端播放控制。
- **3D 体渲染**：基于 VTK 执行服务端体渲染，支持体渲染预设、传输函数、光照、插值、混合模式和图层配置。
- **测量与质量分析**：计算线段、矩形、椭圆、角度、曲线、自由形状 ROI 指标，支持 MTF/FWHM 和水模 QA 分析。
- **实时交互**：通过 Socket.IO 处理滚动、窗宽窗位、缩放、平移、十字线、斜切、3D 旋转、悬停和测量草稿。
- **部署与打包**：可作为 Web 后端部署到 Render，也可构建 Windows 桌面 bundle 供 Electron 客户端内置。

## 产品截图

截图由配套客户端仓库维护，后端 README 使用客户端仓库中的图片资源展示整体效果。

| Stack 阅片 | MPR 重建 |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/stack.png" alt="Stack 阅片" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/mpr.png" alt="MPR 重建" width="420"> |

| 斜切 MPR / 十字线旋转 | 4D 时相播放 |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/mpr_rotate.png" alt="斜切 MPR 与十字线旋转" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/4D.png" alt="4D 时相播放" width="420"> |

| 测量工具 | DICOM 标签 |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/measure.png" alt="测量工具" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/dicomTags.png" alt="DICOM 标签" width="420"> |

| MTF 分析 | FWHM 结果 |
| --- | --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/mtf.png" alt="MTF 分析" width="420"> | <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/mtf_fwhm.png" alt="FWHM 结果" width="420"> |

| 水模 QA |
| --- |
| <img src="https://raw.githubusercontent.com/l5769389/DicomVisionClient/main/screenshots/water_phantom_qa.png" alt="水模 QA" width="420"> |

## 系统架构

后端提供两层通信能力：

- HTTP API：处理加载目录、读取标签、获取缩略图、创建/关闭视口、设置尺寸、分析和导出等请求。
- Socket.IO：处理低延迟交互命令，并向已绑定视口的前端会话推送实时渲染结果。

典型流程：

1. 前端调用 `POST /api/v1/dicom/loadFolder` 或 `POST /api/v1/dicom/loadSample` 注册数据。
2. 后端扫描 DICOM 文件并维护序列、实例和时相元数据。
3. 前端调用 `POST /api/v1/view/create` 创建 Stack、MPR、3D 或其它视口。
4. 前端通过 `bind_view` 将 Socket.IO 会话绑定到视口。
5. 前端发送 `view_operation`、`image_operation`、`view_hover` 或 4D 播放事件。
6. 后端返回 `image_update`、`hover_info`、`measurement_draft`、`view_ack`、4D 播放状态或错误事件。

## 技术栈

- Python 3.13+
- FastAPI
- python-socketio
- pydicom
- NumPy
- SciPy
- Pillow
- VTK
- uv
- PyInstaller（桌面 bundle 构建时使用）

## 目录结构

```text
app/
  api/routes/              HTTP 路由
  core/                    配置、常量、日志
  models/                  内存运行时模型
  schemas/                 请求与响应模型
  services/                DICOM 处理、渲染、分析和注册表
  services/render_layers/  叠加层渲染
  services/volume_rendering/ VTK 体渲染
  sockets/                 Socket.IO 事件与实时推送
  utils/                   通用工具
sample-data/               Web 部署可用的示例数据目录
scripts/                   桌面 bundle 构建与 API 类型生成脚本
tests/                     自动化测试
run.py                     本地启动入口
render.yaml                Render 部署清单
pyproject.toml             项目元数据与依赖
```

## 快速开始

### 环境要求

- Python 3.13 或更高版本
- 可运行 VTK 的系统环境
- 对待加载 DICOM 目录具备读取权限

### 安装依赖

```bash
uv sync
```

如需开发依赖：

```bash
uv sync --extra dev
```

### 启动服务

```bash
uv run python run.py
```

默认地址：

- HTTP：`http://127.0.0.1:8000`
- OpenAPI：`http://127.0.0.1:8000/docs`
- ReDoc：`http://127.0.0.1:8000/redoc`
- Socket.IO：`http://127.0.0.1:8000/socket.io`

## 配置

可按需创建 `.env`：

```env
APP_NAME=DicomVision Server
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
CORS_ORIGINS=["*"]
WEB_SAMPLE_DICOM_PATH=
```

关键配置：

- `APP_ENV`：运行环境，生产环境一般设为 `production`。
- `APP_HOST`：监听地址。
- `APP_PORT`：监听端口，默认 `8000`。
- `CORS_ORIGINS`：允许访问 HTTP 和 Socket.IO 的前端来源。
- `WEB_SAMPLE_DICOM_PATH`：`POST /api/v1/dicom/loadSample` 使用的服务端示例 DICOM 目录。

## HTTP API

基础路径：`/api/v1`

- `GET /health`
- `POST /dicom/loadFolder`
- `POST /dicom/loadSample`
- `POST /dicom/cornerInfo`
- `GET /dicom/thumbnail`
- `POST /dicom/fourD/phases`
- `GET /dicom/fourD/preview`
- `POST /dicom/tags`
- `POST /view/create`
- `POST /view/close`
- `POST /view/setSize`
- `POST /view/mtf/analyze`
- `POST /view/qa/water/analyze`
- `POST /view/export`

准确请求与响应结构以 `/docs` 中的 OpenAPI 文档为准。

## Socket.IO 事件

客户端发送：

- `bind_view`
- `set_view_size`
- `view_operation`
- `image_operation`
- `view_hover`
- `four_d_playback_start`
- `four_d_playback_stop`
- `four_d_playback_fps`

服务端回推：

- `connected`
- `view_bound`
- `view_ack`
- `image_update`
- `hover_info`
- `measurement_draft`
- `four_d_phase_index`
- `four_d_playback_state`
- `image_error`
- `render_error`

## Web 部署

仓库内提供 `render.yaml`，可将服务端部署到 Render。推荐环境变量：

```env
APP_ENV=production
APP_HOST=0.0.0.0
APP_PORT=10000
CORS_ORIGINS=["https://your-vercel-app.vercel.app"]
WEB_SAMPLE_DICOM_PATH=/opt/render/project/src/sample-data
```

Web 前端部署时需要：

- 在前端配置 `VITE_BACKEND_ORIGIN` 指向后端地址。
- 将前端域名写入后端 `CORS_ORIGINS`。
- 如果 Web 端使用示例数据，配置 `WEB_SAMPLE_DICOM_PATH` 并在前端设置 `VITE_WEB_USE_SERVER_SAMPLE=true`。

## Fly.io 部署

仓库内已提供 `fly.toml`、`Dockerfile` 和 `.dockerignore`。推荐优先使用 Fly CLI 部署，因为 CLI 会输出完整构建与启动日志。

### 1. 安装并登录 Fly CLI

Windows PowerShell：

```powershell
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
fly auth login
```

### 2. 创建或确认应用名

`fly.toml` 中默认应用名为：

```toml
app = "dicomvision-server-l5769389"
```

Fly app 名称需要全局唯一。如果创建失败，可改成自己的名称，例如：

```toml
app = "dicomvision-server-yourname"
```

然后创建应用：

```powershell
fly apps create dicomvision-server-l5769389
```

如果该名称已存在，换一个名称后同时修改 `fly.toml`。

### 3. 配置跨域来源

开发阶段可使用当前默认值：

```toml
CORS_ORIGINS = "[\"*\"]"
```

生产环境建议改为你的前端域名：

```powershell
fly secrets set CORS_ORIGINS='["https://your-vercel-app.vercel.app"]'
```

如果使用仓库内示例数据，`fly.toml` 已配置：

```toml
WEB_SAMPLE_DICOM_PATH = "/app/sample-data/test"
```

### 4. 部署

```powershell
fly deploy
```

部署完成后验证：

```powershell
fly status
fly logs
fly open
```

健康检查地址：

```text
https://dicomvision-server-l5769389.fly.dev/health
```

### GitHub 页面部署注意事项

如果使用 Fly.io 的 GitHub Deploy 页面：

- `Working directory` 填 `./`
- `Config path` 填 `fly.toml` 或 `./fly.toml`，不要填 `./`
- `Internal port` 填 `8000`
- `Machine Size` 建议至少 `shared-cpu-1x / 2GB`
- 先把 `fly.toml`、`Dockerfile`、`.dockerignore` push 到 GitHub，否则网页部署读不到这些配置
- 如果提示 `Failed to create app`，优先检查 app 名是否已被占用，或仓库名/配置里的 app 名是否需要改成全小写加短横线的唯一名称

## 桌面 bundle

此仓库可构建供 Electron 客户端安装包内置的 Windows 后端 bundle。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-desktop-bundle.ps1
```

默认输出：

```text
dist/
  DicomVisionServer/
    DicomVisionServer.exe
    ...
```

如需指定输出目录：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-desktop-bundle.ps1 -OutputRoot .\artifacts
```

随后可通过客户端的 `DICOM_VISION_SERVER_BUNDLE_PATH` 或 `npm run release:win` 打入 Electron 安装包。

## 测试

```bash
uv run pytest
```
