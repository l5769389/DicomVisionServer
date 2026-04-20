# DicomVision Server

[English](./README.md)

DicomVision Server 是 DicomVision 阅片系统的后端仓库，负责 DICOM 序列加载、视口生命周期管理、2D 图像渲染、MPR 重建、3D 体渲染，以及通过 Socket.IO 向前端实时推送图像与交互结果。

配套前端仓库 GitHub 地址：[DicomVisionClient](https://github.com/l5769389/DicomVisionClient)

## 项目概述

DicomVision 采用前后端分离的医学影像查看架构：

- `DicomVisionServer`：基于 FastAPI + Socket.IO 的后端服务，负责数据加载、渲染和交互处理
- `DicomVisionClient`：基于 Electron + Vue 的前端应用，负责工作流编排、视口交互和结果展示

后端是整个系统的渲染核心，承担图像计算、视图同步与实时事件分发；前端负责用户操作、工作区管理与渲染结果呈现。

## 软件功能

### DICOM 数据服务

- 加载本地 DICOM 文件夹并发现可读序列
- 在内存中维护序列元数据与运行时注册表
- 为 Web 部署场景提供示例数据加载入口

### 视口与渲染能力

- 创建并管理 `Stack`、`MPR`、`3D` 三类视口
- 渲染 2D Stack 图像与叠加层
- 提供正交重建能力以支持 MPR 工作流
- 基于 VTK 在后端执行 3D 体渲染

### 实时交互处理

- 通过 Socket.IO 提供低延迟交互链路
- 处理平移、缩放、滚动、悬停、重置、十字线与图像操作
- 将图像帧、叠加信息和确认事件实时回推给前端

### 3D 渲染配置

- 提供 `aaa`、`red`、`cardiac`、`muscle`、`mip` 等内置预设
- 支持传输函数归一化和应用
- 支持混合模式、光照、插值、不透明度、颜色、图层等配置
- 支持快速预览与完整质量渲染两条路径

## 系统架构

后端对外暴露两层通信能力：

- HTTP API：处理加载目录、创建视口、设置尺寸等粗粒度操作
- Socket.IO：处理交互命令和实时渲染更新

典型请求流程如下：

1. 前端调用 `POST /api/v1/dicom/loadFolder` 注册目录。
2. 前端调用 `POST /api/v1/view/create` 创建视口。
3. 通过 `POST /api/v1/view/setSize` 或 `set_view_size` 设置视口尺寸。
4. 前端通过 `bind_view` 绑定 socket 会话。
5. 交互事件通过 `view_operation`、`image_operation` 或 `view_hover` 发送到后端。
6. 后端回推 `image_update`、`hover_info`、`view_ack` 或错误事件。

后端核心职责包括：

- 注册并索引可读 DICOM 序列
- 管理 `Stack`、`MPR`、`3D` 视口生命周期与状态
- 渲染图像帧与叠加信息
- 同步多视图交互状态
- 归一化并应用 3D 渲染配置

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

## 目录结构

```text
app/
  api/routes/              HTTP 路由
  core/                    配置、常量、日志
  models/                  内存运行时模型
  schemas/                 请求与响应模型
  services/                DICOM 处理、渲染与注册表
  sockets/                 Socket.IO 事件与运行时中枢
  utils/                   通用工具
sample-data/               Web 部署可用的示例数据
tests/                     自动化测试
run.py                     本地启动入口
render.yaml                Render 部署清单
pyproject.toml             项目元数据与依赖
```

## 核心模块

- `series_registry`：序列发现与元数据管理
- `view_registry`：视口实例生命周期管理
- `dicom_cache`：解码像素缓存
- `viewer_service`：渲染与交互处理的核心编排层
- `view_socket_hub`：按视口定向绑定与推送
- `app/services/volume_rendering/`：基于 VTK 的体渲染管线

## 快速开始

### 环境要求

- Python 3.13 或更高版本
- 可运行 VTK 的环境
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

默认服务地址：

- HTTP：`http://127.0.0.1:8000`
- OpenAPI：`http://127.0.0.1:8000/docs`
- ReDoc：`http://127.0.0.1:8000/redoc`
- Socket.IO：`http://127.0.0.1:8000/socket.io`

## 配置说明

按需创建本地 `.env` 文件，常用配置如下：

```env
APP_NAME=DicomVision Server
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
CORS_ORIGINS=["*"]
WEB_SAMPLE_DICOM_PATH=
```

关键配置项：

- `APP_ENV`：控制开发态行为，例如自动重载
- `APP_HOST`：服务监听地址
- `APP_PORT`：服务端口，默认 `8000`
- `CORS_ORIGINS`：HTTP 与 Socket.IO 允许的来源
- `WEB_SAMPLE_DICOM_PATH`：`POST /api/v1/dicom/loadSample` 使用的示例目录

## API 与实时事件

HTTP 基础路径：`/api/v1`

### HTTP API

- `GET /health`
- `POST /dicom/loadFolder`
- `POST /dicom/loadSample`
- `POST /dicom/cornerInfo`
- `POST /view/create`
- `POST /view/setSize`

准确的请求与响应结构以 `/docs` 中的 OpenAPI 文档为准。

### Socket.IO 事件

客户端发送：

- `bind_view`
- `set_view_size`
- `view_operation`
- `image_operation`
- `view_hover`

服务端回推：

- `connected`
- `view_bound`
- `view_ack`
- `image_update`
- `hover_info`
- `image_error`
- `render_error`

`image_update` 会返回渲染图像数据和视口元数据。

## 前端联调说明

配套前端仓库：

[https://github.com/l5769389/DicomVisionClient](https://github.com/l5769389/DicomVisionClient)

推荐本地联调顺序：

1. 启动 `DicomVisionServer`。
2. 启动 `DicomVisionClient`。
3. 在客户端中加载 DICOM 文件夹。
4. 创建 Stack、MPR 或 3D 视口并开始交互。

当前前端桌面开发模式默认连接 `http://127.0.0.1:8000`，HTTP 与 Socket.IO 均使用该地址。

## Render 部署

仓库内已提供 `render.yaml`，可用于部署到 Render。

建议环境变量：

```env
APP_ENV=production
APP_HOST=0.0.0.0
APP_PORT=10000
CORS_ORIGINS=["https://your-vercel-app.vercel.app"]
WEB_SAMPLE_DICOM_PATH=/opt/render/project/src/sample-data
```

说明：

- `CORS_ORIGINS` 同时用于 FastAPI CORS 与 Socket.IO 允许来源
- `WEB_SAMPLE_DICOM_PATH` 必须指向可读目录
- Web 前端可通过 `POST /api/v1/dicom/loadSample` 加载服务端示例数据，而无需暴露本地路径

## 测试

运行测试：

```bash
uv run pytest
```

## 桌面打包

当前仓库负责构建后端桌面 bundle，由 Electron 客户端安装包消费。

如需安装 PyInstaller：

```bash
uv run python -m pip install pyinstaller
```

构建 Windows 后端 bundle：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-desktop-bundle.ps1
```

默认输出目录：

```text
dist/
  DicomVisionServer/
    DicomVisionServer.exe
    ...
```

如需覆盖输出路径：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-desktop-bundle.ps1 -OutputRoot .\artifacts
```

随后可通过 `DICOM_VISION_SERVER_BUNDLE_PATH` 将该目录交给前端安装包流程使用。
