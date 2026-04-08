# DicomVision Server

[English](./README.md)

DicomVision Server 是 DicomVision 桌面阅片系统的后端服务，负责 DICOM 序列加载、视口生命周期管理、2D 图像渲染、MPR 重建、3D 体渲染，以及通过 Socket.IO 提供实时图像更新。

该服务基于 FastAPI 和 Python 构建，配套前端仓库位于 `D:\ct\git-repo\my\dicomVision\DicomVisionClient`。

## 目录

- [项目概览](#项目概览)
- [核心亮点](#核心亮点)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [架构说明](#架构说明)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API 与实时事件](#api-与实时事件)
- [3D 体渲染](#3d-体渲染)
- [开发说明](#开发说明)
- [测试](#测试)
- [前后端联调](#前后端联调)

## 项目概览

后端对外提供两类通信通道：

- HTTP：用于加载 DICOM 目录、创建视口等粗粒度资源操作
- Socket.IO：用于平移、缩放、滚动、悬停、十字线更新和实时渲染推送等交互操作

典型调用流程：

1. 前端调用 `POST /api/v1/dicom/loadFolder` 注册目录并获取 `seriesId`。
2. 前端调用 `POST /api/v1/view/create` 创建视口并获取 `viewId`。
3. 前端通过 `POST /api/v1/view/setSize` 或 `set_view_size` socket 事件设置视口尺寸。
4. 前端通过 `bind_view` 将 socket 连接绑定到视口。
5. 前端通过 `view_operation` 或 `image_operation` 发送交互命令。
6. 后端回推 `image_update`、`hover_info`、`view_ack` 或错误事件。

## 核心亮点

- 基于 FastAPI 的 HTTP API，用于序列加载与视口创建
- 基于 Socket.IO 的低延迟交互与图像刷新链路
- 支持 2D Stack 渲染、MPR 重建和基于 VTK 的 3D 体渲染
- 使用内存注册表管理序列与视图状态
- 通过 DICOM 像素缓存减少重复文件读取
- 支持角标信息、方向信息、十字线联动与 hover 坐标映射
- 支持 3D 预设与传输函数配置

## 技术栈

- Python 3.13+
- FastAPI
- python-socketio
- pydicom
- NumPy
- Pillow
- SciPy
- VTK
- uv

## 目录结构

```text
app/
  api/routes/              HTTP 路由
  core/                    配置、常量、日志
  models/                  内存领域模型
  schemas/                 请求响应模型
  services/                渲染、注册表与 DICOM 处理
  sockets/                 Socket.IO 事件与运行时中心
  utils/                   通用工具
run.py                     本地启动入口
pyproject.toml             项目与依赖配置
.env.example               环境变量示例
```

## 架构说明

后端核心职责：

- 从本地目录注册并索引可读 DICOM 序列
- 管理 `Stack`、`MPR`、`3D` 三类视口状态
- 渲染图像帧和叠加信息供客户端消费
- 维护多视图交互同步
- 对 3D 预设和自定义配置进行归一化并应用到渲染流程

核心模块：

- `series_registry`：目录级序列发现和元数据管理
- `view_registry`：视口实例与生命周期管理
- `dicom_cache`：解码像素缓存
- `viewer_service`：渲染与交互处理的核心编排层
- `view_socket_hub`：定向 socket 绑定与渲染推送

## 快速开始

### 环境要求

- Python 3.13 或更高版本
- 可运行 VTK 的本地环境
- 后端所在机器需要具备对本地 DICOM 目录的访问权限

### 安装依赖

```bash
uv sync
```

如需安装开发依赖：

```bash
uv sync --extra dev
```

### 启动服务

```bash
uv run python run.py
```

默认启动地址为 `http://127.0.0.1:8000`。

启动后可访问：

- OpenAPI 文档：`http://127.0.0.1:8000/docs`
- ReDoc 文档：`http://127.0.0.1:8000/redoc`
- Socket.IO 路径：`http://127.0.0.1:8000/socket.io`

## 配置说明

如有需要，可根据 `.env.example` 创建本地 `.env` 文件。

```env
APP_NAME=DicomVision Server
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
CORS_ORIGINS=["*"]
```

关键配置项：

- `APP_ENV`：设置为 `development` 时启用自动重载
- `APP_HOST`：Uvicorn 监听地址
- `APP_PORT`：服务端口，默认 `8000`
- `CORS_ORIGINS`：允许的 HTTP 与 Socket.IO 来源

## API 与实时事件

HTTP 基础路径：`/api/v1`

### HTTP API

- `GET /health`
- `POST /dicom/loadFolder`
  - 作用：加载本地目录并注册可读 DICOM 序列
- `POST /dicom/cornerInfo`
  - 作用：获取序列级角标信息
- `POST /view/create`
  - 作用：创建 `Stack`、`MPR` 或 `3D` 视口
- `POST /view/setSize`
  - 作用：保存视口尺寸并触发渲染

精确的请求响应结构请以 `/docs` 中自动生成的 OpenAPI 文档为准。

### Socket.IO 事件

客户端发往服务端：

- `bind_view`
- `set_view_size`
- `view_operation`
- `image_operation`
- `view_hover`

服务端发往客户端：

- `connected`
- `view_bound`
- `view_ack`
- `image_update`
- `hover_info`
- `image_error`
- `render_error`

`image_update` 会返回渲染后的视口元数据及图像字节流。

## 3D 体渲染

基于 VTK 的 3D 渲染器位于 `app/services/volume_rendering/`。

当前 3D 能力包括：

- 基于四元数的相机旋转
- 快速预览与完整渲染两条路径
- 可配置的混合模式
- 可配置的光照与插值策略
- 多层传输函数配置
- 内置 `aaa`、`red`、`cardiac`、`muscle`、`mip` 等预设

所有 3D 配置在应用到渲染会话前，都会先在后端完成归一化处理。

## 开发说明

- 当前实现以“后端负责渲染，前端负责驱动和展示”为主要设计思路。
- 低延迟交互主要依赖 Socket 事件链路。
- 当前前端默认连接 `http://127.0.0.1:8000`。

## 测试

如果后续补充或扩展测试，可执行：

```bash
uv run pytest
```

## 前后端联调

配套前端项目位于：

`D:\ct\git-repo\my\dicomVision\DicomVisionClient`

启动 Electron 客户端前，请先启动本服务。
