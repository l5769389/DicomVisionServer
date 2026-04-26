# 4D 呼吸相位播放需求与后端实现文档

## 背景

前端已经提供 `4D` 视图入口与播放界面，界面从 `POST /api/v1/dicom/loadFolder` 返回的 `seriesList` 中读取 4D 元数据：

- `isFourDSeries`
- `fourDPhaseCount`
- `fourDPhases`

因此后端需要在加载 DICOM 文件夹时完成 4D 呼吸相位识别、相位列表组装与预览图生成。当前阶段不新增前端交互协议，也不把 4D 播放改造成实时 socket 渲染；播放由前端基于相位预览图完成。

为避免前端仅依赖 `loadFolder` 的初始返回，后端同时提供 `POST /api/v1/dicom/fourD/phases`，前端打开 4D 视图时会按 `seriesId` 主动拉取最新 phase manifest。

## 目标

1. 加载文件夹时识别可组成 4D 呼吸相位播放的 DICOM 序列。
2. 在 `SeriesSummary` 中返回每个相位的编号、标签、关联 `seriesId` 与 AX/COR/SAG 预览图。
3. 支持常见 4DCT 数据形态：
   - 多个相位分别存放在多个 DICOM Series 中，序列描述包含 `0%`、`10%`、`20%` 等呼吸相位标识。
   - 单个 Series 内包含 `TemporalPositionIdentifier`、`TriggerTime` 或呼吸相位相关 DICOM tag。
4. 对无法确认相位的普通序列保持现有行为，不返回 4D 字段或返回空字段。
5. 后端实现保持轻量，避免 `loadFolder` 阶段执行完整高质量渲染。

## 非目标

- 不实现后端驱动的 4D 播放定时器。
- 不新增 4D socket 事件。
- 不在本阶段为单个混合 Series 拆分虚拟 Series 给 Stack/MPR/3D 视图使用。
- 不要求所有厂商私有 tag 都能自动识别，但保留规则扩展点。

## 前端契约

`fourDPhases` 的单项结构：

```json
{
  "phaseIndex": 0,
  "label": "Phase 01",
  "seriesId": "phase-series-id",
  "imageSrc": "data:image/png;base64,...",
  "viewportImages": {
    "mpr-ax": "data:image/png;base64,...",
    "mpr-cor": "data:image/png;base64,...",
    "mpr-sag": "data:image/png;base64,..."
  },
  "status": "ready"
}
```

字段说明：

- `phaseIndex`：0 基相位序号，前端用它排序与切换。
- `label`：前端按钮与预览辅助文本，例如 `Phase 01`、`Phase 10%`。
- `seriesId`：该相位对应的后端 Series。多 Series 4D 数据中用于打开相位的 Stack/MPR/3D。
- `imageSrc`：兼容字段，默认使用 AX 预览。
- `viewportImages`：三向预览图，键为 `mpr-ax`、`mpr-cor`、`mpr-sag`。
- `status`：`ready` 表示预览可用，`pending` 表示仅识别到相位但未生成图，`error` 表示生成失败。

## 识别规则

### 多 Series 4D

同一文件夹内多个 Series 满足以下条件时组成一个 4D phase group：

1. `StudyInstanceUID` 相同。
2. `Modality` 相同。
3. Series 描述或文件夹名中存在相位标识，例如：
   - `0%`、`10%`、`90%`
   - `phase 0`、`phase 1`
   - `P0`、`P10`
4. 同组序列的层数和图像尺寸尽量一致。

每个相位 Series 的 `SeriesSummary` 都返回同一份 `fourDPhases`，这样用户从任意相位 Series 进入 4D 视图都能看到完整相位播放。

### 单 Series 4D

单个 Series 内如果存在可用相位 tag，则按 tag 分组：

- `TemporalPositionIdentifier`
- `TriggerTime`
- `NominalPercentageOfRespiratoryPhase`
- `RespiratoryTriggerDelayThreshold`
- `RespiratoryTriggerType`

当前阶段会返回相位预览，但 `seriesId` 仍指向原始 Series。后续如果需要从 4D 中打开某个相位的 Stack/MPR/3D，应增加虚拟 phase series 注册机制。

## 预览图生成

预览生成在 `loadFolder` 后置阶段执行，策略如下：

1. 对每个相位读取少量像素数据。
2. AX 预览取相位体数据中间层。
3. COR 预览取体数据中间行。
4. SAG 预览取体数据中间列。
5. 使用简单 min/max window 映射为 8-bit 灰度 PNG。
6. 输出尺寸限制在缩略图范围内，降低响应体体积。
7. 生成失败时保留相位项并标记 `status: "error"`。

## 后端落点

- `app/schemas/dicom.py`
  - 扩展 `SeriesSummary`。
  - 增加 `FourDPhaseItem` schema。
- `app/services/four_d_service.py`
  - 相位识别。
  - 相位组装。
  - 缩略图生成。
- `app/services/series_registry.py`
  - `load_folder` 注册完 Series 后调用 4D 服务增强 summary。
- `app/api/routes/dicom.py`
  - 暴露 `POST /dicom/fourD/phases`，用于前端打开 4D tab 时补取 phase manifest。

## 验收标准

1. 普通单序列 DICOM 加载结果保持兼容。
2. 多相位测试数据加载后，每个相位 Series 都带有：
   - `isFourDSeries: true`
   - 正确的 `fourDPhaseCount`
   - `fourDPhases.length == 相位数`
   - 每个相位有 `seriesId` 和三向 `viewportImages`
3. 单 Series 带 `TemporalPositionIdentifier` 的测试数据能返回多个相位项。
4. 预览图为 `data:image/png;base64,` URL。
5. 现有测试通过，并新增 4D 后端测试覆盖主要识别路径。

## 后续扩展

- 引入虚拟 phase series，让单 Series 4D 可以从任意相位打开独立 Stack/MPR/3D。
- 增加厂商私有 tag 识别配置。
- 将预览生成改为懒加载 endpoint，减少大型 4D 数据首次加载耗时。
- 支持服务端 4D cine 渲染与 socket 推送。
