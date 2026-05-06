# MPR Reslice Cursor 目标模型与重构实施方案

本文档定义新的 MPR 视图模型，用于替换当前基于 `mpr_frame`、`oblique_planes`、`oblique_line_angles`、`directed_line_angles` 等分散状态拼接出来的实现。

目标不是继续用测试用例修补现象，而是建立一个统一、可推导、可验证的三维 reslice cursor 模型。后续实现时，应以本文档为准，允许移除当前所有不匹配的 MPR 逻辑。

## 1. 当前问题

当前实现的问题不是某个公式单点错误，而是状态模型不对：

- 三个 MPR 视图分别维护 plane、line angle、directed angle、source viewport/source line。
- 渲染方向、十字线方向、物理距离方向、orientation label 不是同一个数据源。
- 为了避免镜像，会局部翻转 row/col/normal；翻转后的 normal 又被拿去计算物理距离，导致旋转和平移组合后符号错误。
- 0~360 度旋转被半圈归一化和 viewport 特殊规则打断，导致方向标签需要额外补丁。
- MPR 与 3D volume 旋转没有统一几何模型，只能后续继续加特殊逻辑。

新的设计要求：

- 所有 MPR 几何只由一个三维 cursor state 驱动。
- AX/COR/SAG 视图只是这个 cursor 的三个派生切面。
- 平移、旋转、reslice、crosshair、orientation、物理距离使用同一组 signed basis。
- 任何地方都不能再单独“猜”方向标签或单独维护线角状态。

## 2. 坐标系统

必须明确区分三类坐标。

### 2.1 IJK voxel 坐标

IJK 是 numpy volume 的采样坐标：

- `i`：volume axis 0
- `j`：volume axis 1
- `k`：volume axis 2

IJK 只能用于采样，不允许用于物理距离和方向标签。

### 2.2 World patient mm 坐标

MPR 的唯一几何真源使用 world patient mm 坐标。

推荐命名为 `world`，不要在实现里混用 `volume direction`、`patient direction`、`canonical direction`。

后端必须构造：

```python
@dataclass(frozen=True)
class VolumeGeometry:
    shape_ijk: tuple[int, int, int]
    ijk_to_world: np.ndarray  # 4x4
    world_to_ijk: np.ndarray  # 4x4
    spacing_hint_mm: tuple[float, float, float]
```

要求：

- `ijk_to_world` 从 DICOM `ImagePositionPatient`、`ImageOrientationPatient`、`PixelSpacing`、slice spacing 构造。
- volume 内部可以继续做标准化、转置、flip，但必须同步更新 affine。
- 物理距离、orientation label、MPR center 一律使用 world mm。
- 如果缺失 DICOM 几何，允许 fallback 到 identity-like affine，但仍然包装成 `VolumeGeometry`。

### 2.3 Viewport screen 坐标

Viewport 坐标只用于鼠标输入和 overlay 绘制：

- `x` 向右
- `y` 向下
- 单位可以是 canvas pixel 或 normalized `[0, 1]`

Viewport 坐标必须通过当前 `PlanePose` 转换为 world mm，不允许直接改 IJK index。

## 3. 核心状态模型

### 3.1 MprCursorState

整个 MPR group 只维护一个 cursor：

```python
@dataclass
class MprCursorState:
    center_world: np.ndarray              # shape=(3,), 当前十字线中心，单位 mm
    reference_center_world: np.ndarray    # shape=(3,), 初始 MPR 中心，单位 mm
    orientation_world: np.ndarray         # shape=(3, 3), 正交矩阵
    linked_to_volume_rotation: bool = False
```

`orientation_world` 是三个正交 cursor 轴。它不是屏幕旋转角，也不是某个 viewport 的 line angle。

推荐约定：

- `orientation_world[:, 0]`：AX 派生方向轴
- `orientation_world[:, 1]`：COR 派生方向轴
- `orientation_world[:, 2]`：SAG 派生方向轴

具体 AX/COR/SAG 的 row/col/normal 不直接存储，而是由 `derive_plane_pose(cursor, viewport)` 派生。

不变量：

```python
orientation_world.T @ orientation_world == I
det(orientation_world) == +1 或固定的项目约定值
```

每次旋转后必须做一次轻量正交化，例如 SVD 或 Gram-Schmidt，避免连续拖拽积累误差。

### 3.2 PlanePose

每个 viewport 渲染前临时派生一个 plane：

```python
@dataclass(frozen=True)
class PlanePose:
    viewport: Literal["mpr-ax", "mpr-cor", "mpr-sag"]
    center_world: np.ndarray
    row_world: np.ndarray     # 屏幕 y 向下的物理方向，单位向量
    col_world: np.ndarray     # 屏幕 x 向右的物理方向，单位向量
    normal_world: np.ndarray  # signed plane normal，单位向量
    pixel_spacing_row_mm: float
    pixel_spacing_col_mm: float
    output_shape: tuple[int, int]
```

要求：

- `row_world`、`col_world`、`normal_world` 必须来自同一个 cursor orientation。
- `normal_world` 是 signed 的，物理距离和 label 都必须使用它。
- 不允许为了 UI 不镜像而单独翻转 `row_world` 或 `col_world` 后继续复用旧 normal。
- 如需调整显示 convention，应只调整 `derive_plane_pose` 的初始 convention 表。

## 4. 默认显示约定

默认 convention 必须用少量 golden test 固定。基于当前参考软件行为，至少锁定：

- AX 初始视图：上方 `A`，下方 `P`。
- AX 初始物理坐标：参考软件显示 `I 1476.88mm` 时，本项目必须显示同样符号。
- AX 垂直线旋转 `-45 / 45 / 135 / 225` 度时，COR 视图物理方向依次为 `P / R / A / L`。
- 初始 MPR center 未移动时，任意 oblique 旋转的物理距离为 `0mm`。
- MPR center 向 A 移动时，相关 oblique 视图显示 `A` 和递增距离，不允许显示 `P`。

实现时不要在多个地方散落这些规则。应集中在：

```python
DEFAULT_MPR_CONVENTION = {
    "mpr-ax": PlaneConvention(...),
    "mpr-cor": PlaneConvention(...),
    "mpr-sag": PlaneConvention(...),
}
```

如果参考软件 convention 后续发生调整，只允许改这个表和对应 golden tests。

## 5. 派生 PlanePose

建议提供唯一入口：

```python
def derive_plane_pose(
    cursor: MprCursorState,
    viewport: str,
    geometry: VolumeGeometry,
    output_shape_policy: OutputShapePolicy,
) -> PlanePose:
    ...
```

职责：

- 从 cursor orientation 取出当前 viewport 的 row/col/normal。
- 根据 `VolumeGeometry` 和 viewport 计算输出尺寸。
- 根据 row/col 在 world 中的采样步长计算 pixel spacing。
- 标记是否 oblique。

判断 oblique 不应来自额外布尔状态，而应由当前 plane 与默认 plane 的夹角决定：

```python
is_oblique = angle_between(plane.normal_world, default_plane.normal_world) > eps
```

## 6. Reslice 渲染

MPR 渲染统一走同一个函数：

```python
def reslice_plane(
    volume: np.ndarray,
    geometry: VolumeGeometry,
    plane: PlanePose,
    mip: MipConfig | None,
) -> np.ndarray:
    ...
```

采样公式：

```python
world = (
    plane.center_world
    + plane.col_world * x_mm
    + plane.row_world * y_mm
    + plane.normal_world * slab_offset_mm
)
ijk = geometry.world_to_ijk @ [world.x, world.y, world.z, 1]
```

然后使用 `scipy.ndimage.map_coordinates` 采样。

优点：

- 正交 MPR 和 oblique MPR 是同一条路径。
- 3D 旋转后的 MPR 也是同一条路径。
- MIP 只是沿 `normal_world` 方向增加 slab offsets。
- 不需要 `_extract_mpr_plane` 和 `_extract_oblique_mpr_plane` 两套逻辑。

## 7. 平移

十字线中心移动只修改：

```python
cursor.center_world
```

Viewport 拖拽转换：

```python
delta_world =
    active_plane.col_world * delta_x_mm
    + active_plane.row_world * delta_y_mm

cursor.center_world = clamp_to_volume_bounds(cursor.center_world + delta_world)
```

注意：

- `delta_x_mm`、`delta_y_mm` 来自当前 viewport 的 image transform 和 plane spacing。
- 不允许再直接写 `mpr_axial_index`、`mpr_coronal_index`、`mpr_sagittal_index`。
- 旧索引若前端仍需要，只能作为派生字段返回。

## 8. 旋转

旋转必须基于 drag-start 快照，不允许每帧用当前角度反推 plane。

拖拽开始时保存：

```python
@dataclass
class MprRotationDragState:
    viewport: str
    line: Literal["horizontal", "vertical"]
    start_angle_rad: float
    start_cursor: MprCursorState
```

拖拽中：

```python
delta = current_angle_rad - start_angle_rad
axis_world = active_plane.normal_world
R = rotation_matrix(axis_world, delta)

cursor.orientation_world = orthonormalize(R @ start_cursor.orientation_world)
cursor.center_world = start_cursor.center_world
```

如果旋转的是某条十字线，而不是整个 active plane，应把 axis 定义为该线对应的 world direction：

```python
line_world =
    active_plane.col_world      # horizontal line
    或 active_plane.row_world   # vertical line

R = rotation_matrix(line_world, delta)
```

实际产品语义需要二选一，并写入测试：

- “旋转 active plane 内的十字线”：围绕 active plane normal 改变另外两个切面的 normal。
- “绕十字线本身翻转切面”：围绕 line_world 旋转目标切面。

当前用户描述更接近第一种：在 AX 视图里旋转十字线，改变 COR/SAG 的切面方向，同时 AX 自身保持当前切片。

无论选择哪种，都必须通过同一个 `MprCursorState.orientation_world` 更新，不再维护 `oblique_line_angles`。

## 9. 十字线 overlay

十字线不是状态，而是 plane intersection 的派生结果。

在 active viewport 中，另一个 plane 的线方向：

```python
line_world = normalize(cross(active_plane.normal_world, target_plane.normal_world))
```

投影到 active plane 屏幕坐标：

```python
x = dot(line_world, active_plane.col_world)
y = dot(line_world, active_plane.row_world)
angle = atan2(y, x)
```

中心点：

```python
center_screen = world_to_plane_screen(cursor.center_world, active_plane)
```

因此不再需要：

- `oblique_line_angles`
- `oblique_directed_line_angles`
- `_sync_mpr_oblique_line_angles`
- `_resolve_mpr_crosshair_line_angle`

## 10. 物理距离与方向标签

物理距离统一：

```python
signed_distance_mm = dot(
    cursor.center_world - cursor.reference_center_world,
    plane.normal_world,
)
```

显示方向：

```python
direction = plane.normal_world if signed_distance_mm >= 0 else -plane.normal_world
label = dominant_patient_axis(direction)
```

显示文本：

```python
f"{label} {abs(signed_distance_mm):.2f}mm"
```

零值规则：

```python
if abs(signed_distance_mm) < 0.005:
    signed_distance_mm = 0
```

禁止事项：

- 禁止用 `dot(patient_point, normal)` 对 patient 原点做绝对投影。
- 禁止物理距离 label 使用另一个 vector，而距离使用另一个 normal。
- 禁止为某个象限添加方向修正分支。

如果 reference software 的标签 convention 与 DICOM LPS 不完全一致，必须通过 `dominant_patient_axis` 的 label policy 统一处理，而不是散落在 MPR 逻辑中。

## 11. Orientation overlay

四边方向标签只从当前 `PlanePose` 生成：

```python
top    = dominant_patient_axis(-plane.row_world)
bottom = dominant_patient_axis( plane.row_world)
right  = dominant_patient_axis( plane.col_world)
left   = dominant_patient_axis(-plane.col_world)
```

这与物理距离使用同一套 label policy。

斜切视图仍显示标准单轴方向，不显示 `PR`、`AL` 这种组合，除非产品明确改需求。

## 12. 3D volume 旋转后的 MPR

3D volume 旋转应拆成两个概念：

### 12.1 VR camera rotation

只影响 3D viewport 的显示相机，不改变 MPR：

```python
volume_view.rotation_quaternion
```

### 12.2 Linked MPR from 3D rotation

如果用户希望“把 3D volume 旋转到某个角度后生成 MPR”，则将 3D 旋转矩阵应用到 MPR cursor orientation：

```python
cursor.orientation_world = R_volume_world @ default_cursor.orientation_world
cursor.center_world = current_volume_focus_world
```

后续 AX/COR/SAG 仍然通过 `derive_plane_pose` 和 `reslice_plane` 渲染，不需要新路径。

推荐新增操作：

```json
{
  "opType": "mprSetFromVolumeRotation",
  "viewId": "...",
  "rotationQuaternion": [x, y, z, w],
  "focusWorld": [x, y, z]
}
```

也可以提供 UI 开关：

- `3D rotation only`：只旋转 VR camera。
- `Sync MPR to 3D`：把 3D 当前姿态同步到 MPR cursor。

## 13. 后端模块拆分

新增模块：

```text
app/services/mpr/
  geometry.py        # VolumeGeometry, affine, vector/matrix helpers
  cursor.py          # MprCursorState, translate, rotate, reset
  planes.py          # derive_plane_pose, PlanePose, convention table
  reslice.py         # reslice_plane, MIP slab
  overlay.py         # crosshair, orientation, corner info
  serializers.py     # schema payload conversion
```

`viewer_service.py` 只负责 orchestration：

- 获取 series volume 和 geometry。
- 获取/更新 `ViewGroupRecord.mpr_cursor`。
- 调用 mpr module。
- 返回 `ViewImageResponse`。

不再在 `viewer_service.py` 里写 MPR 几何公式。

## 14. 数据模型迁移

目标 `ViewGroupRecord`：

```python
@dataclass
class ViewGroupRecord:
    group_id: str
    group_type: str
    series_id: str
    active_viewport: str = "mpr-ax"
    mpr_cursor: MprCursorState | None = None
    mpr_mip: MprMipState = field(default_factory=MprMipState)
    crosshair_drag: MprCrosshairDragState | None = None
    rotation_drag: MprRotationDragState | None = None
```

应删除或废弃：

- `MprFrameState.axis_slice`
- `MprFrameState.axis_row`
- `MprFrameState.axis_col`
- `MprObliquePlaneState`
- `create_default_mpr_oblique_planes`
- `oblique_planes`
- `oblique_line_angles`
- `oblique_directed_line_angles`
- `oblique_source_viewport`
- `oblique_source_line`
- `mpr_reference_center`，改为 `mpr_cursor.reference_center_world`
- `mpr_axial_index` / `mpr_coronal_index` / `mpr_sagittal_index` 作为可写状态

如前端短期还依赖索引，可在 response 中提供派生字段：

```python
derived_indices = geometry.world_to_ijk @ cursor.center_world
```

但这些字段不能反向驱动 MPR。

## 15. API 与前端职责

### 15.1 后端 response

建议返回：

```json
{
  "mprCursor": {
    "centerWorld": [0, 0, 0],
    "referenceCenterWorld": [0, 0, 0],
    "orientationWorld": [[...], [...], [...]]
  },
  "mprPlane": {
    "viewport": "mpr-cor",
    "centerWorld": [0, 0, 0],
    "rowWorld": [0, 0, 0],
    "colWorld": [0, 0, 0],
    "normalWorld": [0, 0, 0],
    "isOblique": true
  },
  "mprCrosshair": {
    "centerX": 0.5,
    "centerY": 0.5,
    "horizontalAngleRad": 0.0,
    "verticalAngleRad": 1.5707963268
  }
}
```

### 15.2 前端职责

前端只负责：

- 显示 image。
- 按 response 画 crosshair。
- 把 pointer down/move/up 和当前 viewport 发给后端。
- 传 canvas 坐标或 normalized 坐标。

前端不负责：

- 推导 MPR plane。
- 维护 crosshair angle。
- 自己判断 oblique 方向标签。
- 自己计算物理距离。

## 16. 实施步骤

### Phase 1：建立纯几何内核

新增 `app/services/mpr/geometry.py`、`cursor.py`、`planes.py`。

完成：

- `VolumeGeometry`
- `MprCursorState`
- `PlanePose`
- 默认 convention table
- `derive_plane_pose`
- `translate_cursor`
- `rotate_cursor`
- `dominant_patient_axis`

只写单元测试，不接入渲染。

### Phase 2：统一 reslice

新增 `reslice_plane`。

用新 reslice 替换：

- `_extract_mpr_plane`
- `_extract_oblique_mpr_plane`

要求正交 MPR 和 oblique MPR 都走同一函数。

### Phase 3：统一 overlay

新增：

- `build_crosshair_overlay_from_planes`
- `build_orientation_overlay_from_plane`
- `build_corner_info_from_plane`

删除旧的：

- `_get_mpr_crosshair_line_angles`
- `_sync_mpr_oblique_line_angles`
- `_resolve_mpr_crosshair_line_angle`
- `_resolve_mpr_oblique_orientation_vector`
- 所有 viewport/line 特殊方向分支

### Phase 4：平移接入

`crosshair` move 改为：

```python
cursor = translate_cursor(cursor, active_plane, delta_screen)
```

不再写三个 MPR index。

### Phase 5：旋转接入

`mprOblique` 改为：

```python
cursor = rotate_cursor_from_drag(rotation_drag_state, current_angle)
```

不再更新 `oblique_planes` 和 line angle cache。

### Phase 6：前端协议收敛

前端 overlay 使用后端返回的 `mprCrosshair`、`mprPlane`。

旧字段保留一版兼容期：

- `mprFrame`
- `mprPlane`
- `mpr_crosshair`

新字段稳定后移除旧字段。

### Phase 7：3D rotation to MPR

新增 `mprSetFromVolumeRotation` 操作。

把 3D rotation matrix 转成 cursor orientation，复用同一套 reslice。

## 17. 测试策略

### 17.1 纯数学测试

必须覆盖：

- `orientation_world` 旋转后仍正交。
- 连续旋转 360 度后回到初始 orientation。
- 任意 plane 的 row/col/normal 一致且无镜像。
- world -> ijk -> world 误差小于 epsilon。

### 17.2 Golden behavior 测试

必须覆盖：

- 初始 AX 上下为 `A/P`。
- 初始 AX 物理坐标符号与参考软件一致。
- AX 垂直线 `-45 / 45 / 135 / 225` 度，COR 物理方向为 `P / R / A / L`。
- 初始中心旋转任意角，物理距离为 `0mm`。
- 先平移到 A，再旋转或旋转后平移到 A，显示 `A` 和递增距离。
- 顺时针/逆时针跨过 0 度、90 度、180 度、270 度无跳变。

### 17.3 集成测试

必须覆盖：

- 正交 MPR rendering shape 与旧实现一致。
- oblique rendering 在简单 synthetic volume 上位置正确。
- MIP slab 沿 plane normal 取样。
- crosshair center drag 后三个视图一致更新。
- reset 恢复 cursor、orientation、reference center。

## 18. 验收标准

重构完成后，应满足：

- MPR 几何公式不再散落在 `viewer_service.py`。
- 不存在 `oblique_line_angles` / `directed_line_angles` 这类状态。
- 每个渲染帧只从 `MprCursorState` 派生三个 `PlanePose`。
- 物理距离、orientation label、crosshair angle、reslice sample 使用同一个 plane basis。
- 新增 3D volume rotation MPR 时不需要新 reslice 路径，只需要设置 cursor orientation。

## 19. 明确不做的事

- 不继续在当前实现里追加 viewport/line/象限特殊补丁。
- 不用图像显示方向反推物理 normal。
- 不把 IJK index 当作物理位置。
- 不在前端维护 MPR 几何真源。

## 20. 后续执行建议

建议下一步直接从 Phase 1 开始，先实现纯几何内核和 golden tests。不要先改现有渲染路径。

原因：

- 几何内核是无 UI、无 DICOM IO、无 socket 的纯函数，最容易验证。
- 一旦 `MprCursorState -> PlanePose` 稳定，后续替换 render/crosshair/overlay 会非常直接。
- 如果先在 `viewer_service.py` 里继续改，会把旧模型和新模型混在一起，风险更高。

## 21. 当前功能覆盖核对

本节用于确认新模型是否覆盖当前产品功能。后续执行时如果遇到功能遗漏，先回到这里补模型，不要在旧实现上打补丁。

| 当前功能 | 新模型覆盖方式 | 说明 |
| --- | --- | --- |
| 初始 AX/COR/SAG 正交 MPR | `MprCursorState + derive_plane_pose` | 默认 convention 生成三个正交 `PlanePose`。 |
| 十字线中心平移 | `translate_cursor(cursor, active_plane, delta_screen)` | 只更新 `cursor.center_world`，不再写三个 index。 |
| 十字线旋转 / oblique | `rotate_cursor_from_drag(rotation_drag_state, current_angle)` | 只更新 `cursor.orientation_world`，不再维护 line angle cache。 |
| 斜切重采样 | `reslice_plane(volume, geometry, plane, mip)` | 正交和 oblique 用同一条 reslice 路径。 |
| MIP slab | `reslice_plane` 沿 `plane.normal_world` 取 slab offsets | 不需要单独的 oblique MIP 分支。 |
| 滚轮翻层 / 沿法线移动 | `cursor.center_world += plane.normal_world * scroll_mm` | scroll 不再操作 axial/coronal/sagittal index。 |
| 物理距离角标 | `dot(cursor.center_world - reference_center_world, plane.normal_world)` | 标签和距离来自同一个 signed normal。 |
| Orientation overlay | `row_world/col_world` 推导 top/right/bottom/left | 不再根据 viewport 特殊判断。 |
| Scale bar | 使用 `plane.pixel_spacing_col_mm` 和 `plane.pixel_spacing_row_mm` | oblique 下也与实际采样间距一致。 |
| Window/level | 保持现有像素强度处理 | 与几何模型无关，不迁移到 MPR geometry。 |
| Pseudocolor | 保持现有渲染后着色 | 与几何模型无关。 |
| Measurement | 测量点从 viewport 坐标转换为 `world` 坐标保存 | 显示时再投影回当前 `PlanePose`。 |
| Reset | 重建默认 `MprCursorState` 和 `reference_center_world` | 不再重置 oblique cache。 |
| 3D VR camera rotation | 保持 3D view 自己的 camera state | 不默认影响 MPR。 |
| 3D rotation 同步 MPR | 把 3D rotation matrix 写入 `cursor.orientation_world` | 复用同一条 `derive_plane_pose + reslice_plane`。 |

## 22. Scroll 规则

MPR scroll 应视为沿当前 viewport plane normal 移动 cursor：

```python
def scroll_cursor(
    cursor: MprCursorState,
    plane: PlanePose,
    scroll_steps: int,
    step_mm: float,
    geometry: VolumeGeometry,
) -> MprCursorState:
    center = cursor.center_world + plane.normal_world * float(scroll_steps) * step_mm
    center = clamp_world_to_volume(center, geometry)
    return replace(cursor, center_world=center)
```

`step_mm` 推荐使用当前 plane normal 对应的体素步长：

```python
step_mm = spacing_along_world_direction(geometry, plane.normal_world)
```

禁止：

- 禁止 scroll 时写 `mpr_axial_index`、`mpr_coronal_index`、`mpr_sagittal_index`。
- 禁止 oblique scroll 继续按正交轴移动。

## 23. Measurement 规则

当前测量功能必须迁移到 world 坐标，否则 oblique 和 3D rotation 同步 MPR 后无法稳定显示。

推荐测量点模型：

```python
@dataclass
class MeasurementPointWorld:
    world: np.ndarray
    source_viewport: str
    source_plane_normal_world: np.ndarray
```

新增测量时：

```python
world = plane_screen_to_world(plane, x_px, y_px)
```

显示测量时：

```python
screen = world_to_plane_screen(plane, measurement_point.world)
distance_to_plane = dot(measurement_point.world - plane.center_world, plane.normal_world)
visible = abs(distance_to_plane) <= measurement_visibility_tolerance_mm
```

距离测量值：

```python
length_mm = norm(point_b.world - point_a.world)
```

角度测量值：

```python
angle = angle_between(point_a.world - vertex.world, point_b.world - vertex.world)
```

这样 measurement 不依赖切片 index，也不依赖某个 viewport 的旧 2D 坐标。

## 24. 执行入口清单

后续如果把本文档再次交给执行者，应按以下顺序操作：

1. 不要继续修改旧 MPR oblique 补丁逻辑。
2. 新建 `app/services/mpr/` 子包。
3. 先实现 `VolumeGeometry`、`MprCursorState`、`PlanePose`、`derive_plane_pose`。
4. 写纯数学 golden tests，覆盖本文档第 17 节和第 21 节。
5. 新建 `reslice_plane`，用 synthetic volume 验证正交和 oblique 采样。
6. 在 `viewer_service.py` 中做最小接入：只把当前渲染路径切到新 `reslice_plane`。
7. 替换 crosshair 平移为 `translate_cursor`。
8. 替换 oblique 旋转为 `rotate_cursor_from_drag`。
9. 替换 overlay、cornerInfo、scaleBar 为从 `PlanePose` 派生。
10. 迁移 measurement 到 world 坐标。
11. 删除旧状态和旧函数。
12. 最后实现 `mprSetFromVolumeRotation`。

每一步都必须保持测试通过；如果某一步需要旧字段兼容，只能把旧字段作为派生 payload，不允许继续作为几何真源。

## 25. 方案结论

在当前需求范围内，`MprCursorState -> PlanePose -> reslice/overlay` 是推荐的最佳方案。

它不是唯一可能方案，但它符合成熟医学影像软件中 reslice cursor 的主流设计，能同时覆盖：

- 正交 MPR。
- Oblique MPR。
- 十字线平移。
- 十字线旋转。
- MIP。
- 物理距离和方向标签。
- measurement。
- scale bar。
- reset。
- 3D volume rotation 同步 MPR。

后续如果继续扩展 curved MPR、multi-planar slab、CPR 或多个独立 reslice cursor，可以在这个模型上扩展；不需要推翻当前设计。
