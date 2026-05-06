# MPR 十字线旋转重实现步骤

本文档记录本次 MPR 十字线旋转的重实现方案。目标是移除旧的“前端计算旋转角度增量、后端按增量补符号”的协议，改成前端只上报交互事实，后端统一推导 MPR plane。

## 目标

- 前端不再计算 `deltaAngleRad`，不再维护旋转角度状态。
- 前端只发送：
  - 当前旋转所在的 MPR 视口，由 `viewId`/`viewportKey` 映射得到；
  - 被拖拽的十字线方向：`horizontal` 或 `vertical`；
  - 拖拽阶段：`start` / `move` / `end`；
  - 当前指针在渲染图像中的归一化位置：`x` / `y`。
- 后端在 `start` 时保存 MPR cursor 快照。
- 后端在 `move` / `end` 时根据当前指针方向，计算拖拽线和垂直线的方向，成对更新另外两个 target plane normal。
- 三个 MPR plane 始终由 `MprCursorState -> derive_plane_pose(...)` 派生，不在前端或协议中传 plane 几何。
- MPR 显示比例、缩放、比例尺统一使用 `PlanePose.pixel_spacing_col_mm` / `pixel_spacing_row_mm`，避免把 world 方向误当成体素方向后造成影像形变。

## 旧逻辑移除范围

### 前端

1. 删除 `mprCrosshairPointerController.ts` 中的旋转拖拽状态和角度增量计算。
2. 删除 `toMprObliqueDeltaAngleRad(...)` 以及相关测试。
3. `useViewerWorkspacePointer.ts` 只保留十字线命中检测、拖拽状态和 `x/y` 上报。
4. Socket payload 不再包含 `angleRad` / `deltaAngleRad`。

### 后端

1. `ViewOperationRequest` 不再接收 `angleRad` / `deltaAngleRad`。
2. 删除 `_apply_mpr_rotation_drag(..., delta_angle_rad, ...)` 的旧增量入口。
3. 删除 `ViewGroupRecord` 中旧的 oblique source 状态。
4. `MprRotationDragRecord` 改为保存：
   - `viewport`
   - `line`
   - `start_cursor`
5. 保留 `mprOblique` 操作名作为现有 socket/API 兼容入口，但语义改为“后端根据 pointer 位置解析旋转”。

## 新后端算法

1. `start`
   - 解析当前 view 对应的 active viewport。
   - 构建当前 `MprPoseContext`。
   - 保存 `MprRotationDragRecord`，其中 `start_cursor` 是拖拽开始时的 cursor 快照。

2. `move` / `end`
   - 读取 `start_cursor` 并重新派生拖拽开始时的三个 planes。
   - 找到当前拖拽线对应的 target viewport：
     - AX horizontal -> COR
     - AX vertical -> SAG
     - COR horizontal -> AX
     - COR vertical -> SAG
     - SAG horizontal -> AX
     - SAG vertical -> COR
   - 根据当前 `x/y` 计算当前指针 screen angle，再转换为 active plane 内的目标 world 方向。
   - 用 `target_line_world × active_normal_world` 得到被拖拽线对应 target plane 的新 normal，并保持 normal 符号与拖拽开始时一致。
   - 用垂直于 target line 的 active plane 内方向，计算另一条十字线对应 target plane 的新 normal。
   - 替换 `start_cursor.orientation_world` 中另外两个 target viewport 对应的 normal 列，active plane normal 保持不变。
   - 同步新的 cursor 回 group。

3. 渲染
   - 不新增前端几何推导。
   - 后端继续通过 `derive_plane_pose(...)` 生成 `mprPlane`、`mprFrame`、`mpr_crosshair`。
   - `mprFrame` 输出保留 cursor 的独立方向列，不再在输出阶段把三列强行正交化。
   - 渲染 transform、fit zoom、hover mapping、measurement spacing、scale bar 都从 `PlanePose` 获取物理像素间距。
   - 前端按响应绘制十字线。

## 实施顺序

1. 文档落地：本文件作为本次重实现的执行清单。
2. 前端删旧协议：
   - 删除旋转角度状态、角度展开、`deltaAngleRad` 转换函数和测试。
   - 旋转拖拽仅发送 `mode: "rotate"`、`line`、`phase`、`x`、`y`。
3. 后端删旧协议字段：
   - schema 和生成类型移除 `angleRad` / `deltaAngleRad`。
   - operation handler 保持 `mprOblique` 入口。
4. 后端实现 pointer-driven 旋转：
   - 新增指针角度解析 helper。
   - 新增由指针方向成对更新 target plane normal 的 helper。
   - `_handle_mpr_oblique` 改为使用新 helper。
   - 修正 MPR 显示比例，禁止用 `MprObliquePlaneState.row/col` 去调用只接受体素方向的 spacing 计算。
5. 测试替换：
   - 删除旧的 `deltaAngleRad` 相关测试。
   - 增加“前端不发送角度增量”的测试。
   - 增加后端通过 `x/y + line` 旋转后另外两个 target plane 成对变化、active plane 保持不变的测试。
   - 增加第二次旋转仍保持 active plane 固定，并广播同步三个 viewport 的回归测试。
   - 增加显示比例使用 `PlanePose` 物理间距的测试。
   - 增加后端不依赖 `deltaAngleRad` 字段的协议测试。
6. 验证：
   - 运行后端 MPR 相关 pytest。
   - 运行前端相关 vitest / typecheck。
