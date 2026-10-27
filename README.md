# GPT — 机器人多子任务执行仿真

为 GPT 高层任务规划准备可调用的机器人执行能力。当前提交是已完成的第一阶段物理场景与实时预览，尚未接入 GPT 或自动抓取技能。

仓库：https://github.com/youseihuayu-wonderful/GPT

第一阶段已实现：Franka Panda 七轴机械臂、双指夹爪、桌面、黑碗、酒瓶、三层抽屉柜、酒架，以及直接读取 MuJoCo 世界坐标位姿的接口。浏览器画面来自正在运行的 MuJoCo Renderer，不是视频或预录轨迹回放。没有视觉识别、GPT、自动抓取或 DP。

## 启动与右侧显示

先克隆仓库，再安装锁定的依赖（需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)）：

```sh
git clone https://github.com/youseihuayu-wonderful/GPT.git
cd GPT
uv sync --locked
uv run python server.py --port 8765
```

服务仅监听 `127.0.0.1`。另一个终端中执行：

```sh
terminal-browser open http://127.0.0.1:8765 --split right --size 0.48
```

也可以直接在浏览器打开 http://127.0.0.1:8765 。服务停止后重新执行启动命令，再点击页面的“重新连接”。没有外部 CDN、API key 或 GPU 服务依赖。

- 暂停／开始运行：停止／继续物理积分。
- 单步：暂停时推进 20 ms，即 10 个 2 ms 物理步。
- 重置：恢复初始状态、清零仿真时间并暂停。
- 视角：全景、正面、俯视。
- 没有浏览器连接时不推进物理；多个页面共享同一个仿真实例与控制状态。
- 画面目标更新频率为 20 Hz。显示的是仿真时间，不是宿主机墙钟时间；机器负载可能降低实际播放速度。

## 位姿接口

```sh
curl http://127.0.0.1:8765/api/state
uv run python simulation.py --seconds 10 --image /tmp/grasp-scene.jpg
```

Python 可直接使用：

```python
from simulation import Simulation

sim = Simulation()
try:
    sim.step(1.0)
    print(sim.get_object_pose("bowl_object"))
finally:
    sim.close()
```

可观测对象：`bowl_object`、`bottle_object`、`cabinet_cabinet_bottom`。`position_m` 为世界坐标中的刚体参考原点，单位米；`quaternion_wxyz` 顺序为 **w,x,y,z**。它们来自 `MjData.xpos/xquat`，不是视觉估计，也不一定是质心或物体表面的高度。每次积分后调用 `mj_forward`，保证位姿与当前积分状态一致。

## 物理与场景范围

- 重力 `-9.81 m/s²`，步长 `0.002 s`，`implicitfast` 积分。
- 机械臂由原模型的位置执行器保持初始姿态；夹爪保持约 80 mm 开口。不包含到达、抓取、抬起控制。
- 黑碗、酒瓶使用自由关节和原资产碰撞几何，受重力和桌面接触支撑。没有绑定到机械臂。
- 抽屉柜和酒架是固定工位设施；保留柜子三个滑动抽屉关节，底层抽屉初始位移 `-0.12 m`。
- 桌面高度 `0.72 m`，台面 `1.30 × 0.90 m`。物体在重置后会短暂下落并在桌上稳定。
- 原视频对应 LIBERO-10 的 `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`。本项目复用同款物体资产、采用 Menagerie Panda，在独立桌面上重新布置；**不是原 LIBERO 任务初始状态、控制器、轨迹或 benchmark 的复现**。
- 原视频用碰撞几何进行软件绘制，本项目用原模型的视觉网格渲染，因此外观有所不同。黑碗材质调暗以便识别，碰撞几何没有改动。

## 文件与来源

- `scene.xml`：桌面、灯光和机械臂／物体安装坐标。
- `simulation.py`：模型组合、物理积分、重置、位姿读取、渲染及命令行入口。
- `server.py`：本地 HTTP／WebSocket 服务。
- `web/`：中文预览界面。
- `vendor/panda/`：[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/da76818e269b82289eba39808e2fb91d679d6994/franka_emika_panda)，提交 `da76818e269b82289eba39808e2fb91d679d6994`，保留 Apache-2.0 LICENSE。
- `vendor/libero/`：[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/assets)，提交 `8f1084e3132a39270c3a13ebe37270a43ece2a01`，保留上游 MIT LICENSE。资源从本机现有研究工程复制，原工程未修改。

## 已执行的物理验证

独立运行 20 s 仿真：黑碗和酒瓶通过桌面接触稳定支撑；无 MuJoCo 数值警告；观测姿态四元数保持归一化；重置精确恢复初始 qpos 和位姿；单步推进 20 ms。验证了实时渲染，不据此声称已完成抓取或具备硬件精度。

浏览器中实际操作并核对了暂停、重置到 `0.000 s`、单步到 `0.020 s`、暂停状态下切换视角后更新真实图像、恢复运行及位姿显示；浏览器没有报告脚本错误。
