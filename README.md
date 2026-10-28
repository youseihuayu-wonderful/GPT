# GPT — 机器人多子任务执行仿真

为 GPT 高层任务规划准备可调用的机器人执行能力。现已完成物理场景、固定脚本 baseline，以及第三阶段参数化抓取与下降／闭合时序控制；尚未接入 GPT、视觉识别或 DP。

仓库：https://github.com/youseihuayu-wonderful/GPT

场景包含 Franka Panda 七轴机械臂、双指夹爪、桌面、黑碗、酒瓶、三层抽屉柜、酒架。浏览器画面由正在运行的 MuJoCo Renderer 生成，不是视频或预录轨迹回放。物体位姿直接取自仿真器。

实施顺序：

1. 搭建仿真场景：已完成，读取物体真值位姿。
2. 固定脚本抓取：已完成，保留顺序执行 baseline。
3. 参数化抓取与时序控制：已完成，支持真实运动重叠及固定场景参数小实验。
4. GPT 规划层：尚未接入；后续输出结构化子任务或参数，由本地控制器执行。
5. 跨物体、位置与方法的对比实验：尚未开展；当前参数小实验不等于这一阶段。

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

- 运行固定 baseline：以 80 mm 开口、100 mm/s 名义下降速度、0 mm 提前闭合执行。技能执行期间不能重复启动；成功或失败后需先重置。
- 运行自定义参数：校验并使用三个输入值，自动开始物理积分。编辑输入框不会改变正在执行的参数；“实际应用”显示后端采用的配置。
- 实验表中的“填入参数”只填表，不启动或重置；“刷新报告”读取最新保存的真实实验结果。
- 暂停／开始运行：停止／继续物理积分。
- 单步：暂停时推进 20 ms，即 10 个 2 ms 物理步。
- 重置：恢复物理初态、后端默认参数并清空计时，暂停仿真；页面输入值保留，便于再次执行同一配置。
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

## 抓取技能接口与判定

界面点击“运行固定 baseline”，或无界面执行：

```sh
uv run python simulation.py --grasp --image /tmp/grasp-result.jpg
uv run python -m unittest discover -s tests -p 'test_grasp.py' -v
```

`--grasp` 默认先运行 1 s 场景稳定过程，再执行完整技能；失败时退出码为 1。渲染图为真实终态，不是成功示意图。

Python 调用 `sim.start_grasp()` 后持续调用 `sim.step(0.02)`，通过 `sim.state()["skill"]` 读取 `idle / running / succeeded / failed` 状态。WebSocket `/ws` 接收 `{"action":"grasp"}`，HTTP `/api/state` 同样返回技能状态。只有 idle 可以启动；重置取消任务、清空计时并恢复物理初态。

执行阶段：等待桌面稳定并设置初始开口、接近、下降、闭合、抬升、保持。默认 baseline 保持顺序执行，没有人为插入停顿；非零触发高度允许下降中开始闭合，但仍须确认下降到位和双侧接触后才抬升。

- 碗碰撞外径约 107 mm，大于夹爪约 80 mm 的开口，因此夹持近侧碗沿：一指在碗内，一指在碗外。没有修改碗的尺寸、壁厚、质量、摩擦或碰撞网格。
- 机械臂用数值逆运动学、五次平滑关节插值和偏置力前馈，仍通过原位置执行器及其力限制运动。不是通用避障规划器。
- 夹爪保留原关节、腱和 ±100 N 执行器力限制，改用腱力输入：打开时为位置 PD，闭合时为 4 N 腱闭合力，并附加与速度成正比的阻尼；力由腱分配至两指。这避免薄碗沿处零开口位置伺服的夹持力随误差缩小而不足；没有向碗直接施加控制力。
- 场景设置 `noslip_iterations="5"`，使用 MuJoCo 的[摩擦防漂移后处理](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#option-noslip-iterations)，抑制软接触模型产生的持续数值滑移。没有增加材质摩擦系数，也没有添加物体绑定或焊接约束。
- 成功要求：相对准备阶段的稳定碗位抬升至少 0.10 m、两指均有实际接触力、没有桌面支撑，并保持 1 s；保持期间碗与夹爪相对位置漂移不得超过 0.01 m。
- 不可达、到位超时、未建立双侧接触、抬升丢失接触或保持失败均返回 failed；首次完成前的执行有 20 s 仿真时间上限。成功后继续监测持物状态，后续掉落或滑移会改报 failed，而不是保留过期的成功状态。失败不自动重试。
- 页面记录总执行时间、各阶段时间、实际抬升高度和保持时间；暂停冻结物理与这些计时。执行时间从调用技能起计，不包含 CLI 的初始场景稳定过程；若完成后失去夹持，失败时间记录至检测时刻。

这是固定工位下的一个可调用抓取技能，不等于完整的 GPT 多子任务规划系统，也尚未实现放入抽屉和关闭抽屉。

## 第三阶段：参数与重叠控制

`GraspParameters` 是不可变的已校验配置；Python／WebSocket 使用 SI 单位，界面和命令行使用 mm、mm/s：

| 字段 | 含义 | 界面允许范围 | 默认值 |
| --- | --- | --- | --- |
| `initial_opening_m` | 初始总开度，两个夹指滑动关节位移之和 | 20–80 mm | 80 mm |
| `descent_speed_m_s` | 用于计算下降轨迹时长的名义速度 | 40–200 mm/s | 100 mm/s |
| `close_trigger_height_m` | TCP 距下降目标的剩余高度，达到后触发闭合 | 0–20 mm | 0 mm |

下降采用五次平滑关节轨迹，速度设置还受关节轨迹速率和最短运动时长约束，**不是 TCP 恒速控制或实测速度保证**。触发高度相对下降目标，不是相对桌面或碗顶；非零值还要求 TCP 横向误差小于 4 mm。0 保持“确认下降完成后闭合”。允许范围不意味着每个配置都能成功。

在已有 `sim` 实例上使用：

```python
from grasp import GraspParameters

sim.start_grasp(GraspParameters(
    initial_opening_m=0.08,
    descent_speed_m_s=0.20,
    close_trigger_height_m=0.004,
))
```

WebSocket 请求：

```json
{"action":"grasp","parameters":{"initial_opening_m":0.08,"descent_speed_m_s":0.20,"close_trigger_height_m":0.004}}
```

省略 `parameters` 仍运行 baseline。非数值、布尔值、NaN、无穷值、越界数值和未知参数在启动前拒绝，不悄悄裁剪或替换默认值。`/api/state` 中的 `skill.parameter_limits` 提供同一套边界，界面不另设范围。

命令行复现同一重置初态：

```sh
uv run python simulation.py --seconds 0 --grasp \
  --initial-opening-mm 80 --descent-speed-mm-s 200 \
  --close-trigger-height-mm 4 --image /tmp/parameter-grasp.jpg
```

三个参数选项均要求 `--grasp`。`--seconds 0` 让技能自身完成准备阶段，与页面重置后运行及参数实验一致；不指定时仍保留原 CLI 的 1 s 预稳定过程。

新增技能状态：

- `parameters`：本次实际应用的参数，不使用页面输入值冒充。
- `close_started_s`、`descent_completed_s`：闭合指令开始、反馈确认下降完成的时刻，均相对任务起点；未发生时为 `null`。闭合开始不等于接触已经建立。
- `descent_close_overlap_s`：下降阶段内，实际 TCP 向下速度与两指总闭合速度均超过 `0.0001 m/s` 时，累计对应物理步长。不是两条指令时间差，也不把静止夹紧计作运动；暂停时不累加。

## 固定场景参数小实验

```sh
uv run python experiments.py --output results/parameter_sweep.json
```

每组创建独立仿真，从相同重置初态启动，不使用重试；首次成功后额外推进 5 s 检查持续夹持。失败按实际结果记录；求解异常会停止实验并报错，不伪造成功数据。报告完整写入后才替换旧文件。页面 `/api/experiments` 读取该报告，缺失或损坏会明确报错。

已实际运行的结果（全部时间为仿真秒）：

| 配置 | 开口 mm | 名义下降速度 mm/s | 提前闭合 mm | 执行时间 s | 实际运动重叠 s | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 80 | 100 | 0 | 13.272 | 0.000 | 成功 |
| smaller_opening | 50 | 100 | 0 | 13.318 | 0.000 | 成功 |
| faster_descent | 80 | 160 | 0 | 12.146 | 0.000 | 成功 |
| overlap | 80 | 100 | 4 | 13.140 | 0.144 | 成功 |
| fast_overlap | 80 | 160 | 4 | 12.012 | 0.142 | 成功 |
| higher_speed_overlap | 80 | 200 | 4 | 11.684 | 0.084 | 成功 |
| combined | 50 | 160 | 4 | 12.102 | 0.096 | 成功 |
| early_closure | 50 | 160 | 20 | 12.100 | 0.160 | 成功 |

表中最短配置比 baseline 减少约 12% 执行时间；不是全局最优结论。缩小开口不必然更快：准备阶段调节开口也需要时间。提前 20 mm 的终态抬升约 12.25 cm，低于提前 4 mm 组合的约 13.68 cm，不能仅按闭合越早越好来选参数。

以上每组仅在一个固定场景运行一次，无位置扰动；不能据此声称统计成功率、跨场景性能或 GPT 优于规则。保存的实验表与正在运行的任务独立；可填入某行参数，再重置并运行以观看真实执行。



## 物理与场景范围

- 重力 `-9.81 m/s²`，步长 `0.002 s`，`implicitfast` 积分。
- 重置时机械臂保持初始姿态、夹爪打开；启动技能后由本地反馈控制器执行抓取、抬升和保持。没有直接设置碗的跟随轨迹或约束绑定。
- 黑碗、酒瓶使用自由关节和原资产碰撞几何，受重力和桌面接触支撑。没有绑定到机械臂。
- 抽屉柜和酒架是固定工位设施；保留柜子三个滑动抽屉关节，底层抽屉初始位移 `-0.12 m`。
- 桌面高度 `0.72 m`，台面 `1.30 × 0.90 m`。物体在重置后会短暂下落并在桌上稳定。
- 原视频对应 LIBERO-10 的 `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`。本项目复用同款物体资产、采用 Menagerie Panda，在独立桌面上重新布置；**不是原 LIBERO 任务初始状态、控制器、轨迹或 benchmark 的复现**。
- 原视频用碰撞几何进行软件绘制，本项目用原模型的视觉网格渲染，因此外观有所不同。黑碗材质调暗以便识别，碰撞几何没有改动。

## 文件与来源

- `scene.xml`：桌面、灯光和机械臂／物体安装坐标。
- `simulation.py`：模型组合、物理积分、重置、位姿读取、渲染及命令行入口。
- `grasp.py`：不可变参数配置、抓取状态机、数值逆运动学、执行器控制、运动重叠测量及成功／失败检查。
- `tests/test_grasp.py`：不依赖图像或 mock 的真实物理行为检查。
- `experiments.py`：独立真实物理参数对照入口。
- `results/parameter_sweep.json`：已执行的八组原始测量报告，不是手工填充数据。
- `server.py`：本地 HTTP／WebSocket 服务。
- `web/`：中文预览界面。
- `vendor/panda/`：[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/da76818e269b82289eba39808e2fb91d679d6994/franka_emika_panda)，提交 `da76818e269b82289eba39808e2fb91d679d6994`，保留 Apache-2.0 LICENSE。
- `vendor/libero/`：[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/assets)，提交 `8f1084e3132a39270c3a13ebe37270a43ece2a01`，保留上游 MIT LICENSE。资源从本机现有研究工程复制，原工程未修改。

## 已执行的物理验证

场景阶段曾独立运行 20 s，验证重力接触稳定、归一化位姿、精确重置与 20 ms 单步。抓取阶段实际 CLI 运行成功：技能耗时约 13.074 s，碗抬升约 0.1392 m，并稳定保持 1 s；无 MuJoCo 数值警告。上述数字只代表当前固定场景，不是成功率、提速或真实硬件性能结论。

9 项行为检查通过，覆盖真实抓起并持续保持、保持途中重置再执行、重复启动拒绝、夹爪失效及成功后掉落、实际初始开口、下降／闭合同时运动与更短耗时、无效参数拒绝和拒绝后的状态不变。baseline、重置重跑及 50/160/4 组合均在成功后继续推进 60 s，检查物理持物而不只看成功标志。此前 baseline 还单独验证了成功后继续仿真 120 s 不掉落。

浏览器实际验证了非法输入阻止执行、从真实实验报告填入参数、按自定义配置运行、下降阶段触发闭合时暂停，以及暂停中切换视角后恢复执行。50/160/4 配置的实际执行时间约 12.102 s，运动重叠 0.096 s；暂停期间物理时间和重叠计时均不变化。无浏览器脚本错误。HTTP／WebSocket 另验证了六类无效参数请求均保持物理状态为暂停且时间为零。
