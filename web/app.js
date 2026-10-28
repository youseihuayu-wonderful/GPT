"use strict";

const byId = (id) => document.getElementById(id);
const ui = {
  status: byId("status"), time: byId("time"), frame: byId("frame"),
  empty: byId("empty-frame"), stale: byId("stale-frame"),
  toggle: byId("toggle"), step: byId("step"), reset: byId("reset"),
  camera: byId("camera"), error: byId("error"), reconnect: byId("reconnect"),
  reconnectPanel: byId("reconnect-panel"), scene: byId("scene-name"),
  robot: byId("robot-name"), gripper: byId("gripper"), contacts: byId("contacts"),
  timestep: byId("timestep"), note: byId("telemetry-note"), objects: byId("objects"),
  joints: byId("joints"), source: byId("source"), warnings: byId("warnings"),
  grasp: byId("grasp"), skillPanel: byId("skill-panel"), skillResult: byId("skill-result"),
  skillHint: byId("skill-hint"), skillPhase: byId("skill-phase"),
  skillMessage: byId("skill-message"), skillFailure: byId("skill-failure"),
  skillElapsed: byId("skill-elapsed"), skillLift: byId("skill-lift"),
  skillHold: byId("skill-hold"), skillPhaseElapsed: byId("skill-phase-elapsed"),
  phaseTimes: byId("phase-times"), skillStages: byId("skill-stages"),
  graspCustom: byId("grasp-custom"), parameterError: byId("parameter-error"),
  appliedParameters: byId("applied-parameters"), earlyClose: byId("early-close"),
  skillOverlap: byId("skill-overlap"), closeStarted: byId("close-started"),
  descentCompleted: byId("descent-completed"),
  experimentsRefresh: byId("experiments-refresh"), experimentsStatus: byId("experiments-status"),
  experimentsError: byId("experiments-error"), experimentsReport: byId("experiments-report"),
  experimentsMeta: byId("experiments-meta"), experimentRuns: byId("experiment-runs"),
  plannerPanel: byId("planner-panel"), plannerStatus: byId("planner-status"),
  plannerTask: byId("planner-task"), plannerHint: byId("planner-hint"),
  planGenerate: byId("plan-generate"), planExecute: byId("plan-execute"),
  planCancel: byId("plan-cancel"), plannerSetup: byId("planner-setup"),
  plannerError: byId("planner-error"), plannerModel: byId("planner-model"),
  plannerModelLabel: byId("planner-model-label"), plannerLatency: byId("planner-latency"),
  plannerResponse: byId("planner-response"), plannerSubmitted: byId("planner-submitted"),
  plannerSubmittedTask: byId("planner-submitted-task"), planPreview: byId("plan-preview"),
  planPreviewTitle: byId("plan-preview-title"), planSummary: byId("plan-summary"),
  planReason: byId("plan-reason"), planParameters: byId("plan-parameters"),
  planSteps: byId("plan-steps"),
};
let socket = null;
let state = null;
let hasFrame = false;
const objectRows = [];
const jointOutputs = [];
let lastSkill = null;
let lastPlanner = null;
let pendingPlannerAction = null;
let plannerDisplayKey = null;
let planPreviewKey = null;
const plannerLabels = {
  unconfigured: "未配置 API 密钥", idle: "待规划", planning: "正在请求 GPT",
  ready: "计划已校验 · 待执行", unsupported: "任务不受支持", failed: "失败",
  executing: "物理执行中", succeeded: "物理抓取成功", cancelled: "规划已取消",
};
const phaseLabels = {
  idle: "待开始", prepare: "准备", approach: "接近", descend: "下降",
  close: "闭合夹爪", lift: "抬升", hold: "稳定保持", done: "完成", failed: "失败",
};
const skillLabels = { idle: "待开始", running: "执行中", succeeded: "抓取成功", failed: "抓取失败" };
const parameterFields = [
  { key: "initial_opening_m", input: byId("initial-opening"), label: "初始开度", unit: "mm", baseline: 0.08 },
  { key: "descent_speed_m_s", input: byId("descent-speed"), label: "下降速度", unit: "mm/s", baseline: 0.10 },
  { key: "close_trigger_height_m", input: byId("close-trigger-height"), label: "提前闭合高度", unit: "mm", baseline: 0 },
];

function validParameters(parameters) {
  return parameters && parameterFields.every(({ key }) => Number.isFinite(parameters[key]));
}

function parameterSummary(parameters) {
  return parameterFields.map(({ key, label, unit }) => `${label} ${number(parameters[key] * 1000, 1)} ${unit}`).join(" · ");
}

function parameterMode(parameters) {
  return parameterFields.every(({ key, baseline }) => parameters[key] === baseline)
    ? "baseline 等效参数（顺序下降后闭合）" : "自定义参数";
}

function parameterLimits(skill) {
  const limits = skill?.parameter_limits;
  return limits && parameterFields.every(({ key }) => Number.isFinite(limits[key]?.min)
    && Number.isFinite(limits[key]?.max) && limits[key].min <= limits[key].max) ? limits : null;
}

function updateParameterControls(ready, skill) {
  const limits = ready ? parameterLimits(skill) : null;
  ui.graspCustom.disabled = ui.grasp.disabled || !limits;
  parameterFields.forEach(({ key, input, unit }) => {
    input.disabled = !limits;
    const range = byId(`${input.id}-range`);
    if (limits) {
      input.min = String(limits[key].min * 1000);
      input.max = String(limits[key].max * 1000);
      range.textContent = `${input.min}–${input.max} ${unit}`;
    } else {
      input.removeAttribute("min");
      input.removeAttribute("max");
      range.textContent = "等待后端有效范围";
    }
  });
}

function readCustomParameters() {
  const limits = parameterLimits(state?.skill);
  if (!limits) throw new Error("后端尚未报告有效参数范围，无法运行自定义抓取。");
  const parameters = {};
  parameterFields.forEach(({ key, input, label, unit }) => {
    const value = input.valueAsNumber;
    if (input.value.trim() === "" || !Number.isFinite(value)) {
      input.setAttribute("aria-invalid", "true");
      throw new Error(`${label}必须填写有限数值（${unit}）。`);
    }
    const siValue = value / 1000;
    if (siValue < limits[key].min || siValue > limits[key].max) {
      input.setAttribute("aria-invalid", "true");
      throw new Error(`${label}须在 ${limits[key].min * 1000}–${limits[key].max * 1000} ${unit} 之间。`);
    }
    input.removeAttribute("aria-invalid");
    parameters[key] = siValue;
  });
  return parameters;
}

function validExperimentRun(run) {
  return run && typeof run.name === "string" && validParameters(run.parameters)
    && ["succeeded", "failed"].includes(run.status)
    && ["elapsed_s", "lift_height_m", "hold_elapsed_s", "descent_close_overlap_s", "validation_elapsed_s"]
      .every((key) => Number.isFinite(run[key]))
    && ["close_started_s", "descent_completed_s"]
      .every((key) => run[key] === null || Number.isFinite(run[key]))
    && (run.failure_reason === null || typeof run.failure_reason === "string")
    && run.phase_times_s && typeof run.phase_times_s === "object" && !Array.isArray(run.phase_times_s)
    && Object.values(run.phase_times_s).every(Number.isFinite)
    && Array.isArray(run.warnings) && run.warnings.every((warning) => typeof warning === "string");
}

function renderExperimentRun(run) {
  const row = document.createElement("tr");
  row.dataset.status = run.status;
  const texts = [
    `${run.name}\n${parameterMode(run.parameters)}\n${parameterSummary(run.parameters)}`,
    `${run.status === "succeeded" ? "成功" : "失败"}${run.failure_reason ? `\n原因：${run.failure_reason}` : run.status === "failed" ? "\n后端未报告失败原因" : ""}`,
    `${number(run.elapsed_s, 3)} / ${number(run.validation_elapsed_s, 3)}`,
    `${number(run.lift_height_m * 100, 2)} / ${number(run.hold_elapsed_s, 3)}`,
    number(run.descent_close_overlap_s, 3),
    `${run.close_started_s === null ? "未开始" : number(run.close_started_s, 3)} / ${run.descent_completed_s === null ? "未完成" : number(run.descent_completed_s, 3)}`,
    `${Object.entries(run.phase_times_s).map(([phase, elapsed]) => `${phaseLabels[phase] || phase}：${number(elapsed, 3)} s`).join("\n") || "未报告阶段耗时"}\n${run.warnings.length ? `警告：${run.warnings.join("\n")}` : "无后端警告"}`,
  ];
  texts.forEach((text) => {
    const cell = document.createElement("td");
    cell.textContent = text;
    row.append(cell);
  });
  const actionCell = document.createElement("td");
  const copy = document.createElement("button");
  copy.type = "button";
  copy.textContent = "填入参数";
  copy.setAttribute("aria-label", `将实验 ${run.name} 的参数填入输入框，不运行`);
  copy.addEventListener("click", () => {
    parameterFields.forEach(({ key, input }) => {
      input.value = String(run.parameters[key] * 1000);
      input.removeAttribute("aria-invalid");
    });
    ui.parameterError.textContent = "";
    ui.parameterError.hidden = true;
    ui.experimentsStatus.textContent = `已填入「${run.name}」的参数，尚未运行或重置。仅连接且任务待开始时可运行；已有结果须先重置。`;
  });
  actionCell.append(copy);
  row.append(actionCell);
  return row;
}

async function loadExperiments() {
  ui.experimentsRefresh.disabled = true;
  ui.experimentsStatus.textContent = "正在读取已存储实验报告…";
  ui.experimentsError.hidden = true;
  ui.experimentsReport.hidden = true;
  ui.experimentRuns.replaceChildren();
  try {
    const response = await fetch("/api/experiments", { cache: "no-store" });
    if (response.status === 404) throw new Error("尚无实验报告（/api/experiments 返回 404）。未显示任何替代结果。");
    if (!response.ok) throw new Error(`实验报告读取失败（HTTP ${response.status}）。`);
    const report = await response.json();
    if (!report || report.kind !== "fixed_scene_parameter_sweep"
        || typeof report.generated_at !== "string" || !Number.isFinite(Date.parse(report.generated_at))
        || !Number.isFinite(report.validation_hold_s) || typeof report.note !== "string"
        || !Array.isArray(report.runs) || !report.runs.every(validExperimentRun)) {
      throw new Error("实验报告格式或运行字段不完整；无法显示可信对照数据。");
    }
    ui.experimentsMeta.textContent = `报告生成：${report.generated_at} · 独立验证保持：${number(report.validation_hold_s, 1)} s。${report.note}`;
    ui.experimentRuns.replaceChildren(...report.runs.map(renderExperimentRun));
    ui.experimentsReport.hidden = false;
    ui.experimentsStatus.textContent = report.runs.length
      ? `已读取 ${report.runs.length} 组已存储单次运行。填入参数不会自动启动任务。`
      : "报告存在，但没有运行记录；未生成替代结果。";
  } catch (error) {
    ui.experimentsStatus.textContent = "实验数据不可用；实时仿真仍独立显示。";
    ui.experimentsError.textContent = error.message;
    ui.experimentsError.hidden = false;
  } finally {
    ui.experimentsRefresh.disabled = false;
  }
}

function plannerLocksScene() {
  return pendingPlannerAction !== null || ["planning", "ready"].includes(state?.planner?.status);
}

function renderPlanner(ready) {
  const planner = ready ? state.planner : lastPlanner;
  const known = planner && Object.hasOwn(plannerLabels, planner.status);
  const idle = ready && state.skill?.status === "idle";
  const locked = known && ["planning", "ready", "executing"].includes(planner.status);
  const pending = pendingPlannerAction !== null;
  const task = ui.plannerTask.value.trim();
  ui.plannerTask.disabled = !idle || !known || locked || pending;
  ui.planGenerate.disabled = !idle || !known || locked || pending || !task || task.length > 1000;
  ui.planExecute.disabled = !idle || !known || planner.status !== "ready"
    || planner.plan?.decision !== "plan" || !Number.isInteger(planner.plan_id) || pending;
  ui.planCancel.disabled = !ready || !known || !["planning", "ready"].includes(planner.status) || pending;
  ui.planCancel.textContent = known && planner.status === "ready" ? "放弃待执行计划" : "取消规划";

  const displayKey = JSON.stringify([ready, planner?.plan_id, planner?.status, planner?.configured,
    planner?.model, planner?.latency_s, planner?.response_id, planner?.error, pendingPlannerAction,
    state?.skill?.status, state?.skill?.phase, state?.paused]);
  if (displayKey === plannerDisplayKey) return;
  plannerDisplayKey = displayKey;
  ui.plannerPanel.dataset.status = !ready ? "stale" : known ? planner.status : "waiting";
  ui.plannerStatus.textContent = known
    ? `${plannerLabels[planner.status]}${!ready ? " · 最后状态" : planner.status === "executing" && state.paused ? " · 已暂停" : ""}`
    : "等待规划状态";
  ui.plannerHint.textContent = !ready
    ? "连接或状态不可用；预览仅为最后状态，不能确认当前规划或物理结果。不会自动重连、请求 API 或执行。"
    : !known ? "后端尚未报告规划功能状态，暂不能请求或执行。"
    : pending ? "操作已发送，等待后端确认；不会重复提交。"
    : planner.status === "planning" ? "正在等待真实 API 响应；场景已冻结。可取消规划或重置，不会自动执行。"
    : planner.status === "ready" ? "请检查下方本次任务和真实计划，再明确点击执行；放弃计划后才能使用手动控制。"
    : planner.status === "unsupported" ? "GPT 判定任务超出固定控制器能力；没有可执行动作。请修改任务后重新生成。"
    : planner.status === "executing" ? "已交由真实物理控制器执行；可暂停或重置，不能用「取消规划」中止物理执行。"
    : !idle ? "物理任务不在待开始状态；请先重置，再生成新计划。抓取结果以下方物理真值为准。"
    : planner.status === "failed" ? "本次规划未获得可执行结果；请查看错误并按需重新生成，不会自动重试。"
    : "生成只请求规划，不会执行。密钥仅从本机 .env 读取；已配置不代表认证成功。";
  ui.plannerSetup.hidden = !known || planner.configured;
  ui.plannerError.textContent = known ? planner.error || "" : "";
  ui.plannerError.hidden = !ui.plannerError.textContent;
  ui.plannerModelLabel.textContent = known && planner.response_id ? "实际 API 响应模型" : "请求模型 · 尚无 API 响应";
  ui.plannerModel.textContent = known ? planner.model : "—";
  ui.plannerLatency.textContent = number(known ? planner.latency_s : undefined, 2);
  ui.plannerResponse.textContent = known && planner.response_id ? `实际 API 响应 ID：${planner.response_id}` : "";
  ui.plannerResponse.hidden = !ui.plannerResponse.textContent;
  ui.plannerSubmitted.hidden = !known || !planner.task;
  ui.plannerSubmittedTask.textContent = known ? planner.task : "";
  const plan = known ? planner.plan : null;
  const previewKey = JSON.stringify([ready, planner?.plan_id, plan?.decision]);
  ui.planPreview.hidden = !plan;
  if (previewKey !== planPreviewKey) {
    planPreviewKey = previewKey;
    ui.planSummary.textContent = plan?.summary || "";
    ui.planReason.textContent = plan ? `规划依据：${plan.reason}` : "";
    ui.planParameters.hidden = !plan?.parameters;
    ui.planParameters.textContent = plan?.parameters ? `已校验参数：${parameterSummary(plan.parameters)}` : "";
    ui.planSteps.replaceChildren();
    for (const step of plan?.steps || []) {
      const item = document.createElement("li");
      item.dataset.phase = step.phase;
      const title = document.createElement("strong");
      title.textContent = phaseLabels[step.phase] || step.phase;
      const instruction = document.createElement("span");
      instruction.textContent = step.instruction;
      item.append(title, instruction);
      ui.planSteps.append(item);
    }
  }
  ui.planPreviewTitle.textContent = plan?.decision === "unsupported"
    ? "GPT 不支持此任务 · 不可执行"
    : planner?.status === "ready" ? "后端已校验计划 · 尚未执行" : "后端已校验计划 · 实际结果见物理状态";
  const executing = known && ["executing", "succeeded", "failed"].includes(planner.status)
    && plan?.decision === "plan";
  for (const item of ui.planSteps.children) {
    const active = executing && state?.skill?.phase === item.dataset.phase;
    item.classList.toggle("active", active);
    if (active) item.setAttribute("aria-current", "step");
    else item.removeAttribute("aria-current");
  }
}

function sendPlanner(command) {
  if (send(command)) {
    pendingPlannerAction = command.action;
    updateControls();
  }
}


function renderSkill(ready) {
  const skill = ready ? state.skill : lastSkill;
  const known = skill && Object.hasOwn(skillLabels, skill.status);
  ui.grasp.disabled = !ready || !known || skill.status !== "idle" || plannerLocksScene();
  updateParameterControls(ready, known ? skill : null);
  ui.skillPanel.dataset.status = !ready ? "stale" : known ? skill.status : "waiting";
  ui.skillResult.textContent = known
    ? `${skillLabels[skill.status]}${!ready ? " · 最后状态" : skill.status === "running" && state.paused ? " · 已暂停" : ""}`
    : "等待状态";
  ui.skillHint.textContent = !ready
    ? "未连接或状态不可用；已有数值仅为最后状态，无法确认当前执行结果。"
    : !known ? "等待后端技能状态；暂不能开始抓取。"
    : plannerLocksScene() ? "GPT 规划或待执行计划已锁定场景；先取消规划、放弃计划或重置，才能运行手动对照。"
    : skill.status === "idle" ? "选择固定 baseline 或自定义参数运行；将自动恢复物理仿真。"
    : skill.status === "running" ? "任务执行中，可暂停；重新尝试前请先重置。"
    : "本次任务已结束；请点击「重置」后重新尝试。";
  ui.skillPhase.textContent = known ? (phaseLabels[skill.phase] || skill.phase || "未报告") : "等待状态";
  ui.skillMessage.textContent = known ? (skill.message || "后端未提供阶段说明。") : "等待后端技能数据，不推断进度或结果。";
  ui.skillFailure.textContent = known && skill.failure_reason ? `失败原因：${skill.failure_reason}` : "";
  ui.skillFailure.hidden = !ui.skillFailure.textContent;
  ui.skillElapsed.textContent = number(known ? skill.elapsed_s : undefined, 2);
  ui.skillLift.textContent = number(known && Number.isFinite(skill.lift_height_m) ? skill.lift_height_m * 100 : undefined, 1);
  ui.skillHold.textContent = number(known ? skill.hold_elapsed_s : undefined, 2);
  ui.skillPhaseElapsed.textContent = number(known ? skill.phase_elapsed_s : undefined, 2);
  ui.skillOverlap.textContent = number(known ? skill.descent_close_overlap_s : undefined, 3);
  ui.closeStarted.textContent = number(known ? skill.close_started_s : undefined, 3);
  ui.descentCompleted.textContent = number(known ? skill.descent_completed_s : undefined, 3);
  const parameters = known ? skill.parameters : null;
  ui.appliedParameters.textContent = validParameters(parameters)
    ? `${!ready ? "最后状态" : skill.status === "idle" ? "后端待用" : "实际应用"}：${parameterMode(parameters)}；${parameterSummary(parameters)}`
    : "后端参数：未报告完整参数，不使用输入框数值代替。";
  const closingDuringDescent = ready && known && skill.status === "running"
    && skill.phase === "descend" && Number.isFinite(skill.close_started_s);
  ui.earlyClose.hidden = !closingDuringDescent;
  ui.earlyClose.textContent = closingDuringDescent
    ? `下降阶段已触发闭合 · 后端开始时间 ${number(skill.close_started_s, 3)} s${state.paused ? " · 已暂停" : ""}；实际运动重叠见下方测量。` : "";
  const stages = Array.from(ui.skillStages.children);
  const current = known ? stages.findIndex((stage) => stage.dataset.phase === skill.phase) : -1;
  stages.forEach((stage, index) => {
    const active = known && stage.dataset.phase === skill.phase;
    stage.classList.toggle("active", active);
    stage.classList.toggle("completed", known && (skill.status === "succeeded" || (current >= 0 && index < current)));
    if (active) stage.setAttribute("aria-current", "step");
    else stage.removeAttribute("aria-current");
  });
  ui.phaseTimes.replaceChildren();
  const times = known && skill.phase_times_s && typeof skill.phase_times_s === "object"
    ? Object.entries(skill.phase_times_s) : [];
  if (!times.length) {
    const waiting = document.createElement("dt");
    waiting.textContent = known ? "后端尚未报告阶段耗时" : "等待后端状态";
    ui.phaseTimes.append(waiting);
  }
  times.forEach(([phase, elapsed]) => {
    const label = document.createElement("dt");
    const value = document.createElement("dd");
    label.textContent = phaseLabels[phase] || phase;
    value.textContent = `${number(elapsed, 2)} s`;
    ui.phaseTimes.append(label, value);
  });
}

function number(value, decimals = 4) {
  return Number.isFinite(value) ? value.toFixed(decimals) : "—";
}

function setStatus(text, kind) {
  ui.status.textContent = text;
  ui.status.dataset.kind = kind;
}

function updateControls() {
  const ready = socket?.readyState === WebSocket.OPEN && state !== null;
  ui.toggle.disabled = !ready || (state.paused && plannerLocksScene());
  ui.reset.disabled = !ready;
  ui.camera.disabled = !ready;
  ui.step.disabled = !ready || !state.paused || plannerLocksScene();
  ui.toggle.textContent = state && !state.paused ? "暂停" : "开始运行";
  renderSkill(ready);
  renderPlanner(ready);
}

function showError(message) {
  ui.error.textContent = message;
  ui.error.hidden = false;
}

function createVector(label, axes) {
  const row = document.createElement("div");
  row.className = "vector";
  const title = document.createElement("span");
  title.className = "vector-label";
  title.textContent = label;
  row.append(title);
  const outputs = axes.map((axis) => {
    const component = document.createElement("span");
    component.className = "component";
    component.append(document.createTextNode(axis));
    const output = document.createElement("output");
    output.setAttribute("aria-label", `${label} ${axis}`);
    component.append(output);
    row.append(component);
    return output;
  });
  return { row, outputs };
}

function updateObjects(objects) {
  while (objectRows.length > objects.length) objectRows.pop().element.remove();
  objects.forEach((object, index) => {
    if (!objectRows[index]) {
      const element = document.createElement("article");
      element.className = "object";
      const title = document.createElement("h3");
      const position = createVector("位置 · m", ["X", "Y", "Z"]);
      const rotation = createVector("四元数", ["w", "x", "y", "z"]);
      element.append(title, position.row, rotation.row);
      ui.objects.append(element);
      objectRows.push({ element, title, position, rotation });
    }
    const row = objectRows[index];
    row.title.textContent = object.name;
    row.position.outputs.forEach((output, i) => { output.textContent = number(object.position_m[i]); });
    row.rotation.outputs.forEach((output, i) => { output.textContent = number(object.quaternion_wxyz[i]); });
  });
}

function updateJoints(positions) {
  if (jointOutputs.length === 0) ui.joints.textContent = "";
  while (jointOutputs.length > positions.length) jointOutputs.pop().parentElement.remove();
  positions.forEach((value, index) => {
    if (!jointOutputs[index]) {
      const component = document.createElement("span");
      component.className = "component";
      component.append(document.createTextNode(`J${index + 1}`));
      const output = document.createElement("output");
      output.setAttribute("aria-label", `关节 ${index + 1} 位置，弧度`);
      component.append(output);
      ui.joints.append(component);
      jointOutputs.push(output);
    }
    jointOutputs[index].textContent = number(value);
  });
}

function renderState(next) {
  if (typeof next.paused !== "boolean" || !Number.isFinite(next.time_s)
      || !Array.isArray(next.objects) || !Array.isArray(next.joint_positions_rad)
      || !Array.isArray(next.warnings)) {
    throw new Error("后端状态字段不完整，无法更新仿真状态。");
  }
  if (state === null) {
    plannerDisplayKey = null;
    planPreviewKey = null;
  }
  if (pendingPlannerAction && (next.planner?.plan_id !== state?.planner?.plan_id
      || next.planner?.status !== state?.planner?.status)) pendingPlannerAction = null;
  state = next;
  if (state.skill) lastSkill = state.skill;
  if (state.planner) lastPlanner = state.planner;
  ui.time.textContent = number(state.time_s, 3);
  ui.scene.textContent = state.scene_name;
  ui.robot.textContent = state.robot_name;
  ui.source.textContent = state.source;
  ui.gripper.textContent = number(state.gripper_opening_m * 1000, 2);
  ui.contacts.textContent = Number.isFinite(state.contacts) ? String(state.contacts) : "—";
  ui.timestep.textContent = number(state.timestep_s * 1000, 2);
  ui.note.textContent = state.objects.length ? "世界坐标 XYZ（m）；姿态四元数按 w、x、y、z 排列。" : "后端当前未报告物体。";
  updateObjects(state.objects);
  updateJoints(state.joint_positions_rad);
  ui.warnings.textContent = state.warnings.join("\n");
  ui.warnings.hidden = state.warnings.length === 0;
  setStatus(state.paused ? "已连接 · 已暂停" : "已连接 · 运行中", state.paused ? "paused" : "running");
  updateControls();
}

function send(command) {
  if (socket?.readyState !== WebSocket.OPEN || !state) {
    showError("仿真器尚未连接或尚未返回状态，无法发送操作。");
    return false;
  }
  try {
    socket.send(JSON.stringify(command));
    ui.error.hidden = true;
    return true;
  } catch (error) {
    showError(`操作未发送：${error.message}`);
    return false;
  }
}

function connect() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;
  state = null;
  pendingPlannerAction = null;
  ui.error.hidden = true;
  ui.reconnectPanel.hidden = true;
  ui.stale.hidden = !hasFrame;
  ui.stale.textContent = "正在重新连接 · 最后一帧";
  setStatus("正在连接", "connecting");
  updateControls();
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  try {
    socket = new WebSocket(`${protocol}//${location.host}/ws`);
  } catch (error) {
    setStatus("连接失败", "error");
    showError(`无法创建仿真连接：${error.message}`);
    ui.reconnectPanel.hidden = false;
    return;
  }
  socket.addEventListener("open", () => {
    setStatus("已连接 · 等待状态", "connecting");
  });
  socket.addEventListener("message", (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.error !== undefined) {
        pendingPlannerAction = null;
        updateControls();
        showError(`仿真器报告错误：${String(message.error)}`);
        setStatus("已连接 · 操作错误", "error");
        return;
      }
      if (!message.state || typeof message.image !== "string" || !message.image) {
        throw new Error("后端消息缺少状态或 JPEG 画面。");
      }
      renderState(message.state);
      ui.frame.src = `data:image/jpeg;base64,${message.image}`;
    } catch (error) {
      state = null;
      pendingPlannerAction = null;
      updateControls();
      setStatus("已连接 · 数据错误", "error");
      showError(`无法显示仿真数据：${error.message}`);
      ui.stale.textContent = "数据更新失败 · 最后一帧";
      ui.stale.hidden = !hasFrame;
    }
  });
  socket.addEventListener("error", () => {
    showError("WebSocket 连接发生错误。请确认本地仿真服务正在运行，再点击重新连接。");
    setStatus("连接错误", "error");
  });
  socket.addEventListener("close", (event) => {
    state = null;
    pendingPlannerAction = null;
    updateControls();
    setStatus("已断开 · 更新停止", "disconnected");
    ui.stale.textContent = "连接已中断 · 最后一帧";
    ui.stale.hidden = !hasFrame;
    ui.note.textContent = "连接已中断，下方数值为最后接收到的状态，不再实时更新。";
    ui.reconnectPanel.hidden = false;
    showError(`仿真连接已关闭（代码 ${event.code}）${event.reason ? `：${event.reason}` : "。"} 无法确认后端是否仍在运行。`);
  });
}

ui.frame.addEventListener("load", () => {
  hasFrame = true;
  ui.frame.hidden = false;
  ui.empty.hidden = true;
  ui.stale.hidden = socket?.readyState === WebSocket.OPEN && state !== null;
});
ui.frame.addEventListener("error", () => {
  showError("后端 JPEG 画面解码失败；未使用替代图像。请检查仿真服务输出。");
  ui.frame.hidden = true;
  ui.empty.hidden = false;
  ui.empty.querySelector("p").textContent = "仿真画面不可用";
  setStatus("画面错误", "error");
});
ui.toggle.addEventListener("click", () => {
  if (!ui.toggle.disabled) send({ action: state?.paused ? "play" : "pause" });
});
ui.step.addEventListener("click", () => {
  if (!ui.step.disabled && state?.paused) send({ action: "step" });
});
ui.reset.addEventListener("click", () => send({ action: "reset" }));
ui.grasp.addEventListener("click", () => {
  if (!ui.grasp.disabled && state?.skill?.status === "idle") send({ action: "grasp" });
});
ui.graspCustom.addEventListener("click", () => {
  if (ui.graspCustom.disabled || state?.skill?.status !== "idle") return;
  try {
    const parameters = readCustomParameters();
    ui.parameterError.textContent = "";
    ui.parameterError.hidden = true;
    send({ action: "grasp", parameters });
  } catch (error) {
    ui.parameterError.textContent = error.message;
    ui.parameterError.hidden = false;
  }
});
parameterFields.forEach(({ input }) => {
  input.addEventListener("input", () => {
    input.removeAttribute("aria-invalid");
    ui.parameterError.textContent = "";
    ui.parameterError.hidden = true;
  });
});
ui.plannerTask.addEventListener("input", updateControls);
ui.planGenerate.addEventListener("click", () => {
  if (!ui.planGenerate.disabled) sendPlanner({ action: "plan", task: ui.plannerTask.value.trim() });
});
ui.planExecute.addEventListener("click", () => {
  if (!ui.planExecute.disabled) sendPlanner({ action: "execute_plan", plan_id: state.planner.plan_id });
});
ui.planCancel.addEventListener("click", () => {
  if (!ui.planCancel.disabled) sendPlanner({ action: "cancel_plan" });
});
ui.camera.addEventListener("change", () => send({ action: "camera", view: ui.camera.value }));
ui.reconnect.addEventListener("click", connect);
ui.experimentsRefresh.addEventListener("click", loadExperiments);
loadExperiments();
connect();
