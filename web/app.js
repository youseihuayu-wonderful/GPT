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
  planContinuous: byId("plan-continuous"), planNext: byId("plan-next"),
  planCancel: byId("plan-cancel"), plannerSetup: byId("planner-setup"),
  plannerError: byId("planner-error"), plannerModel: byId("planner-model"),
  plannerModelLabel: byId("planner-model-label"), plannerLatency: byId("planner-latency"),
  plannerResponse: byId("planner-response"), plannerSubmitted: byId("planner-submitted"),
  plannerSubmittedTask: byId("planner-submitted-task"), planPreview: byId("plan-preview"),
  planPreviewTitle: byId("plan-preview-title"), planSummary: byId("plan-summary"),
  planReason: byId("plan-reason"), planParameters: byId("plan-parameters"),
  planGoal: byId("plan-goal"), planSkill: byId("plan-skill"),
  plannerTotalLatency: byId("planner-total-latency"), plannerMode: byId("planner-mode"),
  plannerGoalChecks: byId("planner-goal-checks"), plannerHistory: byId("planner-history"),
  plannerHistoryEmpty: byId("planner-history-empty"),
  storeScript: byId("store-script"), goalCriteria: byId("goal-criteria"),
  taskOverlay: byId("task-overlay"), overlayTask: byId("overlay-task"), overlayPhase: byId("overlay-phase"),
  storageStatus: byId("storage-status"), storeContained: byId("store-contained"),
  storeSupported: byId("store-supported"), storeClear: byId("store-clear"),
  storeDrawer: byId("store-drawer"), storeClosed: byId("store-closed"),
  storeHomeError: byId("store-home-error"), storeHome: byId("store-home"), storeVerified: byId("store-verified"),
  collisionEpisodes: byId("collision-episodes"), collisionForce: byId("collision-force"),
  manualSkill: byId("manual-skill"), startSkill: byId("start-skill"),
  manualReadiness: byId("manual-skill-readiness"), manualParameters: byId("manual-skill-parameters"),
  manualContract: byId("manual-skill-contract"), manualWatchdog: byId("manual-skill-watchdog"),
  manualTargets: byId("manual-skill-targets"), manualPhases: byId("manual-skill-phases"),
  manualPreconditions: byId("manual-skill-preconditions"), manualPostconditions: byId("manual-skill-postconditions"),
  currentSkill: byId("current-skill"), skillInvocation: byId("skill-invocation"),
  skillPostconditions: byId("skill-postconditions"),
  storeHandClear: byId("store-hand-clear"), storePushContact: byId("store-push-contact"),
  sortingPanel: byId("sorting-panel"), sortStart: byId("sort-start"),
  sortingStatus: byId("sorting-status"), sortingHint: byId("sorting-hint"),
  sortingCurrent: byId("sorting-current"), sortingPhase: byId("sorting-phase"),
  sortingMessage: byId("sorting-message"), sortingFailure: byId("sorting-failure"),
  sortingCount: byId("sorting-count"), sortingProgress: byId("sorting-progress"),
  sortingElapsed: byId("sorting-elapsed"), sortingHold: byId("sorting-hold"),
  sortingQueue: byId("sorting-queue"), sortingChecks: byId("sorting-checks"),
  sortingObjects: byId("sorting-objects"),
  sortingLayout: byId("sorting-layout"),
};
let socket = null;
let state = null;
let hasFrame = false;
const objectRows = [];
const jointOutputs = [];
let lastSkill = null;
let lastPlanner = null;
let pendingPlannerAction = null;
let plannerHistoryKey = null;
let displayedPhaseOrder = "";
let manualContractKey = null;
let fatalRuntimeError = null;
let sceneMode = null;
let lastSorting = null;
let storageReportsLoaded = false;
let sortingReportLoaded = false;
let sortingQueueKey = null;
let sortingObjectsKey = null;
let sortProposalKey = null;
let sortHistoryKey = null;
const sortUi = Object.fromEntries([
  "planner-panel", "planner-status", "planner-task", "planner-hint", "planner-setup",
  "planner-error", "planner-model", "planner-calls", "planner-latency", "planner-total-latency",
  "planner-response", "planner-submitted", "planner-history", "planner-history-empty",
  "plan-generate", "plan-execute", "plan-cancel", "plan-preview", "plan-preview-title",
  "plan-summary", "plan-reason", "plan-mapping", "plan-queue",
].map((key) => [key, byId(`sort-${key}`)]));
const sortPlannerLabels = {
  unconfigured: "未配置订阅工具", idle: "待请求提案", planning: "正在请求 GPT",
  ready: "完整提案待批准", unsupported: "模型报告不支持", executing: "已批准 · 物理执行中",
  succeeded: "整批物理检查通过", failed: "请求或执行失败", cancelled: "请求 / 提案已取消",
};
const storageElements = document.querySelectorAll("[data-storage-only]");
const sortingElements = document.querySelectorAll("[data-sorting-only]");
const colorLabels = { red: "红色", blue: "蓝色", yellow: "黄色" };
const kindLabels = { bowl: "碗", cup: "杯" };
const sortingPhaseLabels = {
  idle: "待开始", prepare: "准备", approach: "接近", descend: "下降",
  close: "双侧接触夹持", lift: "抬升", hold: "稳定保持",
  transfer: "转移至目标色区", place: "放置", release: "释放与放稳",
  withdraw: "撤离物体", return_home: "机械臂归位", verify: "整批稳定保持验证",
  verify_sorting: "整批稳定保持验证", done: "完成", failed: "失败",
};
const sortingCheckLabels = {
  all_correct: "全部进入正确色区", all_supported: "全部受到桌面支撑",
  all_stable: "全部物体稳定", hand_empty: "夹爪为空", home_reached: "机械臂已归位",
};
const plannerLabels = {
  unconfigured: "未找到 Codex CLI", idle: "待请求决策", planning: "正在请求 GPT",
  ready: "单项决策已校验 · 待授权", unsupported: "任务不受支持", failed: "失败",
  executing: "单技能执行中", waiting: "已收到反馈 · 待请求下一项",
  stopping: "正在停止 · 等待当前技能结束", succeeded: "目标实测完成", cancelled: "决策循环已停止",
};
const phaseLabels = {
  idle: "待开始", prepare: "准备", approach: "接近", descend: "下降",
  close: "闭合夹爪", lift: "抬升", hold: "稳定保持", done: "完成", failed: "失败",
  transfer: "移至抽屉上方", place: "放入抽屉", release: "松爪与放稳", withdraw: "撤出夹爪",
  push_approach: "接近抽屉把手", push_drawer: "推动关柜", push_withdraw: "撤离把手",
  return_home: "机械臂归位", verify_store: "验证最终状态",
};
const skillLabels = { idle: "待开始", running: "执行中", succeeded: "执行成功", failed: "执行失败" };
const taskLabels = { grasp: "抓取与保持", store_bowl: "黑碗收纳、关柜与归位" };
const primitiveLabels = {
  grasp_and_hold: "抓取与保持", place_and_release: "放置与释放",
  withdraw_from_drawer: "从抽屉撤出", close_drawer: "关闭抽屉", return_home: "机械臂归位",
};
const goalLabels = {
  hold_target: "抓取目标并稳定保持", store_and_home: "收纳黑碗、关柜与归位",
  close_and_home: "关闭抽屉并归位", home: "机械臂归位",
};
const checkLabels = {
  bilateral_grip: "双侧真实夹持", clear_of_table: "目标离开桌面", target_lifted: "目标达到抬升高度",
  relative_pose_stable: "相对夹持位姿稳定", verified_hold: "稳定保持验证通过",
  bowl_inside_drawer: "黑碗完全位于抽屉内", bowl_supported: "黑碗受到抽屉底面支撑",
  bowl_settled: "黑碗已静稳", hand_empty: "夹爪为空",
  hand_clear_of_drawer: "手部已安全退出抽屉", support_preserved: "物体保持稳定支撑",
  drawer_closed: "抽屉已关闭", home_reached: "机械臂已归位",
};

function checksSummary(checks) {
  return Object.entries(checks || {}).map(([name, passed]) => `${checkLabels[name] || name}：${yesNo(passed)}`).join("；");
}

function primitiveLabel(name) {
  return state?.available_skills?.find((skill) => skill.name === name)?.label || primitiveLabels[name] || name || "未报告";
}

function yesNo(value) {
  return value === true ? "是" : value === false ? "否" : "—";
}
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

const comparisonMethods = { rule: "状态感知规则（主要对照）", gpt: "GPT（真实订阅）", fixed: "固定序列（参考）" };
const comparisonCases = ["table", "table_right", "held", "placed", "complete", "closed_empty", "disturbed"];
const expandedComparisonCases = ["table", "table_right", "table_forward", "held", "placed", "complete", "closed_empty", "disturbed", "hold_bowl", "hold_bottle", "close_only", "home_after_close"];
const comparisonGoalChecks = {
  hold_target: ["bilateral_grip", "clear_of_table", "target_lifted", "relative_pose_stable", "verified_hold"],
  store_and_home: ["bowl_inside_drawer", "bowl_supported", "bowl_settled", "drawer_closed", "home_reached", "hand_empty", "hand_clear_of_drawer", "support_preserved"],
  close_and_home: ["drawer_closed", "home_reached", "hand_empty", "hand_clear_of_drawer", "support_preserved"],
  home: ["home_reached", "hand_empty", "hand_clear_of_drawer", "support_preserved"],
};
const comparisonUi = Object.fromEntries(
  ["refresh", "batch", "download", "status", "error", "report", "meta", "summary", "runs", "analysis", "by-case", "paired"].map((key) => [key, byId(`comparison-${key}`)]),
);

function comparisonCaseMetadata(caseId) {
  return {
    goal: ["hold_bowl", "hold_bottle"].includes(caseId) ? "hold_target"
      : caseId === "close_only" ? "close_and_home" : caseId === "home_after_close" ? "home" : "store_and_home",
    target_object: caseId === "hold_bottle" ? "bottle" : "bowl",
    expected_outcome: ["closed_empty", "disturbed"].includes(caseId) ? "safe_stop" : "success",
  };
}

function comparisonMeasured(run) {
  return run.infrastructure_errors.length === 0 && !["setting_up", "running"].includes(run.status);
}

function comparisonDistribution(values) {
  const measured = values.filter((value) => typeof value === "number" && Number.isFinite(value));
  const n = measured.length;
  const mean = n ? measured.reduce((sum, value) => sum + value, 0) / n : null;
  return {
    n, mean,
    sd: n > 1 ? Math.sqrt(measured.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (n - 1)) : null,
    min: n ? Math.min(...measured) : null, max: n ? Math.max(...measured) : null,
  };
}

function comparisonEqual(actual, expected, field = "") {
  if (typeof expected === "number") {
    const statistic = ["mean", "sd", "min", "max"].includes(field) || field.startsWith("mean_");
    return typeof actual === "number" && Number.isFinite(actual)
      && (!statistic && Number.isInteger(expected) ? actual === expected : Math.abs(actual - expected) <= 1e-8 * Math.max(1, Math.abs(expected)));
  }
  if (expected === null || typeof expected !== "object") return actual === expected;
  if (!actual || typeof actual !== "object" || Array.isArray(actual) !== Array.isArray(expected)) return false;
  const keys = Object.keys(expected);
  return Object.keys(actual).length === keys.length && keys.every((key) => Object.hasOwn(actual, key) && comparisonEqual(actual[key], expected[key], key));
}

function comparisonCollisionCount(runs) {
  const measured = runs.filter((run) => typeof run.collisions?.episodes === "number"
    && Number.isFinite(run.collisions.episodes) && run.collisions.episodes >= 0);
  return measured.length ? measured.filter((run) => run.collisions.episodes > 0).length : null;
}

function comparisonAnalysis(report, invalid) {
  const byCase = [];
  const paired = [];
  for (const caseId of new Set(report.runs.map((run) => run.case_id))) {
    const runs = report.runs.filter((run) => run.case_id === caseId);
    const metadata = comparisonCaseMetadata(caseId);
    for (const method of report.protocol.methods) {
      const group = runs.filter((run) => run.method === method);
      const measured = group.filter(comparisonMeasured);
      const successes = measured.filter((run) => run.expected_outcome === "success" && run.success);
      const reasons = new Map();
      for (const run of group) {
        if (run.failure_reason && !(comparisonMeasured(run) && run.expected_outcome === "safe_stop" && run.safety_pass === true)) {
          reasons.set(run.failure_reason, (reasons.get(run.failure_reason) || 0) + 1);
        }
      }
      byCase.push({
        case_id: caseId, ...metadata, method,
        attempts: measured.length, successes: successes.length,
        safety_passes: measured.filter((run) => run.expected_outcome === "safe_stop" && run.safety_pass === true).length,
        infrastructure_failures: group.filter((run) => run.infrastructure_errors.length > 0).length,
        pending_runs: group.filter((run) => run.infrastructure_errors.length === 0 && !comparisonMeasured(run)).length,
        collision_runs: comparisonCollisionCount(group),
        wall_s: comparisonDistribution(measured.map((run) => run.end_to_end_wall_s)),
        success_wall_s: comparisonDistribution(successes.map((run) => run.end_to_end_wall_s)),
        request_latency_s: comparisonDistribution(group.filter((run) => run.method === "gpt").flatMap((run) => run.history.map((item) => item.latency_s))),
        failure_reasons: [...reasons.keys()].sort().map((reason) => ({ reason, count: reasons.get(reason) })),
      });
    }
    if (metadata.expected_outcome !== "success" || !["rule", "gpt"].every((method) => report.protocol.methods.includes(method))) continue;
    const item = {
      case_id: caseId, goal: metadata.goal, target_object: metadata.target_object,
      complete_pairs: 0, matched_success_pairs: 0, rule_only_successes: 0,
      gpt_only_successes: 0, both_failed: 0, excluded_pairs: 0,
    };
    const deltas = [];
    for (const repetition of new Set(runs.map((run) => run.repetition ?? 1))) {
      const pair = ["rule", "gpt"].map((method) => runs.find((run) => run.method === method && (run.repetition ?? 1) === repetition));
      if (!pair.every((run) => run && comparisonMeasured(run))) {
        item.excluded_pairs += 1;
        continue;
      }
      for (const key of ["initial_state_fingerprint", "initial_observation_fingerprint", "initial_model_fingerprint"]) {
        if (typeof pair[0][key] !== "string" || !pair[0][key] || pair[0][key] !== pair[1][key]) invalid();
      }
      item.complete_pairs += 1;
      const [rule, gpt] = pair;
      if (rule.success && gpt.success) {
        item.matched_success_pairs += 1;
        if (rule.end_to_end_wall_s !== null && gpt.end_to_end_wall_s !== null) deltas.push(gpt.end_to_end_wall_s - rule.end_to_end_wall_s);
      } else if (rule.success) item.rule_only_successes += 1;
      else if (gpt.success) item.gpt_only_successes += 1;
      else item.both_failed += 1;
    }
    paired.push({ ...item, delta_wall_s: comparisonDistribution(deltas) });
  }
  return { by_case: byCase, paired };
}

function validateComparison(report) {
  const count = (value) => Number.isSafeInteger(value) && value >= 0;
  const seconds = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0;
  const nullableSeconds = (value) => value === null || seconds(value);
  const validError = (value) => value === undefined || value === null || typeof value === "string"
    || (typeof value === "object" && !Array.isArray(value) && typeof value.message === "string"
      && ["type", "stage"].every((key) => value[key] === undefined || typeof value[key] === "string"));
  const selection = (values, allowed) => Array.isArray(values) && values.length > 0
    && new Set(values).size === values.length && values.every((value) => allowed.includes(value));
  const invalid = () => { throw new Error("闭环报告字段缺失、数值无效或汇总与原始记录不一致；不显示可能误导的成功率。请查看原始 JSON。"); };
  const expanded = report?.protocol?.suite === "expanded";
  if (!report || report.kind !== "paired_closed_loop_comparison"
      || !["running", "completed", "failed"].includes(report.status)
      || typeof report.generated_at !== "string" || !Number.isFinite(Date.parse(report.generated_at))
      || !selection(report.protocol?.methods, Object.keys(comparisonMethods))
      || !selection(report.protocol?.cases, expanded ? expandedComparisonCases : comparisonCases)
      || !Array.isArray(report.runs) || !Array.isArray(report.summary)
      || !validError(report.error)) invalid();
  const protocol = report.protocol;
  const repetitions = expanded ? protocol.repetitions : 1;
  const total = protocol.methods.length * protocol.cases.length * repetitions;
  if (!count(repetitions) || repetitions < 1 || !count(total) || protocol.planned_runs !== total
      || !seconds(protocol.validation_hold_s) || protocol.validation_hold_s === 0
      || (protocol.suite !== undefined && !["expanded", "original"].includes(protocol.suite))) invalid();
  const identity = (item) => `${item.case_id}:${item.repetition ?? 1}:${item.method}`;
  const specs = new Map();
  const schedule = new Set();
  if (expanded) {
    if (!count(protocol.seed) || !Array.isArray(protocol.case_specs)
        || protocol.case_specs.length !== protocol.cases.length
        || !Array.isArray(protocol.schedule) || protocol.schedule.length !== total
        || protocol.randomization?.xy_jitter_m !== 0.003
        || protocol.randomization?.yaw_range_rad !== 0.06
        || protocol.randomization?.repeats_share_initial_state !== true) invalid();
    for (const [index, spec] of protocol.case_specs.entries()) {
      if (!spec || spec.case_id !== protocol.cases[index] || specs.has(spec.case_id)
          || typeof spec.label !== "string" || !spec.label.trim()
          || typeof spec.task !== "string" || !spec.task.trim()
          || !Array.isArray(spec.preparation_skills) || !spec.preparation_skills.every((skill) => Object.hasOwn(primitiveLabels, skill))
          || !Object.entries(comparisonCaseMetadata(spec.case_id)).every(([key, value]) => spec[key] === value)) invalid();
      const scene = spec.scene;
      const offset = spec.case_id === "table_right" ? [0.04, 0] : spec.case_id === "table_forward" ? [0, 0.04] : [0, 0];
      if (!scene || typeof scene.case_id !== "string" || !scene.case_id
          || typeof scene.position_label !== "string" || !scene.position_label
          || scene.target_object !== spec.target_object || scene.seed !== protocol.seed
          || !Array.isArray(scene.offset_xy_m) || scene.offset_xy_m.length !== 2
          || !scene.offset_xy_m.every((value, axis) => Number.isFinite(value) && Math.abs(value - offset[axis]) <= protocol.randomization.xy_jitter_m + 1e-12)
          || !Number.isFinite(scene.yaw_rad) || Math.abs(scene.yaw_rad) > protocol.randomization.yaw_range_rad) invalid();
      specs.set(spec.case_id, spec);
    }
    for (const item of protocol.schedule) {
      if (!item || !protocol.methods.includes(item.method) || !specs.has(item.case_id)
          || !count(item.repetition) || item.repetition < 1 || item.repetition > repetitions
          || schedule.has(identity(item))) invalid();
      schedule.add(identity(item));
    }
    const goals = new Set(protocol.case_specs.map((spec) => spec.goal));
    if (protocol.goal !== (goals.size === 1 ? [...goals][0] : "mixed")) invalid();
  } else if (protocol.goal !== "store_and_home" || (protocol.repetitions !== undefined && protocol.repetitions !== 1)) invalid();
  const fingerprints = new Map();
  const pairs = new Set();
  for (const run of report.runs) {
    if (!run || !report.protocol.methods.includes(run.method) || !report.protocol.cases.includes(run.case_id)
        || typeof run.label !== "string" || !run.label.trim()
        || typeof run.status !== "string" || !run.status.trim()
        || !["success", "safe_stop"].includes(run.expected_outcome)
        || run.expected_outcome !== comparisonCaseMetadata(run.case_id).expected_outcome
        || typeof run.success !== "boolean"
        || (run.expected_outcome === "success" ? run.safety_pass !== null : typeof run.safety_pass !== "boolean" && run.safety_pass !== null)
        || (run.expected_outcome === "safe_stop" && run.success)
        || (run.success && run.status !== "succeeded")
        || !["executed_skill_count", "decision_count", "gpt_request_count"].every((key) => count(run[key]))
        || !["execution_sim_s", "gpt_latency_s", "validation_sim_s"].every((key) => seconds(run[key]))
        || !["evaluation_sim_s", "end_to_end_wall_s"].every((key) => nullableSeconds(run[key]))
        || (run.failure_reason !== null && typeof run.failure_reason !== "string")
        || (run.method !== "gpt" && (run.gpt_request_count !== 0 || run.gpt_latency_s !== 0))) invalid();
    if (!Array.isArray(run.infrastructure_errors) || !run.infrastructure_errors.every((error) => error && validError(error))
        || !Array.isArray(run.history) || !run.history.every((item) => item && nullableSeconds(item.latency_s))
        || (run.method === "gpt" && run.gpt_request_count !== run.history.length)) invalid();
    const metadata = comparisonCaseMetadata(run.case_id);
    const repetition = run.repetition ?? 1;
    if (!count(repetition) || repetition < 1 || repetition > repetitions
        || Object.entries(metadata).some(([key, value]) => run[key] !== undefined && run[key] !== value)) invalid();
    if (expanded) {
      const spec = specs.get(run.case_id);
      if (run.repetition === undefined || run.goal !== spec.goal || run.target_object !== spec.target_object
          || run.label !== spec.label || !comparisonEqual(run.scene_case, spec.scene) || !schedule.has(identity(run))) invalid();
      if (run.infrastructure_errors.length === 0) {
        for (const key of ["initial_state_fingerprint", "initial_observation_fingerprint", "initial_model_fingerprint"]) {
          if (run[key] === null || run[key] === undefined) {
            if (comparisonMeasured(run)) invalid();
            continue;
          }
          if (typeof run[key] !== "string" || !run[key]) invalid();
          const fingerprintKey = `${run.case_id}:${key}`;
          if (fingerprints.has(fingerprintKey) && fingerprints.get(fingerprintKey) !== run[key]) invalid();
          fingerprints.set(fingerprintKey, run[key]);
        }
      }
    }
    if ((run.success || run.safety_pass === true)
        && (!comparisonMeasured(run) || run.stop_kind === "provider_failure"
          || run.pairing_failed === true || run.model_invariant !== true
          || !run.initial_model_fingerprint || run.final_model_fingerprint !== run.initial_model_fingerprint
          || run.collisions?.episodes !== 0 || !Array.isArray(run.final_state?.warnings)
          || run.final_state.warnings.length || run.final_state.target_object !== metadata.target_object)) invalid();
    if (run.success && (run.validation?.status !== "succeeded"
        || run.validation_sim_s + 1e-6 < protocol.validation_hold_s
        || !comparisonGoalChecks[metadata.goal].every((key) => run.goal_checks?.[metadata.goal]?.[key] === true
          && run.final_state?.goal_checks?.[metadata.goal]?.[key] === true
          && run.validation?.before?.goal_checks?.[metadata.goal]?.[key] === true
          && run.validation?.after?.goal_checks?.[metadata.goal]?.[key] === true))) invalid();
    const pair = identity(run);
    if (pairs.has(pair)) invalid();
    pairs.add(pair);
  }
  if (report.status === "completed" && (pairs.size !== total
      || report.runs.some((run) => ["setting_up", "running"].includes(run.status)))) invalid();
  if (report.summary.length !== report.protocol.methods.length
      || new Set(report.summary.map((item) => item?.method)).size !== report.summary.length) invalid();
  for (const summary of report.summary) {
    if (!summary || !report.protocol.methods.includes(summary.method)) invalid();
    const runs = report.runs.filter((run) => run.method === summary.method);
    const measured = runs.filter(comparisonMeasured);
    const goals = measured.filter((run) => run.expected_outcome === "success");
    const safety = measured.filter((run) => run.expected_outcome === "safe_stop");
    const expected = {
      goal_attempts: goals.length, goal_successes: goals.filter((run) => run.success).length,
      safety_attempts: safety.length, safety_passes: safety.filter((run) => run.safety_pass === true).length,
      executed_skill_count: runs.reduce((sum, run) => sum + run.executed_skill_count, 0),
      gpt_request_count: runs.reduce((sum, run) => sum + run.gpt_request_count, 0),
      infrastructure_failures: runs.filter((run) => run.infrastructure_errors.length > 0).length,
      pending_runs: runs.filter((run) => run.infrastructure_errors.length === 0 && ["setting_up", "running"].includes(run.status)).length,
      timed_goal_attempts: goals.filter((run) => run.end_to_end_wall_s !== null).length,
      timed_successes: goals.filter((run) => run.success && run.end_to_end_wall_s !== null).length,
      timed_gpt_requests: runs.filter((run) => run.method === "gpt").reduce((sum, run) => sum + run.history.filter((item) => item.latency_s !== null).length, 0),
    };
    if (!Object.entries(expected).every(([key, value]) => count(summary[key]) && summary[key] === value)
        || (goals.length === 0 ? summary.goal_success_rate !== null
          : !seconds(summary.goal_success_rate) || Math.abs(summary.goal_success_rate - expected.goal_successes / goals.length) > 0.000001)
        || !nullableSeconds(summary.mean_success_end_to_end_wall_s)
        || !nullableSeconds(summary.mean_end_to_end_wall_s)
        || !nullableSeconds(summary.mean_gpt_latency_s)
        || (summary.collision_runs !== null && (!count(summary.collision_runs) || summary.collision_runs > runs.length))
        || (expected.goal_successes === 0 && summary.mean_success_end_to_end_wall_s !== null)
        || (expected.goal_attempts === 0 && summary.mean_end_to_end_wall_s !== null)
        || (expected.gpt_request_count === 0 && summary.mean_gpt_latency_s !== null)) invalid();
    const means = {
      mean_end_to_end_wall_s: comparisonDistribution(goals.map((run) => run.end_to_end_wall_s)).mean,
      mean_success_end_to_end_wall_s: comparisonDistribution(goals.filter((run) => run.success).map((run) => run.end_to_end_wall_s)).mean,
      mean_gpt_latency_s: comparisonDistribution(runs.filter((run) => run.method === "gpt").flatMap((run) => run.history.map((item) => item.latency_s))).mean,
      collision_runs: comparisonCollisionCount(runs),
    };
    if (!Object.entries(means).every(([key, value]) => comparisonEqual(summary[key], value, key))) invalid();
  }
  if (expanded || report.analysis !== undefined) {
    if (!comparisonEqual(report.analysis, comparisonAnalysis(report, invalid))) invalid();
  }
}

function comparisonRow(texts) {
  const row = document.createElement("tr");
  for (const text of texts) {
    const cell = document.createElement("td");
    cell.textContent = text;
    row.append(cell);
  }
  return row;
}

function renderComparisonAnalysis(report) {
  const repetitions = report.protocol.repetitions ?? 1;
  const distribution = (item) => item.n === 0 ? "n=0 · 未测量"
    : `${item.n} / ${number(item.mean, 3)} / ${item.sd === null ? "不适用（n<2）" : number(item.sd, 3)} / ${number(item.min, 3)} / ${number(item.max, 3)}`;
  const caseLabel = (caseId) => {
    const metadata = comparisonCaseMetadata(caseId);
    const label = report.protocol.case_specs?.find((spec) => spec.case_id === caseId)?.label || caseId;
    return `${label} (${caseId})\n${goalLabels[metadata.goal]} (${metadata.goal})\n物体：${metadata.target_object}`;
  };
  comparisonUi["by-case"].replaceChildren(...report.protocol.cases.flatMap((caseId) => report.protocol.methods.map((method) => {
    const item = report.analysis.by_case.find((entry) => entry.case_id === caseId && entry.method === method);
    if (!item) return comparisonRow([`${caseLabel(caseId)}\n${comparisonMethods[method]}`, `尚未记录：${repetitions}；无有效已结束试次`, "未测量", "未测量", "未测量", "未测量"]);
    const missing = repetitions - item.attempts - item.infrastructure_failures - item.pending_runs;
    return comparisonRow([
      `${caseLabel(caseId)}\n${comparisonMethods[method]}`,
      `${item.expected_outcome === "safe_stop" ? `安全通过 ${item.safety_passes}` : `任务完成 ${item.successes}`} / ${item.attempts}\n基础设施 ${item.infrastructure_failures} / 进行中 ${item.pending_runs} / 尚未记录 ${missing}`,
      distribution(item.wall_s), distribution(item.success_wall_s), distribution(item.request_latency_s),
      `碰撞试次：${item.collision_runs === null ? "未测量" : item.collision_runs}\n${item.failure_reasons.map((entry) => `${entry.count} × ${entry.reason}`).join("\n") || "无记录的非预期失败原因"}`,
    ]);
  })));
  const pairedRows = ["rule", "gpt"].every((method) => report.protocol.methods.includes(method))
    ? report.protocol.cases.filter((caseId) => comparisonCaseMetadata(caseId).expected_outcome === "success").map((caseId) => {
      const item = report.analysis.paired.find((entry) => entry.case_id === caseId);
      if (!item) return comparisonRow([caseLabel(caseId), "尚无完整配对", "未测量", `0 / ${repetitions}`, "n=0 · 未测量"]);
      return comparisonRow([
        caseLabel(caseId), `${item.complete_pairs} / ${item.matched_success_pairs}`,
        `${item.rule_only_successes} / ${item.gpt_only_successes} / ${item.both_failed}`,
        `${item.excluded_pairs} / ${repetitions - item.complete_pairs - item.excluded_pairs}`,
        distribution(item.delta_wall_s),
      ]);
    }) : [comparisonRow(["未同时选择规则与 GPT，不计算配对", "不适用", "不适用", "不适用", "未测量"])];
  comparisonUi.paired.replaceChildren(...pairedRows);
  comparisonUi.analysis.hidden = false;
}

async function loadPlanningComparison() {
  const batch = comparisonUi.batch.value;
  const label = comparisonUi.batch.selectedOptions[0].textContent;
  const reportUrl = `/api/planning-comparison?batch=${encodeURIComponent(batch)}`;
  comparisonUi.refresh.disabled = true;
  comparisonUi.batch.disabled = true;
  comparisonUi.download.href = reportUrl;
  comparisonUi.status.textContent = `正在读取「${label}」；不会启动运行…`;
  comparisonUi.error.hidden = true;
  comparisonUi.report.hidden = true;
  comparisonUi.summary.replaceChildren();
  comparisonUi.runs.replaceChildren();
  comparisonUi.analysis.hidden = true;
  comparisonUi["by-case"].replaceChildren();
  comparisonUi.paired.replaceChildren();
  const time = (value) => value === null ? "未测量" : number(value, 3);
  try {
    const response = await fetch(reportUrl, { cache: "no-store" });
    if (response.status === 404) throw new Error("当前所选批次尚无报告（HTTP 404）；不会显示其他批次替代。");
    if (!response.ok) throw new Error(`闭环对照报告读取失败（HTTP ${response.status}）；未使用替代数据。`);
    const report = await response.json();
    validateComparison(report);
    const total = report.protocol.planned_runs;
    const repetitions = report.protocol.repetitions ?? 1;
    const mixed = report.protocol.goal === "mixed";
    const pending = report.summary.reduce((sum, item) => sum + item.pending_runs, 0);
    const infrastructure = report.summary.reduce((sum, item) => sum + item.infrastructure_failures, 0);
    comparisonUi.status.textContent = `「${label}」${{ running: "未完成 · 运行中", failed: "未完成 / 执行失败 · 保留原始记录", completed: "所选矩阵已完成（不代表全部成功）" }[report.status]}：已记录 ${report.runs.length} / ${total} 次；进行中 ${pending}；基础设施无效 ${infrastructure}；尚未记录 ${total - report.runs.length}。只统计当前批次，不与历史批次合并。${report.runs.length === 0 ? "暂无运行记录，不能计算成功率。" : ""}`;
    const randomization = report.protocol.randomization;
    comparisonUi.meta.textContent = `生成时间：${report.generated_at}。案例：${report.protocol.cases.join("、")}。目标：${mixed ? "多任务（逐场景分开统计）" : report.protocol.goal}。每场景/方法 ${repetitions} 次。${randomization ? `种子 ${report.protocol.seed}；XY 抖动 ±${number(randomization.xy_jitter_m * 1000, 1)} mm；偏航 ±${number(randomization.yaw_range_rad, 3)} rad；同场景各方法与重复共享同一初态，非独立场景抽样。` : "历史单次配对，不能估计重复变异。"}安全例不计入正例完成率，各批次独立统计。`;
    comparisonUi.summary.replaceChildren(...["rule", "gpt", "fixed"]
      .filter((method) => report.protocol.methods.includes(method))
      .map((method) => {
        const item = report.summary.find((summary) => summary.method === method);
        const safetyUnknown = report.runs.filter((run) => run.method === method && run.expected_outcome === "safe_stop" && run.infrastructure_errors.length === 0 && !["setting_up", "running"].includes(run.status) && run.safety_pass === null).length;
        return comparisonRow([
          `${comparisonMethods[method]}\n基础设施无效：${item.infrastructure_failures}；进行中：${item.pending_runs}；尚未记录：${report.protocol.cases.length * repetitions - report.runs.filter((run) => run.method === method).length}（均不计分母）`,
          item.goal_attempts ? `${item.goal_successes} / ${item.goal_attempts}（${number(item.goal_success_rate * 100, 1)}%）` : "无有效已结束正例 · 不计算",
          item.safety_attempts ? `${item.safety_passes} / ${item.safety_attempts}${safetyUnknown ? `\n其中 ${safetyUnknown} 例未测量` : ""}` : "无有效已结束安全例 · 不计算",
          `${item.executed_skill_count} / ${item.gpt_request_count}`,
          mixed ? "混合任务不合并均值；见逐场景表" : `${time(item.mean_end_to_end_wall_s)}（已计时 ${item.timed_goal_attempts}/${item.goal_attempts}）\n${time(item.mean_success_end_to_end_wall_s)}（已计时 ${item.timed_successes}/${item.goal_successes}）`,
          mixed ? "混合任务不合并均值；见逐场景表" : `${time(item.mean_gpt_latency_s)}（已计时 ${item.timed_gpt_requests}/${item.gpt_request_count}）`,
        ]);
      }));
    if (report.analysis) renderComparisonAnalysis(report);
    comparisonUi.runs.replaceChildren(...report.runs.map((run) => comparisonRow([
      `${run.label} (${run.case_id}) · 重复 ${run.repetition ?? 1}/${repetitions}\n目标：${run.goal ?? "store_and_home"} · 物体：${run.target_object ?? "bowl"}\n${run.scene_case ? `位姿：${run.scene_case.position_label} · XY [${run.scene_case.offset_xy_m.map((value) => number(value, 4)).join(", ")}] m · yaw ${number(run.scene_case.yaw_rad, 4)} rad · seed ${run.scene_case.seed}\n` : ""}${comparisonMethods[run.method]}\n预期：${run.expected_outcome === "success" ? "完成任务" : "安全停止（不算任务成功）"}`,
      `${run.status}${run.infrastructure_errors.length ? "\n基础设施失败，不计完成率" : ["setting_up", "running"].includes(run.status) ? "\n尚未结束，不计完成率" : ""}\n任务完成：${run.success ? "是（物理验证）" : "否"}\n安全验收：${run.expected_outcome === "success" ? "不适用（正例）" : run.safety_pass === null ? "未测量" : run.safety_pass ? "通过" : "未通过"}`,
      `${run.executed_skill_count} / ${run.decision_count} / ${run.gpt_request_count}`,
      `${time(run.execution_sim_s)} / ${time(run.evaluation_sim_s)} / ${time(run.validation_sim_s)}`,
      `${time(run.end_to_end_wall_s)} / ${run.gpt_request_count ? time(run.gpt_latency_s) : "无 GPT 请求"}`,
      run.failure_reason || "未报告失败原因",
    ])));
    comparisonUi.report.hidden = false;
    const providerFailures = report.runs.filter((run) => run.method === "gpt" && run.stop_kind === "provider_failure");
    if (report.error) {
      const errorText = typeof report.error === "string" ? report.error
        : [report.error.type, report.error.stage, report.error.message].filter(Boolean).join(" · ");
      comparisonUi.error.textContent = `报告错误：${errorText}`;
      comparisonUi.error.hidden = false;
    } else if (providerFailures.length) {
      const reasons = [...new Set(providerFailures.map((run) => run.failure_reason).filter(Boolean))].join("；");
      comparisonUi.error.textContent = `本存储批次中 ${providerFailures.length} 次 GPT 运行出现提供方失败：${reasons} 这是该批次归档证据，不表示当前账户状态。本表完成率包含当时服务可用性影响，不是模型推理能力排名；安全例未验收不能当作模型不安全的证据。`;
      comparisonUi.error.hidden = false;
    }
  } catch (error) {
    comparisonUi.status.textContent = "闭环对照数据不可用；实时仿真不受影响。";
    comparisonUi.error.textContent = error.message;
    comparisonUi.error.hidden = false;
  } finally {
    comparisonUi.refresh.disabled = false;
    comparisonUi.batch.disabled = false;
  }
}
const evaluationUi = Object.fromEntries(
  ["refresh", "status", "error", "report", "meta", "summary", "runs", "paired"].map((key) => [key, byId(`evaluation-${key}`)]),
);
const evaluationMethods = { fixed_rule: "固定规则", gpt: "GPT（真实订阅）" };
const evaluationOutcomes = {
  pending: "尚无完整配对", succeeded: "物理验证成功", failed: "任务失败",
  infrastructure_error: "基础设施错误 · 不计任务失败",
};

async function loadSortingEvaluation() {
  evaluationUi.refresh.disabled = true;
  evaluationUi.status.textContent = "正在读取冻结评估存档；不会启动试验…";
  evaluationUi.error.hidden = true;
  evaluationUi.report.hidden = true;
  evaluationUi.summary.replaceChildren();
  evaluationUi.runs.replaceChildren();
  evaluationUi.paired.replaceChildren();
  const measured = (value) => Number.isFinite(value) ? number(value, 3) : "未测量";
  try {
    const response = await fetch("/api/sorting-evaluation", { cache: "no-store" });
    if (!response.ok) throw new Error(await response.text() || `HTTP ${response.status}`);
    // The read-only server helper validates identities, evidence and accounting.
    const report = await response.json();
    const total = (key) => report.summary.reduce((sum, item) => sum + item[key], 0);
    const statuses = {
      running: "运行中（存档状态，非实时进度）",
      completed: "预注册矩阵已完成（不代表全部成功）",
      aborted: "已中止 · 未完成，保留部分证据",
    };
    evaluationUi.status.textContent = `${statuses[report.status]}：已尝试 ${total("attempted")} / ${report.expected_runs}；成功 ${total("successes")}；任务失败 ${total("failures")}；待记录 ${total("pending")}；基础设施错误 ${total("infrastructure_failures")}。${report.passed ? "整批验收通过。" : "整批尚未通过。"}未保存结果的项可能尚未运行或正在执行，不能当作失败或成功。`;
    const scope = typeof report.scope === "string" ? report.scope : JSON.stringify(report.scope);
    evaluationUi.meta.textContent = `冻结提交：${report.frozen_commit}。请求模型标识：${report.requested_model}（非服务快照证明）。协议 SHA-256：${report.protocol_sha256}。范围：${scope}`;
    evaluationUi.summary.replaceChildren(...report.summary.map((item) => comparisonRow([
      evaluationMethods[item.method] || item.method,
      `${item.scheduled} / ${item.attempted}`,
      `${item.successes} / ${item.failures}`,
      `${item.pending} / ${item.infrastructure_failures}`,
    ])));
    const runs = new Map(report.runs.map((run) => [run.run_id, run]));
    evaluationUi.runs.replaceChildren(...report.schedule.map((entry) => {
      const run = runs.get(entry.run_id);
      const reason = run?.errors.map((error) => typeof error === "string" ? error
        : [error.type, error.stage, error.message].filter(Boolean).join(" · ")).join("\n");
      const row = comparisonRow([
        `${entry.layout_id} / ${entry.wording_id}\n${evaluationMethods[entry.method] || entry.method}\n${entry.run_id}`,
        run ? `${evaluationOutcomes[run.status] || run.status}\n${Number.isInteger(run.sorted_count) ? `${run.sorted_count} / 6` : "完成数未测量"}` : "待运行 / 尚无已保存结果",
        `${measured(run?.execution_sim_s)} / ${measured(run?.independent_final_hold_s)}`,
        `${measured(run?.execution_wall_s)} / ${measured(run?.gpt_latency_s)} / ${measured(run?.wall_s)}`,
        run ? [run.failure_stage, reason].filter(Boolean).join("\n") || "无记录的失败原因" : "尚未记录",
      ]);
      const evidence = document.createElement("td");
      if (run?.record_path) {
        const link = document.createElement("a");
        link.href = `/api/sorting-evaluation/runs/${encodeURIComponent(run.run_id)}`;
        link.download = `${run.run_id}.json.gz`;
        link.textContent = "下载完整 JSON.gz";
        evidence.append(link);
      } else {
        evidence.textContent = run ? "记录保存失败 · 无可下载文件" : "尚无记录";
      }
      row.append(evidence);
      return row;
    }));
    evaluationUi.paired.replaceChildren(...report.paired.map((pair) => comparisonRow([
      `${pair.layout_id} / ${pair.wording_id}`,
      evaluationOutcomes[pair.status] || pair.status,
      measured(pair.execution_sim_delta_s), measured(pair.execution_wall_delta_s),
    ])));
    if (report.errors.length) {
      evaluationUi.error.textContent = `批次错误：${report.errors.join("；")}`;
      evaluationUi.error.hidden = false;
    }
    evaluationUi.report.hidden = false;
  } catch (error) {
    evaluationUi.status.textContent = "冻结评估存档不可用；不显示旧报告，实时仿真不受影响。";
    evaluationUi.error.textContent = error.message;
    evaluationUi.error.hidden = false;
  } finally {
    evaluationUi.refresh.disabled = false;
  }
}


function plannerLocksScene() {
  return pendingPlannerAction !== null || state?.planner?.locks_scene === true;
}

function renderPlannerHistory(planner) {
  const history = planner?.history || [];
  const key = JSON.stringify(history.map((record) => [
    record.decision_id, record.status, record.error, record.latency_s,
    record.execution_sim_s, record.execution_wall_s, record.result_observation?.time_s,
  ]));
  if (key === plannerHistoryKey) return;
  plannerHistoryKey = key;
  ui.plannerHistoryEmpty.hidden = history.length > 0;
  ui.plannerHistory.replaceChildren(...history.map((record) => {
    const item = document.createElement("li");
    item.dataset.status = record.status;
    const title = document.createElement("strong");
    const decision = record.decision;
    const action = decision?.decision === "execute" ? primitiveLabel(decision.skill)
      : decision?.decision === "complete" ? "提出目标完成" : decision?.decision === "unsupported" ? "不支持" : "等待模型返回";
    const resultLabel = record.status === "succeeded" ? "当前技能实测成功"
      : record.status === "complete" ? "本地目标完成检查通过" : plannerLabels[record.status] || record.status;
    title.textContent = `决策 #${record.decision_id} · ${action} · ${resultLabel}`;
    const timing = document.createElement("p");
    timing.textContent = `观察时刻 ${number(record.observation?.time_s, 3)} 仿真秒 → 反馈时刻 ${number(record.result_observation?.time_s, 3)} 仿真秒\nGPT ${number(record.latency_s, 2)} 墙钟秒 · 执行 ${number(record.execution_sim_s, 3)} 仿真秒 / ${number(record.execution_wall_s, 2)} 墙钟秒`;
    const outcome = document.createElement("p");
    const observed = record.result_observation?.skill;
    outcome.textContent = record.error ? `失败：${record.error}`
      : observed ? `实测反馈：${skillLabels[observed.status] || observed.status} · ${observed.message || ""}${observed.failure_reason ? ` · ${observed.failure_reason}` : ""}`
      : "尚无执行反馈；模型回复不是物理成功证据。";
    const goal = document.createElement("p");
    goal.textContent = checksSummary(record.result_observation?.goal_checks?.[decision?.goal]);
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "本次决策与真实请求 / 结果观测";
    const payload = document.createElement("pre");
    payload.textContent = JSON.stringify({
      decision, model: record.model, thread_id: record.thread_id,
      invocation_id: record.invocation_id, observation: record.observation,
      result_observation: record.result_observation,
    }, null, 2);
    details.append(summary, payload);
    item.append(title, timing, outcome, goal, details);
    return item;
  }));
}

function renderPlanner(ready) {
  const planner = ready ? state.planner : lastPlanner;
  const known = planner && Object.hasOwn(plannerLabels, planner.status);
  const canStart = ready && ["idle", "succeeded"].includes(state.skill?.status);
  const locked = known && planner.locks_scene;
  const pending = pendingPlannerAction !== null;
  const task = ui.plannerTask.value.trim();
  const decision = known ? planner.decision : null;
  const canExecute = ready && known && planner.status === "ready"
    && ["execute", "complete"].includes(decision?.decision)
    && Number.isInteger(planner.decision_id) && !pending;
  ui.plannerTask.disabled = !canStart || !known || locked || pending;
  ui.planGenerate.disabled = ui.plannerTask.disabled || !task || task.length > 1000;
  ui.planExecute.disabled = !canExecute;
  ui.planExecute.textContent = decision?.decision === "complete" ? "确认本地完成检查" : "仅执行当前一项";
  ui.planContinuous.disabled = !canExecute || decision?.decision !== "execute";
  ui.planNext.disabled = !ready || !known || planner.status !== "waiting" || pending;
  ui.planCancel.disabled = !ready || !known || !locked || planner.status === "stopping" || pending;
  ui.plannerPanel.dataset.status = !ready ? "stale" : known ? planner.status : "waiting";
  ui.plannerStatus.textContent = known
    ? `${plannerLabels[planner.status]}${!ready ? " · 最后状态" : state.paused ? " · 物理已暂停" : ""}`
    : "等待决策状态";
  ui.plannerHint.textContent = !ready
    ? "连接或状态不可用；仅显示最后收到的记录，不会自动重连、请求模型或执行。"
    : !known ? "等待后端决策功能状态。"
    : pending ? "操作已发送，等待后端确认；不会重复提交。"
    : planner.status === "planning" ? "GPT 正读取本次实时观测。未暂停时物理与夹持继续运行；第一次生成决策绝不自动执行。"
    : planner.status === "ready" ? "检查本次目标、技能与参数。仅执行当前一项后会等待人工请求；持续授权才会依据实际反馈继续选择技能。"
    : planner.status === "waiting" ? "当前技能已产生实际反馈。可从最新物理状态请求下一项；不会自动请求或重试。"
    : planner.status === "stopping" ? "已停止后续决策；当前技能仍会正常完成，期间手动动作保持锁定。可暂停物理或重置。"
    : planner.status === "unsupported" ? "GPT 判定任务不受支持；未使用替代动作。"
    : planner.status === "executing" ? "仅运行本次选定技能；成功后记录真实结果，再按授权决定是否请求下一项。"
    : planner.status === "failed" ? "决策或物理执行失败；没有自动重试或替代动作。请查看错误与历史。"
    : planner.status === "succeeded" ? "目标已通过本地实测检查；物理仍持续验证，后续失稳会报告失败。"
    : !canStart ? "当前物理执行不能开始新决策；等待成功或检查失败原因。"
    : "使用当前 ChatGPT 订阅，无需 API 密钥。从当前场景生成第一项决策，不执行、不重置。";
  if (ready && state.paused) ui.plannerHint.textContent += " 当前物理时钟已停止；授权不会自动恢复，请点击「开始运行」。GPT 墙钟延迟仍真实计时。";
  ui.plannerSetup.hidden = !known || planner.configured;
  ui.plannerError.textContent = known ? planner.error || "" : "";
  ui.plannerError.hidden = !ui.plannerError.textContent;
  ui.plannerModelLabel.textContent = "Codex 请求模型 · 当前 ChatGPT 订阅";
  ui.plannerModel.textContent = known ? planner.model : "—";
  ui.plannerLatency.textContent = number(known ? planner.latency_s : undefined, 2);
  ui.plannerTotalLatency.textContent = number(known ? planner.total_latency_s : undefined, 2);
  ui.plannerMode.textContent = known && planner.continuous
    ? `持续授权 · 同一目标最多 ${planner.max_skills} 次技能调用 / 9 次模型请求`
    : "逐项授权 · 当前一项结束后等待人工请求";
  ui.plannerResponse.textContent = known && planner.thread_id
    ? `最近 Codex 会话 ID：${planner.thread_id}${planner.diagnostic_count ? ` · 非致命诊断 ${planner.diagnostic_count} 条` : ""}` : "";
  ui.plannerResponse.hidden = !ui.plannerResponse.textContent;
  ui.plannerSubmitted.hidden = !known || !planner.task;
  ui.plannerSubmittedTask.textContent = known ? planner.task : "";
  ui.planPreview.hidden = !decision;
  ui.planSummary.textContent = decision?.summary || "";
  ui.planReason.textContent = decision ? `决策依据：${decision.reason}` : "";
  ui.planParameters.hidden = !decision?.parameters;
  ui.planParameters.textContent = decision?.parameters ? `已校验参数：${parameterSummary(decision.parameters)}` : "";
  const goal = planner?.goal || decision?.goal;
  ui.planGoal.textContent = goal ? `${planner.goal ? "已授权目标" : "待批准目标"}：${goalLabels[goal] || goal} · 目标物体 ${decision?.target_object || "—"}` : "未提出可支持目标";
  ui.planSkill.textContent = decision?.decision === "execute" ? `唯一选定技能：${primitiveLabel(decision.skill)}`
    : decision?.decision === "complete" ? "模型提出完成；必须由本地物理判据验证，不再调用技能。" : "无可执行技能";
  ui.planPreviewTitle.textContent = decision?.decision === "unsupported" ? "GPT 不支持此任务"
    : `单项决策 #${planner?.decision_id} · ${planner?.status === "ready" ? "尚未执行" : "结果见实际反馈"}`;
  ui.plannerGoalChecks.textContent = !ready ? "连接不可用；不能确认当前目标判据。"
    : goal ? checksSummary(state.goal_checks?.[goal]) || "后端尚未报告此目标判据"
    : "生成第一项决策后显示对应目标的实时物理判据。";
  renderPlannerHistory(planner);
}

function sendPlanner(command) {
  if (send(command)) {
    pendingPlannerAction = command.action;
    updateControls();
  }
}

function renderManualSkills(ready) {
  const skills = ready && Array.isArray(state.available_skills) ? state.available_skills : [];
  const names = skills.map((skill) => skill.name);
  const options = Array.from(ui.manualSkill.options);
  if (JSON.stringify(names) !== JSON.stringify(options.map((option) => option.value))) {
    const previous = ui.manualSkill.value;
    ui.manualSkill.replaceChildren(...skills.map((skill) => {
      const option = document.createElement("option");
      option.value = skill.name;
      return option;
    }));
    if (names.includes(previous)) ui.manualSkill.value = previous;
  }
  skills.forEach((skill, index) => {
    const option = ui.manualSkill.options[index];
    option.textContent = `${skill.label || primitiveLabel(skill.name)} · ${skill.available === true ? "物理条件就绪" : "不可用"}`;
  });
  ui.manualSkill.disabled = !ready || skills.length === 0;
  const selected = skills.find((skill) => skill.name === ui.manualSkill.value);
  const needsParameters = selected?.name === "grasp_and_hold";
  const hasLimits = !!parameterLimits(state?.skill);
  const locked = plannerLocksScene();
  ui.startSkill.disabled = !ready || !selected || selected.available !== true || locked || (needsParameters && !hasLimits);
  ui.startSkill.textContent = needsParameters ? "用下方参数运行技能" : "运行所选技能";
  ui.manualReadiness.dataset.ready = String(!ui.startSkill.disabled);
  ui.manualReadiness.textContent = !ready ? "未连接或状态不可用；不能确认技能当前是否可运行。"
    : !selected ? "等待后端技能列表；不推断可用性。"
    : locked ? "GPT 决策循环拥有场景；先停止后续决策并等待当前技能结束，不能插入手动动作。"
    : selected.available !== true ? `不可运行：${selected.unavailable_reason || "后端未确认前置条件满足。"}`
    : needsParameters && !hasLimits ? "物理条件就绪，但后端尚未报告抓取参数范围。"
    : "后端已确认当前物理条件可用；启动时仍会重新检查。仅执行本技能，不自动选择下一项。";
  ui.manualParameters.textContent = needsParameters
    ? "此按钮使用下方三个输入框的当前抓取参数，不使用 GPT 参数。成功后持续物理夹持，不自动放置。"
    : "此技能不发送抓取参数；下方输入框不会影响本次调用。完成后不会自动启动下一技能。";
  ui.manualContract.hidden = !selected;
  const contractKey = JSON.stringify(selected && [selected.name, selected.phases, selected.watchdog_s,
    selected.target_objects, selected.preconditions, selected.postconditions]);
  if (contractKey !== manualContractKey) {
    manualContractKey = contractKey;
    ui.manualWatchdog.textContent = selected ? `超时上限 ${number(selected.watchdog_s, 1)} 仿真秒` : "";
    ui.manualTargets.textContent = selected ? `支持目标：${(selected.target_objects || []).join("、") || "后端未报告"}` : "";
    ui.manualPhases.textContent = selected ? `固定阶段：${(selected.phases || []).map((phase) => phaseLabels[phase] || phase).join(" → ")}` : "";
    for (const [element, conditions] of [[ui.manualPreconditions, selected?.preconditions], [ui.manualPostconditions, selected?.postconditions]]) {
      element.replaceChildren(...(conditions || []).map((condition) => {
        const item = document.createElement("li");
        item.textContent = condition;
        return item;
      }));
    }
  }
}


function renderSkill(ready) {
  const skill = ready ? state.skill : lastSkill;
  const known = skill && Object.hasOwn(skillLabels, skill.status);
  ui.grasp.disabled = !ready || !known || skill.status !== "idle" || plannerLocksScene();
  ui.storeScript.disabled = ui.grasp.disabled || !state?.available_tasks?.some((task) => task.kind === "store_bowl");
  ui.taskOverlay.hidden = !ready || !known || skill.status === "idle";
  const standalone = known && skill.execution_mode === "skill";
  const currentLabel = known ? primitiveLabel(skill.skill_name) : "等待状态";
  ui.overlayTask.textContent = known
    ? `${standalone ? "独立技能" : taskLabels[skill.task_kind] || "物理任务"} · ${currentLabel}` : "";
  ui.overlayPhase.textContent = known ? `${skillLabels[skill.status]} · ${phaseLabels[skill.phase] || skill.phase}` : "";
  updateParameterControls(ready, known ? skill : null);
  renderManualSkills(ready);
  ui.skillPanel.dataset.status = !ready ? "stale" : known ? skill.status : "waiting";
  ui.skillResult.textContent = known
    ? `${skillLabels[skill.status]}${!ready ? " · 最后状态" : skill.status === "running" && state.paused ? " · 已暂停" : ""}`
    : "等待状态";
  ui.skillHint.textContent = !ready
    ? "未连接或状态不可用；已有数值仅为最后状态，无法确认当前执行结果。"
    : !known ? "等待后端技能状态；暂不能开始物理任务。"
    : plannerLocksScene() ? "GPT 决策循环拥有场景；停止后续决策并等待当前技能结束后，才能运行手动对照。"
    : skill.status === "idle" ? "固定任务仅可从待开始状态启动；独立技能由上方后端物理条件决定，不调用 GPT。"
    : skill.status === "running" ? "物理执行中，可暂停；不能插入其他任务或技能。"
    : skill.status === "succeeded" ? "本次执行已成功；可按上方物理条件启动下一独立技能，无需重置。固定任务重跑仍需重置。"
    : "物理执行失败；请查看失败原因并重置，不允许从失败状态继续调用。";
  ui.currentSkill.textContent = currentLabel;
  ui.skillInvocation.textContent = known
    ? `${standalone ? "独立技能" : "固定任务"} · ${standalone ? primitiveLabel(skill.skill_name) : taskLabels[skill.task_kind] || "尚未启动"}${Number.isInteger(skill.invocation_id) ? ` · #${skill.invocation_id}` : ""}${!ready ? " · 最后状态" : ""}`
    : "等待状态";
  ui.skillPhase.textContent = known ? (phaseLabels[skill.phase] || skill.phase || "未报告") : "等待状态";
  ui.skillMessage.textContent = known ? (skill.message || "后端未提供阶段说明。") : "等待后端技能数据，不推断进度或结果。";
  ui.skillFailure.textContent = known && skill.failure_reason ? `失败原因：${skill.failure_reason}` : "";
  ui.skillFailure.hidden = !ui.skillFailure.textContent;
  ui.skillElapsed.textContent = number(known ? skill.elapsed_s : undefined, 2);
  ui.skillLift.textContent = number(known && Number.isFinite(skill.lift_height_m) ? skill.lift_height_m * 100 : undefined, 1);
  const showGraspTiming = known && (!standalone || skill.skill_name === "grasp_and_hold");
  ui.skillHold.textContent = number(showGraspTiming ? skill.hold_elapsed_s : undefined, 2);
  ui.skillPhaseElapsed.textContent = number(known ? skill.phase_elapsed_s : undefined, 2);
  ui.skillOverlap.textContent = number(showGraspTiming ? skill.descent_close_overlap_s : undefined, 3);
  ui.closeStarted.textContent = number(showGraspTiming ? skill.close_started_s : undefined, 3);
  ui.descentCompleted.textContent = number(showGraspTiming ? skill.descent_completed_s : undefined, 3);
  const parameters = known ? skill.parameters : null;
  ui.appliedParameters.textContent = standalone && skill.skill_name !== "grasp_and_hold"
    ? "本次独立技能不接收抓取参数；输入框和此前抓取参数不会被当作本次技能参数。"
    : validParameters(parameters)
      ? `${!ready ? "最后状态" : skill.status === "idle" ? "后端待用" : "实际应用"}：${parameterMode(parameters)}；${parameterSummary(parameters)}`
      : "后端参数：未报告完整参数，不使用输入框数值代替。";
  const closingDuringDescent = ready && known && skill.status === "running"
    && skill.phase === "descend" && Number.isFinite(skill.close_started_s);
  ui.earlyClose.hidden = !closingDuringDescent;
  ui.earlyClose.textContent = closingDuringDescent
    ? `下降阶段已触发闭合 · 后端开始时间 ${number(skill.close_started_s, 3)} s${state.paused ? " · 已暂停" : ""}；实际运动重叠见下方测量。` : "";
  const order = known && Array.isArray(skill.phase_order) ? skill.phase_order : [];
  const orderKey = JSON.stringify([skill?.invocation_id, skill?.execution_mode, order]);
  if (orderKey !== displayedPhaseOrder) {
    displayedPhaseOrder = orderKey;
    ui.skillStages.replaceChildren(...order.map((phase) => {
      const stage = document.createElement("li");
      stage.dataset.phase = phase;
      stage.textContent = phaseLabels[phase] || phase;
      return stage;
    }));
  }
  const storing = known && skill.task_kind === "store_bowl";
  ui.goalCriteria.textContent = !known ? "等待后端任务与完成判据。"
    : standalone ? "独立技能只验证本次契约；下方布尔值来自当前物理状态，不代表其他技能或完整任务已完成。"
    : storing ? "抓取保持只是中间步骤。最终须确认碗在抽屉内并受到支撑、夹爪已与碗分离且手部安全退出、抽屉关闭、机械臂归位且状态稳定；由后端判定。"
    : "抓取目标：实际抬升 ≥ 10 cm，双侧夹持并稳定保持 1 s；由后端判定。";
  const checks = known && skill.postconditions && typeof skill.postconditions === "object"
    ? Object.entries(skill.postconditions) : [];
  ui.skillPostconditions.replaceChildren(...(checks.length ? checks : [["后端尚未报告检查", null]]).map(([name, passed]) => {
    const row = document.createElement("div");
    const label = document.createElement("dt");
    const value = document.createElement("dd");
    label.textContent = name;
    value.textContent = yesNo(passed);
    value.dataset.passed = typeof passed === "boolean" ? String(passed) : "unknown";
    row.append(label, value);
    return row;
  }));
  const storage = known ? skill.storage : null;
  ui.storageStatus.hidden = !storage;
  ui.storeContained.textContent = yesNo(storage?.bowl_inside_drawer);
  ui.storeSupported.textContent = yesNo(storage?.bowl_supported);
  ui.storeClear.textContent = yesNo(storage?.gripper_clear);
  ui.storeHandClear.textContent = yesNo(storage?.hand_clear_of_drawer);
  ui.storePushContact.textContent = yesNo(storage?.push_contact_seen);
  ui.storeDrawer.textContent = number(storage?.drawer_position_m * 1000, 2);
  ui.storeClosed.textContent = yesNo(storage?.drawer_closed);
  ui.storeHomeError.textContent = number(storage?.home_error_rad, 4);
  ui.storeHome.textContent = yesNo(storage?.home_reached);
  ui.storeVerified.textContent = number(storage?.verification_hold_s, 2);
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
function renderSortingChecks(element, checks, labels) {
  element.replaceChildren(...Object.entries(labels).map(([key, text]) => {
    const row = document.createElement("div");
    const label = document.createElement("dt");
    const value = document.createElement("dd");
    label.textContent = text;
    value.textContent = yesNo(checks?.[key]);
    value.dataset.passed = typeof checks?.[key] === "boolean" ? String(checks[key]) : "unknown";
    row.append(label, value);
    return row;
  }));
}

function renderSortPlanner(ready) {
  const planner = ready ? state.planner : lastPlanner;
  const known = planner?.mode === "sorting";
  const pending = pendingPlannerAction !== null;
  const canGenerate = ready && known && planner.configured && planner.status === "idle"
    && state.sorting?.can_start === true && !pending;
  sortUi["planner-task"].disabled = !canGenerate;
  const task = sortUi["planner-task"].value.trim();
  sortUi["plan-generate"].disabled = !canGenerate || !task || task.length > 1000;
  sortUi["plan-execute"].disabled = !ready || !known || pending || planner.status !== "ready"
    || planner.decision?.decision !== "plan" || !Number.isInteger(planner.decision_id);
  sortUi["plan-cancel"].disabled = !ready || !known || pending || planner.can_cancel !== true;
  sortUi["planner-panel"].dataset.status = !ready ? "stale" : planner?.status || "waiting";
  sortUi["planner-status"].textContent = `${sortPlannerLabels[planner?.status] || "等待状态"}${!ready ? " · 最后状态" : state.paused ? " · 物理已暂停" : ""}`;
  sortUi["planner-hint"].textContent = !ready ? "连接不可用；仅显示最后记录，不能批准或确认当前结果。"
    : pending ? "操作已发送，等待后端确认；不会重复提交。"
    : planner?.status === "planning" ? "真实订阅请求进行中；只生成完整提案，不派发物理动作。"
    : planner?.status === "ready" ? "请核对下方实际映射与完整顺序。批准仅限所示编号；场景漂移会拒绝，不会静默修补。"
    : planner?.status === "executing" ? "已批准队列执行中；可暂停、单步或重置。取消按钮不可停止正在执行的机器人。"
    : planner?.status === "succeeded" ? "本地整批物理检查通过；后续目标丢失仍会撤销成功。模型摘要不是成功证据。"
    : ["failed", "unsupported", "cancelled"].includes(planner?.status) ? "没有自动重试或固定动作回退。请查看记录；再次尝试或固定对照须先重置。"
    : !planner?.configured ? "订阅工具未配置；固定对照仍可独立运行，不会调用模型。"
    : !canGenerate ? "物理启动条件尚未满足；须稳定支撑、空手归位且无正在执行的队列。"
    : "请求使用当前仿真真值与 ChatGPT 订阅。生成不执行，批准后执行完整六物体队列。";
  sortUi["planner-setup"].hidden = !known || planner.configured;
  sortUi["planner-error"].textContent = planner?.error || "";
  sortUi["planner-error"].hidden = !planner?.error;
  sortUi["planner-model"].textContent = planner?.model || "—";
  sortUi["planner-calls"].textContent = Number.isInteger(planner?.calls) ? String(planner.calls) : "—";
  sortUi["planner-latency"].textContent = number(planner?.latency_s, 2);
  sortUi["planner-total-latency"].textContent = number(planner?.total_latency_s, 2);
  sortUi["planner-response"].textContent = planner?.thread_id
    ? `Codex 会话 ID：${planner.thread_id} · 非致命诊断 ${planner.diagnostic_count ?? 0} 条` : "";
  sortUi["planner-submitted"].textContent = planner?.task ? `后端本次任务（所有客户端共享）：${planner.task}` : "";
  const decision = planner?.decision;
  sortUi["plan-preview"].hidden = !decision;
  sortUi["plan-preview-title"].textContent = decision?.decision === "unsupported" ? "无可执行提案"
    : `提案 #${planner?.decision_id} · ${planner?.status === "ready" ? "尚未批准执行" : "状态见真实记录"}`;
  sortUi["plan-summary"].textContent = decision ? `模型摘要（非成功证据）：${decision.summary}` : "";
  sortUi["plan-reason"].textContent = decision ? `模型依据：${decision.reason}` : "";
  const objects = (ready ? state.sorting : lastSorting)?.plan || [];
  const proposalKey = JSON.stringify([decision?.steps, objects.map(({ object_id, label, color }) => [object_id, label, color])]);
  if (proposalKey !== sortProposalKey) {
    sortProposalKey = proposalKey;
    const mapping = new Map();
    sortUi["plan-queue"].replaceChildren(...(decision?.steps || []).map((step) => {
      const object = objects.find((entry) => entry.object_id === step.object_id);
      const item = document.createElement("li");
      if (object) {
        item.dataset.color = object.color;
        mapping.set(object.color, step.destination);
      }
      item.textContent = `${object?.label || step.object_id} [${step.object_id}] → ${step.destination} 区`;
      return item;
    }));
    sortUi["plan-mapping"].replaceChildren(...Array.from(mapping, ([color, destination]) => {
      const item = document.createElement("li");
      item.dataset.color = color;
      item.textContent = `${colorLabels[color] || color} → ${destination} 区`;
      return item;
    }));
  }
  const history = planner?.history || [];
  const historyKey = JSON.stringify(history);
  if (historyKey !== sortHistoryKey) {
    sortHistoryKey = historyKey;
    sortUi["planner-history-empty"].hidden = history.length > 0;
    sortUi["planner-history"].replaceChildren(...history.map((record) => {
      const item = document.createElement("li");
      item.dataset.status = record.status;
      const title = document.createElement("strong");
      title.textContent = `提案 #${record.decision_id} · ${sortPlannerLabels[record.status] || record.status}`;
      const timing = document.createElement("p");
      timing.textContent = `GPT ${number(record.latency_s, 2)} 墙钟秒 · 执行 ${number(record.execution_sim_s, 3)} 仿真秒 / ${number(record.execution_wall_s, 2)} 墙钟秒`;
      const outcome = document.createElement("p");
      const observed = record.result_observation?.sorting;
      outcome.textContent = record.error ? `错误：${record.error}`
        : observed ? `实测反馈：${skillLabels[observed.status] || observed.status} · ${observed.message || ""}${observed.failure_reason ? ` · ${observed.failure_reason}` : ""}`
        : "尚无执行反馈；模型回复不是物理成功证据。";
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "完整提案、请求观测与最终物理反馈";
      const payload = document.createElement("pre");
      payload.textContent = JSON.stringify(record, null, 2);
      details.append(summary, payload);
      item.append(title, timing, outcome, details);
      return item;
    }));
  }
}

function renderSorting(ready) {
  const sorting = ready ? state.sorting : lastSorting;
  const planner = ready ? state.planner : lastPlanner;
  const attempted = Boolean(planner?.task || planner?.calls || planner?.history?.length);
  const mode = sorting?.mode === "planned" ? "已批准计划" : "固定对照";
  ui.sortStart.disabled = !ready || sorting?.can_start !== true || plannerLocksScene() || attempted;
  ui.sortingPanel.dataset.status = ready ? sorting.status : "stale";
  ui.sortingStatus.textContent = `${skillLabels[sorting?.status] || "等待状态"}${!ready ? " · 最后状态" : state.paused ? " · 已暂停" : ""}`;
  ui.sortingHint.textContent = !ready ? "连接或状态不可用；以下仅为最后状态，不能确认当前结果。"
    : sorting.status === "running" ? `${mode}队列执行中；可暂停或单步，不接受其他任务，不自动重试。`
    : ["failed", "succeeded"].includes(sorting.status) ? "本次运行已结束；再次运行必须先显式重置。"
    : attempted || plannerLocksScene() ? "GPT 请求 / 提案与固定对照互斥；固定对照须先重置，不能隐藏本次请求历史。"
    : sorting.can_start ? "当前物理条件允许开始；点击后按既定全队列执行并自动恢复物理时钟。"
    : `暂不可开始：${sorting.start_error || "后端尚未确认物体稳定支撑。"}`;
  const current = sorting?.plan?.find((item) => item.object_id === sorting.current_object_id
    && item.index === sorting.current_step_index);
  const phase = sortingPhaseLabels[sorting?.phase] || sorting?.phase || "未报告";
  ui.sortingCurrent.textContent = current ? `#${current.index + 1} ${current.label} → ${current.destination} 区` : "无当前物体";
  ui.sortingPhase.textContent = phase;
  ui.sortingMessage.textContent = sorting?.message || "等待后端阶段说明，不推断进度。";
  ui.sortingFailure.textContent = sorting?.failure_reason ? `失败原因：${sorting.failure_reason}`
    : sorting?.status === "failed" ? "后端报告失败，但未提供失败原因。" : "";
  ui.sortingFailure.hidden = !ui.sortingFailure.textContent;
  ui.sortingCount.textContent = `${number(sorting?.sorted_count, 0)} / ${number(sorting?.object_count, 0)}`;
  if (Number.isFinite(sorting?.object_count) && sorting.object_count > 0) ui.sortingProgress.max = sorting.object_count;
  if (Number.isFinite(sorting?.sorted_count)) ui.sortingProgress.value = sorting.sorted_count;
  else ui.sortingProgress.removeAttribute("value");
  ui.sortingElapsed.textContent = number(sorting?.elapsed_s, 2);
  ui.sortingHold.textContent = number(sorting?.validation_hold_s, 2);
  ui.taskOverlay.hidden = !ready || sorting.status === "idle";
  ui.overlayTask.textContent = `${mode} · ${ui.sortingCount.textContent}${current ? ` · ${current.label}` : ""}`;
  ui.overlayPhase.textContent = `${skillLabels[sorting?.status] || "未报告"} · ${phase}`;
  const queueKey = JSON.stringify([sorting?.plan, sorting?.current_object_id, sorting?.current_step_index]);
  if (queueKey !== sortingQueueKey) {
    sortingQueueKey = queueKey;
    const labels = { pending: "待处理", active: "正在处理", sorted: "实测已分拣", failed: "失败" };
    ui.sortingQueue.replaceChildren(...(sorting?.plan || []).map((entry) => {
      const item = document.createElement("li");
      item.dataset.color = entry.color;
      item.dataset.status = entry.status;
      if (entry.object_id === sorting.current_object_id && entry.index === sorting.current_step_index) item.setAttribute("aria-current", "step");
      const title = document.createElement("strong");
      const detail = document.createElement("span");
      title.textContent = `${entry.label} · ${colorLabels[entry.color] || entry.color}${kindLabels[entry.kind] || entry.kind} → ${entry.destination} 区`;
      detail.textContent = `${labels[entry.status] || entry.status} · 落点 XY (${(entry.slot_xy_m || []).map((value) => number(value, 3)).join(", ")}) m`;
      item.append(title, detail);
      return item;
    }));
  }
  renderSortingChecks(ui.sortingChecks, sorting?.goal_checks, sortingCheckLabels);
  const objectsKey = JSON.stringify(sorting?.objects?.map((object) => [
    object.id, object.label, object.color, object.kind, object.correct_region,
    object.supported, object.stable, object.detached, object.sorted,
  ]));
  if (objectsKey !== sortingObjectsKey) {
    sortingObjectsKey = objectsKey;
    ui.sortingObjects.replaceChildren(...(sorting?.objects || []).map((object) => {
      const card = document.createElement("article");
      card.className = "object";
      card.dataset.color = object.color;
      const title = document.createElement("h3");
      title.textContent = `${object.label} · ${colorLabels[object.color] || object.color}${kindLabels[object.kind] || object.kind}`;
      const checks = document.createElement("dl");
      checks.className = "metadata storage-metrics";
      renderSortingChecks(checks, object, {
        correct_region: "正确色区", supported: "实际支撑", stable: "物体稳定",
        detached: "已与夹爪分离", sorted: "实测已分拣",
      });
      card.append(title, checks);
      return card;
    }));
  }
}

function updateSceneMode() {
  if (!state) return;
  const nextMode = state.sorting ? "sorting" : "storage";
  if (nextMode === sceneMode) return;
  sceneMode = nextMode;
  const sorting = sceneMode === "sorting";
  storageElements.forEach((element) => { element.hidden = sorting; });
  sortingElements.forEach((element) => { element.hidden = !sorting; });
  document.title = sorting ? "GPT / 固定规划 · 彩色碗杯物理分拣" : "GPT 实时单技能决策 · 物理执行与反馈";
  if (!sorting && !storageReportsLoaded) {
    storageReportsLoaded = true;
    loadExperiments();
    loadPlanningComparison();
  }
  if (sorting && !sortingReportLoaded) {
    sortingReportLoaded = true;
    loadSortingEvaluation();
  }
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
  ui.toggle.disabled = !ready;
  ui.reset.disabled = !ready;
  ui.camera.disabled = !ready;
  ui.step.disabled = !ready || !state.paused;
  ui.toggle.textContent = state && !state.paused ? "暂停" : "开始运行";
  updateSceneMode();
  if (sceneMode === "sorting") {
    renderSorting(ready);
    renderSortPlanner(ready);
  } else {
    renderSkill(ready);
    renderPlanner(ready);
  }
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
  if (pendingPlannerAction && (next.planner?.decision_id !== state?.planner?.decision_id
      || next.planner?.status !== state?.planner?.status)) pendingPlannerAction = null;
  state = next;
  if (state.skill) lastSkill = state.skill;
  if (state.planner) lastPlanner = state.planner;
  if (state.sorting) lastSorting = state.sorting;
  ui.time.textContent = number(state.time_s, 3);
  ui.scene.textContent = state.scene_name;
  ui.sortingLayout.textContent = state.sorting_layout ?? "默认布局（非独立评估布局）";
  ui.robot.textContent = state.robot_name;
  ui.source.textContent = state.source;
  ui.gripper.textContent = number(state.gripper_opening_m * 1000, 2);
  ui.contacts.textContent = Number.isFinite(state.contacts) ? String(state.contacts) : "—";
  ui.timestep.textContent = number(state.timestep_s * 1000, 2);
  ui.collisionEpisodes.textContent = Number.isInteger(state.collisions?.episodes) ? String(state.collisions.episodes) : "—";
  ui.collisionForce.textContent = number(state.collisions?.peak_force_n, 2);
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
  ui.error.hidden = fatalRuntimeError === null;
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
        if (message.fatal === true) {
          fatalRuntimeError = String(message.error);
          state = null;
        }
        updateControls();
        showError(`仿真器报告错误：${String(message.error)}`);
        setStatus(fatalRuntimeError ? "仿真线程停止 · 需重启服务" : "已连接 · 操作错误", "error");
        return;
      }
      if (!message.state || typeof message.image !== "string" || !message.image) {
        throw new Error("后端消息缺少状态或 JPEG 画面。");
      }
      renderState(message.state);
      if (fatalRuntimeError !== null) {
        fatalRuntimeError = null;
        ui.error.hidden = true;
      }
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
    showError(fatalRuntimeError || "WebSocket 连接发生错误。请确认本地仿真服务正在运行，再点击重新连接。");
    setStatus(fatalRuntimeError ? "仿真线程停止 · 需重启服务" : "连接错误", "error");
  });
  socket.addEventListener("close", (event) => {
    state = null;
    pendingPlannerAction = null;
    updateControls();
    setStatus(fatalRuntimeError ? "仿真线程停止 · 需重启服务" : "已断开 · 更新停止", fatalRuntimeError ? "error" : "disconnected");
    ui.stale.textContent = fatalRuntimeError ? "仿真线程停止 · 最后一帧" : "连接已中断 · 最后一帧";
    ui.stale.hidden = !hasFrame;
    ui.note.textContent = fatalRuntimeError || "连接已中断，下方数值为最后接收到的状态，不再实时更新。";
    ui.reconnectPanel.hidden = false;
    showError(fatalRuntimeError || `仿真连接已关闭（代码 ${event.code}）${event.reason ? `：${event.reason}` : "。"} 无法确认后端是否仍在运行。`);
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
ui.sortStart.addEventListener("click", () => {
  if (!ui.sortStart.disabled && state?.sorting?.can_start === true) send({ action: "sort" });
});
sortUi["planner-task"].addEventListener("input", updateControls);
sortUi["plan-generate"].addEventListener("click", () => {
  if (!sortUi["plan-generate"].disabled) sendPlanner({ action: "sort_plan", task: sortUi["planner-task"].value.trim() });
});
sortUi["plan-execute"].addEventListener("click", () => {
  if (!sortUi["plan-execute"].disabled) sendPlanner({ action: "sort_execute", decision_id: state.planner.decision_id });
});
sortUi["plan-cancel"].addEventListener("click", () => {
  if (!sortUi["plan-cancel"].disabled) sendPlanner({ action: "sort_cancel" });
});
ui.grasp.addEventListener("click", () => {
  if (!ui.grasp.disabled && state?.skill?.status === "idle") send({ action: "grasp" });
});
ui.storeScript.addEventListener("click", () => {
  if (!ui.storeScript.disabled && state?.skill?.status === "idle") send({ action: "store" });
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
ui.manualSkill.addEventListener("change", updateControls);
ui.startSkill.addEventListener("click", () => {
  if (ui.startSkill.disabled || plannerLocksScene()) return;
  const selected = state?.available_skills?.find((skill) => skill.name === ui.manualSkill.value);
  if (selected?.available !== true) return;
  try {
    const command = { action: "skill", skill: selected.name };
    if (selected.name === "grasp_and_hold") command.parameters = readCustomParameters();
    ui.parameterError.textContent = "";
    ui.parameterError.hidden = true;
    send(command);
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
  if (!ui.planExecute.disabled) sendPlanner({ action: "execute_decision", decision_id: state.planner.decision_id, continuous: false });
});
ui.planContinuous.addEventListener("click", () => {
  if (!ui.planContinuous.disabled) sendPlanner({ action: "execute_decision", decision_id: state.planner.decision_id, continuous: true });
});
ui.planNext.addEventListener("click", () => {
  if (!ui.planNext.disabled) sendPlanner({ action: "plan_next" });
});
ui.planCancel.addEventListener("click", () => {
  if (!ui.planCancel.disabled) sendPlanner({ action: "cancel_plan" });
});
ui.camera.addEventListener("change", () => send({ action: "camera", view: ui.camera.value }));
ui.reconnect.addEventListener("click", connect);
ui.experimentsRefresh.addEventListener("click", loadExperiments);
comparisonUi.refresh.addEventListener("click", loadPlanningComparison);
comparisonUi.batch.addEventListener("change", loadPlanningComparison);
evaluationUi.refresh.addEventListener("click", loadSortingEvaluation);
connect();
