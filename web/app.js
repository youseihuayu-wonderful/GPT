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
};
let socket = null;
let state = null;
let hasFrame = false;
const objectRows = [];
const jointOutputs = [];

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
  state = next;
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
    return;
  }
  try {
    socket.send(JSON.stringify(command));
    ui.error.hidden = true;
  } catch (error) {
    showError(`操作未发送：${error.message}`);
  }
}

function connect() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;
  state = null;
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
ui.toggle.addEventListener("click", () => send({ action: state?.paused ? "play" : "pause" }));
ui.step.addEventListener("click", () => { if (state?.paused) send({ action: "step" }); });
ui.reset.addEventListener("click", () => send({ action: "reset" }));
ui.camera.addEventListener("change", () => send({ action: "camera", view: ui.camera.value }));
ui.reconnect.addEventListener("click", connect);
connect();
