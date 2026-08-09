const STEMS = ["bass", "drums", "other", "vocals", "guitar", "piano"];
const DEFAULT_CHECKED = ["vocals", "piano", "guitar", "bass"];
let settings = {};
let currentJob = null;
let eventSource = null;

const $ = (id) => document.getElementById(id);

// ---------- settings ----------
async function loadSettings() {
  settings = await (await fetch("/api/settings")).json();
  renderSettingsForm();
}
function renderSettingsForm() {
  const fields = [
    ["separation_precision", "Separation precision", ["fp16", "fp32"]],
    ["model_size", "Model size", ["small", "medium", "large"]],
    ["separation_device", "Separation device", ["auto", "cuda", "cpu"]],
    ["transcription_device", "Transcription device", ["auto", "cuda", "cpu"]],
    ["temperature", "Temperature (0 = deterministic)", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5]],
    ["beam_size", "Beam size", [1, 2, 3, 4, 5, 6, 8]],
    ["batch_size", "Batch size", [1, 2, 4, 8]],
    ["output_folder", "Output folder", null],
  ];
  $("settings-form").innerHTML = fields.map(([key, label, options]) => {
    const val = settings[key];
    let control;
    if (Array.isArray(options)) {
      control = `<select id="s-${key}">${options.map((o) => `<option value="${o}" ${String(o) === String(val) ? "selected" : ""}>${o}</option>`).join("")}</select>`;
    } else {
      control = `<input id="s-${key}" type="text" value="${val}">`;
    }
    return `<label class="field"><span>${label}</span>${control}</label>`;
  }).join("");
  [["keep_stems", "Keep stem WAVs after conversion"],
   ["remember_selection", "Remember last stem selection"]].forEach(([key, label]) => {
    $("settings-form").insertAdjacentHTML("beforeend",
      `<label class="field toggle"><input type="checkbox" id="s-${key}" ${settings[key] ? "checked" : ""}> <span>${label}</span></label>`);
  });
}
async function saveSettings() {
  const body = {};
  ["separation_precision", "model_size", "separation_device", "transcription_device",
   "temperature", "beam_size", "batch_size", "output_folder"].forEach((key) => {
    body[key] = $(`s-${key}`).value;
  });
  body.temperature = parseFloat(body.temperature);
  body.beam_size = parseInt(body.beam_size, 10);
  body.batch_size = parseInt(body.batch_size, 10);
  ["keep_stems", "remember_selection"].forEach((key) => {
    body[key] = $(`s-${key}`).checked;
  });
  body.instrument_by_stem = settings.instrument_by_stem || {};
  const resp = await fetch("/api/settings", {
    method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
  });
  settings = await resp.json();
  $("settings-note").textContent = "Settings saved.";
}

// ---------- upload ----------
function setupDropzone() {
  const dz = $("drop-zone");
  const input = $("file-input");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("over");
    if (e.dataTransfer.files.length) upload(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => { if (input.files.length) upload(input.files[0]); });
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
}
async function upload(file) {
  $("job-error").classList.add("hidden");
  const fd = new FormData();
  fd.append("file", file);
  const resp = await fetch("/api/jobs", { method: "POST", body: fd });
  if (!resp.ok) { showError((await resp.json()).detail); return; }
  const { job_id } = await resp.json();
  currentJob = job_id;
  $("job-panel").classList.remove("hidden");
  $("stem-panel").classList.add("hidden");
  $("results").innerHTML = "";
  setProgress(0, "Separating stems…");
  connectEvents(job_id);
}

// ---------- SSE ----------
function connectEvents(jobId) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(`/api/jobs/${jobId}/events`);
  eventSource.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    if (ev.type === "progress") {
      if (ev.phase === "separating") setProgress(ev.pct, "Separating stems…");
      if (ev.phase === "transcribing") setProgress(ev.pct, `Transcribing ${ev.stem}…`);
    } else if (ev.type === "stems") {
      setProgress(100, "Separation done");
      showStems();
    } else if (ev.type === "midi") {
      addResult(ev.stem, ev.file);
    } else if (ev.type === "done") {
      setProgress(100, "Done");
      eventSource.close();
    } else if (ev.type === "error") {
      showError(ev.message);
    } else if (ev.type === "failed" || ev.type === "cancelled") {
      showError(ev.message || ev.type);
      eventSource.close();
    }
  };
}

// ---------- stems ----------
function showStems() {
  const remembered = settings.remember_selection
    ? (JSON.parse(localStorage.getItem("checkedStems") || "null") || DEFAULT_CHECKED)
    : DEFAULT_CHECKED;
  STEMS.forEach((stem) => {
    const card = document.querySelector(`.stem-card[data-stem="${stem}"]`);
    card.querySelector('input[type="checkbox"]').checked = remembered.includes(stem);
    card.querySelector("audio").src = `/output/${currentJob}/stems/${stem}.wav`;
    card.querySelector(".inst").value = settings.instrument_by_stem?.[stem] || "";
  });
  $("stem-panel").classList.remove("hidden");
}
function selectedStems() {
  return [...document.querySelectorAll('input[data-stem]:checked')].map((el) => el.dataset.stem);
}
async function transcribeSelected() {
  const stems = selectedStems();
  if (!stems.length) { showError("Select at least one stem."); return; }
  settings.instrument_by_stem = {};
  [...document.querySelectorAll('input[data-instrument]')].forEach((el) => {
    settings.instrument_by_stem[el.dataset.stem] = el.value.trim();
  });
  if (settings.remember_selection) localStorage.setItem("checkedStems", JSON.stringify(stems));
  await saveSettings();
  await fetch(`/api/jobs/${currentJob}/transcribe`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ stems }),
  });
  setProgress(0, "Transcribing…");
}

// ---------- results ----------
function addResult(stem, file) {
  const link = document.createElement("a");
  link.href = `/output/${currentJob}/midi/${encodeURIComponent(file)}`;
  link.textContent = `download ${file}`;
  link.classList.add("result-link");
  $("results").appendChild(link);
}
function setProgress(pct, text) {
  $("progress-bar").style.width = `${pct}%`;
  $("progress-text").textContent = text;
  $("job-status").textContent = text;
}
function showError(message) {
  const el = $("job-error");
  el.textContent = message;
  el.classList.remove("hidden");
}

$("btn-save-settings").addEventListener("click", saveSettings);
$("btn-cancel").addEventListener("click", async () => {
  if (currentJob) await fetch(`/api/jobs/${currentJob}/cancel`, { method: "POST" });
});
$("btn-transcribe").addEventListener("click", transcribeSelected);
loadSettings();
setupDropzone();
