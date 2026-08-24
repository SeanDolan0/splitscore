const STEMS = ["bass", "drums", "other", "vocals", "guitar", "piano"];
const DEFAULT_CHECKED = ["vocals", "piano", "guitar", "bass"];
let settings = {};
let currentJob = null;
let eventSource = null;
let renderedFiles = new Set(); // dedup midi events replayed by SSE against stored files
let currentPlayingAudio = null; // track the currently-playing audio so only one plays at a time

const $ = (id) => document.getElementById(id);

// ---------- hardware badge ----------
async function loadHardware() {
  try {
    const hw = await (await fetch("/api/hardware")).json();
    const badge = $("hw-badge");
    const text = $("hw-text");
    const icon = badge.querySelector(".hw-icon");
    const sep = hw.separator_device || "unknown";
    const isCuda = sep === "cuda";
    const gpuName = hw.gpu?.name || "No GPU";
    const ortProviders = hw.onnxruntime?.providers || [];
    const ortActive = ortProviders.includes("CUDAExecutionProvider") ? "CUDA" : "CPU";

    if (isCuda && hw.gpu?.vendor === "nvidia") {
      icon.innerHTML = '<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1l6.5 3.75v7.5L8 16l-6.5-3.75v-7.5z"/></svg>';
      badge.classList.add("hw-cuda");
      text.textContent = `${gpuName} · CUDA · ort:${ortActive}`;
    } else if (hw.gpu?.vendor === "apple") {
      icon.textContent = "";
      badge.classList.add("hw-apple");
      text.textContent = `Apple Silicon · ort:${ortActive}`;
    } else {
      icon.innerHTML = '<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1l6.5 3.75v7.5L8 16l-6.5-3.75v-7.5z"/></svg>';
      badge.classList.add("hw-cpu");
      text.textContent = `${gpuName} · ort:${ortActive}`;
    }
    badge.title = `torch ${hw.torch?.version || "?"} · onnxruntime ${hw.onnxruntime?.version || "?"}`;
  } catch {
    // server not up yet or /api/hardware missing — leave "detecting…" text
  }
}

// ---------- settings ----------
async function loadSettings() {
  settings = await (await fetch("/api/settings")).json();
  renderSettingsForm();
}
let instrumentList = [];
async function loadInstruments() {
  // Fetch instrument vocabulary from MuScriptor for custom dropdown
  const { instruments } = await (await fetch("/api/instruments")).json();
  instrumentList = instruments;
  initInstrumentDropdowns();
}
function initInstrumentDropdowns() {
  // Categorize instruments for better UX
  const categorized = categorizeInstruments(instrumentList);
  // Build dropdown HTML once
  const dropdownHTML = buildDropdownHTML(categorized);
  // Inject into each stem's dropdown container
  document.querySelectorAll(".inst-dropdown").forEach((dd) => {
    dd.innerHTML = dropdownHTML;
  });
  // Wire up interactions
  document.querySelectorAll(".inst").forEach((input) => {
    const dropdown = input.nextElementSibling;
    // open on focus/click
    input.addEventListener("focus", () => openDropdown(dropdown, input));
    input.addEventListener("click", (e) => { e.stopPropagation(); openDropdown(dropdown, input); });
    // filter as the user types in the stem input itself
    input.addEventListener("input", () => filterDropdown(dropdown, input.value));
  });
  // filter as the user types in the dropdown's embedded search box
  document.querySelectorAll(".inst-dropdown__search input").forEach((search) => {
    search.addEventListener("input", () => filterDropdown(search.closest(".inst-dropdown"), search.value));
  });
  // Close on outside click
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".inst-wrap")) {
      document.querySelectorAll(".inst-dropdown").forEach(closeDropdown);
    }
  });
  // Keyboard navigation on inputs
  document.querySelectorAll(".inst").forEach((input) => {
    input.addEventListener("keydown", (e) => handleDropdownKeydown(e, input));
  });
}
function categorizeInstruments(instruments) {
  const cats = { keyboard: [], guitar: [], bass: [], strings: [], wind: [], percussion: [], synth: [], vocal: [], other: [] };
  for (const inst of instruments) {
    const l = inst.toLowerCase();
    if (/(piano|keys|organ|rhodes|clav|harpsi|accordion)/.test(l)) cats.keyboard.push(inst);
    else if (/(guitar|acoustic|electric|clean|dist|strat|les paul|tele)/.test(l)) cats.guitar.push(inst);
    else if (/(bass|upright|fretless|precision|jazz bass)/.test(l)) cats.bass.push(inst);
    else if (/(violin|viola|cello|double bass|harp|pizzicato)/.test(l)) cats.strings.push(inst);
    else if (/(sax|trumpet|trombone|flute|clarinet|oboe|horn|woodwind|brass)/.test(l)) cats.wind.push(inst);
    else if (/(drum|kit|perc|cymbal|snare|kick|tom|conga|bongo|shaker|tambourine)/.test(l)) cats.percussion.push(inst);
    else if (/(synth|pad|lead|arp|pluck|bell|fm|analog|moog|modular|wavetable)/.test(l)) cats.synth.push(inst);
    else if (/(voice|vocal|choir|chant|hum|whisper)/.test(l)) cats.vocal.push(inst);
    else cats.other.push(inst);
  }
  // Remove empty categories
  const result = {};
  for (const [k, v] of Object.entries(cats)) if (v.length) result[k] = v;
  return result;
}
function buildDropdownHTML(cats) {
  let html = `<div class="inst-dropdown__search"><input type="search" placeholder="Filter instruments…" aria-label="Filter instruments"></div>
  <div class="inst-dropdown__list" role="listbox">`;
  const order = ["keyboard", "guitar", "bass", "strings", "wind", "percussion", "synth", "vocal", "other"];
  const labels = { keyboard: "KEYBOARD", guitar: "GUITAR", bass: "BASS", strings: "STRINGS", wind: "WIND", percussion: "PERCUSSION", synth: "SYNTH", vocal: "VOCAL", other: "OTHER" };
  for (const cat of order) {
    if (!cats[cat]) continue;
    for (const inst of cats[cat]) {
      html += `<div class="inst-dropdown__item" role="option" data-value="${inst.replace(/"/g, "&quot;")}"><span class="inst-category">${labels[cat]}</span>${inst}</div>`;
    }
  }
  html += `</div>
  <div class="inst-dropdown__footer"><span>Enter=Select</span><span>↑↓=Navigate <kbd>Esc</kbd>=Close</span></div>`;
  return html;
}
function openDropdown(dropdown, input) {
  // Close others
  document.querySelectorAll(".inst-dropdown").forEach((d) => { if (d !== dropdown) closeDropdown(d); });
  dropdown.classList.add("open");
  // Focus search
  const search = dropdown.querySelector(".inst-dropdown__search input");
  if (search) { search.value = input.value; search.focus(); }
  // Highlight current value if present
  highlightMatching(dropdown, input.value);
}
function closeDropdown(dropdown) {
  dropdown.classList.remove("open");
}
function filterDropdown(dropdown, query) {
  const q = query.toLowerCase();
  dropdown.querySelectorAll(".inst-dropdown__item").forEach((item) => {
    const match = item.dataset.value.toLowerCase().includes(q);
    item.style.display = match ? "flex" : "none";
  });
  // Clear highlight if no visible items match
  const visible = dropdown.querySelectorAll(".inst-dropdown__item[style*='flex'], .inst-dropdown__item:not([style*='none'])");
  if (!dropdown.querySelector(".inst-dropdown__item.highlighted") && visible.length) {
    visible[0].classList.add("highlighted");
  }
}
function highlightMatching(dropdown, value) {
  dropdown.querySelectorAll(".inst-dropdown__item").forEach((item) => {
    item.classList.toggle("highlighted", item.dataset.value === value);
    item.classList.toggle("selected", item.dataset.value === value);
  });
}
function handleDropdownKeydown(e, input) {
  const dropdown = input.nextElementSibling;
  if (!dropdown.classList.contains("open")) {
    if (e.key === "ArrowDown" || e.key === "ArrowUp" || e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      openDropdown(dropdown, input);
    }
    return;
  }
  const items = [...dropdown.querySelectorAll(".inst-dropdown__item:not([style*='none'])")];
  const current = dropdown.querySelector(".inst-dropdown__item.highlighted");
  let idx = items.indexOf(current);
  if (e.key === "ArrowDown") {
    e.preventDefault();
    idx = (idx + 1) % items.length;
    items.forEach((it, i) => it.classList.toggle("highlighted", i === idx));
    items[idx]?.scrollIntoView({ block: "nearest" });
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    idx = (idx - 1 + items.length) % items.length;
    items.forEach((it, i) => it.classList.toggle("highlighted", i === idx));
    items[idx]?.scrollIntoView({ block: "nearest" });
  } else if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    if (current) selectDropdownItem(input, dropdown, current.dataset.value);
  } else if (e.key === "Escape") {
    e.preventDefault();
    closeDropdown(dropdown);
    input.blur();
  } else if (e.key === "Tab") {
    closeDropdown(dropdown);
  }
}
function selectDropdownItem(input, dropdown, value) {
  input.value = value;
  input.dispatchEvent(new Event("input", { bubbles: true }));
  closeDropdown(dropdown);
}

function renderSettingsForm() {
  const splitter = [
    ["separation_precision", "Separation precision", ["fp16", "fp32"],
     "fp16 uses a smaller model that's faster and uses less memory with virtually identical quality. Use fp32 only if you notice audio artifacts in the separated stems."],
    ["separation_device", "Separation device", ["auto", "cuda", "cpu"],
     "auto selects the best available hardware — NVIDIA GPU (CUDA) on Windows/Linux, CPU fallback otherwise. On Apple Silicon and AMD GPUs, separation always runs on CPU (ONNX Runtime limitation)."],
  ];
  const general = [
    ["output_folder", "Output folder", null,
     "Where all output is saved. Each job creates a subfolder with separated stems (.wav) and transcribed MIDI files (.mid)."],
  ];
  const midi = [
    ["model_size", "Transcription model", ["small", "medium", "large"],
     "small (103M params) is fastest and works well on CPU. medium (307M) is a balanced choice. large (1.4B) is the most accurate but needs a GPU with 4+ GB VRAM."],
    ["transcription_device", "Transcription device", ["auto", "cuda", "cpu"],
     "auto selects the best available hardware. MuScriptor is significantly faster on NVIDIA CUDA. Apple Silicon uses MPS. CPU works but is much slower."],
    ["temperature", "Temperature", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5],
     "Controls randomness in note prediction. 0 is deterministic — the same audio always produces the same MIDI. Higher values (0.5–1.0) add natural variation but may introduce wrong notes. Above 1.5 becomes unreliable."],
    ["beam_size", "Beam size", [1, 2, 3, 4, 5, 6, 8],
     "How many alternative interpretations the model tracks simultaneously. Higher values find better note sequences but use proportionally more GPU memory. Start with 2–4; drop to 1 if you run out of memory."],
    ["batch_size", "Batch size", [1, 2, 4, 8],
     "How many audio chunks are processed in parallel. Leave at 1 for best quality — higher values can cause artifacts at chunk boundaries. Only increase for faster processing on long files with GPU memory to spare."],
  ];
  const help = (label, desc) =>
    `<span class="q" tabindex="0" role="note" aria-label="Help: ${label}"><span class="tip" role="tooltip">${desc}</span>?</span>`;
  const field = ([key, label, options, desc]) => {
    const val = settings[key];
    const control = Array.isArray(options)
      ? `<select id="s-${key}">${options.map((o) => `<option value="${o}" ${String(o) === String(val) ? "selected" : ""}>${o}</option>`).join("")}</select>`
      : `<input id="s-${key}" type="text" value="${val}">`;
    return `<label class="field"><span class="field-label"><span>${label}</span>${help(label, desc)}</span>${control}</label>`;
  };
  const toggle = ([key, label, group, desc]) =>
    `<label class="field toggle"><input type="checkbox" id="s-${key}" ${settings[key] ? "checked" : ""}> <span>${label}</span>${help(label, desc)}</label>`;
  $("settings-form").innerHTML =
    `<h3 class="settings-group mono-label">Stem splitter</h3>` +
    splitter.map(field).join("") +
    toggle(["keep_stems", "Keep stem WAVs after transcription", "splitter",
            "Keep the separated .wav stem files on disk after MIDI transcription. Turn off to save disk space."]) +
    `<h3 class="settings-group mono-label">Audio to MIDI</h3>` +
    midi.map(field).join("") +
    toggle(["remember_selection", "Remember last stem selection", "midi",
            "Restore the stems you last checked across jobs (browser localStorage)."]) +
    `<h3 class="settings-group mono-label">General</h3>` +
    general.map(field).join("");
}
async function saveSettings() {
  const saveBtn = $("btn-save-settings");
  if (saveBtn) saveBtn.disabled = true;
  try {
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
  } finally {
    if (saveBtn) saveBtn.disabled = false;
  }
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
  localStorage.setItem("jobId", job_id); // survive reload while the job runs
  $("drop-file").textContent = file.name;
  $("drop-zone").classList.add("loaded");
  $("job-panel").classList.remove("hidden");
  $("stem-panel").classList.add("hidden");
  $("results").innerHTML = "";
  $("midi-actions").classList.add("hidden");
  renderedFiles = new Set();
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
      localStorage.removeItem("jobId");
      endJob();
    } else if (ev.type === "error") {
      showError(ev.message);
    } else if (ev.type === "failed") {
      showError(ev.message || "Failed");
      localStorage.removeItem("jobId");
      endJob();
    } else if (ev.type === "cancelled") {
      showNotice(ev.message || "Cancelled");
      localStorage.removeItem("jobId");
      endJob();
    }
  };
  eventSource.onerror = () => { if (currentJob) endJob(); };
}
function endJob() {
  if (eventSource) eventSource.close();
  if ($("btn-transcribe")) $("btn-transcribe").disabled = false;
  if ($("btn-cancel")) $("btn-cancel").disabled = true; // terminal: nothing left to cancel
}

// ---------- reload recovery ----------
function renderStoredMidi(files, songName) {
  (files || []).forEach((file) => {
    const stem = file.replace(`${songName}_`, "").replace(/\.mid$/, "");
    addResult(stem || "midi", file);
  });
}
async function resumeJob() {
  const id = localStorage.getItem("jobId");
  if (!id) return;
  let job;
  try {
    const resp = await fetch(`/api/jobs/${id}`);
    if (!resp.ok) { localStorage.removeItem("jobId"); return; } // job gone (server restarted)
    job = await resp.json();
  } catch { localStorage.removeItem("jobId"); return; }
  currentJob = id;
  $("drop-file").textContent = job.song_name;
  $("drop-zone").classList.add("loaded");
  $("job-panel").classList.remove("hidden");
  $("stem-panel").classList.add("hidden");
  $("results").innerHTML = "";
  $("midi-actions").classList.add("hidden");
  renderedFiles = new Set();
  if (job.status === "ready") {
    setProgress(100, "Separation done");
    showStems();
    connectEvents(id); // catch transcription events once the user restarts it
  } else if (job.status === "done") {
    setProgress(100, "Done");
    renderStoredMidi(job.midi, job.song_name);
    localStorage.removeItem("jobId");
    endJob();
  } else if (job.status === "failed") {
    showError(job.error || "Failed");
    localStorage.removeItem("jobId");
    endJob();
  } else if (job.status === "cancelled") {
    showNotice(job.error || "Cancelled");
    localStorage.removeItem("jobId");
    endJob();
  } else {
    // created / separating / transcribing — reconnect SSE; queued events replay the gap.
    setProgress(0, job.status === "transcribing" ? "Transcribing…" : "Separating stems…");
    renderStoredMidi(job.midi, job.song_name);
    connectEvents(id);
  }
}

// ---------- stems ----------
// Custom player initialization
function initPlayers() {
  STEMS.forEach((stem) => {
    const card = document.querySelector(`input[data-stem="${stem}"]`).closest(".stem-card");
    const audio = card.querySelector("audio");
    if (!audio || !audio.src) return;

    const playBtn = card.querySelector(".play-btn");
    const fill = card.querySelector(".progress-fill");
    const track = card.querySelector(".progress-track");
    const timeCur = card.querySelector(".time-cur");
    const timeDur = card.querySelector(".time-dur");
    const volBtn = card.querySelector(".vol-btn");
    const volSlider = card.querySelector(".vol-slider");
    const dlBtn = card.querySelector(".dl-btn");

    // Play/pause
    const icoPlay = playBtn.querySelector(".ico-play");
    const icoPause = playBtn.querySelector(".ico-pause");
    playBtn.onclick = (e) => {
      e.preventDefault(); e.stopPropagation();
      if (audio.paused) {
        if (currentPlayingAudio && currentPlayingAudio !== audio) {
          currentPlayingAudio.pause();
        }
        audio.play();
        card.classList.add("playing");
        icoPlay.style.display = "none";
        icoPause.style.display = "";
        currentPlayingAudio = audio;
      } else {
        audio.pause();
        card.classList.remove("playing");
        icoPlay.style.display = "";
        icoPause.style.display = "none";
      }
    };

    // Progress update
    audio.ontimeupdate = () => {
      if (!audio.duration) return;
      const pct = (audio.currentTime / audio.duration) * 100;
      fill.style.width = pct + "%";
      track.style.setProperty("--fill", pct + "%");
      timeCur.textContent = fmtTime(audio.currentTime);
    };

    audio.onloadedmetadata = () => { timeDur.textContent = fmtTime(audio.duration); };

    audio.onended = () => {
      card.classList.remove("playing");
      icoPlay.style.display = "";
      icoPause.style.display = "none";
      fill.style.width = "0%";
      track.style.setProperty("--fill", "0%");
      timeCur.textContent = "0:00";
    };

    // Seek
    track.onclick = (e) => {
      e.preventDefault(); e.stopPropagation();
      if (!audio.duration) return;
      const rect = track.getBoundingClientRect();
      const pct = ((e.clientX - rect.left) / rect.width) * 100;
      fill.style.width = pct + "%";
      track.style.setProperty("--fill", pct + "%");
      audio.currentTime = (pct / 100) * audio.duration;
    };

    // Volume
    const volSVGs = {
      off: '<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 5.5h2.5L8 2.5v11l-3.5-3H2z" fill="currentColor" stroke="none"/><line x1="11" y1="5" x2="15" y2="11"/><line x1="15" y1="5" x2="11" y2="11"/></svg>',
      low: '<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 5.5h2.5L8 2.5v11l-3.5-3H2z" fill="currentColor" stroke="none"/><path d="M10.5 5.5c.8.7 1.2 1.6 1.2 2.5s-.4 1.8-1.2 2.5"/></svg>',
      hi: '<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 5.5h2.5L8 2.5v11l-3.5-3H2z" fill="currentColor" stroke="none"/><path d="M10.5 5.5c.8.7 1.2 1.6 1.2 2.5s-.4 1.8-1.2 2.5"/><path d="M12.5 3.5c1.4 1.3 2.2 3.1 2.2 4.5s-.8 3.2-2.2 4.5"/></svg>',
    };
    const volIcon = (vol) => { volBtn.innerHTML = vol === 0 ? volSVGs.off : vol < 0.5 ? volSVGs.low : volSVGs.hi; };
    volSlider.oninput = (e) => {
      e.stopPropagation();
      audio.volume = parseFloat(volSlider.value);
      volIcon(audio.volume);
      volBtn.classList.toggle("muted", audio.volume === 0);
    };

    volBtn.onclick = (e) => {
      e.preventDefault(); e.stopPropagation();
      if (audio.volume > 0) {
        audio._prevVol = audio.volume;
        audio.volume = 0; volSlider.value = 0;
        volIcon(0); volBtn.classList.add("muted");
      } else {
        audio.volume = audio._prevVol || 1; volSlider.value = audio.volume;
        volIcon(audio.volume);
        volBtn.classList.remove("muted");
      }
    };

    // Individual download
    dlBtn.onclick = (e) => {
      e.preventDefault(); e.stopPropagation();
      if (!currentJob) return;
      const a = document.createElement("a");
      a.href = `/output/${currentJob}/stems/${stem}.wav`;
      a.download = `${stem}.wav`; a.click();
    };
  });
}
function fmtTime(s) {
  if (!s || !isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m + ":" + (sec < 10 ? "0" : "") + sec;
}
function downloadAllStems() {
  if (!currentJob) return;
  const a = document.createElement("a");
  a.href = `/output/${currentJob}/stems`;
  a.click();
}
function showStems() {
  const remembered = settings.remember_selection
    ? (JSON.parse(localStorage.getItem("checkedStems") || "null") || DEFAULT_CHECKED)
    : DEFAULT_CHECKED;
  STEMS.forEach((stem) => {
    // data-stem lives on the card's checkbox input, not on the .stem-card
    // label; climb from the input to the card so the panel actually shows.
    const card = document.querySelector(`input[data-stem="${stem}"]`).closest(".stem-card");
    card.querySelector('input[type="checkbox"]').checked = remembered.includes(stem);
    card.querySelector("audio").src = `/output/${currentJob}/stems/${stem}.wav`;
    card.querySelector(".inst").value = settings.instrument_by_stem?.[stem] || "";
  });
  $("stem-panel").classList.remove("hidden");
  initPlayers();
}
function selectedStems() {
  return [...document.querySelectorAll('input[data-stem]:checked')].map((el) => el.dataset.stem);
}
async function transcribeSelected() {
  const stems = selectedStems();
  if (!stems.length) { showError("Select at least one stem."); return; }
  $("btn-transcribe").disabled = true; // guard against double-submit
  try {
  settings.instrument_by_stem = {};
  [...document.querySelectorAll('input[data-instrument]')].forEach((el) => {
    settings.instrument_by_stem[el.dataset.stem] = el.value.trim();
  });
  if (settings.remember_selection) localStorage.setItem("checkedStems", JSON.stringify(stems));
  await saveSettings();
  const resp = await fetch(`/api/jobs/${currentJob}/transcribe`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ stems }),
  });
  if (!resp.ok) throw new Error((await resp.json()).detail || "transcribe failed");
  setProgress(0, "Transcribing…");
  } catch (e) {
    showError(String(e.message || e));
    $("btn-transcribe").disabled = false;
  }
}

// ---------- results ----------
function addResult(stem, file) {
  if (renderedFiles.has(file)) return;
  renderedFiles.add(file);
  const link = document.createElement("a");
  link.href = `/output/${currentJob}/midi/${encodeURIComponent(file)}`;
  link.textContent = `${stem} · ${file}`;
  link.classList.add("result-link");
  $("results").appendChild(link);
  // show the "Download All MIDI" button once at least one result exists
  $("midi-actions").classList.remove("hidden");
}
function downloadAllMidi() {
  const links = $("results").querySelectorAll(".result-link");
  links.forEach((a, i) => {
    setTimeout(() => { a.click(); }, i * 150);
  });
}
function setConsoleState(state) {
  document.body.dataset.console = state;
  const jp = $("job-panel");
  if (jp) jp.dataset.state = state;
}
function setProgress(pct, text) {
  $("progress-bar").style.width = `${pct}%`;
  $("progress-text").textContent = text;
  $("job-status").textContent = text;
  const p = $("progress-pct");
  if (p) p.textContent = `${Math.round(pct)}%`;
  setConsoleState(pct >= 100 ? "done" : "running");
}
function showError(message) {
  const el = $("job-error");
  el.classList.remove("notice");
  el.classList.add("error");
  el.textContent = message;
  el.classList.remove("hidden");
  setConsoleState("error");
}
function showNotice(message) {
  const el = $("job-error");
  el.classList.remove("error");
  el.classList.add("notice");
  el.textContent = message;
  el.classList.remove("hidden");
  $("job-status").textContent = message;
  setConsoleState("cancelled");
}

$("btn-save-settings").addEventListener("click", saveSettings);
$("btn-download-midi").addEventListener("click", downloadAllMidi);
$("btn-cancel").addEventListener("click", async () => {
  if (!currentJob) return;
  $("btn-cancel").disabled = true;
  try {
    await fetch(`/api/jobs/${currentJob}/cancel`, { method: "POST" });
    $("job-status").textContent = "Cancelling…";
    setConsoleState("cancelled");
  } catch {
    $("btn-cancel").disabled = false; // fetch failed; allow a retry
  }
});
$("btn-transcribe").addEventListener("click", transcribeSelected);
$("btn-download-all").addEventListener("click", downloadAllStems);
loadSettings().then(resumeJob); // restore a job left running across a reload
loadInstruments();
setupDropzone();
loadHardware();
