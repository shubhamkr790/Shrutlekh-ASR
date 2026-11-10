// Shrutlekh web client. Plain JavaScript, no build step.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const state = {
    source: null,        // {kind: "file"|"sample", file?: File, name?: string, url: string}
    transcript: null,    // last result from the API (edited in place)
    flagged: [],         // word elements with level medium/low, in order
    flagIndex: -1,
    wordEls: [],         // [{el, word}]
    peaks: null,
    stopAt: null,        // stop playback here (used when replaying one word)
  };

  // ---------- helpers ----------
  const fmtTime = (s) => {
    s = Math.max(0, s || 0);
    return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  };
  const pct = (v) => `${Math.round(v * 100)}%`;
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  function setStatus(msg, kind = "") {
    $("status").className = "status " + kind;
    $("status").innerHTML = msg;
  }

  // ---------- health + samples ----------
  fetch("/api/health").then((r) => r.json()).then((h) => {
    $("health").textContent = `backend ok · v${h.version}`;
    $("health").classList.add("ok");
  }).catch(() => {
    $("health").textContent = "backend offline";
    $("health").classList.add("bad");
  });

  fetch("/api/samples").then((r) => r.json()).then((list) => {
    const ul = $("samples");
    ul.innerHTML = "";
    if (!list.length) { ul.innerHTML = '<li class="muted">No samples found.</li>'; return; }
    list.forEach((s, i) => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      b.innerHTML = `<span>${esc(s.title)}</span><small>${String(i + 1).padStart(2, "0")}</small>`;
      b.title = s.reference;
      b.onclick = () => {
        document.querySelectorAll(".samples button").forEach((x) => x.classList.remove("on"));
        b.classList.add("on");
        pick({ kind: "sample", name: s.file, label: s.title, url: `/api/samples/${encodeURIComponent(s.file)}` });
      };
      li.appendChild(b);
      ul.appendChild(li);
    });
    // Demo links: /?sample=04_medicine.wav&model=base transcribes a sample straight away
    const q = new URLSearchParams(location.search);
    const wanted = q.get("sample");
    if (wanted) {
      if (q.get("model")) $("model").value = q.get("model");
      const i = list.findIndex((s) => s.file === wanted);
      if (i >= 0) { ul.querySelectorAll("button")[i].click(); $("go").click(); }
    }
  });

  // ---------- choosing audio ----------
  function pick(src) {
    if (state.source && state.source.kind === "file") URL.revokeObjectURL(state.source.url);
    state.source = src;
    $("picked").hidden = false;
    $("picked").innerHTML = `<b>${esc(src.label)}</b><span class="muted">${src.kind === "sample" ? "sample" : sizeOf(src.file)}</span>`;
    $("go").disabled = false;
    setStatus("");
  }
  const sizeOf = (f) => (f.size > 1e6 ? `${(f.size / 1e6).toFixed(1)} MB` : `${Math.ceil(f.size / 1e3)} KB`);
  function pickFile(file) {
    if (!file) return;
    document.querySelectorAll(".samples button").forEach((x) => x.classList.remove("on"));
    pick({ kind: "file", file, label: file.name, url: URL.createObjectURL(file) });
  }
  $("file").onchange = (e) => pickFile(e.target.files[0]);
  const drop = $("drop");
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));

  // ---------- recording ----------
  let recorder = null, chunks = [], recStart = 0, recTimer = null;
  $("rec").onclick = async () => {
    if (recorder && recorder.state === "recording") { recorder.stop(); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunks = [];
      recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        clearInterval(recTimer);
        $("rec").classList.remove("recording");
        $("recLabel").textContent = "Record from mic";
        const type = recorder.mimeType || "audio/webm";
        const ext = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "m4a" : "webm";
        pickFile(new File(chunks, `recording-${new Date().toISOString().slice(11, 19).replace(/:/g, "")}.${ext}`, { type }));
      };
      recorder.start();
      recStart = Date.now();
      $("rec").classList.add("recording");
      $("recLabel").textContent = "Stop recording";
      recTimer = setInterval(() => { $("recTime").textContent = fmtTime((Date.now() - recStart) / 1000); }, 250);
    } catch (err) {
      setStatus("Microphone not available: " + esc(err.message), "err");
    }
  };

  // ---------- thresholds ----------
  const syncThresholds = () => {
    if (+$("low").value > +$("high").value) $("low").value = $("high").value;
    $("highOut").textContent = pct(+$("high").value);
    $("lowOut").textContent = pct(+$("low").value);
  };
  $("high").oninput = $("low").oninput = syncThresholds;

  // ---------- transcribe ----------
  $("go").onclick = async () => {
    const src = state.source;
    if (!src) return;
    const opts = { model: $("model").value, language: $("language").value, high: $("high").value, low: $("low").value };
    $("go").disabled = true;
    const started = performance.now();
    const tick = setInterval(() => setStatus(`<span class="spinner"></span>Transcribing with ${opts.model}… ${((performance.now() - started) / 1000).toFixed(0)}s`), 200);
    try {
      let res;
      if (src.kind === "sample") {
        res = await fetch(`/api/transcribe/sample/${encodeURIComponent(src.name)}?${new URLSearchParams(opts)}`, { method: "POST" });
      } else {
        const fd = new FormData();
        fd.append("file", src.file);
        Object.entries(opts).forEach(([k, v]) => fd.append(k, v));
        res = await fetch("/api/transcribe", { method: "POST", body: fd });
      }
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.statusText);
      clearInterval(tick);
      setStatus(`Done in ${data.processing_seconds.toFixed(1)}s for ${data.duration.toFixed(1)}s of audio.`);
      await show(data, src.url);
    } catch (err) {
      clearInterval(tick);
      setStatus("Could not transcribe: " + esc(err.message), "err");
    } finally {
      $("go").disabled = false;
    }
  };

  // ---------- render result ----------
  async function show(t, audioUrl) {
    state.transcript = t;
    $("empty").hidden = true;
    $("out").hidden = false;
    renderStats();
    renderEvaluation(t);
    renderTranscript();
    $("audio").src = audioUrl;
    await loadPeaks(audioUrl);
    drawWave();
  }

  function recount() {
    const words = state.transcript.segments.flatMap((s) => s.words);
    const sm = state.transcript.summary;
    sm.words = words.length;
    sm.high = words.filter((w) => w.level === "high").length;
    sm.medium = words.filter((w) => w.level === "medium").length;
    sm.low = words.filter((w) => w.level === "low").length;
    sm.edited_words = words.filter((w) => w.edited).length;
    sm.review_ratio = words.length ? (sm.medium + sm.low) / words.length : 0;
  }

  function renderStats() {
    const t = state.transcript, s = t.summary;
    const rtf = t.processing_seconds / Math.max(t.duration, 0.01);
    $("stats").innerHTML = [
      ["", s.words, "words"],
      ["medium", s.medium, "to check"],
      ["low", s.low, "likely wrong"],
      ["", pct(s.review_ratio), "of words to review"],
      ["", s.edited_words || 0, "corrected"],
      ["", `${t.language} · ${pct(t.language_probability)}`, "language"],
      ["", `${rtf.toFixed(2)}×`, "real-time factor"],
    ].map(([cls, v, label]) => `<div class="stat ${cls}"><b>${esc(v)}</b><span>${label}</span></div>`).join("");
  }

  function renderEvaluation(t) {
    const box = $("evaluation");
    if (!t.evaluation) { box.hidden = true; return; }
    const e = t.evaluation;
    box.hidden = false;
    box.innerHTML = `<b>Compared with the correct text:</b> word error rate <b>${pct(e.wer)}</b>
      (${e.substitutions} wrong, ${e.deletions} missed, ${e.insertions} extra out of ${e.reference_words} words),
      character error rate ${pct(e.cer)}.<br><span class="ref">Reference: “${esc(t.reference)}”</span>`;
  }

  const FLAG_TEXT = {
    repetitive_text: "repeats itself (possible made-up text)",
    low_average_confidence: "low confidence overall",
    possibly_no_speech: "may be silence or noise",
  };

  function renderTranscript() {
    const box = $("transcript");
    box.innerHTML = "";
    state.wordEls = [];
    state.transcript.segments.forEach((seg) => {
      const row = document.createElement("div");
      row.className = "seg-row";
      const time = document.createElement("time");
      time.textContent = fmtTime(seg.start);
      time.onclick = () => playFrom(seg.start);
      const text = document.createElement("div");
      text.className = "seg-text";
      seg.words.forEach((w) => {
        const lead = w.text.match(/^\s*/)[0];
        if (lead) text.appendChild(document.createTextNode(lead));
        const el = document.createElement("span");
        el.className = `w ${w.edited ? "edited" : w.level}`;
        el.textContent = w.text.trim();
        el.title = `${pct(w.probability)} sure · ${w.start.toFixed(2)}s`;
        el.onclick = () => { if (el.isContentEditable) return; playWord(w); focusWord(el); };
        el.ondblclick = () => editWord(el, w);
        text.appendChild(el);
        state.wordEls.push({ el, word: w });
      });
      if (seg.flags.length) {
        const f = document.createElement("span");
        f.className = "seg-flags";
        f.textContent = "⚠ " + seg.flags.map((x) => FLAG_TEXT[x] || x).join(" · ");
        text.appendChild(f);
      }
      row.dataset.flagged = seg.flags.length || seg.words.some((w) => w.level !== "high") ? "1" : "";
      row.append(time, text);
      box.appendChild(row);
    });
    applyFilter();
    refreshFlagged();
  }

  function refreshFlagged() {
    state.flagged = state.wordEls.filter(({ word }) => word.level !== "high" && !word.edited).map(({ el }) => el);
    $("prevFlag").disabled = $("nextFlag").disabled = state.flagged.length === 0;
    $("nextFlag").textContent = state.flagged.length ? `Next flagged → (${state.flagged.length})` : "Nothing to check ✓";
  }

  function applyFilter() {
    const only = $("onlyFlagged").checked;
    document.querySelectorAll(".seg-row").forEach((r) => r.classList.toggle("hidden", only && !r.dataset.flagged));
  }
  $("onlyFlagged").onchange = applyFilter;

  // ---------- editing ----------
  function editWord(el, w) {
    const before = el.textContent;
    el.contentEditable = "true";
    el.focus();
    document.getSelection().selectAllChildren(el);
    const finish = (save) => {
      el.contentEditable = "false";
      el.onblur = el.onkeydown = null;
      const after = el.textContent.trim();
      if (!save || !after) { el.textContent = before; return; }
      if (after !== before || w.level !== "high") {
        const lead = w.text.match(/^\s*/)[0];
        w.text = lead + after;
        w.edited = true;
        w.level = "high";
        el.className = "w edited";
        recount();
        renderStats();
        refreshFlagged();
        drawWave();
      }
    };
    el.onkeydown = (e) => {
      if (e.key === "Enter") { e.preventDefault(); finish(true); }
      if (e.key === "Escape") { e.preventDefault(); finish(false); }
    };
    el.onblur = () => finish(true);
  }

  // ---------- playback ----------
  const audio = $("audio");
  function playFrom(t) { state.stopAt = null; audio.currentTime = t; audio.play(); }
  function playWord(w) {
    audio.currentTime = Math.max(0, w.start - 0.25);
    state.stopAt = w.end + 0.25;
    audio.play();
  }
  function focusWord(el) {
    document.querySelectorAll(".w.focus").forEach((x) => x.classList.remove("focus"));
    el.classList.add("focus");
    el.scrollIntoView({ block: "center", behavior: "smooth" });
  }
  function jumpFlag(dir) {
    if (!state.flagged.length) return;
    state.flagIndex = (state.flagIndex + dir + state.flagged.length) % state.flagged.length;
    const el = state.flagged[state.flagIndex];
    const entry = state.wordEls.find((x) => x.el === el);
    focusWord(el);
    playWord(entry.word);
  }
  $("nextFlag").onclick = () => jumpFlag(1);
  $("prevFlag").onclick = () => jumpFlag(-1);
  $("play").onclick = () => (audio.paused ? (state.stopAt = null, audio.play()) : audio.pause());
  audio.onloadedmetadata = () => { $("clock").textContent = `0:00 / ${fmtTime(audio.duration)}`; };
  audio.onplay = () => $("playIcon").setAttribute("d", "M6 5h4v14H6zM14 5h4v14h-4z");
  audio.onpause = () => $("playIcon").setAttribute("d", "M7 5v14l12-7z");
  audio.ontimeupdate = () => {
    const t = audio.currentTime;
    if (state.stopAt !== null && t >= state.stopAt) { audio.pause(); state.stopAt = null; }
    $("clock").textContent = `${fmtTime(t)} / ${fmtTime(audio.duration)}`;
    $("cursor").style.left = `${(t / (audio.duration || 1)) * 100}%`;
    state.wordEls.forEach(({ el, word }) => el.classList.toggle("now", !audio.paused && t >= word.start && t < word.end));
  };
  document.addEventListener("keydown", (e) => {
    if (!state.transcript || e.target.isContentEditable || /INPUT|SELECT|TEXTAREA|BUTTON/.test(e.target.tagName)) return;
    if (e.key === " ") { e.preventDefault(); $("play").click(); }
    if (e.key === "n" || e.key === "N") jumpFlag(1);
    if (e.key === "p" || e.key === "P") jumpFlag(-1);
  });

  // ---------- waveform with flag markers ----------
  async function loadPeaks(url) {
    state.peaks = null;
    try {
      const buf = await (await fetch(url)).arrayBuffer();
      // an offline context decodes without needing a sound card (works in headless browsers too)
      const decoded = await new OfflineAudioContext(1, 1, 16000).decodeAudioData(buf);
      const data = decoded.getChannelData(0), n = 600, step = Math.max(1, Math.floor(data.length / n));
      const peaks = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let m = 0;
        for (let j = i * step; j < Math.min((i + 1) * step, data.length); j++) m = Math.max(m, Math.abs(data[j]));
        peaks[i] = m;
      }
      const max = Math.max(...peaks) || 1;
      state.peaks = { values: peaks.map((p) => p / max), duration: decoded.duration };
    } catch { /* waveform is optional; playback still works */ }
  }

  function drawWave() {
    const c = $("wave"), dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = 64;
    c.width = w * dpr; c.height = h * dpr;
    const g = c.getContext("2d");
    g.scale(dpr, dpr);
    g.clearRect(0, 0, w, h);
    const dur = state.peaks?.duration || state.transcript?.duration || 1;
    // flagged regions first, so the waveform sits on top
    state.wordEls.forEach(({ word }) => {
      if (word.level === "high" || word.edited) return;
      g.fillStyle = word.level === "low" ? "rgba(200,64,26,.28)" : "rgba(242,162,12,.30)";
      const x = (word.start / dur) * w, x2 = (word.end / dur) * w;
      g.fillRect(x, 0, Math.max(3, x2 - x), h);
      g.fillStyle = word.level === "low" ? "#c8401a" : "#f2a20c";
      g.fillRect(x, h - 4, Math.max(3, x2 - x), 4);
    });
    if (state.peaks) {
      const vals = state.peaks.values, bw = w / vals.length;
      g.fillStyle = "#17203d";
      vals.forEach((v, i) => {
        const bh = Math.max(1, v * (h - 12));
        g.fillRect(i * bw, (h - 4) / 2 - bh / 2, Math.max(1, bw - 0.6), bh);
      });
    }
  }
  window.addEventListener("resize", () => state.transcript && drawWave());
  document.querySelector(".wave").onclick = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    playFrom(((e.clientX - r.left) / r.width) * (audio.duration || state.transcript.duration));
  };

  // ---------- export ----------
  document.querySelectorAll(".exports [data-fmt]").forEach((b) => {
    b.onclick = async () => {
      const fmt = b.dataset.fmt;
      const res = await fetch(`/api/export?format=${fmt}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(state.transcript),
      });
      if (!res.ok) { setStatus("Export failed.", "err"); return; }
      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `shrutlekh-${state.transcript.id || "transcript"}.${fmt}`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    };
  });
})();
