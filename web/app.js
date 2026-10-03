"use strict";
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (sec) => {
  sec = Math.round(sec || 0);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(s).padStart(2, "0");
};
const span = (a, b) => `${fmt(a)}–${fmt(b)}`;
const highlight = (text, terms) => {
  let h = esc(text);
  for (const t of terms || []) if (t) h = h.split(esc(t)).join(`<mark>${esc(t)}</mark>`);
  return h;
};
const api = (p, opt) => fetch(p, opt).then((r) => r.json());

const EXAMPLES = ["用简体中文描述这段画面", "视频里出现了哪些店铺？", "哪一段有人撑伞？", "有没有看到施工的地方？在什么时候？"];
let VIDEOS = [];
const scope = new Set();

/* ------------------------------------------------------------------ tabs */
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function showTab(name) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
}

/* ---------------------------------------------------------------- status */
async function loadStatus() {
  try {
    const s = await api("/api/status");
    const pills = [];
    const vlm = s.vlm || {};
    pills.push([`画面理解 ${vlm.model_id ? "✓" : "✗"}`, !!vlm.model_id, vlm.model_id || vlm.error]);
    pills.push([`问答 ${typeof s.llm === "string" ? "（由画面模型兼任）" : "✓"}`, true, typeof s.llm === "string" ? s.llm : s.llm.model_id]);
    const p = s.perception || {};
    pills.push([`听声音 ${p.asr ? "✓" : "关"}`, !!p.asr, "SenseVoice 语音识别"]);
    pills.push([`读文字 ${p.ocr ? "✓" : "关"}`, !!p.ocr, "PP-OCR 画面文字识别"]);
    pills.push([`语义检索 ${typeof s.embedding === "string" ? "关" : "✓"}`, typeof s.embedding !== "string", "向量检索（可选）"]);
    if (vlm.max_prefill) pills.push([`单次可读约 ${vlm.max_prefill} token`, true, "模型单次能处理的最大长度，自动读取"]);
    $("#pills").innerHTML = pills.map(([t, on, tip]) => `<span class="pill ${on ? "on" : "off"}" title="${esc(tip || "")}">${esc(t)}</span>`).join("");
  } catch (e) {
    $("#pills").innerHTML = `<span class="pill off">服务未连接</span>`;
  }
}

/* ---------------------------------------------------------------- videos */
async function loadVideos(select) {
  VIDEOS = await api("/api/videos");
  $("#videos").innerHTML = VIDEOS.length ? VIDEOS.map((v) => `
    <div class="vcard" data-name="${esc(v.name)}">
      ${v.thumb ? `<img src="${v.thumb}" loading="lazy">` : `<div style="aspect-ratio:16/9;background:#000"></div>`}
      <div class="b"><div class="n" title="${esc(v.name)}">${esc(v.name)}</div>
        <div class="tags"><span class="tag">${fmt(v.duration)}</span><span class="tag">${v.segments} 段</span>
        ${v.chapters ? `<span class="tag">${v.chapters} 章</span>` : ""}
        ${v.has_speech ? `<span class="tag">有语音</span>` : ""}${v.has_ocr ? `<span class="tag">有文字</span>` : ""}</div>
      </div></div>`).join("") : `<div class="muted">还没有视频，先上传一个吧。</div>`;
  document.querySelectorAll(".vcard").forEach((c) => c.addEventListener("click", () => openVideo(c.dataset.name)));
  renderScope();
  if (select) openVideo(select);
}

function renderScope() {
  for (const n of [...scope]) if (!VIDEOS.find((v) => v.name === n)) scope.delete(n);
  $("#scope").innerHTML = VIDEOS.length
    ? `<span class="muted small">在哪些视频里找：</span>` +
      `<span class="chip ${scope.size ? "" : "sel"}" data-all="1">全部</span>` +
      VIDEOS.map((v) => `<span class="chip ${scope.has(v.name) ? "sel" : ""}" data-v="${esc(v.name)}">${esc(v.name)}</span>`).join("")
    : `<span class="muted small">还没有视频，先到「视频库」上传。</span>`;
  $("#scope").querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => {
    if (c.dataset.all) scope.clear(); else scope.has(c.dataset.v) ? scope.delete(c.dataset.v) : scope.add(c.dataset.v);
    renderScope();
  }));
}

function segCard(s, isNew) {
  return `<div class="seg ${isNew ? "new" : ""}" data-t="${s.start}">
    ${s.thumb ? `<img src="${s.thumb}" loading="lazy">` : ""}
    <div class="b"><div class="t">${span(s.start, s.end)}</div>
      <div class="c">👁 ${esc(s.caption || "（没有画面描述）")}</div>
      ${s.ocr ? `<div class="o">🔤 ${esc(s.ocr)}</div>` : ""}
      ${s.transcript ? `<div class="a">🎤 ${esc(s.transcript)}</div>` : ""}</div></div>`;
}

function timelineHTML(duration, bounds, state, thumbs, head, chapters) {
  const segs = bounds.map(([a, b], i) => {
    const bg = thumbs[i] ? `background-image:url('${thumbs[i]}')` : "";
    return `<div class="tl-seg ${state[i] || ""}" data-t="${a}" style="flex:${Math.max(b - a, 0.1)};${bg}" title="${span(a, b)}"></div>`;
  }).join("");
  let chap = "";
  if (chapters && chapters.length) {
    let t = 0;
    const cells = [];
    for (const c of chapters) {
      if (c.start > t) cells.push(`<div style="flex:${c.start - t};background:none"></div>`);
      cells.push(`<div style="flex:${Math.max(c.end - c.start, 0.1)}" title="${esc(c.text)}">📖 ${esc(c.text)}</div>`);
      t = c.end;
    }
    chap = `<div class="tl-chap">${cells.join("")}</div>`;
  }
  const hd = head == null ? "" : `<div class="tl-head" style="left:${(100 * head) / Math.max(duration, 1)}%"></div>`;
  const axis = [0, 1, 2, 3, 4].map((k) => `<span>${fmt((duration * k) / 4)}</span>`).join("");
  return `<div class="tl-row">${segs}${hd}</div>${chap}<div class="tl-axis">${axis}</div>`;
}

async function openVideo(name) {
  document.querySelectorAll(".vcard").forEach((c) => c.classList.toggle("sel", c.dataset.name === name));
  const d = await api("/api/videos/" + encodeURIComponent(name));
  if (d.detail) return;
  $("#detail").classList.remove("hidden");
  $("#d-title").innerHTML = `${esc(d.name)} <span class="muted small">${fmt(d.duration)} · ${d.segments} 段</span>
    <button class="del" id="d-del">删除</button>`;
  const player = $("#d-player");
  if (player.dataset.src !== d.video_url) { player.src = d.video_url; player.dataset.src = d.video_url; }
  const segs = d.segments_list;
  const chapters = (d.levels && d.levels[0]) || [];
  $("#d-chapters").innerHTML = chapters.length
    ? chapters.map((c) => `<div class="chapter" data-t="${c.start}"><b>${span(c.start, c.end)}</b>${esc(c.text)}</div>`).join("")
    : `<div class="muted small">（无）</div>`;
  const thumbs = {}; segs.forEach((s, i) => { if (s.thumb) thumbs[i] = s.thumb; });
  $("#d-tl").innerHTML = timelineHTML(d.duration, segs.map((s) => [s.start, s.end]), Object.fromEntries(segs.map((_, i) => [i, "ok"])), thumbs, null, chapters);
  $("#d-segs").innerHTML = segs.map((s) => segCard(s)).join("");
  $("#detail").querySelectorAll("[data-t]").forEach((el) => el.addEventListener("click", () => {
    player.currentTime = +el.dataset.t; player.play();
  }));
  $("#d-del").addEventListener("click", async () => {
    if (!confirm(`删除「${name}」的索引？（原视频文件不会删除）`)) return;
    await fetch("/api/videos/" + encodeURIComponent(name), { method: "DELETE" });
    $("#detail").classList.add("hidden");
    loadVideos();
  });
}

/* --------------------------------------------------------------- upload */
const drop = $("#drop");
$("#file").addEventListener("change", (e) => uploadFiles([...e.target.files]));
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => uploadFiles([...e.dataTransfer.files].filter((f) => f.type.startsWith("video/") || /\.(mp4|mov|mkv|avi|webm|flv|ts)$/i.test(f.name))));

async function uploadFiles(files) {
  for (const f of files) {
    const job = await uploadOne(f);
    if (job) await watchJob(job);
  }
  $("#file").value = "";
}

function uploadOne(file) {
  return new Promise((resolve) => {
    const bar = $("#upbar"), msg = $("#upmsg");
    bar.classList.remove("hidden");
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload?name=" + encodeURIComponent(file.name));
    xhr.upload.onprogress = (e) => {
      if (!e.lengthComputable) return;
      $("span", bar).style.width = (100 * e.loaded / e.total) + "%";
      msg.textContent = `正在上传 ${file.name}：${(e.loaded / 1048576).toFixed(1)} / ${(e.total / 1048576).toFixed(1)} MB`;
    };
    xhr.onload = () => {
      bar.classList.add("hidden");
      try { const r = JSON.parse(xhr.responseText); msg.textContent = r.job ? "上传完成，AI 开始看视频…" : "上传失败：" + r.detail; resolve(r.job); }
      catch { msg.textContent = "上传失败"; resolve(null); }
    };
    xhr.onerror = () => { msg.textContent = "上传失败（网络错误）"; bar.classList.add("hidden"); resolve(null); };
    xhr.send(file);
  });
}

/* ------------------------------------------------ watch-along animation */
const W = { bounds: [], duration: 0, state: {}, thumbs: {}, chapters: [], cur: null, cycle: null, typing: null };

function typewrite(el, text, cps = 40) {
  clearInterval(W.typing);
  let i = 0;
  el.textContent = "";
  el.classList.add("cursor");
  W.typing = setInterval(() => {
    i += 2;
    el.textContent = text.slice(0, i);
    if (i >= text.length) { clearInterval(W.typing); el.classList.remove("cursor"); }
  }, 1000 / (cps / 2));
}

function showFrames(frames, s) {
  const scr = $("#w-screen");
  scr.classList.remove("idle");
  scr.querySelectorAll("img").forEach((i) => i.remove());
  frames.forEach((f, i) => {
    const img = document.createElement("img");
    img.src = f;
    if (i === 0) img.classList.add("show");
    scr.insertBefore(img, scr.firstChild);
  });
  clearInterval(W.cycle);
  const imgs = [...scr.querySelectorAll("img")].reverse();
  let k = 0;
  W.cycle = setInterval(() => {
    imgs[k].classList.remove("show");
    k = (k + 1) % imgs.length;
    imgs[k].classList.add("show");
    const t = s.start + ((k + 0.5) * (s.end - s.start)) / imgs.length;
    $("#w-hud").innerHTML = `AI 正在看 ${span(s.start, s.end)}<br>第 ${k + 1}/${imgs.length} 帧 · ${fmt(t)}`;
  }, 700);
  $("#w-hud").innerHTML = `AI 正在看 ${span(s.start, s.end)}<br>第 1/${frames.length} 帧`;
}

function drawWatchTimeline() {
  const head = W.cur ? W.cur.start : null;
  $("#w-tl").innerHTML = timelineHTML(W.duration, W.bounds, W.state, W.thumbs, head, W.chapters);
}

function watchJob(job) {
  return new Promise((resolve) => {
    showTab("library");
    const box = $("#watch");
    box.classList.remove("hidden");
    $("#w-feed").innerHTML = "";
    $("#w-notes").innerHTML = `<div class="muted">等 AI 看完第一段，这里会写下它看到、听到和读到的内容。</div>`;
    $("#w-screen").classList.add("idle");
    Object.assign(W, { bounds: [], duration: 0, state: {}, thumbs: {}, chapters: [], cur: null });
    const es = new EventSource(`/api/jobs/${job}/events`);
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data);
      if (ev.frac != null) $("#w-prog").style.width = (100 * ev.frac) + "%";
      if (ev.msg) $("#w-msg").textContent = ev.msg;
      switch (ev.type) {
        case "plan":
          W.bounds = ev.bounds; W.duration = ev.duration;
          $("#w-title").textContent = `AI 正在看「${ev.name}」`;
          $("#w-sub").textContent = `${fmt(ev.duration)}，分成 ${ev.bounds.length} 段逐段观看`;
          drawWatchTimeline();
          break;
        case "watching":
          for (const k in W.state) if (W.state[k] === "cur") delete W.state[k];  // finishing in background
          W.cur = ev.segment; W.state[ev.segment.index] = "cur";
          showFrames(ev.segment.frames, ev.segment);
          drawWatchTimeline();
          break;
        case "segment": {
          const s = ev.segment;
          W.state[s.index] = "ok"; if (s.thumb) W.thumbs[s.index] = s.thumb;
          drawWatchTimeline();
          const ocr = s.ocr ? s.ocr.split("；").map((t) => `<span class="ocrchip">${esc(t)}</span>`).join("") : "";
          $("#w-notes").innerHTML = `
            <div class="note-h"><span class="t">${span(s.start, s.end)}</span>AI 看到了：</div>
            <div id="w-type"></div>
            ${ocr ? `<div class="note-sec"><div class="l">🔤 画面里的文字</div><div class="ocrchips">${ocr}</div></div>` : ""}
            ${s.transcript ? `<div class="note-sec"><div class="l">🎤 听到的声音</div>${esc(s.transcript)}</div>` : ""}`;
          typewrite($("#w-type"), s.caption || "（没有画面描述）");
          $("#w-feed").insertAdjacentHTML("afterbegin", segCard(s, true));
          break;
        }
        case "chapter":
          if (ev.chapter.level === 1) { W.chapters.push(ev.chapter); W.chapters.sort((a, b) => a.start - b.start); drawWatchTimeline(); }
          break;
        case "finished":
          clearInterval(W.cycle); W.cur = null; drawWatchTimeline();
          $("#w-screen").classList.add("idle");
          $("#w-title").textContent = "看完了"; $("#w-hud").textContent = "完成";
          loadVideos(ev.name);
          break;
        case "error":
          clearInterval(W.cycle); $("#w-screen").classList.add("idle");
          $("#w-title").textContent = "出错了"; $("#w-msg").textContent = ev.msg;
          break;
        case "end":
          es.close(); resolve(); break;
      }
    };
    es.onerror = () => { es.close(); resolve(); };
  });
}

/* ------------------------------------------------------------------ ask */
EXAMPLES.forEach((q) => $("#examples").insertAdjacentHTML("beforeend", `<span class="chip">${esc(q)}</span>`));
$("#examples").querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => { $("#q").value = c.textContent; ask(); }));
$("#askform").addEventListener("submit", (e) => { e.preventDefault(); ask(); });

const METHOD_TEXT = {
  keyword: ["在视频里直接找到了关键词", "在画面描述、画面里的文字和语音中逐段比对问题里的关键词。"],
  chapter: ["关键词没有直接出现，让 AI 按章节判断", "AI 先读了每一章的摘要，挑出最可能相关的章节，再看其中的片段。"],
  vector: ["按意思相近程度查找", "把问题和每段内容的含义做比较，挑出最接近的片段。"],
};

let ES = null;
function ask() {
  const q = $("#q").value.trim();
  if (!q) return;
  if (ES) ES.close();
  const steps = [];
  let terms = [], answer = "", refs = [], items = [];
  LAST_TERMS = [];
  const setStep = (key, state, title, body) => {
    let s = steps.find((x) => x.key === key);
    if (!s) { s = { key }; steps.push(s); }
    Object.assign(s, { state, title, body: body ?? s.body });
    $("#steps").innerHTML = steps.map((x, i) => `<li class="step ${x.state}"><span class="dot">${x.state === "done" ? "✓" : x.state === "fail" ? "!" : i + 1}</span>
      <h4>${x.title}</h4><div class="body">${x.body || ""}</div></li>`).join("");
    $("#steps").querySelectorAll("[data-t]").forEach((el) => el.addEventListener("click", () => playRef({ video: el.dataset.v, start: +el.dataset.t, end: +el.dataset.e })));
  };
  const renderAnswer = (final) => {
    const html = esc(answer).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\n/g, "<br>")
      .replace(/\[(\d{1,3})\]/g, (m, n) => `<button class="cite" data-n="${n}">${n}</button>`);
    $("#answer").innerHTML = html || (final ? "（没有生成回答）" : "");
    $("#answer").classList.toggle("cursor", !final);
    $("#answer").classList.remove("muted");
    $("#answer").querySelectorAll(".cite").forEach((b) => b.addEventListener("click", () => {
      const it = items[+b.dataset.n - 1]; if (it) playRef(it);
    }));
  };

  $("#askbtn").disabled = true;
  $("#answer").innerHTML = `<span class="muted">正在查找…</span>`;
  $("#playercard").classList.add("hidden");
  setStep("understand", "run", "理解问题", "");
  const params = new URLSearchParams({ q });
  if (scope.size) params.set("videos", [...scope].join(","));
  ES = new EventSource("/api/ask?" + params);
  ES.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    switch (ev.type) {
      case "status":
        const run = steps.find((x) => x.state === "run");
        if (run && run.key === "answer") setStep("answer", "run", "AI 生成回答", esc(ev.text));
        break;
      case "plan": {
        const g = ev.global;
        setStep("understand", "done", "理解问题", g
          ? "这是一个<b>概括类</b>问题，会用整段视频的内容来回答。"
          : `在 ${ev.videos} 个视频、共 ${ev.segments} 段内容里查找。`);
        if (ev.mode === "full") setStep("search", "done", "查找相关画面", "视频内容不多，AI 直接通读全部片段，不需要检索。");
        else if (ev.mode === "summarize") setStep("search", "run", "查找相关画面", g ? "使用 AI 看视频时整理好的章节摘要。" : "没找到明确相关的片段，改用整段视频的章节摘要来回答。");
        else setStep("search", "run", "查找相关画面", "");
        break;
      }
      case "hits": {
        terms = ev.terms || []; LAST_TERMS = terms;
        if (terms.length) setStep("understand", "done", "理解问题",
          `问题里的关键词：<div class="terms">${terms.map((t) => `<span class="term">${esc(t)}</span>`).join("")}</div>`);
        const mt = METHOD_TEXT[ev.method] || ["查找相关画面", ""];
        const used = ev.items.filter((it) => it.score >= ev.threshold);
        const cards = (ev.used ? used : ev.items.slice(0, 4)).slice(0, 6).map((it) => `
          <div class="mini ${ev.used ? "hit" : ""}" data-v="${esc(it.video)}" data-t="${it.start}" data-e="${it.end}">
            ${it.thumb ? `<img src="${it.thumb}" loading="lazy">` : ""}
            <div class="t"><span>${span(it.start, it.end)}</span></div>
            <div class="x">${it.ocr ? "🔤 " + highlight(it.ocr, terms) + "<br>" : ""}${highlight(it.text, terms)}${it.transcript ? "<br>🎤 " + highlight(it.transcript, terms) : ""}</div></div>`).join("");
        setStep("search", ev.used ? "done" : "done", ev.used ? mt[0] : "没有找到明确相关的片段",
          `${mt[1]}${ev.used ? `找到 ${used.length} 段：` : "最接近的几段（相关度不够）："}<div class="minis">${cards}</div>`);
        break;
      }
      case "condensed":
        setStep("search", "done", "查找相关画面", `使用章节摘要（${ev.total} 章）。`);
        break;
      case "refined":
        setStep("rewatch", "done", "AI 带着问题重新看了一遍最相关的片段", esc(ev.item.text));
        break;
      case "context": {
        items = ev.items;
        const pct = Math.min(100, Math.round((100 * ev.tokens) / Math.max(1, ev.budget)));
        const chips = ev.items.map((it, i) => `<span class="chip" data-v="${esc(it.video)}" data-t="${it.start}" data-e="${it.end}">[${i + 1}] ${span(it.start, it.end)}</span>`).join(" ");
        setStep("context", "done", `整理出 ${ev.items.length} 条资料交给 AI`,
          `AI 一次能读的内容有限，本次用了约 ${pct}%。<div class="meter"><span style="width:${pct}%"></span></div><div class="row wrap gap8">${chips}</div>`);
        setStep("answer", "run", "AI 生成回答", "");
        break;
      }
      case "delta":
        answer = ev.answer; renderAnswer(false); break;
      case "done":
        answer = ev.answer; refs = ev.refs || []; renderAnswer(true);
        setStep("answer", "done", "AI 生成回答", "完成");
        showRefs(refs);
        break;
      case "error":
        $("#answer").innerHTML = `<span style="color:#ef4444">⚠ ${esc(ev.text)}</span>`;
        const cur = steps.find((x) => x.state === "run");
        if (cur) setStep(cur.key, "fail", cur.title, esc(ev.text));
        break;
      case "end":
        ES.close(); ES = null; $("#askbtn").disabled = false; $("#answer").classList.remove("cursor"); break;
    }
  };
  ES.onerror = () => { if (ES) ES.close(); ES = null; $("#askbtn").disabled = false; };
}

function refLine(text) {
  // prefer the evidence: on-screen text, then speech, then the caption
  const lines = String(text || "").split("\n");
  const get = (k) => (lines.find((l) => l.startsWith(k)) || "").slice(k.length);
  const ocr = get("文字："), asr = get("语音："), cap = get("画面：") || text || "";
  const parts = [];
  if (ocr) parts.push("🔤 " + highlight(ocr.slice(0, 80), LAST_TERMS));
  if (asr && asr !== "（无）") parts.push("🎤 " + highlight(asr.slice(0, 60), LAST_TERMS));
  if (!parts.length) parts.push(highlight(cap.slice(0, 70), LAST_TERMS) + "…");
  return parts.join("<br>");
}

let LAST_TERMS = [];
function showRefs(refs) {
  if (!refs.length) return;
  $("#playercard").classList.remove("hidden");
  $("#refs").innerHTML = refs.map((r, i) => `<div class="ref" data-i="${i}">
      ${r.thumb ? `<img src="${r.thumb}">` : ""}
      <div><b>[${r.n}]</b> ${esc(r.video)} <span class="muted">${span(r.start, r.end)}</span>
      <div class="muted small">${refLine(r.text)}</div></div></div>`).join("");
  $("#refs").querySelectorAll(".ref").forEach((el) => el.addEventListener("click", () => playRef(refs[+el.dataset.i], el)));
  playRef(refs[0], $("#refs .ref"), false);
}

let stopAt = null;
function playRef(r, el, autoplay = true) {
  const p = $("#player");
  const url = r.video_url || "/media/video/" + encodeURIComponent(r.video);
  $("#playercard").classList.remove("hidden");
  document.querySelectorAll(".ref").forEach((x) => x.classList.toggle("playing", x === el));
  const go = () => { p.currentTime = r.start; stopAt = r.end; if (autoplay) p.play(); };
  if (p.dataset.src !== url) { p.src = url; p.dataset.src = url; p.addEventListener("loadedmetadata", go, { once: true }); } else go();
}
$("#player").addEventListener("timeupdate", (e) => { if (stopAt != null && e.target.currentTime >= stopAt) { e.target.pause(); stopAt = null; } });

/* ----------------------------------------------------------------- init */
loadStatus();
loadVideos();
api("/api/jobs").then((jobs) => { if (jobs && jobs.length) watchJob(jobs[0]); }).catch(() => {});
