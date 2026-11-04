"use strict";
const STATE = { past: [], live: [], status: null, eval: null, current: null, loaded: false, failed: false };
const $ = (s, root = document) => root.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = `The server answered ${r.status}.`;
    try { const j = await r.json(); msg = j.error || (typeof j.detail === "string" ? j.detail : msg); } catch (_) { /* keep the status text */ }
    throw Object.assign(new Error(msg), { status: r.status });
  }
  return r.json();
}

const CAUSE = {
  wildfire_smoke: { label: "Wildfire smoke", color: "var(--smoke)" },
  dust: { label: "Dust", color: "var(--dust)" },
  mixed: { label: "Mixed sources", color: "var(--muted)" },
  traffic: { label: "Traffic", color: "var(--muted)" },
  industrial: { label: "Industry", color: "var(--muted)" },
  unknown: { label: "Unclear", color: "var(--muted)" },
};
const cause = (c) => CAUSE[c] || { label: String(c || "unclear").replace(/_/g, " "), color: "var(--muted)" };

const TOOL = {
  detector: { name: "AirLens detector", layer: "area" },
  station_readings: { name: "AirKorea stations", layer: "stations" },
  fire_hotspots: { name: "NASA FIRMS fires", layer: "hotspots" },
  wind: { name: "Open-Meteo wind", layer: "wind" },
  satellite_segments: { name: "SAM 3.1 on NASA GIBS", layer: "mask" },
  cams_forecast: { name: "CAMS forecast", layer: null },
};
const MODEL = {
  "meta-models/muse-spark-open-1026": "Muse Spark Open on Baseten",
  "muse-spark-1.3-contributor": "Muse Spark 1.3 on the Meta API",
  "gpt-6-luna": "GPT-6 Luna on OpenAI",
};
const modelName = (m) => MODEL[m] || m;
const KST = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Asia/Seoul", day: "numeric", month: "short", year: "numeric",
  hour: "2-digit", minute: "2-digit", hour12: false,
});
const KST_TIME = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Seoul", hour: "2-digit", minute: "2-digit", hour12: false });
const fmtTs = (ts) => `${KST.format(new Date(ts))} KST`;
const fmtDay = (d) => new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }).format(new Date(d));

// AirKorea grade bands (ug/m3): the worse of PM10 and PM2.5 decides.
function grade(pm10, pm25) {
  const band = (v, cuts) => (v == null ? -1 : cuts.findIndex((c) => v <= c) === -1 ? 3 : cuts.findIndex((c) => v <= c));
  return Math.max(band(pm10, [30, 80, 150]), band(pm25, [15, 35, 75]));
}
const GRADE_COLOR = ["var(--g-good)", "var(--g-mod)", "var(--g-bad)", "var(--g-vbad)"];

const evidence = (rep, tool) => (rep ? rep.evidence.find((e) => e.tool === tool) : undefined);

// Claims quote evidence field names so the number check can match them; show plain words instead.
const WORDS = [
  [/\bmean_speed_ms\b/g, "mean wind speed (m/s)"], [/\bspeed_ms\b/g, "speed (m/s)"],
  [/\bmean_from_deg\b/g, "mean direction from (°)"], [/\bfrom_deg\b/g, "from (°)"],
  [/\bupwind_bearing_deg\b/g, "upwind bearing (°)"], [/\bbearing_deg\b/g, "bearing (°)"],
  [/\btotal_frp_mw\b/g, "total fire power (MW)"], [/\bfrp_mw\b/g, "fire power (MW)"],
  [/\bnearest_hotspot_km\b/g, "nearest hotspot (km)"], [/\bnearest_km\b/g, "nearest (km)"],
  [/\barea_frac\b/g, "image share"], [/\bn_regions\b/g, "regions"], [/\bat_event\b/g, "at the event hour"],
  [/\bpm2_5\b/g, "PM2.5"], [/\bpm10\b/g, "PM10"], [/\bmax_no2\b/g, "max NO2"], [/\bmax_co\b/g, "max CO"],
  [/\bug\/m3\b/g, "µg/m³"], [/\bcount (\d+)/g, "$1 hotspots"],
  [/\b([a-z]+)_([a-z_]+)\b/g, (m) => m.replace(/_/g, " ")],
];
const plain = (text) => WORDS.reduce((t, [re, to]) => t.replace(re, to), text);

function sceneImage(ev) {
  const seg = evidence(ev.report, "satellite_segments");
  const want = seg && seg.data;
  const match = want && ev.images.find((i) => i.date === want.image_date && i.layer === want.image_layer);
  return match || ev.images.find((i) => i.layer.includes("TrueColor")) || null;
}

/* ---------- case rail and Today panel ---------- */

const caseHref = (ev) => `#/case/${ev.kind}/${ev.id}`;
const isActive = (ev, cur) => !!cur && cur.kind === ev.kind && cur.id === ev.id;

function caseLink(ev, cur) {
  const c = ev.has_report ? cause(ev.cause) : { label: "Not investigated yet", color: "var(--muted)" };
  const tag = ev.has_report ? `${c.label}, ${Math.round(ev.confidence * 100)}%` : c.label;
  return `<li><a href="${caseHref(ev)}" ${isActive(ev, cur) ? 'aria-current="true"' : ""}>
      <span class="case-title">${esc(ev.title)}</span>
      <span class="case-date">${esc(fmtTs(ev.ts))}</span>
      <span class="cause-tag ${ev.has_report ? "" : "pending"}" style="--cause:${c.color}">${esc(tag)}</span>
    </a></li>`;
}

function renderCases(cur) {
  $("#case-list").innerHTML = STATE.past.map((ev) => caseLink(ev, cur)).join("");
  const st = STATE.status && STATE.status.live && STATE.status.live.ran_at ? STATE.status.live : null;
  const line = st
    ? `${st.stations_reporting ?? 0} stations checked at ${KST_TIME.format(new Date(st.ran_at))} KST.`
    : "The hourly check has not run yet.";
  $("#today").innerHTML = STATE.live.length
    ? `<ul class="case-list">${STATE.live.map((ev) => caseLink(ev, cur)).join("")}</ul><p class="fine">${esc(line)}</p>`
    : `<div class="live-empty"><p>No pollution event in the latest hourly check.</p><p class="fine">${esc(line)}</p></div>`;
}

/* ---------- investigation screen ---------- */

function renderHead(ev) {
  const rep = ev.report, head = $("#verdict-head");
  if (!rep) {
    head.style.setProperty("--cause", "var(--muted)");
    head.innerHTML = `
    <p class="cause-word">Not investigated yet</p>
    <h1 class="headline">${esc(ev.title)}</h1>
    <p class="case-meta">${esc(ev.region)}. ${esc(fmtTs(ev.ts))}.</p>`;
    return;
  }
  const c = cause(rep.cause), e1 = (evidence(rep, "detector") || { data: {} }).data;
  head.style.setProperty("--cause", c.color);
  head.innerHTML = `
    <p class="cause-word">${esc(c.label)}</p>
    <p class="confidence"><span class="meter" style="--v:${rep.confidence}"></span>${Math.round(rep.confidence * 100)}% confident</p>
    <h1 class="headline">${esc(rep.headline)}</h1>
    <p class="case-meta">${esc(ev.region)}. ${esc(fmtTs(ev.candidate.ts))}.${e1.n_stations ? ` ${e1.n_stations} stations over ${e1.extent_km} km.` : ""}</p>`;
}

function renderBody(ev) {
  const rep = ev.report;
  if (!rep) {
    const cd = ev.candidate;
    const li = (v, unit) => (v == null ? "" : `<li><b>${esc(v)}</b> ${unit}</li>`);
    $("#verdict-body").innerHTML = `
    <div class="no-report">
      <p><b>No report for this event yet.</b></p>
      <p>${esc(ev.why_no_report || "It has not been investigated.")}</p>
      <ul class="nums">${li(cd.peak_pm10, "µg/m³ peak PM10")}${li(cd.peak_pm25, "µg/m³ peak PM2.5")}${li(cd.n_stations, "stations")}${li(cd.severity == null ? null : Number(cd.severity).toFixed(2), "severity")}</ul>
      <p class="fine">These are the detector's numbers, from rules only. No model has looked at this event.</p>
    </div>`;
    return;
  }
  const chip = (id) => {
    const e = rep.evidence.find((x) => x.evidence_id === id);
    const t = (e && TOOL[e.tool]) || { name: e ? e.source : id };
    return `<button type="button" class="chip" data-evidence="${esc(id)}" aria-pressed="false"><b>${esc(id)}</b>${esc(t.name)}</button>`;
  };
  $("#verdict-body").innerHTML = `
    <p class="summary">${esc(rep.summary)}</p>
    <section aria-labelledby="f-title">
      <h2 id="f-title">What the evidence shows</h2>
      <ul class="findings">${rep.claims.map((cl) => `<li><p>${esc(plain(cl.text))}</p><div class="chips">${cl.evidence_ids.map(chip).join("")}</div></li>`).join("")}</ul>
      <div class="evidence-detail" id="evidence-detail" hidden></div>
    </section>
    <section aria-labelledby="r-title">
      <h2 id="r-title">Causes it ruled out</h2>
      <ul class="ruled">${rep.alternatives.map((a) => `<li><b>${esc(cause(a.cause).label)}</b><span>${esc(a.why_less_likely)}</span></li>`).join("")}</ul>
    </section>
    <section aria-labelledby="l-title">
      <h2 id="l-title">What it could not check</h2>
      <ul class="limits-list">${rep.limitations.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>
    </section>`;
  $("#verdict-body").querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => focusEvidence(ev, b)));
}

function detailRows(data) {
  return Object.entries(data).map(([k, v]) => {
    let val;
    if (Array.isArray(v)) {
      val = v.length && typeof v[0] === "object"
        ? `${v.length} items. First: ${Object.entries(v[0]).map(([a, b]) => `${a} ${Array.isArray(b) ? b.join(", ") : b}`).join(", ")}`
        : v.join(", ");
    } else if (v && typeof v === "object") {
      val = Object.entries(v).map(([a, b]) => `${a} ${typeof b === "object" && b ? JSON.stringify(b) : b}`).join("; ");
    } else val = v;
    return `<dt>${esc(k.replace(/_/g, " "))}</dt><dd>${esc(val)}</dd>`;
  }).join("");
}

function focusEvidence(ev, btn) {
  const frame = $("#scene-frame"), detail = $("#evidence-detail");
  const on = btn.getAttribute("aria-pressed") !== "true";
  document.querySelectorAll(".chip").forEach((b) => b.setAttribute("aria-pressed", "false"));
  delete frame.dataset.focus;
  detail.hidden = true;
  if (!on) return;
  document.querySelectorAll(`.chip[data-evidence="${btn.dataset.evidence}"]`).forEach((b) => b.setAttribute("aria-pressed", "true"));
  const e = ev.report.evidence.find((x) => x.evidence_id === btn.dataset.evidence);
  const layer = TOOL[e.tool] && TOOL[e.tool].layer;
  if (layer) frame.dataset.focus = layer;
  if (layer === "mask" && !frame.classList.contains("no-mask")) setSplit(0);
  detail.innerHTML = `<h3>${esc(e.evidence_id)} ${esc(e.source)}</h3><dl>${detailRows(e.data)}</dl>`;
  detail.hidden = false;
}

function setSplit(v) {
  $("#compare").value = v;
  $("#scene-frame").style.setProperty("--split", `${v}%`);
}

function renderScene(ev) {
  const img = sceneImage(ev), rep = ev.report;
  const [w, s, e, n] = img ? img.bbox : ev.box, W = 1000, H = Math.round((1000 * (n - s)) / (e - w));
  const P = (lon, lat) => [((lon - w) / (e - w)) * W, ((n - lat) / (n - s)) * H];
  const frame = $("#scene-frame");
  delete frame.dataset.focus;
  frame.style.aspectRatio = `${W} / ${H}`;
  $("#scene-base").hidden = !img;
  if (img) {
    $("#scene-base").src = img.file;
    $("#scene-base").alt = `${img.layer.replace(/_/g, " ")} satellite image, ${img.date}`;
  } else $("#scene-base").removeAttribute("src");
  frame.classList.toggle("no-mask", !ev.overlay);
  $("#compare-wrap").hidden = !ev.overlay;
  if (ev.overlay) { $("#scene-mask").src = ev.overlay; $("#scene-mask").alt = "The same image with SAM 3.1's segmented regions"; setSplit(50); }

  const fires = ev.hotspots; // the server already limits these to the 24 hours before the event
  const hot = fires.map((h) => { const [x, y] = P(h.lon, h.lat); return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${Math.min(9, 2 + Math.sqrt(h.frp || 1) / 5).toFixed(1)}"/>`; }).join("");
  const sta = ev.stations.filter((st) => st.lat != null).map((st) => {
    const [x, y] = P(st.lon, st.lat), g = grade(st.pm10, st.pm25);
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${st.in_event ? 6 : 3.5}" fill="${g < 0 ? "transparent" : GRADE_COLOR[g]}" ${st.in_event ? 'stroke="#fff" stroke-width="2"' : 'stroke="rgba(0,0,0,.45)" stroke-width="1"'}/>`;
  }).join("");
  const [bw, bs, be, bn] = ev.candidate.bbox, [x0, y0] = P(bw, bn), [x1, y1] = P(be, bs);
  const wind = evidence(rep, "wind"), [cx, cy] = P(ev.candidate.centroid_lon, ev.candidate.centroid_lat);
  let windSvg = "";
  if (wind) {
    const to = ((wind.data.mean_from_deg + 180) * Math.PI) / 180, L = 50 + 10 * Math.min(wind.data.mean_speed_ms, 15);
    const tx = cx + Math.sin(to) * L, ty = cy - Math.cos(to) * L;
    windSvg = `<line x1="${(cx - Math.sin(to) * L * 0.2).toFixed(1)}" y1="${(cy + Math.cos(to) * L * 0.2).toFixed(1)}" x2="${tx.toFixed(1)}" y2="${ty.toFixed(1)}" stroke="#fff" stroke-width="4" marker-end="url(#arrow)"/>`;
  }
  $("#scene-marks").setAttribute("viewBox", `0 0 ${W} ${H}`);
  $("#scene-marks").setAttribute("preserveAspectRatio", "none");
  $("#scene-marks").innerHTML = `
    <defs><marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L10 5L0 10z" fill="#fff"/></marker></defs>
    <g data-layer="area"><rect x="${x0.toFixed(1)}" y="${y0.toFixed(1)}" width="${(x1 - x0).toFixed(1)}" height="${(y1 - y0).toFixed(1)}" fill="none" stroke="#fff" stroke-width="2" stroke-dasharray="8 6"/></g>
    <g data-layer="hotspots" fill="var(--ember)" fill-opacity=".8">${hot}</g>
    <g data-layer="stations">${sta}</g>
    <g data-layer="wind">${windSvg}</g>`;
  document.querySelectorAll(".layers input").forEach((box) => {
    const g = $(`#scene-marks [data-layer="${box.dataset.layer}"]`);
    if (g) g.style.display = box.checked ? "" : "none";
  });

  const seg = evidence(rep, "satellite_segments");
  const sat = img ? img.layer.split("_").slice(0, 2).join(" ").replace("VIIRS NOAA20", "VIIRS on NOAA-20").replace("MODIS Terra", "MODIS on Terra") : "";
  const segNote = !seg ? "" : seg.data.n_regions
    ? ` SAM 3.1 segmented ${seg.data.n_regions} "${seg.data.prompt}" regions; slide to compare.`
    : ` SAM 3.1 found no "${seg.data.prompt}" region in this pass.`;
  const imgNote = img ? `${sat} true colour, ${fmtDay(img.date)}, from NASA GIBS.` : "No satellite image for this event yet.";
  $("#scene-caption").textContent = `${imgNote}${segNote} Fire hotspots from the 24 hours before ${fmtTs(ev.candidate.ts)}.`;
  const windTxt = wind ? `Wind ${wind.data.mean_speed_ms} m/s from ${wind.data.upwind_direction}` : "";
  $("#legend").innerHTML = `
    <li><i style="background:var(--ember)"></i>Fire hotspot, ${fires.length} in 24 h</li>
    <li><i style="background:var(--g-good)"></i>Good</li><li><i style="background:var(--g-mod)"></i>Moderate</li>
    <li><i style="background:var(--g-bad)"></i>Unhealthy</li><li><i style="background:var(--g-vbad)"></i>Very unhealthy</li>
    <li><i style="background:transparent;box-shadow:inset 0 0 0 2px var(--text)"></i>Station in the event</li>
    ${windTxt ? `<li>${esc(windTxt)}</li>` : ""}`;
}

function renderSteps(ev) {
  const rep = ev.report;
  $(".pipeline").hidden = !rep;
  if (!rep) return;
  const e1 = (evidence(rep, "detector") || { data: {} }).data, tri = rep.triage || { hypotheses: [], checks: [] };
  const tools = rep.evidence.filter((e) => e.tool !== "detector");
  const seg = evidence(rep, "satellite_segments");
  const total = rep.claims.length, ok = total - rep.unverified_claims.length;
  const hyps = tri.hypotheses.map((h) => `${cause(h.cause).label.toLowerCase()} ${Math.round(h.likelihood * 100)}%`).join(", ");
  const steps = [
    ["Detected", "Rules, no model", `${e1.n_stations} stations over ${e1.extent_km} km. Peak PM10 ${e1.peak_pm10}, PM2.5 ${e1.peak_pm25} µg/m³.`],
    ["Triaged", modelName(rep.models[0]), `Ranked ${hyps}. Chose ${tri.checks.length} checks.`],
    ["Investigated", modelName(rep.models[1] || rep.models[0]), `Called ${tools.length} tools: ${tools.map((t) => (TOOL[t.tool] || { name: t.tool }).name).join(", ")}.`],
    ["Segmented", "SAM 3.1", seg ? `Looked for "${seg.data.prompt}" in the ${fmtDay(seg.data.image_date)} image and found ${seg.data.n_regions} regions.` : "No satellite check."],
    ["Checked", "Rules, no model", `${ok} of ${total} findings match the numbers in the evidence.`],
    ["Concluded", cause(rep.cause).label, `${Math.round(rep.confidence * 100)}% confident. Cost $${rep.cost_usd.toFixed(3)}.`],
  ];
  $("#steps").innerHTML = steps.map(([t, m, p]) => `<li><b>${esc(t)}</b><p class="model">${esc(m)}</p><p>${esc(p)}</p></li>`).join("");
}

// "Same cases, three models": hidden when /api/eval has nothing.
function renderEval() {
  const box = $("#eval"), data = STATE.eval;
  const models = data && data.summary ? Object.entries(data.summary) : [];
  box.hidden = !models.length;
  if (!models.length) return;
  const cases = (data.events || []).length;
  $("#eval-note").textContent = `Each model investigated the same ${cases} past event${cases === 1 ? "" : "s"} ${data.runs} time${data.runs === 1 ? "" : "s"}, and its cause was compared with the known one.`;
  $("#eval thead").innerHTML = "<tr><th scope=\"col\">Model</th><th scope=\"col\">Correct / runs</th><th scope=\"col\">Median time</th><th scope=\"col\">Cost per case</th></tr>";
  $("#eval tbody").innerHTML = models.map(([m, g]) => {
    const a = g.all;
    return `<tr><th scope="row">${esc(modelName(m))}</th><td>${a.correct} / ${a.runs}</td><td>${a.median_latency_s} s</td><td>$${Number(a.mean_cost_usd).toFixed(3)}</td></tr>`;
  }).join("");
}

let showSeq = 0;
async function showCase(kind, id) {
  const seq = ++showSeq, frameBox = $("#verdict-head");
  frameBox.innerHTML = '<p class="loading">Loading the case.</p>';
  STATE.current = { kind, id };
  renderCases(STATE.current);
  try {
    const ev = await api(`/api/events/${kind}/${id}`);
    if (seq !== showSeq) return;
    renderHead(ev); renderScene(ev); renderBody(ev); renderSteps(ev);
  } catch (err) {
    if (seq !== showSeq) return;
    frameBox.innerHTML = `<p class="load-error" role="alert">Could not load this case. ${esc(err.message)}</p>`;
  }
}

function defaultCase() {
  const live = STATE.live.find((e) => e.has_report);
  return live || STATE.past[0] || STATE.live[0] || null;
}

/* ---------- question screen ---------- */

const GRADES = {
  good: ["good", "Good"], moderate: ["moderate", "Moderate"], unhealthy: ["bad", "Unhealthy"],
  very_unhealthy: ["very-bad", "Very unhealthy"], unknown: ["unknown", "Not enough data"],
};
const ASK_STEPS = ["Looking up the place.", "Reading station data and forecasts.", "Checking the numbers and writing the answer."];
let askTimer = null;

function renderAnswer(a) {
  const [key, label] = GRADES[a.risk_level] || GRADES.unknown;
  const cites = (a.citations || []).map((c) => {
    const val = c.value !== null && c.value !== undefined && c.value !== "" ? `${c.value}${c.unit ? ` ${c.unit}` : ""}` : "";
    const detail = [c.detail, val && !String(c.detail).includes(val) ? val : "", c.time].filter(Boolean).join(", ");
    return `<li><b>${esc(c.source)}</b><span>${esc(detail)}</span></li>`;
  }).join("");
  const notice = a.cached
    ? '<p class="cached-note">The daily question limit was reached or the agent is unavailable right now, so this is a saved example answer, not a live reading.</p>'
    : "";
  const scale = a.risk_level in GRADES && a.risk_level !== "unknown"
    ? `<ol class="grade-scale" aria-label="Air quality grades">${["good", "moderate", "unhealthy", "very_unhealthy"].map((l) => `<li data-grade="${GRADES[l][0]}" ${l === a.risk_level ? 'aria-current="true"' : ""}>${GRADES[l][1]}</li>`).join("")}</ol>`
    : "";
  const cost = a.cached ? "" : ` Cost $${Number(a.cost_usd || 0).toFixed(3)}.`;
  const models = (a.models || []).length ? `${a.models.map(modelName).join(", ")}.` : "";
  const out = $("#answer");
  out.innerHTML = `
    ${notice}
    <div class="risk">
      <p class="risk-word" data-grade="${key}">${esc(label)}</p>
      ${a.time_window ? `<p class="risk-window">${esc(a.time_window)}</p>` : ""}
      ${a.risk_basis ? `<p class="risk-basis">Grade from the ${esc(a.risk_basis)}</p>` : ""}
      ${scale}
    </div>
    <p class="answer-text">${esc(a.answer)}</p>
    ${a.confidence ? `<p class="confidence-line"><span class="meter" style="--v:${a.confidence}"></span> ${Math.round(a.confidence * 100)}% confident</p>` : ""}
    ${cites ? `<h2>Sources</h2><ul class="citations">${cites}</ul>` : ""}
    ${(a.limitations || []).length ? `<h2>Limits</h2><ul class="limits">${a.limitations.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>` : ""}
    ${models || cost ? `<p class="answer-meta">${esc(`${models}${cost}`.trim())}</p>` : ""}`;
  out.hidden = false;
}

function setAskStatus(text, isError) {
  const el = $("#ask-status");
  el.hidden = !text;
  el.textContent = text || "";
  el.classList.toggle("is-error", !!isError);
  el.setAttribute("role", isError ? "alert" : "status");
}

function fieldError(text) {
  const el = $("#question-error");
  el.hidden = !text;
  el.textContent = text || "";
  $("#question").toggleAttribute("aria-invalid", !!text);
}

async function submitAsk(e) {
  e.preventDefault();
  const q = $("#question"), btn = $("#ask-button");
  const text = q.value.replace(/\s+/g, " ").trim();
  fieldError("");
  if (!text) return fieldError("Please type a question about the air.");
  if (text.length > 300) return fieldError(`Please shorten the question to 300 characters or fewer (now ${text.length}).`);
  btn.disabled = true;
  $("#answer").hidden = true;
  let step = 0;
  setAskStatus(ASK_STEPS[0], false);
  askTimer = setInterval(() => { step = Math.min(step + 1, ASK_STEPS.length - 1); setAskStatus(ASK_STEPS[step], false); }, 8000);
  const ctl = new AbortController(), cut = setTimeout(() => ctl.abort(), 120000);
  try {
    const a = await api("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text }), signal: ctl.signal,
    });
    setAskStatus("", false);
    renderAnswer(a);
    $("#answer").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  } catch (err) {
    const msg = err.name === "AbortError" ? "The agent took too long to answer." : err.status ? err.message : "Could not reach the server.";
    setAskStatus(`${msg} Your question is still here, so you can try again.`, true);
  } finally {
    clearInterval(askTimer); clearTimeout(cut);
    btn.disabled = false;
  }
}

/* ---------- routing and start ---------- */

function route() {
  const m = location.hash.match(/^#\/(ask|case\/(past|live)\/([\w-]+))/);
  const ask = !!m && m[1] === "ask";
  $("#investigate").hidden = ask; $("#ask").hidden = !ask;
  document.querySelectorAll(".site-nav a").forEach((a) => {
    if (a.dataset.nav === (ask ? "ask" : "investigate")) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  if (ask) return;
  if (STATE.failed) return;
  if (!STATE.loaded) return; // boot() calls route() again once the lists are in
  const target = m ? { kind: m[2], id: m[3] } : defaultCase();
  if (target) showCase(target.kind, target.id);
  else $("#verdict-head").innerHTML = '<p class="load-error">No cases are available yet.</p>';
}

async function boot() {
  const jobs = await Promise.allSettled([api("/api/events"), api("/api/status"), api("/api/eval")]);
  if (jobs[0].status === "fulfilled") {
    STATE.live = jobs[0].value.live; STATE.past = jobs[0].value.past;
  } else {
    STATE.failed = true;
    $("#verdict-head").innerHTML = `<p class="load-error" role="alert">Could not load the cases. ${esc(jobs[0].reason.message)}</p>`;
    $("#today").innerHTML = "";
  }
  if (jobs[1].status === "fulfilled") STATE.status = jobs[1].value;
  if (jobs[2].status === "fulfilled") STATE.eval = jobs[2].value;
  STATE.loaded = true;
  if (!STATE.failed) { renderCases(null); renderEval(); }
  route();
}

$("#compare").addEventListener("input", (e) => setSplit(e.target.value));
document.querySelectorAll(".layers input").forEach((box) => box.addEventListener("change", () => {
  const g = $(`#scene-marks [data-layer="${box.dataset.layer}"]`);
  if (g) g.style.display = box.checked ? "" : "none";
}));
const qbox = $("#question");
qbox.addEventListener("input", () => {
  const n = qbox.value.replace(/\s+/g, " ").trim().length;
  $("#count").textContent = `${n} / 300`;
  if (n <= 300 && $("#question-error").textContent.includes("shorten")) fieldError("");
});
document.querySelectorAll(".examples button").forEach((b) => b.addEventListener("click", () => {
  qbox.value = b.textContent; qbox.dispatchEvent(new Event("input")); fieldError(""); qbox.focus();
}));
$("#ask-form").addEventListener("submit", submitAsk);
qbox.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) $("#ask-form").requestSubmit(); });
window.addEventListener("hashchange", route);
boot();
