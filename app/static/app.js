const $ = (s) => document.querySelector(s);
const pct = (x) => (x == null ? "–" : (x * 100).toFixed(1) + "%");
// Safe DOM builder: text is always set via textContent (no HTML injection from filenames/labels)
function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v);
  }
  kids.flat().forEach((c) => c != null && n.append(c));
  return n;
}
const badge = (t, c) => el("span", { class: "badge " + c, text: t });
function bar(p, low) { const b = el("div", { class: "bar" + (low ? " low" : "") }, el("i", { style: `width:${(p * 100).toFixed(0)}%` })); return b; }

async function health() {
  const h = await (await fetch("/health")).json();
  if (h.demo_mode) {
    const b = $("#banner"); b.hidden = false;
    b.textContent = "DEMO MODE: no model file found (model/waste_classifier.onnx). Predictions are placeholders, not real classifications.";
  }
}

/* ---------- Sorting line ---------- */
function initClassify() {
  let files = [];
  const drop = $("#drop"), input = $("#file"), run = $("#run");
  const set = (list) => { files = [...list].filter((f) => f.type.startsWith("image/")).slice(0, 20);
    drop.firstChild.textContent = files.length ? `${files.length} image(s) selected` : "Drop images here or click to choose";
    run.disabled = !files.length; };
  drop.onclick = () => input.click();
  drop.onkeydown = (e) => e.key === "Enter" && input.click();
  input.onchange = () => set(input.files);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); set(e.dataTransfer.files); };
  run.onclick = async () => {
    run.disabled = true; run.textContent = "Classifying…";
    const fd = new FormData(); fd.append("line_id", $("#line").value);
    files.forEach((f) => fd.append("files", f));
    const out = $("#results"); out.replaceChildren();
    try {
      const r = await fetch("/api/predict", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "Request failed");
      j.results.forEach((res, i) => {
        if (res.error) { out.append(el("div", { class: "card" }, el("div", { class: "in" }, el("b", { text: res.source_name }), el("p", { class: "bad badge", text: res.error })))); return; }
        const low = res.route === "manual";
        out.append(el("div", { class: "card" },
          el("img", { src: URL.createObjectURL(files[i]), alt: res.source_name }),
          el("div", { class: "in" },
            el("b", { text: res.predicted_class }), " ", el("small", { text: pct(res.confidence) }),
            bar(res.confidence, low),
            badge(low ? "Manual review" : "Auto-sort lane", low ? "warn" : "ok"),
            badge(res.recyclable ? "Recyclable" : "Residual", res.recyclable ? "ok" : "mute"),
            res.audit_sample ? badge("Audit sample", "mute") : null,
            el("br"), el("small", { text: res.top3.map((t) => `${t.label} ${pct(t.prob)}`).join(" · ") }))));
      });
    } catch (e) { out.append(el("p", { class: "badge bad", text: e.message })); }
    run.textContent = "Classify"; run.disabled = false;
  };
}

/* ---------- Dashboard ---------- */
const charts = {};
function draw(id, cfg) { charts[id]?.destroy(); charts[id] = new Chart($(id), cfg); }
async function loadDash() {
  const q = new URLSearchParams({ hours: $("#hours").value, line: $("#linesel").value });
  const s = await (await fetch("/api/stats?" + q)).json();
  const k = s.kpi;
  const kp = (v, l) => el("div", { class: "kpi" }, el("b", { text: v }), el("span", { text: l }));
  $("#kpis").replaceChildren(
    kp(k.total, "Items classified"), kp(pct(k.auto_rate), "Auto-sorted"),
    kp(pct(k.recyclable_share), "Recyclable share (diversion)"), kp(pct(k.avg_confidence), "Avg confidence"),
    kp(k.pending_review, "Awaiting review"), kp(pct(k.reviewed_accuracy), `Verified accuracy (${k.reviewed} checked)`));
  if (s.demo_rows) { const b = $("#banner"); b.hidden = false; b.textContent = `Contains ${s.demo_rows} demo/synthetic rows: not real plant data.`; }
  const sel = $("#linesel");
  if (sel.options.length === 1) s.lines.forEach((l) => sel.append(el("option", { value: l.line, text: l.line })));
  const pal = ["#1f8a4c", "#3b82c4", "#8a8f94", "#d9b44a", "#8e5bc4", "#b0523c"];
  draw("#c-time", { type: "bar", data: { labels: s.timeline.map((t) => t.t), datasets: [
    { label: "Auto", data: s.timeline.map((t) => t.n - t.manual), backgroundColor: "#1f8a4c" },
    { label: "Manual review", data: s.timeline.map((t) => t.manual), backgroundColor: "#d98a00" }] },
    options: { scales: { x: { stacked: true }, y: { stacked: true, beginAtZero: true } } } });
  draw("#c-class", { type: "doughnut", data: { labels: s.by_class.map((c) => c.label),
    datasets: [{ data: s.by_class.map((c) => c.n), backgroundColor: pal }] } });
  draw("#c-conf", { type: "bar", data: { labels: s.confidence_hist.map((_, i) => `${i * 10}-${i * 10 + 10}%`),
    datasets: [{ label: "Items", data: s.confidence_hist, backgroundColor: s.confidence_hist.map((_, i) => i * 0.1 >= s.threshold - 1e-9 ? "#1f8a4c" : "#d98a00") }] },
    options: { plugins: { legend: { display: false } } } });
  const bands = $("#bands"); bands.replaceChildren();
  ["high", "low"].forEach((b) => { const d = s.bands[b];
    bands.append(el("p", { text: `${b === "high" ? "≥" : "<"} ${s.threshold} confidence: ` + (d ? `${pct(d.accuracy)} correct (${d.reviewed} checked)` : "no verified items yet") })); });
  $("#conf").replaceChildren(s.confusions.length ? el("table", {}, s.confusions.map((c) =>
    el("tr", {}, el("td", { text: `${c.predicted} → ${c.actual}` }), el("td", { text: c.n })))) : el("small", { text: "No verified errors yet." }));
  $("#lines").replaceChildren(el("table", {}, el("tr", {}, ["Line", "Items", "Auto-sort rate", "Avg confidence"].map((h) => el("th", { text: h }))),
    s.lines.map((l) => el("tr", {}, el("td", { text: l.line }), el("td", { text: l.n }), el("td", { text: pct(l.auto_rate) }), el("td", { text: pct(l.avg_conf) })))));
}
function initDash() { $("#hours").onchange = loadDash; $("#linesel").onchange = loadDash; loadDash(); }

/* ---------- Review queue ---------- */
async function loadQueue() {
  const j = await (await fetch("/api/review")).json();
  $("#count").textContent = `${j.pending_total} pending.`;
  const q = $("#queue"); q.replaceChildren();
  if (!j.items.length) q.append(el("p", { class: "muted", text: "Queue is empty. Nice work." }));
  j.items.forEach((it) => {
    const card = el("div", { class: "card" }, it.thumb ? el("img", { src: it.thumb, alt: "item" }) : null,
      el("div", { class: "in" }, el("b", { text: it.predicted_class }), " ", el("small", { text: pct(it.confidence) }), el("br"),
        badge(it.route === "manual" ? "Low confidence" : "Audit sample", it.route === "manual" ? "warn" : "mute"),
        el("small", { text: ` ${it.line_id}` }), el("br"),
        j.labels.map((l) => el("button", { class: "btn alt", text: l, onclick: async () => {
          const r = await fetch("/api/review/" + it.id, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ label: l }) });
          if (r.ok) loadQueue(); } }))));
    q.append(card);
  });
}

health();
({ classify: initClassify, dashboard: initDash, review: loadQueue })[document.body.dataset.page]?.();
