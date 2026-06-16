/* ===== AI Evaluation Factory dashboard ===== */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const api = (p, opt) => fetch(p, opt).then(r => r.json().then(j => ({ ok: r.ok, data: j })));
const fmt = (x, d = 2) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));
const pct = x => (x === null || x === undefined ? "–" : (x * 100).toFixed(0) + "%");
const esc = s => (s ?? "").toString().replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const scoreClass = s => "s" + Math.max(1, Math.min(5, Math.round(s)));

let STATE = { summary: null, results: null, dataset: [] };

/* ---------------- Tabs ---------------- */
$$("#tabs button").forEach(b => b.onclick = () => {
  $$("#tabs button").forEach(x => x.classList.remove("active"));
  b.classList.add("active");
  $$(".view").forEach(v => v.classList.remove("active"));
  $("#view-" + b.dataset.view).classList.add("active");
});

/* ---------------- Init ---------------- */
async function init() {
  const { data: st } = await api("/api/status");
  renderStatus(st);
  renderOverview(st);
  loadDataset();
  loadReports();
}

function renderStatus(st) {
  const online = st.mode === "ONLINE";
  $("#status-pills").innerHTML = `
    <span class="pill"><span class="dot ${online ? "online" : "offline"}"></span> <b>${st.mode}</b></span>
    <span class="pill">Judges: <b>${st.judge_models.join(" + ")}</b></span>
    <span class="pill">Arbiter: <b>${st.arbiter_model}</b></span>
    <span class="pill">Dataset: <b>${st.dataset_size}</b> · Corpus: <b>${st.corpus_size}</b></span>
    <span class="pill">Reports: <b>${st.has_reports ? "✓" : "chưa có"}</b></span>`;
}

function renderOverview(st) {
  const cards = [
    ["Chế độ", st.mode, st.mode === "ONLINE" ? "Gemini thật" : "Mock offline", "🛰️"],
    ["Golden cases", st.dataset_size, "có ground-truth IDs", "🗂️"],
    ["Knowledge chunks", st.corpus_size, "corpus retrieval", "📚"],
    ["Judge models", st.judge_models.length + " + 1", "+ arbiter pro", "⚖️"],
  ];
  $("#overview-cards").innerHTML = cards.map(c => metricCard(...c)).join("");
}

const metricCard = (label, value, sub, spark) => `
  <div class="metric"><div class="spark">${spark || ""}</div>
    <div class="label">${label}</div><div class="value">${value}</div>
    <div class="sub">${sub || ""}</div></div>`;

/* ---------------- Dataset / samples ---------------- */
async function loadDataset() {
  const { ok, data } = await api("/api/dataset");
  if (!ok) return;
  STATE.dataset = data;
  const picks = [
    data.find(d => d.id === "G-014"), data.find(d => d.type === "out-of-context"),
    data.find(d => d.type === "prompt-injection"), data.find(d => d.type === "robustness"),
    data.find(d => d.type === "multi-hop"),
  ].filter(Boolean);
  $("#samples").innerHTML = picks.map(p =>
    `<span class="chip" data-q="${esc(p.question)}">${esc(p.question.slice(0, 48))}${p.question.length > 48 ? "…" : ""}</span>`).join("");
  $$("#samples .chip").forEach(c => c.onclick = () => { $("#q-input").value = c.dataset.q; runDemo(); });
}

/* ---------------- Live demo ---------------- */
$("#run-btn").onclick = runDemo;
$("#q-input").addEventListener("keydown", e => { if (e.key === "Enter") runDemo(); });

async function runDemo() {
  const question = $("#q-input").value.trim();
  if (!question) return;
  const version = $("#version-select").value;
  const btn = $("#run-btn");
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Đang chạy…';
  $("#demo-result").innerHTML = pipelineSkeleton();
  animateStages();

  const { ok, data } = await api("/api/evaluate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, version })
  });
  btn.disabled = false; btn.innerHTML = "▶ Đánh giá";
  if (!ok) { $("#demo-result").innerHTML = `<div class="panel empty">❌ ${esc(data.error || "Lỗi")}</div>`; return; }
  renderTrace(data);
}

function pipelineSkeleton() {
  return `<div class="panel"><div class="pipeline" id="live-pipe">
    ${["🔎 Retrieval", "✍️ Generation", "📐 RAGAS", "⚖️ Multi-Judge"].map((t, i) =>
    `<div class="stage" id="ls-${i}"><div class="num">TẦNG ${i + 1}</div><div class="title">${t}</div>
       <div class="desc"><span class="spinner"></span></div></div>${i < 3 ? '<div class="arrow">→</div>' : ""}`).join("")}
  </div></div>`;
}
function animateStages() {
  [0, 1, 2, 3].forEach((i) => setTimeout(() => { const e = $("#ls-" + i); if (e) e.classList.add("lit"); }, i * 350));
}

function gauge(value, label, color) {
  const r = 30, c = 2 * Math.PI * r, off = c * (1 - value);
  return `<div class="gauge">
    <svg width="74" height="74"><circle cx="37" cy="37" r="${r}" fill="none" stroke="var(--border)" stroke-width="7"/>
      <circle cx="37" cy="37" r="${r}" fill="none" stroke="${color}" stroke-width="7" stroke-linecap="round"
        stroke-dasharray="${c}" stroke-dashoffset="${off}"/></svg>
    <div class="gv">${pct(value)}</div><div class="gl">${label}</div></div>`;
}

function renderTrace(d) {
  const s = d.stages;
  const ret = s.retrieval, gen = s.generation, rag = s.ragas, jd = s.judge;

  const chunks = ret.retrieved.map(r => `
    <div class="chunk ${r.is_expected ? "hit" : ""}">
      <span class="cid">${r.id}</span><span class="ctitle">${esc(r.title)}</span>
      ${r.is_expected ? '<span class="badge good">GT ✓</span>' : ""}
      <div style="width:90px"><div class="scorebar" style="width:${Math.max(4, (r.score || 0) * 100)}%"></div></div>
      <span class="muted" style="font-size:11px;width:38px;text-align:right">${fmt(r.score, 2)}</span>
    </div>`).join("");

  const retMetric = ret.metric.scored
    ? `<span class="badge ${ret.metric.hit_rate ? "good" : "bad"}">Hit ${ret.metric.hit_rate ? "✓" : "✗"}</span>
       <span class="badge info">MRR ${fmt(ret.metric.mrr, 2)}</span>`
    : `<span class="badge warn">không cần retrieval (red-team)</span>`;

  const judges = Object.entries(jd.individual_scores).map(([m, sc]) =>
    `<div class="judge-row"><span class="judge-name">${m}</span>
      <span class="score-pill ${scoreClass(sc)}">${sc}</span></div>`).join("");

  let debateHtml = "";
  if (jd.debate) {
    const b = jd.debate;
    debateHtml = `<div class="debate-box">💬 <b>Tranh luận</b> (${b.converged ? "đã hội tụ ✓" : "vẫn lệch → trọng tài"})<br>
      ${Object.entries(b.before).map(([m, v]) => `${m}: ${v} → <b>${b.after[m]}</b>`).join(" · ")}</div>`;
  }
  const arbiterHtml = jd.arbiter_score != null
    ? `<div class="debate-box">👨‍⚖️ <b>Trọng tài</b> (gemini-2.5-pro) chấm: <b>${jd.arbiter_score}</b></div>` : "";

  const resBadge = { average: "info", debate: "good", "debate+arbiter": "warn", arbiter: "warn" }[jd.conflict_resolution] || "info";

  $("#demo-result").innerHTML = `
    <div class="panel" style="margin-bottom:14px; display:flex; gap:14px; flex-wrap:wrap; align-items:center;">
      <span class="badge info">${d.version}</span>
      <span class="badge ${d.mode.startsWith("ONLINE") ? "good" : "warn"}">${d.mode}</span>
      ${d.matched_golden ? `<span class="badge info">khớp golden: ${d.matched_golden}</span>` : `<span class="badge warn">câu tự do (không có đáp án chuẩn)</span>`}
      <span class="muted">⏱️ ${d.total_time_ms} ms · 💰 $${fmt(d.cost_usd, 6)}</span>
    </div>
    <div class="trace">
      <div class="panel">
        <h3>🔎 Tầng 1 — Retrieval ${retMetric}</h3>
        <div class="muted" style="font-size:12px;margin-bottom:8px;">top_k=${ret.top_k} · ${gen.time_ms} ms</div>
        ${chunks}
      </div>
      <div class="panel">
        <h3>✍️ Tầng 2 — Generation <span class="badge info">${gen.contexts_used} context</span></h3>
        <div class="answer-box">${esc(gen.answer)}</div>
        <div class="muted" style="font-size:12px;margin-top:8px;">${gen.model} · ${gen.tokens.prompt}+${gen.tokens.completion} tokens</div>
      </div>
      <div class="panel">
        <h3>📐 Tầng 3 — RAGAS metrics</h3>
        <div class="gauges">
          ${gauge(rag.faithfulness, "Faithfulness", "var(--good)")}
          ${gauge(rag.relevancy, "Relevancy", "var(--accent)")}
        </div>
        <div class="muted" style="font-size:12px;margin-top:10px;">${esc(rag.reason || "")}</div>
      </div>
      <div class="panel">
        <h3>⚖️ Tầng 4 — Multi-Judge <span class="badge ${resBadge}">${jd.conflict_resolution}</span></h3>
        ${judges}
        ${debateHtml}${arbiterHtml}
        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:12px;">
          <span class="muted">Điểm đồng thuận</span>
          <span class="score-pill ${scoreClass(jd.final_score)}" style="font-size:22px;">${fmt(jd.final_score, 1)}</span>
        </div>
        <div class="muted" style="font-size:12px;margin-top:8px;">${esc(jd.reasoning || "")}</div>
        <div style="margin-top:8px;">${["accuracy", "completeness", "professionalism", "safety"].map(k =>
        `<span class="tag">${k}: ${fmt(jd.criteria[k], 1)}</span>`).join(" ")}</div>
      </div>
    </div>`;
  $$("#live-pipe .stage").forEach(s => { s.classList.remove("lit"); s.classList.add("done"); });
}

/* ---------------- Reports (results + regression) ---------------- */
async function loadReports() {
  const [s, r] = await Promise.all([api("/api/summary"), api("/api/results")]);
  if (s.ok) STATE.summary = s.data;
  if (r.ok) STATE.results = r.data;
  renderResults();
  renderRegression();
  renderCases();
}

function renderResults() {
  const sub = $("#results-sub");
  if (!STATE.summary) { sub.textContent = "Chưa có report. Chạy `python main.py` rồi tải lại trang."; $("#result-cards").innerHTML = ""; return; }
  const m = STATE.summary.metrics, meta = STATE.summary.metadata;
  sub.innerHTML = `Phiên bản <b>${meta.version}</b> · ${meta.total} cases · ${meta.timestamp} · models: ${esc((meta.judge_models || []).join(", "))}`;
  $("#result-cards").innerHTML = [
    metricCard("Avg Judge Score", fmt(m.avg_score, 2) + " / 5", "consensus", "⭐"),
    metricCard("Pass Rate", pct(m.pass_rate), m.num_conflicts + " conflicts", "✅"),
    metricCard("Hit Rate", pct(m.hit_rate), "retrieval", "🎯"),
    metricCard("MRR", fmt(m.mrr, 3), "ranking quality", "📈"),
    metricCard("Faithfulness", pct(m.avg_faithfulness), "chống hallucination", "🛡️"),
    metricCard("Agreement", pct(m.agreement_rate), "Kappa " + fmt(m.cohens_kappa, 2), "🤝"),
    metricCard("Position Bias", pct(m.position_bias_rate), "judge audit", "🔄"),
    metricCard("Cost / eval", "$" + fmt(m.cost_per_eval_usd, 5), "tổng $" + fmt(m.total_cost_usd, 4), "💰"),
    metricCard("Wall time", fmt(m.wall_time_sec, 1) + "s", "avg " + fmt(m.avg_latency_sec, 2) + "s/case", "⚡"),
  ].join("");

  const cb = STATE.summary.cost_breakdown;
  const distHtml = barChartFromCounts(scoreDistribution());
  const costHtml = cb ? Object.entries(cb.per_model).map(([mdl, v]) =>
    barRow(mdl, v.cost_usd / cb.total_cost_usd, "$" + fmt(v.cost_usd, 5))).join("") : "";
  $("#result-charts").innerHTML = `
    <div class="panel"><h3 style="margin-top:0;">Phân bố điểm Judge</h3><div class="bars">${distHtml}</div></div>
    <div class="panel"><h3 style="margin-top:0;">Chi phí theo model</h3><div class="bars">${costHtml}</div>
      <div class="muted" style="font-size:12px;margin-top:8px;">Tổng ${cb ? cb.total_tokens.toLocaleString() : 0} tokens · ${cb ? cb.total_calls : 0} calls</div></div>`;
}

function scoreDistribution() {
  const counts = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
  (STATE.results || []).forEach(r => { const s = Math.round(r.judge.final_score); counts[Math.max(1, Math.min(5, s))]++; });
  return counts;
}
function barChartFromCounts(counts) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  return Object.entries(counts).map(([k, v]) => barRow(k + " ⭐", v / total, v)).join("");
}
function barRow(label, frac, valLabel, cls = "") {
  return `<div class="bar-row"><span class="muted">${label}</span>
    <div class="bar-track"><div class="bar-fill ${cls}" style="width:${Math.max(2, frac * 100)}%"></div></div>
    <span style="text-align:right">${valLabel}</span></div>`;
}

function renderRegression() {
  if (!STATE.summary || !STATE.summary.regression) { $("#gate-box").innerHTML = `<div class="panel empty">Chưa có dữ liệu regression.</div>`; return; }
  const reg = STATE.summary.regression, gate = reg.gate, v1 = reg.v1, v2 = reg.v2;
  $("#gate-box").innerHTML = `
    <div class="panel" style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
      <div><div class="muted">Quyết định phát hành</div>
        <div class="gate-decision gate-${gate.decision}">${gate.decision.replace(/_/g, " ")}</div></div>
      <div style="display:flex;gap:18px;flex-wrap:wrap;margin-left:auto;">
        ${deltaPill("Δ avg_score", gate.delta.avg_score)}
        ${deltaPill("Δ hit_rate", gate.delta.hit_rate)}
        ${deltaPill("Δ faithfulness", gate.delta.faithfulness)}
        <div class="metric" style="min-width:120px"><div class="label">Cost ratio</div><div class="value">${fmt(gate.delta.cost_ratio, 2)}x</div></div>
      </div>
    </div>`;

  const metrics = [["avg_score", 5], ["pass_rate", 1], ["hit_rate", 1], ["mrr", 1], ["avg_faithfulness", 1], ["agreement_rate", 1]];
  $("#regression-bars").innerHTML = metrics.map(([k, max]) => `
    <div style="margin-bottom:6px"><div class="muted" style="font-size:12px">${k}</div>
      ${barRow("V1", v1[k] / max, fmt(v1[k], 2), "v1")}
      ${barRow("V2", v2[k] / max, fmt(v2[k], 2))}</div>`).join("");

  $("#gate-checks").innerHTML = gate.checks.map(c => `
    <div class="check-row"><span class="check-icon">${c.passed ? "✅" : "❌"}</span>
      <span style="flex:1"><b>${c.criterion}</b><br><span class="muted" style="font-size:12px">${esc(c.detail)}</span></span></div>`).join("");
}
function deltaPill(label, v) {
  const good = v >= 0; const cls = good ? "good" : "bad";
  return `<div class="metric" style="min-width:120px"><div class="label">${label}</div>
    <div class="value ${cls === "good" ? "" : ""}" style="color:${good ? "var(--good)" : "var(--bad)"}">${v >= 0 ? "+" : ""}${fmt(v, 3)}</div></div>`;
}

/* ---------------- Cases table ---------------- */
let CASE_FILTER = "all", CASE_SORT = { key: "id", dir: 1 };
function renderCases() {
  if (!STATE.results) { $("#cases-table tbody").innerHTML = `<tr><td colspan="8" class="empty">Chưa có results.</td></tr>`; return; }
  const types = ["all", "pass", "fail", ...new Set(STATE.results.map(r => r.type))];
  $("#case-filters").innerHTML = types.map(t =>
    `<span class="chip ${CASE_FILTER === t ? "" : ""}" data-f="${t}" style="${CASE_FILTER === t ? "border-color:var(--accent);color:var(--text)" : ""}">${t}</span>`).join("");
  $$("#case-filters .chip").forEach(c => c.onclick = () => { CASE_FILTER = c.dataset.f; renderCases(); });

  $$("#cases-table th").forEach(th => th.onclick = () => {
    const k = th.dataset.sort; CASE_SORT.dir = (CASE_SORT.key === k ? -CASE_SORT.dir : 1); CASE_SORT.key = k; renderCases();
  });

  let rows = STATE.results.slice();
  if (CASE_FILTER === "pass" || CASE_FILTER === "fail") rows = rows.filter(r => r.status === CASE_FILTER);
  else if (CASE_FILTER !== "all") rows = rows.filter(r => r.type === CASE_FILTER);

  const getv = (r, k) => ({
    id: r.id, question: r.question, type: r.type, status: r.status,
    hit: r.retrieval.hit_rate ?? -1, mrr: r.retrieval.mrr ?? -1,
    faith: r.ragas.faithfulness, score: r.judge.final_score
  }[k]);
  rows.sort((a, b) => { const x = getv(a, CASE_SORT.key), y = getv(b, CASE_SORT.key); return (x > y ? 1 : x < y ? -1 : 0) * CASE_SORT.dir; });

  $("#cases-table tbody").innerHTML = rows.map((r, i) => `
    <tr data-i="${STATE.results.indexOf(r)}">
      <td><code class="k">${r.id}</code></td>
      <td>${esc(r.question.slice(0, 60))}${r.question.length > 60 ? "…" : ""}</td>
      <td><span class="tag">${r.type}</span></td>
      <td>${r.retrieval.scored ? (r.retrieval.hit_rate ? "✓" : "✗") : "–"}</td>
      <td>${r.retrieval.scored ? fmt(r.retrieval.mrr, 2) : "–"}</td>
      <td>${pct(r.ragas.faithfulness)}</td>
      <td><span class="score-pill ${scoreClass(r.judge.final_score)}" style="font-size:13px;padding:2px 7px">${fmt(r.judge.final_score, 1)}</span></td>
      <td class="st-${r.status}">${r.status.toUpperCase()}</td>
    </tr>`).join("");
  $$("#cases-table tbody tr").forEach(tr => tr.onclick = () => showCaseModal(STATE.results[+tr.dataset.i]));
}

function showCaseModal(r) {
  const jd = r.judge;
  $("#modal-root").innerHTML = `
    <div style="position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;display:flex;align-items:center;justify-content:center;padding:20px" id="modal-bg">
      <div class="panel" style="max-width:760px;width:100%;max-height:85vh;overflow:auto" onclick="event.stopPropagation()">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <h3 style="margin:0"><code class="k">${r.id}</code> <span class="tag">${r.type}</span> <span class="st-${r.status}">${r.status.toUpperCase()}</span></h3>
          <button class="btn" id="modal-close">✕</button></div>
        <p><b>Câu hỏi:</b> ${esc(r.question)}</p>
        <p><b>Đáp án chuẩn:</b> <span class="muted">${esc(r.expected_answer)}</span></p>
        <div class="answer-box"><b>Agent trả lời:</b><br>${esc(r.agent_answer)}</div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin:12px 0">
          <span class="badge ${r.retrieval.scored ? (r.retrieval.hit_rate ? "good" : "bad") : "warn"}">
            Retrieval: ${r.retrieval.scored ? (r.retrieval.hit_rate ? "Hit ✓" : "Miss ✗") : "n/a"} ${r.retrieval.scored ? "MRR " + fmt(r.retrieval.mrr, 2) : ""}</span>
          <span class="badge info">Faith ${pct(r.ragas.faithfulness)}</span>
          <span class="badge info">Rel ${pct(r.ragas.relevancy)}</span>
          <span class="badge ${jd.conflict ? "warn" : "good"}">${jd.conflict_resolution}</span>
        </div>
        <p class="muted" style="font-size:12px">retrieved: ${(r.retrieved_ids || []).join(", ")} · expected: ${(r.expected_retrieval_ids || []).join(", ") || "–"}</p>
        <table><tr><th>Judge</th><th>Score</th></tr>
          ${Object.entries(jd.individual_scores).map(([m, s]) => `<tr><td>${m}</td><td><span class="score-pill ${scoreClass(s)}" style="font-size:13px;padding:2px 7px">${s}</span></td></tr>`).join("")}
          ${jd.arbiter_score != null ? `<tr><td>arbiter (pro)</td><td><span class="score-pill ${scoreClass(jd.arbiter_score)}" style="font-size:13px;padding:2px 7px">${jd.arbiter_score}</span></td></tr>` : ""}
          <tr><td><b>Consensus</b></td><td><span class="score-pill ${scoreClass(jd.final_score)}">${fmt(jd.final_score, 1)}</span></td></tr>
        </table>
        ${jd.debate ? `<div class="debate-box">💬 Tranh luận: ${Object.entries(jd.debate.before).map(([m, v]) => `${m} ${v}→${jd.debate.after[m]}`).join(" · ")} (${jd.debate.converged ? "hội tụ" : "không hội tụ"})</div>` : ""}
        <p class="muted" style="font-size:13px;margin-top:10px"><b>Lý do giám khảo:</b> ${esc(jd.reasoning || "")}</p>
      </div>
    </div>`;
  $("#modal-bg").onclick = () => $("#modal-root").innerHTML = "";
  $("#modal-close").onclick = () => $("#modal-root").innerHTML = "";
}

init();
