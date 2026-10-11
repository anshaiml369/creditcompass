/* Credit Default Model — frontend logic. Talks to main.py: POST /predict (unchanged API). */
(function () {
  "use strict";
  const { $, $$, el, api, fmt, renderBars, plotXY, legend, setBusy, setStatus, callout, toast, copyText } = window.RUI;

  /* Exactly the fields of main.LoanApplication, in its order. Hard limits mirror the notebook's cleaning rules
     or basic validity; soft warnings use the range observed in the cleaned training data (schema.json). */
  const FIELDS = [
    { key: "person_age", group: "applicant", label: "Age", unit: "years", type: "int", min: 18, max: 100, step: 1 },
    { key: "person_income", group: "applicant", label: "Annual income", unit: "per year", type: "num", minExclusive: 0, step: 1000 },
    { key: "person_home_ownership", group: "applicant", label: "Home ownership", type: "cat" },
    { key: "person_emp_length", group: "applicant", label: "Employment length", unit: "years", type: "num", min: 0, max: 60, step: 1 },
    { key: "loan_intent", group: "loan", label: "Loan purpose", type: "cat" },
    { key: "loan_grade", group: "loan", label: "Loan grade", type: "cat", hint: "Lender-assigned grade, A (best) to G" },
    { key: "loan_amnt", group: "loan", label: "Loan amount", unit: "amount", type: "num", minExclusive: 0, step: 500 },
    { key: "loan_int_rate", group: "loan", label: "Interest rate", unit: "%", type: "num", minExclusive: 0, max: 100, step: 0.01 },
    { key: "loan_percent_income", group: "loan", label: "Loan-to-income ratio", unit: "ratio", type: "num", min: 0, max: 10, step: 0.01, derived: true },
    { key: "cb_person_default_on_file", group: "bureau", label: "Previous default on file", type: "cat" },
    { key: "cb_person_cred_hist_length", group: "bureau", label: "Credit history length", unit: "years", type: "int", min: 0, max: 80, step: 1 },
  ];
  const CAT_LABELS = {
    RENT: "Rent", MORTGAGE: "Mortgage", OWN: "Own", OTHER: "Other",
    DEBTCONSOLIDATION: "Debt consolidation", EDUCATION: "Education", HOMEIMPROVEMENT: "Home improvement",
    MEDICAL: "Medical", PERSONAL: "Personal", VENTURE: "Venture", Y: "Yes", N: "No",
  };
  const catLabel = (v) => CAT_LABELS[v] || v;

  const state = { schema: null, evaluation: null, ratioAuto: true, last: null };

  /* ------------------------------------------------------------------ */
  /* Form                                                               */
  /* ------------------------------------------------------------------ */
  function buildForm() {
    const s = state.schema.features;
    for (const f of FIELDS) {
      const meta = s[f.key];
      const id = `f-${f.key}`;
      let control;
      if (f.type === "cat") {
        control = el("select", { class: "select", id, name: f.key, required: true }, meta.values.map((v) => el("option", { value: v, text: `${catLabel(v)}${CAT_LABELS[v] && v !== catLabel(v) && f.key !== "cb_person_default_on_file" ? ` (${v})` : ""}` })));
      } else {
        const input = el("input", {
          class: "input", id, name: f.key, type: "number", inputmode: f.type === "int" ? "numeric" : "decimal",
          step: f.step, min: f.min ?? (f.minExclusive !== undefined ? f.minExclusive : undefined), max: f.max, required: true,
          "aria-describedby": `${id}-hint ${id}-err ${id}-warn`,
        });
        control = f.unit ? el("div", { class: "input-affix" }, [input, el("span", { class: "affix", text: f.unit })]) : input;
      }
      const rangeHint = f.type === "cat"
        ? (f.hint || `${meta.values.length} categories in training data`)
        : `Training range ${fmt.num(meta.min, meta.max < 2 ? 2 : 0)} – ${fmt.num(meta.max, meta.max < 2 ? 2 : 0)} · median ${fmt.num(meta.p50, meta.max < 2 ? 2 : 0)}`;
      const labelChildren = [el("span", { text: f.label }), el("span", { class: "field-code", text: f.key })];
      const field = el("div", { class: "field", "data-field": f.key }, [
        el("label", { class: "field-label", for: id }, labelChildren),
        control,
        el("span", { class: "hint", id: `${id}-hint` }, [rangeHint, f.derived ? el("span", {}, [" · ", el("button", { type: "button", class: "lock-toggle", id: "ratio-toggle", text: "auto: amount ÷ income" })]) : null]),
        el("span", { class: "field-error", id: `${id}-err`, role: "alert" }),
        el("span", { class: "field-warn", id: `${id}-warn` }),
      ]);
      $(`[data-group="${f.group}"]`).append(field);
    }
    $("#ratio-toggle").addEventListener("click", () => setRatioAuto(!state.ratioAuto));
    for (const k of ["person_income", "loan_amnt"]) $(`#f-${k}`).addEventListener("input", syncRatio);
    $("#f-loan_percent_income").addEventListener("input", () => { if (state.ratioAuto) setRatioAuto(false); });
    for (const f of FIELDS) $(`#f-${f.key}`).addEventListener("change", () => validateField(f, true));
    fillTypical();
  }

  function setRatioAuto(on) {
    state.ratioAuto = on;
    $("#ratio-toggle").textContent = on ? "auto: amount ÷ income" : "manual — click to recompute automatically";
    if (on) syncRatio();
  }

  function syncRatio() {
    if (!state.ratioAuto) return;
    const inc = parseFloat($("#f-person_income").value), amt = parseFloat($("#f-loan_amnt").value);
    if (inc > 0 && amt >= 0) $("#f-loan_percent_income").value = (Math.round((amt / inc) * 100) / 100).toFixed(2);
  }

  function fillTypical() {
    const s = state.schema.features;
    for (const f of FIELDS) {
      const meta = s[f.key];
      const v = f.type === "cat" ? Object.entries(meta.counts).sort((a, b) => b[1] - a[1])[0][0] : meta.p50;
      $(`#f-${f.key}`).value = f.type === "int" ? Math.round(v) : v;
    }
    setRatioAuto(true);
    FIELDS.forEach((f) => validateField(f, false));
    $("#form-summary").textContent = "Filled with training-data medians (numeric) and most common categories.";
  }

  function readValue(f) {
    const raw = $(`#f-${f.key}`).value;
    if (f.type === "cat") return raw;
    if (raw === "") return null;
    return f.type === "int" ? Number(raw) : parseFloat(raw);
  }

  function validateField(f, show) {
    const v = readValue(f);
    const meta = state.schema.features[f.key];
    let err = "", warn = "";
    if (f.type === "cat") {
      if (!meta.values.includes(v)) err = "Choose one of the listed options.";
    } else if (v === null || Number.isNaN(v)) err = "Required.";
    else if (f.type === "int" && !Number.isInteger(v)) err = "Must be a whole number.";
    else if (f.min !== undefined && v < f.min) err = `Must be at least ${f.min}.`;
    else if (f.minExclusive !== undefined && v <= f.minExclusive) err = `Must be greater than ${f.minExclusive}.`;
    else if (f.max !== undefined && v > f.max) err = `Must be at most ${f.max}.`;
    if (!err && f.key === "person_emp_length") {
      const age = readValue(FIELDS[0]);
      if (age !== null && v > age) err = "Cannot exceed age (such rows were removed from training).";
    }
    if (!err && f.type !== "cat" && (v < meta.min || v > meta.max)) warn = `Outside the training range (${meta.min}–${meta.max}); the model is extrapolating.`;
    if (show) {
      $(`#f-${f.key}-err`).textContent = err;
      $(`#f-${f.key}-warn`).textContent = warn;
      $(`#f-${f.key}`).setAttribute("aria-invalid", err ? "true" : "false");
    }
    return { err, warn };
  }

  function collect(show = true) {
    const payload = {};
    const errors = [], warnings = [];
    for (const f of FIELDS) {
      const { err, warn } = validateField(f, show);
      if (err) errors.push(f);
      if (warn) warnings.push(f.label);
      payload[f.key] = readValue(f);
    }
    return { payload, errors, warnings };
  }

  /* ------------------------------------------------------------------ */
  /* Predict                                                            */
  /* ------------------------------------------------------------------ */
  const predict = (payload) => api("/predict", { method: "POST", body: payload, timeout: 20000 });

  async function onSubmit(e) {
    e.preventDefault();
    const { payload, errors, warnings } = collect(true);
    if (errors.length) {
      $("#form-summary").textContent = `Fix ${errors.length} field${errors.length === 1 ? "" : "s"} before estimating.`;
      $(`#f-${errors[0].key}`).focus();
      return;
    }
    $("#form-summary").textContent = "";
    const btn = $("#predict-btn");
    setBusy(btn, true, "Scoring…");
    try {
      const res = await predict(payload);
      state.last = { payload, res, warnings };
      renderResult(payload, res, warnings);
    } catch (err) {
      $("#result-body").replaceChildren(callout("err", err.status === 422 ? "The API rejected the input" : "Prediction failed", [err.message]));
      $("#result-badge").textContent = "error";
    } finally {
      setBusy(btn, false);
    }
  }

  function renderResult(payload, res, warnings) {
    const p = res.default_probability, t = res.threshold;
    const high = res.default_prediction === 1;
    const ev = state.evaluation;
    let context = null;
    if (ev) {
      const bin = ev.probability_bins.find((b) => p >= b.lo && (p < b.hi || (b.hi === 1 && p <= 1)));
      if (bin && bin.n) context = el("div", { class: "context-box" }, [
        "Among the ", el("strong", { text: fmt.num(bin.n) }), ` test-set applications the model scored between ${fmt.pct(bin.lo, 0)} and ${fmt.pct(bin.hi, 0)}, `,
        el("strong", { text: fmt.pct(bin.observed_default_rate, 1) }), " actually defaulted.",
        bin.n < 100 ? el("span", { class: "muted" }, [" Small bin — treat this rate as noisy."]) : null,
      ]);
    }
    $("#result-badge").textContent = "live";
    $("#result-body").replaceChildren(el("div", { class: "stack" }, [
      el("div", {}, [
        el("p", { class: "eyebrow", text: "Estimated probability of default" }),
        el("div", { class: "prob-figure" }, [el("b", { text: (p * 100).toFixed(1) }), el("span", { text: "%" })]),
      ]),
      el("div", { class: "scale", role: "img", "aria-label": `Probability ${fmt.pct(p)} on a 0 to 100% scale; decision threshold at ${fmt.pct(t)}` }, [
        el("div", { class: "scale-track" }),
        el("div", { class: "scale-threshold", style: { left: `${t * 100}%` } }, [el("span", { text: `threshold ${fmt.pct(t, 1)}` })]),
        el("div", { class: "scale-marker", style: { left: `${p * 100}%` } }),
        el("div", { class: "scale-ticks" }, ["0%", "25%", "50%", "75%", "100%"].map((x) => el("span", { text: x }))),
      ]),
      el("div", { class: "verdict-row" }, [
        el("span", { class: `verdict-chip ${high ? "high" : "low"}`, text: res.Result }),
        el("span", { class: "hint", text: `API label: probability ${high ? "≥" : "<"} stored threshold ${t.toFixed(3)}` }),
      ]),
      context,
      warnings.length ? callout("warn", "Outside the training range", [`${warnings.join(", ")} ${warnings.length === 1 ? "is" : "are"} outside the values seen in training. The estimate is an extrapolation.`]) : null,
      el("p", { class: "hint", text: "The label is a fixed cut-off on the probability, not a lending decision. Use the sensitivity tool below to see which inputs move this estimate." }),
      el("details", { class: "disclosure" }, [
        el("summary", { text: "Request and response" }),
        el("div", { class: "disclosure-body stack-sm" }, [
          el("pre", { class: "code", text: `POST /predict\n${JSON.stringify(payload, null, 2)}\n\n→ ${JSON.stringify(res, null, 2)}` }),
          el("button", { class: "btn btn-sm", type: "button", text: "Copy JSON", onclick: () => copyText(JSON.stringify({ request: payload, response: res }, null, 2)) }),
        ]),
      ]),
    ]));
  }

  /* ------------------------------------------------------------------ */
  /* Sensitivity (ICE)                                                  */
  /* ------------------------------------------------------------------ */
  function buildWhatIf() {
    $("#whatif-feature").replaceChildren(...FIELDS.map((f) => el("option", { value: f.key, text: `${f.label} (${f.key})` })));
    $("#whatif-feature").value = "loan_int_rate";
    const toggleRatio = () => { $("#whatif-ratio-wrap").hidden = !["person_income", "loan_amnt"].includes($("#whatif-feature").value); };
    $("#whatif-feature").addEventListener("change", toggleRatio);
    toggleRatio();
  }

  async function runWhatIf(e) {
    e.preventDefault();
    const { payload, errors } = collect(true);
    const out = $("#whatif-out");
    if (errors.length) { out.replaceChildren(callout("err", "Complete the application form first", ["The sensitivity curve varies one input of the current, valid application."])); return; }
    const key = $("#whatif-feature").value;
    const f = FIELDS.find((x) => x.key === key);
    const meta = state.schema.features[key];
    let values;
    if (f.type === "cat") values = meta.values;
    else {
      const lo = meta.p01, hi = meta.p99, n = 14;
      values = Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1));
      values.push(payload[key]);
      values = values.map((v) => (f.type === "int" ? Math.round(v) : +v.toFixed(f.step < 1 ? 2 : 0)));
      values = [...new Set(values)].sort((a, b) => a - b);
    }
    const keepRatio = $("#whatif-ratio").checked && ["person_income", "loan_amnt"].includes(key);
    const skipped = [];
    const jobs = values.map((v) => {
      const p = { ...payload, [key]: v };
      if (keepRatio && p.person_income > 0) p.loan_percent_income = Math.round((p.loan_amnt / p.person_income) * 100) / 100;
      if (p.person_emp_length > p.person_age) { skipped.push(v); return null; }
      return { v, p };
    }).filter(Boolean);

    const btn = $("#whatif-btn");
    setBusy(btn, true, `Querying 0 / ${jobs.length}…`);
    const results = [];
    let done = 0;
    try {
      const queue = [...jobs];
      const worker = async () => {
        while (queue.length) {
          const j = queue.shift();
          const r = await predict(j.p);
          results.push({ v: j.v, prob: r.default_probability, t: r.threshold, sent: j.p });
          $(".btn-label", btn).textContent = `Querying ${++done} / ${jobs.length}…`;
        }
      };
      await Promise.all([worker(), worker(), worker(), worker()]);
    } catch (err) {
      setBusy(btn, false);
      out.replaceChildren(callout("err", "Sensitivity run failed", [err.message]));
      return;
    }
    setBusy(btn, false);
    const thr = results[0]?.t ?? 0.5;
    const current = results.find((r) => r.v === payload[key]);
    const chart = el("div");
    if (f.type === "cat") {
      const ordered = meta.values.map((v) => results.find((r) => r.v === v)).filter(Boolean);
      renderBars(chart, ordered.map((r) => ({ label: `${catLabel(r.v)}${r.v === payload[key] ? " (current)" : ""}`, value: r.prob, color: r.prob >= thr ? "var(--risk)" : "var(--accent)", top: r.v === payload[key] })), { ariaLabel: `Default probability by ${f.label}` });
    } else {
      results.sort((a, b) => a.v - b.v);
      const xs = results.map((r) => r.v);
      const xmin = Math.min(...xs), xmax = Math.max(...xs);
      plotXY(chart, [{ name: "p", color: "#2563c9", points: results.map((r) => [r.v, r.prob]), markers: true, highlight: current ? [current.v, current.prob] : null }],
        { xLabel: `${f.label}${f.unit ? ` (${f.unit})` : ""}`, yLabel: "P(default)", xDomain: [xmin, xmax === xmin ? xmin + 1 : xmax], yDomain: [0, 1], height: 300, marks: [{ y: thr, label: `threshold ${thr.toFixed(3)}`, color: "#b4321f" }], ariaLabel: `Default probability as ${f.label} varies` });
    }
    const probs = results.map((r) => r.prob);
    const crossing = results.some((r) => r.prob >= thr) && results.some((r) => r.prob < thr);
    out.replaceChildren(el("div", { class: "whatif-grid" }, [
      el("div", { class: "stack-sm" }, [chart, f.type === "cat" ? null : legend([{ label: "Model output at each queried value", color: "#2563c9" }, { label: "Stored decision threshold", color: "#b4321f" }])]),
      el("div", { class: "stack" }, [
        el("dl", { class: "kv" }, [
          el("dt", { text: "Feature" }), el("dd", { class: "mono", text: key }),
          el("dt", { text: "Queries" }), el("dd", { text: `${results.length} real API calls` }),
          el("dt", { text: "Range of P(default)" }), el("dd", { text: `${fmt.pct(Math.min(...probs))} – ${fmt.pct(Math.max(...probs))}` }),
          el("dt", { text: "Crosses threshold" }), el("dd", { text: crossing ? "Yes — the API label changes within this range" : "No" }),
        ]),
        keepRatio ? el("p", { class: "hint", text: "Loan-to-income ratio was recomputed for each point, as the dataset defines it." }) : null,
        skipped.length ? el("p", { class: "hint", text: `${skipped.length} value(s) skipped because employment length would exceed age.` }) : null,
        el("p", { class: "hint", text: "This shows how the model responds for this one application with everything else held fixed. It is not a causal effect: in real data, inputs such as grade and interest rate move together." }),
      ]),
    ]));
  }

  /* ------------------------------------------------------------------ */
  /* Evaluation                                                         */
  /* ------------------------------------------------------------------ */
  function stat(label, value, sub, prov) {
    return el("div", { class: "stat" }, [el("div", { class: "stat-label", text: label }), el("div", { class: "stat-value", text: value }), sub ? el("div", { class: "stat-sub", text: sub }) : null, prov ? el("div", { class: "provenance", style: { marginTop: "8px" }, text: prov }) : null]);
  }

  function cm2(c) {
    const cell = (v, cls, sub) => el("div", { class: `c ${cls}` }, [fmt.num(v), el("small", { text: sub })]);
    return el("div", { class: "cm2", role: "table", "aria-label": "Confusion matrix at the selected threshold" }, [
      el("div", { class: "h" }), el("div", { class: "h", text: "Predicted repaid" }), el("div", { class: "h", text: "Predicted default" }),
      el("div", { class: "h", text: "Repaid" }), cell(c.tn, "ok", "true negative"), cell(c.fp, "bad", "false positive"),
      el("div", { class: "h", text: "Defaulted" }), cell(c.fn, "bad", "false negative"), cell(c.tp, "ok", "true positive"),
    ]);
  }

  function renderEvaluation() {
    const e = state.evaluation;
    const root = $("#eval-root");
    const st = e.stored_threshold;

    // Threshold explorer (grid of thresholds + the stored one)
    const options = [...e.threshold_sweep, st].sort((a, b) => a.threshold - b.threshold);
    const storedIdx = options.indexOf(st);
    const slider = el("input", { type: "range", class: "range", min: 0, max: options.length - 1, step: 1, value: storedIdx, "aria-label": "Decision threshold" });
    const readout = el("div");
    const drawThr = () => {
      const o = options[+slider.value];
      const isStored = o === st;
      readout.replaceChildren(
        el("p", { class: "row" }, [el("strong", { class: "mono", text: `threshold = ${o.threshold.toFixed(3)}` }), isStored ? el("span", { class: "badge badge-accent", text: "stored in best_threshold.pkl" }) : null]),
        el("dl", { class: "kv" }, [
          el("dt", { text: "Precision" }), el("dd", { text: o.precision.toFixed(3) }),
          el("dt", { text: "Recall" }), el("dd", { text: o.recall.toFixed(3) }),
          el("dt", { text: "F1" }), el("dd", { text: o.f1.toFixed(3) }),
          el("dt", { text: "Accuracy" }), el("dd", { text: o.accuracy.toFixed(3) }),
          el("dt", { text: "Flagged as high risk" }), el("dd", { text: fmt.pct(o.flagged_rate) }),
        ])
      );
      cmBox.replaceChildren(cm2(o.confusion));
    };
    const cmBox = el("div");
    slider.addEventListener("input", drawThr);
    drawThr();

    const pr = el("div"), roc = el("div"), cal = el("div"), imp = el("div");
    plotXY(pr, [{ name: "PR", color: "#2563c9", points: e.curves.pr, highlight: [st.recall, st.precision] }], { xLabel: "Recall", yLabel: "Precision", height: 280, ariaLabel: "Precision–recall curve" });
    plotXY(roc, [{ name: "chance", color: "#b8bfc8", points: [[0, 0], [1, 1]], dashed: true, width: 1.2 }, { name: "ROC", color: "#12355b", points: e.curves.roc }], { xLabel: "False positive rate", yLabel: "True positive rate", height: 280, ariaLabel: "ROC curve" });
    plotXY(cal, [{ name: "ideal", color: "#b8bfc8", points: [[0, 0], [1, 1]], dashed: true, width: 1.2 }, { name: "model", color: "#2563c9", points: e.probability_bins.filter((b) => b.n).map((b) => [b.mean_predicted, b.observed_default_rate]), markers: true }], { xLabel: "Mean predicted probability (bin)", yLabel: "Observed default rate", height: 280, ariaLabel: "Calibration curve" });
    const feats = e.permutation_importance.features;
    const maxImp = Math.max(...feats.map((f) => f.mean), 1e-9);
    renderBars(imp, feats.map((f) => ({ label: f.feature, value: Math.max(0, f.mean) / maxImp, raw: f })), { valueFormat: (v) => (v * maxImp).toFixed(3), ariaLabel: "Permutation importance (drop in ROC-AUC)" });

    const nb = e.notebook_recorded;
    const nbRows = [
      ...Object.entries(nb.cv_5fold_train).map(([m, r]) => ({ model: m, setting: "5-fold CV on train", ...r })),
      ...Object.entries(nb.test_default_threshold).map(([m, r]) => ({ model: m, setting: "test split, default threshold", ...r })),
    ];
    const f3 = (v) => (v === undefined ? "—" : Number.isInteger(Math.round(v * 1e6) / 1e4) ? v.toFixed(2) : v.toFixed(3));

    root.replaceChildren(
      el("div", { class: "grid-4" }, [
        stat("ROC-AUC", e.ranking.roc_auc.toFixed(3), "Ranking quality, threshold-free", "re-evaluated"),
        stat("PR-AUC (average precision)", e.ranking.pr_auc.toFixed(3), `Default rate in test split ${fmt.pct(e.data.test_positives / e.data.test_rows)}`, "re-evaluated"),
        stat("Brier score", e.ranking.brier.toFixed(3), "Mean squared probability error (lower is better)", "re-evaluated"),
        stat("Recall at stored threshold", st.recall.toFixed(3), `Precision ${st.precision.toFixed(3)} · ${fmt.num(st.confusion.fn)} defaults missed`, "re-evaluated"),
      ]),
      el("div", { class: "eval-grid" }, [
        el("div", { class: "panel span-all" }, [
          el("div", { class: "panel-head" }, [el("h3", { class: "panel-title", text: "Threshold explorer" }), el("span", { class: "panel-sub", text: `${fmt.num(e.data.test_rows)} held-out applications · ${fmt.num(e.data.test_positives)} defaults` })]),
          el("div", { class: "panel-body thr-layout" }, [
            el("div", { class: "stack" }, [slider, el("div", { class: "scale-ticks" }, [el("span", { text: options[0].threshold.toFixed(2) }), el("span", { text: options[options.length - 1].threshold.toFixed(2) })]), readout]),
            el("div", { class: "stack-sm" }, [cmBox, el("p", { class: "hint", text: `Best F1 on the 0.05-step grid: ${e.best_f1_on_grid.f1.toFixed(3)} at ${e.best_f1_on_grid.threshold.toFixed(2)}. The stored threshold gives F1 ${st.f1.toFixed(3)}.` })]),
          ]),
        ]),
        el("div", { class: "panel" }, [
          el("div", { class: "panel-head" }, [el("h3", { class: "panel-title", text: "Precision–recall curve" }), el("span", { class: "provenance", text: "re-evaluated" })]),
          el("div", { class: "panel-body stack-sm" }, [pr, el("p", { class: "hint", text: "The ringed point is the stored decision threshold." })]),
        ]),
        el("div", { class: "panel" }, [
          el("div", { class: "panel-head" }, [el("h3", { class: "panel-title", text: "ROC curve" }), el("span", { class: "provenance", text: "re-evaluated" })]),
          el("div", { class: "panel-body" }, [roc]),
        ]),
        el("div", { class: "panel" }, [
          el("div", { class: "panel-head" }, [el("h3", { class: "panel-title", text: "Calibration" }), el("span", { class: "provenance", text: "re-evaluated" })]),
          el("div", { class: "panel-body stack-sm" }, [cal, legend([{ label: "Model (10 equal-width bins)", color: "#2563c9" }, { label: "Perfect calibration", color: "#b8bfc8" }]),
            el("p", { class: "hint", text: `Points below the diagonal mean the model overestimates risk in that range. Bin sizes: ${e.probability_bins.map((b) => b.n).join(", ")} — most applications fall in the lowest and highest bins.` })]),
        ]),
        el("div", { class: "panel" }, [
          el("div", { class: "panel-head" }, [el("h3", { class: "panel-title", text: "Permutation importance" }), el("span", { class: "provenance", text: "re-evaluated" })]),
          el("div", { class: "panel-body stack-sm" }, [imp, el("p", { class: "hint", text: "Drop in test ROC-AUC when one input column is shuffled (mean of 5 repeats, fixed seed). It measures reliance on each raw input, not causal influence; correlated inputs (e.g. grade and interest rate) share credit." })]),
        ]),
        el("div", { class: "panel span-all" }, [
          el("div", { class: "panel-head" }, [el("h3", { class: "panel-title", text: "Recorded during development" }), el("span", { class: "provenance", text: "notebook" })]),
          el("div", { class: "table-wrap" }, [el("table", { class: "table" }, [
            el("thead", {}, [el("tr", {}, ["Model", "Setting", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1"].map((h, i) => el("th", { class: i > 1 ? "num" : "", text: h })))]),
            el("tbody", {}, nbRows.map((r) => el("tr", {}, [
              el("td", { text: r.model }), el("td", { class: "muted", text: r.setting }),
              el("td", { class: "num", text: r.roc_auc ? r.roc_auc.toFixed(3) : "—" }), el("td", { class: "num", text: f3(r.accuracy) }),
              el("td", { class: "num", text: f3(r.precision) }), el("td", { class: "num", text: f3(r.recall) }), el("td", { class: "num", text: f3(r.f1) }),
            ]))),
          ])]),
          el("div", { class: "panel-foot", text: `${nb.tuning}. The notebook printed test metrics to two decimals. The deployed artifact is the calibrated version of the tuned model, evaluated above.` }),
        ]),
      ]),
      el("p", { class: "hint", style: { marginTop: "16px" }, text: `Generated ${new Date(e.generated_at).toLocaleString()} by ${e.script}. Data: ${fmt.num(e.data.raw_rows)} rows → ${fmt.num(e.data.rows_after_cleaning)} after cleaning → ${fmt.num(e.data.train_rows)} train / ${fmt.num(e.data.test_rows)} test.` })
    );
  }

  /* ------------------------------------------------------------------ */
  /* Boot                                                               */
  /* ------------------------------------------------------------------ */
  async function boot() {
    try {
      [state.schema, state.evaluation] = await Promise.all([
        api("data/schema.json", { timeout: 10000 }),
        api("data/evaluation.json", { timeout: 10000 }).catch(() => null),
      ]);
    } catch (err) {
      $("#loan-form").replaceChildren(el("div", { class: "panel-body" }, [callout("err", "Feature schema not found", ["static/data/schema.json is missing. Generate it with: python evaluation/evaluate_artifact.py", err.message])]));
      return;
    }
    buildForm();
    buildWhatIf();
    if (state.evaluation) renderEvaluation();
    else $("#eval-root").replaceChildren(callout("warn", "Evaluation results not found", ["Generate them with: python evaluation/evaluate_artifact.py"]));

    // Same liveness probe as the previous UI: the API has no /health route, so check the OpenAPI document.
    try {
      await api("/openapi.json", { timeout: 6000 });
      setStatus($("#api-status"), "ok", "Model API online");
    } catch {
      setStatus($("#api-status"), "err", "API unreachable");
    }
  }

  $("#loan-form").addEventListener("submit", onSubmit);
  $("#fill-typical").addEventListener("click", () => { if (state.schema) fillTypical(); });
  $("#whatif-form").addEventListener("submit", runWhatIf);
  boot();
})();
