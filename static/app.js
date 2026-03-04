/**
 * Hybrid Grader AI — Frontend Logic
 *
 * Handles form submission, constructs multipart/form-data, calls
 * POST /api/v1/evaluate-full, and renders the FinalEvaluationResponse
 * onto the results panel.
 */

"use strict";

// ─── Constants ───────────────────────────────────────────────────────────────

const API_ENDPOINT = "/api/v1/evaluate-full";

/** Score-ring SVG geometry (r = 52, circumference = 2π×52) */
const RING_CIRCUMFERENCE = 326.73;

/** Default rubric for the Photosynthesis example question. */
const DEFAULT_RUBRIC = {
  nodes: [
    { id: "photosynthesis",  label: "Photosynthesis",  weight: 1.5 },
    { id: "chlorophyll",     label: "Chlorophyll",      weight: 1.0 },
    { id: "sunlight",        label: "Sunlight",         weight: 1.0 },
    { id: "carbon_dioxide",  label: "Carbon Dioxide",   weight: 1.0 },
    { id: "water",           label: "Water",            weight: 0.8 },
    { id: "glucose",         label: "Glucose",          weight: 1.2 },
    { id: "oxygen",          label: "Oxygen",           weight: 0.8 },
    { id: "chloroplast",     label: "Chloroplast",      weight: 1.0 },
  ],
  edges: [
    { source: "sunlight",       target: "photosynthesis", relationship: "drives"    },
    { source: "chlorophyll",    target: "photosynthesis", relationship: "enables"   },
    { source: "carbon_dioxide", target: "photosynthesis", relationship: "reactant"  },
    { source: "water",          target: "photosynthesis", relationship: "reactant"  },
    { source: "photosynthesis", target: "glucose",        relationship: "produces"  },
    { source: "photosynthesis", target: "oxygen",         relationship: "produces"  },
    { source: "chloroplast",    target: "chlorophyll",    relationship: "contains"  },
  ],
};

// ─── DOM references ───────────────────────────────────────────────────────────

const form            = document.getElementById("eval-form");
const fileInput       = document.getElementById("file-input");
const fileLabelContent = document.getElementById("file-label-content");
const alphaSlider     = document.getElementById("alpha");
const alphaValueLabel = document.getElementById("alpha-value");
const rubricTextarea  = document.getElementById("rubric-json");
const submitBtn       = document.getElementById("submit-btn");

// Right-column panels
const placeholderPanel = document.getElementById("placeholder-panel");
const loadingPanel     = document.getElementById("loading-panel");
const errorPanel       = document.getElementById("error-panel");
const errorMsg         = document.getElementById("error-msg");
const resultsPanel     = document.getElementById("results-panel");

// Result DOM nodes
const extractedTextEl  = document.getElementById("extracted-text");
const kgScoreEl        = document.getElementById("kg-score");
const kgBarEl          = document.getElementById("kg-bar");
const kgScorePill      = document.getElementById("kg-score-pill");
const matchedConceptsEl = document.getElementById("matched-concepts");
const missingConceptsEl = document.getElementById("missing-concepts");
const coherenceScoreEl = document.getElementById("coherence-score");
const coherenceBarEl   = document.getElementById("coherence-bar");
const correctnessScoreEl = document.getElementById("correctness-score");
const correctnessBarEl = document.getElementById("correctness-bar");
const justificationEl  = document.getElementById("justification");
const finalScoreTextEl = document.getElementById("final-score-text");
const scoreRingEl      = document.getElementById("score-ring");
const llmScorePill     = document.getElementById("llm-score-pill");

// ─── Initialisation ───────────────────────────────────────────────────────────

/** Pre-fill the rubric textarea with the photosynthesis example. */
rubricTextarea.value = JSON.stringify(DEFAULT_RUBRIC, null, 2);

/** Sync slider gradient and label on load and on input. */
function syncSlider(val) {
  const pct = val * 100;
  alphaSlider.style.background =
    `linear-gradient(to right, #6366f1 0%, #6366f1 ${pct}%, #374151 ${pct}%, #374151 100%)`;
  alphaValueLabel.textContent = parseFloat(val).toFixed(2);
}
syncSlider(alphaSlider.value);

alphaSlider.addEventListener("input", () => syncSlider(alphaSlider.value));

/** Show selected filename in the file drop zone. */
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  fileLabelContent.innerHTML = `
    <svg class="w-6 h-6 text-green-400 mx-auto mb-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
      <path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
    </svg>
    <p class="text-sm font-medium text-green-400 truncate max-w-xs">${escapeHtml(file.name)}</p>
    <p class="text-xs text-gray-500 mt-0.5">${formatBytes(file.size)}</p>`;
});

// ─── Panel helpers ────────────────────────────────────────────────────────────

function showPanel(name) {
  for (const [panelName, el] of [
    ["placeholder", placeholderPanel],
    ["loading",     loadingPanel],
    ["error",       errorPanel],
    ["results",     resultsPanel],
  ]) {
    el.classList.toggle("hidden", panelName !== name);
  }
}

// ─── Form submission ──────────────────────────────────────────────────────────

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Validate file selection (native required attribute handles empty, but belt-and-braces)
  if (!fileInput.files || fileInput.files.length === 0) {
    showError("Please select an answer-sheet image before submitting.");
    return;
  }

  // Validate rubric JSON is parseable before sending
  let parsedRubric;
  try {
    parsedRubric = JSON.parse(rubricTextarea.value);
  } catch {
    showError("Rubric JSON is not valid JSON. Please fix the syntax and try again.");
    return;
  }
  if (!parsedRubric.nodes || !parsedRubric.edges) {
    showError('Rubric JSON must contain "nodes" and "edges" arrays.');
    return;
  }

  // Build multipart form data
  const formData = new FormData();
  formData.append("file",       fileInput.files[0]);
  formData.append("question",   document.getElementById("question").value.trim());
  formData.append("rubric_json", rubricTextarea.value);
  formData.append("alpha",      alphaSlider.value);

  // Disable submit, show loading
  submitBtn.disabled = true;
  submitBtn.classList.add("opacity-60", "cursor-not-allowed");
  showPanel("loading");

  try {
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let detail = `HTTP ${response.status}: ${response.statusText}`;
      try {
        const errBody = await response.json();
        if (errBody.detail) detail = errBody.detail;
      } catch { /* ignore parse failure */ }
      throw new Error(detail);
    }

    /** @type {FinalEvaluationResponse} */
    const data = await response.json();
    renderResults(data);
    showPanel("results");

  } catch (err) {
    showError(err.message || "An unexpected error occurred. Please try again.");
  } finally {
    submitBtn.disabled = false;
    submitBtn.classList.remove("opacity-60", "cursor-not-allowed");
  }
});

// ─── Result rendering ─────────────────────────────────────────────────────────

/**
 * Maps a FinalEvaluationResponse onto all result DOM nodes.
 *
 * @param {Object} data - FinalEvaluationResponse JSON object.
 * @param {string}  data.extracted_text
 * @param {{coverage_score: number, matched_concepts: string[], missing_concepts: string[]}} data.kg_result
 * @param {{coherence_score: number, correctness_score: number, justification: string}} data.llm_result
 * @param {number}  data.final_score
 */
function renderResults(data) {
  const { extracted_text, kg_result, llm_result, final_score } = data;

  // ── Extracted Text
  extractedTextEl.textContent = extracted_text || "(No text extracted)";

  // ── KG Results
  const kgPct = pct(kg_result.coverage_score);
  kgScoreEl.textContent  = kgPct;
  kgScorePill.textContent = kgPct;
  // Defer bar width for CSS transition to fire
  requestAnimationFrame(() => { kgBarEl.style.width = kgPct; });

  renderBadges(matchedConceptsEl, kg_result.matched_concepts, "matched");
  renderBadges(missingConceptsEl, kg_result.missing_concepts, "missing");

  // ── LLM Results
  const coherencePct    = pct(llm_result.coherence_score);
  const correctnessPct  = pct(llm_result.correctness_score);
  coherenceScoreEl.textContent   = coherencePct;
  correctnessScoreEl.textContent = correctnessPct;
  requestAnimationFrame(() => {
    coherenceBarEl.style.width   = coherencePct;
    correctnessBarEl.style.width = correctnessPct;
  });
  justificationEl.textContent = llm_result.justification;

  // LLM pill (average of coherence + correctness)
  const llmAvg = (llm_result.coherence_score + llm_result.correctness_score) / 2;
  llmScorePill.textContent = pct(llmAvg);

  // ── Final Score ring
  const finalPct = Math.round(final_score * 100);
  finalScoreTextEl.textContent = `${finalPct}%`;
  // Animate ring: dashoffset shrinks as score rises
  requestAnimationFrame(() => {
    scoreRingEl.style.strokeDashoffset =
      (RING_CIRCUMFERENCE * (1 - final_score)).toFixed(2);
  });

  // Colour the score text based on performance
  finalScoreTextEl.className = `text-4xl font-bold tabular-nums ${scoreColour(final_score)}`;
}

/**
 * Renders concept-ID badges into a container element.
 * @param {HTMLElement} container
 * @param {string[]} concepts
 * @param {"matched"|"missing"} type
 */
function renderBadges(container, concepts, type) {
  container.innerHTML = "";
  if (!concepts.length) {
    const empty = document.createElement("span");
    empty.className = "text-xs text-gray-600 italic";
    empty.textContent = "None";
    container.appendChild(empty);
    return;
  }

  const isMatched = type === "matched";
  concepts.forEach((concept, i) => {
    const badge = document.createElement("span");
    badge.className = [
      "badge inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium",
      isMatched
        ? "bg-green-500/15 text-green-300 ring-1 ring-inset ring-green-500/30"
        : "bg-red-500/15 text-red-300 ring-1 ring-inset ring-red-500/30",
    ].join(" ");
    badge.style.animationDelay = `${i * 40}ms`;

    const dot = document.createElement("span");
    dot.className = `w-1.5 h-1.5 rounded-full ${isMatched ? "bg-green-400" : "bg-red-400"}`;
    badge.appendChild(dot);
    badge.appendChild(document.createTextNode(concept));
    container.appendChild(badge);
  });
}

// ─── Error helper ─────────────────────────────────────────────────────────────

function showError(message) {
  errorMsg.textContent = message;
  showPanel("error");
}

// ─── Utility ─────────────────────────────────────────────────────────────────

/** Format a 0–1 float as a percentage string, e.g. "73.5%". */
function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

/** Pick a Tailwind text-colour class based on score magnitude. */
function scoreColour(score) {
  if (score >= 0.75) return "text-green-400";
  if (score >= 0.5)  return "text-yellow-400";
  return "text-red-400";
}

/** Escape HTML special characters to prevent XSS. */
function escapeHtml(str) {
  return str.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

/** Format bytes to a human-readable string. */
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
