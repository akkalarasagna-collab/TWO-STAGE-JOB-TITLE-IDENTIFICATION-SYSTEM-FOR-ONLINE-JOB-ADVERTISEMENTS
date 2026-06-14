/**
 * main.js — Two Stage Job Title Identification System
 *
 * Handles:
 *  - AJAX form submission (no page reload)
 *  - Cycling loading messages (Stage 1 → DistilBERT → Skills → Score)
 *  - Dynamic rendering of all result cards
 *  - Animated confidence progress bar
 *  - Graceful error handling
 */

'use strict';

/* ── DOM references ──────────────────────────────────────────────────────── */
const jobDescriptionEl  = document.getElementById('jobDescription');
const analyzeBtnEl      = document.getElementById('analyzeBtn');
const loadingSpinnerEl  = document.getElementById('loadingSpinner');
const loadingMsgEl      = document.getElementById('loadingMsg');
const errorAlertEl      = document.getElementById('errorAlert');
const errorMessageEl    = document.getElementById('errorMessage');
const resultsSectionEl  = document.getElementById('resultsSection');

// Result card elements
const stageBadgeEl       = document.getElementById('stageBadge');
const predictedTitleEl   = document.getElementById('predictedTitle');
const jobCategoryEl      = document.getElementById('jobCategory');
const confidenceLabelEl  = document.getElementById('confidenceLabel');
const confidenceBarEl    = document.getElementById('confidenceBar');
const alternativesCardEl = document.getElementById('alternativesCard');
const alternativesListEl = document.getElementById('alternativesList');
const technicalSkillsEl  = document.getElementById('technicalSkills');
const softSkillsEl       = document.getElementById('softSkills');
const noTechSkillsEl     = document.getElementById('noTechSkills');
const noSoftSkillsEl     = document.getElementById('noSoftSkills');

/* ── Loading message cycling ─────────────────────────────────────────────── */
const loadingMessages = [
  'Checking for explicit job title...',
  'Running semantic zero-shot classification...',
  'Extracting skills from description...',
  'Calculating confidence score...',
];

let _msgInterval = null;

function startLoadingMessages() {
  if (!loadingMsgEl) return;
  let idx = 0;
  loadingMsgEl.textContent = loadingMessages[0];
  _msgInterval = setInterval(() => {
    idx = (idx + 1) % loadingMessages.length;
    loadingMsgEl.textContent = loadingMessages[idx];
  }, 1800);
}

function stopLoadingMessages() {
  if (_msgInterval !== null) {
    clearInterval(_msgInterval);
    _msgInterval = null;
  }
  if (loadingMsgEl) loadingMsgEl.textContent = loadingMessages[0];
}


/* ── Main analysis function ──────────────────────────────────────────────── */
async function analyzeJob() {
  const description = jobDescriptionEl.value.trim();

  // Client-side validation
  if (!description) {
    showError('Please paste a job description before analyzing.');
    return;
  }
  if (description.length < 20) {
    showError('Job description is too short. Please provide more detail.');
    return;
  }

  // UI: loading state
  hideError();
  hideResults();
  setLoading(true);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ description }),
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      showError(data.error || 'An unexpected error occurred. Please try again.');
      return;
    }

    renderResults(data);

  } catch (networkError) {
    showError('Network error: Unable to reach the server. Is Flask running?');
  } finally {
    setLoading(false);
  }
}


/* ── Render all result cards ─────────────────────────────────────────────── */
function renderResults(data) {

  // ── Stage badge ───────────────────────────────────────────────────────────
  // Use detection_stage from the server response if present; fall back to
  // the numeric stage field for backward compatibility.
  if (data.stage === 1) {
    stageBadgeEl.textContent = '⚡ Stage 1: Pattern Match';
    stageBadgeEl.className   = 'badge fs-6 px-3 py-2 badge-stage1';
  } else {
    stageBadgeEl.textContent = '🤖 Stage 2: Semantic Match (MiniLM)';
    stageBadgeEl.className   = 'badge fs-6 px-3 py-2 badge-stage2';
  }

  // Show full detection_stage string as a tooltip / title attribute
  if (data.detection_stage) {
    stageBadgeEl.title = data.detection_stage;
  }

  // ── Job title & category ──────────────────────────────────────────────────
  predictedTitleEl.textContent = data.job_title   || '—';
  jobCategoryEl.textContent    = data.job_category || '—';

  // ── Confidence bar (animate after short delay) ────────────────────────────
  const confidence = parseFloat(data.confidence) || 0;
  confidenceLabelEl.textContent = confidence.toFixed(1) + '%';

  // Reset bar width first so the CSS transition fires on every call
  confidenceBarEl.style.width = '0%';
  confidenceBarEl.setAttribute('aria-valuenow', 0);

  // Colour by confidence tier
  if (confidence >= 75) {
    confidenceBarEl.style.background = 'linear-gradient(90deg, #198754, #20c997)';
  } else if (confidence >= 45) {
    confidenceBarEl.style.background = 'linear-gradient(90deg, #fd7e14, #ffc107)';
  } else {
    confidenceBarEl.style.background = 'linear-gradient(90deg, #dc3545, #e87070)';
  }

  // Trigger animation on next tick
  setTimeout(() => {
    confidenceBarEl.style.width = confidence + '%';
    confidenceBarEl.setAttribute('aria-valuenow', confidence);
  }, 100);

  // ── Alternatives ──────────────────────────────────────────────────────────
  alternativesListEl.innerHTML = '';
  if (data.alternatives && data.alternatives.length > 0) {
    alternativesCardEl.classList.remove('d-none');
    data.alternatives.forEach((alt, idx) => {
      // BERT alternatives use 'confidence'; LR used 'probability' — support both
      const pct = parseFloat(alt.probability ?? alt.confidence) || 0;
      alternativesListEl.insertAdjacentHTML('beforeend', `
        <div class="alt-item">
          <div class="d-flex justify-content-between align-items-center">
            <span class="alt-title">${idx + 1}. ${escapeHtml(alt.title)}</span>
          </div>
          <div class="alt-bar-wrap mt-1">
            <div class="alt-bar" id="altBar${idx}" style="width: 0%"></div>
          </div>
          <div class="alt-pct">${pct.toFixed(1)}% probability</div>
        </div>
      `);
      setTimeout(() => {
        const bar = document.getElementById(`altBar${idx}`);
        if (bar) bar.style.width = pct + '%';
      }, 200 + idx * 100);
    });
  } else {
    alternativesCardEl.classList.add('d-none');
  }

  // ── Technical skills ──────────────────────────────────────────────────────
  technicalSkillsEl.innerHTML = '';
  if (data.technical_skills && data.technical_skills.length > 0) {
    noTechSkillsEl.classList.add('d-none');
    data.technical_skills.forEach(skill => {
      technicalSkillsEl.insertAdjacentHTML('beforeend',
        `<span class="skill-badge skill-tech">
           <i class="fa-solid fa-code fa-xs"></i>${escapeHtml(skill)}
         </span>`
      );
    });
  } else {
    noTechSkillsEl.classList.remove('d-none');
  }

  // ── Soft skills ───────────────────────────────────────────────────────────
  softSkillsEl.innerHTML = '';
  if (data.soft_skills && data.soft_skills.length > 0) {
    noSoftSkillsEl.classList.add('d-none');
    data.soft_skills.forEach(skill => {
      softSkillsEl.insertAdjacentHTML('beforeend',
        `<span class="skill-badge skill-soft">
           <i class="fa-solid fa-star fa-xs"></i>${escapeHtml(skill)}
         </span>`
      );
    });
  } else {
    noSoftSkillsEl.classList.remove('d-none');
  }

  // ── Show results with fade-in animation ───────────────────────────────────
  resultsSectionEl.classList.remove('d-none');
  resultsSectionEl.classList.add('fade-in');
  resultsSectionEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
}


/* ── UI helper functions ─────────────────────────────────────────────────── */

function setLoading(isLoading) {
  if (isLoading) {
    loadingSpinnerEl.classList.remove('d-none');
    analyzeBtnEl.disabled  = true;
    analyzeBtnEl.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i>Analyzing...';
    startLoadingMessages();
  } else {
    stopLoadingMessages();
    loadingSpinnerEl.classList.add('d-none');
    analyzeBtnEl.disabled  = false;
    analyzeBtnEl.innerHTML = '<i class="fa-solid fa-magnifying-glass-chart me-2"></i>Analyze Job Description';
  }
}

function showError(message) {
  errorMessageEl.textContent = message;
  errorAlertEl.classList.remove('d-none');
  errorAlertEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
  errorAlertEl.classList.add('d-none');
  errorMessageEl.textContent = '';
}

function hideResults() {
  resultsSectionEl.classList.add('d-none');
  resultsSectionEl.classList.remove('fade-in');
}

/** Prevent XSS when inserting user-derived strings into innerHTML */
function escapeHtml(str) {
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
  return String(str).replace(/[&<>"']/g, ch => map[ch]);
}


/* ── Allow Ctrl+Enter to submit ──────────────────────────────────────────── */
jobDescriptionEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    analyzeJob();
  }
});
