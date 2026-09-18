/**
 * analyzer.js — Core analysis logic for the main page
 * Handles API calls, loading states, result rendering, and history storage.
 */

const API_BASE = 'http://localhost:8000';

// ─── QUICK EXAMPLE DATA ──────────────────────────────────────

const EXAMPLES = [
  {
    label: '✅ Reliable',
    type: 'safe',
    text: 'Researchers at MIT published a peer-reviewed study in the journal Nature showing that regular moderate exercise significantly improves memory consolidation and cognitive function in adults over 40, with effects observed after just 12 weeks.'
  },
  {
    label: '⚠️ Suspicious',
    type: 'warn',
    text: 'Scientists have PROVEN that 5G towers are secretly designed to track your location and the government doesn\'t want you to know. Share before they delete this! The deep state is hiding the truth!'
  },
  {
    label: '🎯 Clickbait',
    type: 'danger',
    text: 'You Won\'t BELIEVE What This Doctor Discovered!! 10 Signs You\'re Doing Everything WRONG — Number 7 Will Shock You! Doctors HATE This Simple Trick That Cures Belly Fat Overnight!'
  },
  {
    label: '🌍 Hindi',
    type: 'safe',
    text: 'नई दिल्ली: भारत सरकार ने आज एक नई स्वास्थ्य नीति की घोषणा की जिसके तहत ग्रामीण क्षेत्रों में स्वास्थ्य सेवाओं को बेहतर बनाने के लिए 500 करोड़ रुपये का बजट आवंटित किया गया है।'
  },
  {
    label: '🇫🇷 Français',
    type: 'warn',
    text: 'CHOQUANT: Les chercheurs CACHENT la vérité sur les vaccins! Le gouvernement ne veut pas que vous sachiez! Partagez avant que ce soit supprimé! La vérité sur Big Pharma révélée!'
  },
  {
    label: '🇪🇸 Español',
    type: 'safe',
    text: 'El gobierno de España ha anunciado nuevas medidas para reducir las emisiones de carbono en un 40% para 2030, siguiendo las recomendaciones del Panel Intergubernamental sobre Cambio Climático.'
  },
];

// ─── STATE ───────────────────────────────────────────────────

let currentResult = null;
let isAnalyzing = false;

// ─── DOM REFS ─────────────────────────────────────────────────

const textInput     = document.getElementById('text-input');
const charCount     = document.getElementById('char-count');
const langSelect    = document.getElementById('lang-select');
const analyzeBtn    = document.getElementById('analyze-btn');
const resultPanel   = document.getElementById('result-panel');
const apiStatusDot  = document.getElementById('api-status-dot');
const apiStatusText = document.getElementById('api-status-text');

// ─── INIT ─────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  renderExampleChips();
  loadSupportedLanguages();
  checkApiHealth();
  updateCharCount();
  if (textInput) textInput.addEventListener('input', updateCharCount);
  if (analyzeBtn) analyzeBtn.addEventListener('click', handleAnalyze);
});

// ─── CHAR COUNT ────────────────────────────────────────────────

function updateCharCount() {
  if (!textInput || !charCount) return;
  const len = textInput.value.length;
  charCount.textContent = `${len.toLocaleString()} / 10,000 characters`;
  charCount.style.color = len > 9000 ? 'var(--color-danger)' : len > 7000 ? 'var(--color-warning)' : 'var(--text-muted)';
}

// ─── API HEALTH CHECK ──────────────────────────────────────────

async function checkApiHealth() {
  if (!apiStatusDot || !apiStatusText) return;
  apiStatusDot.className = 'api-dot checking';
  apiStatusText.textContent = 'Checking...';
  try {
    const r = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      const data = await r.json();
      apiStatusDot.className = 'api-dot online';
      const misinfo = data.models?.misinformation || 'unknown';
      const mode = misinfo === 'loaded' ? 'ML Mode' : 'Heuristic Mode';
      apiStatusText.textContent = `API Online · ${mode}`;
    } else {
      throw new Error('bad status');
    }
  } catch {
    apiStatusDot.className = 'api-dot offline';
    apiStatusText.textContent = 'API Offline — run: uvicorn app:app';
  }
}

// ─── LOAD LANGUAGES ────────────────────────────────────────────

async function loadSupportedLanguages() {
  if (!langSelect) return;
  try {
    const r = await fetch(`${API_BASE}/supported-languages`);
    if (!r.ok) return;
    const { languages } = await r.json();
    languages.forEach(lang => {
      const opt = document.createElement('option');
      opt.value = lang.code;
      opt.textContent = `${lang.flag} ${lang.name}`;
      langSelect.appendChild(opt);
    });
  } catch {
    // API offline — default options already in HTML
  }
}

// ─── EXAMPLE CHIPS ─────────────────────────────────────────────

function renderExampleChips() {
  const wrap = document.getElementById('quick-chips');
  if (!wrap) return;
  EXAMPLES.forEach((ex, i) => {
    const chip = document.createElement('button');
    chip.className = `quick-chip chip-${ex.type}`;
    chip.title = ex.text;
    chip.textContent = ex.label;
    chip.setAttribute('id', `chip-${i}`);
    chip.addEventListener('click', () => {
      if (textInput) {
        textInput.value = ex.text;
        updateCharCount();
        textInput.focus();
        textInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    });
    wrap.appendChild(chip);
  });
}

// ─── ANALYZE HANDLER ───────────────────────────────────────────

async function handleAnalyze() {
  const text = textInput?.value?.trim();
  if (!text) {
    showToast('Please enter some text to analyze.', 'warn');
    textInput?.focus();
    return;
  }
  if (text.length < 5) {
    showToast('Text is too short. Please enter at least 5 characters.', 'warn');
    return;
  }
  if (isAnalyzing) return;

  isAnalyzing = true;
  setAnalyzeLoading(true);
  showResultLoading();

  const langOverride = langSelect?.value || null;

  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        language_override: langOverride || null,
      }),
      signal: AbortSignal.timeout(15000),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${response.status}`);
    }

    const result = await response.json();
    currentResult = result;
    saveToHistory(result);
    renderResult(result);
  } catch (err) {
    if (err.name === 'TimeoutError') {
      showToast('Request timed out. Make sure the API server is running.', 'error');
    } else if (err.message.includes('fetch') || err.message.includes('Failed')) {
      showToast('Cannot reach API. Run: cd backend && uvicorn app:app --reload', 'error');
    } else {
      showToast(`Error: ${err.message}`, 'error');
    }
    showResultEmpty();
  } finally {
    isAnalyzing = false;
    setAnalyzeLoading(false);
  }
}

// ─── LOADING STATES ────────────────────────────────────────────

function setAnalyzeLoading(loading) {
  if (!analyzeBtn) return;
  if (loading) {
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="spinner"></span> Analyzing...';
  } else {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<i data-lucide="zap" style="width:18px;height:18px;"></i> Analyze Text';
    if (window.lucide) lucide.createIcons();
  }
}

function showResultLoading() {
  if (!resultPanel) return;
  resultPanel.innerHTML = `
    <div class="scan-overlay" style="position:relative; min-height:300px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:20px;">
      <div class="scan-line"></div>
      <div style="font-size: 2rem; animation: float 2s ease-in-out infinite;">🔍</div>
      <p class="scan-text">Analyzing content across languages...</p>
      <div style="width:100%; max-width:260px; display:flex; flex-direction:column; gap:10px; padding: 0 20px;">
        <div class="skeleton skeleton-line" style="height:10px; border-radius:9999px;"></div>
        <div class="skeleton skeleton-line" style="height:10px; border-radius:9999px; width:80%;"></div>
        <div class="skeleton skeleton-line" style="height:10px; border-radius:9999px; width:60%;"></div>
      </div>
    </div>
  `;
}

function showResultEmpty() {
  if (!resultPanel) return;
  resultPanel.innerHTML = `
    <div class="result-empty">
      <div class="result-empty-icon">🛡️</div>
      <p class="result-empty-text">Enter text and click Analyze to detect<br>misinformation and clickbait.</p>
    </div>
  `;
}

// ─── RESULT RENDERER ───────────────────────────────────────────

function renderResult(result) {
  if (!resultPanel) return;

  const { language, misinformation, clickbait, overall, highlighted_spans, processing_time_ms } = result;

  // Verdict styling
  const verdictColorMap = { safe: '#10b981', warning: '#f59e0b', danger: '#ef4444' };
  const verdictIconMap  = { safe: '✅', warning: '⚠️', danger: '🚨' };
  const riskPct = Math.round(overall.risk_score * 100);

  // Build highlighted text
  const highlightedText = buildHighlightedText(result.text_preview, highlighted_spans);

  // Confidence ring offset (circumference 314)
  const misinfoPct = Math.round(misinformation.confidence * 100);
  const clickbaitPct = Math.round(clickbait.confidence * 100);
  const misinfoOffset = 314 - (314 * misinformation.confidence);
  const ringColor = overall.color === 'safe' ? '#10b981' : overall.color === 'warning' ? '#f59e0b' : '#ef4444';

  // Reasoning HTML
  const reasoningHTML = misinformation.reasoning.map(r => `
    <div class="reasoning-chip">
      <span class="reasoning-chip-dot"></span>
      <span>${escHtml(r)}</span>
    </div>
  `).join('');

  // Red flag phrases
  const redFlagHTML = [...(misinformation.red_flag_phrases || []), ...(clickbait.detected_patterns || [])]
    .slice(0, 6)
    .map(p => `<span class="badge badge-danger" style="margin:2px;">${escHtml(p)}</span>`)
    .join('');

  resultPanel.innerHTML = `
    <div class="result-reveal">
      <!-- Language -->
      <div class="result-lang">
        <span class="lang-flag">${language.flag}</span>
        <div class="lang-info">
          <div class="lang-name">${escHtml(language.name)}</div>
          <div class="lang-confidence">Detected with ${Math.round(language.confidence * 100)}% confidence · ${processing_time_ms}ms</div>
        </div>
        ${language.rtl ? '<span class="badge badge-info" style="margin-left:auto;">RTL</span>' : ''}
      </div>

      <!-- Overall verdict banner -->
      <div class="verdict-banner ${overall.color}">
        <div class="verdict-icon">${verdictIconMap[overall.color]}</div>
        <div class="verdict-text-wrap">
          <div class="verdict-title" style="color:${verdictColorMap[overall.color]}">${escHtml(overall.verdict)}</div>
          <div class="verdict-subtitle">Combined risk score</div>
        </div>
        <div class="verdict-score" style="color:${verdictColorMap[overall.color]}">${riskPct}%</div>
      </div>

      <!-- Metrics row -->
      <div class="metrics-row">
        <div class="metric-card">
          <div class="confidence-ring-wrap" style="margin-bottom:8px;">
            <svg class="confidence-ring" width="80" height="80" viewBox="0 0 100 100">
              <circle class="confidence-ring-bg" cx="50" cy="50" r="42" stroke-width="7"/>
              <circle
                id="misinfo-ring"
                class="confidence-ring-fill"
                cx="50" cy="50" r="42"
                stroke="${ringColor}"
                stroke-width="7"
                stroke-dasharray="314"
                stroke-dashoffset="314"
              />
            </svg>
            <div class="confidence-ring-label">
              <span class="confidence-ring-pct" id="misinfo-pct">0%</span>
              <span class="confidence-ring-text">Risk</span>
            </div>
          </div>
          <div class="metric-label">Misinformation</div>
          <div class="metric-badge">
            <span class="badge ${misinfoLabelToBadge(misinformation.label)}">${escHtml(misinformation.label)}</span>
          </div>
        </div>

        <div class="metric-card">
          <div style="width:100%; margin-bottom:8px;">
            <div class="score-bar-wrap">
              <div class="score-bar-header">
                <span class="score-bar-label">Clickbait Score</span>
                <span class="score-bar-value" id="clickbait-pct-text" style="color:${clickbaitColor(clickbait.score)}">0%</span>
              </div>
              <div class="score-bar-track">
                <div id="clickbait-bar" class="score-bar-fill" style="background: linear-gradient(90deg, var(--accent-purple), ${clickbaitColor(clickbait.score)});"></div>
              </div>
            </div>
          </div>
          <div class="metric-label" style="margin-top:8px;">Clickbait</div>
          <div class="metric-badge" style="margin-top:4px;">
            <span class="badge ${clickbait.is_clickbait ? 'badge-warning' : 'badge-safe'}">${escHtml(clickbait.label)}</span>
          </div>
        </div>
      </div>

      <!-- Reasoning -->
      ${reasoningHTML ? `
      <div class="reasoning-section">
        <div class="reasoning-title">
          <span>🧠</span> Analysis Reasoning
        </div>
        <div class="reasoning-chips">${reasoningHTML}</div>
      </div>` : ''}

      <!-- Red flag phrases -->
      ${redFlagHTML ? `
      <div class="reasoning-section">
        <div class="reasoning-title"><span>🚩</span> Detected Red Flags</div>
        <div style="display:flex; flex-wrap:wrap; gap:6px;">${redFlagHTML}</div>
      </div>` : ''}

      <!-- Highlighted text -->
      <div class="highlighted-text-section">
        <div class="reasoning-title"><span>📄</span> Text Preview</div>
        <div class="highlighted-text-box">${highlightedText}</div>
      </div>
    </div>
  `;

  // Animate rings after paint
  requestAnimationFrame(() => {
    setTimeout(() => {
      // Confidence ring
      const ring = document.getElementById('misinfo-ring');
      if (ring) ring.style.strokeDashoffset = misinfoOffset;

      // Pct counter
      animateCounter('misinfo-pct', 0, Math.round(misinformation.risk_score * 100), '%', 1000);

      // Clickbait bar
      const bar = document.getElementById('clickbait-bar');
      if (bar) bar.style.width = `${Math.round(clickbait.score * 100)}%`;
      animateCounter('clickbait-pct-text', 0, Math.round(clickbait.score * 100), '%', 1000);

    }, 80);
  });
}

// ─── HIGHLIGHT TEXT ─────────────────────────────────────────────

function buildHighlightedText(text, spans) {
  if (!spans || spans.length === 0) return escHtml(text);
  let result = '';
  let lastIdx = 0;
  for (const span of spans) {
    if (span.start > text.length) break;
    result += escHtml(text.slice(lastIdx, span.start));
    const cls = span.type === 'clickbait' ? 'highlight-clickbait' : 'highlight-misinfo';
    const title = span.type === 'clickbait' ? 'Clickbait pattern' : 'Misinformation red flag';
    result += `<mark class="${cls}" title="${title}">${escHtml(span.text)}</mark>`;
    lastIdx = span.end;
  }
  result += escHtml(text.slice(lastIdx));
  return result;
}

// ─── HELPERS ───────────────────────────────────────────────────

function misinfoLabelToBadge(label) {
  if (label === 'Reliable') return 'badge-safe';
  if (label === 'Suspicious') return 'badge-warning';
  return 'badge-danger';
}

function clickbaitColor(score) {
  if (score >= 0.65) return '#ef4444';
  if (score >= 0.35) return '#f59e0b';
  return '#10b981';
}

function escHtml(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

function animateCounter(id, from, to, suffix, duration) {
  const el = document.getElementById(id);
  if (!el) return;
  const start = performance.now();
  const update = (now) => {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(from + (to - from) * eased) + suffix;
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

// ─── HISTORY STORAGE ────────────────────────────────────────────

function saveToHistory(result) {
  try {
    const history = JSON.parse(localStorage.getItem('misinfo_history') || '[]');
    history.unshift({
      id: Date.now(),
      text_preview: result.text_preview,
      language: result.language,
      misinfo_label: result.misinformation.label,
      misinfo_score: result.misinformation.risk_score,
      clickbait_label: result.clickbait.label,
      clickbait_score: result.clickbait.score,
      overall_verdict: result.overall.verdict,
      timestamp: result.timestamp,
    });
    localStorage.setItem('misinfo_history', JSON.stringify(history.slice(0, 200)));
  } catch { /* storage quota exceeded or disabled */ }
}

// ─── TOAST NOTIFICATIONS ────────────────────────────────────────

function showToast(message, type = 'info') {
  const colorMap = { info: '#6d51ff', warn: '#f59e0b', error: '#ef4444', success: '#10b981' };
  const iconMap  = { info: 'ℹ️', warn: '⚠️', error: '❌', success: '✅' };
  const container = document.getElementById('toast-container') || createToastContainer();

  const toast = document.createElement('div');
  toast.style.cssText = `
    display: flex; align-items: center; gap: 10px;
    padding: 12px 18px;
    background: rgba(13,17,32,0.95);
    border: 1px solid ${colorMap[type]}55;
    border-left: 3px solid ${colorMap[type]};
    border-radius: 10px;
    font-size: 0.85rem;
    color: #f1f5f9;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    animation: fadeInDown 0.3s ease;
    max-width: 360px;
    backdrop-filter: blur(12px);
  `;
  toast.innerHTML = `<span>${iconMap[type]}</span><span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(-10px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

function createToastContainer() {
  const el = document.createElement('div');
  el.id = 'toast-container';
  el.style.cssText = `
    position: fixed; top: 88px; right: 20px;
    z-index: 9999; display: flex; flex-direction: column; gap: 10px;
  `;
  document.body.appendChild(el);
  return el;
}

// Expose globally for inline onclick use
window.handleAnalyze = handleAnalyze;
window.showToast = showToast;
