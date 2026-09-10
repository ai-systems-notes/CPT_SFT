document.addEventListener('DOMContentLoaded', () => {
  let appData = null;
  let isRevealed = true;

  const MODEL_INFO = {
    base: { name: 'Qwen3 0.6B Base', family: 'base' },
    transformers_cpt: { name: 'Transformers CPT', family: 'transformers_cpt' },
    unsloth_cpt: { name: 'Unsloth CPT', family: 'unsloth_cpt' },
    base_sft: { name: 'Base + SFT', family: 'base_sft' },
    transformers_cpt_sft: { name: 'Transformers CPT + SFT', family: 'transformers_cpt_sft' },
    unsloth_cpt_sft: { name: 'Unsloth CPT + SFT', family: 'unsloth_cpt_sft' }
  };

  const byId = id => document.getElementById(id);
  const metaGrid = byId('meta-grid');
  const runtimeGrid = byId('runtime-grid');
  const questionsList = byId('questions-list');
  const searchInput = byId('search-input');
  const searchClear = byId('search-clear');
  const categoryFilter = byId('category-filter');
  const jumpSelect = byId('jump-select');
  const visibleCount = byId('visible-count');
  const totalCount = byId('total-count');
  const blindToggle = byId('blind-toggle');
  const modeStatusText = byId('mode-status-text');
  const btnViewNote = byId('btn-view-note');
  const noteModal = byId('note-modal');
  const modalClose = byId('modal-close');
  const noteTextContent = byId('note-text-content');

  const escapeHtml = value => String(value ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
  const modelName = key => MODEL_INFO[key]?.name || key;
  const modelClass = key => `model-${String(key).replace(/[^a-zA-Z0-9_-]/g, '-')}`;
  const modelKeys = () => {
    const configured = appData?.metadata?.models;
    const fromMetadata = Array.isArray(configured)
      ? configured.map(item => item.label).filter(Boolean)
      : Object.keys(configured || {});
    return fromMetadata.length ? fromMetadata : Object.keys(appData?.items?.[0]?.answers || {});
  };

  async function loadData() {
    try {
      const response = await fetch('/api/data');
      if (!response.ok) throw new Error('Failed to load dataset');
      appData = await response.json();
      const hasBlindData = appData.items.some(
        item => Object.keys(item.blind_candidates || {}).length > 0
      );
      blindToggle.disabled = !hasBlindData;
      if (!hasBlindData) {
        blindToggle.checked = true;
        isRevealed = true;
        modeStatusText.textContent = 'Revealed (blind data unavailable)';
      }
      renderMetadata();
      populateFilters();
      renderQuestions();
    } catch (error) {
      questionsList.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
    }
  }

  function renderMetadata() {
    const meta = appData.metadata || {};
    const generation = meta.generation || {};
    totalCount.textContent = meta.item_count ?? appData.items.length;
    metaGrid.innerHTML = `
      <div class="meta-item"><span class="meta-label">Dataset</span><span class="meta-val">${escapeHtml((meta.dataset || 'qa_input.jsonl').split('/').pop())}</span></div>
      <div class="meta-item"><span class="meta-label">Status</span><span class="meta-val">${escapeHtml(meta.status || 'unknown')}</span></div>
      <div class="meta-item"><span class="meta-label">Models</span><span class="meta-val">${modelKeys().length}</span></div>
      <div class="meta-item"><span class="meta-label">Decoding</span><span class="meta-val">${escapeHtml(generation.decoding || 'greedy')} / ${escapeHtml(generation.dtype || '--')}</span></div>`;

    const runtime = meta.runtime || {};
    runtimeGrid.innerHTML = modelKeys().map(key => {
      const value = runtime[key] || {};
      return `<div class="runtime-box ${modelClass(key)}">
        <div class="model-title">${escapeHtml(modelName(key))}</div>
        <div class="stat-highlight">${Number.isFinite(value.generation_seconds) ? value.generation_seconds.toFixed(2) : '--'}s</div>
        <div class="stat-row">Load: ${Number.isFinite(value.model_load_seconds) ? value.model_load_seconds.toFixed(2) : '--'}s</div>
        <div class="stat-row">VRAM: ${Number.isFinite(value.torch_peak_allocated_mib) ? value.torch_peak_allocated_mib.toFixed(1) : '--'} MiB</div>
      </div>`;
    }).join('');
  }

  function populateFilters() {
    const categories = [...new Set(appData.items.map(item => item.category).filter(Boolean))].sort();
    categoryFilter.innerHTML = '<option value="all">All Categories</option>' + categories
      .map(category => `<option value="${escapeHtml(category)}">${escapeHtml(category)}</option>`).join('');
    jumpSelect.innerHTML = '<option value="">Select Question...</option>' + appData.items
      .map(item => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.id)}: ${escapeHtml(item.question.slice(0, 45))}</option>`).join('');
  }

  function filteredItems() {
    const search = searchInput.value.toLowerCase().trim();
    const category = categoryFilter.value;
    return appData.items.filter(item => {
      if (category !== 'all' && item.category !== category) return false;
      if (!search) return true;
      const values = [item.question, item.answer, item.topic, ...(item.keywords || []), ...Object.values(item.answers || {})];
      return values.some(value => String(value || '').toLowerCase().includes(search));
    });
  }

  function answerColumns(item) {
    if (isRevealed) {
      return modelKeys().map(key => ({
        key, name: modelName(key), text: item.answers?.[key] || '', cssClass: modelClass(key)
      }));
    }
    const candidates = item.blind_candidates || {};
    const mapping = item.candidate_to_model || {};
    return Object.keys(candidates).sort().map(letter => {
      const key = mapping[letter] || '';
      return {
        key, letter, name: `Candidate ${letter}`, text: candidates[letter] || '', cssClass: modelClass(key)
      };
    });
  }

  function judgeHtml(item) {
    const judge = item.judge;
    if (!judge) return '';
    const winner = isRevealed ? judge.judge_winner_model : judge.judge_winner_candidate;
    const scores = judge.judge_scores_by_model || {};
    const scoreText = modelKeys().map(key => `${modelName(key)} ${scores[key] ?? '--'}`).join(' / ');
    return `<div class="judge-box">
      <div class="judge-title">Gemini Judge</div>
      <div><strong>Winner:</strong> ${escapeHtml(winner || 'Tie')}</div>
      <div><strong>Scores:</strong> ${escapeHtml(scoreText)}</div>
      <div class="judge-reason">${escapeHtml(judge.judge_reason || '')}</div>
    </div>`;
  }

  function questionCard(item) {
    const keywords = item.keywords || [];
    const cards = answerColumns(item).map(column => {
      const hits = new Set(keywords.filter(keyword => column.text.toLowerCase().includes(keyword.toLowerCase())));
      const badge = column.letter ? `<span class="blind-candidate-badge">Blind ${escapeHtml(column.letter)}</span>` : '';
      return `<div class="answer-card ${column.cssClass}">
        <div class="model-header"><span class="model-name">${escapeHtml(column.name)} ${badge}</span><span>${column.text.length} chars</span></div>
        <div class="answer-content">${escapeHtml(column.text || '(Empty Response)')}</div>
        <div class="keywords-list">${keywords.map(keyword => `<span class="kw-badge ${hits.has(keyword) ? 'hit' : ''}">${hits.has(keyword) ? '✓ ' : ''}${escapeHtml(keyword)}</span>`).join('')}</div>
      </div>`;
    }).join('');
    const source = item.source_url
      ? `<a href="${escapeHtml(item.source_url)}" target="_blank" rel="noopener noreferrer" class="q-source-link">Source Doc</a>` : '';
    return `<article class="q-card" id="card-${escapeHtml(item.id)}">
      <div class="q-card-header"><div class="q-title-area"><div><span class="q-id-badge">${escapeHtml(item.id)}</span>
      ${item.category ? `<span class="q-category-tag">${escapeHtml(item.category)}</span>` : ''}
      ${item.topic ? `<span class="q-topic-tag">${escapeHtml(item.topic)}</span>` : ''}</div>
      <h2 class="q-question-text">${escapeHtml(item.question)}</h2></div>${source}</div>
      <div class="reference-box"><div class="reference-title">Reference Answer</div><div class="reference-text">${escapeHtml(item.answer)}</div></div>
      ${judgeHtml(item)}<div class="answers-grid">${cards}</div></article>`;
  }

  function renderQuestions() {
    if (!appData) return;
    const items = filteredItems();
    visibleCount.textContent = items.length;
    questionsList.innerHTML = items.length ? items.map(questionCard).join('') : '<p>No matching questions.</p>';
  }

  blindToggle.addEventListener('change', event => {
    isRevealed = event.target.checked;
    modeStatusText.textContent = isRevealed ? 'De-blind (Revealed)' : 'Blind (Gemini candidates)';
    renderQuestions();
  });
  searchInput.addEventListener('input', () => { searchClear.style.display = searchInput.value ? 'block' : 'none'; renderQuestions(); });
  searchClear.addEventListener('click', () => { searchInput.value = ''; searchClear.style.display = 'none'; renderQuestions(); });
  categoryFilter.addEventListener('change', renderQuestions);
  jumpSelect.addEventListener('change', event => {
    const element = document.getElementById(`card-${event.target.value}`);
    if (element) element.scrollIntoView({ behavior: 'smooth', block: 'center' });
  });
  btnViewNote.addEventListener('click', async () => {
    noteModal.style.display = 'flex';
    try { noteTextContent.textContent = await (await fetch('/api/report')).text(); }
    catch (error) { noteTextContent.textContent = `Failed to load report: ${error.message}`; }
  });
  modalClose.addEventListener('click', () => { noteModal.style.display = 'none'; });
  noteModal.addEventListener('click', event => { if (event.target === noteModal) noteModal.style.display = 'none'; });

  loadData();
});
