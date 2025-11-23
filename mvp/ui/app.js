// app.js - UI behaviour for MVP dashboard

const API_ROOT = ""; // same origin
const state = {
  taxonomy: null,
  lastPrediction: null,
};
const els = {};

// ---- Theme toggler (robust) ----
(function themeInit(){
  const root = document.documentElement;
  const stored = localStorage.getItem('txcat_theme'); // 'dark' or 'light'
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

  function applyTheme(name){
    if(name === 'dark'){
      root.setAttribute('data-theme', 'dark');
      root.classList.add('theme-dark');
      root.classList.remove('theme-light');
    } else {
      root.setAttribute('data-theme', 'light');
      root.classList.add('theme-light');
      root.classList.remove('theme-dark');
    }
  }

  const initial = stored || (prefersDark ? 'dark' : 'light');
  applyTheme(initial);

  document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('theme-toggle');
    if(!btn) return;
    function toggle(){
      const cur = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
      const nxt = cur === 'dark' ? 'light' : 'dark';
      applyTheme(nxt);
      localStorage.setItem('txcat_theme', nxt);
      btn.setAttribute('aria-pressed', nxt === 'dark');
    }
    btn.addEventListener('click', toggle);
  });
})();

// ---- Bootstrapping ----
document.addEventListener('DOMContentLoaded', () => {
  cacheDom();
  wireEvents();
  hydrateTaxonomy();
});

function cacheDom(){
  els.txInput = document.getElementById('input-transaction');
  els.btnPredict = document.getElementById('btn-predict');
  els.btnRebuild = document.getElementById('btn-rebuild-index');
  els.resultCard = document.getElementById('result-card');
  els.predictedLabel = document.getElementById('predicted-label');
  els.predictedConfidence = document.getElementById('predicted-confidence');
  els.neighborsList = document.getElementById('neighbors-list');
  els.rationale = document.getElementById('rationale');
  els.rawJson = document.getElementById('raw-json');
  els.csvFile = document.getElementById('file-csv');
  els.btnPredictCsv = document.getElementById('btn-predict-csv');
  els.btnClearCsv = document.getElementById('btn-clear-csv');
  els.csvResults = document.getElementById('csv-results');
  els.correctionTransaction = document.getElementById('correction-transaction');
  els.correctionInput = document.getElementById('correction-input');
  els.btnSubmitCorrection = document.getElementById('btn-submit-correction');
  els.btnSaveAdd = document.getElementById('btn-save-add');
  els.correctionStatus = document.getElementById('correction-status');
  els.taxonomyFile = document.getElementById('taxonomy-file');
  els.btnUploadTaxonomy = document.getElementById('btn-upload-taxonomy');
  els.statusText = document.getElementById('status-text');
  els.indexCount = document.getElementById('index-count');
  els.modelBadge = document.getElementById('model-badge');
}

function wireEvents(){
  if(els.btnPredict && els.txInput){
    els.btnPredict.addEventListener('click', () => handlePredict());
    els.txInput.addEventListener('keydown', (ev) => {
      if((ev.ctrlKey || ev.metaKey) && ev.key === 'Enter'){
        handlePredict();
      }
    });
  }

  if(els.btnRebuild){
    els.btnRebuild.addEventListener('click', async () => {
      setBusy(els.btnRebuild, true);
      setStatus('Rebuilding index…');
      try{
        const resp = await fetch(`${API_ROOT}/rebuild_index`, { method: 'POST' });
        const data = await expectJson(resp);
        setStatus(`Index rebuilt (${data.count ?? '0'} docs)`, 'success');
        await hydrateTaxonomy();
      } catch (err) {
        console.error(err);
        setStatus(err.message || 'Rebuild failed', 'error');
      } finally {
        setBusy(els.btnRebuild, false);
      }
    });
  }

  if(els.btnPredictCsv){
    els.btnPredictCsv.addEventListener('click', () => handleCsvPredict());
  }
  if(els.btnClearCsv){
    els.btnClearCsv.addEventListener('click', () => clearCsvResults());
  }

  if(els.btnSubmitCorrection){
    els.btnSubmitCorrection.addEventListener('click', () => handleCorrection('correct'));
  }
  if(els.btnSaveAdd){
    els.btnSaveAdd.addEventListener('click', () => handleCorrection('add_to_index'));
  }

  if(els.btnUploadTaxonomy){
    els.btnUploadTaxonomy.addEventListener('click', () => handleTaxonomyUpload());
  }
}

async function hydrateTaxonomy(){
  setStatus('Loading taxonomy…');
  try{
    const resp = await fetch(`${API_ROOT}/taxonomy`);
    const data = await expectJson(resp);
    state.taxonomy = data;
    updateMetaBadges(data);
    populateCategoryInputs(data.categories || []);
    setStatus('Ready', 'success');
  } catch (err) {
    console.error(err);
    setStatus(err.message || 'Failed to load taxonomy', 'error');
  }
}

function updateMetaBadges(meta){
  if(els.modelBadge){
    els.modelBadge.textContent = meta.model ? `Model: ${meta.model}` : 'Model: —';
  }
  if(els.indexCount){
    const count = meta.index_count ?? meta.count ?? '—';
    els.indexCount.textContent = `Index: ${count}`;
  }
}

function populateCategoryInputs(categories){
  if(!els.correctionInput) return;
  
  // Clear existing options except the first placeholder
  const select = els.correctionInput;
  while(select.options.length > 1){
    select.remove(1);
  }
  
  // Add categories as options
  (categories || []).forEach((cat) => {
    const option = document.createElement('option');
    option.value = cat.id;
    option.textContent = `${cat.name || cat.id} (${cat.id})`;
    select.appendChild(option);
  });
}

async function handlePredict(){
  if(!els.txInput) return;
  const tx = (els.txInput.value || '').trim();
  if(!tx){
    setStatus('Enter a transaction to predict', 'error');
    return;
  }
  setBusy(els.btnPredict, true);
  setStatus('Predicting…');
  try{
    const form = new FormData();
    form.append('transaction', tx);
    const resp = await fetch(`${API_ROOT}/predict`, { method: 'POST', body: form });
    const data = await expectJson(resp);
    state.lastPrediction = data;
    renderPrediction(data);
    prefillCorrection(data, tx);
    setStatus('Prediction ready', 'success');
  } catch (err) {
    console.error(err);
    setStatus(err.message || 'Prediction failed', 'error');
  } finally {
    setBusy(els.btnPredict, false);
  }
}

function renderPrediction(pred){
  if(!els.resultCard) return;
  els.resultCard.classList.remove('hidden');
  const labelText = pred.predicted_category_name || pred.predicted_category_id || pred.category || 'UNKNOWN';
  if(els.predictedLabel) els.predictedLabel.textContent = labelText;

  const confidenceText = formatConfidence(pred.confidence);
  if(els.predictedConfidence){
    els.predictedConfidence.textContent = `Confidence ${confidenceText}`;
    els.predictedConfidence.classList.toggle('low', !!pred.is_low_confidence);
  }

  if(els.rationale){
    const matches = (pred.keyword_matches || []).map(k => k.toUpperCase());
    const matchText = matches.length ? `Keywords: ${matches.join(', ')}. ` : '';
    els.rationale.textContent = `${matchText}${pred.rationale || ''}`.trim() || 'No rationale provided.';
  }

  renderNeighbors(pred.explanations || []);
  
  renderFeatureImportance(pred.feature_importance || {});

  if(els.rawJson){
    els.rawJson.textContent = JSON.stringify(pred, null, 2);
  }
}

function renderFeatureImportance(featureImportance){
  const container = document.getElementById('feature-importance');
  if(!container) return;
  
  const keywordContainer = document.getElementById('keyword-importance');
  const neighborContainer = document.getElementById('neighbor-importance');
  
  if(!featureImportance || (!featureImportance.keywords && !featureImportance.neighbors)){
    container.classList.add('hidden');
    return;
  }
  
  container.classList.remove('hidden');
  
  // Render keyword importance
  if(keywordContainer && featureImportance.keywords){
    keywordContainer.innerHTML = '';
    const keywords = Object.entries(featureImportance.keywords);
    if(keywords.length > 0){
      const title = document.createElement('div');
      title.className = 'muted small';
      title.style.marginBottom = '0.5rem';
      title.textContent = 'Keyword Matches:';
      keywordContainer.appendChild(title);
      
      keywords.forEach(([keyword, data]) => {
        const item = document.createElement('div');
        item.className = 'feature-item';
        item.style.display = 'flex';
        item.style.justifyContent = 'space-between';
        item.style.alignItems = 'center';
        item.style.padding = '0.3rem 0';
        item.style.borderBottom = '1px solid rgba(255,255,255,0.1)';
        
        const left = document.createElement('div');
        left.textContent = `${keyword} (${data.count}x)`;
        left.style.fontWeight = '500';
        
        const right = document.createElement('div');
        right.className = 'muted small';
        right.textContent = `${Math.round(data.importance * 100)}% importance`;
        
        const bar = document.createElement('div');
        bar.style.width = `${data.importance * 100}%`;
        bar.style.height = '4px';
        bar.style.backgroundColor = 'var(--primary, #4CAF50)';
        bar.style.borderRadius = '2px';
        bar.style.marginTop = '0.2rem';
        
        const barContainer = document.createElement('div');
        barContainer.style.width = '100%';
        barContainer.style.marginTop = '0.2rem';
        barContainer.appendChild(bar);
        
        const wrapper = document.createElement('div');
        wrapper.style.width = '100%';
        wrapper.appendChild(left);
        wrapper.appendChild(right);
        wrapper.appendChild(barContainer);
        
        item.appendChild(wrapper);
        keywordContainer.appendChild(item);
      });
    }
  }
  
  // Render neighbor importance
  if(neighborContainer && featureImportance.neighbors){
    neighborContainer.innerHTML = '';
    const neighbors = Object.entries(featureImportance.neighbors);
    if(neighbors.length > 0){
      const title = document.createElement('div');
      title.className = 'muted small';
      title.style.marginTop = '1rem';
      title.style.marginBottom = '0.5rem';
      title.textContent = 'Neighbor Contributions:';
      neighborContainer.appendChild(title);
      
      neighbors.forEach(([key, data]) => {
        const item = document.createElement('div');
        item.className = 'feature-item';
        item.style.display = 'flex';
        item.style.justifyContent = 'space-between';
        item.style.alignItems = 'center';
        item.style.padding = '0.3rem 0';
        item.style.borderBottom = '1px solid rgba(255,255,255,0.1)';
        
        const left = document.createElement('div');
        left.textContent = data.description || key;
        left.style.fontSize = '0.85rem';
        left.style.maxWidth = '60%';
        left.style.overflow = 'hidden';
        left.style.textOverflow = 'ellipsis';
        left.style.whiteSpace = 'nowrap';
        
        const right = document.createElement('div');
        right.className = 'muted small';
        right.textContent = `${Math.round(data.weight * 100)}% weight`;
        
        const bar = document.createElement('div');
        bar.style.width = `${data.weight * 100}%`;
        bar.style.height = '4px';
        bar.style.backgroundColor = 'var(--accent, #2196F3)';
        bar.style.borderRadius = '2px';
        bar.style.marginTop = '0.2rem';
        
        const barContainer = document.createElement('div');
        barContainer.style.width = '100%';
        barContainer.style.marginTop = '0.2rem';
        barContainer.appendChild(bar);
        
        const wrapper = document.createElement('div');
        wrapper.style.width = '100%';
        wrapper.appendChild(left);
        wrapper.appendChild(right);
        wrapper.appendChild(barContainer);
        
        item.appendChild(wrapper);
        neighborContainer.appendChild(item);
      });
    }
  }
}

function renderNeighbors(explanations){
  if(!els.neighborsList) return;
  els.neighborsList.innerHTML = '';
  if(!explanations.length){
    const empty = document.createElement('div');
    empty.className = 'muted small';
    empty.textContent = 'No similar transactions found.';
    els.neighborsList.appendChild(empty);
    return;
  }
  const frag = document.createDocumentFragment();
  explanations.forEach((exp, idx) => {
    const row = document.createElement('div');
    row.className = 'neighbor';

    const desc = document.createElement('div');
    desc.className = 'desc';
    desc.textContent = exp.description || '(no description)';

    const lbl = document.createElement('div');
    lbl.className = 'lbl';
    lbl.textContent = exp.category || 'UNKNOWN';

    const sim = document.createElement('div');
    sim.className = 'sim';
    const similarity = typeof exp.similarity === 'number' ? `${Math.round(exp.similarity * 100)}%` : '—';
    sim.textContent = similarity;

    row.appendChild(desc);
    row.appendChild(lbl);
    row.appendChild(sim);
    frag.appendChild(row);
  });
  els.neighborsList.appendChild(frag);
}

function prefillCorrection(prediction, fallbackTx){
  if(els.correctionTransaction){
    els.correctionTransaction.value = prediction.description || fallbackTx || '';
  }
  if(els.correctionInput){
    els.correctionInput.value = prediction.predicted_category_id || prediction.category || '';
  }
}

async function handleCsvPredict(){
  if(!els.csvFile || !els.csvFile.files.length){
    setStatus('Choose a CSV file first', 'error');
    return;
  }
  const file = els.csvFile.files[0];
  const form = new FormData();
  form.append('file', file);
  setBusy(els.btnPredictCsv, true);
  setStatus('Running batch prediction…');
  try{
    const resp = await fetch(`${API_ROOT}/predict_batch`, { method: 'POST', body: form });
    const data = await expectJson(resp);
    renderCsvResults(data.predictions || []);
    setStatus(`Batch ready (${data.count ?? 0} rows)`, 'success');
  } catch (err) {
    console.error(err);
    setStatus(err.message || 'Batch prediction failed', 'error');
  } finally {
    setBusy(els.btnPredictCsv, false);
  }
}

function renderCsvResults(rows){
  if(!els.csvResults) return;
  els.csvResults.classList.remove('hidden');
  els.csvResults.innerHTML = '';
  if(!rows.length){
    els.csvResults.textContent = 'No rows predicted.';
    return;
  }
  const frag = document.createDocumentFragment();
  rows.slice(0, 25).forEach((row) => {
    const card = document.createElement('div');
    card.className = 'csv-row';
    const title = document.createElement('div');
    title.className = 'desc';
    title.textContent = row.description || '(no description)';
    const details = document.createElement('div');
    details.className = 'muted small';
    const conf = formatConfidence(row.confidence);
    details.textContent = `Prediction: ${row.predicted_category || 'UNKNOWN'} • ${conf}`;
    card.appendChild(title);
    card.appendChild(details);
    frag.appendChild(card);
  });
  if(rows.length > 25){
    const note = document.createElement('div');
    note.className = 'muted small';
    note.textContent = `Showing 25 of ${rows.length} rows. Download CSV to inspect all rows from the API response.`;
    frag.appendChild(note);
  }
  els.csvResults.appendChild(frag);
}

function clearCsvResults(){
  if(els.csvFile) els.csvFile.value = '';
  if(els.csvResults){
    els.csvResults.classList.add('hidden');
    els.csvResults.innerHTML = '';
  }
}

async function handleCorrection(endpoint){
  if(!els.correctionTransaction){
    setStatus('Correction UI missing', 'error');
    return;
  }
  const tx = (els.correctionTransaction.value || state.lastPrediction?.description || '').trim();
  if(!tx){
    setStatus('Correct the last prediction or paste a transaction first.', 'error');
    return;
  }
  let label = (els.correctionInput && els.correctionInput.value || '').trim();
  if(!label && state.taxonomy?.categories?.length === 1){
    label = state.taxonomy.categories[0].id;
  }
  if(!label){
    setStatus('Choose a category to submit the correction.', 'error');
    return;
  }

  const form = new FormData();
  form.append('transaction', tx);
  form.append('correct_label', label);
  const target = endpoint === 'add_to_index' ? 'add_to_index' : 'correct';
  setBusy(endpoint === 'add_to_index' ? els.btnSaveAdd : els.btnSubmitCorrection, true);
  if(els.correctionStatus){
    els.correctionStatus.textContent = 'Submitting…';
  }
  try{
    const resp = await fetch(`${API_ROOT}/${target}`, { method: 'POST', body: form });
    const data = await expectJson(resp);
    const success = target === 'correct' ? data.saved : data.added;
    if(success){
      const msg = target === 'correct' ? 'Correction saved' : 'Added to index';
      if(els.correctionStatus) els.correctionStatus.textContent = msg;
      setStatus(msg, 'success');
      if(target === 'add_to_index'){
        await hydrateTaxonomy();
      }
    } else {
      throw new Error('Server did not confirm save');
    }
  } catch (err) {
    console.error(err);
    if(els.correctionStatus) els.correctionStatus.textContent = 'Failed';
    setStatus(err.message || 'Correction failed', 'error');
  } finally {
    setBusy(endpoint === 'add_to_index' ? els.btnSaveAdd : els.btnSubmitCorrection, false);
  }
}

async function handleTaxonomyUpload(){
  if(!els.taxonomyFile || !els.taxonomyFile.files.length){
    setStatus('Choose a taxonomy file first', 'error');
    return;
  }
  const fd = new FormData();
  fd.append('file', els.taxonomyFile.files[0]);
  setBusy(els.btnUploadTaxonomy, true);
  setStatus('Uploading taxonomy…');
  try{
    const resp = await fetch(`${API_ROOT}/upload_taxonomy`, { method: 'POST', body: fd });
    const data = await expectJson(resp);
    setStatus(`Taxonomy uploaded (${data.categories ?? '0'} categories). Rebuild to apply.`, 'success');
    await hydrateTaxonomy();
  } catch (err) {
    console.error(err);
    setStatus(err.message || 'Upload failed', 'error');
  } finally {
    setBusy(els.btnUploadTaxonomy, false);
  }
}

// ---- Helpers ----
function setBusy(element, isBusy){
  if(!element) return;
  element.disabled = !!isBusy;
  element.classList.toggle('is-busy', !!isBusy);
}

function formatConfidence(value){
  if(typeof value !== 'number' || Number.isNaN(value)){
    return '—';
  }
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const precision = pct < 1 ? 2 : pct < 100 ? 1 : 0;
  return `${pct.toFixed(precision)}%`;
}

async function expectJson(resp){
  if(!resp.ok){
    const text = await resp.text();
    throw new Error(text || `Request failed with ${resp.status}`);
  }
  return resp.json();
}

function setStatus(message, tone = 'info'){
  if(els.statusText){
    els.statusText.textContent = message;
    els.statusText.dataset.tone = tone;
  }
}
