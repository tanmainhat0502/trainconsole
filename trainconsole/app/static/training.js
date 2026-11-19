let PROJECTS = [];
let MODELS = [];
let pollTimer = null;
let LAST_COMPARE = null;

async function loadProjects() {
  PROJECTS = await api('/api/projects');
  const select = document.getElementById('projectSelect');
  const preferred = PROJECTS.find((p) => p.name === 'Train Console') || PROJECTS[0];
  select.innerHTML = PROJECTS.map((p) =>
    `<option value="${p.id}" ${preferred && p.id === preferred.id ? 'selected' : ''}>${p.name}</option>`).join('');
}

function parseTasks(id) {
  return document.getElementById(id).value
    .split(',').map((value) => Number(value.trim())).filter(Boolean);
}

async function loadJobSummary() {
  const projectId = Number(document.getElementById('projectSelect').value);
  if (!projectId) return;
  const summary = await api(`/api/jobs/summary?project_id=${projectId}`);
  document.getElementById('completedJobs').textContent = summary.jobs_by_state.completed || 0;
  document.getElementById('newJobs').textContent = summary.jobs_by_state.new || 0;
  document.getElementById('completedBadge').textContent = `${summary.jobs_by_state.completed || 0} job completed`;
  document.getElementById('newBadge').textContent = `${summary.jobs_by_state.new || 0} job new`;
}

async function loadModels() {
  const projectId = Number(document.getElementById('projectSelect').value) || null;
  MODELS = await api(`/api/models${projectId ? `?project_id=${projectId}` : ''}`);
  const yolo = MODELS.find((model) => (model.family || 'yolo26') === 'yolo26');
  const detr = MODELS.find((model) => model.family === 'rfdetr');
  document.getElementById('productionModel').textContent = [
    yolo ? `YOLO: ${yolo.name}` : null,
    detr ? `DETR: ${detr.name}` : null,
  ].filter(Boolean).join(' · ') || '—';
}

function durationText(run) {
  const start = run.started_at || (run.state === 'queued' ? null : run.created_at);
  if (!start) return '—';
  const active = ['queued', 'running'].includes(run.state) && !run.completed_at;
  if (!run.completed_at && !active) return '—';
  const end = run.completed_at ? new Date(run.completed_at) : new Date();
  const seconds = Math.max(0, Math.round((end - new Date(start)) / 1000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  let text;
  if (hours) text = `${hours}h ${minutes}m`;
  else if (minutes) text = `${minutes}m ${String(secs).padStart(2, '0')}s`;
  else text = `${secs}s`;
  return active ? `${text} <span class="sub">running</span>` : text;
}

function renderTrainRuns(runs) {
  const body = document.getElementById('runRows');
  const latest = runs.slice(0, 5);
  if (!latest.length) {
    body.innerHTML = '<tr><td colspan="7" class="empty">No runs yet.</td></tr>';
    return;
  }
  body.innerHTML = latest.map((run) => {
    const canCancel = ['queued', 'running', 'cancelling'].includes(run.state);
    const action = canCancel
      ? `<button class="btn sm danger" data-cancel-run="${run.id}" ${run.state === 'cancelling' ? 'disabled' : ''}>${run.state === 'cancelling' ? 'Cancelling…' : 'Cancel'}</button>`
      : '';
    return `
    <tr>
      <td>#${run.id} ${run.name}</td>
      <td>${badge(run.state, statusKind(run.state))}</td>
      <td>${run.progress}%</td>
      <td class="muted">${run.stage}${run.error ? `<div class="mono" style="color:var(--danger);max-width:420px;white-space:pre-wrap">${run.error.slice(0, 500)}</div>` : ''}</td>
      <td>${run.metrics?.seen_map50 != null ? `${(Number(run.metrics.seen_map50) * 100).toFixed(2)}%` : run.metrics?.map50 != null ? `${(Number(run.metrics.map50) * 100).toFixed(2)}%` : '—'}
        ${run.metrics?.classes?.val_images != null ? `<div class="sub">${run.metrics.classes.val_images} seen images</div>` : ''}</td>
      <td class="muted">${durationText(run)}</td>
      <td>${action}</td>
    </tr>`;
  }).join('');
}

function renderInferRuns(runs) {
  const body = document.getElementById('inferRows');
  const latest = runs.slice(0, 5);
  if (!latest.length) {
    body.innerHTML = '<tr><td colspan="6" class="empty">No runs yet.</td></tr>';
    return;
  }
  body.innerHTML = latest.map((run) => `
    <tr>
      <td>#${run.id} ${run.name}</td>
      <td>${badge(run.state, statusKind(run.state))}</td>
      <td>${run.progress}%</td>
      <td>${run.counts?.jobs ?? '—'}${run.counts?.skipped ? ` <span class="muted">(${run.counts.skipped} skipped)</span>` : ''}</td>
      <td>${run.counts?.predicted ?? '—'}</td>
      <td class="muted">${durationText(run)}${run.error ? `<div class="mono" style="color:var(--danger)">${run.error.slice(0, 300)}</div>` : ''}</td>
    </tr>`).join('');
}

async function loadRuns() {
  const [trainRuns, inferRuns] = await Promise.all([
    api('/api/train-runs'),
    api('/api/infer-runs'),
  ]);
  renderTrainRuns(trainRuns);
  renderInferRuns(inferRuns);
  const active = trainRuns.some((run) => ['queued', 'running', 'cancelling'].includes(run.state))
    || inferRuns.some((run) => ['queued', 'running'].includes(run.state));
  if (active && !pollTimer) {
    pollTimer = setInterval(async () => {
      try {
        await loadRuns();
        await loadJobSummary();
      } catch (error) { /* transient */ }
    }, 3000);
  }
  if (!active && pollTimer) {
    clearInterval(pollTimer); pollTimer = null;
  }
}

async function startTraining() {
  const projectId = Number(document.getElementById('projectSelect').value);
  if (!projectId) return toast('Select a project first.', 'error');
  const button = document.getElementById('trainBtn');
  button.disabled = true; button.textContent = 'Queueing…';
  try {
    const run = await api('/api/train-runs', {
      method: 'POST',
      body: JSON.stringify({
        project_id: projectId,
        task_ids: parseTasks('trainTaskIds'),
        epochs: Number(document.getElementById('epochs').value),
        batch: Number(document.getElementById('batch').value),
        imgsz: Number(document.getElementById('imgsz').value),
        gpu: document.getElementById('gpu').value,
        family: document.getElementById('trainFamily').value,
      }),
    });
    toast(`Queued train + infer (run #${run.id}).`);
    await loadRuns();
  } catch (error) {
    toast(error.message, 'error');
  } finally {
    button.disabled = false; button.textContent = 'Train & Infer';
  }
}

function parseJobIds(value) {
  return [...new Set(
    value.split(/[\s,]+/)
      .map((part) => Number(part.trim()))
      .filter((jobId) => Number.isInteger(jobId) && jobId > 0),
  )];
}

function syncInferScope() {
  const specific = document.getElementById('inferScope').value === 'ids';
  document.getElementById('inferIdsField').hidden = !specific;
}

function openInferModal() {
  const projectId = Number(document.getElementById('projectSelect').value);
  if (!projectId) return toast('Select a project first.', 'error');
  document.getElementById('inferScope').value = 'all';
  document.getElementById('inferJobIds').value = '';
  const hasYolo = MODELS.some((model) => (model.family || 'yolo26') === 'yolo26');
  const hasDetr = MODELS.some((model) => model.family === 'rfdetr');
  document.getElementById('inferFamily').value = hasYolo || !hasDetr ? 'yolo26' : 'rfdetr';
  syncInferScope();
  document.getElementById('inferModal').hidden = false;
}

async function queueInference() {
  const projectId = Number(document.getElementById('projectSelect').value);
  if (!projectId) return toast('Select a project first.', 'error');
  const mode = document.getElementById('inferScope').value;
  const jobIds = mode === 'ids'
    ? parseJobIds(document.getElementById('inferJobIds').value)
    : [];
  if (mode === 'ids' && !jobIds.length) {
    return toast('Enter at least one job ID.', 'error');
  }
  const button = document.getElementById('inferQueueBtn');
  button.disabled = true; button.textContent = 'Queueing…';
  try {
    const run = await api('/api/infer-runs', {
      method: 'POST',
      body: JSON.stringify({
        project_id: projectId,
        task_ids: mode === 'ids' ? [] : parseTasks('trainTaskIds'),
        job_ids: jobIds,
        family: document.getElementById('inferFamily').value,
        gpu: document.getElementById('gpu').value,
        batch: Number(document.getElementById('batch').value) || 1,
      }),
    });
    toast(mode === 'ids'
      ? `Queued infer run #${run.id} for ${jobIds.length} job(s).`
      : `Queued infer run #${run.id} for all new jobs.`);
    document.getElementById('inferModal').hidden = true;
    await loadRuns();
  } catch (error) {
    toast(error.message, 'error');
  } finally {
    button.disabled = false; button.textContent = 'Queue inference';
  }
}

function closeExportMenu() {
  document.getElementById('exportMenu').hidden = true;
}

async function exportLabels(scope) {
  const projectId = Number(document.getElementById('projectSelect').value);
  if (!projectId) return toast('Select a project first.', 'error');
  const button = document.getElementById('exportBtn');
  button.disabled = true;
  button.textContent = 'Exporting…';
  try {
    const response = await fetch(`/api/labels/export?project_id=${projectId}&scope=${scope}`);
    if (response.status === 401) { window.location = '/login'; return; }
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const payload = await response.json();
        detail = payload.detail || detail;
      } catch (error) { /* keep status text */ }
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `labels_${scope}_${projectId}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    toast(`Exported labels: ${scope === 'completed' ? 'completed jobs' : 'all jobs'}.`);
  } catch (error) {
    toast(error.message, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Export labels ▾';
  }
}

async function cancelRun(runId) {
  if (!window.confirm(`Cancel training run #${runId}?`)) return;
  try {
    await api(`/api/train-runs/${runId}/cancel`, { method: 'POST' });
    toast(`Cancelling training run #${runId}…`);
    await loadRuns();
  } catch (error) {
    toast(error.message, 'error');
  }
}

async function testGpu() {
  const button = document.getElementById('gpuTestBtn');
  const result = document.getElementById('gpuTestResult');
  const gpu = document.getElementById('gpu').value.trim();
  if (!gpu) return toast('Enter GPU first, e.g. 0,1,3', 'error');
  button.disabled = true;
  button.textContent = 'Testing…';
  result.style.display = 'block';
  result.textContent = 'Measuring free VRAM…';
  try {
    const data = await api('/api/gpu/test', {
      method: 'POST',
      body: JSON.stringify({
        gpu,
        imgsz: Number(document.getElementById('imgsz').value) || 640,
        project_id: Number(document.getElementById('projectSelect').value),
      }),
    });
    const yolo = data.yolo || data;
    const rf = data.rfdetr || {};
    const freeStr = data.gpus.map((g) => `${g}:${data.free_gb[g]}GB`).join(' · ');
    const yoloText = yolo.error
      ? `YOLO26x: <span style="color:var(--danger)">${escapeHtml(yolo.error)}</span>`
      : `YOLO26x: max <b>${yolo.suggested}</b>/GPU (total ${yolo.suggested * data.gpus.length})`;
    const rfText = rf.batch
      ? `RF-DETR-2XL: <b>${rf.batch}</b>/GPU (auto${rf.grad_accum ? `, accum ${rf.grad_accum}` : ''}${rf.source ? ` · ${escapeHtml(rf.source)}` : ''})`
      : 'RF-DETR-2XL: auto (no completed run yet)';
    result.innerHTML = `${yoloText} · ${rfText} · free VRAM ${freeStr} · imgsz ${data.imgsz}`;
  } catch (error) {
    toast(error.message, 'error');
    result.textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = 'Test';
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[char]));
}

function syncFamilyControls() {
  const rfdetr = document.getElementById('trainFamily').value === 'rfdetr';
  document.getElementById('gpuTestBtn').disabled = false;
  document.getElementById('familyHint').textContent = rfdetr
    ? 'RF-DETR-2XL: training uses batch_size="auto"; this batch is used for inference/compare.'
    : 'YOLO26x: batch is per GPU; Test suggests the maximum batch.';
}

function renderCompare(data) {
  const grid = document.getElementById('compareGrid');
  const status = document.getElementById('compareStatus');
  document.getElementById('compareMeta').textContent =
    `${data.model_yolo || 'YOLO'} vs ${data.model_detr || 'RF-DETR'}`;
  const when = data.created_at ? ` · ${new Date(data.created_at).toLocaleString()}` : '';
  status.textContent =
    `${data.images.length} images · GT green · YOLO blue · DETR red${when}`;
  grid.innerHTML = data.images.map((item) => `
    <figure>
      <img src="${item.image}" loading="lazy" alt="">
      <figcaption>${escapeHtml(item.name)}<br>GT ${item.gt} · YOLO ${item.yolo} · DETR ${item.detr}</figcaption>
    </figure>`).join('');
}

async function openCompareModal() {
  const projectId = Number(document.getElementById('projectSelect').value);
  if (!projectId) return toast('Select a project first.', 'error');
  const viewButton = document.getElementById('compareViewBtn');
  const modal = document.getElementById('compareModal');
  LAST_COMPARE = null;
  viewButton.disabled = true;
  document.getElementById('compareGrid').innerHTML = '';
  document.getElementById('compareMeta').textContent = '';
  document.getElementById('compareStatus').textContent =
    'Choose “View last compare” or “Run new 50 images”.';
  modal.hidden = false;
  try {
    const data = await api(`/api/compare/latest?project_id=${projectId}`);
    LAST_COMPARE = data;
    viewButton.disabled = false;
    document.getElementById('compareMeta').textContent =
      `Last: ${data.model_yolo || 'YOLO'} vs ${data.model_detr || 'RF-DETR'}`;
  } catch (error) {
    document.getElementById('compareMeta').textContent = 'No previous compare yet.';
  }
}

function viewLastCompare() {
  if (!LAST_COMPARE) return toast('No previous compare yet.', 'error');
  renderCompare(LAST_COMPARE);
}

async function runNewCompare() {
  const projectId = Number(document.getElementById('projectSelect').value);
  if (!projectId) return toast('Select a project first.', 'error');
  const button = document.getElementById('compareRunBtn');
  const viewButton = document.getElementById('compareViewBtn');
  button.disabled = true; button.textContent = 'Running 50 images…';
  document.getElementById('compareGrid').innerHTML = '';
  document.getElementById('compareMeta').textContent = '';
  document.getElementById('compareStatus').textContent =
    'Running YOLO26x and RF-DETR-2XL on 50 random images…';
  try {
    const data = await api('/api/compare', {
      method: 'POST',
      body: JSON.stringify({
        project_id: projectId,
        count: 50,
        gpu: document.getElementById('gpu').value,
        batch: Number(document.getElementById('batch').value) || 1,
      }),
    });
    LAST_COMPARE = data;
    viewButton.disabled = false;
    renderCompare(data);
  } catch (error) {
    document.getElementById('compareStatus').textContent = error.message;
  } finally {
    button.disabled = false; button.textContent = 'Run new 50 images';
  }
}

async function refreshAll() {
  await loadJobSummary();
  await loadModels();
  await loadRuns();
}

document.addEventListener('DOMContentLoaded', async () => {
  if (window.TC.page !== 'training') return;
  try {
    await loadProjects();
    document.getElementById('projectSelect').onchange = () => refreshAll().catch((error) => toast(error.message, 'error'));
    document.getElementById('refreshBtn').onclick = () => refreshAll().catch((error) => toast(error.message, 'error'));
    document.getElementById('trainBtn').onclick = startTraining;
    document.getElementById('trainFamily').onchange = syncFamilyControls;
    syncFamilyControls();
    document.getElementById('compareBtn').onclick = openCompareModal;
    document.getElementById('compareViewBtn').onclick = viewLastCompare;
    document.getElementById('compareRunBtn').onclick = runNewCompare;
    document.getElementById('inferBtn').onclick = openInferModal;
    document.getElementById('inferScope').onchange = syncInferScope;
    document.getElementById('inferQueueBtn').onclick = queueInference;
    document.getElementById('gpuTestBtn').onclick = testGpu;
    document.querySelectorAll('[data-close]').forEach((button) => {
      button.onclick = () => { document.getElementById(button.dataset.close).hidden = true; };
    });
    document.querySelectorAll('.modal').forEach((modal) => {
      modal.addEventListener('click', (event) => {
        if (event.target === modal) modal.hidden = true;
      });
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        document.querySelectorAll('.modal').forEach((modal) => { modal.hidden = true; });
      }
    });
    document.getElementById('runRows').addEventListener('click', (event) => {
      const button = event.target.closest('[data-cancel-run]');
      if (button && !button.disabled) cancelRun(button.dataset.cancelRun);
    });
    document.getElementById('exportBtn').onclick = (event) => {
      event.stopPropagation();
      const menu = document.getElementById('exportMenu');
      menu.hidden = !menu.hidden;
    };
    document.getElementById('exportMenu').addEventListener('click', (event) => {
      const item = event.target.closest('[data-scope]');
      if (!item) return;
      closeExportMenu();
      exportLabels(item.dataset.scope);
    });
    document.addEventListener('click', (event) => {
      if (!event.target.closest('.dropdown')) closeExportMenu();
    });
    await Promise.all([loadModels(), loadRuns(), loadJobSummary()]);
  } catch (error) { toast(error.message, 'error'); }
});
