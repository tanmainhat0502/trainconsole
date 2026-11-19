let PROJECTS = [];
let USERS = [];
let PREVIEW = null;
let LAST_SUMMARY = null;
let SELECTED_USER = null;

const $ = (id) => document.getElementById(id);

const PICKER = { input: null, index: 0 };

function escapeAttr(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;').replace(/"/g, '&quot;')
    .replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

async function loadUsers() {
  USERS = await api('/api/users');
  $('usersCount').textContent = USERS.length;
  $('userListCount').textContent = `${USERS.length} user`;
  $('assignees').innerHTML = USERS.map((user) =>
    `<option value="${user.id}">${user.username}${user.is_superuser ? ' (admin)' : ''}</option>`).join('')
    || '<option value="">No users yet</option>';
  $('userRows').innerHTML = USERS.map((user) => `
    <tr data-user="${user.id}" style="cursor:pointer">
      <td><b>${user.username}</b></td>
      <td>${[user.first_name, user.last_name].filter(Boolean).join(' ') || '—'}</td>
      <td>${user.is_superuser ? badge('admin', 'ok') : badge('labeler', 'info')}</td>
      <td class="sub">${user.email || '—'}</td>
      <td style="text-align:right">
        <button class="btn sm danger" data-delete-user="${user.id}" data-username="${user.username}"
          ${user.is_superuser ? 'disabled title="Admin cannot be deleted"' : ''}>Delete</button>
      </td>
    </tr>`).join('');
  if (SELECTED_USER) {
    const current = USERS.find((user) => user.id === SELECTED_USER.id);
    if (current) showUserDetail(current);
    else {
      SELECTED_USER = null;
      $('userDetail').style.display = 'none';
    }
  }
}

async function loadProjects() {
  PROJECTS = await api('/api/projects');
  const select = $('projectSelect');
  const preferred = PROJECTS.find((p) => p.name === 'Train Console') || PROJECTS[0];
  select.innerHTML = PROJECTS.map((p) =>
    `<option value="${p.id}" ${preferred && p.id === preferred.id ? 'selected' : ''}>${p.name}</option>`).join('');
  syncBatchProject();
}

function currentProject() {
  const projectId = Number($('projectSelect').value);
  return PROJECTS.find((project) => Number(project.id) === projectId) || null;
}

function syncBatchProject() {
  const project = currentProject();
  $('batchProject').value = project ? project.name : '';
  $('poolRoot').value = project && project.pool_root ? project.pool_root : '';
  $('takeAll').checked = Boolean(project && project.take_all);
  if (LAST_SUMMARY) {
    const remaining = LAST_SUMMARY.not_on_cvat || LAST_SUMMARY.pool_total;
    if (remaining) $('budget').value = remaining;
  }
}

async function loadJobs() {
  const projectId = $('projectSelect').value;
  const state = $('stateFilter').value;
  if (!projectId) return;
  if (PICKER.input) closePicker();
  const [jobs, summary, progress] = await Promise.all([
    api(`/api/jobs?project_id=${projectId}${state ? `&state=${encodeURIComponent(state)}` : ''}`),
    api(`/api/jobs/summary?project_id=${projectId}`),
    api(`/api/jobs/user-progress?project_id=${projectId}`),
  ]);
  renderKpis(summary);
  LAST_SUMMARY = summary;
  renderJobs(jobs);
  renderUserProgress(progress);
}

function renderKpis(summary) {
  const cards = [
    ['Images in pool', summary.pool_total],
    ['On CVAT', summary.on_cvat],
    ['Labeled', summary.labeled],
    ['Unlabeled', summary.unlabeled],
    ['Not on CVAT', summary.not_on_cvat],
  ];
  $('jobKpis').innerHTML = cards.map(([label, value]) => `
    <div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>`).join('');
}

function renderJobs(jobs) {
  const rows = $('jobRows');
  if (!jobs.length) {
    rows.innerHTML = '<tr><td colspan="6" class="empty">No jobs match this filter.</td></tr>';
    return;
  }
  rows.innerHTML = jobs.map((job) => `
    <tr>
      <td>${job.task_name || `#${job.task_id}`}</td>
      <td>#${job.id}</td>
      <td>${badge(job.state, statusKind(job.state))}</td>
      <td>
        <input class="assignee-input" data-job="${job.id}" data-previous="${escapeAttr(job.assignee || '')}"
          value="${escapeAttr(job.assignee || '')}" placeholder="Select user" autocomplete="off" spellcheck="false">
      </td>
      <td>${job.frames}</td>
      <td><a class="btn sm ghost" href="${job.url}" target="_blank" rel="noopener">CVAT ↗</a></td>
    </tr>`).join('');
}

function renderUserProgress(rows) {
  $('userProgressCount').textContent = `${rows.length} user`;
  if (!rows.length) {
    $('userProgress').innerHTML = '<div class="empty">No jobs yet.</div>';
    return;
  }
  $('userProgress').innerHTML = rows.map((row) => {
    const percent = row.frames ? Math.round((row.completed_frames / row.frames) * 100) : 0;
    return `<div style="margin-bottom:10px">
      <div class="between">
        <span>${row.assignee}</span>
        <span class="sub">${row.completed_frames}/${row.frames} frames · ${row.jobs} job</span>
      </div>
      <div class="progress sm"><div style="width:${percent}%"></div></div>
    </div>`;
  }).join('');
}

/* ------------------------------------------------------- assignee picker */

function pickerItems(query) {
  const cleaned = (query || '').trim().toLowerCase();
  const reset = { value: '', name: '', label: '— Unassigned —' };
  const users = USERS
    .filter((user) => !cleaned || user.username.toLowerCase().includes(cleaned))
    .map((user) => ({
      value: String(user.id),
      name: user.username,
      label: `${user.username}${user.is_superuser ? ' (admin)' : ''}`,
    }));
  return [reset].concat(users);
}

function renderPicker(query) {
  const menu = $('assigneeMenu');
  const items = pickerItems(query);
  if (PICKER.index > items.length - 1) PICKER.index = -1;
  menu.innerHTML = items.length
    ? items.map((item, index) => `
        <div class="picker-item ${index === PICKER.index ? 'active' : ''}"
          data-value="${item.value}" data-name="${escapeAttr(item.name)}">${item.label}</div>`).join('')
    : '<div class="picker-empty">No matching user.</div>';
  PICKER.items = items;
}

function openPicker(input) {
  const menu = $('assigneeMenu');
  PICKER.input = input;
  PICKER.index = -1;
  const rect = input.getBoundingClientRect();
  menu.style.left = `${rect.left}px`;
  menu.style.top = `${rect.bottom + 4}px`;
  menu.style.minWidth = `${Math.max(rect.width, 190)}px`;
  menu.hidden = false;
  input.value = '';
  renderPicker('');
}

function closePicker() {
  $('assigneeMenu').hidden = true;
  PICKER.input = null;
}

function movePicker(step) {
  if (!PICKER.items || !PICKER.items.length) return;
  if (PICKER.index === -1) {
    PICKER.index = step > 0 ? 0 : PICKER.items.length - 1;
  } else {
    PICKER.index = (PICKER.index + step + PICKER.items.length) % PICKER.items.length;
  }
  renderPicker(PICKER.input ? PICKER.input.value : '');
}

async function saveAssignee(input, userId) {
  const jobId = input.dataset.job;
  const previous = input.dataset.previous || '';
  input.disabled = true;
  try {
    const job = await api(`/api/jobs/${jobId}/assignee`, {
      method: 'PATCH',
      body: JSON.stringify({ user_id: userId }),
    });
    input.dataset.previous = job.assignee || '';
    input.value = job.assignee || '';
    toast(job.assignee ? `Job #${jobId} → ${job.assignee}` : `Job #${jobId} cleared assignee`);
  } catch (error) {
    input.value = previous;
    toast(error.message, 'error');
  } finally {
    input.disabled = false;
  }
}

function choosePickerItem(element) {
  const input = PICKER.input;
  if (!input || !element) return;
  const value = element.dataset.value;
  input.value = element.dataset.name || '';
  closePicker();
  saveAssignee(input, value ? Number(value) : null);
}

/* ------------------------------------------------------------ user modal */

async function createUser() {
  const button = $('createUserBtn');
  button.disabled = true;
  button.textContent = 'Creating…';
  try {
    const result = await api('/api/users', {
      method: 'POST',
      body: JSON.stringify({
        username: $('username').value.trim(),
        password: $('password').value,
        email: $('email').value.trim(),
        first_name: $('firstName').value.trim(),
        last_name: $('lastName').value.trim(),
      }),
    });
    const box = $('createUserResult');
    box.style.display = 'block';
    box.innerHTML = `
      ${badge('User created', 'ok')}
      <div class="stat-line"><span>Username</span><span class="mono">${result.username}</span></div>
      <div class="stat-line"><span>Password</span><span class="mono">${result.password}</span></div>
      <div class="stat-line"><span>Sign in at</span><span>CVAT UI</span></div>`;
    $('username').value = '';
    $('password').value = '';
    $('email').value = '';
    $('firstName').value = '';
    $('lastName').value = '';
    toast(`Created user ${result.username}`);
    await loadUsers();
    await loadJobs();
  } catch (error) {
    toast(error.message, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Create account';
  }
}

async function deleteUser(button) {
  const id = button.dataset.deleteUser;
  const username = button.dataset.username;
  if (!window.confirm(`Delete user "${username}" from CVAT? Jobs assigned to this user will be unassigned.`)) return;
  button.disabled = true;
  try {
    await api(`/api/users/${id}`, { method: 'DELETE' });
    toast(`Deleted user ${username}`);
    await loadUsers();
    await loadJobs();
  } catch (error) {
    toast(error.message, 'error');
    button.disabled = false;
  }
}

function showUserDetail(user) {
  SELECTED_USER = user;
  const box = $('userDetail');
  box.style.display = 'block';
  const role = user.is_superuser ? badge('admin', 'ok') : badge('labeler', 'info');
  box.innerHTML = `
    <div class="between"><span><b>${user.username}</b> ${role}</span><span class="sub">${user.email || ''}</span></div>
    <div class="stat-line"><span>Username</span><span class="mono">${user.username}</span></div>
    <div class="stat-line"><span>Password</span><span class="mono">${user.password || '— not stored —'}</span></div>
    <div class="row" style="margin-top:8px">
      <input id="newPassword" placeholder="New password (empty = generate)" style="flex:1">
      <button class="btn sm" id="setPasswordBtn">Set password</button>
    </div>`;
  $('setPasswordBtn').onclick = () => setUserPassword(user);
}

async function setUserPassword(user) {
  const input = $('newPassword');
  const password = input ? input.value.trim() : '';
  const button = $('setPasswordBtn');
  button.disabled = true;
  button.textContent = 'Setting…';
  try {
    const result = await api(`/api/users/${user.id}/password`, {
      method: 'PATCH',
      body: JSON.stringify({ password }),
    });
    toast(`Password for ${result.username}: ${result.password}`);
    await loadUsers();
    const current = USERS.find((item) => item.id === user.id);
    if (current) showUserDetail(current);
  } catch (error) {
    toast(error.message, 'error');
    button.disabled = false;
    button.textContent = 'Set password';
  }
}

/* ----------------------------------------------------------- batch modal */

async function previewBatch() {
  const button = $('previewBtn');
  button.disabled = true;
  button.textContent = 'Selecting samples…';
  try {
    PREVIEW = await api('/api/preview', {
      method: 'POST',
      body: JSON.stringify({
        project_id: Number($('projectSelect').value),
        pool_root: $('poolRoot').value,
        budget: Number($('budget').value),
        seed: Number($('seed').value),
        take_all: $('takeAll').checked,
      }),
    });
    const box = $('batchPreview');
    box.style.display = 'block';
    const report = PREVIEW.report || {};
    box.innerHTML = `
      <div class="between">
        <b>${report.selected || 0} images selected</b>
        ${badge('preview ready', 'ok')}
      </div>
      <div class="stat-line"><span>Sources</span><span>${Object.entries(report.per_source || {}).map(([k, v]) => `${k}: ${v}`).join(' · ')}</span></div>
      <div class="stat-line"><span>Reasons</span><span>${JSON.stringify(report.per_reason || {})}</span></div>
      <div class="stat-line"><span>Groups</span><span>${report.groups_used || 0} · max ${report.max_per_group || 0}/group</span></div>
      <div class="stat-line"><span>Preview ID</span><span class="mono">${PREVIEW.preview_id}</span></div>
      <div class="stat-line"><span>Existing labels</span><span>${PREVIEW.source_coco ? 'will import from COCO' : 'no COCO found'}</span></div>
      <div class="sub" style="margin-top:8px">First images: ${PREVIEW.items.slice(0, 3).map((item) => item.path).join(' · ')}</div>`;
    $('createBatchBtn').disabled = false;
    toast('Preview ready. Check the distribution, then create the task.');
  } catch (error) {
    toast(error.message, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Preview sample';
  }
}

async function createBatch() {
  if (!PREVIEW) return;
  const button = $('createBatchBtn');
  button.disabled = true;
  button.textContent = 'Uploading to CVAT…';
  try {
    const assignees = Array.from($('assignees').selectedOptions)
      .map((option) => Number(option.value)).filter(Boolean);
    const result = await api('/api/batches', {
      method: 'POST',
      body: JSON.stringify({
        name: $('batchName').value || `Batch ${new Date().toLocaleString()}`,
        preview_id: PREVIEW.preview_id,
        project_id: Number($('projectSelect').value),
        job_size: Number($('jobSize').value),
        assignee_ids: assignees,
        import_labels: $('importLabels').checked,
      }),
    });
    const box = $('batchPreview');
    box.innerHTML = `
      ${badge('CVAT task created', 'ok')}
      <div class="stat-line"><span>Project</span><span>#${result.project_id}</span></div>
      <div class="stat-line"><span>Task</span><span>#${result.task_id} — ${result.task_name}</span></div>
      <div class="stat-line"><span>Jobs</span><span>${result.jobs.length}</span></div>
      <div class="stat-line"><span>Imported labels</span><span>${result.imported_annotations || 0} shapes</span></div>
      ${result.import_error ? `<div class="sub" style="color:var(--danger)">${result.import_error}</div>` : ''}
      <div class="row" style="margin-top:10px">
        <a class="btn sm primary" href="${result.jobs && result.jobs.length ? result.jobs[0].url : '#'}" target="_blank" rel="noopener">Open first job in CVAT ↗</a>
      </div>`;
    toast(`Created task #${result.task_id} with ${result.jobs.length} jobs`);
    PREVIEW = null;
    $('createBatchBtn').disabled = true;
    $('createBatchBtn').textContent = 'Create CVAT task/jobs';
    await loadJobs();
  } catch (error) {
    toast(error.message, 'error');
    button.disabled = false;
    button.textContent = 'Create CVAT task/jobs';
  }
}

function openModal(id) {
  $(id).hidden = false;
}

function closeModal(id) {
  $(id).hidden = true;
}

/* ------------------------------------------------------------------ boot */

document.addEventListener('DOMContentLoaded', async () => {
  if (window.TC.page !== 'management') return;

  document.querySelectorAll('[data-close]').forEach((button) => {
    button.onclick = () => closeModal(button.dataset.close);
  });
  document.querySelectorAll('.modal').forEach((modal) => {
    modal.addEventListener('click', (event) => {
      if (event.target === modal) modal.hidden = true;
    });
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') document.querySelectorAll('.modal').forEach((modal) => { modal.hidden = true; });
  });

  $('openBatchBtn').onclick = () => {
    syncBatchProject();
    openModal('batchModal');
  };
  $('openUsersBtn').onclick = () => openModal('usersModal');
  $('refreshBtn').onclick = loadJobs;
  $('previewBtn').onclick = previewBatch;
  $('createBatchBtn').onclick = createBatch;
  $('createUserBtn').onclick = createUser;
  $('projectSelect').onchange = async () => {
    await loadJobs();
    syncBatchProject();
  };
  $('stateFilter').onchange = loadJobs;

  $('userRows').addEventListener('click', (event) => {
    const button = event.target.closest('[data-delete-user]');
    if (button) {
      if (!button.disabled) deleteUser(button);
      return;
    }
    const row = event.target.closest('tr[data-user]');
    if (!row) return;
    const user = USERS.find((item) => String(item.id) === row.dataset.user);
    if (user) showUserDetail(user);
  });

  $('jobRows').addEventListener('focusin', (event) => {
    const input = event.target.closest('.assignee-input');
    if (input) openPicker(input);
  });
  $('jobRows').addEventListener('input', (event) => {
    const input = event.target.closest('.assignee-input');
    if (!input || PICKER.input !== input) return;
    PICKER.index = USERS.length ? 1 : -1;
    renderPicker(input.value);
  });
  $('jobRows').addEventListener('keydown', (event) => {
    const input = event.target.closest('.assignee-input');
    if (!input || PICKER.input !== input) return;
    if (event.key === 'ArrowDown') { event.preventDefault(); movePicker(1); }
    if (event.key === 'ArrowUp') { event.preventDefault(); movePicker(-1); }
    if (event.key === 'Enter') {
      event.preventDefault();
      const item = $('assigneeMenu').querySelector('.picker-item.active');
      if (item) choosePickerItem(item);
    }
    if (event.key === 'Escape') {
      input.value = input.dataset.previous || '';
      closePicker();
      input.blur();
    }
  });
  $('jobRows').addEventListener('focusout', (event) => {
    const input = event.target.closest('.assignee-input');
    if (!input) return;
    setTimeout(() => {
      if (PICKER.input !== input) return;
      if (document.activeElement && document.activeElement.closest && document.activeElement.closest('#assigneeMenu')) return;
      if (input.value !== (input.dataset.previous || '')) input.value = input.dataset.previous || '';
      closePicker();
    }, 120);
  });

  $('assigneeMenu').addEventListener('mousedown', (event) => {
    const item = event.target.closest('.picker-item');
    if (!item) return;
    event.preventDefault();
    choosePickerItem(item);
  });

  document.addEventListener('mousedown', (event) => {
    if (!PICKER.input) return;
    if (event.target.closest('#assigneeMenu') || event.target === PICKER.input) return;
    PICKER.input.value = PICKER.input.dataset.previous || '';
    closePicker();
  });
  window.addEventListener('resize', closePicker);

  try {
    await loadProjects();
    await loadUsers();
    await loadJobs();
    const params = new URLSearchParams(window.location.search);
    if (params.get('batch')) openModal('batchModal');
    if (params.get('users')) openModal('usersModal');
  } catch (error) { toast(error.message, 'error'); }
});
