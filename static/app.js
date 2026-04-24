let clients = [];
let allEntries = [];
let currentPage = 1;
const PAGE_SIZE = 20;
let expenseModalState = {};
let invoiceClientId = null;
let receiptQueue = [];      // pending crops waiting for review

// ── Utilities ────────────────────────────────────────────────────────────────

function formatDate(isoStr) {
  if (!isoStr) return '';
  const hastz = isoStr.endsWith('Z') || /[+-]\d{2}:\d{2}$/.test(isoStr);
  const d = new Date(hastz ? isoStr : isoStr + 'Z');
  return isNaN(d) ? isoStr : d.toLocaleString();
}

function fmtShortDate(dateStr) {
  if (!dateStr) return '';
  const [y, m, d] = dateStr.split('-');
  return `${parseInt(m)}/${parseInt(d)}/${y.slice(2)}`;
}

function fmtDuration(minutes) {
  if (!minutes) return '0h';
  const hrs = Math.round(minutes / 15) * 0.25;
  return hrs % 1 === 0 ? hrs + 'h' : hrs.toFixed(2) + 'h';
}

function liveDurationMin(e) {
  if (e.end_ts) return e.duration_min || 0;
  const baseStr = e.resumed_at || e.start_ts;
  if (!baseStr) return e.duration_min || 0;
  const hastz = baseStr.endsWith('Z') || /[+-]\d{2}:\d{2}$/.test(baseStr);
  const base = new Date(hastz ? baseStr : baseStr + 'Z');
  const elapsed = Math.floor((Date.now() - base) / 60000);
  const baseline = e.resumed_at ? (e.duration_min || 0) : 0;
  return baseline + elapsed;
}

async function api(path, method = 'GET', body) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  if (res.headers.get('content-type')?.includes('application/json')) return res.json();
  return res;
}

// ── Navigation ───────────────────────────────────────────────────────────────

document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const section = btn.dataset.section;
    document.querySelectorAll('.section').forEach(s => s.style.display = 'none');
    document.getElementById(`section-${section}`).style.display = 'block';
  });
});

// ── Time Entries (with pagination) ───────────────────────────────────────────

function renderEntriesPage() {
  const totalPages = Math.max(1, Math.ceil(allEntries.length / PAGE_SIZE));
  currentPage = Math.min(currentPage, totalPages);

  const start = (currentPage - 1) * PAGE_SIZE;
  const pageEntries = allEntries.slice(start, start + PAGE_SIZE);

  document.getElementById('pageInfo').textContent = `Page ${currentPage} of ${totalPages}`;
  document.getElementById('prevPage').disabled = currentPage <= 1;
  document.getElementById('nextPage').disabled = currentPage >= totalPages;

  const tbody = document.querySelector('#entriesTable tbody');
  tbody.innerHTML = '';
  pageEntries.forEach(e => {
    const clientName = clients.find(c => c.id == e.client_id)?.name || '';
    const tr = document.createElement('tr');
    tr.dataset.id = e.id;
    const isActive = !e.end_ts;
    const resumedAt = e.resumed_at || '';
    const baselineMin = resumedAt ? (e.duration_min || 0) : 0;
    const timerAnchor = resumedAt || e.start_ts || '';
    const canResume = !isActive && !e.invoice_id;
    tr.classList.toggle('active-row', isActive);
    tr.innerHTML = `
      <td>${e.id}</td>
      <td class="editable" data-field="client_id">${clientName}</td>
      <td class="editable" data-field="project">${e.project}</td>
      <td class="editable" data-field="description">${e.description || ''}</td>
      <td class="editable" data-field="start_ts" data-raw="${e.start_ts || ''}">${formatDate(e.start_ts)}</td>
      <td class="editable" data-field="end_ts" data-raw="${e.end_ts || ''}">${formatDate(e.end_ts)}</td>
      <td class="duration-cell" data-start="${timerAnchor}" data-baseline="${baselineMin}">${fmtDuration(liveDurationMin(e))}</td>
      <td>${isActive ? (resumedAt ? '<span class="status-badge active">● Resumed</span>' : '<span class="status-badge active">● Active</span>') : e.invoice_id ? '<span class="status-badge billed">Billed</span>' : '<span class="status-badge">Done</span>'}</td>
      <td>
        ${canResume ? `<button data-id="${e.id}" class="resume">Resume</button>` : ''}
        <button data-id="${e.id}" class="del">Delete</button>
      </td>
    `;
    tbody.appendChild(tr);
  });

  document.querySelectorAll('.del').forEach(b => b.addEventListener('click', async ev => {
    const id = ev.target.dataset.id;
    if (confirm(`Delete entry ${id}?`)) {
      await api('/api/delete', 'POST', { id });
      loadAll();
    }
  }));

  document.querySelectorAll('.resume').forEach(b => b.addEventListener('click', async ev => {
    const id = ev.target.dataset.id;
    const res = await api('/api/resume', 'POST', { id });
    if (res && res.error) { alert(res.error); return; }
    loadAll();
  }));

  document.querySelectorAll('#entriesTable .editable').forEach(cell => {
    cell.addEventListener('dblclick', ev => {
      const field = ev.target.dataset.field;
      const id = ev.target.parentNode.dataset.id;

      if (field === 'client_id') {
        const originalValue = ev.target.textContent;
        const sel = document.createElement('select');
        sel.innerHTML = '<option value="">Select Client</option>';
        clients.forEach(c => {
          const opt = document.createElement('option');
          opt.value = c.id;
          opt.textContent = c.name;
          if (c.id == originalValue) opt.selected = true;
          sel.appendChild(opt);
        });
        sel.addEventListener('change', async () => {
          await api('/api/edit', 'POST', { id, client_id: sel.value });
          loadAll();
        });
        ev.target.innerHTML = '';
        ev.target.appendChild(sel);
        sel.focus();
      } else if (field === 'start_ts' || field === 'end_ts') {
        // Use raw ISO string → convert to datetime-local value (local time, no tz suffix)
        const rawIso = ev.target.dataset.raw;
        const hastz = rawIso.endsWith('Z') || /[+-]\d{2}:\d{2}$/.test(rawIso);
        const d = rawIso ? new Date(hastz ? rawIso : rawIso + 'Z') : new Date();
        const localIso = new Date(d.getTime() - d.getTimezoneOffset() * 60000)
          .toISOString().slice(0, 16);
        const inp = document.createElement('input');
        inp.type = 'datetime-local';
        inp.value = localIso;
        inp.style.width = '180px';
        let clickingButton = false;
        document.addEventListener('mousedown', e => {
          if (e.target.tagName === 'BUTTON') clickingButton = true;
        }, { capture: true, once: true });
        const save = async (e) => {
          if (clickingButton) { clickingButton = false; loadAll(); return; }
          const chosen = inp.value;
          if (!chosen) { loadAll(); return; }
          // Store as UTC ISO string
          const utc = new Date(chosen).toISOString();
          const body = { id };
          body[field] = utc;
          await api('/api/edit', 'POST', body);
          loadAll();
        };
        inp.addEventListener('blur', save);
        inp.addEventListener('keydown', e => { if (e.key === 'Enter') inp.blur(); if (e.key === 'Escape') loadAll(); });
        ev.target.textContent = '';
        ev.target.appendChild(inp);
        inp.focus();
      } else {
        const originalValue = ev.target.textContent;
        const newValue = prompt(`Edit ${field}`, originalValue);
        if (newValue !== null && newValue !== originalValue) {
          const body = { id };
          body[field] = newValue;
          api('/api/edit', 'POST', body).then(loadAll);
        }
      }
    });
  });
}

function isoDate(d) {
  const tzOffset = d.getTimezoneOffset() * 60000;
  return new Date(d.getTime() - tzOffset).toISOString().slice(0, 10);
}

// Compute [start, end] dates for a named preset. Weeks are Monday–Sunday.
function presetRange(preset) {
  const today = new Date(); today.setHours(0, 0, 0, 0);
  if (preset === 'last_week') {
    const dayOfWeek = (today.getDay() + 6) % 7; // Mon=0 … Sun=6
    const thisMonday = new Date(today); thisMonday.setDate(today.getDate() - dayOfWeek);
    const lastMonday = new Date(thisMonday); lastMonday.setDate(thisMonday.getDate() - 7);
    const lastSunday = new Date(lastMonday); lastSunday.setDate(lastMonday.getDate() + 6);
    return [isoDate(lastMonday), isoDate(lastSunday)];
  }
  if (preset === 'last_month') {
    const first = new Date(today.getFullYear(), today.getMonth() - 1, 1);
    const last = new Date(today.getFullYear(), today.getMonth(), 0);
    return [isoDate(first), isoDate(last)];
  }
  return null;
}

function renderFilteredTotals() {
  const filterClientId = document.getElementById('filterClient').value;
  const filterStartDate = document.getElementById('filterStartDate').value;
  const filterEndDate = document.getElementById('filterEndDate').value;
  const filterStatus = document.getElementById('filterStatus').value;
  const active = !!(filterClientId || filterStartDate || filterEndDate || (filterStatus && filterStatus !== 'all'));
  const el = document.getElementById('filteredTotals');
  if (!active) { el.style.display = 'none'; return; }

  let totalMin = 0, totalValue = 0;
  allEntries.forEach(e => {
    if (!e.duration_min) return;
    totalMin += e.duration_min;
    const rate = clients.find(c => c.id == e.client_id)?.hourly_rate || 0;
    totalValue += (e.duration_min / 60) * rate;
  });
  const hrs = Math.round(totalMin / 15) * 0.25;
  document.getElementById('ftEntries').textContent = allEntries.length;
  document.getElementById('ftHours').textContent = (hrs % 1 === 0 ? hrs + 'h' : hrs.toFixed(2) + 'h');
  document.getElementById('ftValue').textContent = '$' + totalValue.toFixed(2);
  el.style.display = '';
}

async function loadEntries() {
  const filterClientId = document.getElementById('filterClient').value;
  const filterStartDate = document.getElementById('filterStartDate').value;
  const filterEndDate = document.getElementById('filterEndDate').value;
  const filterStatus = document.getElementById('filterStatus').value;

  let url = '/api/entries?';
  const params = new URLSearchParams();
  if (filterClientId) params.append('client_id', filterClientId);
  if (filterStartDate) params.append('start_date', filterStartDate);
  if (filterEndDate) params.append('end_date', filterEndDate);
  if (filterStatus && filterStatus !== 'all') params.append('status', filterStatus);

  allEntries = await api(url + params.toString());
  currentPage = 1;
  renderEntriesPage();
  renderFilteredTotals();
}

document.getElementById('applyFilterBtn').addEventListener('click', loadEntries);
document.getElementById('clearFilterBtn').addEventListener('click', () => {
  document.getElementById('filterClient').value = '';
  document.getElementById('filterPreset').value = '';
  document.getElementById('filterStartDate').value = '';
  document.getElementById('filterEndDate').value = '';
  document.getElementById('filterStatus').value = 'all';
  loadEntries();
});

document.getElementById('filterPreset').addEventListener('change', ev => {
  const range = presetRange(ev.target.value);
  if (!range) return;
  document.getElementById('filterStartDate').value = range[0];
  document.getElementById('filterEndDate').value = range[1];
  loadEntries();
});

document.getElementById('prevPage').addEventListener('click', () => {
  if (currentPage > 1) { currentPage--; renderEntriesPage(); }
});
document.getElementById('nextPage').addEventListener('click', () => {
  const totalPages = Math.ceil(allEntries.length / PAGE_SIZE);
  if (currentPage < totalPages) { currentPage++; renderEntriesPage(); }
});

// ── Clients ──────────────────────────────────────────────────────────────────

async function loadClients() {
  clients = await api('/api/clients');

  // Time tracking client dropdown
  const select = document.getElementById('clientSelect');
  select.innerHTML = '<option value="">Select Client</option>';
  clients.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c.id;
    opt.textContent = c.name;
    select.appendChild(opt);
  });
  const lastClient = localStorage.getItem('lastClient');
  if (lastClient) select.value = lastClient;

  
  const filterClient = document.getElementById('filterClient');
  if (filterClient) {
    const currentFilterClient = filterClient.value;
    filterClient.innerHTML = '<option value="">All Clients</option>';
    clients.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.id;
      opt.textContent = c.name;
      filterClient.appendChild(opt);
    });
    if (currentFilterClient) filterClient.value = currentFilterClient;
  }

  // Expense client dropdowns
  ['expClientSelect', 'mClientSelect'].forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    const current = sel.value;
    sel.innerHTML = '<option value="">No Client</option>';
    clients.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.id;
      opt.textContent = c.name;
      sel.appendChild(opt);
    });
    if (current) sel.value = current;
  });

  // Clients table
  const tbody = document.querySelector('#clientsTable tbody');
  tbody.innerHTML = '';
  clients.forEach(c => {
    const tr = document.createElement('tr');
    tr.dataset.id = c.id;
    tr.innerHTML = `
      <td>${c.id}</td>
      <td class="editable" data-field="name">${c.name}</td>
      <td class="editable" data-field="email">${c.email || ''}</td>
      <td class="editable" data-field="phone">${c.phone || ''}</td>
      <td class="editable" data-field="address">${c.address || ''}</td>
      <td class="editable" data-field="hourly_rate">${c.hourly_rate || ''}</td>
      <td>
        <button data-id="${c.id}" data-name="${c.name}" class="generateInvoice">Invoice</button>
        <button data-id="${c.id}" class="delClient">Delete</button>
      </td>
    `;
    tbody.appendChild(tr);
  });

  document.querySelectorAll('.generateInvoice').forEach(b => b.addEventListener('click', ev => {
    invoiceClientId = ev.target.dataset.id;
    document.getElementById('invoiceClientName').textContent = ev.target.dataset.name;
    document.getElementById('invStartDate').value = '';
    document.getElementById('invEndDate').value = '';
    document.getElementById('invDraft').checked = false;
    document.getElementById('invIncludeExpenses').checked = true;
    document.getElementById('invoiceModal').style.display = 'flex';
  }));

  document.querySelectorAll('.delClient').forEach(b => b.addEventListener('click', async ev => {
    const id = ev.target.dataset.id;
    if (confirm(`Delete client ${id}?`)) {
      await api(`/api/clients/${id}`, 'DELETE');
      loadAll();
    }
  }));

  document.querySelectorAll('#clientsTable .editable').forEach(cell => {
    cell.addEventListener('dblclick', ev => {
      const field = ev.target.dataset.field;
      const id = ev.target.parentNode.dataset.id;
      const originalValue = ev.target.textContent;
      const newValue = prompt(`Edit ${field}`, originalValue);
      if (newValue !== null && newValue !== originalValue) {
        const body = {};
        body[field] = newValue;
        api(`/api/clients/${id}`, 'PUT', body).then(loadAll);
      }
    });
  });
}

// ── Invoice modal ─────────────────────────────────────────────────────────────

document.getElementById('invGenerateBtn').addEventListener('click', () => {
  if (!invoiceClientId) return;
  const params = new URLSearchParams();
  const start = document.getElementById('invStartDate').value;
  const end = document.getElementById('invEndDate').value;
  if (start) params.set('start_date', start);
  if (end) params.set('end_date', end);
  if (!document.getElementById('invIncludeExpenses').checked) params.set('include_expenses', '0');
  const isDraft = document.getElementById('invDraft').checked;
  if (isDraft) params.set('draft', '1');
  const qs = params.toString() ? `?${params.toString()}` : '';
  window.open(`/generate_invoice/${invoiceClientId}${qs}`, '_blank');
  document.getElementById('invoiceModal').style.display = 'none';

  // If not a draft, auto-navigate to Invoices tab and refresh list
  if (!isDraft) {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.section').forEach(s => s.style.display = 'none');
    const btn = document.querySelector('[data-section="invoices"]');
    if (btn) btn.classList.add('active');
    const sec = document.getElementById('section-invoices');
    if (sec) sec.style.display = '';
    setTimeout(() => loadInvoices(), 800);
  }
});

document.getElementById('invCancelBtn').addEventListener('click', () => {
  document.getElementById('invoiceModal').style.display = 'none';
});

// ── Expense Report modal ───────────────────────────────────────────────────────

document.getElementById('expReportBtn').addEventListener('click', () => {
  const sel = document.getElementById('erClientSelect');
  sel.innerHTML = '<option value="">All Clients</option>';
  clients.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c.id;
    opt.textContent = c.name;
    sel.appendChild(opt);
  });
  document.getElementById('erStartDate').value = '';
  document.getElementById('erEndDate').value = '';
  document.getElementById('expReportModal').style.display = 'flex';
});

document.getElementById('erPdfBtn').addEventListener('click', () => {
  const params = new URLSearchParams();
  const clientId = document.getElementById('erClientSelect').value;
  const start = document.getElementById('erStartDate').value;
  const end = document.getElementById('erEndDate').value;
  if (clientId) params.set('client_id', clientId);
  if (start) params.set('start_date', start);
  if (end) params.set('end_date', end);
  const qs = params.toString() ? `?${params.toString()}` : '';
  window.open(`/api/expenses/report${qs}`, '_blank');
  document.getElementById('expReportModal').style.display = 'none';
});

document.getElementById('erCsvBtn').addEventListener('click', () => {
  const params = new URLSearchParams();
  const clientId = document.getElementById('erClientSelect').value;
  const start = document.getElementById('erStartDate').value;
  const end = document.getElementById('erEndDate').value;
  if (clientId) params.set('client_id', clientId);
  if (start) params.set('start_date', start);
  if (end) params.set('end_date', end);
  const qs = params.toString() ? `?${params.toString()}` : '';
  window.location = `/api/expenses/export${qs}`;
  document.getElementById('expReportModal').style.display = 'none';
});

document.getElementById('erZipBtn').addEventListener('click', () => {
  const params = new URLSearchParams();
  const clientId = document.getElementById('erClientSelect').value;
  const start = document.getElementById('erStartDate').value;
  const end = document.getElementById('erEndDate').value;
  if (clientId) params.set('client_id', clientId);
  if (start) params.set('start_date', start);
  if (end) params.set('end_date', end);
  const qs = params.toString() ? `?${params.toString()}` : '';
  window.location = `/api/expenses/receipts-zip${qs}`;
  document.getElementById('expReportModal').style.display = 'none';
});

document.getElementById('erCancelBtn').addEventListener('click', () => {
  document.getElementById('expReportModal').style.display = 'none';
});

// ── Project Summary ───────────────────────────────────────────────────────────

async function loadProjectsSummary() {
  const data = await api('/api/projects/summary');
  const tbody = document.querySelector('#projectsTable tbody');
  tbody.innerHTML = '';
  data.forEach(p => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${p.project}</td>
      <td>${p.total_hours}</td>
      <td>$${p.billed_value.toFixed(2)}</td>
      <td class="editable target-hours" data-project="${p.project}">${p.target_hours || '0'}</td>
    `;
    tbody.appendChild(tr);
  });
  document.querySelectorAll('.target-hours').forEach(cell => {
    cell.addEventListener('dblclick', async ev => {
      const project = ev.target.dataset.project;
      const newValue = prompt(`Set target hours for ${project}`, ev.target.textContent);
      if (newValue !== null) {
        await api('/api/projects/target', 'POST', { project, target_hours: parseFloat(newValue) });
        loadProjectsSummary();
      }
    });
  });
}

// ── Expenses ──────────────────────────────────────────────────────────────────

const EXPENSE_CATS = ['other', 'travel', 'hotel', 'meal', 'office'];

function escHtml(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

async function loadExpenses() {
  const data = await api('/api/expenses');

  // Refresh bulk-bar client dropdown
  const bulkClientSel = document.getElementById('expBulkClient');
  bulkClientSel.innerHTML = '<option value="">No Client</option>';
  clients.forEach(c => {
    const o = document.createElement('option');
    o.value = c.id; o.textContent = c.name;
    bulkClientSel.appendChild(o);
  });

  const tbody = document.querySelector('#expensesTable tbody');
  tbody.innerHTML = '';

  data.forEach(exp => {
    const tr = document.createElement('tr');
    tr.dataset.id = exp.id;
    tr.innerHTML = `
      <td><input type="checkbox" class="exp-check" data-id="${exp.id}" /></td>
      <td>${exp.id}</td>
      <td class="exp-cell" data-field="expense_date" data-id="${exp.id}" data-raw="${exp.expense_date || ''}">${exp.expense_date || ''}</td>
      <td class="exp-cell" data-field="vendor"       data-id="${exp.id}" data-raw="${escHtml(exp.vendor || '')}">${escHtml(exp.vendor || '')}</td>
      <td class="exp-cell" data-field="amount"       data-id="${exp.id}" data-raw="${exp.amount}">$${parseFloat(exp.amount).toFixed(2)}</td>
      <td class="exp-cell" data-field="category"     data-id="${exp.id}" data-raw="${exp.category || 'other'}">${exp.category || ''}</td>
      <td class="exp-cell" data-field="client_id"    data-id="${exp.id}" data-raw="${exp.client_id || ''}">${exp.client_name || '—'}</td>
      <td class="exp-cell" data-field="project"      data-id="${exp.id}" data-raw="${escHtml(exp.project || '')}">${escHtml(exp.project || '')}</td>
      <td class="exp-cell" data-field="reimbursable" data-id="${exp.id}" data-raw="${exp.reimbursable ? 1 : 0}">${exp.reimbursable ? 'Yes' : 'No'}</td>
      <td>${exp.source || ''}</td>
      <td class="receipt-cell">
        ${exp.receipt_path
          ? `<div class="receipt-wrap">
               <a href="/${exp.receipt_path}" target="_blank" title="View receipt">
                 <img src="/${exp.receipt_path}" class="receipt-thumb" alt="receipt" />
               </a>
               <button class="removeReceipt btn-secondary" data-id="${exp.id}" title="Remove receipt">✕</button>
             </div>`
          : `<label class="attach-btn btn-secondary" title="Attach receipt">
               + <input type="file" class="attachReceiptInput" data-id="${exp.id}" accept="image/*,.pdf" style="display:none" />
             </label>`}
      </td>
      <td><button class="delExp btn-secondary" data-id="${exp.id}">Delete</button></td>
    `;
    tbody.appendChild(tr);
  });

  tbody.querySelectorAll('.exp-cell').forEach(cell => cell.addEventListener('click', expCellClick));
  tbody.querySelectorAll('.exp-check').forEach(cb => cb.addEventListener('change', updateExpBulkBar));
  tbody.querySelectorAll('.attachReceiptInput').forEach(input => {
    input.addEventListener('change', async ev => {
      const file = ev.target.files[0];
      if (!file) return;
      const id = ev.target.dataset.id;
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(`/api/expenses/${id}/receipt`, { method: 'POST', body: formData });
      const data = await res.json();
      if (data.error) return alert(`Upload error: ${data.error}`);
      loadExpenses();
    });
  });

  tbody.querySelectorAll('.removeReceipt').forEach(b => b.addEventListener('click', async ev => {
    const id = ev.target.dataset.id;
    if (!confirm('Remove this receipt?')) return;
    await api(`/api/expenses/${id}`, 'PUT', { receipt_path: '' });
    loadExpenses();
  }));

  tbody.querySelectorAll('.delExp').forEach(b => b.addEventListener('click', async ev => {
    const id = ev.target.dataset.id;
    if (confirm(`Delete expense ${id}?`)) {
      await api(`/api/expenses/${id}`, 'DELETE');
      loadExpenses();
    }
  }));

  updateExpBulkBar();
}

// Inline cell editor
function expCellClick(ev) {
  const cell = ev.currentTarget;
  if (cell.querySelector('input, select')) return; // already open

  const { field, id, raw } = cell.dataset;
  const originalText = cell.textContent.trim();

  // Reimbursable: just toggle, no input needed
  if (field === 'reimbursable') {
    const newVal = raw === '1' ? 0 : 1;
    api(`/api/expenses/${id}`, 'PUT', { reimbursable: newVal }).then(() => loadExpenses());
    return;
  }

  let el;
  if (field === 'expense_date') {
    el = Object.assign(document.createElement('input'), { type: 'date', value: raw });
  } else if (field === 'amount') {
    el = Object.assign(document.createElement('input'), { type: 'number', step: '0.01', value: raw });
  } else if (field === 'category') {
    el = document.createElement('select');
    EXPENSE_CATS.forEach(cat => {
      const o = Object.assign(document.createElement('option'), { value: cat, textContent: cat });
      if (cat === raw) o.selected = true;
      el.appendChild(o);
    });
  } else if (field === 'client_id') {
    el = document.createElement('select');
    el.innerHTML = '<option value="">No Client</option>';
    clients.forEach(c => {
      const o = Object.assign(document.createElement('option'), { value: c.id, textContent: c.name });
      if (String(c.id) === String(raw)) o.selected = true;
      el.appendChild(o);
    });
  } else {
    el = Object.assign(document.createElement('input'), { type: 'text', value: raw });
  }

  const done = { v: false };

  async function save() {
    if (done.v) return; done.v = true;
    let val = el.value;
    if (field === 'client_id') val = val || null;
    if (field === 'amount') val = parseFloat(val);
    await api(`/api/expenses/${id}`, 'PUT', { [field]: val });
    loadExpenses();
  }

  function cancel() {
    if (done.v) return; done.v = true;
    cell.textContent = originalText;
    cell.addEventListener('click', expCellClick);
  }

  cell.textContent = '';
  cell.appendChild(el);
  el.focus();
  if (el.type !== 'select-one' && el.type !== 'number' && el.select) el.select();

  el.addEventListener('blur', save);
  el.addEventListener('keydown', e => {
    if (e.key === 'Enter')  { e.preventDefault(); el.blur(); }
    if (e.key === 'Escape') { e.preventDefault(); cancel(); }
  });
  el.addEventListener('click', e => e.stopPropagation());
}

// Bulk helpers
function getCheckedExpenseIds() {
  return [...document.querySelectorAll('.exp-check:checked')].map(cb => cb.dataset.id);
}

function updateExpBulkBar() {
  const ids = getCheckedExpenseIds();
  const allCbs = document.querySelectorAll('.exp-check');
  const allCb  = document.getElementById('expCheckAll');
  document.getElementById('expBulkBar').style.display = ids.length ? 'flex' : 'none';
  document.getElementById('expBulkCount').textContent  = `${ids.length} selected`;
  if (allCb) {
    allCb.checked       = ids.length > 0 && ids.length === allCbs.length;
    allCb.indeterminate = ids.length > 0 && ids.length < allCbs.length;
  }
}

// Select-all checkbox
document.getElementById('expCheckAll').addEventListener('change', ev => {
  document.querySelectorAll('.exp-check').forEach(cb => cb.checked = ev.target.checked);
  updateExpBulkBar();
});

// Bulk apply date
document.getElementById('expBulkApplyDate').addEventListener('click', async () => {
  const date = document.getElementById('expBulkDate').value;
  if (!date) return alert('Enter a date to apply');
  const ids = getCheckedExpenseIds();
  await Promise.all(ids.map(id => api(`/api/expenses/${id}`, 'PUT', { expense_date: date })));
  loadExpenses();
  loadInvoices();
});

// Bulk apply client
document.getElementById('expBulkApplyClient').addEventListener('click', async () => {
  const clientId = document.getElementById('expBulkClient').value || null;
  const ids = getCheckedExpenseIds();
  await Promise.all(ids.map(id => api(`/api/expenses/${id}`, 'PUT', { client_id: clientId })));
  loadExpenses();
  loadInvoices();
});

// Clear selection
document.getElementById('expBulkClear').addEventListener('click', () => {
  document.querySelectorAll('.exp-check').forEach(cb => cb.checked = false);
  const allCb = document.getElementById('expCheckAll');
  if (allCb) { allCb.checked = false; allCb.indeterminate = false; }
  updateExpBulkBar();
});

// Expense tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab-panel').forEach(panel => {
      panel.style.display = panel.id === `tab-${tab}` ? 'flex' : 'none';
    });
  });
});

// Manual expense add
document.getElementById('addExpenseBtn').addEventListener('click', async () => {
  const vendor = document.getElementById('expVendor').value.trim();
  const amount = parseFloat(document.getElementById('expAmount').value);
  const expense_date = document.getElementById('expDate').value;
  if (!vendor || isNaN(amount) || !expense_date) return alert('Vendor, amount, and date are required');
  await api('/api/expenses', 'POST', {
    vendor, amount, expense_date,
    client_id: document.getElementById('expClientSelect').value || null,
    project: document.getElementById('expProject').value.trim(),
    category: document.getElementById('expCategory').value,
    description: document.getElementById('expDescription').value.trim(),
    reimbursable: document.getElementById('expReimbursable').checked,
    source: 'manual',
  });
  ['expVendor', 'expAmount', 'expDate', 'expDescription', 'expProject'].forEach(id => {
    document.getElementById(id).value = '';
  });
  loadAll();
});

// Receipt upload / OCR (with multi-receipt queue)
document.getElementById('uploadReceiptBtn').addEventListener('click', async () => {
  const fileInput = document.getElementById('receiptFile');
  const status = document.getElementById('receiptStatus');
  if (!fileInput.files[0]) return alert('Please select an image file');
  status.textContent = 'Uploading & scanning…';
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  try {
    const res = await fetch('/api/expenses/parse-receipt', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) { status.textContent = `Error: ${data.error}`; return; }
    const n = data.count;
    status.textContent = n > 1 ? `Found ${n} receipts — reviewing one by one.` : 'Parsed — please review.';
    receiptQueue = [...data.results];
    openNextFromQueue();
  } catch (e) {
    status.textContent = `Error: ${e.message}`;
  }
});

function openNextFromQueue() {
  if (receiptQueue.length === 0) { loadAll(); return; }
  const item = receiptQueue.shift();
  openExpenseModal(item.parsed, 'receipt', item.receipt_path, item.raw_text, receiptQueue.length);
}

// Email parse
document.getElementById('parseEmailBtn').addEventListener('click', async () => {
  const text = document.getElementById('emailText').value.trim();
  if (!text) return alert('Please paste email confirmation text first');
  const data = await api('/api/expenses/parse-email', 'POST', { text });
  if (data.error) return alert(`Error: ${data.error}`);
  if (data.results && data.results.length > 0) {
    const status = document.getElementById('receiptStatus');
    status.textContent = `Found ${data.count} trips — reviewing one by one.`;
    receiptQueue = [...data.results];
    openNextFromQueue();
  } else {
    openExpenseModal(data.parsed, 'email', null, null);
  }
});

// Auto-categorise when vendor name is entered in the modal
document.getElementById('mVendor').addEventListener('blur', async () => {
  const vendor = document.getElementById('mVendor').value.trim();
  if (!vendor) return;
  const data = await api(`/api/expenses/lookup-vendor?name=${encodeURIComponent(vendor)}`);
  const hint = document.getElementById('categoryHint');
  if (data.category) {
    document.getElementById('mCategory').value = data.category;
    hint.textContent = `✓ auto-categorized`;
  } else {
    hint.textContent = '';
  }
});

// Expense review modal
function openExpenseModal(parsed, source, receiptPath, rawText, queueRemaining = 0) {
  expenseModalState = { source, receiptPath, queueRemaining };
  const label = document.getElementById('modalQueueLabel');
  label.textContent = queueRemaining > 0 ? `(${queueRemaining} more after this)` : '';
  const saveBtn = document.getElementById('modalSaveBtn');
  saveBtn.textContent = queueRemaining > 0 ? 'Save & Next' : 'Save';
  const cancelBtn = document.getElementById('modalCancelBtn');
  cancelBtn.textContent = queueRemaining > 0 ? 'Skip' : 'Cancel';
  document.getElementById('mVendor').value = parsed.vendor || '';
  document.getElementById('mAmount').value = parsed.amount || '';
  document.getElementById('mDate').value = parsed.expense_date || '';
  document.getElementById('mEndDate').value = parsed.end_date || '';
  document.getElementById('mCategory').value = parsed.category || 'other';
  document.getElementById('mProject').value = parsed.project || '';
  document.getElementById('mDescription').value = parsed.description || '';
  document.getElementById('mReimbursable').checked = parsed.reimbursable !== false;
  const mSel = document.getElementById('mClientSelect');
  mSel.innerHTML = '<option value="">No Client</option>';
  clients.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c.id;
    opt.textContent = c.name;
    mSel.appendChild(opt);
  });
  if (parsed.client_id) mSel.value = parsed.client_id;
  const rawBlock = document.getElementById('rawTextBlock');
  if (rawText) {
    rawBlock.style.display = 'block';
    document.getElementById('rawTextContent').textContent = rawText;
  } else {
    rawBlock.style.display = 'none';
  }
  document.getElementById('expenseModal').style.display = 'flex';
}

document.getElementById('modalSaveBtn').addEventListener('click', async () => {
  const vendor = document.getElementById('mVendor').value.trim();
  const amount = parseFloat(document.getElementById('mAmount').value);
  const expense_date = document.getElementById('mDate').value;
  if (!vendor || isNaN(amount) || !expense_date) return alert('Vendor, amount, and date are required');
  await api('/api/expenses', 'POST', {
    vendor, amount, expense_date,
    category: document.getElementById('mCategory').value,
    client_id: document.getElementById('mClientSelect').value || null,
    project: document.getElementById('mProject').value.trim(),
    description: document.getElementById('mDescription').value.trim(),
    reimbursable: document.getElementById('mReimbursable').checked,
    source: expenseModalState.source || 'manual',
    receipt_path: expenseModalState.receiptPath || '',
  });
  document.getElementById('expenseModal').style.display = 'none';
  // Continue queue if there are more receipts, otherwise reload
  if (receiptQueue.length > 0) openNextFromQueue(); else loadAll();
});

document.getElementById('modalCancelBtn').addEventListener('click', () => {
  document.getElementById('expenseModal').style.display = 'none';
  // Skip = advance queue without saving
  if (receiptQueue.length > 0) openNextFromQueue();
});

// ── Controls ──────────────────────────────────────────────────────────────────

document.getElementById('startBtn').addEventListener('click', async () => {
  const client_id = document.getElementById('clientSelect').value;
  const project = document.getElementById('project').value || 'Default';
  localStorage.setItem('lastProject', project);
  localStorage.setItem('lastClient', client_id);
  const description = document.getElementById('description').value || '';
  await api('/api/start', 'POST', { client_id, project, description });
  loadAll();
});

document.getElementById('stopBtn').addEventListener('click', async () => {
  await api('/api/stop', 'POST', {});
  loadAll();
});

document.getElementById('exportBtn').addEventListener('click', () => {
  window.location = '/api/export';
});

document.getElementById('addClientBtn').addEventListener('click', async () => {
  const name = document.getElementById('clientName').value;
  const email = document.getElementById('clientEmail').value;
  const phone = document.getElementById('clientPhone').value;
  const address = document.getElementById('clientAddress').value;
  const hourly_rate = parseFloat(document.getElementById('clientRate').value) || null;
  if (!name) return alert('Client name required');
  await api('/api/clients', 'POST', { name, email, phone, address, hourly_rate });
  ['clientName', 'clientEmail', 'clientPhone', 'clientAddress', 'clientRate'].forEach(id => {
    document.getElementById(id).value = '';
  });
  loadAll();
});

document.getElementById('project').value = localStorage.getItem('lastProject') || '';

// ── Load all ──────────────────────────────────────────────────────────────────

async function loadAll() {
  await loadClients();
  loadEntries();
  loadProjectsSummary();
  loadExpenses();
  loadInvoices();
}

// Show the LAN URL next to the QR code in Settings
fetch('/qr.png', { method: 'HEAD' }).then(() => {
  const el = document.getElementById('qrUrl');
  if (el) el.textContent = window.location.origin;
});

loadAll();

// ── Sync status indicator ─────────────────────────────────────────────────

const syncEl = document.getElementById('syncIndicator');
const syncDot = syncEl.querySelector('.sync-dot');
const syncLabel = syncEl.querySelector('.sync-label');

function renderSyncStatus(s) {
  syncEl.classList.remove('synced', 'error', 'syncing');
  if (s.in_sync === true) {
    syncEl.classList.add('synced');
    syncLabel.textContent = `✓ ${s.local_commit}`;
    syncEl.title = `In sync with remote\nCommit: ${s.local_commit}\nLast sync: ${s.last_sync ? new Date(s.last_sync).toLocaleString() : '—'}\nClick to sync now`;
  } else if (s.success === false && s.last_sync) {
    syncEl.classList.add('error');
    syncLabel.textContent = `✗ ${s.local_commit || '?'}`;
    syncEl.title = `Sync error: ${s.error}\nClick to retry`;
  } else if (s.last_sync === null) {
    syncLabel.textContent = '—';
    syncEl.title = 'Sync status unknown — click to sync now';
  } else {
    syncEl.classList.add('error');
    syncLabel.textContent = `✗ ${s.local_commit || '?'}`;
    syncEl.title = `Out of sync: local ${s.local_commit} / remote ${s.remote_commit}\nClick to sync now`;
  }
}

async function fetchSyncStatus() {
  try {
    const s = await api('/sync-status');
    renderSyncStatus(s);
  } catch (_) {}
}

syncEl.addEventListener('click', async () => {
  syncEl.classList.remove('synced', 'error');
  syncEl.classList.add('syncing');
  syncLabel.textContent = 'Syncing…';
  try {
    const s = await api('/sync-now', 'POST');
    renderSyncStatus(s);
  } catch (_) {
    syncEl.classList.remove('syncing');
    syncEl.classList.add('error');
    syncLabel.textContent = '✗ failed';
  }
});

fetchSyncStatus();
setInterval(fetchSyncStatus, 60000);

// PWA service worker
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {});
}

setInterval(() => {
  document.querySelectorAll('.active-row .duration-cell').forEach(cell => {
    const startStr = cell.dataset.start;
    if (!startStr) return;
    const baseline = parseInt(cell.dataset.baseline) || 0;
    const hastz = startStr.endsWith('Z') || /[+-]\d{2}:\d{2}$/.test(startStr);
    const start = new Date(hastz ? startStr : startStr + 'Z');
    const now = new Date();
    cell.textContent = fmtDuration(baseline + Math.floor((now - start) / 60000));
  });
}, 30000);


// ── Invoices ──────────────────────────────────────────────────────────────────

async function loadInvoices() {
  const data = await api('/api/invoices');
  const tbody = document.querySelector('#invoicesTable tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  const today = new Date(); today.setHours(0,0,0,0);
  let outstanding = 0, overdue = 0, dueSoon = 0;

  data.forEach(inv => {
    const tr = document.createElement('tr');
    tr.dataset.id = inv.id;
    const isPaid = inv.status === 'paid';
    const total = parseFloat(inv.total_amount || 0) + parseFloat(inv.expense_total || 0);

    // Due-date awareness
    let daysLabel = '—';
    if (inv.due_date) {
      const due = new Date(inv.due_date); due.setHours(0,0,0,0);
      const diff = Math.round((due - today) / 86400000);
      if (!isPaid) {
        if (diff < 0) {
          daysLabel = `${Math.abs(diff)}d overdue`;
          tr.classList.add('row-overdue');
          overdue += total;
        } else if (diff <= 7) {
          daysLabel = `${diff}d`;
          tr.classList.add('row-due-soon');
          dueSoon += total;
        } else if (diff <= 30) {
          daysLabel = `${diff}d`;
          tr.classList.add('row-due-soon-mild');
          dueSoon += total;
        } else {
          daysLabel = `${diff}d`;
        }
      } else {
        daysLabel = diff < 0 ? 'Paid (was overdue)' : 'Paid';
        tr.classList.add('row-paid');
      }
    }
    if (!isPaid) outstanding += total;

    tr.innerHTML = `
      <td>${inv.id}</td>
      <td>${inv.invoice_number}</td>
      <td>${inv.client_name || '—'}</td>
      <td>${inv.invoice_date}</td>
      <td>${inv.due_date || '—'}</td>
      <td class="days-cell">${daysLabel}</td>
      <td>$${total.toFixed(2)}</td>
      <td>${isPaid ? '<span class="status-badge billed">Paid</span>' : '<span class="status-badge">Unpaid</span>'}</td>
      <td>
        <label class="inline-check">
          <input type="checkbox" class="invoice-status-cb" data-id="${inv.id}" ${isPaid ? 'checked' : ''} />
          Paid
        </label>
        <a href="/api/invoices/${inv.id}/download" target="_blank" class="btn-secondary" style="margin-left:8px;padding:4px 8px;text-decoration:none;font-size:0.85rem">PDF</a>
      </td>
    `;
    tbody.appendChild(tr);
  });

  // Update invoice summary cards
  document.getElementById('summaryOutstanding').textContent = `$${outstanding.toFixed(2)}`;
  document.getElementById('summaryOverdue').textContent = `$${overdue.toFixed(2)}`;
  document.getElementById('summaryDueSoon').textContent = `$${dueSoon.toFixed(2)}`;

  // Fetch and display unbilled time
  const unbilled = await api('/api/invoices/unbilled-summary');
  document.getElementById('summaryUnbilled').textContent = `$${unbilled.total.toFixed(2)}`;
  document.getElementById('summaryUnbilledBreakdown').innerHTML = unbilled.clients.map(c => {
    const dateRange = (c.earliest && c.latest)
      ? `<span class="breakdown-dates">${fmtShortDate(c.earliest)} – ${fmtShortDate(c.latest)}</span>`
      : '';
    return `<div class="breakdown-row"><span>${c.client}${dateRange}</span><span>$${c.amount.toFixed(2)}</span></div>`;
  }).join('');

  document.querySelectorAll('.invoice-status-cb').forEach(cb => {
    cb.addEventListener('change', async ev => {
      const id = ev.target.dataset.id;
      const newStatus = ev.target.checked ? 'paid' : 'unpaid';
      await api('/api/invoices/' + id, 'PUT', { status: newStatus });
      loadInvoices();
    });
  });
}

// ── Shutdown ─────────────────────────────────────────────────────────────────

document.getElementById('shutdownBtn').addEventListener('click', async () => {
  if (!confirm('Stop the Time Tracker server?')) return;
  await fetch('/api/shutdown', { method: 'POST' });
  document.body.innerHTML = '<p style="font-family:sans-serif;padding:2rem">Server stopped. You can close this tab.</p>';
});
