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
  const date = new Date(isoStr.endsWith('Z') ? isoStr : isoStr + 'Z');
  return date.toLocaleString();
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
    tr.classList.toggle('active-row', isActive);
    tr.innerHTML = `
      <td>${e.id}</td>
      <td class="editable" data-field="client_id">${clientName}</td>
      <td class="editable" data-field="project">${e.project}</td>
      <td class="editable" data-field="description">${e.description || ''}</td>
      <td class="editable" data-field="start_ts">${formatDate(e.start_ts)}</td>
      <td class="editable" data-field="end_ts">${formatDate(e.end_ts)}</td>
      <td class="duration-cell" data-start="${e.start_ts}">${e.duration_min || '0'}</td>
      <td>${isActive ? '<span class="status-badge active">● Active</span>' : '<span class="status-badge">Done</span>'}</td>
      <td><button data-id="${e.id}" class="del">Delete</button></td>
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

  document.querySelectorAll('#entriesTable .editable').forEach(cell => {
    cell.addEventListener('dblclick', ev => {
      const field = ev.target.dataset.field;
      const id = ev.target.parentNode.dataset.id;
      const originalValue = ev.target.textContent;
      if (field === 'client_id') {
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
      } else {
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

async function loadEntries() {
  allEntries = await api('/api/entries');
  currentPage = 1;
  renderEntriesPage();
}

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
  if (document.getElementById('invDraft').checked) params.set('draft', '1');
  const qs = params.toString() ? `?${params.toString()}` : '';
  window.open(`/generate_invoice/${invoiceClientId}${qs}`, '_blank');
  document.getElementById('invoiceModal').style.display = 'none';
});

document.getElementById('invCancelBtn').addEventListener('click', () => {
  document.getElementById('invoiceModal').style.display = 'none';
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

async function loadExpenses() {
  const data = await api('/api/expenses');
  const tbody = document.querySelector('#expensesTable tbody');
  tbody.innerHTML = '';
  data.forEach(exp => {
    const tr = document.createElement('tr');
    tr.dataset.id = exp.id;
    tr.innerHTML = `
      <td>${exp.id}</td>
      <td>${exp.expense_date}</td>
      <td>${exp.vendor}</td>
      <td>$${parseFloat(exp.amount).toFixed(2)}</td>
      <td>${exp.category || ''}</td>
      <td>${exp.client_name || ''}</td>
      <td>${exp.project || ''}</td>
      <td>${exp.reimbursable ? 'Yes' : 'No'}</td>
      <td>${exp.source || ''}</td>
      <td><button class="delExp" data-id="${exp.id}">Delete</button></td>
    `;
    tbody.appendChild(tr);
  });
  document.querySelectorAll('.delExp').forEach(b => b.addEventListener('click', async ev => {
    const id = ev.target.dataset.id;
    if (confirm(`Delete expense ${id}?`)) {
      await api(`/api/expenses/${id}`, 'DELETE');
      loadExpenses();
    }
  }));
}

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
  openExpenseModal(data.parsed, 'email', null, null);
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
}

// Show the LAN URL next to the QR code in Settings
fetch('/qr.png', { method: 'HEAD' }).then(() => {
  const el = document.getElementById('qrUrl');
  if (el) el.textContent = window.location.origin;
});

loadAll();

// PWA service worker
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {});
}

setInterval(() => {
  document.querySelectorAll('.active-row .duration-cell').forEach(cell => {
    const startStr = cell.dataset.start;
    const start = new Date(startStr.endsWith('Z') ? startStr : startStr + 'Z');
    const now = new Date();
    cell.textContent = Math.floor((now - start) / 60000);
  });
}, 30000);
