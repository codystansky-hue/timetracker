let clients = [];

function formatDate(isoStr) {
  if (!isoStr) return '';
  // Ensure we append 'Z' if it is missing and it's UTC from server
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

async function loadEntries() {
  const data = await api('/api/entries');
  const tbody = document.querySelector('#entriesTable tbody');
  tbody.innerHTML = '';
  data.forEach(e => {
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
  document.querySelectorAll('.del').forEach(b => b.addEventListener('click', async (ev) => {
    const id = ev.target.dataset.id;
    if (confirm(`Delete entry ${id}?`)) {
      await api('/api/delete', 'POST', { id });
      loadAll();
    }
  }));
  document.querySelectorAll('#entriesTable .editable').forEach(cell => {
    cell.addEventListener('dblclick', (ev) => {
      const field = ev.target.dataset.field;
      const id = ev.target.parentNode.dataset.id;
      const originalValue = ev.target.textContent;
      if (field === 'client_id') {
        // Handle client selection differently
        const clientSelect = document.createElement('select');
        clientSelect.innerHTML = '<option value="">Select Client</option>';
        clients.forEach(c => {
          const option = document.createElement('option');
          option.value = c.id;
          option.textContent = c.name;
          if (c.id == originalValue) option.selected = true;
          clientSelect.appendChild(option);
        });
        clientSelect.addEventListener('change', async () => {
          const newValue = clientSelect.value;
          if (newValue !== originalValue) {
            await api('/api/edit', 'POST', { id, client_id: newValue });
            loadAll();
          }
          ev.target.textContent = clientSelect.selectedOptions[0]?.textContent || '';
        });
        ev.target.innerHTML = '';
        ev.target.appendChild(clientSelect);
        clientSelect.focus();
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

async function loadClients() {
  clients = await api('/api/clients');
  const select = document.getElementById('clientSelect');
  select.innerHTML = '<option value="">Select Client</option>';
  clients.forEach(c => {
    const option = document.createElement('option');
    option.value = c.id;
    option.textContent = c.name;
    select.appendChild(option);
  });
  const lastClient = localStorage.getItem('lastClient');
  if (lastClient) select.value = lastClient;

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
        <button data-id="${c.id}" class="generateInvoice">Generate Invoice</button>
        <button data-id="${c.id}" class="delClient">Delete</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
  document.querySelectorAll('.generateInvoice').forEach(b => b.addEventListener('click', (ev) => {
    const id = ev.target.dataset.id;
    window.location = `/generate_invoice/${id}`;
  }));
  document.querySelectorAll('.delClient').forEach(b => b.addEventListener('click', async (ev) => {
    const id = ev.target.dataset.id;
    if (confirm(`Delete client ${id}?`)) {
      await api(`/api/clients/${id}`, 'DELETE');
      loadAll();
    }
  }));
  document.querySelectorAll('#clientsTable .editable').forEach(cell => {
    cell.addEventListener('dblclick', (ev) => {
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
    cell.addEventListener('dblclick', async (ev) => {
      const project = ev.target.dataset.project;
      const originalValue = ev.target.textContent;
      const newValue = prompt(`Set target hours for ${project}`, originalValue);
      if (newValue !== null && newValue !== originalValue) {
        await api('/api/projects/target', 'POST', { project, target_hours: parseFloat(newValue) });
        loadProjectsSummary();
      }
    });
  });
}

async function loadAll() {
  await loadClients();
  loadEntries();
  loadProjectsSummary();
}

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
  document.getElementById('clientName').value = '';
  document.getElementById('clientEmail').value = '';
  document.getElementById('clientPhone').value = '';
  document.getElementById('clientAddress').value = '';
  document.getElementById('clientRate').value = '';
  loadAll();
});

document.getElementById('project').value = localStorage.getItem('lastProject') || '';

loadAll();

setInterval(() => {
  document.querySelectorAll('.active-row .duration-cell').forEach(cell => {
    const startStr = cell.dataset.start;
    const start = new Date(startStr.endsWith('Z') ? startStr : startStr + 'Z');
    const now = new Date();
    const diffMin = Math.floor((now - start) / 60000);
    cell.textContent = diffMin;
  });
}, 30000);
