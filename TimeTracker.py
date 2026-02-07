from flask import Flask, jsonify, request, send_file, render_template
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
import csv
import io
import os
import glob
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.graphics.shapes import Line
from reportlab.platypus import Flowable
import config

class HLine(Flowable):
    def __init__(self, width):
        Flowable.__init__(self)
        self.width = width

    def draw(self):
        self.canv.line(0, 0, self.width, 0)

APP_ROOT = Path(__file__).parent
DB_PATH = APP_ROOT / 'time_tracker.db'

app = Flask(__name__, static_folder=str(APP_ROOT / 'static'), template_folder=str(APP_ROOT / 'templates'))

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/entries')
def entries():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT * FROM entries ORDER BY id DESC')
    rows = [dict(x) for x in cur.fetchall()]
    conn.close()
    return jsonify(rows)

@app.route('/api/start', methods=['POST'])
def start():
    data = request.json or {}
    client_id = data.get('client_id')
    project = data.get('project', 'Default')
    description = data.get('description', '')
    start_ts = datetime.utcnow().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('INSERT INTO entries (client_id, project, description, start_ts) VALUES (?, ?, ?, ?)', (client_id, project, description, start_ts))
    conn.commit()
    eid = cur.lastrowid
    conn.close()
    return jsonify({'id': eid, 'start_ts': start_ts})

@app.route('/api/stop', methods=['POST'])
def stop():
    data = request.json or {}
    eid = data.get('id')
    now = datetime.utcnow()
    conn = get_conn()
    cur = conn.cursor()
    if eid:
        cur.execute('SELECT * FROM entries WHERE id = ?', (eid,))
        row = cur.fetchone()
    else:
        cur.execute('SELECT * FROM entries WHERE end_ts IS NULL ORDER BY id DESC LIMIT 1')
        row = cur.fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'No running entry found'}), 404
    start = datetime.fromisoformat(row['start_ts'])
    end_ts = now.isoformat()
    duration_min = int((now - start).total_seconds() // 60)
    cur.execute('UPDATE entries SET end_ts = ?, duration_min = ? WHERE id = ?', (end_ts, duration_min, row['id']))
    conn.commit()
    conn.close()
    return jsonify({'id': row['id'], 'end_ts': end_ts, 'duration_min': duration_min})

@app.route('/api/add', methods=['POST'])
def add_manual():
    data = request.json or {}
    client_id = data.get('client_id')
    project = data.get('project', 'Default')
    description = data.get('description', '')
    start_ts = data.get('start_ts')
    end_ts = data.get('end_ts')
    if not start_ts or not end_ts:
        return jsonify({'error': 'start_ts and end_ts required (ISO format)'}), 400
    start = datetime.fromisoformat(start_ts)
    end = datetime.fromisoformat(end_ts)
    duration_min = int((end - start).total_seconds() // 60)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('INSERT INTO entries (client_id, project, description, start_ts, end_ts, duration_min) VALUES (?, ?, ?, ?, ?, ?)', (client_id, project, description, start_ts, end_ts, duration_min))
    conn.commit()
    eid = cur.lastrowid
    conn.close()
    return jsonify({'id': eid})

@app.route('/api/delete', methods=['POST'])
def delete():
    data = request.json or {}
    eid = data.get('id')
    if not eid:
        return jsonify({'error': 'id required'}), 400
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('DELETE FROM entries WHERE id = ?', (eid,))
    conn.commit()
    conn.close()
    return jsonify({'deleted': eid})

@app.route('/api/export')
def export_csv():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT * FROM entries ORDER BY id')
    rows = cur.fetchall()
    conn.close()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['id', 'client_id', 'project', 'description', 'start_ts', 'end_ts', 'duration_min'])
    for r in rows:
        writer.writerow([r['id'], r['client_id'], r['project'], r['description'], r['start_ts'], r['end_ts'], r['duration_min']])
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='time_entries.csv')

@app.route('/api/edit', methods=['POST'])
def edit():
    data = request.json or {}
    eid = data.get('id')
    if not eid:
        return jsonify({'error': 'id required'}), 400

    conn = get_conn()
    cur = conn.cursor()

    cur.execute('SELECT * FROM entries WHERE id = ?', (eid,))
    existing_entry = cur.fetchone()

    if not existing_entry:
        conn.close()
        return jsonify({'error': 'Entry not found'}), 404

    client_id = data.get('client_id', existing_entry['client_id'])
    project = data.get('project', existing_entry['project'])
    description = data.get('description', existing_entry['description'])
    start_ts_str = data.get('start_ts', existing_entry['start_ts'])
    end_ts_str = data.get('end_ts', existing_entry['end_ts'])

    duration_min = existing_entry['duration_min']

    # Recalculate duration if start_ts or end_ts changed or were passed
    if 'start_ts' in data or 'end_ts' in data:
        if start_ts_str and end_ts_str:
            start = datetime.fromisoformat(start_ts_str)
            end = datetime.fromisoformat(end_ts_str)
            duration_min = int((end - start).total_seconds() // 60)
        else:
            duration_min = None

    cur.execute(
        'UPDATE entries SET client_id = ?, project = ?, description = ?, start_ts = ?, end_ts = ?, duration_min = ? WHERE id = ?',
        (client_id, project, description, start_ts_str, end_ts_str, duration_min, eid)
    )
    conn.commit()
    conn.close()
    return jsonify({'id': eid, 'updated': True})

@app.route('/api/projects/summary')
def projects_summary():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('''
        SELECT 
            e.project, 
            SUM(e.duration_min) as total_duration,
            SUM(CAST(e.duration_min AS REAL) / 60.0 * IFNULL(c.hourly_rate, 0)) as billed_value,
            IFNULL(p.target_hours, 0) as target_hours
        FROM entries e
        LEFT JOIN clients c ON e.client_id = c.id
        LEFT JOIN projects p ON e.project = p.name
        WHERE e.duration_min IS NOT NULL
        GROUP BY e.project
        ORDER BY total_duration DESC
    ''')
    rows = [dict(x) for x in cur.fetchall()]
    for row in rows:
        row['total_hours'] = round(row['total_duration'] / 60, 2)
        row['billed_value'] = round(row['billed_value'], 2)
    conn.close()
    return jsonify(rows)

@app.route('/api/projects/target', methods=['POST'])
def update_project_target():
    data = request.json or {}
    name = data.get('project')
    target = data.get('target_hours', 0)
    if not name:
        return jsonify({'error': 'project name required'}), 400
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO projects (name, target_hours) 
        VALUES (?, ?) 
        ON CONFLICT(name) DO UPDATE SET target_hours = excluded.target_hours
    ''', (name, target))
    conn.commit()
    conn.close()
    return jsonify({'project': name, 'target_hours': target})

@app.route('/api/clients')
def get_clients():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT * FROM clients ORDER BY id')
    rows = [dict(x) for x in cur.fetchall()]
    conn.close()
    return jsonify(rows)

@app.route('/api/clients', methods=['POST'])
def add_client():
    data = request.json or {}
    name = data.get('name')
    email = data.get('email')
    phone = data.get('phone')
    address = data.get('address')
    hourly_rate = data.get('hourly_rate', config.HOURLY_RATE)
    if not name:
        return jsonify({'error': 'name required'}), 400
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('INSERT INTO clients (name, email, phone, address, hourly_rate) VALUES (?, ?, ?, ?, ?)', (name, email, phone, address, hourly_rate))
    conn.commit()
    cid = cur.lastrowid
    conn.close()
    return jsonify({'id': cid})

@app.route('/api/clients/<int:client_id>', methods=['PUT'])
def update_client(client_id):
    data = request.json or {}
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT * FROM clients WHERE id = ?', (client_id,))
    if not cur.fetchone():
        conn.close()
        return jsonify({'error': 'Client not found'}), 404
    name = data.get('name')
    email = data.get('email')
    phone = data.get('phone')
    address = data.get('address')
    hourly_rate = data.get('hourly_rate')
    updates = []
    params = []
    if name is not None:
        updates.append('name = ?')
        params.append(name)
    if email is not None:
        updates.append('email = ?')
        params.append(email)
    if phone is not None:
        updates.append('phone = ?')
        params.append(phone)
    if address is not None:
        updates.append('address = ?')
        params.append(address)
    if hourly_rate is not None:
        updates.append('hourly_rate = ?')
        params.append(hourly_rate)
    if not updates:
        conn.close()
        return jsonify({'error': 'No fields to update'}), 400
    params.append(client_id)
    cur.execute(f'UPDATE clients SET {", ".join(updates)} WHERE id = ?', params)
    conn.commit()
    conn.close()
    return jsonify({'updated': client_id})

@app.route('/api/clients/<int:client_id>', methods=['DELETE'])
def delete_client(client_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('DELETE FROM clients WHERE id = ?', (client_id,))
    conn.commit()
    conn.close()
    return jsonify({'deleted': client_id})

@app.route('/generate_invoice/<int:client_id>')
def generate_invoice(client_id):
    draft = request.args.get('draft', '0') == '1'
    conn = get_conn()
    cur = conn.cursor()
    # Get client
    cur.execute('SELECT * FROM clients WHERE id = ?', (client_id,))
    client = cur.fetchone()
    if not client:
        conn.close()
        return jsonify({'error': 'Client not found'}), 404

    # Get time entries for client
    cur.execute('SELECT * FROM entries WHERE client_id = ? AND duration_min IS NOT NULL ORDER BY start_ts', (client_id,))
    entries = cur.fetchall()

    if not entries:
        conn.close()
        return jsonify({'error': 'No time entries for client'}), 404

    # Calculate totals
    total_min = sum(e['duration_min'] for e in entries)
    total_hours = round(total_min / 60, 2)
    hourly_rate = client['hourly_rate'] or config.HOURLY_RATE
    total_amount = round(total_hours * hourly_rate, 2)

    # Date range
    start_dates = [datetime.fromisoformat(e['start_ts']).date() for e in entries]
    min_date = min(start_dates)
    max_date = max(start_dates)
    date_range = f"{min_date} to {max_date}"

    # Invoice number based on folder and DB
    invoices_files = glob.glob('invoices/INV-*.pdf')
    file_nums = [int(f.split('-')[1].split('.')[0]) for f in invoices_files] if invoices_files else [0]
    max_file = max(file_nums)

    cur.execute('SELECT MAX(CAST(SUBSTR(invoice_number, 5) AS INTEGER)) FROM invoices')
    max_db = cur.fetchone()[0] or 0

    max_num = max(max_file, max_db)
    invoice_number = f"{config.INVOICE_PREFIX}{max_num + 1:03d}"

    # Invoice date and due date
    invoice_date = datetime.now().date()
    due_date = invoice_date + timedelta(days=30)

    # Insert invoice into DB only if not draft
    if not draft:
        cur.execute('INSERT INTO invoices (invoice_number, client_id, invoice_date, due_date, total_hours, total_amount) VALUES (?, ?, ?, ?, ?, ?)',
                    (invoice_number, client_id, invoice_date.isoformat(), due_date.isoformat(), total_hours, total_amount))
        conn.commit()
    conn.close()

    # Create invoices folder
    invoices_dir = Path('invoices')
    invoices_dir.mkdir(exist_ok=True)
    prefix = 'DRAFT-' if draft else ''
    pdf_path = invoices_dir / f'{prefix}{invoice_number}.pdf'

    # Generate PDF
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='RightAlign', alignment=TA_RIGHT))
    styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='LeftAlign', alignment=TA_LEFT))
    story = []

    # Header table: Business left, Client right, aligned per row
    data = [
        [Paragraph(f"{config.MY_NAME}", styles['LeftAlign']), Paragraph(f"Bill To: {client['name']}", styles['RightAlign'])],
    ]
    if config.BUSINESS_NAME:
        data.append([Paragraph(config.BUSINESS_NAME, styles['LeftAlign']), ''])
    data.append([Paragraph(config.MY_ADDRESS, styles['LeftAlign']), Paragraph(client['address'] or '', styles['RightAlign'])])
    data.append([Paragraph(f"{config.MY_PHONE} | {config.MY_EMAIL}", styles['LeftAlign']), Paragraph(f"{client['email'] or ''} | {client['phone'] or ''}".strip(' | '), styles['RightAlign'])])

    table = Table(data, colWidths=[300, 200])
    table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Horizontal line
    story.append(HLine(500))
    story.append(Spacer(1, 12))

    # Invoice title
    story.append(Paragraph(f"Invoice #{invoice_number}", styles['Title']))
    story.append(Spacer(1, 12))

    # Invoice details
    data = [
        ['Invoice Date:', str(invoice_date), 'Due Date:', str(due_date)],
    ]
    table = Table(data, colWidths=[100, 100, 100, 100])
    table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Horizontal line
    story.append(HLine(500))
    story.append(Spacer(1, 12))

    # Service description
    service_desc = f"Consulting Services: {total_hours} hours @ ${hourly_rate}/hr ({date_range})"
    story.append(Paragraph(service_desc, styles['Normal']))
    story.append(Spacer(1, 12))

    # Table of entries
    data = [['Date', 'Description', 'Hours', 'Subtotal']]
    for e in entries:
        date = datetime.fromisoformat(e['start_ts']).date()
        desc = e['description'] or e['project']
        hours = round(e['duration_min'] / 60, 2)
        subtotal = round(hours * hourly_rate, 2)
        data.append([str(date), desc, f"{hours}", f"${subtotal:.2f}"])
    data.append(['', 'Total:', str(total_hours), f"${total_amount:.2f}"])

    table = Table(data, colWidths=[80, 250, 60, 80])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Horizontal line
    story.append(HLine(500))
    story.append(Spacer(1, 12))

    # Payment terms
    story.append(Paragraph("Payment Terms:", styles['Heading3']))
    story.append(Paragraph(config.PAYMENT_TERMS, styles['Normal']))
    story.append(Paragraph(config.LATE_FEE_POLICY, styles['Normal']))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Payment Instructions:", styles['Heading3']))
    story.append(Paragraph(f"Bank: {config.BANK_NAME} | Routing: {config.ROUTING_NUM} | Account: {config.ACCT_NUM}", styles['Normal']))
    story.append(Paragraph(f"PayPal: {config.PAYPAL_EMAIL}", styles['Normal']))

    doc.build(story)

    return send_file(str(pdf_path), mimetype='application/pdf', as_attachment=True, download_name=f'{prefix}{invoice_number}.pdf')

if __name__ == '__main__':
    app.run(debug=True)
