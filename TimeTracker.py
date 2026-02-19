from flask import Flask, jsonify, request, send_file, render_template
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import csv
import io
import os
import glob
import re
import uuid
from collections import defaultdict
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.graphics.shapes import Line
from reportlab.platypus import Flowable
import pytesseract
import qrcode
from PIL import Image
from werkzeug.utils import secure_filename
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
import config

class HLine(Flowable):
    def __init__(self, width):
        Flowable.__init__(self)
        self.width = width

    def draw(self):
        self.canv.line(0, 0, self.width, 0)

APP_ROOT = Path(__file__).parent
DB_PATH = APP_ROOT / 'time_tracker.db'
UPLOADS_DIR = APP_ROOT / 'uploads'
UPLOADS_DIR.mkdir(exist_ok=True)
ALLOWED_RECEIPT_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.pdf'}

app = Flask(__name__, static_folder=str(APP_ROOT / 'static'), template_folder=str(APP_ROOT / 'templates'))

def git_sync(message="Sync database"):
    """Sync the database file with the git repository."""
    if not os.path.exists(DB_PATH):
        return
    try:
        # Check if we are in a git repo
        subprocess.run(['git', 'rev-parse', '--is-inside-work-tree'], check=True, capture_output=True, cwd=APP_ROOT)
        
        # Pull latest changes first
        subprocess.run(['git', 'pull', '--rebase'], check=True, cwd=APP_ROOT)
        
        # Add, commit and push
        subprocess.run(['git', 'add', str(DB_PATH)], check=True, cwd=APP_ROOT)
        # Check for changes to avoid empty commits
        status = subprocess.run(['git', 'status', '--porcelain', str(DB_PATH)], check=True, capture_output=True, text=True, cwd=APP_ROOT)
        if status.stdout.strip():
            subprocess.run(['git', 'commit', '-m', message], check=True, cwd=APP_ROOT)
            # Push to origin
            subprocess.run(['git', 'push'], check=True, cwd=APP_ROOT)
    except Exception as e:
        app.logger.error(f"Git sync failed: {e}")

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
    git_sync(f"Start entry: {project} - {description}")
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
    git_sync(f"Stop entry: {row['id']}")
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
    git_sync(f"Manual entry added: {eid}")
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
    git_sync(f"Deleted entry: {eid}")
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
    git_sync(f"Updated entry: {eid}")
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
    git_sync(f"Updated client: {client_id}")
    return jsonify({'updated': client_id})

@app.route('/api/clients/<int:client_id>', methods=['DELETE'])
def delete_client(client_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('DELETE FROM clients WHERE id = ?', (client_id,))
    conn.commit()
    conn.close()
    git_sync(f"Deleted client: {client_id}")
    return jsonify({'deleted': client_id})

# ---------------------------------------------------------------------------
# PDF support
# ---------------------------------------------------------------------------

def _pdf_to_images(pdf_path):
    """
    Render each page of a PDF to a PIL Image at 300 DPI.
    Returns an empty list if pdf2image or poppler-utils are unavailable.
    """
    try:
        from pdf2image import convert_from_path
        return convert_from_path(str(pdf_path), dpi=300)
    except ImportError:
        app.logger.error('pdf2image not installed — run: pip install pdf2image')
        return []
    except Exception as e:
        app.logger.error(f'PDF render failed: {e}')
        return []


# ---------------------------------------------------------------------------
# Multi-receipt segmentation (OpenCV)
# ---------------------------------------------------------------------------

def _order_points(pts):
    """Return points ordered: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image, pts):
    """Perspective-correct a quadrilateral region to a flat rectangle."""
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = max(int(widthA),  int(widthB))
    maxH = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxW - 1, 0],
                    [maxW - 1, maxH - 1], [0, maxH - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def _find_gap_splits(projection, dim, min_gap_px=20, edge_margin=0.15):
    """
    Find sustained low-density gaps in a 1-D projection profile.

    A position is a 'gap' when its smoothed value is less than 3 % of the
    local maximum within a window of ~8 % of the dimension on each side.
    This local-contrast approach handles the scanner case where receipts
    (white on white) create only subtle density differences.
    """
    if len(projection) < min_gap_px * 2:
        return []

    # Smooth first to suppress isolated spikes (e.g., a stray dark pixel
    # in the middle of a genuine gap between two receipts)
    win    = max(3, min_gap_px // 2)
    smooth = np.convolve(projection.astype(float),
                         np.ones(win) / win, mode='same')

    # Local window for computing the surrounding maximum
    srch = max(50, int(dim * 0.08))

    # Precompute rolling max with numpy for speed
    padded   = np.pad(smooth, srch, mode='edge')
    roll_max = np.array([padded[i:i + 2 * srch + 1].max()
                         for i in range(dim)])

    # A position is a gap if its smoothed value < 3 % of its local maximum
    is_gap = smooth <= np.maximum(roll_max * 0.03, 2.0)

    margin_lo = int(dim * edge_margin)
    margin_hi = int(dim * (1 - edge_margin))
    splits    = []
    in_run    = False
    run_start = 0

    for i, g in enumerate(is_gap.tolist()):
        if g and not in_run:
            run_start = i
            in_run    = True
        elif not g and in_run:
            run_len = i - run_start
            centre  = (run_start + i) // 2
            if run_len >= min_gap_px and margin_lo <= centre <= margin_hi:
                splits.append(centre)
            in_run = False

    return splits


def _deepest_gap(projection, lo_frac=0.2, hi_frac=0.8, smooth_win=31):
    """
    Find the single deepest local-minimum position in a projection within the
    middle lo_frac–hi_frac fraction of the dimension.  Returns None if the
    minimum is not substantially lower than its neighbours (no real gap).
    """
    dim = len(projection)
    lo  = int(dim * lo_frac)
    hi  = int(dim * hi_frac)
    if hi <= lo:
        return None

    # Smooth to remove spikes
    win    = max(3, smooth_win)
    smooth = np.convolve(projection.astype(float), np.ones(win) / win, mode='same')

    mid       = smooth[lo:hi]
    min_local = float(mid.min())
    max_local = float(mid.max())

    # Only consider it a real gap if the minimum is < 5 % of the local max
    if max_local < 1 or min_local > max_local * 0.05:
        return None

    return int(np.argmin(mid)) + lo


def _segment_by_projection(img):
    """
    Scanner fallback: split a page into receipt regions.

    Finds the single deepest column gap and deepest row gap in the middle
    portion of the page, then cuts into up to a 2×2 grid.  Empty cells
    (< 500 dark pixels) are discarded.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = (gray < 200).astype(np.int32)

    col_proj = dark.sum(axis=0)
    row_proj = dark.sum(axis=1)

    col_split = _deepest_gap(col_proj, lo_frac=0.2, hi_frac=0.8, smooth_win=max(15, w // 100))
    row_split = _deepest_gap(row_proj, lo_frac=0.2, hi_frac=0.8, smooth_win=max(15, h // 100))

    col_boundaries = ([0, col_split, w] if col_split else [0, w])
    row_boundaries = ([0, row_split, h] if row_split else [0, h])

    pad   = max(5, int(min(w, h) * 0.004))
    crops = []
    for ri in range(len(row_boundaries) - 1):
        for ci in range(len(col_boundaries) - 1):
            y0 = max(0, row_boundaries[ri]     - pad)
            y1 = min(h, row_boundaries[ri + 1] + pad)
            x0 = max(0, col_boundaries[ci]     - pad)
            x1 = min(w, col_boundaries[ci + 1] + pad)
            region = img[y0:y1, x0:x1]
            if dark[y0:y1, x0:x1].sum() < 500:
                continue
            crops.append(Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB)))

    return crops


def _segment_receipts(image_path):
    """
    Detect individual receipts in a photo/scan.

    Uses a two-stage pipeline:
    1. Edge/threshold-based contour detection (works well for photos on dark
       backgrounds).
    2. Text-density blob detection fallback (works well for flatbed scans
       where all paper is white-on-white and edge contrast is minimal).
    """
    if not CV2_AVAILABLE:
        return [Image.open(str(image_path))]

    img = cv2.imread(str(image_path))
    if img is None:
        return [Image.open(str(image_path))]

    orig = img.copy()
    h, w = img.shape[:2]

    scale = min(1.0, 1600 / max(h, w))
    small = cv2.resize(img, (int(w * scale), int(h * scale)))
    sh, sw = small.shape[:2]

    gray    = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Method A: Canny edges — good for any background with visible edges
    canny = cv2.Canny(blurred, 20, 80)
    k_canny = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, k_canny)
    canny = cv2.dilate(canny, k_canny, iterations=2)

    # Method B: Otsu threshold — good for light receipts on dark backgrounds
    _, otsu = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([otsu[0, :], otsu[-1, :], otsu[:, 0], otsu[:, -1]])
    light_background = np.mean(border) > 128
    if light_background:
        otsu = cv2.bitwise_not(otsu)
    k_otsu = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k_otsu)

    combined = cv2.bitwise_or(canny, otsu)

    contours, _ = cv2.findContours(combined.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    min_area   = sw * sh * 0.03
    crops      = []
    used_rects = []

    for c in contours:
        if cv2.contourArea(c) < min_area:
            break
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, cw, ch = cv2.boundingRect(approx)

        aspect = ch / cw if cw else 0
        if not (0.15 <= aspect <= 12):
            continue

        overlap = False
        for (rx, ry, rw, rh) in used_rects:
            ix = max(0, min(x + cw, rx + rw) - max(x, rx))
            iy = max(0, min(y + ch, ry + rh) - max(y, ry))
            if ix * iy > 0.4 * min(cw * ch, rw * rh):
                overlap = True
                break
        if overlap:
            continue

        used_rects.append((x, y, cw, ch))
        mg = int(0.005 * max(sw, sh))

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype('float32') / scale
            try:
                warped = _four_point_transform(orig, pts)
                crops.append(Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)))
            except Exception:
                pass
        else:
            x0 = max(0, int(x / scale) - mg)
            y0 = max(0, int(y / scale) - mg)
            x1 = min(w, int((x + cw) / scale) + mg)
            y1 = min(h, int((y + ch) / scale) + mg)
            crop = orig[y0:y1, x0:x1]
            if crop.size:
                crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

    # If we're on a light-background (scanner) image and found fewer than 3
    # distinct receipts, the edge/threshold approach likely missed receipts.
    # Fall back to projection-profile splitting which handles white-on-white better.
    if light_background and len(crops) < 3:
        proj_crops = _segment_by_projection(orig)
        if len(proj_crops) > len(crops):
            crops = proj_crops

    # Final pass: split any crop that covers > 45 % of the page width and is
    # wider than it is tall — two receipts may have merged horizontally.
    final_crops = []
    for crop_pil in (crops if crops else [Image.open(str(image_path))]):
        cw2, ch2 = crop_pil.size
        if cw2 > w * 0.45 and cw2 > ch2 * 1.5:
            mid = cw2 // 2
            final_crops.extend([crop_pil.crop((0, 0, mid, ch2)),
                                 crop_pil.crop((mid, 0, cw2, ch2))])
        elif ch2 > h * 0.45 and ch2 > cw2 * 2.0 and cw2 > w * 0.4:
            mid = ch2 // 2
            final_crops.extend([crop_pil.crop((0, 0, cw2, mid)),
                                 crop_pil.crop((0, mid, cw2, ch2))])
        else:
            final_crops.append(crop_pil)

    return final_crops if final_crops else [Image.open(str(image_path))]


def _fix_rotation(pil_img):
    """
    Ask Tesseract's OSD engine what rotation is needed and apply it.
    Handles 90 / 180 / 270-degree flips — the most common cause of garbled
    OCR on scanned receipts.
    """
    try:
        osd   = pytesseract.image_to_osd(pil_img, output_type=pytesseract.Output.DICT)
        angle = osd.get('rotate', 0)
        if angle:
            # OSD 'rotate' is the clockwise degrees needed to correct;
            # PIL.rotate() is counter-clockwise, so negate.
            pil_img = pil_img.rotate(-angle, expand=True)
    except Exception:
        pass          # OSD needs ≥ a few lines of text; silently skip if it fails
    return pil_img


def _preprocess_for_ocr(pil_img):
    """
    Correct orientation, sharpen contrast, and binarise a receipt image
    before handing it to Tesseract.
    """
    # Rotation correction first (on the full-colour image — OSD is more
    # reliable before binarisation strips colour information)
    pil_img = _fix_rotation(pil_img)

    if not CV2_AVAILABLE:
        return pil_img

    img = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2GRAY)

    # Scale up if small (Tesseract prefers ≥ 300 DPI equivalent)
    h, w = img.shape
    if max(h, w) < 1500:
        sc  = 1500 / max(h, w)
        img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)

    img = cv2.fastNlMeansDenoising(img, h=12)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(img)


# ---------------------------------------------------------------------------
# Expense helpers / parsers
# ---------------------------------------------------------------------------

def _find_amount(text):
    """
    Extract the most likely total amount from receipt text.

    Strategy:
    - Collect candidates from lines matching total-related keywords, grouped
      by priority: HIGH (grand total / amount due) > MED (total) > LOW (subtotal).
    - Skip lines that are clearly payment lines (cash tendered, change, tip).
    - Within each priority group take the LAST match (lowest on the receipt =
      most likely to be the grand total, not an intermediate subtotal).
    - Fall back to the bottom third, then any amount.
    """
    lines = text.splitlines()
    n     = len(lines)

    def _amounts_in(line):
        """Parse dollar amounts, including OCR artefacts like '27 .05'."""
        out = []
        for m in re.finditer(r'\$?\s*([\d,]+\.\d{2})', line):
            try:
                v = float(m.group(1).replace(',', ''))
                if 0.01 <= v <= 99999:
                    out.append(v)
            except ValueError:
                pass
        for m in re.finditer(r'\$?\s*([\d,]+)\s*\.\s*(\d{2})\b', line):
            try:
                v = float(m.group(1).replace(',', '') + '.' + m.group(2))
                if 0.01 <= v <= 99999:
                    out.append(v)
            except ValueError:
                pass
        return list(dict.fromkeys(out))

    def _amount_for_line(i):
        """Return the amount on line i, or line i+1 if line i has none."""
        a = _amounts_in(lines[i])
        if a:
            return max(a)
        if i + 1 < n:
            a = _amounts_in(lines[i + 1])
            if a:
                return max(a)
        return None

    # Keywords that indicate a payment LINE, not the expense total
    SKIP_KWS = ['change due', 'change', 'cash tendered', 'cash paid',
                'tip', 'gratuity', 'you saved', 'savings', 'discount',
                'coupon', 'points redeemed']

    # HIGH: definitively the grand total
    HIGH_KWS = ['grand total', 'total amount', 'amount due', 'balance due',
                'your total', 'order total', 'total charges', 'you paid',
                'total due']
    # MED: plain "total" (may be a subtotal on some receipts, so prefer HIGH)
    MED_KWS  = ['total']
    # LOW: subtotals / sale amounts
    LOW_KWS  = ['subtotal', 'sub total', 'amount charged', 'sale amount', 'sale']

    high_vals, med_vals, low_vals = [], [], []

    for i, line in enumerate(lines):
        ll = line.lower()
        # Skip payment / change lines
        if any(kw in ll for kw in SKIP_KWS):
            continue
        val = _amount_for_line(i)
        if val is None:
            continue
        if any(kw in ll for kw in HIGH_KWS):
            high_vals.append(val)
        elif any(kw in ll for kw in MED_KWS):
            med_vals.append(val)
        elif any(kw in ll for kw in LOW_KWS):
            low_vals.append(val)

    # Last entry = closest to bottom = grand total
    if high_vals:
        return high_vals[-1]
    if med_vals:
        return med_vals[-1]
    if low_vals:
        return low_vals[-1]

    # Fall back: bottom third of receipt
    bottom = [v for ln in lines[n * 2 // 3:] for v in _amounts_in(ln)]
    if bottom:
        return max(bottom)

    # Last resort: any two-decimal amount
    all_vals = [v for ln in lines for v in _amounts_in(ln)]
    return max(all_vals) if all_vals else 0.0


def _normalize_date(date_str):
    """Convert various date format strings to YYYY-MM-DD, or return None."""
    # Strip trailing time portions before parsing (e.g. "3 Feb'26 7:58 AM")
    date_str = re.sub(r'\s+\d{1,2}:\d{2}(\s*(AM|PM))?$', '', date_str.strip(),
                      flags=re.IGNORECASE)
    date_str = date_str.strip()
    formats = [
        '%Y-%m-%d', '%m/%d/%Y', '%m/%d/%y',
        '%B %d, %Y', '%b %d, %Y', '%B %d %Y', '%b %d %Y',
        '%d %B %Y', '%d %b %Y', '%Y/%m/%d', '%m-%d-%Y', '%m-%d-%y',
        # Dot-separated: US "02.18.2026" and short "02.18.26"
        '%m.%d.%Y', '%m.%d.%y',
        # Receipt printer formats: "3 Feb'26", "12 Feb'26"
        "%d %b'%y", "%d %B'%y",
        # Day-first variants common in hotel folios: "18-Feb-2026"
        '%d-%b-%Y', '%d-%B-%Y',
        # Written-out month-first without comma: "February 18 2026"
        '%B %d %Y', '%b %d %Y',
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
        except ValueError:
            pass
    return None


def _is_plausible_receipt_date(date_str):
    """True if date is within the last 4 years and not in the future (>1 yr)."""
    try:
        d = datetime.strptime(date_str, '%Y-%m-%d')
        now = datetime.now()
        return (now - timedelta(days=4 * 365)) <= d <= (now + timedelta(days=365))
    except Exception:
        return False


def _find_date(text):
    """Find the most plausible transaction date, return YYYY-MM-DD or today.

    Collects ALL date-like strings in the text, normalises each, then returns
    the first one that falls in a plausible range (last 4 years).  This avoids
    picking up card-expiry years, future booking dates typed in error, etc.
    """
    patterns = [
        # ISO: 2026-02-18
        r'\b(\d{4}-\d{2}-\d{2})\b',
        # US slash: 02/18/2026 or 02/18/26
        r'\b(\d{1,2}/\d{1,2}/\d{4})\b',
        r'\b(\d{1,2}/\d{1,2}/\d{2})\b',
        # US dash: 02-18-2026 or 02-18-26
        r'\b(\d{1,2}-\d{1,2}-\d{4})\b',
        r'\b(\d{1,2}-\d{1,2}-\d{2})\b',
        # US dot: 02.18.2026 or 02.18.26
        r'\b(\d{1,2}\.\d{1,2}\.\d{4})\b',
        r'\b(\d{1,2}\.\d{1,2}\.\d{2})\b',
        # "Feb 18, 2026" or "February 18 2026"
        r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b',
        # "18 Feb 2026" or "18-Feb-2026"
        r'\b(\d{1,2}[\s\-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?[\s\-]\d{4})\b',
        # Compact receipt: "3 Feb'26" or "12 Feb'26 7:58 AM"
        r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'?\d{2}(?:\s+\d{1,2}:\d{2}(?:\s*[AP]M)?)?)\b",
        # Compact YYYYMMDD (e.g. some parking kiosk tickets)
        r'\b(2\d{7})\b',
    ]
    candidates = []
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            raw = m.group(1)
            # Handle compact YYYYMMDD
            if re.match(r'^2\d{7}$', raw):
                try:
                    normalized = datetime.strptime(raw, '%Y%m%d').strftime('%Y-%m-%d')
                except ValueError:
                    continue
            else:
                normalized = _normalize_date(raw)
            if normalized:
                candidates.append(normalized)

    if not candidates:
        return datetime.now().strftime('%Y-%m-%d')

    # Prefer plausible dates (within the last 4 years); fall back to first found
    plausible = [d for d in candidates if _is_plausible_receipt_date(d)]
    return plausible[0] if plausible else candidates[0]


def _find_vendor(text):
    """Return the most likely vendor name using a scored-candidate approach.

    Filters obvious noise (POS codes, addresses, prices, URLs, phone numbers,
    city/state lines) from the first 40 lines, then scores remaining candidates
    by capitalisation, word count, brand-map membership, and position.
    """
    # POS metadata lines (keywords that appear at the start of noise lines)
    _SKIP_STARTS = re.compile(
        r'^(?:'
        r'date[\s:]|time[\s:]|receipt|invoice|tax\b|total|subtotal|amount|'
        r'thank|welcome|sorry|please|call us|visit us|follow us|'
        r'order\s*[:#]|station\s*[:#]|server\s*[:#]|table\s*[:#]|'
        r'check\s*[:#]|ticket\s*[:#]|seat\s*[:#]|'
        r'transaction|reference\s*[:#]|auth|approval|entry\s*method|'
        r'terminal\s*(id)?[:#]?|merchant\s*(id)?[:#]?|mid[:\s]|tid[:\s]|'
        r'paid\s*with|visa|mastercard|amex|american\s*express|discover|'
        r'debit|credit card|card\s*type|card\s*number|acct\b|'
        r'your\s*server|cashier|clerk|operator|host\b|'
        r'balance|change\s*due|change\b|tip\b|gratuity|'
        r'suite\s*\d|floor\s*\d|p\.?o\.?\s*box|'
        r'to\s*go|for\s*here|dine\s*in|take\s*out|carry\s*out|pick.?up|'
        r'member|loyalty|rewards|points\b|savings|coupon|promo|'
        r'\*+|={3,}|-{4,}|_{4,}'
        r')',
        re.IGNORECASE,
    )
    # "Code# XXXX" — catches "Order# 7069", "Station# POSZ", "otation# POSZ"
    _POS_CODE    = re.compile(r'\w+\s*#\s*\w', re.IGNORECASE)
    # Address: starts with house number + street type
    _ADDR        = re.compile(
        r'^\d+\s+\w[\w\s]*\s+(?:st|ave|blvd|dr|rd|ln|way|pkwy|ste|hwy|'
        r'route|court|ct|pl|place|terr|ter|cir|loop|trail|trl)\b',
        re.IGNORECASE,
    )
    # Pure numbers / symbols
    _NUMERIC     = re.compile(r'^[\d\s\-\+\.\,\$\#\*\/\\()_=]+$')
    # Price on the line → item or total row, not a name
    _PRICE       = re.compile(r'\$\s*\d+\.\d{2}|\d+\.\d{2}\s*$')
    # "City, ST 12345" or "City ST 12345"
    _CITY_STATE  = re.compile(r'^[A-Za-z\s\.]+,?\s+[A-Z]{2}\s+\d{5}')
    # Phone number
    _PHONE       = re.compile(r'^\+?1?\s*\(?\d{3}\)?[\s\-\.]\d{3}[\s\-\.]\d{4}')
    # URL or e-mail
    _URL         = re.compile(r'(?:www\.|https?://|\.(com|net|org|io|co|us)\b)',
                              re.IGNORECASE)
    _EMAIL       = re.compile(r'\S+@\S+\.\S+')
    # Item line: description  TAB/many-spaces  amount (right-aligned layout)
    _ITEM_LINE   = re.compile(r'^.{4,}\s{3,}\d[\d,]*\.\d{2}\s*$')
    # Separator / decoration lines
    _SEPARATOR   = re.compile(r'^[\-=\*\.~_]{3,}$')

    candidates = []
    for line in text.splitlines()[:45]:
        line = line.strip()
        if len(line) < 3 or len(line) > 80:
            continue
        if _SEPARATOR.match(line):
            continue
        if _NUMERIC.match(line):
            continue
        if _SKIP_STARTS.match(line):
            continue
        pos_hit = _POS_CODE.search(line)
        if pos_hit:
            # Extract whatever precedes the '#' (e.g. "STARBUCKS" from "STARBUCKS #12345")
            hash_pos = line.find('#', pos_hit.start())
            before = line[:hash_pos].strip() if hash_pos >= 0 else ''
            # Only keep it if it isn't itself a generic POS keyword like "Order"
            _POS_KW = re.compile(
                r'^(?:order|station|server|table|check|ticket|seat|ref|'
                r'trans|auth|terminal|item|sku|upc|acct|account|lot)\s*$',
                re.IGNORECASE,
            )
            if (len(before) >= 3 and re.search(r'[A-Za-z]{2}', before)
                    and not _POS_KW.match(before)):
                candidates.append(before)
            continue
        if _ADDR.match(line):
            continue
        if _PRICE.search(line):
            continue
        if _CITY_STATE.match(line):
            continue
        if _PHONE.match(line):
            continue
        if _URL.search(line):
            continue
        if _EMAIL.search(line):
            continue
        if _ITEM_LINE.match(line):
            continue
        if not re.search(r'[A-Za-z]{2}', line):
            continue
        candidates.append(line)

    if not candidates:
        return 'Unknown Vendor'

    # Score each candidate; higher score = more likely to be the business name
    def _score(line, pos):
        s = 0
        words = line.split()
        alpha_words = [w for w in words if w and w[0].isalpha()]

        # Known brand is the strongest signal
        if _find_vendor_category(line):
            s += 25

        # All words title-case or ALL-CAPS → looks like a header / business name
        if alpha_words and all(w[0].isupper() for w in alpha_words):
            s += 4

        # Multi-word names are more credible than single tokens
        n_alpha = len(alpha_words)
        if n_alpha >= 2:
            s += 3
        if n_alpha >= 3:
            s += 1

        # Digits mixed into the line → likely a code, item, or address
        if re.search(r'\d', line):
            s -= 3

        # Very long lines are more likely descriptions than business names
        if len(line) > 45:
            s -= 2

        # Proximity to the top of the receipt (business names come first)
        s += max(0, 5 - pos)

        # Common prepositions/articles suggest a phrase, not a proper name
        if re.search(r'\b(?:for|the|and|our|your|we|you|at|in|on|of|a |is |are )\b',
                     line, re.IGNORECASE):
            s -= 2

        # All-lowercase line → usually body text
        if line == line.lower() and len(line) > 8:
            s -= 3

        return s

    scored = sorted(
        [(_score(c, i), i, c) for i, c in enumerate(candidates)],
        key=lambda x: (-x[0], x[1]),
    )
    return scored[0][2][:80]


# ---------------------------------------------------------------------------
# Brand → category lookup table
# ---------------------------------------------------------------------------
_BRAND_MAP = {
    'travel': [
        # Rideshare / taxi
        'uber', 'lyft', 'taxi', 'grab', 'curb', 'gett', 'waymo', 'via ride',
        # US airlines
        'american airlines', 'delta airlines', 'delta air', 'united airlines',
        'southwest airlines', 'jetblue', 'alaska airlines', 'spirit airlines',
        'frontier airlines', 'allegiant', 'sun country', 'breeze airways',
        'hawaiian airlines', 'avelo airlines',
        # International airlines
        'air canada', 'westjet', 'british airways', 'lufthansa', 'emirates',
        'qatar airways', 'air france', 'klm', 'ryanair', 'easyjet',
        'singapore airlines', 'cathay pacific', 'ana ', 'japan airlines',
        # Ground transport
        'amtrak', 'via rail', 'greyhound', 'megabus', 'flixbus',
        'metro', 'metrolink', 'metro rail', 'metro bus', 'big blue bus',
        'dash bus', 'culver citybus', 'torrance transit', 'foothill transit',
        # Car rental
        'enterprise', 'hertz', 'avis', 'budget rent', 'national car',
        'alamo', 'dollar rent', 'thrifty', 'sixt', 'zipcar', 'turo',
        'fox rent', 'payless car',
        # Parking
        'parkwhiz', 'spothero', 'parking', 'ez pass', 'fastrak',
        'lax parking', 'lawa parking', 'los angeles world airports',
        'sp+ parking', 'sp plus', 'ggp parking', 'park24', 'parkway',
        'parkmobile', 'passport parking', 'ace parking', 'central parking',
        'imperial parking', 'propark', 'republic parking',
        'airport parking', 'economy parking', 'valet parking',
        'parking garage', 'parking structure',
        # Fuel / road
        'mileage', 'toll', 'fuel', 'gasoline', 'shell', 'bp ', 'chevron',
        'exxon', 'mobil', 'sunoco', 'wawa', 'arco', '76 gas', 'circle k',
        'pilot travel', 'love\'s travel', 'ta travel', 'speedway',
    ],
    'hotel': [
        # Major chains
        'marriott', 'hilton', 'hyatt', 'ihg', 'wyndham', 'accor',
        'best western', 'holiday inn', 'sheraton', 'westin', 'w hotel',
        'courtyard', 'hampton inn', 'doubletree', 'embassy suites',
        'aloft', 'element hotel', 'le meridien', 'st. regis', 'st regis',
        'four seasons', 'ritz-carlton', 'ritz carlton', 'waldorf astoria',
        'intercontinental', 'crowne plaza', 'kimpton', 'hotel indigo',
        'radisson', 'ramada', 'days inn', 'super 8', 'motel 6',
        'red roof', 'la quinta', 'comfort inn', 'quality inn',
        'sleep inn', 'extended stay', 'residence inn', 'homewood suites',
        # Boutique / short-term
        'airbnb', 'vrbo', 'homeaway', 'sonder', 'vacasa',
        # LA-area hotels
        'loews hollywood', 'the standard', 'ace hotel', 'line hotel',
        'nomad hotel', 'chateau marmont', 'beverly hills hotel',
        'beverly wilshire', 'hotel bel-air', 'sunset tower', 'mama shelter',
        'dream hollywood', 'mondrian', 'soho house',
    ],
    'meal': [
        # Burgers / fast food
        "mcdonald's", 'mcdonalds', 'burger king', "wendy's", 'wendys',
        'five guys', 'shake shack', 'in-n-out', 'in n out', 'whataburger',
        'sonic drive', 'hardees', "carl's jr", 'carls jr', 'jack in the box',
        'smashburger', 'fatburger', 'habit burger', 'the habit',
        # Chicken
        'kfc', 'chick-fil-a', 'chick fil a', 'popeyes', 'raising canes',
        "cane's", 'wingstop', 'buffalo wild wings', "zaxby's", 'el pollo loco',
        # Mexican
        'chipotle', 'qdoba', "moe's", 'taco bell', 'del taco', 'taco bueno',
        'baja fresh', 'green burrito', 'chronic tacos',
        # Sandwiches / subs
        'subway', "jimmy john's", "jersey mike's", 'firehouse subs',
        'potbelly', "jason's deli", 'togos',
        # Pizza
        'pizza hut', "domino's", "papa john's", 'little caesars', 'sbarro',
        "round table pizza", 'blaze pizza', 'mod pizza',
        # Asian
        'panda express', 'pei wei',
        'sushi', 'hissho', 'hissho sushi', 'ramen', 'hibachi', 'teriyaki',
        'benihana', 'nobu', 'kura', 'sakura', 'kome', 'waba grill',
        'yoshinoya', 'l&l hawaiian', 'flame broiler',
        # Coffee / bakery
        'starbucks', 'dunkin', 'dunkin donuts', 'tim hortons',
        'dutch bros', 'caribou coffee', "peet's coffee", "peets coffee",
        'panera bread', 'panera', 'einstein bagels', "bruegger's",
        'coffee bean', 'urth caffe', 'alfred coffee', 'blue bottle',
        'philz coffee', 'groundwork coffee', 'verve coffee',
        # Casual dining
        'olive garden', "applebee's", "chili's", 'outback steakhouse',
        'longhorn steakhouse', 'red lobster', 'red robin',
        'cheesecake factory', 'ihop', "denny's", 'cracker barrel',
        'waffle house', 'first watch', 'corner bakery', 'bj\'s restaurant',
        # Delivery platforms
        'doordash', 'grubhub', 'uber eats', 'ubereats', 'seamless',
        'postmates', 'instacart', 'caviar',
        # Generic restaurant descriptors (used in brand-map fulltext scan)
        'restaurant', 'ristorante', 'brasserie', 'bistro', 'tavern',
        'cafe', 'diner', 'eatery', 'kitchen', 'grill', 'steakhouse',
        'chophouse', 'trattoria', 'cantina', 'taqueria', 'pizzeria',
        'smokehouse', 'brew pub', 'gastropub', 'oyster bar',
        # Common LA / airport spots
        'courtesy bistro', 'courtesy', 'urth', 'sugarfish', 'jon & vinny',
        'bottega louie', 'perino\'s', 'ink.sack', 'lax connector',
    ],
    'office': [
        'staples', 'office depot', 'officemax', 'office max',
        'best buy', 'micro center', "fry's electronics", 'b&h photo', 'adorama',
        'newegg', 'amazon business',
        'fedex', 'fedex office', 'ups store', 'the ups store', 'usps', 'dhl',
        'microsoft', 'adobe', 'google workspace', 'dropbox', 'zoom',
        'slack', 'notion', 'atlassian', 'github', 'aws ', 'amazon web',
        'apple store', 'dell', 'hp inc', 'lenovo',
        'staples print', 'office print', 'kinko',
    ],
}


def _find_vendor_category(vendor_name):
    """Return the category for a known brand name, or None if unknown."""
    name = vendor_name.lower()
    for category, brands in _BRAND_MAP.items():
        for brand in brands:
            if brand in name or name in brand:
                return category
    return None


def _find_category(text):
    """Categorize an expense from full receipt/email text using the brand map."""
    tl = text.lower()
    for category, brands in _BRAND_MAP.items():
        for brand in brands:
            # Require at least 6 characters to avoid short substring false
            # positives (e.g. "accor" matching "according to card issuer")
            if len(brand) >= 6 and brand in tl:
                return category
    # Generic keyword fallback (words that appear in the receipt body)
    generic = {
        'travel':  [
            'flight', 'airline', 'airport', 'boarding pass', 'boarding',
            'departure', 'arrival', 'itinerary', 'gate ', 'terminal ',
            'parking', 'valet', 'shuttle', 'rideshare', 'ride share',
            'mileage', 'toll', 'fuel', 'gasoline', 'refuel',
            'car rental', 'vehicle rental', 'rental car',
        ],
        'hotel':   [
            'hotel', 'inn', 'motel', 'resort', 'lodging', 'check-in',
            'check in', 'check out', 'checkout', 'room charge', 'room rate',
            'nightly rate', 'room night', 'accommodation', 'folio',
            'guest room', 'suite ', 'concierge',
        ],
        'meal':    [
            'restaurant', 'cafe', 'coffee', 'food', 'lunch', 'dinner',
            'breakfast', 'brunch', 'beverage', 'cocktail', 'appetizer',
            'entree', 'dessert', 'takeout', 'take out', 'delivery',
            'dine in', 'dine-in', 'to go', 'carry out',
        ],
        'office':  [
            'office', 'printing', 'shipping', 'postage', 'subscription',
            'software', 'license', 'storage', 'cloud', 'domain',
            'supplies', 'stationery', 'toner', 'cartridge',
        ],
    }
    for cat, words in generic.items():
        if any(re.search(r'\b' + re.escape(w.rstrip()) + r'\b', tl) for w in words):
            return cat
    return 'other'


def _parse_receipt_text(text):
    """Parse OCR text from a paper receipt."""
    vendor   = _find_vendor(text)
    category = _find_vendor_category(vendor) or _find_category(text)

    # If _find_vendor couldn't identify a named business, scan the full text
    # for the first brand-map brand that appears, and use that as the vendor.
    # This handles receipts where the header is in a large/stylized font that
    # Tesseract reads poorly but the business name appears in the body (e.g.
    # on the credit-card slip: "www.hisshosushi.com", "Hissho Sushi 7000 NE...").
    if vendor in ('Unknown Vendor',) or not _find_vendor_category(vendor):
        tl = text.lower()
        for cat, brands in _BRAND_MAP.items():
            for brand in brands:
                # Use brands of ≥ 6 characters or multi-word brands as vendor
                # names.  Very short brands (bar, cafe, etc.) are still excluded.
                qualifies = len(brand) >= 6 or ' ' in brand
                if qualifies and brand in tl:
                    # Capitalise properly and prefer this over the guessed vendor
                    vendor   = brand.title()
                    category = cat
                    break
            else:
                continue
            break

    return {
        'vendor':       vendor,
        'amount':       _find_amount(text),
        'expense_date': _find_date(text),
        'category':     category,
    }


def _parse_flight_email(text):
    """Parse a flight booking confirmation email."""
    result = {'category': 'travel'}
    airline_m = re.search(
        r'\b(American|Delta|United|Southwest|JetBlue|Alaska|Spirit|Frontier|'
        r'Air Canada|British Airways|Lufthansa|Emirates|Ryanair)\b',
        text, re.IGNORECASE,
    )
    result['vendor'] = (airline_m.group(1) + ' Airlines') if airline_m else _find_vendor(text)

    date_pat = r'([A-Za-z]+\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})'
    depart_m = re.search(r'(?:depart(?:ure)?|outbound|flight date)[:\s]+' + date_pat, text, re.IGNORECASE)
    result['expense_date'] = (_normalize_date(depart_m.group(1)) or _find_date(text)) if depart_m else _find_date(text)

    return_m = re.search(r'(?:return|inbound|return date)[:\s]+' + date_pat, text, re.IGNORECASE)
    if return_m:
        result['end_date'] = _normalize_date(return_m.group(1))

    total_m = re.search(r'(?:total|amount charged|fare)[:\s]*\$?\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
    if total_m:
        try:
            result['amount'] = float(total_m.group(1).replace(',', ''))
        except ValueError:
            result['amount'] = _find_amount(text)
    else:
        result['amount'] = _find_amount(text)

    result['description'] = 'Flight booking'
    return result


def _parse_hotel_email(text):
    """Parse a hotel booking confirmation email."""
    result = {'category': 'hotel'}
    hotel_m = re.search(r'(?:hotel|inn|resort|property)[:\s]+([^\n]+)', text, re.IGNORECASE)
    result['vendor'] = hotel_m.group(1).strip()[:80] if hotel_m else _find_vendor(text)

    date_pat = r'([A-Za-z]+\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})'
    checkin_m = re.search(r'(?:check[\s-]?in|arrival)[:\s]+' + date_pat, text, re.IGNORECASE)
    result['expense_date'] = (_normalize_date(checkin_m.group(1)) or _find_date(text)) if checkin_m else _find_date(text)

    checkout_m = re.search(r'(?:check[\s-]?out|departure)[:\s]+' + date_pat, text, re.IGNORECASE)
    if checkout_m:
        result['end_date'] = _normalize_date(checkout_m.group(1))

    total_m = re.search(r'(?:total|amount|charges?)[:\s]*\$?\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
    if total_m:
        try:
            result['amount'] = float(total_m.group(1).replace(',', ''))
        except ValueError:
            result['amount'] = _find_amount(text)
    else:
        result['amount'] = _find_amount(text)

    result['description'] = 'Hotel stay'
    return result


def _parse_email_text(text):
    """Dispatch to flight/hotel parser or fall back to generic; adds confidence key."""
    text_lower = text.lower()
    is_flight = any(w in text_lower for w in ['flight', 'airline', 'departure', 'boarding', 'itinerary', 'ticket number'])
    is_hotel = any(w in text_lower for w in ['hotel', 'check-in', 'checkout', 'check out', 'reservation', 'lodging'])

    if is_flight and not is_hotel:
        result = _parse_flight_email(text)
        result['confidence'] = 'flight'
    elif is_hotel and not is_flight:
        result = _parse_hotel_email(text)
        result['confidence'] = 'hotel'
    elif is_flight and is_hotel:
        result = _parse_flight_email(text)
        result['confidence'] = 'mixed'
    else:
        result = _parse_receipt_text(text)
        result['confidence'] = 'generic'
    return result


# ---------------------------------------------------------------------------
# Expense API routes
# ---------------------------------------------------------------------------

@app.route('/api/expenses', methods=['GET'])
def get_expenses():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('''
        SELECT e.*, c.name as client_name
        FROM expenses e
        LEFT JOIN clients c ON e.client_id = c.id
        ORDER BY e.id DESC
    ''')
    rows = [dict(x) for x in cur.fetchall()]
    conn.close()
    return jsonify(rows)


@app.route('/api/expenses', methods=['POST'])
def add_expense():
    data = request.json or {}
    vendor = data.get('vendor', '').strip()
    amount = data.get('amount')
    expense_date = data.get('expense_date', '').strip()
    if not vendor or amount is None or not expense_date:
        return jsonify({'error': 'vendor, amount, expense_date required'}), 400
    client_id = data.get('client_id') or None
    project = data.get('project', '')
    category = data.get('category', 'other')
    description = data.get('description', '')
    reimbursable = 1 if data.get('reimbursable', True) else 0
    source = data.get('source', 'manual')
    receipt_path = data.get('receipt_path', '')
    created_at = datetime.utcnow().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO expenses
            (client_id, project, vendor, amount, expense_date, category,
             description, reimbursable, source, receipt_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (client_id, project, vendor, float(amount), expense_date, category,
          description, reimbursable, source, receipt_path, created_at))
    conn.commit()
    eid = cur.lastrowid
    conn.close()
    git_sync(f"Added expense: {vendor} ${amount}")
    return jsonify({'id': eid})


@app.route('/api/expenses/<int:expense_id>', methods=['PUT'])
def update_expense(expense_id):
    data = request.json or {}
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT * FROM expenses WHERE id = ?', (expense_id,))
    if not cur.fetchone():
        conn.close()
        return jsonify({'error': 'Expense not found'}), 404
    allowed = ['client_id', 'project', 'vendor', 'amount', 'expense_date',
               'category', 'description', 'reimbursable', 'source']
    updates, params = [], []
    for field in allowed:
        if field in data:
            updates.append(f'{field} = ?')
            params.append(data[field])
    if not updates:
        conn.close()
        return jsonify({'error': 'No fields to update'}), 400
    params.append(expense_id)
    cur.execute(f'UPDATE expenses SET {", ".join(updates)} WHERE id = ?', params)
    conn.commit()
    conn.close()
    git_sync(f"Updated expense: {expense_id}")
    return jsonify({'updated': expense_id})


@app.route('/api/expenses/<int:expense_id>', methods=['DELETE'])
def delete_expense(expense_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('DELETE FROM expenses WHERE id = ?', (expense_id,))
    conn.commit()
    conn.close()
    git_sync(f"Deleted expense: {expense_id}")
    return jsonify({'deleted': expense_id})


@app.route('/api/expenses/export')
def export_expenses_csv():
    client_id = request.args.get('client_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    conn = get_conn()
    cur = conn.cursor()
    q = '''SELECT e.id, e.expense_date, e.vendor, e.amount, e.category,
                  c.name as client_name, e.project, e.description,
                  e.reimbursable, e.source
           FROM expenses e
           LEFT JOIN clients c ON e.client_id = c.id
           WHERE 1=1'''
    params = []
    if client_id:
        q += ' AND e.client_id = ?'
        params.append(int(client_id))
    if start_date:
        q += ' AND e.expense_date >= ?'
        params.append(start_date)
    if end_date:
        q += ' AND e.expense_date <= ?'
        params.append(end_date)
    q += ' ORDER BY e.expense_date, e.id'
    cur.execute(q, params)
    rows = cur.fetchall()
    conn.close()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['id', 'date', 'vendor', 'amount', 'category',
                     'client', 'project', 'description', 'reimbursable', 'source'])
    for r in rows:
        writer.writerow([r['id'], r['expense_date'], r['vendor'], r['amount'],
                         r['category'], r['client_name'] or '', r['project'] or '',
                         r['description'] or '', 'yes' if r['reimbursable'] else 'no',
                         r['source'] or ''])
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode('utf-8')),
                     mimetype='text/csv', as_attachment=True,
                     download_name='expenses.csv')


@app.route('/api/expenses/report')
def generate_expense_report():
    client_id = request.args.get('client_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    conn = get_conn()
    cur = conn.cursor()

    client = None
    if client_id:
        cur.execute('SELECT * FROM clients WHERE id = ?', (int(client_id),))
        client = cur.fetchone()

    q = '''SELECT e.*, c.name as client_name
           FROM expenses e
           LEFT JOIN clients c ON e.client_id = c.id
           WHERE 1=1'''
    params = []
    if client_id:
        q += ' AND e.client_id = ?'
        params.append(int(client_id))
    if start_date:
        q += ' AND e.expense_date >= ?'
        params.append(start_date)
    if end_date:
        q += ' AND e.expense_date <= ?'
        params.append(end_date)
    q += ' ORDER BY e.expense_date, e.id'
    cur.execute(q, params)
    expenses = cur.fetchall()
    conn.close()

    if not expenses:
        return jsonify({'error': 'No expenses found for the selected filters'}), 404

    total = round(sum(e['amount'] for e in expenses), 2)
    report_date = datetime.now().date()

    if start_date and end_date:
        date_label = f"{start_date} to {end_date}"
    elif start_date:
        date_label = f"From {start_date}"
    elif end_date:
        date_label = f"Through {end_date}"
    else:
        dates = [e['expense_date'] for e in expenses]
        date_label = f"{min(dates)} to {max(dates)}"

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ERRightAlign', alignment=TA_RIGHT))
    styles.add(ParagraphStyle(name='ERLeftAlign', alignment=TA_LEFT))
    story = []

    # Header: business left, client right (if client selected)
    right_top = f"Prepared For: {client['name']}" if client else ''
    right_addr = (client['address'] or '') if client else ''
    right_contact = (f"{client['email'] or ''} | {client['phone'] or ''}".strip(' | ')) if client else ''
    hdata = [
        [Paragraph(config.MY_NAME, styles['ERLeftAlign']),
         Paragraph(right_top, styles['ERRightAlign'])],
    ]
    if config.BUSINESS_NAME:
        hdata.append([Paragraph(config.BUSINESS_NAME, styles['ERLeftAlign']), ''])
    hdata.append([Paragraph(config.MY_ADDRESS, styles['ERLeftAlign']),
                  Paragraph(right_addr, styles['ERRightAlign'])])
    hdata.append([Paragraph(f"{config.MY_PHONE} | {config.MY_EMAIL}", styles['ERLeftAlign']),
                  Paragraph(right_contact, styles['ERRightAlign'])])
    htable = Table(hdata, colWidths=[300, 200])
    htable.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(htable)
    story.append(Spacer(1, 12))
    story.append(HLine(500))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Expense Report", styles['Title']))
    story.append(Spacer(1, 12))

    ddata = [['Report Date:', str(report_date), 'Period:', date_label]]
    dtable = Table(ddata, colWidths=[100, 100, 60, 240])
    dtable.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
    ]))
    story.append(dtable)
    story.append(Spacer(1, 12))
    story.append(HLine(500))
    story.append(Spacer(1, 12))

    # Expense table — include Client column only when not filtered by client
    show_client = not client_id
    if show_client:
        headers = ['Date', 'Vendor', 'Category', 'Client', 'Project', 'Description', 'Amount']
        col_widths = [55, 100, 60, 75, 60, 100, 50]
    else:
        headers = ['Date', 'Vendor', 'Category', 'Project', 'Description', 'Amount']
        col_widths = [60, 130, 70, 80, 110, 50]

    edata = [headers]
    for e in expenses:
        row = [e['expense_date'], e['vendor'], (e['category'] or '').capitalize()]
        if show_client:
            row.append(e['client_name'] or '')
        row += [e['project'] or '', e['description'] or '', f"${e['amount']:.2f}"]
        edata.append(row)

    total_label_offset = -2 if show_client else -2
    blank_cols = len(headers) - 2
    edata.append([''] * blank_cols + ['Total:', f"${total:.2f}"])

    etable = Table(edata, colWidths=col_widths)
    etable.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (-1, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
    ]))
    story.append(etable)
    story.append(Spacer(1, 12))

    # Category summary (only useful when more than one category)
    by_cat = defaultdict(float)
    for e in expenses:
        by_cat[(e['category'] or 'other').capitalize()] += e['amount']

    if len(by_cat) > 1:
        story.append(Paragraph("Summary by Category", styles['Heading3']))
        story.append(Spacer(1, 6))
        cdata = [['Category', 'Total']]
        for cat, amt in sorted(by_cat.items()):
            cdata.append([cat, f"${amt:.2f}"])
        cdata.append(['Grand Total', f"${total:.2f}"])
        ctable = Table(cdata, colWidths=[200, 100])
        ctable.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('LINEABOVE', (0, -1), (-1, -1), 1, colors.black),
        ]))
        story.append(ctable)

    doc.build(story)
    buf.seek(0)
    client_slug = f"_{client['name'].replace(' ', '_')}" if client else ''
    filename = f"expense_report{client_slug}_{report_date}.pdf"
    return send_file(buf, mimetype='application/pdf', as_attachment=True, download_name=filename)


@app.route('/api/expenses/parse-receipt', methods=['POST'])
def parse_receipt():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_RECEIPT_EXTENSIONS:
        allowed = ', '.join(sorted(ALLOWED_RECEIPT_EXTENSIONS))
        return jsonify({'error': f'Invalid file type: {ext}. Allowed: {allowed}'}), 400

    filename = f"{uuid.uuid4().hex}_{secure_filename(f.filename)}"
    receipt_path = UPLOADS_DIR / filename
    f.save(str(receipt_path))

    # --- Build the list of PIL Images to OCR ---
    # PDF: render each page at 300 DPI, then segment each page for multi-receipt layouts.
    # Image: run segmentation directly on the file.
    crops = []

    if ext == '.pdf':
        pages = _pdf_to_images(receipt_path)
        if not pages:
            return jsonify({
                'error': 'PDF rendering failed. '
                         'Run: sudo apt install poppler-utils'
            }), 500
        for page_img in pages:
            # Fix page rotation BEFORE segmentation so that when we split the
            # page into individual receipt crops each crop spans the full
            # receipt (header → footer) in the correct reading direction.
            page_img = _fix_rotation(page_img)
            # Save rotated page so _segment_receipts can read it via OpenCV
            tmp = UPLOADS_DIR / f"{uuid.uuid4().hex}_page.png"
            page_img.save(str(tmp))
            try:
                crops.extend(_segment_receipts(tmp))
            except Exception:
                crops.append(page_img)
    else:
        try:
            crops = _segment_receipts(receipt_path)
        except Exception as e:
            app.logger.warning(f'Segmentation failed, using full image: {e}')
            crops = [Image.open(str(receipt_path))]

    # --- OCR each crop ---
    safe_stem = secure_filename(Path(f.filename).stem)
    results = []
    for i, crop_img in enumerate(crops):
        try:
            enhanced = _preprocess_for_ocr(crop_img)
            raw_text = pytesseract.image_to_string(
                enhanced, config='--psm 4 --oem 1'
            )
        except Exception:
            raw_text = ''

        parsed = _parse_receipt_text(raw_text)

        if len(crops) > 1:
            crop_name = f"{uuid.uuid4().hex}_crop{i}_{safe_stem}.png"
            crop_img.save(str(UPLOADS_DIR / crop_name))
            rp = f'uploads/{crop_name}'
        else:
            rp = f'uploads/{filename}'

        results.append({
            'parsed':       parsed,
            'raw_text':     raw_text,
            'receipt_path': rp,
            'crop_index':   i,
        })

    return jsonify({'results': results, 'count': len(results)})


@app.route('/api/expenses/lookup-vendor')
def lookup_vendor():
    name = request.args.get('name', '').strip()
    if not name:
        return jsonify({'category': None})
    category = _find_vendor_category(name)
    return jsonify({'category': category})


@app.route('/qr.png')
def qr_png():
    """Generate a QR code using the URL the browser actually used to reach us."""
    url = request.host_url.rstrip('/')
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/api/expenses/parse-email', methods=['POST'])
def parse_email():
    data = request.json or {}
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'text required'}), 400
    parsed = _parse_email_text(text)
    return jsonify({'parsed': parsed})


@app.route('/generate_invoice/<int:client_id>')
def generate_invoice(client_id):
    draft = request.args.get('draft', '0') == '1'
    start_date = request.args.get('start_date')  # YYYY-MM-DD, optional
    end_date = request.args.get('end_date')        # YYYY-MM-DD, optional
    conn = get_conn()
    cur = conn.cursor()
    # Get client
    cur.execute('SELECT * FROM clients WHERE id = ?', (client_id,))
    client = cur.fetchone()
    if not client:
        conn.close()
        return jsonify({'error': 'Client not found'}), 404

    # Get time entries, optionally filtered by date range
    entry_q = 'SELECT * FROM entries WHERE client_id = ? AND duration_min IS NOT NULL'
    entry_p = [client_id]
    if start_date:
        entry_q += ' AND DATE(start_ts) >= ?'
        entry_p.append(start_date)
    if end_date:
        entry_q += ' AND DATE(start_ts) <= ?'
        entry_p.append(end_date)
    entry_q += ' ORDER BY start_ts'
    cur.execute(entry_q, entry_p)
    entries = cur.fetchall()

    # Get unbilled reimbursable expenses, optionally filtered by date range
    exp_q = 'SELECT * FROM expenses WHERE client_id = ? AND reimbursable = 1 AND invoice_id IS NULL'
    exp_p = [client_id]
    if start_date:
        exp_q += ' AND expense_date >= ?'
        exp_p.append(start_date)
    if end_date:
        exp_q += ' AND expense_date <= ?'
        exp_p.append(end_date)
    exp_q += ' ORDER BY expense_date'
    cur.execute(exp_q, exp_p)
    reimbursable_expenses = cur.fetchall()

    if not entries and not reimbursable_expenses:
        conn.close()
        return jsonify({'error': 'No time entries or reimbursable expenses for client'}), 404

    # Calculate time totals
    total_min = sum(e['duration_min'] for e in entries) if entries else 0
    total_hours = round(total_min / 60, 2)
    hourly_rate = client['hourly_rate'] or config.HOURLY_RATE
    total_amount = round(total_hours * hourly_rate, 2)

    # Expense totals
    expense_total = round(sum(e['amount'] for e in reimbursable_expenses), 2)
    grand_total = round(total_amount + expense_total, 2)

    # Date range
    if entries:
        start_dates = [datetime.fromisoformat(e['start_ts']).date() for e in entries]
        min_date = min(start_dates)
        max_date = max(start_dates)
        date_range = f"{min_date} to {max_date}"
    else:
        date_range = str(datetime.now().date())

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
    invoice_id = None
    if not draft:
        cur.execute('''
            INSERT INTO invoices
                (invoice_number, client_id, invoice_date, due_date, total_hours, total_amount, expense_total)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (invoice_number, client_id, invoice_date.isoformat(), due_date.isoformat(),
              total_hours, total_amount, expense_total))
        conn.commit()
        invoice_id = cur.lastrowid
        if reimbursable_expenses:
            exp_ids = [e['id'] for e in reimbursable_expenses]
            placeholders = ','.join('?' * len(exp_ids))
            cur.execute(
                f'UPDATE expenses SET invoice_id = ? WHERE id IN ({placeholders})',
                [invoice_id] + exp_ids,
            )
            conn.commit()
    conn.close()
    if not draft:
        git_sync(f"Generated invoice: {invoice_number}")

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

    # Time entries section (only if there are entries)
    if entries:
        service_desc = f"Consulting Services: {total_hours} hours @ ${hourly_rate}/hr ({date_range})"
        story.append(Paragraph(service_desc, styles['Normal']))
        story.append(Spacer(1, 12))

        time_label = 'Subtotal (Hours):' if reimbursable_expenses else 'Total:'
        tdata = [['Date', 'Description', 'Hours', 'Subtotal']]
        for e in entries:
            d = datetime.fromisoformat(e['start_ts']).date()
            desc = e['description'] or e['project']
            hrs = round(e['duration_min'] / 60, 2)
            subtotal = round(hrs * hourly_rate, 2)
            tdata.append([str(d), desc, f"{hrs}", f"${subtotal:.2f}"])
        tdata.append(['', time_label, str(total_hours), f"${total_amount:.2f}"])

        time_table = Table(tdata, colWidths=[80, 250, 60, 80])
        time_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ]))
        story.append(time_table)
        story.append(Spacer(1, 12))

    # Reimbursable expenses section
    if reimbursable_expenses:
        story.append(Paragraph("Reimbursable Expenses", styles['Heading3']))
        story.append(Spacer(1, 6))

        edata = [['Date', 'Vendor', 'Category', 'Description', 'Amount']]
        for exp in reimbursable_expenses:
            edata.append([
                exp['expense_date'],
                exp['vendor'],
                (exp['category'] or '').capitalize(),
                exp['description'] or '',
                f"${exp['amount']:.2f}",
            ])
        edata.append(['', '', '', 'Expense Total:', f"${expense_total:.2f}"])

        exp_table = Table(edata, colWidths=[70, 130, 70, 140, 70])
        exp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (4, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ]))
        story.append(exp_table)
        story.append(Spacer(1, 12))

    # Grand total (shown when both time and expenses are present)
    if entries and reimbursable_expenses:
        gt_data = [
            ['', 'Hours Total:', f"${total_amount:.2f}"],
            ['', 'Expense Total:', f"${expense_total:.2f}"],
            ['', 'Grand Total:', f"${grand_total:.2f}"],
        ]
        gt_table = Table(gt_data, colWidths=[320, 100, 80])
        gt_table.setStyle(TableStyle([
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (1, -1), (-1, -1), 'Helvetica-Bold'),
            ('LINEABOVE', (1, -1), (-1, -1), 1, colors.black),
            ('TOPPADDING', (0, -1), (-1, -1), 6),
        ]))
        story.append(gt_table)
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

@app.route('/sw.js')
def service_worker():
    return send_file(str(APP_ROOT / 'static' / 'sw.js'), mimetype='application/javascript')


if __name__ == '__main__':
    print("Performing initial git sync...")
    git_sync("Startup sync")
    app.run(debug=True, host='0.0.0.0')
