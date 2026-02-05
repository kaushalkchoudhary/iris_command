"""Report generation: IRIS Command Enhanced Report."""

import io
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta

from fpdf import FPDF

REPORTS_DIR = Path(__file__).parent / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ALERTS_DIR = Path(__file__).parent / "data" / "alerts"
LOGO_PATH = Path(__file__).parent / "data" / "iris_logo.svg"

# Color Palette (RGB)
COLOR_BG_DARK = (5, 10, 20)
COLOR_TEXT_PRIMARY = (255, 255, 255)
COLOR_TEXT_SECONDARY = (160, 174, 192)
COLOR_TEXT_DARK = (40, 40, 40)

# Mode Colors
MODE_COLORS = {
    "congestion": (239, 68, 68),  # Red
    "vehicle": (16, 185, 129),    # Emerald
    "flow": (168, 85, 247),       # Purple
    "crowd": (20, 184, 166),      # Teal
    "forensics": (6, 182, 212),   # Cyan
}

def hex_to_rgb(hex_str):
    return tuple(int(hex_str[i:i+2], 16) for i in (1, 3, 5))

class IRISReportEnhanced(FPDF):
    """Refined PDF with IRIS Command branding."""

    def __init__(self, mode="congestion"):
        super().__init__()
        self.mode = mode
        self.accent = MODE_COLORS.get(mode, (6, 182, 212))

    def circle(self, x, y, r, style='D'):
        self.ellipse(x - r, y - r, 2 * r, 2 * r, style)

    def draw_grid(self, x, y, w, h, step=5):
        """Draws a faint grid pattern."""
        # FPDF 1.7 doesn't support alpha easily without extension. 
        # We will use specific faint color instead.
        # We will use specific faint gray color instead.
        r, g, b = self.accent
        faint_color = (int(r*0.2 + 20), int(g*0.2 + 20), int(b*0.2 + 30)) # Very dark version
        self.set_draw_color(*faint_color)
        self.set_line_width(0.1)
        
        # Vertical lines
        for i in range(0, int(w), step):
            self.line(x + i, y, x + i, y + h)
        # Horizontal lines
        for i in range(0, int(h), step):
            self.line(x, y + i, x + w, y + i)



    def header(self):
        # Dark Header Background
        header_h = 28 # Thinner
        self.set_fill_color(*COLOR_BG_DARK)
        self.rect(0, 0, 210, header_h, 'F')
        
        # Grid Pattern
        self.draw_grid(0, 0, 210, header_h)

        # Draw Logo (Vector Approx)
        self.set_draw_color(*self.accent)
        self.set_line_width(0.4)
        center_x, center_y = 16, 14
        self.set_fill_color(*self.accent)
        self.circle(center_x, center_y, 1.2, 'F')
        self.set_fill_color(*COLOR_BG_DARK)
        self.circle(center_x, center_y, 4.5, 'D')
        self.circle(center_x, center_y, 7, 'D')
        self.line(center_x, center_y, center_x + 4, center_y - 4)

        # Branding Text
        branding = "IRIS COMMAND"
        if self.mode == "forensics":
            branding = "IRIS FORENSICS"

        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*self.accent)
        self.set_xy(28, 6.5)
        self.cell(0, 8, branding, ln=True)
        
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*COLOR_TEXT_SECONDARY)
        self.set_xy(28, 11.5)
        self.cell(0, 8, "INTEGRATED REALTIME INTELLIGENCE SYSTEM", ln=True)

        # Right side: Report Type
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(255, 255, 255)
        self.set_xy(-60, 6)
        self.cell(50, 6, f"{self.mode.upper()} ANALYSIS REPORT", align='R', ln=True)
        
        # Confidential marker
        self.set_font("Helvetica", "I", 6)
        self.set_text_color(*COLOR_TEXT_SECONDARY)
        self.set_xy(-60, 10)
        self.cell(50, 6, "CONFIDENTIAL / INTERNAL USE", align='R', ln=True)

        # Bottom Accent Line
        self.set_draw_color(*self.accent)
        self.set_line_width(0.5)
        self.line(0, header_h, 210, header_h)
        self.ln(15)

    def footer(self):
        footer_h = 12
        y_start = 297 - footer_h
        
        # Footer Background
        self.set_fill_color(*COLOR_BG_DARK)
        self.rect(0, y_start, 210, footer_h, 'F')
        
        self.draw_grid(0, y_start, 210, footer_h)

        self.set_y(-8)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.set_draw_color(*self.accent)
        # Top line
        self.line(0, y_start, 210, y_start)
        
        self.cell(0, 4, f"IRIS Command v1.0  |  Integrated Realtime Intelligence System  |  Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.ln(6)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(0, 0, 0)
        self.set_fill_color(245, 247, 250)
        self.cell(0, 8, f"  {title.upper()}", fill=True, ln=True)
        self.set_draw_color(*self.accent)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 60, self.get_y())
        self.ln(4)

def generate_report(
    source_name: str,
    screenshot_bytes: bytes | None,
    metrics_data: dict | None,
    alerts_list: list,
    active_mode: str | None,
    vlm_narrative: str | None = None,
) -> bytes:
    
    mode = active_mode or "congestion"
    pdf = IRISReportEnhanced(mode=mode)
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    now = datetime.now()
    m = metrics_data or {}

    # ── 1. Metadata Grid ──
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(60, 60, 60)
    
    col_w = 95
    x_start = 10
    y_start = pdf.get_y()
    pdf.set_xy(x_start, y_start)
    
    # Left Column (Standard)
    _meta_row(pdf, "Source ID", source_name, x_start, col_w)
    _meta_row(pdf, "Analysis Mode", mode.upper(), x_start, col_w, color=pdf.accent)
    _meta_row(pdf, "Generated At", now.strftime("%Y-%m-%d %H:%M:%S"), x_start, col_w)
    
    # Right Column (Mode Specific)
    x_start = 110
    pdf.set_xy(x_start, y_start)
    
    # Common: Duration
    start_time = m.get("start_time")
    duration_str = "N/A"
    if start_time:
        d = timedelta(seconds=int(time.time() - start_time))
        duration_str = str(d)
    _meta_row(pdf, "Video Duration", duration_str, x_start, col_w)
    
    # Mode-Specific Metrics
    if mode == "congestion":
        hot_regions = m.get("hot_regions", {})
        active_hot = hot_regions.get("active_count", 0)
        peak_con = max([a.get("congestion", 0) for a in alerts_list] + [m.get("congestion_index", 0)])
        _meta_row(pdf, "Peak Congestion", f"{peak_con}%", x_start, col_w, bold=True)
        _meta_row(pdf, "Hot Regions", str(active_hot), x_start, col_w)
        
    elif mode == "vehicle":
        total_veh = m.get("detection_count", 0)
        state_counts = m.get("state_counts", {})
        stopped = state_counts.get("stopped", 0)
        _meta_row(pdf, "Total Vehicles", str(total_veh), x_start, col_w, bold=True)
        _meta_row(pdf, "Stopped Vehicles", str(stopped), x_start, col_w)

    elif mode == "flow":
        mobility = m.get("mobility_index", 0)
        stalled = m.get("stalled_pct", 0)
        _meta_row(pdf, "Mobility Index", str(mobility), x_start, col_w, bold=True)
        _meta_row(pdf, "Stalled Traffic", f"{stalled}%", x_start, col_w)

    elif mode == "crowd":
        crowd_cnt = m.get("crowd_count", m.get("detection_count", 0))
        risk = m.get("risk_score", 0)
        _meta_row(pdf, "Crowd Count", str(crowd_cnt), x_start, col_w, bold=True)
        _meta_row(pdf, "Risk Score", str(risk), x_start, col_w)

    elif mode == "forensics":
        count = m.get("detection_count", 0)
        prompt = m.get("prompt", "N/A") # Not actually in metrics usually, but handled gracefully
        _meta_row(pdf, "Detections", str(count), x_start, col_w, bold=True)
        pdf.set_font("Helvetica", "I", 8)
        # Extra custom line for prompt
        # _meta_row can't easily handle long text, manual call
        pdf.set_x(x_start)
        pdf.cell(col_w/2, 6, "Search Query", border=0)
        pdf.cell(col_w/2, 6, prompt[:20] + "..." if len(prompt)>20 else prompt, ln=True)

    pdf.ln(10)
    
    # ── 2. Executive Summary ──
    pdf.section_title("Executive Summary")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    
    summary = f"During the {duration_str} analysis of '{source_name}', "
    
    if mode == "congestion":
        hot_cnt = m.get("hot_regions", {}).get("active_count", 0)
        summary += f"IRIS detected {hot_cnt} active congestion zones. "
        if hot_cnt > 0:
            summary += "Traffic flow is significantly impacted in key areas. "
        else:
            summary += "Traffic flow remained efficient. "
            
    elif mode == "vehicle":
        cnt = m.get("detection_count", 0)
        stopped = m.get("state_counts", {}).get("stopped", 0)
        summary += f"a total of {cnt} vehicles were classified. {stopped} vehicles were observed in a stopped state. "
        
    elif mode == "flow":
        mob = m.get("mobility_index", 0)
        summary += f"the average Mobility Index was {mob}. "
        if mob < 30:
            summary += "Traffic conditions indicate severe stagnation. "
        elif mob < 60:
            summary += "Traffic is moving with moderate delays. "
        else:
            summary += "Traffic is free-flowing. "
            
    elif mode == "crowd":
        cnt = m.get("crowd_count", m.get("detection_count", 0))
        risk = m.get("risk_score", 0)
        summary += f"approximately {cnt} individuals were monitored. The calculated Risk Score is {risk}/100. "
        if risk > 50:
            summary += "Crowd density has reached alert levels. "
            
    elif mode == "forensics":
        cnt = m.get("detection_count", 0)
        summary += f"the system detected {cnt} instances matching the search criteria. "

    if alerts_list:
        summary += f"Specifically, {len(alerts_list)} distinct incidents were logged for review."
    else:
        summary += "No critical incidents were flagged."
        
    pdf.multi_cell(0, 6, summary)
    pdf.ln(6)

    # ── 3. Live Snapshot ──
    if mode == "forensics":
        pdf.section_title("Part 1: Tactical Search Report")
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, f"Search Criteria: {m.get('prompt', 'Visual Segment')}", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Detections matching criteria: {m.get('detection_count', 0)}", ln=True)
        pdf.ln(2)
    else:
        pdf.section_title("Live Analysis Snapshot")

    if screenshot_bytes:
        _embed_image(pdf, screenshot_bytes, h=90)
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 10, "No live snapshot available.", ln=True)
    pdf.ln(4)

    # ── 3b. Strategic Feed Analysis (Forensics Only) ──
    if mode == "forensics" and vlm_narrative:
        pdf.add_page()
        pdf.section_title("Part 2: Strategic Feed Analysis")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        
        # VLM Narrative can be long, use multi_cell
        pdf.multi_cell(0, 6, vlm_narrative)
        pdf.ln(6)
        
        # If we have session history, embed snapshots of key events
        history = m.get("session_history", [])
        if history:
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 10, "TACTICAL TIMELINE & KEY EVENTS", ln=True)
            
            # Show up to 3 major snapshots
            snapshots_found = 0
            for entry in history:
                shot_path = entry.get("screenshot_path")
                if shot_path and Path(shot_path).exists():
                    t_str = time.strftime("%H:%M:%S", time.localtime(entry['timestamp']))
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.cell(0, 8, f"Event Snapshot @ {t_str}", ln=True)
                    
                    # Embed local file
                    try:
                        pdf.image(shot_path, x=15, h=50)
                        pdf.ln(2)
                    except:
                        pdf.cell(0, 5, "[Image Load Error]", ln=True)
                    
                    pdf.set_font("Helvetica", "I", 8)
                    pdf.set_x(15)
                    det_brief = f"Detections: {len(entry['detections'])} | Prompt: {entry['prompt']}"
                    pdf.cell(0, 5, det_brief, ln=True)
                    pdf.ln(5)
                    
                    snapshots_found += 1
                    if snapshots_found >= 3: break
                    
            if snapshots_found == 0:
                pdf.set_font("Helvetica", "I", 9)
                pdf.cell(0, 8, "No visual event snapshots available for this session.", ln=True)
            pdf.ln(4)

    # ── 4. Incident Log ──
    if alerts_list:
        pdf.add_page()
        pdf.section_title("Incident Log")
        
        display_alerts = alerts_list[-20:] if len(alerts_list) > 20 else alerts_list
        
        for i, alert in enumerate(reversed(display_alerts)):
            # Check space
            if pdf.get_y() + 95 > 280:
                pdf.add_page()
            
            _render_alert_card(pdf, alert, mode)
            pdf.ln(4)
    else:
        pdf.ln(4)
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, "No incidents recorded.", ln=True)

    out = pdf.output(dest='S')
    if isinstance(out, str):
        return out.encode("latin-1")
    return out

def _meta_row(pdf, label, value, x, w, color=(0,0,0), bold=False):
    pdf.set_x(x)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(w/2, 6, label, border=0)
    pdf.set_font("Helvetica", "B" if bold else "", 9)
    pdf.set_text_color(*color)
    pdf.cell(w/2, 6, str(value), border=0, ln=True)

def _embed_image(pdf, img_data, h=None):
    if not img_data: return
    if not isinstance(img_data, (bytes, bytearray)): return

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    try:
        tmp.write(img_data)
        tmp.flush()
        tmp.close()
        pdf.image(tmp.name, x=pdf.get_x(), h=h, w=pdf.w-20 if not h else 0)
    except Exception:
        pass
    finally:
        Path(tmp.name).unlink(missing_ok=True)

def _render_alert_card(pdf, alert, mode):
    start_y = pdf.get_y()
    
    # Severity Color
    sev = alert.get("severity", "medium").lower()
    if sev == "critical": color = (220, 50, 50)
    elif sev == "high": color = (220, 140, 20)
    else: color = (40, 160, 200)
    
    # Header
    pdf.set_fill_color(250, 250, 250)
    pdf.rect(10, start_y, 190, 85, 'F')
    pdf.set_fill_color(*color)
    pdf.rect(10, start_y, 2, 85, 'F')
    
    # Content
    pdf.set_xy(15, start_y + 3)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*color)
    pdf.cell(0, 6, f"{sev.upper()} ALERT", ln=True)
    
    # Primary Metric (Title)
    pdf.set_xy(15, start_y + 10)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 20, 20)
    
    m = alert.get("metrics", {})
    if mode == "congestion":
        pdf.cell(50, 6, f"{alert.get('congestion', 0)}% Congestion", ln=False)
    elif mode == "crowd":
        pdf.cell(50, 6, f"{m.get('crowd_count', 0)} People Detected", ln=False)
    elif mode == "flow":
        pdf.cell(50, 6, f"Mobility: {m.get('mobility_index', 0)}", ln=False)
    else:
        pdf.cell(50, 6, f"{sev.upper()} Event", ln=False) # Fallback

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f" @ {alert.get('time_str', 'Unknown')}", ln=True)
    
    # Detailed Metrics (2 Lines)
    pdf.set_xy(15, start_y + 18)
    pdf.set_font("Helvetica", "", 8)
    
    line1, line2 = "", ""
    
    if mode == "congestion":
        line1 = f"Density: {m.get('traffic_density', 0)}%  |  Mobility: {m.get('mobility_index', 0)}%  |  Vehicles: {m.get('detection_count', 0)}"
        line2 = f"Flow: Fast {m.get('fast_pct', 0)}% | Med {m.get('medium_pct', 0)}% | Slow {m.get('slow_pct', 0)}% | Stalled {m.get('stalled_pct', 0)}%"
    elif mode == "vehicle":
        line1 = f"Moving: {m.get('state_counts', {}).get('moving',0)} | Stopped: {m.get('state_counts', {}).get('stopped',0)}"
        line2 = f"Class Breakdown: {str(m.get('class_counts', {}))}"
    elif mode == "crowd":
        line1 = f"Density: {m.get('crowd_density', 0)}% (Class: {m.get('density_class', 'N/A')})"
        line2 = f"Risk Score: {m.get('risk_score', 0)} | Trend: {m.get('crowd_trend', 'stable')}"
    elif mode == "flow":
        line1 = f"Avg Speed: {m.get('avg_speed', 0)} km/h (est.)"
        line2 = f"Flow: Fast {m.get('fast_pct', 0)}% | Med {m.get('medium_pct', 0)}% | Slow {m.get('slow_pct', 0)}% | Stalled {m.get('stalled_pct', 0)}%"
    else:
        line1 = f"Details: {str(m)}"
        
    pdf.cell(0, 5, line1, ln=True)
    pdf.set_x(15)
    pdf.cell(0, 5, line2, ln=True)
    
    # Screenshot
    aid = alert.get("id")
    if aid:
        alert_jpg = ALERTS_DIR / f"{aid}.jpg"
        if alert_jpg.exists():
            pdf.set_xy(15, start_y + 30)
            try:
                pdf.image(str(alert_jpg), h=50) 
            except:
                pdf.cell(0, 10, "[Image Load Error]", ln=True)
    
    pdf.set_y(start_y + 90)
