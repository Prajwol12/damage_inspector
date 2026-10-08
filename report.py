import io
from datetime import datetime

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image as RLImage, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

SEVERITY_INFO = {
    "minor": {
        "color": "#2EB67D",
        "advice": "Cosmetic or limited damage. Repair can typically be scheduled at your convenience. "
                  "Treat any exposed metal promptly to prevent rust.",
    },
    "moderate": {
        "color": "#FFB020",
        "advice": "Noticeable damage that likely needs panel repair or part replacement. "
                  "An in-person professional inspection is recommended before deciding on repairs.",
    },
    "severe": {
        "color": "#E5484D",
        "advice": "Significant damage. Avoid driving the vehicle until it has been inspected by a "
                  "qualified professional; hidden structural or safety-system damage is possible.",
    },
}

DAMAGE_INFO = {
    "dent": "Deformation of a body panel. Check for cracked paint and for distortion of the underlying panel or structure.",
    "scratch": "Surface mark. Clear-coat scratches can often be polished out; deeper ones reach primer or metal and risk corrosion.",
    "crack": "Fracture in a panel, bumper or component. Cracks can spread and may affect structural or safety parts.",
    "glass shatter": "Broken or shattered glass. Safety-critical: the vehicle should not be driven until the glazing is replaced or secured.",
    "lamp broken": "Damaged headlamp or tail lamp. Affects visibility and legal roadworthiness.",
    "tire flat": "Flat or deflated tyre. Do not drive; check the wheel rim and suspension for related damage.",
}


def _location(box, w, h):
    cx = (box[0] + box[2]) / 2 / w
    cy = (box[1] + box[3]) / 2 / h
    horiz = "left" if cx < 0.33 else "right" if cx > 0.66 else "center"
    vert = "upper" if cy < 0.33 else "lower" if cy > 0.66 else "middle"
    if horiz == "center" and vert == "middle":
        return "center"
    if horiz == "center":
        return f"{vert} center"
    if vert == "middle":
        return horiz
    return f"{vert} {horiz}"


def enrich_detections(detections, size):
    """Add location and image-coverage % to each detection."""
    w, h = size
    out = []
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        area = max(0, x2 - x1) * max(0, y2 - y1) / (w * h) * 100
        out.append({**d, "location": _location(d["box"], w, h), "coverage": area})
    return out


def _flowable(pil_img, max_w, max_h):
    pil_img = pil_img.convert("RGB")
    ratio = min(max_w / pil_img.width, max_h / pil_img.height)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return RLImage(buf, width=pil_img.width * ratio, height=pil_img.height * ratio)


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#8A94A8"))
    canvas.drawString(18 * mm, 10 * mm, "AI-assisted estimate - not a substitute for a professional inspection")
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(original, annotated, result, detections):
    """Return the PDF report as bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm,
                            title="Vehicle Damage Assessment Report")
    ss = getSampleStyleSheet()
    ink = colors.HexColor("#101828")
    muted = colors.HexColor("#5B667A")
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=22, textColor=ink, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, textColor=ink, spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14, textColor=ink)
    small = ParagraphStyle("s", parent=body, fontSize=8.5, textColor=muted, leading=12)

    sev = result["severity"]
    sev_color = colors.HexColor(SEVERITY_INFO[sev]["color"])
    sev_conf = result["severity_probs"][sev] * 100
    combined = min(100.0, sum(d["coverage"] for d in detections))

    def base_style(extra=()):
        return TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("TEXTCOLOR", (0, 0), (-1, -1), ink),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5DBE6")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            *extra,
        ])

    story = [
        Paragraph("Vehicle Damage Assessment Report", title),
        Paragraph(f"Generated {datetime.now():%d %B %Y, %H:%M}", small),
        Spacer(1, 8),
    ]

    # Summary
    summary = Table(
        [["OVERALL SEVERITY", "DAMAGE AREAS FOUND", "SEVERITY CONFIDENCE", "APPROX. COVERAGE"],
         [sev.upper(), str(len(detections)), f"{sev_conf:.0f}%", f"{combined:.1f}% of image"]],
        colWidths=[43.5 * mm] * 4,
    )
    summary.setStyle(base_style([
        ("FONTSIZE", (0, 0), (-1, 0), 7.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), muted),
        ("FONTSIZE", (0, 1), (-1, 1), 12),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 1), (0, 1), sev_color),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F5F9")),
    ]))
    story += [summary, Spacer(1, 10)]

    # Annotated image
    story += [Paragraph("Detected damage", h2), _flowable(annotated, 174 * mm, 110 * mm)]

    # Detections table
    story.append(Paragraph("Detection details", h2))
    if detections:
        rows = [["#", "Damage type", "Location on image", "Confidence", "Image coverage"]]
        for i, d in enumerate(sorted(detections, key=lambda x: -x["confidence"]), 1):
            rows.append([str(i), d["type"].title(), d["location"].title(),
                         f"{d['confidence'] * 100:.0f}%", f"{d['coverage']:.1f}%"])
        t = Table(rows, colWidths=[10 * mm, 45 * mm, 45 * mm, 36 * mm, 38 * mm])
        t.setStyle(base_style([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F5F9")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No individual damage areas were detected at the selected sensitivity.", body))

    # Severity
    story.append(Paragraph("Severity analysis", h2))
    prob_rows = [["Severity level", "Probability"]]
    for name, p in result["severity_probs"].items():
        prob_rows.append([name.title(), f"{p * 100:.1f}%"])
    pt = Table(prob_rows, colWidths=[87 * mm, 87 * mm])
    pt.setStyle(base_style([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F5F9")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story += [pt, Spacer(1, 6),
              Paragraph("Severity is rated on the whole photo, not on each damage area separately.", small)]

    # Notes per damage type
    types = sorted({d["type"] for d in detections})
    if types:
        story.append(Paragraph("About the detected damage types", h2))
        for t_name in types:
            story.append(Paragraph(f"<b>{t_name.title()}</b> - {DAMAGE_INFO.get(t_name, '')}", body))
            story.append(Spacer(1, 4))

    # Recommendation
    story += [Paragraph("Recommended action", h2),
              Paragraph(SEVERITY_INFO[sev]["advice"], body),
              Spacer(1, 12)]

    # Original photo
    story += [Paragraph("Original photo", h2), _flowable(original, 120 * mm, 80 * mm), Spacer(1, 12)]

    # Disclaimer
    story.append(Paragraph(
        "<b>Disclaimer.</b> This report was produced automatically by AI models and is intended as a "
        "preliminary assessment only. Results depend on image quality, lighting and angle, and hidden or "
        "structural damage cannot be seen in a photo. It does not include repair cost estimates. "
        "Always have a qualified professional inspect the vehicle.", small))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()