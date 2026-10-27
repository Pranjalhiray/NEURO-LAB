"""Build a short patient-facing PDF with only the requested finding and images."""
import base64
from datetime import date
from io import BytesIO

from PIL import Image


def _decode_png(value: str | None) -> Image.Image | None:
    if not value:
        return None
    try:
        return Image.open(BytesIO(base64.b64decode(value, validate=True))).convert("RGBA")
    except Exception:
        return None


def _overlay_png(source: Image.Image | None, segmentation: dict | None) -> Image.Image | None:
    if source is None or not segmentation:
        return None
    mask = _decode_png(segmentation.get("mask_png_base64"))
    if mask is None:
        return None
    mask = mask.convert("L").resize(source.size, Image.Resampling.NEAREST)
    tint = Image.new("RGBA", source.size, (225, 74, 84, 0))
    tint.putalpha(mask.point(lambda value: int(value * .48)))
    return Image.alpha_composite(source.convert("RGBA"), tint)


def _pdf_image(image: Image.Image, max_width: float, max_height: float):
    from reportlab.platypus import Image as PdfImage
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    width, height = image.size
    scale = min(max_width / max(width, 1), max_height / max(height, 1))
    return PdfImage(buffer, width=width * scale, height=height * scale)


def _tumor_name(label: str | None) -> tuple[str, str]:
    names = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "pituitary": "Pituitary tumor",
    }
    if label == "no_tumor":
        return "No tumor pattern identified by the model", "None suggested by the model"
    if label in names:
        return "A tumor pattern was suggested by the model", names[label]
    return "An AI result is not available", "Not available"


def make_report(analysis: dict, patient_details: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape

    output = BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=letter, rightMargin=38, leftMargin=38,
        topMargin=34, bottomMargin=32, title="NEURO/LAB Patient MRI Summary",
        author="NEURO/LAB Research Workstation",
    )
    styles = getSampleStyleSheet()
    navy = colors.HexColor("#1d394e")
    teal = colors.HexColor("#167c70")
    text = colors.HexColor("#40586a")
    muted = colors.HexColor("#748694")
    line = colors.HexColor("#dce6e9")
    tint = colors.HexColor("#f5faf8")
    title = ParagraphStyle("PatientTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=navy, alignment=TA_LEFT, spaceAfter=3)
    eyebrow = ParagraphStyle("PatientEyebrow", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=teal, spaceAfter=5)
    body = ParagraphStyle("PatientBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=10, leading=14, textColor=text)
    small = ParagraphStyle("PatientSmall", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.5, leading=12, textColor=muted)
    card_label = ParagraphStyle("PatientCardLabel", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=teal)
    footer_style = ParagraphStyle("PatientFooter", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=8.5, leading=12, textColor=colors.HexColor("#795a31"), alignment=TA_CENTER)

    clean_name = escape(str(patient_details.get("name") or "Patient")[:120])
    age = patient_details.get("age")
    date_of_birth = patient_details.get("date_of_birth")
    if hasattr(date_of_birth, "isoformat"):
        date_of_birth = date_of_birth.isoformat()
    patient_fields = [
        ["PATIENT", clean_name],
        ["AGE", f"{int(age)} years" if age is not None else "Not provided"],
        ["DATE OF BIRTH", escape(str(date_of_birth)) if date_of_birth else "Not provided"],
        ["EMAIL", escape(str(patient_details.get("email") or "Not provided"))],
        ["CONTACT", escape(str(patient_details.get("phone") or "Not provided"))],
    ]

    classification = analysis.get("classification") or {}
    finding, tumor_type = _tumor_name(classification.get("label"))
    segmentation = analysis.get("segmentation")
    has_highlight = bool(segmentation and segmentation.get("bounding_box"))
    location = (
        "The coral highlight shows the area selected by the model in the scan image."
        if has_highlight else
        "The model did not highlight an area in this image."
    ) if segmentation else "A highlighted area was not available for this image."

    story = [
        Paragraph("PATIENT MRI SUMMARY", eyebrow),
        Paragraph("Your scan information", title),
        Paragraph(f"Prepared {date.today().strftime('%d %B %Y')}", small),
        Spacer(1, 11),
    ]
    # A compact two-column information card keeps patient details readable.
    patient_table = Table([[Paragraph(f"<b>{label}</b>", card_label), Paragraph(value, body)] for label, value in patient_fields], colWidths=[1.3 * inch, 5.6 * inch])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), .45, line),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([patient_table, Spacer(1, 12)])

    finding_rows = [
        [Paragraph("WHAT THE MODEL FLAGGED", card_label), Paragraph(escape(finding), body)],
        [Paragraph("AI-SUGGESTED TYPE", card_label), Paragraph(escape(tumor_type), body)],
        [Paragraph("WHERE ON THE SCAN", card_label), Paragraph(escape(location), body)],
    ]
    finding_table = Table(finding_rows, colWidths=[1.75 * inch, 5.15 * inch])
    finding_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), tint),
        ("BOX", (0, 0), (-1, -1), .7, colors.HexColor("#cfe4dc")),
        ("INNERGRID", (0, 0), (-1, -1), .4, colors.HexColor("#dbeae4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([finding_table, Spacer(1, 13)])

    source = _decode_png(analysis.get("source_png_base64"))
    overlay = _overlay_png(source, segmentation)
    if source is not None and overlay is None:
        # Keep the requested two-image layout even when no segmentation was returned.
        overlay = source.copy()
    image_width, image_height = 3.24 * inch, 2.7 * inch
    image_cells = []
    overlay_caption = "Coral highlight shows the model-selected area." if has_highlight else "No highlighted area was available for this scan."
    for label, image, caption in (("ORIGINAL MRI", source, "Original uploaded scan."), ("AI-HIGHLIGHTED AREA", overlay, overlay_caption)):
        cell = [Paragraph(label, card_label), Spacer(1, 6)]
        if image is not None:
            cell.append(_pdf_image(image, image_width, image_height))
        else:
            cell.append(Paragraph("Image not available", small))
        cell.extend([Spacer(1, 4), Paragraph(escape(caption), small)])
        image_cells.append(cell)
    image_table = Table([image_cells], colWidths=[3.45 * inch, 3.45 * inch])
    image_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), .5, line),
        ("INNERGRID", (0, 0), (-1, -1), .5, line),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.extend([image_table, Spacer(1, 12)])

    caution_box = Table([[Paragraph("This is an experimental AI result, not a medical diagnosis. Please discuss your MRI with a qualified clinician.", footer_style)]], colWidths=[6.9 * inch])
    caution_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff8ec")),
        ("BOX", (0, 0), (-1, -1), .5, colors.HexColor("#eadbb8")),
        ("LEFTPADDING", (0, 0), (-1, -1), 11), ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(caution_box)
    doc.build(story)
    return output.getvalue()
