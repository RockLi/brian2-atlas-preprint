#!/usr/bin/env python3
"""Build the short NMDA 2025 external-validation technical note as a PDF."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports" / "nmda2025_external_validation_note.md"
DEFAULT_OUTPUT = ROOT.parents[2] / "output" / "pdf" / "nmda2025_external_validation_note.pdf"


def inline_markup(text: str) -> str:
    text = html.escape(text.strip())
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="\2" color="#245a88">\1</a>', text)
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\times", "x")
    text = re.sub(r"\^\{([^}]+)\}", r"<super>\1</super>", text)
    return text


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PaperTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=22,
            textColor=colors.HexColor("#17324d"),
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "author": ParagraphStyle(
            "Author",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4c6173"),
            spaceAfter=11,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=15,
            textColor=colors.HexColor("#17324d"),
            spaceBefore=11,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#2f5d78"),
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.1,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=5,
            allowWidows=0,
            allowOrphans=0,
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.4,
            leading=11,
            alignment=TA_JUSTIFY,
            leftIndent=9 * mm,
            rightIndent=9 * mm,
            borderColor=colors.HexColor("#b9c9d5"),
            borderWidth=0.5,
            borderPadding=7,
            backColor=colors.HexColor("#f5f8fa"),
            spaceAfter=7,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=9.5,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#536675"),
            leftIndent=4 * mm,
            rightIndent=4 * mm,
            spaceBefore=3,
            spaceAfter=7,
        ),
        "table": ParagraphStyle(
            "TableCell",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.1,
            leading=8.3,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#1f2933"),
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=8.2,
            textColor=colors.white,
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7,
            textColor=colors.HexColor("#687b8a"),
        ),
    }


def add_header_footer(canvas, doc):
    canvas.saveState()
    page_width, page_height = A4
    canvas.setStrokeColor(colors.HexColor("#cad6df"))
    canvas.setLineWidth(0.4)
    canvas.line(doc.leftMargin, page_height - 15 * mm, page_width - doc.rightMargin, page_height - 15 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#687b8a"))
    canvas.drawString(doc.leftMargin, page_height - 12.2 * mm, "NMDA 2025 external validation")
    canvas.drawRightString(page_width - doc.rightMargin, 10 * mm, f"Technical validation note  |  {doc.page}")
    canvas.restoreState()


def parse_table(lines: list[str], style_map: dict) -> Table:
    rows = []
    for row_index, line in enumerate(lines):
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if row_index == 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        style = style_map["table_header"] if not rows else style_map["table"]
        rows.append([Paragraph(inline_markup(cell), style) for cell in cells])
    width = A4[0] - 36 * mm
    columns = len(rows[0])
    col_widths = [width / columns] * columns
    table = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#315f7d")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c6d0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f7f9")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ]
        )
    )
    return table


def image_flowable(markdown_path: Path, alt: str, relative: str, style_map: dict):
    path = (markdown_path.parent / relative).resolve()
    with PILImage.open(path) as im:
        pixel_width, pixel_height = im.size
    max_width = A4[0] - 42 * mm
    max_height = 95 * mm
    scale = min(max_width / pixel_width, max_height / pixel_height)
    figure = Image(str(path), width=pixel_width * scale, height=pixel_height * scale)
    figure.hAlign = "CENTER"
    return KeepTogether(
        [
            Spacer(1, 3),
            figure,
            Paragraph(f"Figure. {inline_markup(alt)}", style_map["caption"]),
        ]
    )


def markdown_to_story(path: Path, style_map: dict):
    lines = path.read_text(encoding="utf-8").splitlines()
    story = []
    paragraph = []
    abstract_next = False
    index = 0

    def flush():
        nonlocal paragraph, abstract_next
        if paragraph:
            joined = " ".join(item.strip() for item in paragraph)
            style = style_map["abstract"] if abstract_next else style_map["body"]
            story.append(Paragraph(inline_markup(joined), style))
            paragraph = []
            abstract_next = False

    while index < len(lines):
        line = lines[index]
        if line.startswith("| "):
            flush()
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.extend([Spacer(1, 3), KeepTogether([parse_table(table_lines, style_map)]), Spacer(1, 5)])
            continue
        image_match = re.fullmatch(r"!\[([^]]+)\]\(([^)]+)\)", line.strip())
        if image_match:
            flush()
            story.append(image_flowable(path, image_match.group(1), image_match.group(2), style_map))
        elif line.startswith("# "):
            flush()
            story.append(Spacer(1, 7 * mm))
            story.append(Paragraph(inline_markup(line[2:]), style_map["title"]))
        elif line.startswith("## "):
            flush()
            heading = line[3:]
            story.append(Paragraph(inline_markup(heading), style_map["h1"]))
            abstract_next = heading == "Abstract"
        elif line.startswith("### "):
            flush()
            story.append(Paragraph(inline_markup(line[4:]), style_map["h2"]))
        elif line.startswith("**Rock"):
            flush()
            author_lines = [line]
            while index + 1 < len(lines) and lines[index + 1].strip():
                index += 1
                author_lines.append(lines[index])
            story.append(Paragraph("<br/>".join(inline_markup(x.replace("  ", "")) for x in author_lines), style_map["author"]))
        elif not line.strip():
            flush()
        else:
            paragraph.append(line)
        index += 1
    flush()
    return story


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    style_map = styles()
    doc = SimpleDocTemplate(
        str(args.output),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=17 * mm,
        title="External Validation of Brian2-Atlas on a Published Explicit NMDA Workload",
        author="Rock (rock@bettiai.fr), Independent Researcher",
        subject="External scientific validation using the Skaar et al. 2025 explicit NMDA benchmark",
    )
    doc.build(markdown_to_story(args.input, style_map), onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    print(args.output)


if __name__ == "__main__":
    main()
