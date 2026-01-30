"""EDA PDF report generation using ReportLab."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


class EDAReportBuilder:
    """Build PDF reports from EDA results."""

    def __init__(self, output_dir: str = "./output_eda", tag: str = "eda") -> None:
        self.output_dir = output_dir
        self.tag = tag
        os.makedirs(output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name="SectionHeader", fontSize=14, leading=18, spaceAfter=6, spaceBefore=10))
        self.styles.add(ParagraphStyle(name="SubHeader", fontSize=11, leading=14, spaceAfter=4, spaceBefore=6))
        self.styles.add(ParagraphStyle(name="Small", fontSize=8, leading=10))

    def build(
        self,
        results: Dict[str, Any],
        skipped_sections: List[Dict[str, Any]],
        config: Dict[str, Any],
        filename: str = "EDA_Report.pdf",
    ) -> str:
        """Build PDF report."""
        pdf_path = os.path.join(self.output_dir, filename)
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, title="EDA Report")

        elements: List[Any] = []
        elements.extend(self._cover_page(config))

        for sec_key, payload in results.items():
            if not payload:
                continue
            elements.append(PageBreak())
            elements.append(Paragraph(sec_key.replace("_", " ").title(), self.styles["SectionHeader"]))

            summary = payload.get("summary", [])
            if summary:
                for line in summary:
                    elements.append(Paragraph(f"- {line}", self.styles["Normal"]))
                elements.append(Spacer(1, 8))

            for table_def in payload.get("tables", []):
                elements.append(Paragraph(table_def.get("title", "Table"), self.styles["SubHeader"]))
                elements.append(self._safe_table(table_def.get("headers", []), table_def.get("rows", [])))
                elements.append(Spacer(1, 6))

            plots = payload.get("plots", {})
            if plots:
                elements.append(Paragraph("Charts", self.styles["SubHeader"]))
                elements.extend(self._image_grid(list(plots.values())))

        elements.append(PageBreak())
        elements.append(Paragraph("Skipped Sections", self.styles["SectionHeader"]))
        if skipped_sections:
            rows = [[s.get("section", ""), s.get("reason", "")] for s in skipped_sections]
            elements.append(self._safe_table(["Section", "Reason"], rows))
        else:
            elements.append(Paragraph("No sections were skipped.", self.styles["Normal"]))

        elements.append(Spacer(1, 8))
        elements.append(Paragraph("Run Configuration", self.styles["SubHeader"]))
        config_rows = [[k, str(v)] for k, v in config.items()]
        elements.append(self._safe_table(["Key", "Value"], config_rows))

        doc.build(elements)
        return pdf_path

    def _cover_page(self, config: Dict[str, Any]) -> List[Any]:
        elements: List[Any] = []
        elements.append(Paragraph("EDA Report", self.styles["Title"]))
        elements.append(Paragraph(f"Experiment: {self.tag}", self.styles["Normal"]))
        elements.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", self.styles["Normal"]))
        if config:
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("Dataset", self.styles["SubHeader"]))
            data_path = config.get("data_path", "")
            elements.append(Paragraph(f"Data path: {data_path}", self.styles["Small"]))
            elements.append(Paragraph(f"Rows used: {config.get('rows_used', '')}", self.styles["Small"]))
            elements.append(Paragraph(f"Target column: {config.get('target_col', '')}", self.styles["Small"]))
            elements.append(Paragraph(f"Time column: {config.get('time_col', '')}", self.styles["Small"]))
        return elements

    def _build_table(self, headers: List[str], rows: List[List[Any]]) -> Table:
        col_count = len(headers) if headers else 0
        if not col_count:
            for row in rows:
                if isinstance(row, (list, tuple)) and row:
                    col_count = max(col_count, len(row))
            if col_count == 0:
                return Table([[Paragraph("No data available.", self.styles["Small"])]])

        data: List[List[Any]] = []
        if headers:
            data.append([Paragraph(str(h), self.styles["Small"]) for h in headers])

        for row in rows[:200]:
            if not isinstance(row, (list, tuple)):
                row = [row]
            if len(row) < col_count:
                row = list(row) + [""] * (col_count - len(row))
            if len(row) > col_count:
                row = list(row)[:col_count]
            cleaned = ["" if cell is None else cell for cell in row]
            data.append([Paragraph(str(cell), self.styles["Small"]) for cell in cleaned])

        table = Table(data, repeatRows=1 if headers else 0, hAlign="LEFT")
        style = TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
        table.setStyle(style)
        return table

    def _safe_table(self, headers: List[str], rows: List[List[Any]]) -> Any:
        try:
            table = self._build_table(headers, rows)
            table.wrap(500, 700)
            return table
        except Exception:
            lines: List[str] = []
            if headers:
                lines.append(" | ".join([str(h) for h in headers]))
            for row in rows[:50]:
                if not isinstance(row, (list, tuple)):
                    row = [row]
                cleaned = ["" if cell is None else str(cell) for cell in row]
                lines.append(" | ".join(cleaned))
            if not lines:
                lines = ["No data available."]
            return Preformatted("\n".join(lines), self.styles["Small"])

    def _image_grid(self, image_paths: List[str], cols: int = 2) -> List[Any]:
        images: List[Image] = []
        for path in image_paths:
            if not os.path.exists(path):
                continue
            images.append(self._scaled_image(path, max_width=220))

        if not images:
            return [Paragraph("No charts available.", self.styles["Normal"])]

        rows: List[List[Any]] = []
        row: List[Any] = []
        for img in images:
            row.append(img)
            if len(row) == cols:
                rows.append(row)
                row = []
        if row:
            while len(row) < cols:
                row.append(Spacer(1, 1))
            rows.append(row)

        table = Table(rows, hAlign="LEFT")
        table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return [table]

    def _scaled_image(self, path: str, max_width: int = 240) -> Image:
        img = ImageReader(path)
        width, height = img.getSize()
        if width == 0:
            return Image(path)
        scale = max_width / float(width)
        return Image(path, width=width * scale, height=height * scale)
