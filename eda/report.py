"""EDA PDF report generation using ReportLab."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


class EDAReportBuilder:
    """Build PDF reports from EDA results with structured layout."""

    def __init__(
        self,
        output_dir: str = "./output_eda",
        tag: str = "eda",
        max_table_rows: int = 100,
        page_size=A4,
    ) -> None:
        self.output_dir = output_dir
        self.tag = tag
        self.max_table_rows = max_table_rows
        self.page_size = page_size
        os.makedirs(output_dir, exist_ok=True)

        styles = getSampleStyleSheet()
        self.styles = styles
        self.styles.add(
            ParagraphStyle(
                name="TitleStyle",
                parent=styles["Title"],
                fontSize=20,
                leading=24,
                spaceAfter=10,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SectionHeaderStyle",
                parent=styles["Heading2"],
                fontSize=14,
                leading=18,
                spaceBefore=8,
                spaceAfter=6,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SubHeaderStyle",
                parent=styles["Heading3"],
                fontSize=11,
                leading=14,
                spaceBefore=6,
                spaceAfter=4,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="BodyTextStyle",
                parent=styles["BodyText"],
                fontSize=9,
                leading=12,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SmallNoteStyle",
                parent=styles["BodyText"],
                fontSize=8,
                leading=10,
                textColor=colors.grey,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="TableCell",
                parent=styles["BodyText"],
                fontSize=8,
                leading=10,
                wordWrap="CJK",
                splitLongWords=True,
            )
        )

        self._header_title = "EDA Report"
        self._header_subtitle = ""
        self._generated_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def build(
        self,
        results: Dict[str, Any],
        skipped_sections: List[Dict[str, Any]],
        config: Dict[str, Any],
        filename: str = "EDA_Report.pdf",
    ) -> str:
        """Build PDF report."""
        pdf_path = os.path.join(self.output_dir, filename)
        report_data = {
            "results": results,
            "skipped_sections": skipped_sections,
            "config": config,
        }
        return self.build_pdf(report_data, pdf_path)

    def build_pdf(self, report_data: Dict[str, Any], output_path: str) -> str:
        """Render the PDF report to output_path."""
        results = report_data.get("results", {})
        skipped_sections = report_data.get("skipped_sections", [])
        config = report_data.get("config", {})

        dataset_label = os.path.basename(config.get("data_path", "")) or "Auto-detected dataset"
        self._header_title = "EDA Report"
        self._header_subtitle = dataset_label
        self._generated_at = time.strftime("%Y-%m-%d %H:%M:%S")

        doc = BaseDocTemplate(
            output_path,
            pagesize=self.page_size,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.9 * inch,
            bottomMargin=0.7 * inch,
            title="EDA Report",
        )

        frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
        template = PageTemplate(id="report", frames=[frame], onPage=self._on_page)
        doc.addPageTemplates([template])

        elements: List[Any] = []
        elements.extend(self._cover_section(config))

        for idx, (sec_key, payload) in enumerate(results.items()):
            if not payload:
                continue
            if idx > 0:
                elements.append(PageBreak())

            if sec_key == "data_quality":
                self._render_data_quality_section(elements, payload, doc.width)
                continue
            if sec_key == "univariate":
                self._render_univariate_summary(elements, payload, doc.width)
                continue

            section_title = sec_key.replace("_", " ").title()
            header = Paragraph(section_title, self.styles["SectionHeaderStyle"])
            summary_flow = self._summary_list(payload.get("summary", []))
            if summary_flow:
                elements.append(KeepTogether([header, summary_flow]))
            else:
                elements.append(header)

            for table_def in payload.get("tables", []):
                table = self._table_from_def(table_def)
                if table is None:
                    continue
                elements.append(Spacer(1, 6))
                elements.append(Paragraph(table_def.get("title", "Table"), self.styles["SubHeaderStyle"]))
                elements.append(table)

            plot_paths = list(payload.get("plots", {}).values())
            if plot_paths:
                elements.append(Spacer(1, 8))
                elements.append(Paragraph("Charts", self.styles["SubHeaderStyle"]))
                elements.append(self._chart_grid(plot_paths, doc.width))

        elements.append(PageBreak())
        elements.append(Paragraph("Skipped EDA Sections", self.styles["SectionHeaderStyle"]))
        if skipped_sections:
            rows = [[s.get("section", ""), s.get("reason", "")] for s in skipped_sections]
            table_def = {
                "title": "Skipped Sections",
                "headers": ["Section Name", "Reason for Skipping"],
                "rows": rows,
            }
            table = self._table_from_def(table_def)
            if table is not None:
                elements.append(table)
        else:
            elements.append(Paragraph("No sections were skipped.", self.styles["BodyTextStyle"]))

        elements.append(Spacer(1, 10))
        elements.append(Paragraph("Run Configuration", self.styles["SubHeaderStyle"]))
        config_rows = [[k, str(v)] for k, v in config.items()]
        config_table = self._table_from_def({"headers": ["Key", "Value"], "rows": config_rows})
        if config_table is not None:
            elements.append(config_table)

        doc.build(elements)
        return output_path

    def _on_page(self, canvas, doc) -> None:
        """Header and footer on each page."""
        canvas.saveState()
        width, height = self.page_size

        header_y = height - 0.5 * inch
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(doc.leftMargin, header_y, self._header_title)
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(width - doc.rightMargin, header_y, self._header_subtitle)

        footer_y = 0.35 * inch
        canvas.setFont("Helvetica", 8)
        canvas.drawString(doc.leftMargin, footer_y, f"Generated: {self._generated_at}")
        canvas.drawRightString(width - doc.rightMargin, footer_y, f"Page {doc.page}")
        canvas.restoreState()

    def _cover_section(self, config: Dict[str, Any]) -> List[Any]:
        elements: List[Any] = []
        elements.append(Paragraph("EDA Report", self.styles["TitleStyle"]))
        elements.append(Paragraph(f"Generated: {self._generated_at}", self.styles["SmallNoteStyle"]))
        elements.append(Spacer(1, 8))
        if config:
            elements.append(Paragraph("Dataset Overview", self.styles["SubHeaderStyle"]))
            info = [
                f"Data path: {config.get('data_path', '')}",
                f"Rows used: {config.get('rows_used', '')}",
                f"Target column: {config.get('target_col', '')}",
                f"Time column: {config.get('time_col', '')}",
            ]
            elements.append(self._summary_list(info) or Spacer(1, 1))
        return elements

    def _render_data_quality_section(self, elements: List[Any], payload: Dict[str, Any], doc_width: float) -> None:
        elements.append(Paragraph("Data Quality", self.styles["SectionHeaderStyle"]))
        elements.append(Spacer(1, 6))

        summary_flow = self._summary_list(payload.get("summary", []))
        if summary_flow:
            elements.append(summary_flow)
            elements.append(Spacer(1, 8))

        missingness_payload = payload.get("metrics", {}).get("missingness_payload", {})
        missing_cols = missingness_payload.get("missing_columns", []) or []
        non_missing_cols = missingness_payload.get("non_missing_columns", []) or []

        elements.append(Paragraph("Missingness", self.styles["SubHeaderStyle"]))
        if missing_cols:
            rows = [
                [r["column"], r["missing_count"], r["missing_rate"]]
                for r in missing_cols
            ]
            table_def = {
                "title": "Missingness",
                "headers": ["Column", "Missing Count", "Missing Rate"],
                "rows": rows,
                "style": "missingness",
            }
            table = self._table_from_def(table_def)
            if table is not None:
                elements.append(table)
            elements.append(Spacer(1, 6))

            if non_missing_cols:
                non_missing_text = self._truncate_columns(non_missing_cols, max_cols=10)
                elements.append(
                    Paragraph(
                        f"The following columns have no missing values: {non_missing_text} "
                        f"({len(non_missing_cols)} columns total).",
                        self.styles["BodyTextStyle"],
                    )
                )
        else:
            elements.append(
                Paragraph(
                    "No columns with missing values were detected in this dataset.",
                    self.styles["BodyTextStyle"],
                )
            )

        elements.append(Spacer(1, 10))
        elements.append(Paragraph("Null-like / Placeholder Values", self.styles["SubHeaderStyle"]))
        elements.append(
            Paragraph(
                "These values are not nulls but may represent missing or invalid data placeholders.",
                self.styles["SmallNoteStyle"],
            )
        )

        null_like_payload = payload.get("metrics", {}).get("null_like_payload", []) or []
        null_like_skip = payload.get("metrics", {}).get("null_like_skipped_reason")
        if null_like_skip:
            elements.append(Paragraph(null_like_skip, self.styles["BodyTextStyle"]))
        elif not null_like_payload:
            elements.append(Paragraph("No null-like placeholder values were detected.", self.styles["BodyTextStyle"]))
        else:
            rows = [
                [r["column"], r["null_like_count"], r["null_like_rate"], ", ".join(r.get("examples", []))]
                for r in null_like_payload
            ]
            table_def = {
                "title": "Null-like / Placeholder Values",
                "headers": ["Column", "Null-like Count", "Null-like Rate", "Example Values"],
                "rows": rows,
                "style": "null_like",
            }
            table = self._table_from_def(table_def)
            if table is not None:
                elements.append(table)

        elements.append(Spacer(1, 12))

        for table_def in payload.get("tables", []):
            if table_def.get("title") in ("Missingness", "Null-like / Placeholder Values"):
                continue
            table = self._table_from_def(table_def)
            if table is None:
                continue
            elements.append(Paragraph(table_def.get("title", "Table"), self.styles["SubHeaderStyle"]))
            elements.append(table)
            elements.append(Spacer(1, 8))

        plot_paths = list(payload.get("plots", {}).values())
        if plot_paths:
            elements.append(Spacer(1, 8))
            elements.append(Paragraph("Charts", self.styles["SubHeaderStyle"]))
            elements.append(self._chart_grid(plot_paths, doc_width))

    def _summary_list(self, lines: Sequence[str]) -> Optional[ListFlowable]:
        items = [line for line in lines if line]
        if not items:
            return None
        flow_items = [ListItem(Paragraph(str(item), self.styles["BodyTextStyle"])) for item in items]
        return ListFlowable(
            flow_items,
            bulletType="bullet",
            start="circle",
            leftIndent=18,
            bulletFontSize=8,
            bulletOffsetY=1,
            spaceAfter=4,
        )

    def _table_from_def(self, table_def: Dict[str, Any]) -> Optional[Table]:
        headers = table_def.get("headers", []) or []
        rows = table_def.get("rows", []) or []
        style_name = table_def.get("style", "") or ""
        if not headers and not rows:
            return None
        if headers and not rows:
            return None

        col_count = len(headers) if headers else max((len(r) for r in rows), default=0)
        if col_count == 0:
            return None

        rows = rows[: self.max_table_rows]
        data: List[List[Any]] = []
        if headers:
            data.append([Paragraph(str(h), self.styles["TableCell"]) for h in headers])

        for row in rows:
            if not isinstance(row, (list, tuple)):
                row = [row]
            row = list(row)
            if len(row) < col_count:
                row.extend([""] * (col_count - len(row)))
            if len(row) > col_count:
                row = row[:col_count]
            formatted: List[str] = []
            for idx, cell in enumerate(row):
                header = headers[idx] if idx < len(headers) else ""
                formatted.append(self._format_cell(cell, header))
            data.append([Paragraph(str(cell), self.styles["TableCell"]) for cell in formatted])

        available_width = self.page_size[0] - 1.5 * inch
        col_widths = self._coerce_col_widths(table_def.get("col_widths"), available_width)
        if not col_widths:
            col_widths = self._column_widths(headers, rows, available_width)
        table = Table(data, colWidths=col_widths, repeatRows=1 if headers else 0, hAlign="LEFT")

        style = TableStyle()
        header_bg = colors.HexColor("#2F3B52")
        header_fg = colors.white
        style.add("BACKGROUND", (0, 0), (-1, 0), header_bg)
        style.add("TEXTCOLOR", (0, 0), (-1, 0), header_fg)
        style.add("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
        style.add("ALIGN", (0, 0), (-1, 0), "CENTER")
        style.add("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey)
        style.add("VALIGN", (0, 0), (-1, -1), "TOP")
        style.add("LEFTPADDING", (0, 0), (-1, -1), 4)
        style.add("RIGHTPADDING", (0, 0), (-1, -1), 4)
        style.add("TOPPADDING", (0, 0), (-1, -1), 3)
        style.add("BOTTOMPADDING", (0, 0), (-1, -1), 3)
        style.add("WORDWRAP", (0, 0), (-1, -1), "CJK")

        if style_name == "wide_numeric_stats":
            style.add("FONTSIZE", (0, 0), (-1, 0), 8.5)
            style.add("FONTSIZE", (0, 1), (-1, -1), 8)
            style.add("LEFTPADDING", (0, 0), (-1, -1), 3)
            style.add("RIGHTPADDING", (0, 0), (-1, -1), 3)
            style.add("TOPPADDING", (0, 0), (-1, -1), 2)
            style.add("BOTTOMPADDING", (0, 0), (-1, -1), 2)

        for r in range(1, len(data)):
            if r % 2 == 0:
                style.add("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F5F7FA"))

        alignments = self._infer_alignments(headers, rows, col_count)
        for idx, align in enumerate(alignments):
            style.add("ALIGN", (idx, 1), (idx, -1), align)

        table.setStyle(style)
        return table

    def _coerce_col_widths(self, widths: Any, available_width: float) -> Optional[List[float]]:
        if not widths:
            return None
        try:
            values = [float(w) for w in widths]
        except Exception:
            return None
        if not values:
            return None
        max_val = max(values)
        if max_val <= 5:
            values = [w * inch for w in values]
        total = sum(values)
        if total > available_width and total > 0:
            scale = available_width / total
            values = [w * scale for w in values]
        return values

    def _format_cell(self, value: Any, header: str) -> str:
        if value is None:
            return ""
        header_lower = (header or "").lower()
        if "rate" in header_lower:
            return self._format_percent(value)
        return self._format_number(value)

    def _column_widths(self, headers: List[str], rows: List[List[Any]], available_width: float) -> List[float]:
        col_count = len(headers) if headers else max((len(r) for r in rows), default=1)
        weights = [1] * col_count
        sample_rows = rows[: min(len(rows), 30)]
        for col_idx in range(col_count):
            values = []
            if headers and col_idx < len(headers):
                values.append(str(headers[col_idx]))
            for row in sample_rows:
                if col_idx < len(row):
                    values.append(str(row[col_idx]))
            max_len = max((len(v) for v in values), default=1)
            weights[col_idx] = min(max_len, 40)

        total = sum(weights) or col_count
        return [available_width * (w / total) for w in weights]

    def _infer_alignments(self, headers: List[str], rows: List[List[Any]], col_count: int) -> List[str]:
        aligns = []
        for idx in range(col_count):
            if idx < len(headers):
                header = headers[idx].lower()
                if any(tok in header for tok in ["count", "mean", "std", "min", "max", "25", "50", "75", "rate"]):
                    aligns.append("RIGHT")
                    continue
            column_values = [row[idx] for row in rows if isinstance(row, (list, tuple)) and idx < len(row)]
            aligns.append("RIGHT" if self._is_numeric_column(column_values) else "LEFT")
        return aligns

    def _is_numeric_column(self, values: Sequence[Any]) -> bool:
        has_value = False
        for val in values:
            if val is None or val == "":
                continue
            text = str(val).strip().replace(",", "")
            if text.endswith("%"):
                text = text[:-1]
            try:
                float(text)
                has_value = True
            except ValueError:
                return False
        return has_value

    def _chart_grid(self, image_items: List[Any], available_width: float) -> Table:
        items: List[Dict[str, str]] = []
        for item in image_items:
            if isinstance(item, dict):
                path = item.get("path", "")
                title = item.get("title", "")
            else:
                path = str(item)
                title = ""
            if path and os.path.exists(path):
                items.append({"path": path, "title": title})

        if not items:
            return Table([[Paragraph("No charts available.", self.styles["BodyTextStyle"])]])

        cols = 2
        gap = 12
        chart_width = (available_width - gap) / cols
        cells: List[Any] = []
        for item in items:
            img = self._scaled_image(item["path"], chart_width)
            if item["title"]:
                cell = [Paragraph(item["title"], self.styles["SmallNoteStyle"]), Spacer(1, 2), img]
            else:
                cell = [img]
            cells.append(cell)

        rows: List[List[Any]] = []
        row: List[Any] = []
        for cell in cells:
            row.append(cell)
            if len(row) == cols:
                rows.append(row)
                row = []
        if row:
            while len(row) < cols:
                row.append(Spacer(1, 1))
            rows.append(row)

        table = Table(rows, colWidths=[chart_width, chart_width], hAlign="LEFT")
        table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return table

    def _scaled_image(self, path: str, target_width: float) -> Image:
        img = ImageReader(path)
        width, height = img.getSize()
        if width == 0:
            return Image(path)
        scale = target_width / float(width)
        return Image(path, width=target_width, height=height * scale)

    def _render_univariate_summary(self, elements: List[Any], payload: Dict[str, Any], doc_width: float) -> None:
        """Render Univariate section using a dedicated layout."""
        elements.append(Paragraph("Univariate - Summary", self.styles["SectionHeaderStyle"]))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Univariate", self.styles["SubHeaderStyle"]))
        elements.append(Spacer(1, 4))

        summary_flow = self._summary_list(payload.get("summary", []))
        if summary_flow:
            elements.append(summary_flow)
        elements.append(Spacer(1, 10))

        numeric_table = None
        categorical_tables: List[Dict[str, Any]] = []
        for table_def in payload.get("tables", []):
            title = table_def.get("title", "")
            style = table_def.get("style", "")
            if style == "wide_numeric_stats" or "Numeric Summary Statistics" in title:
                numeric_table = table_def
            elif style in ("categorical_topk", "categorical_topk_flat") or "Top K" in title or "Categorical" in title:
                categorical_tables.append(table_def)

        if numeric_table:
            elements.append(Paragraph("Numeric Summary Statistics", self.styles["SubHeaderStyle"]))
            num_table = self._table_from_def(numeric_table)
            if num_table is not None:
                elements.append(num_table)
            elements.append(Spacer(1, 12))

        if categorical_tables:
            elements.append(Paragraph("Categorical Frequency (Top K)", self.styles["SubHeaderStyle"]))
            elements.append(Spacer(1, 6))
            for table_def in categorical_tables:
                if table_def.get("style") == "categorical_topk":
                    elements.append(Paragraph(table_def.get("title", ""), self.styles["BodyTextStyle"]))
                    elements.append(Spacer(1, 4))
                table = self._table_from_def(table_def)
                if table is not None:
                    elements.append(table)
                elements.append(Spacer(1, 8))

        uni = payload.get("univariate_payload") or {}
        chart_items = uni.get("chart_paths") if isinstance(uni.get("chart_paths"), list) else []
        if not chart_items:
            plots = payload.get("plots", {})
            if isinstance(plots, dict):
                chart_items = [{"title": k, "path": v} for k, v in plots.items()]
            elif isinstance(plots, list):
                chart_items = plots

        if chart_items:
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("Univariate Plots", self.styles["SubHeaderStyle"]))
            elements.append(self._chart_grid(chart_items, doc_width))

    def _truncate_columns(self, cols: Sequence[str], max_cols: int = 12) -> str:
        cols = list(cols)
        if len(cols) <= max_cols:
            return ", ".join(cols)
        extra = len(cols) - max_cols
        return ", ".join(cols[:max_cols]) + f" (+{extra} more)"


    def _format_number(self, value: Any) -> str:
        if value is None:
            return ""
        try:
            val = float(value)
        except (TypeError, ValueError):
            return str(value)
        if val.is_integer():
            return f"{int(val):,}"
        return f"{val:,.3f}"

    def _format_percent(self, value: Any) -> str:
        if value is None:
            return ""
        try:
            val = float(value)
        except (TypeError, ValueError):
            return str(value)
        if val <= 1:
            val = val * 100
        return f"{val:.2f}%"
