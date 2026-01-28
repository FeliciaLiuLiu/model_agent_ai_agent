"""EDA PDF report generation."""
from __future__ import annotations

import os
import time
import textwrap
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class EDAReportBuilder:
    """Build PDF reports from EDA results."""

    def __init__(self, output_dir: str = "./output_eda", tag: str = "eda"):
        self.output_dir = output_dir
        self.tag = tag
        os.makedirs(output_dir, exist_ok=True)

    def build(self, results: Dict[str, Any], filename: str = "EDA_Report.pdf") -> str:
        """Build PDF report."""
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.image as mpimg

        pdf_path = os.path.join(self.output_dir, filename)
        section_order = [
            "overview",
            "missingness",
            "numeric",
            "categorical",
            "correlation",
            "target",
            "outliers",
            "time",
        ]
        ordered = [(k, results.get(k)) for k in section_order if k in results]

        with PdfPages(pdf_path) as pdf:
            # Cover
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.5, 0.96, "EDA Report", ha="center", fontsize=18, weight="bold")
            fig.text(0.1, 0.90, f"Experiment: {self.tag}", fontsize=11)
            fig.text(0.1, 0.86, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=11)
            pdf.savefig(fig)
            plt.close(fig)

            for sec_key, payload in ordered:
                if not payload:
                    continue
                title = sec_key.replace("_", " ").title()
                metrics = payload.get("metrics", {})
                plots = payload.get("plots", {})
                summary = payload.get("summary", [])

                # Metrics pages
                lines = self._flatten_metrics(metrics)
                self._write_text_pages(pdf, f"{title} - Metrics", lines)

                # Summary pages
                if summary:
                    self._write_text_pages(pdf, f"{title} - Explanation", summary)

                # Plot pages
                for plot_key, img_path in self._collect_images(plots):
                    try:
                        img = mpimg.imread(img_path)
                        fig = plt.figure(figsize=(8.27, 11.69))
                        ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
                        ax.axis("off")
                        ax.imshow(img)
                        pdf.savefig(fig)
                        plt.close(fig)
                    except Exception:
                        continue

        return pdf_path

    def _flatten_metrics(self, metrics: Dict[str, Any]) -> List[str]:
        """Flatten nested metrics into lines."""
        lines: List[str] = []

        def add_line(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    add_line(f"{prefix}{k}.", v)
                return
            if isinstance(value, list):
                joined = ", ".join([str(x) for x in value])
                lines.append(f"{prefix[:-1]}: {joined}")
                return
            lines.append(f"{prefix[:-1]}: {value}")

        for k, v in metrics.items():
            add_line(f"{k}.", v)
        return lines

    def _write_text_pages(self, pdf, title: str, lines: List[str]) -> None:
        """Render wrapped text lines across one or more pages."""
        idx = 0
        while idx < len(lines):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.5, 0.96, title, ha="center", fontsize=14, weight="bold")
            y = 0.90
            while idx < len(lines) and y > 0.08:
                line = lines[idx]
                for wrapped in textwrap.wrap(line, width=110):
                    if y <= 0.08:
                        break
                    fig.text(0.06, y, f"- {wrapped}", fontsize=9)
                    y -= 0.018
                idx += 1
            pdf.savefig(fig)
            plt.close(fig)

    def _collect_images(self, plots: Dict[str, Any], prefix: str = "") -> List[Tuple[str, str]]:
        imgs: List[Tuple[str, str]] = []
        for k, v in plots.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, str) and v.endswith(".png"):
                imgs.append((key, v))
            elif isinstance(v, dict):
                imgs.extend(self._collect_images(v, prefix=f"{key}."))
        return imgs
