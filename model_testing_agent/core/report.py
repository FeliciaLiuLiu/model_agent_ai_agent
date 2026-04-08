"""PDF Report generation."""
import os, time
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ReportBuilder:
    """Build PDF reports from evaluation results."""

    def __init__(self, output_dir="./output", tag="model_testing"):
        self.output_dir, self.tag = output_dir, tag
        os.makedirs(output_dir, exist_ok=True)

    def build(self, results: Dict[str, Any], filename=None) -> str:
        """Build PDF report."""
        filename = filename or "model_testing_agent_Model_Testing_Report.pdf"
        pdf_path = os.path.join(self.output_dir, filename)
        bundles, segmentation_meta = self._bundle_payloads(results)
        self._build_pdf(pdf_path, bundles, segmentation_meta)
        return pdf_path

    def _bundle_payloads(self, results: Dict[str, Any]):
        """Normalize plain and segmented results into report bundles."""
        section_order = ['effectiveness', 'efficiency', 'stability', 'interpretability']
        segmentation_meta = results.get("segmentation")

        if "overall" in results or "segments" in results:
            bundles = []
            if isinstance(results.get("overall"), dict):
                ordered = [(k, results["overall"].get(k)) for k in section_order if k in results["overall"]]
                bundles.append(("Overall", ordered))
            for segment_name, payload in (results.get("segments") or {}).items():
                ordered = [(k, payload.get(k)) for k in section_order if k in payload]
                bundles.append((segment_name, ordered))
            return bundles, segmentation_meta

        ordered = [(k, results.get(k)) for k in section_order if k in results]
        return [("Overall", ordered)], segmentation_meta

    def _build_pdf(self, pdf_path, bundles, segmentation_meta=None):
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.image as mpimg
        import textwrap

        with PdfPages(pdf_path) as pdf:
            # Cover page
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.5, 0.96, 'Model Testing Report', ha='center', fontsize=18, weight='bold')
            fig.text(0.1, 0.90, f"Experiment: {self.tag}", fontsize=11)
            fig.text(0.1, 0.86, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=11)
            if segmentation_meta:
                fig.text(0.1, 0.82, f"Segmentation Column: {segmentation_meta.get('column')}", fontsize=11)
                fig.text(0.1, 0.79, f"Segments Defined: {len(segmentation_meta.get('segments', []))}", fontsize=11)
                y = 0.74
                for segment in segmentation_meta.get("segments", [])[:12]:
                    line = f"- {segment.get('name')}: {segment.get('row_count')} rows ({segment.get('status')})"
                    fig.text(0.1, y, line, fontsize=9)
                    y -= 0.022
            pdf.savefig(fig); plt.close(fig)

            titles = {'effectiveness': '1) Model Effectiveness', 'efficiency': '2) Model Efficiency',
                      'stability': '3) Model Stability', 'interpretability': '4) Model Interpretability'}

            for bundle_title, ordered in bundles:
                for sec_key, payload in ordered:
                    if not payload:
                        continue
                    metrics = payload.get('metrics', {})
                    plots = payload.get('plots', {})
                    explanations = payload.get('explanations', None)
                    summary_lines = []
                    plot_expl = {}
                    if isinstance(explanations, list):
                        summary_lines = explanations
                    elif isinstance(explanations, dict):
                        if isinstance(explanations.get('summary'), list):
                            summary_lines = explanations.get('summary', [])
                        elif isinstance(explanations.get('metrics'), dict):
                            summary_lines = [f"{k}: {v}" for k, v in explanations.get('metrics', {}).items()]
                        plot_expl = explanations.get('plots', {}) if isinstance(explanations.get('plots'), dict) else {}

                    title_prefix = f"{bundle_title} - " if len(bundles) > 1 or segmentation_meta else ""

                    fig = plt.figure(figsize=(8.27, 11.69))
                    fig.text(0.5, 0.96, f"{title_prefix}{titles.get(sec_key, sec_key)}", ha='center', fontsize=16, weight='bold')
                    y = 0.90
                    for k, v in self._flatten(metrics):
                        fig.text(0.08, y, f"- {k}: {self._fmt(v)}", fontsize=9)
                        y -= 0.018
                        if y < 0.10:
                            break
                    pdf.savefig(fig); plt.close(fig)

                    if summary_lines:
                        idx = 0
                        while idx < len(summary_lines):
                            fig = plt.figure(figsize=(8.27, 11.69))
                            fig.text(
                                0.5,
                                0.96,
                                f"{title_prefix}{titles.get(sec_key, sec_key)} - Explanations",
                                ha='center',
                                fontsize=14,
                                weight='bold',
                            )
                            y = 0.90
                            while idx < len(summary_lines) and y > 0.08:
                                line = summary_lines[idx]
                                for wrapped in textwrap.wrap(line, width=110):
                                    if y <= 0.08:
                                        break
                                    fig.text(0.06, y, f"- {wrapped}", fontsize=9)
                                    y -= 0.018
                                idx += 1
                            pdf.savefig(fig); plt.close(fig)

                    for plot_key, img_path in self._collect_images(plots):
                        try:
                            img = mpimg.imread(img_path)
                            fig = plt.figure(figsize=(8.27, 11.69))
                            ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])
                            ax.axis('off'); ax.imshow(img)
                            expl = plot_expl.get(plot_key)
                            if expl:
                                fig.text(0.05, 0.02, f"{plot_key}: {expl}", fontsize=8)
                            pdf.savefig(fig); plt.close(fig)
                        except Exception:
                            pass

    def _flatten(self, d, prefix=''):
        items = []
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict): items.extend(self._flatten(v, f"{key}."))
            else: items.append((key, v))
        return items

    def _fmt(self, val):
        if isinstance(val, float): return f"{val:.4f}"
        if isinstance(val, (list, tuple)): return str(val[:5]) + ('...' if len(val) > 5 else '')
        return str(val)

    def _collect_images(self, plots, prefix=""):
        imgs = []
        for k, v in plots.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, str) and v.endswith('.png'):
                imgs.append((key, v))
            elif isinstance(v, dict):
                imgs.extend(self._collect_images(v, prefix=f"{key}."))
        return imgs
