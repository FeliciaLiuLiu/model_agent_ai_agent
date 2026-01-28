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
        section_order = ['effectiveness', 'efficiency', 'stability', 'interpretability']
        ordered = [(k, results.get(k)) for k in section_order if k in results]
        self._build_pdf(pdf_path, ordered)
        return pdf_path

    def _build_pdf(self, pdf_path, ordered):
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.image as mpimg
        import textwrap

        with PdfPages(pdf_path) as pdf:
            # Cover page
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.5, 0.96, 'Model Testing Report', ha='center', fontsize=18, weight='bold')
            fig.text(0.1, 0.90, f"Experiment: {self.tag}", fontsize=11)
            fig.text(0.1, 0.86, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=11)
            pdf.savefig(fig); plt.close(fig)

            titles = {'effectiveness': '1) Model Effectiveness', 'efficiency': '2) Model Efficiency',
                      'stability': '3) Model Stability', 'interpretability': '4) Model Interpretability'}

            for sec_key, payload in ordered:
                if not payload: continue
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

                # Metrics page
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.text(0.5, 0.96, titles.get(sec_key, sec_key), ha='center', fontsize=16, weight='bold')
                y = 0.90
                for k, v in self._flatten(metrics):
                    fig.text(0.08, y, f"- {k}: {self._fmt(v)}", fontsize=9)
                    y -= 0.018
                    if y < 0.10: break
                pdf.savefig(fig); plt.close(fig)

                # Explanations page(s)
                if summary_lines:
                    page = 1
                    idx = 0
                    while idx < len(summary_lines):
                        fig = plt.figure(figsize=(8.27, 11.69))
                        fig.text(0.5, 0.96, f"{titles.get(sec_key, sec_key)} - Explanations", ha='center', fontsize=14, weight='bold')
                        y = 0.90
                        while idx < len(summary_lines) and y > 0.08:
                            line = summary_lines[idx]
                            for wrapped in textwrap.wrap(line, width=110):
                                if y <= 0.08: break
                                fig.text(0.06, y, f"- {wrapped}", fontsize=9)
                                y -= 0.018
                            idx += 1
                        pdf.savefig(fig); plt.close(fig)
                        page += 1

                # Plot pages
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
                    except: pass

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
