"""PDF Report generation."""
import os
import textwrap
import time
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ReportBuilder:
    """Build PDF reports from evaluation results."""

    SECTION_TITLES = {
        "effectiveness": "1) Model Effectiveness",
        "efficiency": "2) Model Efficiency",
        "stability": "3) Model Stability",
        "interpretability": "4) Model Interpretability",
    }
    TABLE_HEADERS = ["Metric", "Figure", "Explanation"]

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
        section_order = ["effectiveness", "efficiency", "stability", "interpretability"]
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

        with PdfPages(pdf_path) as pdf:
            self._render_cover_page(pdf, segmentation_meta)

            for bundle_title, ordered in bundles:
                for sec_key, payload in ordered:
                    if not payload:
                        continue
                    metrics = payload.get("metrics", {})
                    plots = payload.get("plots", {})
                    explanations = payload.get("explanations", None)
                    plot_expl = self._extract_plot_explanations(explanations)
                    title_prefix = f"{bundle_title} - " if len(bundles) > 1 or segmentation_meta else ""
                    section_title = f"{title_prefix}{self.SECTION_TITLES.get(sec_key, sec_key)}"

                    rows = self._build_metric_rows(sec_key, metrics)
                    if rows:
                        self._render_metric_table_pages(pdf, section_title, rows)

                    for plot_key, img_path in self._collect_images(plots):
                        try:
                            img = mpimg.imread(img_path)
                            fig = plt.figure(figsize=(8.27, 11.69))
                            ax = fig.add_axes([0.05, 0.07, 0.90, 0.84])
                            ax.axis("off")
                            ax.imshow(img)
                            fig.text(0.5, 0.96, section_title, ha="center", fontsize=16, weight="bold")
                            expl = plot_expl.get(plot_key)
                            if expl:
                                caption = "\n".join(textwrap.wrap(f"{plot_key}: {expl}", width=110))
                                fig.text(0.05, 0.02, caption, fontsize=8)
                            pdf.savefig(fig)
                            plt.close(fig)
                        except Exception:
                            continue

    def _render_cover_page(self, pdf, segmentation_meta=None):
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.96, "Model Testing Report", ha="center", fontsize=18, weight="bold")
        fig.text(0.1, 0.90, f"Experiment: {self.tag}", fontsize=11)
        fig.text(0.1, 0.86, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=11)

        if segmentation_meta:
            fig.text(0.1, 0.82, f"Segmentation Column: {segmentation_meta.get('column')}", fontsize=11)
            fig.text(0.1, 0.79, f"Segments Defined: {len(segmentation_meta.get('segments', []))}", fontsize=11)
            rows = [
                [
                    segment.get("name", ""),
                    str(segment.get("row_count", "")),
                    str(segment.get("status", "")),
                ]
                for segment in segmentation_meta.get("segments", [])[:12]
            ]
            if rows:
                ax = fig.add_axes([0.08, 0.12, 0.84, 0.60])
                ax.axis("off")
                table = ax.table(
                    cellText=rows,
                    colLabels=["Segment", "Rows", "Status"],
                    cellLoc="left",
                    colLoc="left",
                    colWidths=[0.46, 0.18, 0.20],
                    bbox=[0, 0, 1, 1],
                )
                self._style_cover_table(table, len(rows))

        pdf.savefig(fig)
        plt.close(fig)

    def _style_cover_table(self, table, row_count: int):
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        header_color = "#1F4E79"
        for col in range(3):
            cell = table[0, col]
            cell.set_facecolor(header_color)
            cell.set_edgecolor("#D0D7DE")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.get_text().set_ha("left")
        for row in range(1, row_count + 1):
            for col in range(3):
                cell = table[row, col]
                cell.set_edgecolor("#D0D7DE")
                cell.set_facecolor("#F8FBFF" if row % 2 else "white")
                cell.get_text().set_ha("left")
                cell.get_text().set_va("center")

    def _extract_plot_explanations(self, explanations):
        if isinstance(explanations, dict) and isinstance(explanations.get("plots"), dict):
            return explanations["plots"]
        return {}

    def _build_metric_rows(self, sec_key: str, metrics: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        rows = []
        for metric_name, value in self._flatten(metrics):
            rows.append(
                (
                    self._display_metric_name(metric_name),
                    self._fmt(value),
                    self._metric_explanation(sec_key, metric_name, value, metrics),
                )
            )
        return rows

    def _render_metric_table_pages(self, pdf, title: str, rows: List[Tuple[str, str, str]]):
        prepared = [self._prepare_table_row(*row) for row in rows]
        chunks = self._chunk_table_rows(prepared)

        for page_index, chunk in enumerate(chunks, start=1):
            fig = plt.figure(figsize=(8.27, 11.69))
            page_title = title if len(chunks) == 1 else f"{title} ({page_index}/{len(chunks)})"
            fig.text(0.5, 0.96, page_title, ha="center", fontsize=16, weight="bold")

            ax = fig.add_axes([0.04, 0.05, 0.92, 0.86])
            ax.axis("off")
            table = ax.table(
                cellText=[[row["metric"], row["figure"], row["explanation"]] for row in chunk],
                colLabels=self.TABLE_HEADERS,
                cellLoc="left",
                colLoc="left",
                colWidths=[0.23, 0.15, 0.62],
                bbox=[0, 0, 1, 1],
            )
            self._style_metric_table(table, chunk)
            pdf.savefig(fig)
            plt.close(fig)

    def _style_metric_table(self, table, rows: List[Dict[str, Any]]):
        table.auto_set_font_size(False)
        table.set_fontsize(8)

        header_units = 1.4
        total_units = header_units + sum(row["units"] for row in rows)
        header_height = header_units / total_units
        body_heights = [row["units"] / total_units for row in rows]

        for col in range(len(self.TABLE_HEADERS)):
            cell = table[0, col]
            cell.set_facecolor("#1F4E79")
            cell.set_edgecolor("#D0D7DE")
            cell.set_height(header_height)
            cell.PAD = 0.02
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.get_text().set_ha("left")
            cell.get_text().set_va("center")
            cell.get_text().set_wrap(True)

        for row_idx, row_height in enumerate(body_heights, start=1):
            for col in range(len(self.TABLE_HEADERS)):
                cell = table[row_idx, col]
                cell.set_edgecolor("#D0D7DE")
                cell.set_facecolor("#F8FBFF" if row_idx % 2 else "white")
                cell.set_height(row_height)
                cell.PAD = 0.02
                cell.get_text().set_ha("left")
                cell.get_text().set_va("center")
                cell.get_text().set_wrap(True)

    def _prepare_table_row(self, metric: str, figure: str, explanation: str) -> Dict[str, Any]:
        metric_text = "\n".join(textwrap.wrap(metric, width=24)) or metric
        figure_text = "\n".join(textwrap.wrap(figure, width=18)) or figure
        explanation_text = "\n".join(textwrap.wrap(explanation, width=74)) or explanation
        units = max(
            metric_text.count("\n") + 1,
            figure_text.count("\n") + 1,
            explanation_text.count("\n") + 1,
        ) + 0.8
        return {
            "metric": metric_text,
            "figure": figure_text,
            "explanation": explanation_text,
            "units": units,
        }

    def _chunk_table_rows(self, rows: List[Dict[str, Any]], max_units: float = 26.0) -> List[List[Dict[str, Any]]]:
        chunks: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        used_units = 1.4
        for row in rows:
            if current and used_units + row["units"] > max_units:
                chunks.append(current)
                current = [row]
                used_units = 1.4 + row["units"]
            else:
                current.append(row)
                used_units += row["units"]
        if current:
            chunks.append(current)
        return chunks

    def _flatten(self, d, prefix=""):
        items = []
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(self._flatten(v, f"{key}."))
            else:
                items.append((key, v))
        return items

    def _display_metric_name(self, metric_name: str) -> str:
        if metric_name.startswith("precision_at_k."):
            return f"precision_at_k[{metric_name.split('.', 1)[1]}]"
        if metric_name.startswith("recall_at_k."):
            return f"recall_at_k[{metric_name.split('.', 1)[1]}]"
        if metric_name.startswith("fpr_at_thresholds.t_"):
            return f"fpr_at_thresholds[{metric_name.split('t_', 1)[1]}]"
        if metric_name.startswith("confusion_matrix."):
            return f"confusion_matrix[{metric_name.split('.', 1)[1]}]"
        return metric_name

    def _fmt(self, val):
        if isinstance(val, float):
            return f"{val:.4f}"
        if isinstance(val, (list, tuple)):
            return ", ".join(str(x) for x in list(val)[:5]) + (" ..." if len(val) > 5 else "")
        return str(val)

    def _collect_images(self, plots, prefix=""):
        imgs = []
        for k, v in plots.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, str) and v.endswith(".png"):
                imgs.append((key, v))
            elif isinstance(v, dict):
                imgs.extend(self._collect_images(v, prefix=f"{key}."))
        return imgs

    def _metric_explanation(self, sec_key: str, metric_name: str, value: Any, metrics: Dict[str, Any]) -> str:
        handlers = {
            "effectiveness": self._effectiveness_metric_explanation,
            "efficiency": self._efficiency_metric_explanation,
            "stability": self._stability_metric_explanation,
            "interpretability": self._interpretability_metric_explanation,
        }
        handler = handlers.get(sec_key)
        if handler:
            return handler(metric_name, value, metrics)
        return "Summarizes the reported output. Next: compare this result across segments and investigate any material change before sign-off."

    def _effectiveness_metric_explanation(self, metric_name: str, value: Any, metrics: Dict[str, Any]) -> str:
        cm = metrics.get("confusion_matrix", {})
        tp = cm.get("TP", 0)
        fp = cm.get("FP", 0)
        fn = cm.get("FN", 0)
        tn = cm.get("TN", 0)
        total = tp + fp + fn + tn
        positive_rate = (tp + fn) / total if total else 0.0

        if metric_name == "auc_roc":
            if value < 0.70:
                strength = "limited"
            elif value < 0.80:
                strength = "good"
            elif value < 0.90:
                strength = "strong"
            else:
                strength = "excellent"
            return (
                f"Measures ranking power across all thresholds. {value:.4f} indicates {strength} separation. "
                "Next: compare AUC-ROC across segments and keep threshold tuning separate from ranking validation."
            )
        if metric_name == "auc_pr":
            return (
                f"Measures precision-recall performance under class imbalance; the observed event rate is {positive_rate:.2%}. "
                "Next: use this together with alert capacity to decide whether the high-score queue is practically useful."
            )
        if metric_name == "precision":
            return (
                "Shows the share of flagged cases that are true positives at the current threshold. "
                "Next: if this is below review expectations, raise the threshold or improve exclusion features."
            )
        if metric_name == "recall":
            return (
                "Shows the share of true positives captured at the current threshold. "
                "Next: if missed cases are too high, lower the threshold or add stronger risk features."
            )
        if metric_name == "f1":
            return (
                "Balances precision and recall in one score. "
                "Next: use it as a threshold-selection aid when both missed cases and false alerts matter."
            )
        if metric_name == "ks_statistic":
            return (
                "Measures score separation between positive and negative classes. "
                "Next: low KS suggests the model needs better features or a different specification."
            )
        if metric_name == "ks_threshold":
            return (
                "This is the threshold that maximizes KS separation. "
                "Next: treat it as a candidate operating point, then compare it with business capacity and policy limits."
            )
        if metric_name.startswith("confusion_matrix."):
            bucket = metric_name.split(".", 1)[1]
            messages = {
                "TN": "True negatives correctly cleared. Next: preserve this base while tuning the alert threshold.",
                "FP": "False positives create review cost without SAR conversion. Next: reduce this count with threshold tuning or sharper risk features.",
                "FN": "False negatives are missed SAR cases. Next: if this count is unacceptable, lower the threshold or add features that capture suspicious behavior earlier.",
                "TP": "True positives are correctly escalated cases. Next: assess whether this volume is sufficient relative to expected SAR capture goals.",
            }
            return messages.get(
                bucket,
                "Confusion-matrix count at the current threshold. Next: use it with precision and recall to tune the operating point.",
            )
        if metric_name.startswith("precision_at_k."):
            k = metric_name.split(".", 1)[1]
            return (
                f"Shows how accurate the top-{k} highest-risk queue is. "
                "Next: align K with investigator capacity and confirm that top alerts stay high quality."
            )
        if metric_name.startswith("recall_at_k."):
            k = metric_name.split(".", 1)[1]
            return (
                f"Shows how much of the positive population is captured within the top-{k} queue. "
                "Next: increase K if coverage is too low, or improve ranking quality if capacity cannot expand."
            )
        return (
            "Summarizes classification effectiveness at the current run. "
            "Next: compare this metric across segments and confirm the operating threshold still matches business objectives."
        )

    def _efficiency_metric_explanation(self, metric_name: str, value: Any, metrics: Dict[str, Any]) -> str:
        if metric_name == "fpr":
            if value < 0.05:
                level = "low"
            elif value < 0.10:
                level = "moderate"
            else:
                level = "high"
            return (
                f"Measures the share of negatives incorrectly flagged; {value:.4f} is {level}. "
                "Next: if review demand is too high, raise the threshold or add features that better suppress false alerts."
            )
        if metric_name == "tn":
            return (
                "Counts negatives correctly cleared without manual review. "
                "Next: preserve this clearance rate while checking that recall does not fall below policy expectations."
            )
        if metric_name == "fp":
            return (
                "Counts non-events that would still be sent to review. "
                "Next: use this with staffing capacity to decide whether threshold or feature refinement is required."
            )
        if metric_name == "threshold":
            return (
                "This is the active operating cutoff used to flag alerts. "
                "Next: tune it jointly with recall, precision, and review capacity rather than in isolation."
            )
        if metric_name.startswith("fpr_at_thresholds."):
            threshold = metric_name.split("t_", 1)[1]
            return (
                f"Shows the expected false-positive rate if the threshold were set to {threshold}. "
                "Next: use these candidate cutoffs to choose a reviewable alert volume before production sign-off."
            )
        return (
            "Summarizes operational efficiency of the alerting rule. "
            "Next: compare this result with review capacity and adjust the operating threshold if needed."
        )

    def _stability_metric_explanation(self, metric_name: str, value: Any, metrics: Dict[str, Any]) -> str:
        if metric_name == "psi":
            if value < 0.10:
                shift = "negligible"
            elif value < 0.25:
                shift = "moderate"
            else:
                shift = "material"
            return (
                f"Measures score-distribution shift between reference and current populations; {value:.4f} is {shift}. "
                "Next: investigate segment mix, data drift, and recalibration needs when PSI is material."
            )
        if metric_name == "cv_auc_roc_mean":
            return (
                "Average AUC-ROC across validation folds. "
                "Next: compare it with overall AUC-ROC; a large gap can indicate overfitting or unstable sampling."
            )
        if metric_name == "cv_auc_roc_std":
            return (
                "Standard deviation of AUC-ROC across folds. "
                "Next: if variability is high, simplify the model or increase training stability before deployment."
            )
        if metric_name == "cv_auc_pr_mean":
            return (
                "Average AUC-PR across validation folds. "
                "Next: confirm that precision-recall performance remains acceptable under resampling."
            )
        if metric_name == "cv_auc_pr_std":
            return (
                "Standard deviation of AUC-PR across folds. "
                "Next: high dispersion suggests unstable alert quality and warrants deeper validation."
            )
        if metric_name == "bootstrap_auc_roc_mean":
            return (
                "Bootstrap mean AUC-ROC estimates expected ranking performance under repeated sampling. "
                "Next: compare it with the point estimate to judge optimism in the current test run."
            )
        if metric_name == "bootstrap_auc_roc_ci_lower":
            return (
                "Lower confidence bound for bootstrap AUC-ROC. "
                "Next: use it as a conservative performance floor in model-risk discussions."
            )
        if metric_name == "bootstrap_auc_roc_ci_upper":
            return (
                "Upper confidence bound for bootstrap AUC-ROC. "
                "Next: review the interval width with the lower bound to assess uncertainty."
            )
        if metric_name == "concept_drift_detected":
            if value:
                return (
                    "Flags whether performance changed materially across ordered data slices. "
                    "Next: investigate time windows or segments driving the drift and consider challenger monitoring."
                )
            return (
                "No material concept-drift signal was detected in the evaluated slices. "
                "Next: retain ongoing monitoring because stable historical results do not guarantee future stability."
            )
        if metric_name == "concept_drift_score":
            return (
                "Quantifies the magnitude of performance change across ordered slices. "
                "Next: rising scores should trigger deeper review of segment behavior and model refresh timing."
            )
        return (
            "Summarizes model stability under resampling or drift analysis. "
            "Next: combine this with segmentation results before concluding the model is production-stable."
        )

    def _interpretability_metric_explanation(self, metric_name: str, value: Any, metrics: Dict[str, Any]) -> str:
        if metric_name == "model_type":
            return (
                "Identifies the estimator family so explanation methods can be interpreted in the right context. "
                "Next: confirm the chosen explanation methods are appropriate for this model class."
            )
        if metric_name == "methods_used":
            return (
                "Lists the interpretability methods produced in this run. "
                "Next: compare signals across methods before using them in documentation or model challenge."
            )
        if metric_name == "perm_top_features":
            return (
                "Shows features with the largest performance drop when permuted, which highlights global importance. "
                "Next: validate that the top drivers align with business logic, feature governance, and policy expectations."
            )
        if metric_name == "shap_top_features":
            return (
                "Shows the most influential features under SHAP-based contribution analysis. "
                "Next: inspect whether directionality and ranking remain sensible across key segments."
            )
        if metric_name == "lime_instances":
            return (
                "Counts the locally explained observations reviewed with LIME. "
                "Next: examine representative true-positive and false-positive cases to understand local decision logic."
            )
        if metric_name == "pdp_features":
            return (
                "Lists features with partial-dependence plots. "
                "Next: review whether average model response is monotonic, stable, and operationally sensible."
            )
        if metric_name == "ice_features":
            return (
                "Lists features with individual conditional expectation plots. "
                "Next: inspect heterogeneity and interactions that may be hidden in average-only views."
            )
        return (
            "Summarizes interpretability evidence for this run. "
            "Next: use these outputs to document model drivers and challenge unexpected dependencies."
        )
