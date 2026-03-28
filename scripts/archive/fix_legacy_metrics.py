import json
import re
import sqlite3
from pathlib import Path


def fix_legacy_metrics():
    root_path = Path(__file__).parent.parent
    db_path = root_path / "platform" / "backend" / "storage" / "dev.db"
    reports_dir = root_path / "reports"

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    print(f"Connecting to database at {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get evaluations that were imported from CLI (empty metrics)
    cursor.execute("SELECT id, notes FROM evaluations WHERE tags LIKE '%legacy%'")
    rows = cursor.fetchall()

    updated_count = 0
    for row_id, notes in rows:
        # Extract filename from notes: "Imported from eval_..."
        match = re.search(r"Imported from (eval_.*\.json)", notes)
        if not match:
            continue

        report_filename = match.group(1)
        report_path = reports_dir / report_filename

        if not report_path.exists():
            print(f"Report file {report_filename} not found in {reports_dir}")
            continue

        try:
            with open(report_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {report_filename}: {e}")
            continue

        # Get metrics_summary from JSON
        metrics_summary = data.get("metrics_summary", {})
        if not metrics_summary:
            # Fallback to 'metrics' if metrics_summary is missing (older versions?)
            metrics_summary = data.get("metrics", {})

        # Ensure we have the _avg keys for the UI
        final_metrics = {}
        mapping = {
            "faithfulness": "faithfulness_avg",
            "answer_relevancy": "relevancy_avg",
            "contextual_precision": "precision_avg",
            "contextual_recall": "recall_avg",
        }

        for k, v in metrics_summary.items():
            if "_avg" in k:
                final_metrics[k] = v
            elif k in mapping:
                final_metrics[mapping[k]] = v

        # Fill in relevancy from answer_relevancy if needed
        if "answer_relevancy_avg" in metrics_summary and "relevancy_avg" not in final_metrics:
            final_metrics["relevancy_avg"] = metrics_summary["answer_relevancy_avg"]
        if "answer_relevancy" in metrics_summary and "relevancy_avg" not in final_metrics:
            final_metrics["relevancy_avg"] = metrics_summary["answer_relevancy"]

        # Calculate overall_avg if missing
        if "overall_avg" not in final_metrics:
            avg_scores = [v for k, v in final_metrics.items() if "_avg" in k and v is not None]
            if avg_scores:
                final_metrics["overall_avg"] = sum(avg_scores) / len(avg_scores)

        if final_metrics:
            new_json = json.dumps(final_metrics)
            cursor.execute(
                "UPDATE evaluations SET summary_metrics = ? WHERE id = ?", (new_json, row_id)
            )
            updated_count += 1
            print(f"Successfully restored metrics for evaluation: {row_id} from {report_filename}")

    conn.commit()
    conn.close()
    print(f"\nSuccessfully updated {updated_count} evaluations.")


if __name__ == "__main__":
    fix_legacy_metrics()
