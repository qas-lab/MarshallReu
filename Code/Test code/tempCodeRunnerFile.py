report_dict = {
    "API Tools": {"precision": 1.00, "recall": 0.14, "f1-score": 0.25, "support": 14},
    "APT": {"precision": 1.00, "recall": 1.00, "f1-score": 1.00, "support": 1},
    "Ant": {"precision": 0.81, "recall": 0.55, "f1-score": 0.65, "support": 144},
    "Build": {"precision": 0.00, "recall": 0.00, "f1-score": 0.00, "support": 4},
    "CVS": {"precision": 0.00, "recall": 0.00, "f1-score": 0.00, "support": 9},
    "Compare": {"precision": 0.26, "recall": 0.21, "f1-score": 0.23, "support": 24},
    "Core": {"precision": 0.58, "recall": 0.19, "f1-score": 0.28, "support": 96},
    "Debug": {"precision": 0.67, "recall": 0.84, "f1-score": 0.74, "support": 592},
    "Doc": {"precision": 0.00, "recall": 0.00, "f1-score": 0.00, "support": 16},
    "IDE": {"precision": 0.50, "recall": 0.06, "f1-score": 0.10, "support": 18},
    "PMC": {"precision": 0.00, "recall": 0.00, "f1-score": 0.00, "support": 1},
    "Releng": {"precision": 0.83, "recall": 0.25, "f1-score": 0.38, "support": 20},
    "Resources": {"precision": 0.00, "recall": 0.00, "f1-score": 0.00, "support": 3},
    "Runtime": {"precision": 0.00, "recall": 0.00, "f1-score": 0.00, "support": 1},
    "SWT": {"precision": 0.71, "recall": 0.12, "f1-score": 0.20, "support": 43},
    "Search": {"precision": 0.00, "recall": 0.00, "f1-score": 0.00, "support": 16},
    "Team": {"precision": 0.38, "recall": 0.33, "f1-score": 0.36, "support": 15},
    "Text": {"precision": 0.82, "recall": 0.67, "f1-score": 0.74, "support": 153},
    "UI": {"precision": 0.71, "recall": 0.82, "f1-score": 0.76, "support": 808},
    "Update  (deprecated - use RT>Equinox>p2)": {"precision": 0.00, "recall": 0.00, "f1-score": 0.00, "support": 3},
    "User Assistance": {"precision": 0.62, "recall": 0.42, "f1-score": 0.50, "support": 19},
}

# Filter out rows with all zero values
filtered_dict = {k: v for k, v in report_dict.items() if not (v["precision"] == 0.0 and v["recall"] == 0.0 and v["f1-score"] == 0.0)}

# Extract precision, recall, f1-score, and support values
precision = [v["precision"] for v in filtered_dict.values()]
recall = [v["recall"] for v in filtered_dict.values()]
f1_score = [v["f1-score"] for v in filtered_dict.values()]
support = [v["support"] for v in filtered_dict.values()]

# Recalculate macro and weighted averages
macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_f1_score = np.mean(f1_score)

weighted_precision = np.average(precision, weights=support)
weighted_recall = np.average(recall, weights=support)
weighted_f1_score = np.average(f1_score, weights=support)

# Print recalculated values
print(f"Macro Precision: {macro_precision:.2f}")
print(f"Macro Recall: {macro_recall:.2f}")
print(f"Macro F1-Score: {macro_f1_score:.2f}")
print(f"Weighted Precision: {weighted_precision:.2f}")
print(f"Weighted Recall: {weighted_recall:.2f}")
print(f"Weighted F1-Score: {weighted_f1_score:.2f}")