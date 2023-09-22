scores = {
    "FastGCN": [74.93, 53.26, 53.73, 38.02, 41.06, 26.42, 29.91, 33.01],
    "LADIES": [69.90, 67.85, 56.36, 81.88, 43.23, 16.02, 44.58, 67.72],
    "GraphSAINT": [86.20, 77.63, 83.07, 80.50, 44.69, 33.13, 53.71, 59.57],
    "GAS": [80.70, 70.42, 79.22, 94.83, 51.32, 33.79, 69.38, 75.12],
    "AS-GCN": [86.42, 78.80, 89.76, 92.44, 48.21, 30.99, 65.95],  # OOM result is ignored
    "GFGS": [87.29, 75.17, 90.11, 93.68, 47.33, 44.91, 64.54, 73.65]
}

averages = {}
for method, method_scores in scores.items():
    averages[method] = sum(method_scores) / len(method_scores)

for method, avg_score in averages.items():
    print(f"{method}: {avg_score:.2f}%")
