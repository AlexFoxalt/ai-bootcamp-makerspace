from statistics import mean
from comparison import draw_metrics


metrics = [
    {
        "lat": mean([3.33, 2.76, 3.16, 3.08, 2.42, 1.79, 1.08, 2.39, 2.04, 5.01])
    },  # naive
    {
        "lat": mean([2.31, 3.01, 4.08, 2.56, 2.35, 2.46, 1.03, 1.64, 2.07, 2.67])
    },  # naive-with-sem
    {"lat": mean([2.15, 2.32, 2.10, 2.40, 2.08, 1.04, 0.60, 1.40, 1.42, 1.60])},  # bm25
    {
        "lat": mean([7.71, 2.53, 2.78, 2.65, 2.15, 1.81, 1.27, 1.62, 2.38, 2.44])
    },  # context compression
    {
        "lat": mean([5.81, 6.10, 5.88, 6.92, 5.04, 3.05, 2.79, 3.76, 7.76, 4.13])
    },  # multiquery
    {
        "lat": mean([2.50, 3.17, 3.01, 2.67, 2.41, 1.80, 1.69, 1.43, 2.48, 2.05])
    },  # parent
    {
        "lat": mean([6.24, 5.10, 8.31, 7.54, 5.36, 4.01, 2.67, 4.87, 4.71, 4.90])
    },  # ensemble
    {
        "lat": mean([7.43, 7.14, 4.74, 5.28, 6.00, 3.44, 4.31, 3.22, 4.51, 3.77])
    },  # fusion
]
names = [
    "Naive",
    "Naive-with-Semantic",
    "BM25",
    "Contextual Compression",
    "Multi-Query",
    "Parent",
    "Ensemble",
    "Fusion",
]

draw_metrics(metrics, names)
