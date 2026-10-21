import json

import numpy as np

from scripts.run_full_clustering_evaluation import (
    aggregate_reference_topics,
    compute_cluster_size_stats,
    compute_topic_coherence,
)


def test_compute_cluster_size_stats_handles_outliers_and_singletons():
    labels = np.array([0, 0, 0, 1, 1, -1, -1, 2])
    stats = compute_cluster_size_stats(labels)

    assert stats["cluster_count"] == 3
    assert stats["min_cluster_size"] == 1
    assert stats["max_cluster_size"] == 3
    assert stats["mean_cluster_size"] == 2.0
    assert stats["median_cluster_size"] == 2.0
    assert stats["singleton_cluster_count"] == 1
    assert stats["largest_cluster_ratio"] == 0.5


def test_compute_topic_coherence_returns_numeric_metric():
    docs = [
        "deep learning model training neural network",
        "deep learning model optimizer training",
        "database query indexing system design",
        "database query optimization index system",
    ]
    words = ["deep", "learning", "model", "database", "query"]
    score = compute_topic_coherence(words, docs)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0


def test_aggregate_reference_topics_builds_broad_taxonomy():
    labels = [
        "machine learning / deep learning",
        "deep learning",
        "computer vision",
        "computer vision / image recognition",
        "database systems",
    ]

    aggregated = aggregate_reference_topics(labels)

    assert aggregated["deep learning"] >= 2
    assert aggregated["computer vision"] >= 2
    assert aggregated["database systems"] >= 1


def test_run_config_and_environment_json_are_valid():
    from scripts.run_full_clustering_evaluation import build_environment_report, build_run_config

    env = build_environment_report()
    cfg = build_run_config()

    assert "python_version" in env
    assert "package_versions" in env
    assert cfg["dataset"]["benchmark_name"] == "current_12k_cs_arxiv"
    assert isinstance(json.dumps(cfg), str)
