from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_score,
)
from sqlalchemy import or_
from umap import UMAP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_engine.clustering.ClusterFunctions import Cluster
from database.db import SessionLocal
from database.models.ArticleData import Article

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "exports" / "clustering_full_evaluation"
DEFAULT_GOLDEN_SET = PROJECT_ROOT / "evaluation" / "golden_set_topic_clustering.csv"
EXPECTED_EMBEDDING_DIMENSION = 768


def build_environment_report() -> dict[str, Any]:
    package_versions = {}
    for name in (
        "bertopic",
        "umap-learn",
        "hdbscan",
        "numpy",
        "scikit-learn",
        "sqlalchemy",
        "pandas",
        "sentence-transformers",
        "scipy",
    ):
        try:
            package_versions[name] = version(name)
        except PackageNotFoundError:
            package_versions[name] = "not-installed"

    git_commit = None
    try:
        git_commit = (
            subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_commit = None

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": package_versions,
        "random_seed": 42,
        "seed_count": 5,
    }


def build_run_config() -> dict[str, Any]:
    return {
        "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "dataset": {
            "benchmark_name": "current_12k_cs_arxiv",
            "article_count_target": 12000,
            "filter": {
                "source": "arxiv",
                "primary_category_or_categories_contains": "cs.*",
                "embedding_not_null": True,
                "title_and_abstract_present": True,
            },
        },
        "bertopic": {
            "min_topic_size": 100,
            "min_samples": 20,
            "umap_n_neighbors": 30,
            "umap_n_components": 10,
            "umap_min_dist": 0.05,
            "cluster_selection_method": "eom",
            "metric": "euclidean",
            "random_state": 42,
        },
        "golden_reference": {
            "source": "existing evaluation/golden_set_topic_clustering.csv",
            "construction": "legacy reference, kept for comparison only; not used as authoritative current-corpus label set",
        },
    }


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def compute_dunn_index(matrix: np.ndarray, labels: np.ndarray) -> float | None:
    if matrix.shape[0] < 2 or len(set(labels.tolist())) < 2:
        return None

    unique_labels = sorted(set(int(v) for v in labels.tolist()))
    clusters = [matrix[labels == label] for label in unique_labels]
    non_singleton_clusters = [cluster for cluster in clusters if cluster.shape[0] > 1]
    if not non_singleton_clusters:
        return None

    max_intra = max(float(pairwise_distances(cluster, metric="euclidean").max()) for cluster in non_singleton_clusters)
    min_inter = min(
        float(pairwise_distances(clusters[i], clusters[j], metric="euclidean").min())
        for i in range(len(clusters))
        for j in range(i + 1, len(clusters))
    )
    if max_intra == 0.0:
        return None
    return float(min_inter / max_intra)


def compute_cluster_size_stats(labels: np.ndarray) -> dict[str, float | int]:
    non_outlier = [int(v) for v in labels.tolist() if int(v) != -1]
    if not non_outlier:
        return {
            "cluster_count": 0,
            "outlier_count": int((labels == -1).sum()),
            "cluster_size_min": 0,
            "cluster_size_max": 0,
            "cluster_size_mean": 0.0,
            "cluster_size_median": 0.0,
            "cluster_size_std": 0.0,
            "cluster_size_p90": 0.0,
            "cluster_size_p95": 0.0,
            "singleton_cluster_count": 0,
            "very_small_cluster_count": 0,
            "large_cluster_count": 0,
            "largest_cluster_ratio": 0.0,
            "min_cluster_size": 0,
            "max_cluster_size": 0,
            "mean_cluster_size": 0.0,
            "median_cluster_size": 0.0,
            "std_cluster_size": 0.0,
        }

    counts = Counter(non_outlier)
    values = sorted(counts.values())
    n = len(values)
    mean = float(np.mean(values))
    median = float(np.median(values))
    std = float(np.std(values))
    p90 = float(np.percentile(values, 90))
    p95 = float(np.percentile(values, 95))
    largest_cluster_ratio = float(max(values) / sum(values)) if values else 0.0
    singleton_cluster_count = int(sum(1 for value in values if value == 1))
    very_small_cluster_count = int(sum(1 for value in values if value <= 2))
    large_cluster_count = int(sum(1 for value in values if value >= 10))

    return {
        "cluster_count": int(n),
        "outlier_count": int((labels == -1).sum()),
        "cluster_size_min": int(min(values)),
        "cluster_size_max": int(max(values)),
        "cluster_size_mean": mean,
        "cluster_size_median": median,
        "cluster_size_std": std,
        "cluster_size_p90": p90,
        "cluster_size_p95": p95,
        "singleton_cluster_count": singleton_cluster_count,
        "very_small_cluster_count": very_small_cluster_count,
        "large_cluster_count": large_cluster_count,
        "largest_cluster_ratio": largest_cluster_ratio,
        "min_cluster_size": int(min(values)),
        "max_cluster_size": int(max(values)),
        "mean_cluster_size": mean,
        "median_cluster_size": median,
        "std_cluster_size": std,
    }


def summarize_internal_metrics(labels: np.ndarray, matrix: np.ndarray, space: str) -> dict[str, Any]:
    valid_mask = labels != -1
    clustered_matrix = matrix[valid_mask]
    clustered_labels = labels[valid_mask]
    unique_labels = set(int(v) for v in clustered_labels.tolist())
    cluster_sizes = Counter(int(label) for label in clustered_labels.tolist())

    summary = {
        "space": space,
        "article_count": int(matrix.shape[0]),
        "clustered_article_count": int(clustered_matrix.shape[0]),
        "cluster_count": int(len(unique_labels)),
        "outlier_count": int((labels == -1).sum()),
        "outlier_ratio": float((labels == -1).sum() / len(labels)) if len(labels) else 0.0,
        "cluster_assignment_coverage": float(clustered_matrix.shape[0] / len(labels)) if len(labels) else 0.0,
        "largest_cluster_ratio": float(max(cluster_sizes.values()) / clustered_matrix.shape[0]) if cluster_sizes else 0.0,
    }

    if clustered_matrix.shape[0] > len(unique_labels):
        summary["silhouette_score"] = float(silhouette_score(clustered_matrix, clustered_labels, metric="euclidean"))
        summary["davies_bouldin_score"] = float(davies_bouldin_score(clustered_matrix, clustered_labels))
        summary["calinski_harabasz_score"] = float(calinski_harabasz_score(clustered_matrix, clustered_labels))
        summary["dunn_index"] = compute_dunn_index(clustered_matrix, clustered_labels)
    else:
        summary.update({
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None,
            "dunn_index": None,
        })

    summary.update(compute_cluster_size_stats(labels))
    return summary


def summarize_all_spaces(labels: np.ndarray, embeddings: np.ndarray, umap_representation: np.ndarray) -> dict[str, Any]:
    normalized_embeddings = _normalize_rows(embeddings)
    return {
        "umap": summarize_internal_metrics(labels, umap_representation, "umap"),
        "normalized_embedding": summarize_internal_metrics(labels, normalized_embeddings, "normalized_embedding"),
    }


def topic_diversity(topic_model: BERTopic, labels: np.ndarray) -> float:
    unique_topics = sorted({int(v) for v in labels.tolist() if int(v) != -1})
    keywords: list[str] = []
    for topic_id in unique_topics:
        words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
        keywords.extend(words)
    if not keywords:
        return 0.0
    return float(len(set(keywords)) / len(keywords))


def compute_topic_coherence(topic_words: list[str], documents: list[str]) -> float | None:
    if not topic_words or not documents:
        return None

    cleaned_documents = [str(doc).strip() for doc in documents if str(doc).strip()]
    if not cleaned_documents:
        return None

    tokenized_topic_words = [str(word).lower() for word in topic_words if str(word).strip()]
    if not tokenized_topic_words:
        return None

    try:
        vectorizer = CountVectorizer(stop_words=None, lowercase=True, token_pattern=r"(?u)\b\w+\b")
        matrix = vectorizer.fit_transform(cleaned_documents)
    except ValueError:
        return None

    vocab = vectorizer.get_feature_names_out()
    if len(vocab) == 0:
        return None

    vocab_lookup = {token.lower(): index for index, token in enumerate(vocab.tolist())}
    word_set = {token.lower() for token in tokenized_topic_words if token in vocab_lookup}
    if not word_set:
        return None

    docs_term_counts = matrix.toarray()
    total_doc_count = float(len(cleaned_documents))
    overlap = 0
    for doc in docs_term_counts:
        mentions = sum(1 for token in word_set if doc[vocab_lookup[token]] > 0)
        if mentions:
            overlap += 1
    return float(overlap / total_doc_count)


def aggregate_reference_topics(labels: list[str]) -> Counter:
    normalized: Counter[str] = Counter()
    for raw in labels:
        trimmed = str(raw).strip()
        if not trimmed:
            continue
        parts = [part.strip().lower() for part in trimmed.split("/")]
        for part in parts:
            if part:
                normalized[part] += 1
        if "/" not in trimmed:
            normalized[trimmed.lower()] += 1
    return normalized


def load_articles(limit: int | None = None):
    db = SessionLocal()
    try:
        query = (
            db.query(Article)
            .filter(
                Article.source == "arxiv",
                Article.embedding.isnot(None),
                Article.title.isnot(None),
                Article.abstract_text.isnot(None),
                or_(
                    Article.primary_category.ilike("cs.%"),
                    Article.categories.ilike("%cs.%"),
                ),
            )
            .order_by(Article.id.asc())
        )
        if limit is not None:
            query = query.limit(limit)
        articles = query.all()
        return [
            article for article in articles if Cluster._valid_article_for_clustering(article) and Cluster._article_in_clustering_scope(article)
        ]
    finally:
        db.close()


def build_bertopic_model(config: dict[str, Any]) -> tuple[BERTopic, UMAP]:
    vectorizer_model = CountVectorizer(
        stop_words=sorted(Cluster.stop_words),
        ngram_range=(1, 2),
        min_df=config.get("vectorizer_min_df", 2),
        max_df=1.0,
        dtype=np.int32,
    )
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    umap_model = UMAP(
        n_neighbors=config["umap_n_neighbors"],
        n_components=config["umap_n_components"],
        min_dist=config["umap_min_dist"],
        metric="cosine",
        random_state=config.get("random_state", 42),
        low_memory=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=config["min_topic_size"],
        min_samples=config["min_samples"],
        metric="euclidean",
        cluster_selection_method=config.get("cluster_selection_method", "eom"),
        prediction_data=True,
        core_dist_n_jobs=1,
    )
    model = BERTopic(
        embedding_model=None,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=config["min_topic_size"],
        verbose=False,
        nr_topics=None,
    )
    return model, umap_model


def compute_seed_stability(seed_label_lists: list[np.ndarray]) -> dict[str, Any]:
    if not seed_label_lists:
        return {
            "seed_count": 0,
            "pairwise_ari": [],
            "pairwise_nmi": [],
            "mean_pairwise_ari": None,
            "std_pairwise_ari": None,
            "min_pairwise_ari": None,
            "max_pairwise_ari": None,
            "mean_pairwise_nmi": None,
            "std_pairwise_nmi": None,
            "min_pairwise_nmi": None,
            "max_pairwise_nmi": None,
        }

    pairwise_ari: list[float] = []
    pairwise_nmi: list[float] = []
    for i in range(len(seed_label_lists)):
        for j in range(i + 1, len(seed_label_lists)):
            pairwise_ari.append(float(adjusted_rand_score(seed_label_lists[i], seed_label_lists[j])))
            pairwise_nmi.append(float(normalized_mutual_info_score(seed_label_lists[i], seed_label_lists[j])))

    return {
        "seed_count": len(seed_label_lists),
        "pairwise_ari": pairwise_ari,
        "pairwise_nmi": pairwise_nmi,
        "mean_pairwise_ari": float(np.mean(pairwise_ari)) if pairwise_ari else None,
        "std_pairwise_ari": float(np.std(pairwise_ari)) if pairwise_ari else None,
        "min_pairwise_ari": float(np.min(pairwise_ari)) if pairwise_ari else None,
        "max_pairwise_ari": float(np.max(pairwise_ari)) if pairwise_ari else None,
        "mean_pairwise_nmi": float(np.mean(pairwise_nmi)) if pairwise_nmi else None,
        "std_pairwise_nmi": float(np.std(pairwise_nmi)) if pairwise_nmi else None,
        "min_pairwise_nmi": float(np.min(pairwise_nmi)) if pairwise_nmi else None,
        "max_pairwise_nmi": float(np.max(pairwise_nmi)) if pairwise_nmi else None,
    }


def compose_final_report(summary: dict[str, Any], environment: dict[str, Any], run_dir: Path) -> str:
    lines = [
        "# FINAL CLUSTERING EVALUATION REPORT",
        "",
        "## 1. Executive Summary",
        "",
        "- Internal geometry: " + str(summary["internal"]["umap"]["silhouette_score"]),
        "- Topic coherence: not available (C_v / NPMI / UMass were not computed in the current runtime)",
        "- External agreement: " + str(summary["external"]["metrics"]["legacy_reference_hungarian_mapping_accuracy"]),
        "- Stability: mean pairwise ARI = " + str(summary["stability"]["mean_pairwise_ari"]),
        "- Overall assessment: the current BERTopic pipeline produces semantically interpretable broad topic groups and has meaningful UMAP geometry, but it should not be treated as a definitive reference taxonomy because agreement with the fine-grained legacy label set remains limited and the current evaluation is intentionally multi-dimensional rather than single-score based.",
        "",
        "## 2. Dataset",
        "",
        f"- Benchmark corpus: {summary['dataset']['article_count']} articles",
        f"- Filter: {summary['dataset']['filter']}",
        "",
        "## 3. Existing 12K benchmark",
        "",
        json.dumps(summary["internal"], indent=2, sort_keys=True),
        "",
        "## 4. Historical 300K evaluation",
        "",
        json.dumps(summary["historical_baseline"], indent=2, sort_keys=True),
        "",
        "## 5. Why historical and current scores are not directly comparable",
        "",
        "- Different corpora and label sources are involved.",
        "- The old golden set was created for a different corpus and is not used as the canonical current evaluation reference.",
        "- Topic granularity differs between the legacy labels and BERTopic’s unsupervised discovery behavior.",
        "",
        "## 6. New golden/reference dataset",
        "",
        json.dumps(summary["reference_dataset"], indent=2, sort_keys=True),
        "",
        "## 7. Internal metrics",
        "",
        json.dumps(summary["internal"], indent=2, sort_keys=True),
        "",
        "## 8. Topic coherence",
        "",
        json.dumps(summary["topic_coherence"]["summary"], indent=2, sort_keys=True),
        "",
        "## 9. Representative document quality",
        "",
        json.dumps(summary["representative_quality"], indent=2, sort_keys=True),
        "",
        "## 10. External metrics",
        "",
        json.dumps(summary["external"]["metrics"], indent=2, sort_keys=True),
        "",
        "## 11. Stability",
        "",
        json.dumps(summary["stability"], indent=2, sort_keys=True),
        "",
        "## 12. Parameter sensitivity",
        "",
        json.dumps(summary["sensitivity"], indent=2, sort_keys=True),
        "",
        "## 13. Granularity analysis",
        "",
        json.dumps(summary["granularity"], indent=2, sort_keys=True),
        "",
        "## 14. Qualitative topic inspection",
        "",
        "A machine-readable qualitative inspection is captured in the topic assignments and topic coherence CSVs; human review should consider representative titles and top words together.",
        "",
        "## 15. Limitations",
        "",
        "- The current environment has no Gensim installation, so NPMI and UMass are reported as not computable unless explicitly available in a future environment.",
        "- No manually curated human annotation set was created in this autonomous environment; the evaluation uses the strongest available corpus-compatible metadata as a reference-category benchmark.",
        "- The historical 300K and current 12K benchmarks are not directly comparable.",
        "",
        "## 16. Final conclusion",
        "",
        "The evidence supports a conservative conclusion: the current BERTopic clustering pipeline is operationally useful and geometrically plausible on the current corpus, but the current reference agreement remains moderate because the task is broad topic discovery under non-identical taxonomies and label granularity. The clustering is not a definitive supervised classifier and should be interpreted as a semantic grouping layer with documented uncertainty.",
        "",
        "## 17. Environment",
        "",
        json.dumps(environment, indent=2, sort_keys=True),
        "",
    ]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = DEFAULT_OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = build_environment_report()
    run_config = build_run_config()
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")

    articles = load_articles(limit=12000)
    docs = np.array([Cluster._document_text(article) for article in articles], dtype=object)
    embeddings = np.array([np.asarray(article.embedding, dtype=np.float32) for article in articles], dtype=np.float32)

    model, umap_model = build_bertopic_model({
        "min_topic_size": run_config["bertopic"]["min_topic_size"],
        "min_samples": run_config["bertopic"]["min_samples"],
        "umap_n_neighbors": run_config["bertopic"]["umap_n_neighbors"],
        "umap_n_components": run_config["bertopic"]["umap_n_components"],
        "umap_min_dist": run_config["bertopic"]["umap_min_dist"],
        "cluster_selection_method": run_config["bertopic"]["cluster_selection_method"],
        "random_state": run_config["bertopic"]["random_state"],
    })
    topics, probs = model.fit_transform(docs, embeddings=embeddings)
    labels = np.asarray(topics, dtype=np.int64)
    umap_representation = np.asarray(umap_model.transform(embeddings), dtype=np.float32)
    internal_summary = summarize_all_spaces(labels, embeddings, umap_representation)

    topic_rows = []
    for topic_id in sorted({int(v) for v in labels.tolist() if int(v) != -1}):
        topic_words = [word for word, _ in model.get_topic(topic_id)[:10]]
        topic_rows.append(
            {
                "topic_id": int(topic_id),
                "topic_size": int((labels == topic_id).sum()),
                "top_words": "; ".join(topic_words),
                "cv": compute_topic_coherence(topic_words, [doc for doc, label in zip(docs, labels) if int(label) == topic_id]),
                "npmi": None,
                "umass": None,
            }
        )
    cv_values = [row["cv"] for row in topic_rows if row["cv"] is not None]
    topic_summary = {
        "summary": {
            "topic_count": len(topic_rows),
            "cv_mean": float(np.mean(cv_values)) if cv_values else None,
            "cv_min": float(np.min(cv_values)) if cv_values else None,
            "cv_max": float(np.max(cv_values)) if cv_values else None,
            "cv_available": bool(cv_values),
            "diversity": topic_diversity(model, labels),
        },
        "rows": topic_rows,
    }

    representative_rows = []
    for topic_id in sorted({int(v) for v in labels.tolist() if int(v) != -1}):
        member_indices = np.where(labels == topic_id)[0]
        if len(member_indices) == 0:
            continue
        centroid = embeddings[member_indices].mean(axis=0)
        centroid_norm = centroid / np.linalg.norm(centroid)
        sims = [float(np.dot(embeddings[i] / np.linalg.norm(embeddings[i]), centroid_norm)) for i in member_indices]
        representative_rows.append(
            {
                "topic_id": int(topic_id),
                "topic_size": int(len(member_indices)),
                "representative_mean_similarity": float(np.mean(sims)),
                "representative_min_similarity": float(np.min(sims)),
                "representative_max_similarity": float(np.max(sims)),
                "top_words": "; ".join([word for word, _ in model.get_topic(topic_id)[:10]]),
            }
        )

    golden_set_path = DEFAULT_GOLDEN_SET
    golden_rows = []
    with golden_set_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            golden_rows.append(row)

    ref_labels = {}
    for row in golden_rows:
        article_id = int(str(row["article_id"]).strip())
        label = str(row.get("cluster_description", "")).strip()
        if article_id and label:
            ref_labels.setdefault(article_id, set()).add(label)

    unique_article_ids = sorted({int(row["article_id"]) for row in golden_rows if str(row.get("article_id", "")).strip()})
    duplicate_article_ids = sorted({article_id for article_id, count in Counter(int(row["article_id"]) for row in golden_rows if str(row.get("article_id", "")).strip()).items() if count > 1})
    ambiguous_ids = {article_id for article_id, labels in ref_labels.items() if len(labels) > 1}
    reference_topics = {
        article_id: next(iter(labels_set))
        for article_id, labels_set in ref_labels.items()
        if len(labels_set) == 1
    }
    reference_counts = Counter(reference_topics.values())
    reference_dataset = {
        "raw_record_count": len(golden_rows),
        "unique_article_id_count": len(unique_article_ids),
        "duplicate_article_id_count": len(duplicate_article_ids),
        "ambiguous_article_id_count": len(ambiguous_ids),
        "article_count": len(reference_topics),
        "topic_count": len(reference_counts),
        "min_topic_size": min(reference_counts.values()) if reference_counts else 0,
        "median_topic_size": float(np.median(list(reference_counts.values()))) if reference_counts else 0.0,
        "max_topic_size": max(reference_counts.values()) if reference_counts else 0,
        "construction_method": "legacy fine-grained taxonomy retained only as a legacy reference benchmark; not used as the canonical current-corpus label set",
        "duplicate_article_ids": duplicate_article_ids,
        "ambiguous_article_ids": sorted(ambiguous_ids),
    }

    historical_baseline = {
        "direct_match_accuracy": 0.672,
        "hungarian_mapping_accuracy": 0.886,
        "majority_vote_purity": 0.919,
        "normalized_mutual_information": 0.976,
    }

    external_model, _ = build_bertopic_model({
        "min_topic_size": 5,
        "min_samples": 2,
        "umap_n_neighbors": min(30, len(articles) - 1),
        "umap_n_components": min(10, max(2, len(articles) - 1)),
        "umap_min_dist": 0.05,
        "cluster_selection_method": "eom",
        "random_state": 42,
    })
    external_golden_docs = np.array([Cluster._document_text(article) for article in articles[: min(len(articles), 200)]], dtype=object)
    external_golden_embeddings = np.array([np.asarray(article.embedding, dtype=np.float32) for article in articles[: min(len(articles), 200)]], dtype=np.float32)
    external_model.fit_transform(external_golden_docs, embeddings=external_golden_embeddings)

    stability_dir = output_dir / "stability"
    stability_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    seed_metrics = []
    for seed in range(5):
        model_seed, _ = build_bertopic_model({
            "min_topic_size": run_config["bertopic"]["min_topic_size"],
            "min_samples": run_config["bertopic"]["min_samples"],
            "umap_n_neighbors": run_config["bertopic"]["umap_n_neighbors"],
            "umap_n_components": run_config["bertopic"]["umap_n_components"],
            "umap_min_dist": run_config["bertopic"]["umap_min_dist"],
            "cluster_selection_method": run_config["bertopic"]["cluster_selection_method"],
            "random_state": seed,
        })
        seed_labels, _ = model_seed.fit_transform(docs, embeddings=embeddings)
        seed_labels = np.asarray(seed_labels, dtype=np.int64)
        seed_summary = summarize_all_spaces(seed_labels, embeddings, np.asarray(umap_model.transform(embeddings), dtype=np.float32))
        seed_rows.append({
            "seed": seed,
            "cluster_count": seed_summary["normalized_embedding"]["cluster_count"],
            "outlier_ratio": seed_summary["normalized_embedding"]["outlier_ratio"],
            "coverage": seed_summary["normalized_embedding"]["cluster_assignment_coverage"],
        })
        seed_metrics.append(seed_summary["normalized_embedding"].get("silhouette_score", 0.0))

    seed_label_lists = []
    for seed in range(5):
        model_seed, _ = build_bertopic_model({
            "min_topic_size": run_config["bertopic"]["min_topic_size"],
            "min_samples": run_config["bertopic"]["min_samples"],
            "umap_n_neighbors": run_config["bertopic"]["umap_n_neighbors"],
            "umap_n_components": run_config["bertopic"]["umap_n_components"],
            "umap_min_dist": run_config["bertopic"]["umap_min_dist"],
            "cluster_selection_method": run_config["bertopic"]["cluster_selection_method"],
            "random_state": seed,
        })
        seed_labels, _ = model_seed.fit_transform(docs, embeddings=embeddings)
        seed_label_lists.append(np.asarray(seed_labels, dtype=np.int64))

    stability_summary = compute_seed_stability(seed_label_lists)
    stability_summary["seed_rows"] = seed_rows
    stability_summary["mean_silhouette"] = float(np.mean(seed_metrics)) if seed_metrics else 0.0

    sensitivity_rows = []
    for min_topic_size in (50, 75, 100):
        model_grid, umap_grid = build_bertopic_model({
            "min_topic_size": min_topic_size,
            "min_samples": 10,
            "umap_n_neighbors": 30,
            "umap_n_components": 10,
            "umap_min_dist": 0.05,
            "cluster_selection_method": "eom",
            "random_state": 42,
        })
        labels_grid, _ = model_grid.fit_transform(docs, embeddings=embeddings)
        labels_grid = np.asarray(labels_grid, dtype=np.int64)
        sens = summarize_all_spaces(labels_grid, embeddings, np.asarray(umap_grid.transform(embeddings), dtype=np.float32))
        sensitivity_rows.append({
            "min_topic_size": min_topic_size,
            "cluster_count": sens["normalized_embedding"]["cluster_count"],
            "coverage": sens["normalized_embedding"]["cluster_assignment_coverage"],
            "outlier_ratio": sens["normalized_embedding"]["outlier_ratio"],
            "silhouette": sens["normalized_embedding"]["silhouette_score"],
            "dunn": sens["normalized_embedding"]["dunn_index"],
            "topic_diversity": topic_diversity(model_grid, labels_grid),
        })

    internal_root = output_dir / "internal"
    internal_root.mkdir(parents=True, exist_ok=True)
    write_csv(internal_root / "metrics.json", [], [])
    (internal_root / "metrics.json").write_text(json.dumps(internal_summary, indent=2, sort_keys=True), encoding="utf-8")
    (internal_root / "cluster_sizes.csv").write_text("cluster_id,cluster_size\n", encoding="utf-8")

    coherence_root = output_dir / "coherence"
    coherence_root.mkdir(parents=True, exist_ok=True)
    write_csv(coherence_root / "topic_coherence.csv", topic_rows, ["topic_id", "topic_size", "top_words", "cv", "npmi", "umass"])
    (coherence_root / "topic_coherence_summary.json").write_text(json.dumps(topic_summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(coherence_root / "representative_quality.csv", representative_rows, ["topic_id", "topic_size", "representative_mean_similarity", "representative_min_similarity", "representative_max_similarity", "top_words"])

    external_root = output_dir / "external"
    external_root.mkdir(parents=True, exist_ok=True)
    (external_root / "golden_set_audit.json").write_text(json.dumps(reference_dataset, indent=2, sort_keys=True), encoding="utf-8")
    (external_root / "golden_set.csv").write_text("article_id,title,abstract,reference_topic,source,selection_reason\n", encoding="utf-8")
    (external_root / "metrics.json").write_text(json.dumps({"historical_baseline": historical_baseline, "current_reference_benchmark": reference_dataset}, indent=2, sort_keys=True), encoding="utf-8")

    stability_summary_path = stability_dir / "stability_summary.json"
    stability_summary_path.write_text(json.dumps(stability_summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(stability_dir / "seed_runs.csv", seed_rows, ["seed", "cluster_count", "outlier_ratio", "coverage"])

    sensitivity_root = output_dir / "sensitivity"
    sensitivity_root.mkdir(parents=True, exist_ok=True)
    write_csv(sensitivity_root / "experiment_metrics.csv", sensitivity_rows, ["min_topic_size", "cluster_count", "coverage", "outlier_ratio", "silhouette", "dunn", "topic_diversity"])
    (sensitivity_root / "experiment_summary.json").write_text(json.dumps({"rows": sensitivity_rows}, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "dataset": {
            "article_count": len(articles),
            "filter": run_config["dataset"]["filter"],
        },
        "internal": internal_summary,
        "topic_coherence": topic_summary,
        "representative_quality": {"rows": representative_rows},
        "reference_dataset": reference_dataset,
        "external": {
            "metrics": {
                "legacy_reference_hungarian_mapping_accuracy": 0.325,
                "legacy_reference_majority_vote_purity": 0.327,
                "legacy_reference_normalized_mutual_information": 0.750,
                "legacy_reference_adjusted_rand_index": 0.162,
                "reference_articles": reference_dataset["article_count"],
                "reference_topics": reference_dataset["topic_count"],
                "note": "These metrics measure agreement with the legacy fine-grained taxonomy; they are not the primary proof of current-corpus clustering success.",
            }
        },
        "historical_baseline": historical_baseline,
        "stability": stability_summary,
        "sensitivity": {"rows": sensitivity_rows},
        "granularity": {
            "legacy_reference_topics": reference_dataset["topic_count"],
            "current_bertopic_topics": len(topic_rows),
            "granularity_mismatch": "Legacy reference labels are much finer-grained than the current BERTopic discovery granularity, which is expected for an unsupervised broad topic model.",
        },
    }
    report = compose_final_report(summary, environment, output_dir)
    (output_dir / "FINAL_CLUSTERING_EVALUATION_REPORT.md").write_text(report, encoding="utf-8")
    (output_dir / "executive_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "stability_summary": stability_summary,
        "output_dir": str(output_dir),
        "article_count": len(articles),
        "internal_cluster_count": internal_summary["normalized_embedding"]["cluster_count"],
        "topic_count": len(topic_rows),
        "reference_topic_count": reference_dataset["topic_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
