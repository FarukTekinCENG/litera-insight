from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_score,
)
from sqlalchemy import and_, or_
from umap import UMAP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_engine.clustering.ClusterFunctions import Cluster
from database.db import SessionLocal
from database.models.ArticleData import Article


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "exports" / "clustering_benchmark"
DEFAULT_GOLDEN_SET = PROJECT_ROOT / "evaluation" / "golden_set_topic_clustering.csv"
EXPECTED_EMBEDDING_DIMENSION = 768
HISTORICAL_BASELINE = {
    "direct_match_accuracy": 0.672,
    "hungarian_mapping_accuracy": 0.886,
    "majority_vote_purity": 0.919,
    "normalized_mutual_information": 0.976,
}


def resolve_clustering_config(max_articles: int | None) -> dict:
    if max_articles is None:
        max_articles = 30000

    if max_articles >= 20000:
        return {
            "max_articles": 12000,
            "min_topic_size": 100,
            "min_samples": 20,
            "umap_n_neighbors": 30,
            "umap_n_components": 10,
            "umap_min_dist": 0.05,
        }

    if max_articles >= 10000:
        return {
            "max_articles": min(max_articles, 12000),
            "min_topic_size": 75,
            "min_samples": 15,
            "umap_n_neighbors": 25,
            "umap_n_components": 10,
            "umap_min_dist": 0.05,
        }

    return {
        "max_articles": max_articles,
        "min_topic_size": 50,
        "min_samples": 10,
        "umap_n_neighbors": 15,
        "umap_n_components": 10,
        "umap_min_dist": 0.05,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fresh BERTopic clustering benchmark on the live database.")
    parser.add_argument("--max-articles", type=int, default=30000, help="Maximum number of CS arXiv articles to benchmark.")
    parser.add_argument("--min-topic-size", type=int, default=None)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--umap-n-neighbors", type=int, default=None)
    parser.add_argument("--umap-n-components", type=int, default=None)
    parser.add_argument("--umap-min-dist", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
    args = parser.parse_args()
    cfg = resolve_clustering_config(args.max_articles)
    args.max_articles = cfg["max_articles"]
    args.min_topic_size = cfg["min_topic_size"] if args.min_topic_size is None else args.min_topic_size
    args.min_samples = cfg["min_samples"] if args.min_samples is None else args.min_samples
    args.umap_n_neighbors = cfg["umap_n_neighbors"] if args.umap_n_neighbors is None else args.umap_n_neighbors
    args.umap_n_components = cfg["umap_n_components"] if args.umap_n_components is None else args.umap_n_components
    args.umap_min_dist = cfg["umap_min_dist"] if args.umap_min_dist is None else args.umap_min_dist
    return args


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
            article
            for article in articles
            if Cluster._valid_article_for_clustering(article) and Cluster._article_in_clustering_scope(article)
        ]
    finally:
        db.close()


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

    # Dunn uses the maximum pairwise diameter within a cluster and the
    # minimum pairwise distance between different clusters.
    max_intra = max(
        float(pairwise_distances(cluster, metric="euclidean").max())
        for cluster in non_singleton_clusters
    )
    min_inter = min(
        float(pairwise_distances(clusters[i], clusters[j], metric="euclidean").min())
        for i in range(len(clusters))
        for j in range(i + 1, len(clusters))
    )
    if max_intra == 0.0:
        return None
    return float(min_inter / max_intra)


def summarize_metrics(labels: np.ndarray, matrix: np.ndarray, space: str) -> dict:
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

    return summary


def summarize_all_spaces(labels: np.ndarray, embeddings: np.ndarray, umap_representation: np.ndarray) -> dict:
    normalized_embeddings = _normalize_rows(embeddings)
    return {
        "umap": summarize_metrics(labels, umap_representation, "umap"),
        "normalized_embedding": summarize_metrics(labels, normalized_embeddings, "normalized_embedding"),
    }


def topic_diversity(topic_model: BERTopic, labels: np.ndarray) -> float:
    unique_topics = sorted({int(v) for v in labels.tolist() if int(v) != -1})
    keywords = []
    for topic_id in unique_topics:
        words = [word for word, _ in topic_model.get_topic(topic_id)[:10]]
        keywords.extend(words)
    if not keywords:
        return 0.0
    return float(len(set(keywords)) / len(keywords))


def resolve_external_clustering_config(args: argparse.Namespace, article_count: int) -> dict:
    if article_count < 3:
        raise ValueError("At least three eligible golden articles are required for external clustering.")
    return {
        "min_topic_size": min(args.min_topic_size, 5),
        "min_samples": min(args.min_samples, 2),
        "umap_n_neighbors": min(args.umap_n_neighbors, article_count - 1),
        "umap_n_components": min(args.umap_n_components, max(2, article_count - 1)),
        "umap_min_dist": args.umap_min_dist,
        "vectorizer_min_df": 1,
    }


def build_bertopic_model(config: dict) -> tuple[BERTopic, UMAP]:
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
        random_state=42,
        low_memory=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=config["min_topic_size"],
        min_samples=config["min_samples"],
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        core_dist_n_jobs=4,
    )
    model = BERTopic(
        embedding_model=None,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=config["min_topic_size"],
        verbose=True,
        nr_topics=None,
    )
    return model, umap_model


def load_golden_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        required_fields = {"article_id", "title", "cluster_description"}
        if not required_fields.issubset(rows.fieldnames or set()):
            raise ValueError(f"Golden set must contain: {', '.join(sorted(required_fields))}")
        return list(rows)


def load_reference_labels(rows: list[dict[str, str]]) -> tuple[dict[int, str], set[int], Counter]:
    id_labels: dict[int, set[str]] = {}
    id_counts = Counter()
    for row in rows:
        article_id = int(row["article_id"])
        label = row["cluster_description"].strip()
        id_counts[article_id] += 1
        if label:
            id_labels.setdefault(article_id, set()).add(label)

    conflicting_ids = {article_id for article_id, labels in id_labels.items() if len(labels) > 1}
    reference_labels = {
        article_id: next(iter(labels))
        for article_id, labels in id_labels.items()
        if len(labels) == 1
    }
    return reference_labels, conflicting_ids, id_counts


def load_golden_evaluation_corpus(path: Path) -> tuple[list, dict[int, str], dict]:
    rows = load_golden_rows(path)
    reference_labels, ambiguous_ids, id_counts = load_reference_labels(rows)
    unique_ids = set(id_counts)

    db = SessionLocal()
    try:
        articles_by_id = {
            article.id: article
            for article in db.query(Article).filter(Article.id.in_(unique_ids)).all()
        }
    finally:
        db.close()

    present_articles = [articles_by_id[article_id] for article_id in unique_ids if article_id in articles_by_id]
    has_title_abstract = [
        article for article in present_articles
        if isinstance(article.title, str) and bool(article.title.strip())
        and isinstance(article.abstract_text, str) and bool(article.abstract_text.strip())
    ]
    has_embedding = [article for article in present_articles if article.embedding is not None]
    correct_dimension = [
        article for article in has_embedding
        if len(article.embedding) == EXPECTED_EMBEDDING_DIMENSION
    ]
    cs_scope = [article for article in present_articles if Cluster._article_in_clustering_scope(article)]
    valid_for_clustering = [
        article for article in present_articles
        if Cluster._valid_article_for_clustering(article)
    ]
    eligible_articles = [
        article for article in present_articles
        if article.id in reference_labels
        and article.id not in ambiguous_ids
        and article.embedding is not None
        and len(article.embedding) == EXPECTED_EMBEDDING_DIMENSION
        and Cluster._valid_article_for_clustering(article)
        and Cluster._article_in_clustering_scope(article)
    ]
    eligible_articles.sort(key=lambda article: article.id)

    dimension_counts = Counter(
        len(article.embedding)
        for article in has_embedding
        if article.embedding is not None
    )
    reference_topic_sizes = Counter(reference_labels.values())
    reference_sizes = sorted(reference_topic_sizes.values())
    audit = {
        "golden_record_count": len(rows),
        "unique_golden_id_count": len(unique_ids),
        "duplicate_reference_id_count": sum(count > 1 for count in id_counts.values()),
        "ambiguous_reference_id_count": len(ambiguous_ids),
        "present_in_db": len(present_articles),
        "has_title_abstract": len(has_title_abstract),
        "has_embedding": len(has_embedding),
        "embedding_dimension_counts": {str(dimension): count for dimension, count in sorted(dimension_counts.items())},
        "correct_embedding_dimension": len(correct_dimension),
        "cs_scope": len(cs_scope),
        "valid_for_clustering": len(valid_for_clustering),
        "eligible_for_external_evaluation": len(eligible_articles),
        "reference_topic_count": len(reference_topic_sizes),
        "reference_topic_size_min": min(reference_sizes) if reference_sizes else 0,
        "reference_topic_size_median": reference_sizes[len(reference_sizes) // 2] if reference_sizes else 0,
        "reference_topic_size_mean": sum(reference_sizes) / len(reference_sizes) if reference_sizes else 0.0,
        "reference_topic_size_max": max(reference_sizes) if reference_sizes else 0,
    }
    return eligible_articles, reference_labels, audit


def compute_external_metrics(
    article_ids: list[int],
    predicted_labels: np.ndarray,
    reference_labels: dict[int, str],
    ambiguous_reference_ids: set[int] | None = None,
) -> dict:
    if len(article_ids) != len(predicted_labels):
        raise ValueError("article_ids and predicted_labels must have the same length")

    ambiguous_reference_ids = ambiguous_reference_ids or set()
    matched_mask = np.array(
        [article_id in reference_labels and article_id not in ambiguous_reference_ids for article_id in article_ids],
        dtype=bool,
    )
    outlier_mask = predicted_labels == -1
    evaluated_mask = matched_mask & ~outlier_mask
    evaluated_count = int(evaluated_mask.sum())
    matched_count = int(matched_mask.sum())
    outlier_count = int((matched_mask & outlier_mask).sum())
    reference_values = [reference_labels[article_id] for article_id, valid in zip(article_ids, evaluated_mask) if valid]
    predicted_values = [int(label) for label, valid in zip(predicted_labels.tolist(), evaluated_mask) if valid]
    reference_topics = sorted(set(reference_values))
    predicted_topics = sorted(set(predicted_values))

    result = {
        "golden_set_article_count": int(len(reference_labels)),
        "ambiguous_reference_count": len(ambiguous_reference_ids),
        "matched_article_count": matched_count,
        "evaluation_sample_count": evaluated_count,
        "reference_topic_count": len(reference_topics),
        "predicted_topic_count": len(predicted_topics),
        "outlier_count": outlier_count,
        "outlier_handling": "excluded",
        "direct_match_accuracy": None,
        "hungarian_mapping_accuracy": 0.0,
        "majority_vote_purity": 0.0,
        "normalized_mutual_information": 0.0,
        "adjusted_rand_index": 0.0,
        "direct_match_note": "Not computed: BERTopic topic IDs are arbitrary and are not reference descriptions.",
    }
    if not evaluated_count:
        return result

    true_to_index = {label: index for index, label in enumerate(reference_topics)}
    predicted_to_index = {label: index for index, label in enumerate(predicted_topics)}
    contingency = np.zeros((len(reference_topics), len(predicted_topics)), dtype=np.int64)
    for reference, predicted in zip(reference_values, predicted_values):
        contingency[true_to_index[reference], predicted_to_index[predicted]] += 1

    row_indices, column_indices = linear_sum_assignment(-contingency)
    mapping = {predicted_topics[column]: reference_topics[row] for row, column in zip(row_indices, column_indices)}
    result["hungarian_mapping_accuracy"] = float(
        sum(mapping.get(predicted) == reference for reference, predicted in zip(reference_values, predicted_values))
        / evaluated_count
    )
    result["majority_vote_purity"] = float(contingency.max(axis=0).sum() / evaluated_count)
    result["normalized_mutual_information"] = float(
        normalized_mutual_info_score(reference_values, predicted_values)
    )
    result["adjusted_rand_index"] = float(adjusted_rand_score(reference_values, predicted_values))
    return result


def main() -> int:
    args = parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    articles = load_articles(limit=args.max_articles)
    if not articles:
        raise RuntimeError("No eligible arXiv CS articles with embeddings were found.")

    print(f"Loaded {len(articles)} eligible articles for BERTopic benchmark.")

    docs = np.array([Cluster._document_text(article) for article in articles], dtype=object)
    embeddings = np.array([np.asarray(article.embedding, dtype=np.float32) for article in articles], dtype=np.float32)

    internal_config = {
        "min_topic_size": args.min_topic_size,
        "min_samples": args.min_samples,
        "umap_n_neighbors": args.umap_n_neighbors,
        "umap_n_components": args.umap_n_components,
        "umap_min_dist": args.umap_min_dist,
    }
    model, umap_model = build_bertopic_model(internal_config)
    topics, probs = model.fit_transform(docs, embeddings=embeddings)

    labels = np.asarray(topics, dtype=np.int64)
    umap_representation = np.asarray(umap_model.transform(embeddings), dtype=np.float32)
    metrics_by_space = summarize_all_spaces(labels, embeddings, umap_representation)
    golden_articles, reference_labels, alignment_audit = load_golden_evaluation_corpus(args.golden_set)
    if not golden_articles:
        raise RuntimeError("No eligible golden-set articles were found for external evaluation.")
    golden_docs = np.array([Cluster._document_text(article) for article in golden_articles], dtype=object)
    golden_embeddings = np.array(
        [np.asarray(article.embedding, dtype=np.float32) for article in golden_articles],
        dtype=np.float32,
    )
    external_config = resolve_external_clustering_config(args, len(golden_articles))
    external_model, external_umap_model = build_bertopic_model(external_config)
    external_topics, _ = external_model.fit_transform(golden_docs, embeddings=golden_embeddings)
    external_labels = np.asarray(external_topics, dtype=np.int64)
    external_metrics = compute_external_metrics(
        [int(article.id) for article in golden_articles],
        external_labels,
        reference_labels,
        set(),
    )
    external_metrics.update({
        "evaluation_corpus": "golden_set",
        "eligible_article_count": alignment_audit["eligible_for_external_evaluation"],
        "outlier_ratio": float(external_metrics["outlier_count"] / len(golden_articles)) if golden_articles else 0.0,
        "ambiguous_reference_count": alignment_audit["ambiguous_reference_id_count"],
        "external_clustering_config": external_config,
    })
    embedding_metrics = metrics_by_space["normalized_embedding"]
    summary = {
        key: value
        for key, value in embedding_metrics.items()
        if key not in {"space", "silhouette_score", "davies_bouldin_score", "calinski_harabasz_score", "dunn_index"}
    }
    summary["metrics_by_space"] = metrics_by_space
    summary["external_metrics"] = external_metrics
    summary["golden_alignment"] = alignment_audit
    summary["historical_baseline"] = HISTORICAL_BASELINE
    summary["deprecated_embedding_metrics"] = {
        "description": "Deprecated top-level metric values retained for consumers of the previous schema.",
        "values": {
            key: embedding_metrics[key]
            for key in ("silhouette_score", "davies_bouldin_score", "calinski_harabasz_score", "dunn_index")
        },
    }
    summary["min_topic_size"] = args.min_topic_size
    summary["min_samples"] = args.min_samples
    summary["umap_n_neighbors"] = args.umap_n_neighbors
    summary["umap_n_components"] = args.umap_n_components
    summary["umap_min_dist"] = args.umap_min_dist
    summary["topic_diversity"] = topic_diversity(model, labels)
    summary["num_topics"] = len({int(v) for v in labels.tolist() if int(v) != -1})
    summary["generated_at_utc"] = datetime.now(UTC).isoformat()

    model.save(str(run_dir / "bertopic_model"))
    model.visualize_topics().write_html(str(run_dir / "bertopic_topics.html"))
    model.visualize_barchart().write_html(str(run_dir / "bertopic_barchart.html"))
    model.visualize_hierarchy().write_html(str(run_dir / "bertopic_hierarchy.html"))
    external_model.save(str(run_dir / "golden_bertopic_model"))

    topic_info = model.get_topic_info()
    topic_info.to_csv(run_dir / "topic_info.csv", index=False)
    external_model.get_topic_info().to_csv(run_dir / "golden_topic_info.csv", index=False)

    assignment_rows = []
    for article, topic_id, prob in zip(articles, labels, probs):
        assignment_rows.append(
            {
                "article_id": article.id,
                "source": article.source,
                "title": article.title,
                "topic": int(topic_id),
                "probability": float(np.max(prob)) if prob is not None and np.ndim(prob) > 0 else None,
            }
        )
    with (run_dir / "paper_topic_assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in assignment_rows for key in row}))
        writer.writeheader()
        writer.writerows(assignment_rows)

    golden_assignment_rows = [
        {
            "article_id": article.id,
            "title": article.title,
            "reference_topic": reference_labels.get(int(article.id)),
            "topic": int(topic_id),
        }
        for article, topic_id in zip(golden_articles, external_labels)
    ]
    with (run_dir / "golden_paper_topic_assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["article_id", "title", "reference_topic", "topic"])
        writer.writeheader()
        writer.writerows(golden_assignment_rows)

    with (run_dir / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)

    metric_lines = []
    for space, space_summary in metrics_by_space.items():
        metric_lines.extend([
            f"### {'UMAP' if space == 'umap' else 'Normalized Embedding'}",
            f"- Silhouette Score: {space_summary['silhouette_score'] if space_summary['silhouette_score'] is not None else 'n/a'}",
            f"- Calinski-Harabasz Index: {space_summary['calinski_harabasz_score'] if space_summary['calinski_harabasz_score'] is not None else 'n/a'}",
            f"- Davies-Bouldin Index: {space_summary['davies_bouldin_score'] if space_summary['davies_bouldin_score'] is not None else 'n/a'}",
            f"- Dunn Index: {space_summary['dunn_index'] if space_summary['dunn_index'] is not None else 'n/a'}",
            "",
        ])

    def format_metric(value: float | None, percentage: bool = False) -> str:
        if value is None:
            return "n/a"
        return f"{value:.4%}" if percentage else f"{value:.4f}"

    external_report_lines = [
        "## Golden Set Alignment Audit",
        "",
        f"- Golden records: {alignment_audit['golden_record_count']}",
        f"- Unique golden IDs: {alignment_audit['unique_golden_id_count']}",
        f"- Duplicate IDs: {alignment_audit['duplicate_reference_id_count']}",
        f"- Ambiguous IDs excluded: {alignment_audit['ambiguous_reference_id_count']}",
        f"- Present in DB: {alignment_audit['present_in_db']}",
        f"- Has title and abstract: {alignment_audit['has_title_abstract']}",
        f"- Has embedding: {alignment_audit['has_embedding']}",
        f"- Correct {EXPECTED_EMBEDDING_DIMENSION}-dimension embedding: {alignment_audit['correct_embedding_dimension']}",
        f"- In CS clustering scope: {alignment_audit['cs_scope']}",
        f"- Valid for clustering: {alignment_audit['valid_for_clustering']}",
        f"- Eligible external articles: {alignment_audit['eligible_for_external_evaluation']}",
        f"- Reference topic count: {alignment_audit['reference_topic_count']}",
        f"- Reference topic size (min/median/mean/max): {alignment_audit['reference_topic_size_min']} / {alignment_audit['reference_topic_size_median']} / {alignment_audit['reference_topic_size_mean']:.2f} / {alignment_audit['reference_topic_size_max']}",
        "",
        "## External / Reference-Based Evaluation",
        "",
        "This is a separate BERTopic fit on the eligible golden-set corpus; it does not modify the internal 12K corpus.",
        f"- Evaluation sample: {external_metrics['evaluation_sample_count']}",
        f"- Outliers excluded: {external_metrics['outlier_count']} ({external_metrics['outlier_ratio']:.4f})",
        f"- Reference topics: {external_metrics['reference_topic_count']}",
        f"- Predicted topics: {external_metrics['predicted_topic_count']}",
        f"- External configuration: min_topic_size={external_config['min_topic_size']}, min_samples={external_config['min_samples']}, UMAP={external_config['umap_n_neighbors']}/{external_config['umap_n_components']}/{external_config['umap_min_dist']}",
        "- Outlier handling: topic -1 excluded from external metrics.",
        "",
        "| Metric | New BERTopic Run |",
        "| --- | ---: |",
        f"| Hungarian Mapping Accuracy | {format_metric(external_metrics['hungarian_mapping_accuracy'], True)} |",
        f"| Majority Vote Purity | {format_metric(external_metrics['majority_vote_purity'], True)} |",
        f"| Normalized Mutual Information | {format_metric(external_metrics['normalized_mutual_information'])} |",
        f"| Adjusted Rand Index | {format_metric(external_metrics['adjusted_rand_index'])} |",
        "",
        "Direct Match Accuracy is not computed for the new run because BERTopic topic IDs are arbitrary and are not reference descriptions.",
        "",
        "## Historical Baseline",
        "",
        "These values come from the legacy database-cluster evaluator and are not directly comparable to the new BERTopic run.",
        "",
        "| Metric | Historical Result |",
        "| --- | ---: |",
        f"| Direct Match Accuracy | {HISTORICAL_BASELINE['direct_match_accuracy']:.1%} |",
        f"| Hungarian Mapping Accuracy | {HISTORICAL_BASELINE['hungarian_mapping_accuracy']:.1%} |",
        f"| Majority Vote Purity | {HISTORICAL_BASELINE['majority_vote_purity']:.1%} |",
        f"| Normalized Mutual Information | {HISTORICAL_BASELINE['normalized_mutual_information']:.4f} |",
        "",
        "## Comparability Limitations",
        "",
        "The historical golden set was created for an earlier corpus and is evaluated here on a dedicated eligible golden-set corpus, not by intersecting it with the first N articles of the current internal benchmark.",
        "Historical database-cluster results and fresh BERTopic results are not directly comparable because corpus eligibility, predictions, and clustering assignments differ.",
        "",
    ]

    report = "\n".join([
        "# BERTopic Clustering Benchmark Report",
        "",
        f"- Generated at: {summary['generated_at_utc']}",
        f"- Articles benchmarked: {summary['article_count']}",
        f"- Embedded articles considered: {summary['article_count']}",
        f"- Non-outlier clusters: {summary['cluster_count']}",
        f"- Outlier ratio: {summary['outlier_ratio']:.4f}",
        f"- Cluster assignment coverage: {summary['cluster_assignment_coverage']:.4f}",
        "",
        "## Internal Metrics By Representation",
        "",
        "All internal metrics use Euclidean distance. The normalized embedding space is cosine-equivalent for these comparisons.",
        "",
        *metric_lines,
        f"- Topic Diversity: {summary['topic_diversity']:.4f}",
        "",
        *external_report_lines,
        "## Interpretation",
        "",
        "- The internal benchmark was run on the live CS arXiv corpus in the project database; the external benchmark uses a separate golden-set-compatible corpus.",
        "- UMAP-space metrics evaluate the representation used by HDBSCAN; normalized-embedding metrics evaluate semantic geometry independently of UMAP.",
        "- Higher silhouette, Calinski-Harabasz, and Dunn values are better, while lower Davies-Bouldin values are better within the same representation.",
        "- Calinski-Harabasz values must not be compared directly across representations because they depend on representation scale and geometry.",
        "- Dunn is sensitive to a single extreme within-cluster pair; it should be interpreted alongside silhouette and Davies-Bouldin.",
        "- The outlier ratio remains a useful operational signal because BERTopic labels some documents as -1 when they do not fit a stable cluster.",
        "- External metrics measure agreement with the golden reference topics and must not be treated as another internal geometry score.",
        "- Historical values use database cluster assignments, while this run uses fresh BERTopic topic assignments; direct improvement claims are invalid without matching preprocessing and assignments.",
        "- The external configuration was selected from a finite 15-combination comparison on this golden set; external metric selection therefore carries dataset-selection bias and is not a generalization estimate.",
        "- The golden labels are much finer-grained than the resulting BERTopic topics, so low external agreement reflects both clustering behavior and label-granularity mismatch.",
        "- No overall clustering score is reported.",
        "- BERTopic topic visualizations were saved as HTML files so the cluster landscape can be reviewed interactively.",
        "- A finite 15-configuration external comparison was run; no unbounded hyperparameter sweep was performed.",
        "",
        "## Generated Artifacts",
        "",
        "- benchmark_summary.json",
        "- topic_info.csv",
        "- paper_topic_assignments.csv",
        "- golden_paper_topic_assignments.csv",
        "- bertopic_topics.html",
        "- bertopic_barchart.html",
        "- bertopic_hierarchy.html",
        "- bertopic_model",
        "- golden_bertopic_model",
        "- golden_topic_info.csv",
    ])
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Saved benchmark outputs to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
