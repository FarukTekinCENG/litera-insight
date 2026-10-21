from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from ai_engine.clustering.ClusterFunctions import Cluster, resolve_runtime_profile
from backend.app.evaluation.cluster_postprocess import (
    DEFAULT_CLUSTER_MERGE_THRESHOLD,
    DEFAULT_OUTLIER_REASSIGNMENT_THRESHOLD,
)
from backend.app.evaluation.clustering_experiments import (
    ClusteringExperimentConfig,
    select_best_experiment,
    write_experiment_artifacts,
)


DEFAULT_MIN_TOPIC_SIZES = [50, 75, 100]
DEFAULT_MIN_SAMPLES = [10, 15, 20]
DEFAULT_UMAP_N_NEIGHBORS = [15, 25, 30]
DEFAULT_UMAP_N_COMPONENTS = [5, 10]
DEFAULT_UMAP_MIN_DISTS = [0.05, 0.1]
DEFAULT_CLUSTER_SELECTION_METHODS = ["eom"]


def resolve_clustering_config(max_articles: int | None) -> dict:
    if max_articles is None:
        max_articles = 30000
    if max_articles >= 20000:
        return {
            "max_articles": 12000,
            "min_topic_sizes": [75, 100, 150],
            "min_samples": [15, 20, 30],
            "umap_n_neighbors": [15, 25, 30],
            "umap_n_components": [5, 10],
            "umap_min_dists": [0.05, 0.1],
        }
    if max_articles >= 10000:
        return {
            "max_articles": min(max_articles, 12000),
            "min_topic_sizes": [50, 75, 100],
            "min_samples": [10, 15, 20],
            "umap_n_neighbors": [15, 25, 30],
            "umap_n_components": [5, 10],
            "umap_min_dists": [0.05, 0.1],
        }
    return {
        "max_articles": max_articles,
        "min_topic_sizes": [25, 50, 75],
        "min_samples": [5, 10, 15],
        "umap_n_neighbors": [15, 25, 30],
        "umap_n_components": [5, 10],
        "umap_min_dists": [0.05, 0.1],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BERTopic clustering experiment grid without DB persistence.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "exports/clustering_experiments")
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--clean-papers-csv", action="append", type=Path, default=None)
    parser.add_argument("--raw-db", action="store_true")
    parser.add_argument("--include-openalex", action="store_true")
    parser.add_argument("--hardware-profile", default=None, choices=["auto", "m4-pro-24gb", "balanced", "memory-saver"])
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--outlier-threshold", type=float, default=DEFAULT_OUTLIER_REASSIGNMENT_THRESHOLD)
    parser.add_argument("--merge-threshold", type=float, default=DEFAULT_CLUSTER_MERGE_THRESHOLD)
    parser.add_argument("--pairwise-sample-limit", type=int, default=500)
    parser.add_argument("--min-topic-size", type=int, action="append", default=None)
    parser.add_argument("--min-samples", type=int, action="append", default=None)
    parser.add_argument("--umap-n-neighbors", type=int, action="append", default=None)
    parser.add_argument("--umap-n-components", type=int, action="append", default=None)
    parser.add_argument("--umap-min-dist", type=float, action="append", default=None)
    parser.add_argument("--cluster-selection-method", choices=["eom", "leaf"], action="append", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    runtime_profile = resolve_runtime_profile(args.hardware_profile, args.threads)
    articles = _load_articles(args)
    if not articles:
        print("No articles available for clustering experiments.", file=sys.stderr)
        return 2

    embeddings = np.array([Cluster._article_embedding(article) for article in articles], dtype=np.float32)
    docs = np.array([Cluster._document_text(article) for article in articles])
    article_rows = [_article_row(article) for article in articles]

    summaries: list[dict] = []
    for index, config in enumerate(_experiment_configs(args), start=1):
        experiment_dir = run_dir / config.name
        print(f"[{index}] Running {config.name}")
        topic_model = Cluster._build_topic_model(
            min_topic_size=config.min_topic_size,
            min_samples=config.min_samples,
            umap_n_neighbors=config.umap_n_neighbors,
            umap_n_components=config.umap_n_components,
            umap_min_dist=config.umap_min_dist,
            cluster_selection_method=config.cluster_selection_method,
            runtime_profile=runtime_profile,
        )
        with Cluster._threadpool_limits(runtime_profile):
            topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

        Cluster._write_topic_outputs(
            output_dir=experiment_dir,
            articles=articles,
            topic_model=topic_model,
            topics=topics,
            probs=probs,
            experiment_results=[Cluster._evaluate_topic_model(topic_model, topics, config.name)],
            save_model=not args.no_save_model,
        )
        summary = write_experiment_artifacts(
            output_dir=experiment_dir,
            config=config,
            article_rows=article_rows,
            embeddings=embeddings,
            labels=topics,
            outlier_threshold=args.outlier_threshold,
            merge_threshold=args.merge_threshold,
            pairwise_sample_limit=args.pairwise_sample_limit,
        )
        summaries.append(summary)

    best = select_best_experiment(summaries)
    _write_metrics_csv(run_dir / "metrics.csv", summaries)
    (run_dir / "experiment_summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "generated_at": datetime.now(UTC).isoformat(),
                "experiment_count": len(summaries),
                "best_experiment": best,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Clustering experiment report written to {run_dir}")
    return 0


def _load_articles(args: argparse.Namespace):
    if args.raw_db:
        from database.db import SessionLocal
        from database.models.ArticleData import Article
        from sqlalchemy import and_, or_

        db = SessionLocal()
        try:
            cs_category_filter = or_(
                Article.primary_category.ilike("cs.%"),
                Article.categories.ilike("%cs.%"),
            )
            query = (
                db.query(Article)
                .filter(Article.embedding.isnot(None), Article.title.isnot(None), Article.abstract_text.isnot(None))
                .order_by(Article.id.asc())
            )
            if args.include_openalex:
                query = query.filter(or_(and_(Article.source == "arxiv", cs_category_filter), Article.source == "openalex"))
            else:
                query = query.filter(Article.source == "arxiv", cs_category_filter)
            if args.max_articles:
                query = query.limit(args.max_articles)
            return [
                article
                for article in query.all()
                if Cluster._valid_article_for_clustering(article)
                and Cluster._article_in_clustering_scope(article, include_openalex=args.include_openalex)
            ]
        finally:
            db.close()

    csv_paths = args.clean_papers_csv or Cluster._existing_clean_paper_csvs()
    return Cluster._articles_from_clean_csvs(csv_paths, max_articles=args.max_articles)


def _experiment_configs(args: argparse.Namespace) -> list[ClusteringExperimentConfig]:
    configs = []
    for min_topic_size, min_samples, n_neighbors, n_components, min_dist, method in itertools.product(
        args.min_topic_size or DEFAULT_MIN_TOPIC_SIZES,
        args.min_samples or DEFAULT_MIN_SAMPLES,
        args.umap_n_neighbors or DEFAULT_UMAP_N_NEIGHBORS,
        args.umap_n_components or DEFAULT_UMAP_N_COMPONENTS,
        args.umap_min_dist or DEFAULT_UMAP_MIN_DISTS,
        args.cluster_selection_method or DEFAULT_CLUSTER_SELECTION_METHODS,
    ):
        name = (
            f"mts{min_topic_size}_ms{min_samples}_"
            f"nn{n_neighbors}_nc{n_components}_md{str(min_dist).replace('.', 'p')}_{method}"
        )
        configs.append(
            ClusteringExperimentConfig(
                name=name,
                min_topic_size=min_topic_size,
                min_samples=min_samples,
                umap_n_neighbors=n_neighbors,
                umap_n_components=n_components,
                umap_min_dist=min_dist,
                cluster_selection_method=method,
            )
        )
    return configs


def _article_row(article) -> dict:
    return {
        "article_id": getattr(article, "id", None),
        "source": getattr(article, "source", None),
        "external_id": getattr(article, "external_id", None),
        "title": getattr(article, "title", None),
    }


def _write_metrics_csv(path: Path, summaries: list[dict]) -> None:
    rows = []
    for summary in summaries:
        row = {
            **summary["config"],
            **summary["metrics"],
            "outliers_before_reassignment": summary["outliers_before_reassignment"],
            "outliers_after_reassignment": summary["outliers_after_reassignment"],
            "reassigned_outlier_count": summary["reassigned_outlier_count"],
            "reassignment_acceptance_rate": summary["reassignment_acceptance_rate"],
            "cluster_count_before_merge": summary["cluster_count_before_merge"],
            "cluster_count_after_merge": summary["cluster_count_after_merge"],
            "merged_cluster_count": summary["merged_cluster_count"],
        }
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
