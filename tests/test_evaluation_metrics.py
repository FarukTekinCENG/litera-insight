import asyncio
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.evaluation.clustering_metrics import evaluate_clustering_records
from backend.app.evaluation.report_writer import write_evaluation_report
from backend.app.evaluation.retrieval_metrics import (
    evaluate_retrieval_questions,
    load_golden_questions,
    score_retrieval,
    summarize_retrieval_results,
)
from backend.app.evaluation.rag_golden_set import (
    calculate_metrics,
    extract_cited_article_ids,
    load_validated_golden_set,
)
from backend.app.evaluation.schemas import GoldenQuestion
from backend.app.schemas.retrieval import RetrievedArticle
from backend.app.schemas.source import SourceReference
from scripts.run_current_clustering_benchmark import (
    compute_external_metrics,
    compute_dunn_index,
    load_reference_labels,
    resolve_clustering_config,
    resolve_external_clustering_config,
    summarize_all_spaces,
)
from scripts.run_evaluation import main as run_evaluation_main


def test_large_corpus_clustering_tuning_uses_sampled_config():
    cfg = resolve_clustering_config(30000)

    assert cfg["max_articles"] == 12000
    assert cfg["min_topic_size"] >= 75
    assert cfg["umap_n_neighbors"] <= 30
    assert cfg["umap_min_dist"] >= 0.05


def test_benchmark_reports_umap_and_normalized_embedding_spaces():
    labels = np.array([0, 0, 1, 1, -1])
    embeddings = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.5, 0.5]],
        dtype=np.float32,
    )
    umap_representation = embeddings * 2.0

    result = summarize_all_spaces(labels, embeddings, umap_representation)

    assert set(result) == {"umap", "normalized_embedding"}
    assert result["umap"]["space"] == "umap"
    assert result["normalized_embedding"]["space"] == "normalized_embedding"
    assert result["umap"]["silhouette_score"] is not None
    assert result["normalized_embedding"]["silhouette_score"] is not None


def test_dunn_index_uses_pairwise_cluster_diameter():
    matrix = np.array([[0.0], [3.0], [10.0], [11.0]])
    labels = np.array([0, 0, 1, 1])

    assert compute_dunn_index(matrix, labels) == pytest.approx(7.0 / 3.0)


def test_external_metrics_handle_label_permutation():
    result = compute_external_metrics(
        [1, 2, 3, 4],
        np.array([1, 1, 0, 0]),
        {1: "A", 2: "A", 3: "B", 4: "B"},
    )

    assert result["direct_match_accuracy"] is None
    assert result["hungarian_mapping_accuracy"] == pytest.approx(1.0)
    assert result["majority_vote_purity"] == pytest.approx(1.0)
    assert result["normalized_mutual_information"] == pytest.approx(1.0)
    assert result["adjusted_rand_index"] == pytest.approx(1.0)


def test_external_metrics_calculate_mixed_cluster_purity():
    result = compute_external_metrics(
        [1, 2, 3, 4, 5],
        np.array([0, 0, 0, 0, 1]),
        {1: "A", 2: "A", 3: "A", 4: "B", 5: "B"},
    )

    assert result["majority_vote_purity"] == pytest.approx(4 / 5)


def test_external_metrics_exclude_outliers_and_missing_references():
    result = compute_external_metrics(
        [1, 2, 3, 99],
        np.array([0, 0, -1, 1]),
        {1: "A", 2: "A", 3: "B", 4: "C"},
    )

    assert result["golden_set_article_count"] == 4
    assert result["matched_article_count"] == 3
    assert result["evaluation_sample_count"] == 2
    assert result["outlier_count"] == 1
    assert result["outlier_handling"] == "excluded"


def test_reference_loader_deduplicates_and_excludes_conflicts():
    rows = [
        {"article_id": "1", "title": "A", "cluster_description": "Topic A"},
        {"article_id": "1", "title": "A", "cluster_description": "Topic A"},
        {"article_id": "2", "title": "B", "cluster_description": "Topic B"},
        {"article_id": "2", "title": "B", "cluster_description": "Topic C"},
    ]

    labels, ambiguous_ids, counts = load_reference_labels(rows)

    assert labels == {1: "Topic A"}
    assert ambiguous_ids == {2}
    assert counts == {1: 2, 2: 2}


def test_external_metrics_return_safe_empty_result():
    result = compute_external_metrics([], np.array([], dtype=np.int64), {})

    assert result["evaluation_sample_count"] == 0
    assert result["hungarian_mapping_accuracy"] == 0.0
    assert result["normalized_mutual_information"] == 0.0


def test_external_config_is_bounded_for_small_golden_corpus():
    args = SimpleNamespace(
        min_topic_size=100,
        min_samples=20,
        umap_n_neighbors=30,
        umap_n_components=10,
        umap_min_dist=0.05,
    )

    config = resolve_external_clustering_config(args, 44)

    assert config["min_topic_size"] == 5
    assert config["min_samples"] == 2
    assert config["umap_n_neighbors"] == 30


def test_retrieval_metric_math_and_duplicate_deduplication():
    scores = score_retrieval(
        expected_article_ids=[1, 2],
        retrieved_article_ids=[3, 1, 1, 2],
        top_k=3,
    )

    expected_dcg = (1 / math.log2(2 + 1)) + (1 / math.log2(3 + 1))
    expected_idcg = (1 / math.log2(1 + 1)) + (1 / math.log2(2 + 1))

    assert scores["retrieved_article_ids"] == [3, 1, 2]
    assert scores["hit_at_k"] is True
    assert scores["recall_at_k"] == 1.0
    assert scores["precision_at_k"] == pytest.approx(2 / 3)
    assert scores["mrr"] == pytest.approx(1 / 2)
    assert scores["ndcg_at_k"] == pytest.approx(expected_dcg / expected_idcg)


def test_rag_golden_set_citation_and_answer_metrics():
    retrieved = [
        RetrievedArticle(source=SourceReference(source_id="S1", article_id=10, title="Paper 10")),
        RetrievedArticle(source=SourceReference(source_id="S2", article_id=20, title="Paper 20")),
    ]
    cited_ids = extract_cited_article_ids("Paper 10 is relevant [S1]. Paper 20 is background [S2].", retrieved)
    question = GoldenQuestion(
        id="q1",
        question="Find papers about RAG",
        expected_article_ids=[10],
        expected_answer_keywords=["relevant"],
        top_k=2,
    )

    metrics = calculate_metrics(
        question=question,
        retrieved_article_ids=[10, 20],
        cited_article_ids=cited_ids,
        answer="Paper 10 is relevant [S1]. Paper 20 is background [S2].",
        top_k=2,
    )

    assert cited_ids == [10, 20]
    assert metrics["hit_at_1"] is True
    assert metrics["hit_at_3"] is True
    assert metrics["citation_precision"] == pytest.approx(0.5)
    assert metrics["citation_recall"] == pytest.approx(1.0)
    assert metrics["answer_keyword_coverage"] == pytest.approx(1.0)


def test_empty_expected_set_raises_validation_error():
    with pytest.raises(ValueError, match="expected_article_ids"):
        GoldenQuestion(id="bad", question="paper?", expected_article_ids=[])

    with pytest.raises(ValueError, match="expected_article_ids"):
        score_retrieval([], [1], top_k=5)


def test_golden_question_supports_plan_fields_and_question_id_alias():
    question = GoldenQuestion(
        question_id="q-out",
        question="Give me a dinner recipe.",
        expected_article_ids=[],
        requires_retrieval=False,
        question_type="out_of_domain",
        expected_answer_keywords=["academic publication assistant"],
    )

    assert question.id == "q-out"
    assert question.question_type == "out_of_domain"
    assert question.expected_answer_keywords == ["academic publication assistant"]


def test_load_validated_golden_set_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            {
                "questions": [
                    {"id": "q1", "question": "Find RAG papers", "expected_article_ids": [10]},
                    {"id": "q1", "question": "Find dense retrieval papers", "expected_article_ids": [11]},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate golden question ids"):
        load_validated_golden_set(path)


def test_clustering_evaluator_skips_when_fewer_than_two_clusters():
    result = evaluate_clustering_records(
        [
            SimpleNamespace(embedding=[1.0, 0.0], cluster_id=1),
            SimpleNamespace(embedding=[0.9, 0.1], cluster_id=1),
        ]
    )

    assert result.cluster_count == 1
    assert result.cluster_assignment_coverage == 1.0
    assert result.silhouette_score is None
    assert result.skipped_reason == "At least two non-outlier clusters are required."


def test_clustering_outlier_ratio_handles_minus_one_label():
    result = evaluate_clustering_records(
        [
            SimpleNamespace(embedding=[1.0, 0.0], cluster_id=1),
            SimpleNamespace(embedding=[0.9, 0.1], cluster_id=1),
            SimpleNamespace(embedding=[0.0, 1.0], cluster_id=2),
            SimpleNamespace(embedding=[0.1, 0.9], cluster_id=2),
            SimpleNamespace(embedding=[0.5, 0.5], cluster_id=-1),
        ],
        pairwise_sample_limit=10,
    )

    assert result.outlier_count == 1
    assert result.clustered_article_count == 4
    assert result.cluster_count == 2
    assert result.outlier_ratio == pytest.approx(0.2)
    assert result.cluster_assignment_coverage == pytest.approx(0.8)
    assert result.avg_intra_cluster_cosine_similarity is not None
    assert result.avg_centroid_similarity is not None


def test_retrieval_evaluator_scores_fake_retrieval_service():
    question = GoldenQuestion(
        id="q1",
        question="Find papers about RAG",
        expected_article_ids=[10],
        top_k=3,
    )

    class FakeEmbeddingService:
        def embed_query(self, query: str):
            assert query
            return [0.1, 0.2]

    class FakeRetrievalService:
        def retrieve(self, query_embedding, filters, top_k, sort_by, query_text):
            assert query_embedding == [0.1, 0.2]
            assert query_text == "Find papers about RAG"
            assert top_k == 3
            assert sort_by == "relevance"
            return [
                _retrieved_article(30),
                _retrieved_article(10),
                _retrieved_article(10),
            ]

    results = asyncio.run(
        evaluate_retrieval_questions(
            db=SimpleNamespace(),
            questions=[question],
            top_k=5,
            embedding_service=FakeEmbeddingService(),
            retrieval_service=FakeRetrievalService(),
        )
    )

    assert len(results) == 1
    assert results[0].retrieved_article_ids == [30, 10]
    assert results[0].hit_at_k is True
    assert results[0].mrr == pytest.approx(0.5)
    assert results[0].uses_rag is True
    assert results[0].source_count == 3
    assert results[0].rewritten_query == "Find papers about RAG"
    assert results[0].sort_by == "relevance"


def test_retrieval_evaluator_can_force_rag_for_router_isolation():
    question = GoldenQuestion(
        id="q1",
        question="Explain this without retrieval",
        expected_article_ids=[10],
        top_k=1,
    )

    class FakeEmbeddingService:
        def embed_query(self, query: str):
            return [0.1, 0.2]

    class FakeRetrievalService:
        def retrieve(self, query_embedding, filters, top_k, sort_by, query_text):
            assert query_text == "Explain this without retrieval"
            return [_retrieved_article(10)]

    results = asyncio.run(
        evaluate_retrieval_questions(
            db=SimpleNamespace(),
            questions=[question],
            top_k=1,
            embedding_service=FakeEmbeddingService(),
            retrieval_service=FakeRetrievalService(),
            force_rag=True,
        )
    )

    assert results[0].uses_rag is True
    assert results[0].hit_at_k is True
    assert "Forced RAG" in results[0].route_reason


def test_retrieval_evaluator_can_disable_keyword_branch():
    question = GoldenQuestion(
        id="q1",
        question="Find papers about dense retrieval",
        expected_article_ids=[10],
        top_k=1,
    )

    class FakeEmbeddingService:
        def embed_query(self, query: str):
            return [0.1, 0.2]

    class FakeRetrievalService:
        def retrieve(self, query_embedding, filters, top_k, sort_by, query_text):
            assert query_text is None
            return [_retrieved_article(10)]

    results = asyncio.run(
        evaluate_retrieval_questions(
            db=SimpleNamespace(),
            questions=[question],
            top_k=1,
            embedding_service=FakeEmbeddingService(),
            retrieval_service=FakeRetrievalService(),
            force_rag=True,
            use_keyword=False,
        )
    )

    assert results[0].hit_at_k is True


def test_retrieval_evaluator_passes_mode_options_to_service(monkeypatch):
    captured = {}

    class FakeEmbeddingService:
        def embed_query(self, query: str):
            return [0.1, 0.2]

    class FakeRetrievalService:
        def __init__(
            self,
            db,
            retrieval_mode,
            fusion_method,
            vector_top_k,
            bm25_top_k,
            final_top_k,
        ):
            captured.update(
                {
                    "retrieval_mode": retrieval_mode,
                    "fusion_method": fusion_method,
                    "vector_top_k": vector_top_k,
                    "bm25_top_k": bm25_top_k,
                    "final_top_k": final_top_k,
                }
            )

        def retrieve(self, query_embedding, filters, top_k, sort_by, query_text):
            return [_retrieved_article(10)]

    monkeypatch.setattr("backend.app.evaluation.retrieval_metrics.RetrievalService", FakeRetrievalService)

    results = asyncio.run(
        evaluate_retrieval_questions(
            db=SimpleNamespace(),
            questions=[
                GoldenQuestion(
                    id="q1",
                    question="Find papers about RAG",
                    expected_article_ids=[10],
                    top_k=1,
                )
            ],
            top_k=1,
            embedding_service=FakeEmbeddingService(),
            force_rag=True,
            retrieval_mode="bm25",
            fusion_method="weighted",
            vector_top_k=11,
            bm25_top_k=12,
            final_top_k=3,
        )
    )

    assert captured == {
        "retrieval_mode": "bm25",
        "fusion_method": "weighted",
        "vector_top_k": 11,
        "bm25_top_k": 12,
        "final_top_k": 3,
    }
    assert results[0].retrieval_mode == "bm25"
    assert results[0].fusion_method == "weighted"


def test_report_writer_creates_summary_json_and_retrieval_csv(tmp_path):
    result = asyncio.run(
        evaluate_retrieval_questions(
            db=SimpleNamespace(),
            questions=[
                GoldenQuestion(
                    id="q1",
                    question="Find papers about RAG",
                    expected_article_ids=[10],
                    top_k=2,
                )
            ],
            top_k=2,
            embedding_service=SimpleNamespace(embed_query=lambda query: [0.1, 0.2]),
            retrieval_service=SimpleNamespace(
                retrieve=lambda query_embedding, filters, top_k, sort_by, query_text: [_retrieved_article(10)]
            ),
        )
    )
    summary = summarize_retrieval_results(result)

    run_dir = write_evaluation_report(
        output_root=tmp_path,
        run_id="test-run",
        suite="retrieval",
        retrieval_results=result,
        retrieval_summary=summary,
    )

    assert (run_dir / "summary.json").exists()
    assert (run_dir / "retrieval_results.csv").exists()
    csv_text = (run_dir / "retrieval_results.csv").read_text(encoding="utf-8")
    assert "rewritten_query" in csv_text
    assert "filters" in csv_text
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["retrieval"]["summary"]["question_count"] == 1


def test_load_golden_questions_rejects_malformed_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed golden questions JSON"):
        load_golden_questions(path)

    assert run_evaluation_main(["--suite", "retrieval", "--golden-file", str(path)]) == 2


def _retrieved_article(article_id: int) -> RetrievedArticle:
    return RetrievedArticle(
        source=SourceReference(
            source_id=f"S{article_id}",
            article_id=article_id,
            title=f"Paper {article_id}",
        ),
        abstract_text="Abstract",
    )
