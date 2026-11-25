"""V2 检索融合、证据门控拒答与引用核验单元测试（完全离线可运行）。"""

from types import SimpleNamespace

from ecom_rag.config import Settings
from ecom_rag.embeddings import DeterministicHashEmbeddingProvider
from ecom_rag.graph import AgenticRAGService, evaluate_evidence_gate, verify_citations
from ecom_rag.retriever import RetrievalResult, RetrievedChunk, rrf_fuse_and_rerank
from ecom_rag.schemas import Product


def test_rrf_fuse_prioritizes_both_semantic_and_bm25_hits() -> None:
    """同时在语义和 BM25 靠前且已验证购买的高质量评论应排在首位。"""
    chunks = [
        RetrievedChunk(
            chunk_id="c1", sku_id="SKU1", review_id="R1", content="普通好评",
            rating=5, verified_purchase=False, helpful_votes=0, quality_score=0.3, vector_score=0.72,
        ),
        RetrievedChunk(
            chunk_id="c2", sku_id="SKU1", review_id="R2", content="2.4GHz 配网实测很稳定",
            rating=5, verified_purchase=True, helpful_votes=30, quality_score=0.95, vector_score=0.70,
        ),
    ]
    bm25_ranked = [("c2", 4.5), ("c1", 0.0)]
    fused = rrf_fuse_and_rerank(chunks, bm25_ranked, top_k=2)
    assert fused[0].chunk_id == "c2"
    assert fused[0].bm25_score == 4.5


def test_evidence_gate_abstains_when_zero_reviews_for_experience_question() -> None:
    """无评论商品（如 PET-CARRIER-AIR）被问买家体验时必须触发拒答。"""
    product_dict = {"sku_id": "PET-CARRIER-AIR", "title": "航空透气折叠宠物包", "specs_json": {}}
    ok, reason, _warn = evaluate_evidence_gate("这款宠物包买家实测耐用吗？", product_dict, [])
    assert ok is False
    assert "暂无已核验买家评论" in (reason or "")


def test_citation_verification_catches_hallucinated_review_id() -> None:
    """回答中若伪造未召回的 [R9999]，引用核验必须检出。"""
    chunks = [
        RetrievedChunk(
            chunk_id="c1", sku_id="SKU1", review_id="R1001", content="很静音",
            rating=5, verified_purchase=True, helpful_votes=10, quality_score=0.8,
        )
    ]
    cited, invalid, is_valid = verify_citations("实测很静音 [R1001]，且不漏水 [R9999]。", chunks)
    assert cited == ["R1001", "R9999"]
    assert invalid == ["R9999"]
    assert is_valid is False


def test_langgraph_v2_abstain_short_circuits_llm() -> None:
    """验证证据不足时 LangGraph 直接走 abstain 分支，不调用大模型（0 Token）。"""

    class DummyRetriever:
        def retrieve_v2_hybrid_rerank(self, **_kwargs) -> RetrievalResult:
            return RetrievalResult(
                product=Product(sku_id="PET-CARRIER-AIR", title="航空宠物包", category="宠物出行"),
                chunks=[],
                retrieve_ms=2,
            )

    class FailIfCalledLLM:
        def __getattr__(self, name):
            raise AssertionError(f"证据不足时不应调用 LLM: {name}")

    rag = AgenticRAGService(
        settings=Settings(),
        retriever=DummyRetriever(),  # type: ignore[arg-type]
        embedder=DeterministicHashEmbeddingProvider(dim=64),
        llm_client=FailIfCalledLLM(),  # type: ignore[arg-type]
    )
    state = rag.ask(question="买家实测这款包透气性和耐用度怎么样？", sku_id="PET-CARRIER-AIR")
    assert state["abstained"] is True
    assert state["prompt_tokens"] == 0
    assert "资料不足" in state["answer"]


def test_langgraph_v2_generate_and_verify_with_stub_llm() -> None:
    """验证证据充分时走 generate -> verify_citations，并正确记录 Token。"""

    class DummyRetriever:
        def retrieve_v2_hybrid_rerank(self, **_kwargs) -> RetrievalResult:
            return RetrievalResult(
                product=Product(sku_id="PET-FOUNTAIN-PRO", title="静音饮水机Pro", category="饮水"),
                chunks=[
                    RetrievedChunk(
                        chunk_id="R1001_c0", sku_id="PET-FOUNTAIN-PRO", review_id="R1001",
                        content="放卧室床头实测几乎听不到马达声音",
                        rating=5, verified_purchase=True, helpful_votes=42,
                        quality_score=0.92, vector_score=0.85, bm25_score=2.1, final_score=0.88,
                    )
                ],
                retrieve_ms=3,
            )

    class StubLLM:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **_kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="买家实测放卧室几乎听不到声音 [R1001]。"))],
                usage=SimpleNamespace(prompt_tokens=120, completion_tokens=25),
            )

    rag = AgenticRAGService(
        settings=Settings(),
        retriever=DummyRetriever(),  # type: ignore[arg-type]
        embedder=DeterministicHashEmbeddingProvider(dim=64),
        llm_client=StubLLM(),  # type: ignore[arg-type]
    )
    state = rag.ask(question="放卧室吵不吵？", sku_id="PET-FOUNTAIN-PRO")
    assert state["abstained"] is False
    assert state["citation_valid"] is True
    assert state["cited_review_ids"] == ["R1001"]
    assert state["prompt_tokens"] == 120


def test_frontend_chinese_ui_served() -> None:
    """验证根路径处理函数返回存在且包含完整中文工作台的 HTML 文件。"""
    from pathlib import Path
    from ecom_rag.api import serve_frontend

    resp = serve_frontend()
    html_path = Path(resp.path)
    assert html_path.is_file()
    html = html_path.read_text(encoding="utf-8")
    assert "跨境电商商品评论 Agentic RAG 智能问答工作台" in html
    assert "V1 / V2 同屏双栏对比" in html
