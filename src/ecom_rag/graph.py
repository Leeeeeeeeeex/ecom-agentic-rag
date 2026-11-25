"""LangGraph Agentic RAG 状态机（检索 → 证据充分性门控 → 拒答或结构化带引用生成 → 引用核验）。"""

import json
import re
import time
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from ecom_rag.config import Settings
from ecom_rag.embeddings import EmbeddingProvider
from ecom_rag.retriever import RetrievalResult, RetrievedChunk, ReviewRetriever

_CITATION_PATTERN = re.compile(r"\[(R\d+)\]")

# 显式区分 V1 基线与 V2 生产 Prompt（便于 Langfuse 与 RAGAS 对比）
PROMPT_V1_SYSTEM = (
    "你是电商售前客服助手。请根据下方参考评论回答买家问题，语气友好简洁。"
)

PROMPT_V2_SYSTEM = """你是跨境电商资深售前顾问，你的回答必须严格基于本轮提供的【官方商品规格】与【已核验买家评论】，严禁编造或借用其他型号的参数。

强制输出规则：
1. 将「官方参数说明」与「买家实测反馈」分开表述，不要混淆页面宣传与真实体验。
2. 每一条来自买家评论的事实或体验结论，必须在句末标注对应评论编号，例如 [R1001]。
3. 严禁引用本轮上下文中未出现的评论编号；如当前证据未覆盖问题中的某个细节，必须明确说明“现有资料与评论暂未提及该细节，建议联系人工确认”。
4. 若同一问题既有正面体验也有负面注意点（如水垢清理、电池模式休眠），必须客观同时呈现，避免片面承诺。"""


class RAGState(TypedDict, total=False):
    """LangGraph 状态定义。"""

    question: str
    sku_id: str | None
    pipeline_version: Literal["v1_vector", "v2_agentic_hybrid"]
    top_k: int

    # 检索与门控中间态
    embed_ms: int
    retrieve_ms: int
    generate_ms: int
    product_dict: dict | None
    chunks: list[RetrievedChunk]
    evidence_sufficient: bool
    abstain_reason: str | None
    low_sample_warning: str | None

    # 输出与审计字段
    answer: str
    abstained: bool
    cited_review_ids: list[str]
    invalid_citations: list[str]
    citation_valid: bool
    prompt_tokens: int
    completion_tokens: int


def verify_citations(answer: str, retrieved_chunks: list[RetrievedChunk]) -> tuple[list[str], list[str], bool]:
    """核验回答中的 [Rxxxx] 引用是否全部来自本轮真实召回切片。"""
    valid_ids = {c.review_id for c in retrieved_chunks}
    cited = list(dict.fromkeys(_CITATION_PATTERN.findall(answer or "")))
    invalid = [cid for cid in cited if cid not in valid_ids]
    is_valid = len(invalid) == 0
    return cited, invalid, is_valid


def evaluate_evidence_gate(
    question: str,
    product_dict: dict | None,
    chunks: list[RetrievedChunk],
    min_vector_score: float = 0.22,
    min_final_score: float = 0.25,
) -> tuple[bool, str | None, str | None]:
    """V2 证据充分性门控：判断是否足以回答、是否必须拒答、是否提示样本偏少。"""
    if product_dict is None and not chunks:
        return False, "未查询到对应商品资料，且无任何相关买家评论证据。", None

    if not chunks:
        # 有商品主数据但零评论：如果问的是纯使用体验/评价，必须拒答避免脑补
        experience_terms = ("评价", "反馈", "实测", "买家", "差评", "口碑", "用起来", "吵不吵", "耐用", "卡不卡")
        if any(term in question for term in experience_terms):
            return (
                False,
                f"商品 {product_dict['sku_id']}（{product_dict['title']}）目前暂无已核验买家评论，无法基于真实使用体验给出结论。",
                None,
            )
        # 仅询问规格参数时允许依据官方 specs_json 回答，但标注无评论
        return True, None, "当前商品暂无买家评论，仅能依据官方规格参数回答。"

    top_vec = max(c.vector_score for c in chunks)
    top_final = max(c.final_score for c in chunks)
    top_bm25 = max(c.bm25_score for c in chunks)

    # 若语义分、综合分和词法分同时过低，说明召回内容与问题无关，触发拒答
    if top_vec < min_vector_score and top_final < min_final_score and top_bm25 <= 0.0:
        return (
            False,
            "已检索该商品评论，但未找到与当前问题直接相关的有效买家证据（相关性低于阈值）。",
            None,
        )

    unique_reviews = {c.review_id for c in chunks}
    warning = None
    if len(unique_reviews) < 2:
        warning = f"当前仅命中 {len(unique_reviews)} 条独立买家评论，样本量较少，请谨慎参考。"
    return True, None, warning


def _build_user_context(question: str, product_dict: dict | None, chunks: list[RetrievedChunk]) -> str:
    """将商品规格与买家评论分开拼装，防止模型混淆官方参数与主观反馈。"""
    sections: list[str] = []
    if product_dict:
        specs_str = json.dumps(product_dict.get("specs_json", {}), ensure_ascii=False, indent=2)
        sections.append(
            f"【官方商品规格】\n"
            f"- SKU: {product_dict['sku_id']}\n"
            f"- 标题: {product_dict['title']}\n"
            f"- 价格: ${product_dict.get('price_usd')}\n"
            f"- 是否为配件: {'是' if product_dict.get('is_accessory') else '否'}\n"
            f"- 规格参数:\n{specs_str}"
        )
    else:
        sections.append("【官方商品规格】\n未指定或未找到对应商品主数据。")

    if chunks:
        review_lines = []
        for c in chunks:
            vp = "已验证购买" if c.verified_purchase else "未验证"
            review_lines.append(
                f"- [{c.review_id}] (SKU={c.sku_id}, 评分={c.rating}星, {vp}, 有用票={c.helpful_votes}): {c.content}"
            )
        sections.append("【已核验买家评论证据】\n" + "\n".join(review_lines))
    else:
        sections.append("【已核验买家评论证据】\n暂无匹配评论。")

    sections.append(f"【买家/导购问题】\n{question}")
    return "\n\n".join(sections)


class AgenticRAGService:
    """封装 V1 与 V2 LangGraph 编排的问答核心引擎。"""

    def __init__(
        self,
        settings: Settings,
        retriever: ReviewRetriever,
        embedder: EmbeddingProvider,
        llm_client: OpenAI | None = None,
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._embedder = embedder
        self._llm_client = llm_client
        self._graph = self._compile_graph()

    def _get_llm(self) -> OpenAI:
        if self._llm_client is not None:
            return self._llm_client
        if not self._settings.llm_api_key or "please-replace" in self._settings.llm_api_key:
            raise ValueError("请先在 .env 中填写有效的 LLM_API_KEY。")
        self._llm_client = OpenAI(
            api_key=self._settings.llm_api_key,
            base_url=self._settings.llm_base_url,
        )
        return self._llm_client

    def _node_retrieve(self, state: RAGState) -> RAGState:
        question = state["question"]
        sku_id = state.get("sku_id")
        version = state.get("pipeline_version", "v2_agentic_hybrid")
        top_k = state.get("top_k", 5)

        vectors, embed_ms = self._embedder.embed_texts([question])
        query_vec = vectors[0]

        res: RetrievalResult
        if version == "v1_vector":
            res = self._retriever.retrieve_v1_pure_vector(query_vec, sku_id=sku_id, top_k=top_k)
        else:
            res = self._retriever.retrieve_v2_hybrid_rerank(
                question=question,
                query_vec=query_vec,
                sku_id=sku_id,
                exclude_accessories=True,
                top_k=top_k,
            )

        return {
            "embed_ms": embed_ms,
            "retrieve_ms": res.retrieve_ms,
            "product_dict": res.product.model_dump() if res.product else None,
            "chunks": res.chunks,
        }

    def _node_grade_evidence(self, state: RAGState) -> RAGState:
        version = state.get("pipeline_version", "v2_agentic_hybrid")
        # V1 基线不做严格证据门控，直接放行生成（用于对比 V1 幻觉率与 V2 拒答精度）
        if version == "v1_vector":
            return {
                "evidence_sufficient": True,
                "abstain_reason": None,
                "low_sample_warning": None,
            }

        ok, reason, warning = evaluate_evidence_gate(
            question=state["question"],
            product_dict=state.get("product_dict"),
            chunks=state.get("chunks", []),
        )
        return {
            "evidence_sufficient": ok,
            "abstain_reason": reason,
            "low_sample_warning": warning,
        }

    def _route_after_grade(self, state: RAGState) -> Literal["abstain", "generate"]:
        return "generate" if state.get("evidence_sufficient", False) else "abstain"

    def _node_abstain(self, state: RAGState) -> RAGState:
        reason = state.get("abstain_reason") or "当前资料与评论证据不足，无法给出可靠回答。"
        answer = f"【资料不足·暂无法确认】{reason} 建议补充商品具体型号（SKU）或转人工客服进一步核实，避免基于无关商品猜测。"
        return {
            "answer": answer,
            "abstained": True,
            "generate_ms": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cited_review_ids": [],
            "invalid_citations": [],
            "citation_valid": True,
        }

    def _node_generate(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        version = state.get("pipeline_version", "v2_agentic_hybrid")
        system_prompt = PROMPT_V1_SYSTEM if version == "v1_vector" else PROMPT_V2_SYSTEM
        user_content = _build_user_context(
            question=state["question"],
            product_dict=state.get("product_dict"),
            chunks=state.get("chunks", []),
        )

        client = self._get_llm()
        resp = client.chat.completions.create(
            model=self._settings.llm_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        answer = (resp.choices[0].message.content or "").strip()
        if not answer:
            answer = "模型暂时未返回有效内容，请稍后重试或简化问题后再次咨询。"

        if state.get("low_sample_warning"):
            answer = f"{answer}\n\n> 提示：{state['low_sample_warning']}"

        usage = resp.usage
        return {
            "answer": answer,
            "abstained": False,
            "generate_ms": int((time.perf_counter() - t0) * 1000),
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }

    def _node_verify_citations(self, state: RAGState) -> RAGState:
        cited, invalid, is_valid = verify_citations(
            answer=state.get("answer", ""),
            retrieved_chunks=state.get("chunks", []),
        )
        answer = state.get("answer", "")
        if not is_valid and state.get("pipeline_version") == "v2_agentic_hybrid":
            answer += f"\n\n> [系统核验提醒] 检测到未在本轮召回中的引用编号 {invalid}，请以附带评论原文为准。"
        return {
            "answer": answer,
            "cited_review_ids": cited,
            "invalid_citations": invalid,
            "citation_valid": is_valid,
        }

    def _compile_graph(self):
        builder = StateGraph(RAGState)
        builder.add_node("retrieve", self._node_retrieve)
        builder.add_node("grade_evidence", self._node_grade_evidence)
        builder.add_node("abstain", self._node_abstain)
        builder.add_node("generate", self._node_generate)
        builder.add_node("verify_citations", self._node_verify_citations)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "grade_evidence")
        builder.add_conditional_edges(
            "grade_evidence",
            self._route_after_grade,
            {"abstain": "abstain", "generate": "generate"},
        )
        builder.add_edge("abstain", END)
        builder.add_edge("generate", "verify_citations")
        builder.add_edge("verify_citations", END)
        return builder.compile()

    def ask(
        self,
        question: str,
        sku_id: str | None = None,
        pipeline_version: Literal["v1_vector", "v2_agentic_hybrid"] = "v2_agentic_hybrid",
        top_k: int = 5,
    ) -> RAGState:
        """执行完整问答状态图并返回全量状态与指标。"""
        return self._graph.invoke(
            {
                "question": question,
                "sku_id": sku_id,
                "pipeline_version": pipeline_version,
                "top_k": top_k,
            }
        )
