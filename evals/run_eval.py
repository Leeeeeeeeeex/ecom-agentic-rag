"""V1（纯向量基线）vs V2（混合检索+证据门控+引用核验）黄金评测集自动化回归脚本。

产出：
1. 确定性检索指标：SKU 纯净度（防串品）、必需评论 Recall@5、配件污染率
2. 安全与护栏指标：拒答准确率、引用合法率（零伪造引用）
3. LLM-as-a-Judge 质量指标（RAGAS 核心口径）：忠实度 Faithfulness、答案相关性 Answer Relevancy
4. 系统与成本指标：P50/P95 延迟（拆解 embed/retrieve/generate）、总 Token 消耗与预估成本
"""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from ecom_rag.config import get_settings
from ecom_rag.embeddings import OpenAICompatibleEmbeddingProvider
from ecom_rag.graph import AgenticRAGService
from ecom_rag.observability import ObservabilityRecorder
from ecom_rag.retriever import ReviewRetriever

JUDGE_SYSTEM_PROMPT = """你是严格的 AI 应用评测裁判（RAGAS 口径）。请根据【问题】【参考答案】【召回上下文】【模型回答】给出两个 0.0~1.0 的分数，并严格返回合法 JSON：
{
  "faithfulness": 0.0到1.0的小数（回答中的事实是否完全忠实于召回上下文与官方规格，未编造、未把其他型号的缺陷安在当前商品上）,
  "answer_relevancy": 0.0到1.0的小数（回答是否准确、完整回应了用户问题，或在资料不足时恰当拒答/说明未覆盖）,
  "reason": "不超过40字的简短评语"
}
只输出 JSON，不要包含 Markdown 代码块或多余文字。"""


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    k = max(0, min(len(s) - 1, math.ceil(len(s) * p) - 1))
    return s[k]


def judge_answer(
    client: OpenAI,
    model: str,
    question: str,
    ground_truth: str,
    contexts: list[str],
    answer: str,
) -> tuple[float, float, str]:
    """使用 LLM-as-a-Judge 按 RAGAS 忠实度与相关性口径打分。"""
    payload = (
        f"【问题】{question}\n"
        f"【参考标准答案】{ground_truth}\n"
        f"【本轮召回上下文】\n" + ("\n".join(contexts) if contexts else "（空）") + "\n"
        f"【模型实际回答】\n{answer}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        return (
            round(float(data.get("faithfulness", 0.0)), 3),
            round(float(data.get("answer_relevancy", 0.0)), 3),
            str(data.get("reason", "")),
        )
    except Exception as exc:
        return 0.0, 0.0, f"裁判解析失败: {exc}"


def evaluate_version(
    version: str,
    dataset: list[dict],
    rag: AgenticRAGService,
    judge_client: OpenAI,
    judge_model: str,
    recorder: ObservabilityRecorder | None = None,
) -> dict:
    """对指定版本运行全量评测集并汇总全部量化指标。"""
    rows: list[dict] = []
    latencies: list[int] = []
    embed_ms_list: list[int] = []
    retrieve_ms_list: list[int] = []
    generate_ms_list: list[int] = []

    total_prompt_tokens = 0
    total_completion_tokens = 0

    sku_purity_scores: list[float] = []
    recall_scores: list[float] = []
    trap_polluted_cnt = 0
    trap_total_cnt = 0
    abstain_correct = 0
    citation_valid_cnt = 0
    faithfulness_scores: list[float] = []
    relevancy_scores: list[float] = []

    for item in dataset:
        t0 = time.perf_counter()
        state = rag.ask(
            question=item["question"],
            sku_id=item["sku_id"],
            pipeline_version=version,  # type: ignore[arg-type]
            top_k=5,
        )
        total_ms = int((time.perf_counter() - t0) * 1000)
        chunks = state.get("chunks", [])
        abstained = bool(state.get("abstained", False))
        answer = state.get("answer", "")
        citation_valid = bool(state.get("citation_valid", True))
        p_tok = int(state.get("prompt_tokens", 0))
        c_tok = int(state.get("completion_tokens", 0))

        latencies.append(total_ms)
        embed_ms_list.append(int(state.get("embed_ms", 0)))
        retrieve_ms_list.append(int(state.get("retrieve_ms", 0)))
        generate_ms_list.append(int(state.get("generate_ms", 0)))
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok

        # 1. SKU 纯净度（召回的评论是否全属于目标 SKU）
        if chunks:
            same_sku = sum(1 for c in chunks if c.sku_id == item["sku_id"])
            sku_purity = same_sku / len(chunks)
            sku_purity_scores.append(sku_purity)
        else:
            # 无评论商品检索为空记为 1.0 纯净（未混入其他商品评论）
            sku_purity = 1.0 if item["expected_abstain"] else 0.0
            sku_purity_scores.append(sku_purity)

        # 2. 跨型号/配件陷阱污染检测
        forbidden = set(item.get("forbidden_sku_ids", []))
        polluted = any(c.sku_id in forbidden for c in chunks)
        if forbidden:
            trap_total_cnt += 1
            if polluted:
                trap_polluted_cnt += 1

        # 3. 必需评论 Recall@5
        expected_ids = set(item.get("expected_review_ids", []))
        retrieved_ids = {c.review_id for c in chunks}
        if expected_ids:
            rec = len(expected_ids & retrieved_ids) / len(expected_ids)
            recall_scores.append(rec)
        else:
            rec = None

        # 4. 拒答准确度与引用合法率
        if abstained == item["expected_abstain"]:
            abstain_correct += 1
        if citation_valid:
            citation_valid_cnt += 1

        # 5. LLM-as-a-Judge 忠实度与相关性
        ctx_texts = [f"[{c.review_id}][SKU={c.sku_id}] {c.content}" for c in chunks]
        if state.get("product_dict"):
            ctx_texts.insert(0, f"[官方规格][SKU={state['product_dict']['sku_id']}] {json.dumps(state['product_dict'].get('specs_json', {}), ensure_ascii=False)}")
        faith, rel, reason = judge_answer(
            client=judge_client,
            model=judge_model,
            question=item["question"],
            ground_truth=item["ground_truth"],
            contexts=ctx_texts,
            answer=answer,
        )
        faithfulness_scores.append(faith)
        relevancy_scores.append(rel)

        # 可选同步上报到数据库与 Langfuse
        if recorder is not None:
            recorder.record_qa(
                pipeline_version=version,
                sku_id=item["sku_id"],
                question=item["question"],
                answer=answer,
                abstained=abstained,
                retrieved_count=len(chunks),
                embed_ms=int(state.get("embed_ms", 0)),
                retrieve_ms=int(state.get("retrieve_ms", 0)),
                generate_ms=int(state.get("generate_ms", 0)),
                total_ms=total_ms,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                citation_valid=citation_valid,
                cache_hit=False,
            )

        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "sku_id": item["sku_id"],
                "question": item["question"],
                "expected_abstain": item["expected_abstain"],
                "actual_abstain": abstained,
                "sku_purity": round(sku_purity, 3),
                "recall_at_5": round(rec, 3) if rec is not None else None,
                "trap_polluted": polluted,
                "citation_valid": citation_valid,
                "faithfulness": faith,
                "answer_relevancy": rel,
                "judge_reason": reason,
                "retrieved_skus": list(dict.fromkeys(c.sku_id for c in chunks)),
                "retrieved_review_ids": [c.review_id for c in chunks],
                "total_ms": total_ms,
                "answer_preview": answer[:180],
            }
        )
        print(
            f"[{version}] {item['id']} ({item['category']}) | purity={sku_purity:.2f} | "
            f"abstain={abstained}/{item['expected_abstain']} | faith={faith:.2f} | rel={rel:.2f} | {total_ms}ms"
        )

    n = len(dataset)
    # DeepSeek-V3 参考单价估算：输入约 ¥2/百万 tokens，输出约 ¥8/百万 tokens
    est_cost_cny_per_1k = (
        (total_prompt_tokens / max(1, n)) * 1000 * 2.0 / 1_000_000
        + (total_completion_tokens / max(1, n)) * 1000 * 8.0 / 1_000_000
    )

    summary = {
        "pipeline_version": version,
        "cases": n,
        "sku_purity_avg": round(sum(sku_purity_scores) / max(1, len(sku_purity_scores)), 4),
        "expected_review_recall_at_5": round(sum(recall_scores) / max(1, len(recall_scores)), 4),
        "cross_model_pollution_rate": round(trap_polluted_cnt / max(1, trap_total_cnt), 4),
        "abstain_accuracy": round(abstain_correct / max(1, n), 4),
        "citation_valid_rate": round(citation_valid_cnt / max(1, n), 4),
        "faithfulness_avg": round(sum(faithfulness_scores) / max(1, n), 4),
        "answer_relevancy_avg": round(sum(relevancy_scores) / max(1, n), 4),
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "avg_embed_ms": round(sum(embed_ms_list) / max(1, n)),
        "avg_retrieve_ms": round(sum(retrieve_ms_list) / max(1, n)),
        "avg_generate_ms": round(sum(generate_ms_list) / max(1, n)),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "est_cost_cny_per_1k_queries": round(est_cost_cny_per_1k, 3),
    }
    return {"summary": summary, "details": rows}


def render_markdown_report(v1: dict, v2: dict) -> str:
    s1, s2 = v1["summary"], v2["summary"]

    def pct(val: float) -> str:
        return f"{val * 100:.1f}%"

    lines = [
        "# V1（纯向量基线）vs V2（混合检索 + 证据门控 + 引用核验）自动化评测报告",
        "",
        f"- 评测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 评测样本数：{s1['cases']} 条（覆盖正常问答、跨型号/配件混淆陷阱、无评论拒答边界、正负权衡场景）",
        "",
        "## 核心指标对比",
        "",
        "| 指标维度 | V1 纯向量基线 | V2 Agentic 混合检索 | 变化幅度 |",
        "| :--- | :---: | :---: | :---: |",
        f"| **SKU 召回纯净度（防串品）** | {pct(s1['sku_purity_avg'])} | **{pct(s2['sku_purity_avg'])}** | +{(s2['sku_purity_avg']-s1['sku_purity_avg'])*100:.1f} pp |",
        f"| **跨型号/配件污染率（越低越好）** | {pct(s1['cross_model_pollution_rate'])} | **{pct(s2['cross_model_pollution_rate'])}** | {(s2['cross_model_pollution_rate']-s1['cross_model_pollution_rate'])*100:.1f} pp |",
        f"| **必需评论 Recall@5** | {pct(s1['expected_review_recall_at_5'])} | **{pct(s2['expected_review_recall_at_5'])}** | +{(s2['expected_review_recall_at_5']-s1['expected_review_recall_at_5'])*100:.1f} pp |",
        f"| **拒答边界准确率** | {pct(s1['abstain_accuracy'])} | **{pct(s2['abstain_accuracy'])}** | +{(s2['abstain_accuracy']-s1['abstain_accuracy'])*100:.1f} pp |",
        f"| **引用合法率（无伪造编号）** | {pct(s1['citation_valid_rate'])} | **{pct(s2['citation_valid_rate'])}** | {(s2['citation_valid_rate']-s1['citation_valid_rate'])*100:+.1f} pp |",
        f"| **忠实度 Faithfulness (Judge)** | {s1['faithfulness_avg']:.3f} | **{s2['faithfulness_avg']:.3f}** | {s2['faithfulness_avg']-s1['faithfulness_avg']:+.3f} |",
        f"| **答案相关性 Answer Relevancy** | {s1['answer_relevancy_avg']:.3f} | **{s2['answer_relevancy_avg']:.3f}** | {s2['answer_relevancy_avg']-s1['answer_relevancy_avg']:+.3f} |",
        f"| **P50 / P95 端到端耗时** | {s1['p50_latency_ms']} / {s1['p95_latency_ms']} ms | {s2['p50_latency_ms']} / {s2['p95_latency_ms']} ms | 分阶段拆解可观测 |",
        f"| **平均阶段耗时 (Embed / Retrieve / LLM)** | {s1['avg_embed_ms']} / {s1['avg_retrieve_ms']} / {s1['avg_generate_ms']} ms | {s2['avg_embed_ms']} / {s2['avg_retrieve_ms']} / {s2['avg_generate_ms']} ms | 拒答短路节省 LLM |",
        f"| **20 题总 Token（入/出）** | {s1['total_prompt_tokens']} / {s1['total_completion_tokens']} | {s2['total_prompt_tokens']} / {s2['total_completion_tokens']} | — |",
        f"| **预估千次问答模型成本（CNY）** | ¥{s1['est_cost_cny_per_1k_queries']:.3f} | ¥{s2['est_cost_cny_per_1k_queries']:.3f} | — |",
        "",
        "## 典型 Badcase 对比（V1 翻车 vs V2 修复）",
        "",
    ]

    v1_map = {r["id"]: r for r in v1["details"]}
    v2_map = {r["id"]: r for r in v2["details"]}
    shown = 0
    for qid, r1 in v1_map.items():
        r2 = v2_map[qid]
        if (r1["trap_polluted"] and not r2["trap_polluted"]) or (
            r1["actual_abstain"] != r1["expected_abstain"] and r2["actual_abstain"] == r2["expected_abstain"]
        ) or (r2["faithfulness"] - r1["faithfulness"] >= 0.2):
            lines.extend(
                [
                    f"### {qid}（{r1['category']} · 目标 SKU: `{r1['sku_id']}`）",
                    f"- **问题**：{r1['question']}",
                    f"- **V1 表现**：召回 SKU `{r1['retrieved_skus']}`，拒答={r1['actual_abstain']}，Faithfulness={r1['faithfulness']}（{r1['judge_reason']}）",
                    f"- **V2 表现**：召回 SKU `{r2['retrieved_skus']}`，拒答={r2['actual_abstain']}，Faithfulness={r2['faithfulness']}（{r2['judge_reason']}）",
                    "",
                ]
            )
            shown += 1
            if shown >= 5:
                break

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 V1 vs V2 黄金评测集回归对比")
    parser.add_argument("--dataset", type=Path, default=Path("evals/golden_dataset.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("evals/reports"))
    parser.add_argument("--record-db", action="store_true", help="将评测链路写入审计表及 Langfuse")
    args = parser.parse_args()

    settings = get_settings()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    embedder = OpenAICompatibleEmbeddingProvider(settings)
    retriever = ReviewRetriever(settings.postgres_dsn)
    rag = AgenticRAGService(settings=settings, retriever=retriever, embedder=embedder)
    judge_client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    recorder = ObservabilityRecorder(settings) if args.record_db else None

    print(f"=== 开始运行黄金评测集（共 {len(dataset)} 题）===")
    v1_res = evaluate_version("v1_vector", dataset, rag, judge_client, settings.llm_model, recorder)
    v2_res = evaluate_version("v2_agentic_hybrid", dataset, rag, judge_client, settings.llm_model, recorder)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "latest_comparison.json"
    md_path = args.out_dir / "v1_vs_v2_report.md"

    json_path.write_text(json.dumps({"v1": v1_res, "v2": v2_res}, ensure_ascii=False, indent=2), encoding="utf-8")
    md_text = render_markdown_report(v1_res, v2_res)
    md_path.write_text(md_text, encoding="utf-8")

    print("\n" + md_text)
    print(f"\n报告已保存至: {md_path} 与 {json_path}")


if __name__ == "__main__":
    main()
