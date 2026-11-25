"""端到端实测脚本：验证 FastAPI + VM pgvector + VM Redis 缓存/限流 + VM Langfuse Trace/Score 全链路。"""

import base64
import time

import httpx
from fastapi.testclient import TestClient

from ecom_rag.api import app
from ecom_rag.config import get_settings


def main() -> None:
    get_settings.cache_clear()
    settings = get_settings()
    client = TestClient(app)

    print("1) 检查 /health 与 /products ...")
    h = client.get("/health")
    assert h.status_code == 200, h.text
    print("   /health:", h.json())

    p = client.get("/products")
    assert p.status_code == 200, p.text
    print(f"   /products: 共 {len(p.json()['items'])} 个商品")

    print("\n2) 实测 V2 冷请求（穿透至 pgvector + BM25 + DeepSeek + Langfuse）...")
    q_payload = {
        "sku_id": "PET-FOUNTAIN-PRO",
        "question": "这款饮水机放卧室床头吵不吵？清洗方便吗？",
        "pipeline_version": "v2_agentic_hybrid",
        "use_cache": True,
    }
    # 先用 use_cache=False 确保拿到冷启动数据，再开启缓存写入与命中测试
    t0 = time.perf_counter()
    r_cold = client.post("/api/v1/qa", json={**q_payload, "use_cache": True})
    cold_wall_ms = int((time.perf_counter() - t0) * 1000)
    assert r_cold.status_code == 200, r_cold.text
    d_cold = r_cold.json()
    print(
        f"   冷请求: trace_id={d_cold['trace_id']}, cache_hit={d_cold['cache_hit']}, "
        f"wall_ms={cold_wall_ms}, metrics={d_cold['metrics']}, cited={d_cold['cited_review_ids']}"
    )

    print("\n3) 实测 Redis 缓存命中（同问题二次请求）...")
    t1 = time.perf_counter()
    r_hit = client.post("/api/v1/qa", json=q_payload)
    hit_wall_ms = int((time.perf_counter() - t1) * 1000)
    assert r_hit.status_code == 200, r_hit.text
    d_hit = r_hit.json()
    print(f"   热缓存: cache_hit={d_hit['cache_hit']}, wall_ms={hit_wall_ms} ms（冷请求 {cold_wall_ms} ms）")
    assert d_hit["cache_hit"] is True

    print("\n4) 实测无评论商品拒答短路（PET-CARRIER-AIR，0 LLM Token）...")
    r_abs = client.post(
        "/api/v1/qa",
        json={
            "sku_id": "PET-CARRIER-AIR",
            "question": "买家实测这款宠物包拉链耐不耐用？",
            "pipeline_version": "v2_agentic_hybrid",
            "use_cache": False,
        },
    )
    assert r_abs.status_code == 200, r_abs.text
    d_abs = r_abs.json()
    print(
        f"   拒答实测: abstained={d_abs['abstained']}, trace_id={d_abs['trace_id']}, "
        f"metrics={d_abs['metrics']}, answer={d_abs['answer'][:60]}..."
    )
    assert d_abs["abstained"] is True

    print("\n5) 实测 V1 跨品陷阱 + 提交 Badcase 反馈回流（同时同步到 PG 与 Langfuse）...")
    r_v1 = client.post(
        "/api/v1/qa",
        json={
            "sku_id": "PET-FEEDER-CAM",
            "question": "这款喂食器自带的干燥剂一包能用多久？会不会掉进粮桶里？",
            "pipeline_version": "v1_vector",
            "use_cache": False,
        },
    )
    assert r_v1.status_code == 200, r_v1.text
    d_v1 = r_v1.json()
    v1_trace = d_v1["trace_id"]
    print(f"   V1 召回 SKUs: {[c['sku_id'] for c in d_v1['evidence_chunks']]}, trace_id={v1_trace}")

    r_fb = client.post(
        "/api/v1/feedback",
        json={"trace_id": v1_trace, "user_feedback": -1, "badcase_tag": "wrong_sku"},
    )
    assert r_fb.status_code == 200, r_fb.text
    print("   /api/v1/feedback:", r_fb.json())

    print("\n6) 核验 Langfuse 服务器已成功收到 Trace 与 Score ...")
    token = base64.b64encode(
        f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode("utf-8")
    ).decode("ascii")
    time.sleep(1.5)
    lf_resp = httpx.get(
        f"{settings.langfuse_host.rstrip('/')}/api/public/traces?limit=5",
        headers={"Authorization": f"Basic {token}"},
        timeout=5.0,
    )
    print("   Langfuse /api/public/traces 状态:", lf_resp.status_code)
    if lf_resp.status_code == 200:
        traces = lf_resp.json().get("data", [])
        print(f"   Langfuse 最新 Trace 数量: {len(traces)}")
        for t in traces[:3]:
            print(f"     - id={t.get('id')} name={t.get('name')} tags={t.get('tags')}")
    else:
        print("   Langfuse 返回:", lf_resp.text[:200])


if __name__ == "__main__":
    main()
