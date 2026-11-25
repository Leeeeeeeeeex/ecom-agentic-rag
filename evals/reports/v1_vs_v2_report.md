# V1（纯向量基线）vs V2（混合检索 + 证据门控 + 引用核验）自动化评测报告

- 评测时间：2026-11-25 16:45:45
- 评测样本数：20 条（覆盖正常问答、跨型号/配件混淆陷阱、无评论拒答边界、正负权衡场景）

## 核心指标对比

| 指标维度 | V1 纯向量基线 | V2 Agentic 混合检索 | 变化幅度 |
| :--- | :---: | :---: | :---: |
| **SKU 召回纯净度（防串品）** | 44.0% | **100.0%** | +56.0 pp |
| **跨型号/配件污染率（越低越好）** | 100.0% | **0.0%** | -100.0 pp |
| **必需评论 Recall@5** | 96.9% | **100.0%** | +3.1 pp |
| **拒答边界准确率** | 85.0% | **100.0%** | +15.0 pp |
| **引用合法率（无伪造编号）** | 100.0% | **100.0%** | +0.0 pp |
| **忠实度 Faithfulness (Judge)** | 0.930 | **0.995** | +0.065 |
| **答案相关性 Answer Relevancy** | 0.953 | **0.983** | +0.030 |
| **P50 / P95 端到端耗时** | 1674 / 3069 ms | 2248 / 3694 ms | 分阶段拆解可观测 |
| **平均阶段耗时 (Embed / Retrieve / LLM)** | 297 / 33 / 1475 ms | 226 / 59 / 1846 ms | 拒答短路节省 LLM |
| **20 题总 Token（入/出）** | 13480 / 2054 | 13163 / 4292 | — |
| **预估千次问答模型成本（CNY）** | ¥2.170 | ¥3.033 | — |

## 典型 Badcase 对比（V1 翻车 vs V2 修复）

### Q09（cross_model_trap · 目标 SKU: `PET-FOUNTAIN-PRO`）
- **问题**：这款 Pro 饮水机是不是塑料材质容易让猫长黑下巴，晚上还有嗡嗡低频噪音？
- **V1 表现**：召回 SKU `['PET-FOUNTAIN-LITE', 'PET-FOUNTAIN-PRO', 'PET-FEEDER-CAM']`，拒答=False，Faithfulness=1.0（准确区分Pro与Lite，材质噪音回应完整，补充提醒均源自上下文）
- **V2 表现**：召回 SKU `['PET-FOUNTAIN-PRO']`，拒答=False，Faithfulness=1.0（准确澄清非塑料款、不锈钢盘不易黑下巴、噪音低，并补充硬水与有线注意点）

### Q10（cross_model_trap · 目标 SKU: `PET-FOUNTAIN-LITE`）
- **问题**：这款 Lite 饮水机能不能用无线水泵直接进洗碗机洗？
- **V1 表现**：召回 SKU `['PET-FOUNTAIN-PRO', 'PET-FOUNTAIN-LITE']`，拒答=False，Faithfulness=1.0（准确回应无线与洗碗机问题，事实均来自召回上下文，无编造。）
- **V2 表现**：召回 SKU `['PET-FOUNTAIN-LITE']`，拒答=False，Faithfulness=1.0（准确指出非无线水泵且塑料件不可洗碗机，忠实于上下文并恰当说明未覆盖细节。）

### Q11（cross_model_trap · 目标 SKU: `PET-FOUNTAIN-FILTER`）
- **问题**：这个8片装滤芯能不能装在 PurrFlow Lite 基础款饮水机上？
- **V1 表现**：召回 SKU `['PET-FOUNTAIN-FILTER', 'PET-FOUNTAIN-LITE', 'PET-FOUNTAIN-PRO', 'PET-FEEDER-DESICCANT']`，拒答=False，Faithfulness=1.0（准确回答不兼容，引用官方规格和买家反馈，无编造。）
- **V2 表现**：召回 SKU `['PET-FOUNTAIN-FILTER']`，拒答=False，Faithfulness=1.0（回答准确引用官方规格与买家反馈，明确否定兼容性，完整回应问题。）

### Q12（cross_model_trap · 目标 SKU: `PET-FEEDER-CAM`）
- **问题**：这款喂食器自带的干燥剂一包能用多久？会不会掉进粮桶里？
- **V1 表现**：召回 SKU `['PET-FEEDER-DESICCANT', 'PET-FEEDER-CAM', 'PET-LITTER-AUTO']`，拒答=False，Faithfulness=0.6（干燥剂信息来自配件SKU，非本体评论，但未冒充本体；回答基本回应问题）
- **V2 表现**：召回 SKU `['PET-FEEDER-CAM']`，拒答=False，Faithfulness=1.0（忠实引用评论，未编造周期或防掉落细节，恰当说明资料不足并建议人工确认。）

### Q13（abstain_boundary · 目标 SKU: `PET-CARRIER-AIR`）
- **问题**：这款航空宠物包买家实测拉链耐不耐用、底板塌不塌？
- **V1 表现**：召回 SKU `['PET-LITTER-AUTO', 'PET-FEEDER-CAM', 'PET-FOUNTAIN-PRO']`，拒答=False，Faithfulness=1.0（如实说明无实测评论，引用官方规格未编造，恰当回应并给出建议）
- **V2 表现**：召回 SKU `[]`，拒答=True，Faithfulness=1.0（正确拒答，未编造实测，忠实于无评论事实）
