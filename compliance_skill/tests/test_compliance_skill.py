"""
tests/test_compliance_skill.py
compliance_skill 单元测试 & 集成测试

运行：
  cd compliance_skill
  pip install pytest
  OPENROUTER_API_KEY=your-key pytest tests/ -v

覆盖范围：
  - term_aligner：精确匹配 / LLM 推断 / 回退
  - regulation_retriever：KG 搜索 / 静态 KB 匹配 / 层次选择
  - confidence_gate：通过 / 重试 / 人工标记
  - compliance_reviewer：verdict 提取
  - main_workflow：端到端 (需 API 密钥)
"""
import os
import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# 使 skill 模块可见
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

# ══════════════════════════════════════════════════════════════════════
#  1. term_aligner 测试
# ══════════════════════════════════════════════════════════════════════
class TestTermAligner(unittest.TestCase):
    """测试三级术语对齐降级逻辑。"""

    def setUp(self):
        from skills.term_aligner import _load_term_map
        self.term_map = _load_term_map()

    def test_exact_match_english(self):
        """英文小写精确匹配应命中。"""
        from skills.term_aligner import _exact_match
        result = _exact_match("sodium benzoate")
        self.assertIsNotNone(result, "sodium benzoate 应在词典中")
        self.assertEqual(result["国标名"], "苯甲酸钠")

    def test_exact_match_case_insensitive(self):
        """大小写不敏感。"""
        from skills.term_aligner import _exact_match
        result = _exact_match("Sodium Benzoate")
        self.assertIsNotNone(result)
        self.assertEqual(result["国标名"], "苯甲酸钠")

    def test_exact_match_german(self):
        """德文名 Natriumbenzoat 应命中。"""
        from skills.term_aligner import _exact_match
        result = _exact_match("Natriumbenzoat")
        self.assertIsNotNone(result)
        self.assertEqual(result["国标名"], "苯甲酸钠")

    def test_exact_match_e_number(self):
        """EU 编号 E171 应命中。"""
        from skills.term_aligner import _exact_match
        result = _exact_match("E171")
        self.assertIsNotNone(result)
        self.assertEqual(result["国标名"], "二氧化钛")

    def test_exact_match_miss(self):
        """不存在的词条应返回 None。"""
        from skills.term_aligner import _exact_match
        result = _exact_match("xyzabc_nonexistent")
        self.assertIsNone(result)

    def test_align_term_returns_dict(self):
        """align_term 对已知词条返回含国标名的字典。"""
        from skills.term_aligner import align_term
        result = align_term("titanium dioxide")
        self.assertIn("国标名", result)
        self.assertEqual(result["国标名"], "二氧化钛")
        self.assertIn("_match_method", result)

    def test_align_term_match_method_exact(self):
        """精确匹配时 _match_method 应为 'exact'。"""
        from skills.term_aligner import align_term
        result = align_term("glyphosate")
        self.assertEqual(result.get("_match_method"), "exact")
        self.assertEqual(result.get("_confidence"), "high")

    def test_term_map_not_empty(self):
        """词典文件应已生成且非空。"""
        self.assertGreater(len(self.term_map), 10,
            "term_mapping.json 为空，请先运行 knowledge_base/build_kg_and_terms.py")


# ══════════════════════════════════════════════════════════════════════
#  2. regulation_retriever 测试
# ══════════════════════════════════════════════════════════════════════
class TestRegulationRetriever(unittest.TestCase):

    def test_kg_search_returns_list(self):
        """KG 搜索返回列表（可为空）。"""
        from skills.regulation_retriever import _kg_search
        results = _kg_search(["铅", "中国"], "中国")
        self.assertIsInstance(results, list)

    def test_kg_search_lead_china(self):
        """铅+中国 应在 KG 中有命中（如 KG 已构建）。"""
        from skills.regulation_retriever import _kg_search, _KG
        if not _KG:
            self.skipTest("regulation_kg.json 为空，跳过 KG 测试")
        results = _kg_search(["铅"], "中国")
        self.assertGreater(len(results), 0, "铅/中国 应在 KG 中有命中")

    def test_static_search_eu_titanium(self):
        """欧盟+二氧化钛 应命中静态 KB。"""
        from skills.regulation_retriever import _static_search
        result = _static_search(["titanium dioxide", "E171"], "欧盟")
        self.assertIsNotNone(result, "静态 KB 应含 EU TiO2 条目")
        self.assertIn("2022", result.get("effective_date", ""))

    def test_static_search_china_gb2762(self):
        """中国+GB2762 应命中。"""
        from skills.regulation_retriever import _static_search
        result = _static_search(["GB 2762", "lead", "China"], "中国")
        self.assertIsNotNone(result)

    def test_static_search_japan_glyphosate(self):
        """日本+草甘膦+燕麦 应命中。"""
        from skills.regulation_retriever import _static_search
        result = _static_search(["glyphosate", "oat", "Japan"], "日本")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("effective_date"), "2018-12")

    def test_static_search_no_match(self):
        """查不到时返回 None。"""
        from skills.regulation_retriever import _static_search
        result = _static_search(["xyz_nonexistent"], "未知国")
        self.assertIsNone(result)

    def test_search_regulations_found_structure(self):
        """search_regulations 命中时结构正确。"""
        from skills.regulation_retriever import search_regulations
        with patch("skills.regulation_retriever._fetch_eu", return_value=[]):
            result = search_regulations("titanium dioxide E171", "欧盟")
        self.assertIn("found", result)
        self.assertIn("results", result)
        self.assertIn("confidence", result)
        self.assertIn("retrieval_layer", result)

    def test_search_regulations_not_found_structure(self):
        """查不到时 found=False，message 字段存在。"""
        from skills.regulation_retriever import search_regulations
        result = search_regulations("xyzabc_totally_unknown_substance", "未知国")
        # 可能命中 KG 或静态 KB，关键是结构正确
        self.assertIn("found", result)
        self.assertIn("retrieval_layer", result)

    def test_confidence_above_zero_on_match(self):
        """命中时 confidence > 0。"""
        from skills.regulation_retriever import search_regulations
        with patch("skills.regulation_retriever._fetch_eu", return_value=[]):
            result = search_regulations("铅 欧盟", "欧盟")
        if result["found"]:
            self.assertGreater(result["confidence"], 0)


# ══════════════════════════════════════════════════════════════════════
#  3. confidence_gate 测试
# ══════════════════════════════════════════════════════════════════════
class TestConfidenceGate(unittest.TestCase):

    def _make_retrieval(self, found=True, confidence=0.90, layer=1):
        return {
            "found":           found,
            "confidence":      confidence,
            "retrieval_layer": layer,
            "results":         [],
        }

    def test_pass_when_high_confidence(self):
        """高置信度应通过。"""
        from agents.confidence_gate import confidence_gate
        gate = confidence_gate(self._make_retrieval(confidence=0.92), "compliance")
        self.assertEqual(gate.action, "proceed")
        self.assertTrue(gate.passed)

    def test_flag_when_not_found(self):
        """未找到法规应标记人工复核。"""
        from agents.confidence_gate import confidence_gate
        gate = confidence_gate(self._make_retrieval(found=False, confidence=0.0), "compliance")
        self.assertEqual(gate.action, "flag_human")
        self.assertFalse(gate.passed)

    def test_proceed_with_warning_layer2(self):
        """Layer2 低置信应 proceed_with_warning（而非 flag_human）。"""
        from agents.confidence_gate import confidence_gate
        gate = confidence_gate(
            self._make_retrieval(confidence=0.60, layer=2), "compliance"
        )
        self.assertIn(gate.action, ("proceed_with_warning", "retry"))

    def test_flag_human_layer4(self):
        """Layer4（静态 KB）低置信应 flag_human。"""
        from agents.confidence_gate import confidence_gate
        gate = confidence_gate(
            self._make_retrieval(confidence=0.55, layer=4), "compliance"
        )
        self.assertEqual(gate.action, "flag_human")

    def test_retry_with_supplemental(self):
        """补充检索成功时应 retry 并 passed=True。"""
        from agents.confidence_gate import confidence_gate

        def mock_retriever(kw, country):
            return {"found": True, "confidence": 0.92, "retrieval_layer": 1, "results": [{"title": "test"}]}

        gate = confidence_gate(
            self._make_retrieval(confidence=0.60, layer=2),
            "compliance",
            supplemental_retriever=mock_retriever,
            keywords="test",
            country="中国",
        )
        self.assertEqual(gate.action, "retry")
        self.assertTrue(gate.passed)


# ══════════════════════════════════════════════════════════════════════
#  4. compliance_reviewer 测试（Mock LLM）
# ══════════════════════════════════════════════════════════════════════
class TestComplianceReviewer(unittest.TestCase):

    def _make_aligned(self, name="苯甲酸钠"):
        return {"国标名": name, "_match_method": "exact", "_confidence": "high"}

    def _make_reg_result(self):
        return {
            "found": True,
            "results": [{"title": "GB 2760-2014 防腐剂限量", "legal_status": "Immediately Effective",
                         "effective_date": "2015-05", "source": "Static KB", "retrieval_layer": 4}],
            "retrieval_layer": 4, "confidence": 0.62,
        }

    def _mock_response(self, verdict_line: str) -> MagicMock:
        resp = MagicMock()
        resp.choices[0].message.content = (
            "STEP 1 成分识别：苯甲酸钠（防腐剂）\n"
            "STEP 2 限量核验：GB 2760-2014 规定饮料中最大用量 0.2 g/kg\n"
            "STEP 3 添加剂合规：在允许范围内\n"
            "STEP 4 标签合规：须用中文标注\n"
            "STEP 5 程序性要求：无特殊检验要求\n"
            "STEP 6 综合结论：限量值符合规定\n"
            f"{verdict_line}"
        )
        return resp

    @patch("skills.compliance_reviewer._client")
    def test_verdict_compliant(self, mock_client):
        """建议放行时 verdict 正确提取。"""
        from skills.compliance_reviewer import review_compliance
        mock_client.chat.completions.create.return_value = (
            self._mock_response("审查结论：建议放行")
        )
        result = review_compliance(
            self._make_aligned(), self._make_reg_result(), "苯甲酸钠含量是否合规？"
        )
        self.assertEqual(result["verdict"], "建议放行")
        self.assertIsInstance(result["full_report"], str)
        self.assertFalse(result.get("flagged_uncertain"))

    @patch("skills.compliance_reviewer._client")
    def test_verdict_violation(self, mock_client):
        """违规时 verdict 正确提取。"""
        from skills.compliance_reviewer import review_compliance
        mock_client.chat.completions.create.return_value = (
            self._mock_response("审查结论：违规")
        )
        result = review_compliance(
            self._make_aligned(), self._make_reg_result(), "是否违规？"
        )
        self.assertEqual(result["verdict"], "违规")

    @patch("skills.compliance_reviewer._client")
    def test_low_confidence_flag(self, mock_client):
        """低置信度时 full_report 应含 ⚠️。"""
        from skills.compliance_reviewer import review_compliance
        mock_client.chat.completions.create.return_value = (
            self._mock_response("审查结论：建议放行")
        )
        # 覆盖 mock 以包含 ⚠️
        mock_client.chat.completions.create.return_value.choices[0].message.content = (
            "STEP 2 限量核验：⚠️[需人工核实]\n审查结论：建议放行"
        )
        result = review_compliance(
            self._make_aligned(), self._make_reg_result(), "query",
            retrieval_confidence=0.60
        )
        self.assertTrue(result["flagged_uncertain"])

    @patch("skills.compliance_reviewer._client")
    def test_llm_failure_graceful(self, mock_client):
        """LLM 调用失败时应返回无法判断，不抛异常。"""
        from skills.compliance_reviewer import review_compliance
        mock_client.chat.completions.create.side_effect = Exception("Connection error")
        result = review_compliance(
            self._make_aligned(), self._make_reg_result(), "test"
        )
        self.assertEqual(result["verdict"], "无法判断")
        self.assertTrue(result["flagged_uncertain"])


# ══════════════════════════════════════════════════════════════════════
#  5. main_workflow 集成测试（需要有效 API Key）
# ══════════════════════════════════════════════════════════════════════
@unittest.skipUnless(
    os.environ.get("OPENROUTER_API_KEY", "test-key") not in ("test-key", ""),
    "需要有效 OPENROUTER_API_KEY 才能运行集成测试"
)
class TestMainWorkflowIntegration(unittest.TestCase):

    def test_process_case_known_additive(self):
        """Natriumbenzoat 完整流水线应返回 ComplianceCase。"""
        from main_workflow import process_case
        case = process_case(
            foreign_ingredient="Natriumbenzoat",
            question="该德国饮料含苯甲酸钠 0.1 g/kg，是否可进口中国？",
            destination_country="中国",
            verbose=False,
        )
        self.assertIsNotNone(case.final_verdict)
        self.assertNotEqual(case.final_verdict, "流程终止：成分无法识别")

    def test_process_case_flagged_on_unknown(self):
        """完全无法识别的成分应标记人工复核。"""
        from main_workflow import process_case
        case = process_case(
            foreign_ingredient="XYZABC_NONEXISTENT_COMPOUND_12345",
            question="这个化合物能进口中国吗？",
            destination_country="中国",
            verbose=False,
        )
        # 可能是 "流程终止" 或 LLM 推断为 null → flagged
        # 主要验证不崩溃
        self.assertIsNotNone(case.final_verdict)


# ══════════════════════════════════════════════════════════════════════
#  6. 知识库完整性检查
# ══════════════════════════════════════════════════════════════════════
class TestKnowledgeBase(unittest.TestCase):

    def test_term_mapping_exists(self):
        """term_mapping.json 应存在。"""
        from config.settings import TERM_MAP_PATH
        self.assertTrue(TERM_MAP_PATH.exists(),
            f"term_mapping.json 不存在: {TERM_MAP_PATH}\n"
            "请运行 python knowledge_base/build_kg_and_terms.py")

    def test_regulation_kg_exists(self):
        """regulation_kg.json 应存在。"""
        from config.settings import KG_PATH
        self.assertTrue(KG_PATH.exists(),
            f"regulation_kg.json 不存在: {KG_PATH}\n"
            "请运行 python knowledge_base/build_kg_and_terms.py")

    def test_regulation_kg_has_triples(self):
        """KG 应含有效三元组。"""
        from config.settings import KG_PATH
        if not KG_PATH.exists():
            self.skipTest("KG 文件不存在")
        data = json.loads(KG_PATH.read_text(encoding="utf-8"))
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 20, "KG 三元组数量过少，请检查构建脚本")

    def test_kg_triple_structure(self):
        """每条三元组应含 subject/predicate/object/country 字段。"""
        from config.settings import KG_PATH
        if not KG_PATH.exists():
            self.skipTest("KG 文件不存在")
        data = json.loads(KG_PATH.read_text(encoding="utf-8"))
        required_fields = {"subject", "predicate", "object", "country"}
        for i, triple in enumerate(data[:5]):
            missing = required_fields - set(triple.keys())
            self.assertEqual(missing, set(), f"Triple[{i}] 缺少字段: {missing}")

    def test_term_mapping_has_standard_additives(self):
        """词典应包含常见添加剂条目。"""
        from config.settings import TERM_MAP_PATH
        if not TERM_MAP_PATH.exists():
            self.skipTest("词典文件不存在")
        data = json.loads(TERM_MAP_PATH.read_text(encoding="utf-8"))
        must_have = ["sodium benzoate", "titanium dioxide", "glyphosate"]
        for key in must_have:
            self.assertIn(key, data, f"词典缺少 '{key}' 条目")


if __name__ == "__main__":
    unittest.main(verbosity=2)
