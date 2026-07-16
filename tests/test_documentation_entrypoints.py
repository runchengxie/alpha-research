from __future__ import annotations

import re
from pathlib import Path

from alpha_research.modeling import SUPPORTED_MODEL_TYPES
from alpha_research.signal_artifact import CANONICAL_SIGNAL_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
NEW_CONTRACT_DOCS = (
    ROOT / "docs" / "concepts" / "minute-factors.md",
    ROOT / "docs" / "reference" / "signal-artifacts.md",
)
RESEARCH_GUIDE_DOCS = (
    ROOT / "docs" / "concepts" / "model-landscape.md",
    ROOT / "docs" / "concepts" / "model-selection.md",
    ROOT / "docs" / "concepts" / "overfitting-controls.md",
)
STYLE_PATTERNS = (
    re.compile(r"不是.{0,40}而是"),
    re.compile(r"并非.{0,40}而是"),
    re.compile(r"\*\*"),
    re.compile("\uff1b"),
    re.compile("\u2014\u2014"),
    re.compile("[\u201c\u201d]"),
)
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def test_docs_use_concise_chinese_style() -> None:
    offenders: list[str] = []

    paths = (ROOT / "README.md", ROOT / "AGENTS.md", *sorted((ROOT / "docs").rglob("*.md")))
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            for pattern in STYLE_PATTERNS:
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{line_number}:{pattern.pattern}")

    assert offenders == []


def test_local_markdown_links_resolve() -> None:
    missing: list[str] = []
    paths = (ROOT / "README.md", ROOT / "AGENTS.md", *sorted((ROOT / "docs").rglob("*.md")))

    for path in paths:
        docs = path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK_PATTERN.finditer(docs):
            target = match.group(1).strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            local_target = target.split("#", 1)[0].strip("<>")
            if local_target and not (path.parent / local_target).resolve().exists():
                missing.append(f"{path.relative_to(ROOT)} -> {target}")

    assert missing == []


def test_testing_docs_match_script_modes() -> None:
    script = (ROOT / "scripts" / "dev" / "run_tests.sh").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "testing.md").read_text(encoding="utf-8")

    for mode in (
        "all",
        "fast",
        "unit",
        "lint",
        "format",
        "typecheck",
        "basedpyright",
        "maintainability",
    ):
        assert f"`{mode}`" in docs
        assert mode in script


def test_docs_record_current_automation_status() -> None:
    docs = (ROOT / "docs" / "testing.md").read_text(encoding="utf-8")

    assert "当前仓库没有启用 GitHub Actions 远端测试" in docs
    assert "本地质量门禁" in docs
    assert ".github/workflows/tests.yml" not in docs


def test_model_landscape_matches_current_registry_and_research_state() -> None:
    docs = (ROOT / "docs" / "concepts" / "model-landscape.md").read_text(encoding="utf-8")

    for model_type in SUPPORTED_MODEL_TYPES:
        assert f"`{model_type}`" in docs
    assert "当前默认研究主线是 A 股" in docs
    assert "随机森林尚未进入模型注册表" in docs
    assert "Triple Barrier 标签已经" in docs
    assert "HK quarterly" not in docs
    assert len(docs.splitlines()) < 120


def test_model_selection_covers_training_and_artifact_roles() -> None:
    docs = (ROOT / "docs" / "concepts" / "model-selection.md").read_text(encoding="utf-8")

    for model_type in SUPPORTED_MODEL_TYPES:
        assert f"`{model_type}`" in docs
    assert "不训练预测模型" in docs
    assert "A 股预设当前使用 `xgb_regressor`" in docs


def test_minute_and_signal_contract_docs_are_indexed_and_complete() -> None:
    index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    minute_docs = NEW_CONTRACT_DOCS[0].read_text(encoding="utf-8")
    signal_docs = NEW_CONTRACT_DOCS[1].read_text(encoding="utf-8")

    assert "concepts/minute-factors.md" in index
    assert "reference/signal-artifacts.md" in index
    assert "concepts/afml-methodology.md" in index
    assert "minute_friend_factors" in minute_docs
    assert "minute_factors" in minute_docs
    for column in CANONICAL_SIGNAL_COLUMNS:
        assert f"`{column}`" in signal_docs


def test_overfitting_docs_use_owner_relative_source_path() -> None:
    docs = (ROOT / "docs" / "concepts" / "overfitting-controls.md").read_text(encoding="utf-8")

    assert "`src/alpha_research/split.py`" in docs
    assert "../alpha-research/src/alpha_research/split.py" not in docs
    assert "根目录 `docs/platform-workflow.md`" in docs
