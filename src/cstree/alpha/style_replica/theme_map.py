"""AI hardware chain theme mapping for StyleReplica-A80B20-v0.

Maps A-share stocks to their AI hardware chain themes based on Shenwan 2021
industry classification (L3 level) and concept board membership.

The 7 themes and their quotas match the design document:

| Theme                                  | Quota |
|----------------------------------------|------:|
| semiconductor_chip_equipment_materials |    18 |
| electronic_components_passive_ceramic  |    17 |
| pcb_ccl_electronic_substrate           |    14 |
| electronic_chemicals_polymer_materials |    12 |
| optical_cpo_communication_equipment    |     7 |
| datacenter_storage_cooling             |     6 |
| minor_metals_rare_metal_powder          |     6 |
| **Total**                              |  **80** |
"""

from __future__ import annotations

from typing import Final

import pandas as pd

# ── Theme definitions ──────────────────────────────────────────────────────────

AI_HARDWARE_THEME_QUOTAS: Final[dict[str, int]] = {
    "semiconductor_chip_equipment_materials": 18,
    "electronic_components_passive_ceramic": 17,
    "pcb_ccl_electronic_substrate": 14,
    "electronic_chemicals_polymer_materials": 12,
    "optical_cpo_communication_equipment": 7,
    "datacenter_storage_cooling": 6,
    "minor_metals_rare_metal_powder": 6,
}

THEME_LABELS: Final[dict[str, str]] = {
    "semiconductor_chip_equipment_materials": "半导体/国产芯片/设备材料",
    "electronic_components_passive_ceramic": "电子元器件/被动件/电子陶瓷",
    "pcb_ccl_electronic_substrate": "PCB/覆铜板/电子基材",
    "electronic_chemicals_polymer_materials": "电子化学品/高分子材料",
    "optical_cpo_communication_equipment": "光模块/CPO/通信设备",
    "datacenter_storage_cooling": "数据中心/存储/温控",
    "minor_metals_rare_metal_powder": "小金属/稀有金属/粉体",
}

# ── Industry → Theme mapping (Shenwan 2021 L3) ────────────────────────────────

_INDUSTRY_TO_THEME: Final[dict[str, str]] = {
    # Semiconductor
    "集成电路": "semiconductor_chip_equipment_materials",
    "半导体材料": "semiconductor_chip_equipment_materials",
    "半导体设备": "semiconductor_chip_equipment_materials",
    "分立器件": "semiconductor_chip_equipment_materials",
    "芯片设计": "semiconductor_chip_equipment_materials",
    "数字芯片设计": "semiconductor_chip_equipment_materials",
    "模拟芯片设计": "semiconductor_chip_equipment_materials",
    # Electronic components
    "印制电路板": "pcb_ccl_electronic_substrate",
    "被动元件": "electronic_components_passive_ceramic",
    "电子化学品Ⅲ": "electronic_chemicals_polymer_materials",
    "电子陶瓷": "electronic_components_passive_ceramic",
    "消费电子零部件及组装": "electronic_components_passive_ceramic",
    "其他电子Ⅲ": "electronic_components_passive_ceramic",
    "LED": "electronic_components_passive_ceramic",
    "面板": "electronic_components_passive_ceramic",
    "光学元件": "electronic_components_passive_ceramic",
    "连接器": "electronic_components_passive_ceramic",
    "覆铜板": "pcb_ccl_electronic_substrate",
    # Communication / Optical
    "通信网络设备及器件": "optical_cpo_communication_equipment",
    "通信线缆及配套": "optical_cpo_communication_equipment",
    "通信终端及配件": "optical_cpo_communication_equipment",
    "其他通信设备": "optical_cpo_communication_equipment",
    "光模块": "optical_cpo_communication_equipment",
    # Chemicals / Materials
    "有机硅": "electronic_chemicals_polymer_materials",
    "氟化工": "electronic_chemicals_polymer_materials",
    "磷化工及磷酸盐": "electronic_chemicals_polymer_materials",
    "膜材料": "electronic_chemicals_polymer_materials",
    "改性塑料": "electronic_chemicals_polymer_materials",
    "合成树脂": "electronic_chemicals_polymer_materials",
    "涂料油墨": "electronic_chemicals_polymer_materials",
    "其他化学制品": "electronic_chemicals_polymer_materials",
    "其他塑料制品": "electronic_chemicals_polymer_materials",
    "高分子材料": "electronic_chemicals_polymer_materials",
    # Datacenter / Storage / Cooling
    "温控设备": "datacenter_storage_cooling",
    "制冷空调设备": "datacenter_storage_cooling",
    "其他通用设备": "datacenter_storage_cooling",
    "数据存储": "datacenter_storage_cooling",
    "IT服务Ⅲ": "datacenter_storage_cooling",
    # Minor metals
    "钨": "minor_metals_rare_metal_powder",
    "锂": "minor_metals_rare_metal_powder",
    "稀土": "minor_metals_rare_metal_powder",
    "其他小金属": "minor_metals_rare_metal_powder",
    "金属新材料": "minor_metals_rare_metal_powder",
    "磁性材料": "minor_metals_rare_metal_powder",
    "粉体材料": "minor_metals_rare_metal_powder",
    # Equipment / Machinery
    "激光设备": "semiconductor_chip_equipment_materials",
    "仪器仪表Ⅲ": "semiconductor_chip_equipment_materials",
    "其他专用设备": "semiconductor_chip_equipment_materials",
    "其他自动化设备": "semiconductor_chip_equipment_materials",
    "机器人": "semiconductor_chip_equipment_materials",
    "工控设备": "semiconductor_chip_equipment_materials",
}


def map_stock_to_theme(
    symbol: str,
    industry_name: str | None = None,
    *,
    concept_tags: list[str] | None = None,
) -> str | None:
    """Map a single stock to its AI hardware chain theme.

    Priority order:
    1. Industry L3 name → theme mapping
    2. Concept board tags (keyword match)

    Returns ``None`` if the stock doesn't belong to any AI hardware theme.
    """
    concepts = concept_tags or []

    # Industry-based mapping (highest confidence)
    if industry_name and industry_name.strip():
        name = industry_name.strip()
        # Exact match first
        if name in _INDUSTRY_TO_THEME:
            return _INDUSTRY_TO_THEME[name]
        # Substring match
        for kw, theme in _INDUSTRY_TO_THEME.items():
            if kw in name:
                return theme

    # Concept-based fallback
    concept_keywords: dict[str, str] = {
        "半导体": "semiconductor_chip_equipment_materials",
        "芯片": "semiconductor_chip_equipment_materials",
        "光刻": "semiconductor_chip_equipment_materials",
        "PCB": "pcb_ccl_electronic_substrate",
        "覆铜板": "pcb_ccl_electronic_substrate",
        "电子基材": "pcb_ccl_electronic_substrate",
        "被动元件": "electronic_components_passive_ceramic",
        "电子陶瓷": "electronic_components_passive_ceramic",
        "MLCC": "electronic_components_passive_ceramic",
        "电子化学品": "electronic_chemicals_polymer_materials",
        "电子材料": "electronic_chemicals_polymer_materials",
        "光模块": "optical_cpo_communication_equipment",
        "CPO": "optical_cpo_communication_equipment",
        "光通信": "optical_cpo_communication_equipment",
        "数据中心": "datacenter_storage_cooling",
        "温控": "datacenter_storage_cooling",
        "液冷": "datacenter_storage_cooling",
        "小金属": "minor_metals_rare_metal_powder",
        "稀有金属": "minor_metals_rare_metal_powder",
        "钨": "minor_metals_rare_metal_powder",
        "钽": "minor_metals_rare_metal_powder",
        "铌": "minor_metals_rare_metal_powder",
        "粉体": "minor_metals_rare_metal_powder",
        "AI算力": "datacenter_storage_cooling",
    }

    for tag in concepts:
        tag_lower = tag.strip()
        for keyword, theme in concept_keywords.items():
            if keyword.lower() in tag_lower.lower():
                return theme

    return None


def build_theme_map(
    industry_frame: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    industry_col: str = "industry_name",
    concept_frame: pd.DataFrame | None = None,
    concept_symbol_col: str = "symbol",
    concept_name_col: str = "concept_name",
) -> pd.Series:
    """Build a symbol → theme mapping for an entire universe.

    Args:
        industry_frame: DataFrame with symbol and industry_name columns.
        concept_frame: Optional concept board membership DataFrame.

    Returns:
        Series indexed by symbol, values are theme keys (or NaN if unmapped).
    """
    # Build concept tag dict per symbol
    concept_by_symbol: dict[str, list[str]] = {}
    if concept_frame is not None and not concept_frame.empty:
        for _, row in concept_frame.iterrows():
            sym = str(row.get(concept_symbol_col, ""))
            tag = str(row.get(concept_name_col, ""))
            if sym and tag:
                concept_by_symbol.setdefault(sym, []).append(tag)

    # Build industry name dict
    industry_by_symbol: dict[str, str] = {}
    if industry_col in industry_frame.columns:
        for _, row in industry_frame.iterrows():
            sym = str(row.get(symbol_col, ""))
            ind = str(row.get(industry_col, ""))
            if sym and ind:
                industry_by_symbol[sym] = ind

    # Map all symbols
    all_symbols = sorted(set(industry_frame[symbol_col].astype(str)))
    theme_map: dict[str, str | None] = {}
    for symbol in all_symbols:
        theme_map[symbol] = map_stock_to_theme(
            symbol,
            industry_name=industry_by_symbol.get(symbol),
            concept_tags=concept_by_symbol.get(symbol, []),
        )

    return pd.Series(theme_map, name="theme")


def get_theme_quota(theme: str) -> int:
    """Return the A-leg quota for a given theme key. Returns 0 for unmapped."""
    return AI_HARDWARE_THEME_QUOTAS.get(theme, 0)


def get_theme_label(theme: str) -> str:
    """Return the Chinese display label for a theme key."""
    return THEME_LABELS.get(theme, theme)
