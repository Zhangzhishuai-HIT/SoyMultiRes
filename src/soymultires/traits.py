"""Trait definitions for SoyMultiRes (32 GRIN stress descriptors + 5 Mendelian calibration traits).

Codes and scale directions follow the GRIN descriptor definition pages downloaded on 2026-11-11
(down_data/GRIN_soybean_descriptors/definitions_20261111/). Every score is oriented so that a HIGHER value means
MORE resistant / tolerant.
"""
from __future__ import annotations

import re

# (GRIN code, descriptor id, short label, Chinese name, stress category)
STRESS = [
    ("NEMATCYST_1", 51128, "SCN1", "胞囊线虫1号小种", "SCN"),
    ("NEMATCYST_3", 51131, "SCN3", "胞囊线虫3号小种", "SCN"),
    ("NEMATCYST_4", 51132, "SCN4", "胞囊线虫4号小种", "SCN"),
    ("NEMATCYST_5", 51133, "SCN5", "胞囊线虫5号小种", "SCN"),
    ("NEMATCYST_14", 51129, "SCN14", "胞囊线虫14号小种", "SCN"),
    ("PHYTOROT_1", 51137, "PR1", "疫霉根腐病1号小种", "Phytophthora"),
    ("PHYTOROT_3", 51144, "PR3", "疫霉根腐病3号小种", "Phytophthora"),
    ("PHYTOROT_4", 51150, "PR4", "疫霉根腐病4号小种", "Phytophthora"),
    ("PHYTOROT_5", 51151, "PR5", "疫霉根腐病5号小种", "Phytophthora"),
    ("PHYTOROT_7", 51153, "PR7", "疫霉根腐病7号小种", "Phytophthora"),
    ("PHYTOROT_12", 51139, "PR12", "疫霉根腐病12号小种", "Phytophthora"),
    ("PHYTOROT_17", 51140, "PR17", "疫霉根腐病17号小种", "Phytophthora"),
    ("PHYTOROT_20", 51142, "PR20", "疫霉根腐病20号小种", "Phytophthora"),
    ("PHYTOROT_25", 51143, "PR25", "疫霉根腐病25号小种", "Phytophthora"),
    ("SDS", 51110, "SDS", "猝死综合征", "fungi_oomycetes"),
    ("BRNSTEMROT", 51025, "BSR", "褐色茎腐病", "fungi_oomycetes"),
    ("NSCANKER", 51094, "NSC", "北方茎溃疡病", "fungi_oomycetes"),
    ("FROGEYE2", 51033, "FE2", "灰斑病2号小种", "fungi_oomycetes"),
    ("FROGEYE3", 51103, "FE3", "灰斑病C-32分离物", "fungi_oomycetes"),
    ("RUST_TAN", 51136, "RUST", "大豆锈病（TAN型病斑）", "fungi_oomycetes"),
    ("PYTHIUMROT", 51040, "PYT", "腐霉根腐病", "fungi_oomycetes"),
    ("BP", 51023, "BP", "细菌性斑疹病", "bacteria"),
    ("PMV", 51107, "PMV", "花生斑驳病毒", "virus"),
    ("BPMV", 51108, "BPMV", "菜豆荚斑驳病毒", "virus"),
    ("MEXBEANBTL", 51085, "MBB", "墨西哥豆瓢虫", "chewing_insects"),
    ("LOOPER", 51099, "SBL", "大豆尺蠖", "chewing_insects"),
    ("VBC", 51097, "VBC", "黎豆夜蛾", "chewing_insects"),
    ("APHID", 51111, "APH", "大豆蚜", "sucking_insects"),
    ("POTATOLHOP", 51034, "PLH", "马铃薯小绿叶蝉", "sucking_insects"),
    ("CHLOROSIS", 51074, "IDC", "缺铁黄化", "abiotic"),
    ("SALTREACT", 51054, "SALT", "耐盐", "abiotic"),
    ("HIGHTEMP", 51091, "HT", "高温种子发芽力", "abiotic"),
]
TRAITS = [t[0] for t in STRESS]
SHORT = {t[0]: t[2] for t in STRESS}
ZH = {t[0]: t[3] for t in STRESS}
CATEGORY = {t[0]: t[4] for t in STRESS}
DESCRIPTOR_ID = {t[0]: t[1] for t in STRESS}

# categorical scales: code -> ordinal score (higher = more resistant)
_LEVELS = {
    "SCN": {"S": 0, "MS": 1, "MR": 2, "R": 3},
    "PHYTOROT": {"S": 0, "H": 1, "R": 2},          # H = heterogeneous (resistant and susceptible plants)
    "PYTHIUMROT": {"S": 0, "H": 1, "R": 2},
    "SDS": {"S": 0, "MS": 1, "MR": 2},
    "APHID": {"S": 0, "MR": 1, "R": 2},
    "BRNSTEMROT": {"U": 0, "S": 0, "R": 2, "MR": 3},  # U = susceptible in preliminary field test; MR = "most resistant"
    "NSCANKER": {"S": 0, "MS": 1, "I": 2, "MR": 3, "R": 4},
    "PMV": {"S": 0, "U": 1, "R": 2},               # U = field-resistant, not confirmed in the greenhouse
    "BPMV": {"S": 0, "T": 1},
    "SALTREACT": {"S": 0, "T": 1},
    "HIGHTEMP": {"S": 0, "I": 1, "T": 2},
}
_RESISTANT = {
    "SCN": {"R", "MR"}, "PHYTOROT": {"R"}, "PYTHIUMROT": {"R"}, "SDS": {"MR"}, "APHID": {"R", "MR"},
    "BRNSTEMROT": {"R", "MR"}, "NSCANKER": {"R", "MR"}, "PMV": {"R"}, "BPMV": {"T"}, "SALTREACT": {"T"},
    "HIGHTEMP": {"T"},
}
# numeric damage / severity scores: higher raw value = more damage, so score = -value
NUMERIC = {"BP", "FROGEYE2", "FROGEYE3", "MEXBEANBTL", "LOOPER", "VBC", "POTATOLHOP", "RUST_TAN", "CHLOROSIS"}
NUMERIC_TOP_FRACTION = 0.10   # "resistant" for numeric traits = least-damaged 10% of a batch (ties included)


def family(trait: str) -> str:
    return "SCN" if trait.startswith("NEMATCYST") else "PHYTOROT" if trait.startswith("PHYTOROT") else trait


def kind(trait: str) -> str:
    return "numeric" if trait in NUMERIC else "categorical"


def levels(trait: str) -> dict:
    return _LEVELS[family(trait)]


def resistant_codes(trait: str) -> set:
    return _RESISTANT[family(trait)]


def batch_year(method: str):
    """Evaluation year parsed from a GRIN method name (None if absent)."""
    m = re.search(r"(19\d\d|20\d\d)", method)
    if m:
        return int(m.group(1))
    m = re.search(r"[A-Z](\d\d)(?:\.\d)?$", method) or re.search(r"\.(\d\d)(?:\.\d)?$", method)
    if m:
        y = int(m.group(1))
        return 1900 + y if y > 30 else 2000 + y
    return None


# Mendelian calibration traits (HapGAM's binary definitions; maturity group numeric as in HapGAM)
MENDEL = {
    "FLWRCOLOR": ("flower_purple", {"P", "Dp", "Lp"}, {"W", "Nw"}),
    "PUBCOLOR": ("pubescence_tawny", {"T", "Lt"}, {"G", "Ng"}),
    "STEMTERM": ("indeterminate", {"N"}, {"D"}),
    "SCOATCOLOR": ("seedcoat_pigmented", {"Bl", "Br", "Rbr", "Gnbr"}, {"Y"}),
}
MG_VALUE = {"000": -2.0, "00": -1.0, "0": 0.0, "I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0,
            "V": 5.0, "VI": 6.0, "VII": 7.0, "VIII": 8.0, "IX": 9.0, "X": 10.0}


def mg_group(v) -> str:
    if v is None or v != v:
        return "MG_NA"
    return "MG000-0" if v <= 0 else "MGI-II" if v <= 2 else "MGIII-IV" if v <= 4 else "MGV-VI" if v <= 6 else "MGVII-X"
