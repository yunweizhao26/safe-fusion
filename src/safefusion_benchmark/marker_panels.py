"""Marker panels for the colon and pancreatic islet benchmarks.

The primary panels are the markers that the source studies used to annotate
the cell types evaluated here:

* Colon epithelium: Table S2 ("Marker genes for fine-grained cell type
  identification") of Kong et al., Immunity 56:444-458 (2023),
  doi:10.1016/j.immuni.2023.01.002.  Gene symbols are kept as published.
* Pancreatic islets: the Garnett cell type marker file (Supplementary Table 17)
  of Fasolino et al., Nature Metabolism 4:284-299 (2022),
  doi:10.1038/s42255-022-00531-x.

A panel maps each gene to the annotated cell-type labels it marks.  Genes that
are absent from a benchmark matrix are skipped by the evaluators.  The original
hand-assembled panels are kept for a sensitivity analysis.
"""

from __future__ import annotations

KONG_TABLE_S2: dict[str, tuple[str, ...]] = {
    "Activated fibroblasts CCL19+ ADAMADEC1+": ("CCL19", "ADAMDEC1"),
    "B cells": ("MS4A1", "CD19", "VPREB3", "CD79A", "BANK1", "CD79B", "CD22"),
    "B cells AICDA+ LRMP+": ("AICDA", "LRMP", "TCL1A", "SNX29P2", "NEIL1"),
    "DC1": ("CLEC9A", "IDO1", "CPNE3", "BATF3"),
    "DC2 CD1D-": ("FCER1A", "CLEC10A", "CD1D"),
    "DC2 CD1D+": ("FCER1A", "CLEC10A", "CD1D"),
    "Endothelial cells CA4+ CD36+": ("CD4", "CD36"),
    "Endothelial cells CD36+": ("CD36", "RBP7", "TMEM88", "PLVAP", "COL15A1"),
    "Endothelial cells DARC+": ("DARC", "SELE", "C2CD4B", "GPR126", "CPE"),
    "Endothelial cells LTC4S+ SEMA3G+": ("SEMA3G", "LTC4S", "C10orf10"),
    "Enterochromaffin cells": ("CHGA", "TPH1", "CES1", "SLC38A11", "RAB3C"),
    "Enterocytes BEST4+": ("BEST4", "CA7", "CA4", "SPIB", "OTOP2", "NOTCH2NL"),
    "Enterocytes CA1+ CA2+ CA4-": (
        "CA1", "SLC26A2", "CA2", "SLC26A3", "KRT19", "SELENBP1", "PKIB", "UGT2B17", "CES2",
    ),
    "Enterocytes TMIGD1+ MEP1A+": ("TMIGD1", "MEP1A", "APOA4", "APOC3", "APOA1", "FABP6"),
    "Enterocytes TMIGD1+ MEP1A+ GSTA1+": ("GSTA1", "GSTA2", "TMIGD1", "MEP1A"),
    "Enteroendocrine cells": (
        "PCSK1N", "PYY", "CHGA", "GCG", "CRYBA2", "SCGN", "FEV", "SCG5", "INSL5", "MS4A8",
    ),
    "Epithelial cells HBB+ HBA+": ("HBB", "HBA2", "HBA1"),
    "Epithelial cells METTL12+ MAFB+": ("MAFB", "METTL12"),
    "Epithelial Cycling cells": (
        "UBE2C", "PTTG1", "HMGB2", "TOP2A", "CKS2", "CENPW", "CDKN3", "STMN1", "TUBB4B", "HIST1H4C",
    ),
    "Fibroblasts ADAMDEC1+": ("CCL11", "ADAMDEC1", "CCL13", "HAPLN1"),
    "Fibroblasts KCNN3+ LY6H+": ("KCNN3", "LY6H", "DPT", "C7", "SCN7A"),
    "Fibroblasts NPY+ SLITRK6+": ("NPY", "SLITRK6", "F3", "EDNRB", "NSG1"),
    "Fibroblasts SFRP2+ SLPI+": ("SLPI", "SFRP2", "IGFBP6", "MFAP5"),
    "Fibroblasts SMOC2+ PTGIS+": ("SMOC2", "PTGIS", "F3", "PCSK6", "ADAMTSL3"),
    "Glial cells": ("GPM6B", "S100B", "PLP1", "NRXN1", "CDH19", "SCN7A", "LGI4", "SPP1"),
    "Goblet cells MUC2+ TFF1-": (
        "MUC2", "RETNLB", "SPINK4", "ITLN1", "CLCA1", "FCGBP", "TFF3", "ST6GALNAC1", "LRRC26", "REP15",
    ),
    "Goblet cells MUC2+ TFF1+": ("MUC2", "SPINK4", "FCGBP", "CLCA1", "ZG16", "TFF1", "BCAS1", "CEACAM5"),
    "Goblet cells SPINK4+": (
        "SPINK4", "MUC2", "FCGBP", "CLCA1", "ITLN1", "TFF3", "TFF1", "S100P", "RETNLB", "LRRC26",
    ),
    "IELs ID3+ ENTPD1+": ("ID3", "ENTPD1", "GZMA", "CD247", "CD7", "HOPX"),
    "ILCs": ("ALDOC", "LINC00299", "LST1", "IL4I1", "AREG"),
    "Immune Cycling cells": (
        "STMN1", "HMGB2", "TCL1A", "NUSAP1", "KIAA0101", "TOP2A", "TYMS", "CDK1", "UBE2C", "PTTG1",
    ),
    "Inflammatory fibroblasts IL11+ CHI3L1+": ("CHI3L1", "IL11", "MMP3", "MMP1", "TNC"),
    "L cells": ("CHGA", "NTS", "PYY", "GCG", "CCK"),
    "Lymphatics": ("CCL21", "MMRN1", "LYVE1", "TFPI", "PPFIBP1"),
    "Macrophages": ("CD163", "C1QC", "C1QA", "C1QB"),
    "Macrophages CCL3+ CCL4+": ("CCL3", "CCL4", "DAB2", "A2M"),
    "Macrophages CXCL9+ CXCL10+": ("CXCL10", "CXCL9", "GBP1", "CXCL11"),
    "Macrophages LYVE1+": ("LYVE1", "F13A1", "CCL18"),
    "Macrophages Metallothionein": ("MT1G", "MT1X", "MT2A", "MT1H", "MT1E", "MT1F", "MT1M"),
    "Macrophages PLA2G2D+": ("PLA2G2D", "MMP9", "PTGDS"),
    "Mast cells": ("TPSAB1", "CPA3", "CTSG", "HDC", "GATA2", "VWA5A", "SLC18A2"),
    "Mature DCs": ("LAMP3", "FSCN1", "CCL19", "CCL22", "IDO1", "CCR7", "MARCKSL1"),
    "Monocytes CHI3L1+ CYP27A1+": ("CHI3L1", "CYP27A1"),
    "Monocytes S100A8+ S100A9+": ("S100A9", "S100A8", "FCN1", "G0S2", "EREG", "FPR1"),
    "Myofibroblasts GREM1+ GREM2+": ("GREM1", "GREM2", "ACTG2", "DES", "TAGLN", "MYH11"),
    "Myofibroblasts HHIP+ NPNT+": ("HHIP", "NPNT", "SOSTDC1", "ACTG2", "ACTA2", "MYH11", "TAGLN"),
    "Neutrophils S100A8+ S100A9+": (
        "S100A8", "S100A9", "FCGR3B", "APOBEC3A", "S100A12", "FCN1", "ACSL1", "FPR2", "FPR1",
    ),
    "NK cells KLRF1+ CD3G-": ("KLRF1", "NCAM1", "KLRD1"),
    "Paneth cells": ("DEFA5", "DEFA6", "REG3A", "PRSS1", "ITLN2", "PLA2G2A"),
    "Pericytes HIGD1B+ STEAP4+": ("NOTCH3", "HIGD1B", "STEAP4", "COX4I2", "FABP4"),
    "Pericytes RERGL+ NTRK2+": ("NTRK2", "RERGL", "PLN", "NOTCH3"),
    "Plasma cells": (
        "IGJ", "MZB1", "IGLL5", "DERL3", "SSR4", "TNFRSF17", "FKBP11", "SEC11C", "ANKRD28", "AL928768.3",
    ),
    "Stem cells OLFM4+": ("OLFM4", "REG1A"),
    "Stem cells OLFM4+ GSTA1+": (
        "FABP1", "GSTA1", "AKR1C3", "KRT19", "MAOA", "CES2", "CBR1", "RBP2", "PTGR1", "LIMA1",
    ),
    "Stem cells OLFM4+ LGR5+": ("LGR5", "OLFM4"),
    "Stem cells OLFM4+ PCNA+": ("PCNA", "RANBP1", "OLFM4", "STRA13", "DUT", "SIVA1"),
    "Stromal Cycling cells": (
        "HMGB2", "UBE2C", "PTTG1", "TOP2A", "MKI67", "CDC20", "H2AFZ", "CCNB1", "BIRC5", "NUSAP1",
    ),
    "T cells CD4+ FOSB+": ("CD4", "FOSB", "IL7R", "RORA", "CD2"),
    "T cells CD4+ IL17A+": ("IL17A", "IL22", "CXCR6", "CCL20"),
    "T cells CD8+": ("CD8A", "CD8B"),
    "T cells CD8+ KLRG1+": ("KLRG1", "GZMH", "IFNG", "CD8B", "CD8A"),
    "T cells Naive CD4+": ("CCR7", "SELL", "TCF7"),
    "T cells OGT+": ("OGT", "MIAT", "CELF2", "RORA", "ANKRD44", "ARAP2", "AKNA", "CBLB"),
    "Tregs": ("CTLA4", "TIGIT", "TBC1D4", "BATF", "TNFRSF4"),
    "Tuft cells": ("SH2D6", "LRMP", "7SK_ENSG00000260682", "AVIL", "BMX", "AZGP1", "MATK", "TRPM5"),
}

# Colon benchmark labels (annotation of the prepared colon data) and the
# Table S2 subset that each label corresponds to.
COLON_LABEL_TO_KONG_SUBSET: dict[str, str] = {
    "Enterocytes BEST4": "Enterocytes BEST4+",
    "Enterocytes CA1 CA2 CA4-": "Enterocytes CA1+ CA2+ CA4-",
    "Enterocytes TMIGD1 MEP1A": "Enterocytes TMIGD1+ MEP1A+",
    "Enteroendocrine cells": "Enteroendocrine cells",
    "Epithelial Cycling cells": "Epithelial Cycling cells",
    "Goblet cells MUC2 TFF1": "Goblet cells MUC2+ TFF1+",
    "Goblet cells MUC2 TFF1-": "Goblet cells MUC2+ TFF1-",
    "Goblet cells SPINK4": "Goblet cells SPINK4+",
    "Paneth cells": "Paneth cells",
    "Stem cells OLFM4": "Stem cells OLFM4+",
    "Stem cells OLFM4 LGR5": "Stem cells OLFM4+ LGR5+",
    "Stem cells OLFM4 PCNA": "Stem cells OLFM4+ PCNA+",
    "Tuft cells": "Tuft cells",
}

FASOLINO_TABLE_S17: dict[str, tuple[str, ...]] = {
    "Beta Cells": ("INS", "IAPP"),
    "Alpha Cells": ("GCG", "GC", "TTR"),
    "Delta Cells": ("LEPR", "PRG4", "SST"),
    "PP/Gamma Cells": ("CARTPT", "PCDH10", "PLAC8", "PPY"),
    "Epsilon Cells": ("GHRL",),
    "Acinar Cells": ("CPA1", "CPA2"),
    "Ductal Cells": ("SFRP5", "MMP7"),
    "Stellates_MesenchymalCells": ("RGS10", "THY1"),
    "Endothelial Cells": ("VWF",),
    "Immune Cells": ("PTPRC",),
}

# Pancreas benchmark labels that each Fasolino cell type covers.
PANCREAS_TYPE_TO_LABELS: dict[str, tuple[str, ...]] = {
    "Beta Cells": ("beta_major", "beta_minor"),
    "Alpha Cells": ("alpha",),
    "Delta Cells": ("delta",),
    "PP/Gamma Cells": ("pp",),
    "Epsilon Cells": ("epsilon",),
    "Acinar Cells": ("acinar", "acinar_minor_mhcclassII", "duct_acinar_related"),
    "Ductal Cells": ("duct_major", "duct_acinar_related"),
    "Stellates_MesenchymalCells": ("stellates", "immune_stellates"),
    "Endothelial Cells": ("endothelial",),
    "Immune Cells": ("immune_stellates",),
}


def invert(markers_by_type: dict[str, tuple[str, ...]], labels_by_type: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    """Map each gene to every benchmark label whose cell type lists it."""
    panel: dict[str, set[str]] = {}
    for cell_type, genes in markers_by_type.items():
        for gene in genes:
            panel.setdefault(gene, set()).update(labels_by_type[cell_type])
    return {gene: tuple(sorted(labels)) for gene, labels in sorted(panel.items())}


COLON_SOURCE_PANEL = invert(
    {label: KONG_TABLE_S2[subset] for label, subset in COLON_LABEL_TO_KONG_SUBSET.items()},
    {label: (label,) for label in COLON_LABEL_TO_KONG_SUBSET},
)
PANCREAS_SOURCE_PANEL = invert(FASOLINO_TABLE_S17, PANCREAS_TYPE_TO_LABELS)

_GOBLET = ("Goblet cells MUC2 TFF1", "Goblet cells MUC2 TFF1-", "Goblet cells SPINK4")
_STEM = ("Stem cells OLFM4", "Stem cells OLFM4 LGR5", "Stem cells OLFM4 PCNA")
COLON_ORIGINAL_PANEL: dict[str, tuple[str, ...]] = {
    **{gene: ("Enterocytes BEST4",) for gene in ("BEST4", "OTOP2", "CA7", "GUCA2A", "GUCA2B")},
    **{gene: ("Enterocytes CA1 CA2 CA4-",) for gene in ("CA1", "CA2")},
    **{gene: ("Enterocytes TMIGD1 MEP1A",) for gene in ("TMIGD1", "MEP1A")},
    **{gene: _GOBLET for gene in ("MUC2", "TFF3", "AGR2", "SPDEF")},
    "TFF1": ("Goblet cells MUC2 TFF1",),
    "SPINK4": ("Goblet cells SPINK4",),
    **{gene: ("Tuft cells",) for gene in ("POU2F3", "DCLK1")},
    **{gene: ("Enteroendocrine cells",) for gene in ("CHGA", "GCG", "GIP", "CCK")},
    **{gene: ("Paneth cells",) for gene in ("LYZ", "DEFA5")},
    **{gene: ("Epithelial Cycling cells",) for gene in ("MKI67", "PCNA")},
    **{gene: _STEM for gene in ("OLFM4", "LGR5", "ASCL2", "SOX9")},
}

PANCREAS_ORIGINAL_PANEL: dict[str, tuple[str, ...]] = {
    **{gene: ("beta_major", "beta_minor") for gene in ("INS", "IAPP", "PCSK1", "PCSK2", "MAFA", "NKX6-1", "PDX1")},
    **{gene: ("alpha",) for gene in ("GCG", "TTR")},
    "SST": ("delta",),
    "PPY": ("pp",),
    "GHRL": ("epsilon",),
    **{gene: ("acinar", "acinar_minor_mhcclassII", "duct_acinar_related") for gene in ("PRSS1", "PRSS2", "REG1A", "REG1B", "CPA1", "CTRB2")},
    **{gene: ("duct_major", "duct_acinar_related") for gene in ("KRT8", "KRT18", "KRT19", "MUC1", "KRT17", "KRT7")},
    **{gene: ("stellates", "immune_stellates") for gene in ("COL1A1", "COL1A2", "COL3A1", "SPARC", "DCN")},
    **{gene: ("endothelial",) for gene in ("KDR", "EMCN", "PLVAP", "VWF")},
    **{gene: ("immune_stellates",) for gene in ("PTPRC", "CD74", "HLA-DRA", "HLA-DPA1", "HLA-DPB1")},
}

PANELS = {
    "colon": {"source": COLON_SOURCE_PANEL, "original": COLON_ORIGINAL_PANEL},
    "pancreas": {"source": PANCREAS_SOURCE_PANEL, "original": PANCREAS_ORIGINAL_PANEL},
}
