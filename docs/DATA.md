# Data used by the paper

Raw and prepared matrices are not stored in Git. Paper runs reject inputs whose
SHA-256 hashes do not match the locked configs.

| Study | Public source | Prepared path | SHA-256 |
|---|---|---|---|
| Colon epithelium | [CELLxGENE](https://cellxgene.cziscience.com/), artifact `63ff2c52-cb63-44f0-bac3-d0b33373e312` | `external_data/prepared/colon_epithelial.h5ad` | `532261970fca1061cd00883054c86858ffdfd37e863e13a56e6ff5c2f7f6896f` |
| Pancreatic islets | [CELLxGENE](https://cellxgene.cziscience.com/), artifact `f89a618b-fe4b-404e-bd39-7c574529b1f5` | `external_data/prepared/pancreas_islets.h5ad` | `02e68751d0add44ba6e6e0f694d14fbaa4413ba6ce2d5614f414474b9ff58ede` |
| PBMC CITE-seq | [GEO GSE100501](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100501) | `external_data/prepared/pbmc_citeseq.h5ad` | `f8672f72bc8fce7ea932d01e3032e2d90b0f54aacf6da19acd182667b8c7a105` |
| Norman CRISPRa | [GEO GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344) | `external_data/prepared/norman_crispra.h5ad` | recorded by preparation manifest |
| Adamson CRISPRi | [GEO GSE90546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546) | `external_data/scperturb/AdamsonWeissman2016_GSM2406681_10X010.h5ad` | `e70fcd49808cab8d724de8d5a332940911206e1c8ef44cc7b568d048ed795c85` |
| Dixit KO | [GEO GSE90063](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063) | `external_data/scperturb/DixitRegev2016.h5ad` | `a448af6aa250a6cdca472d26bcba5aec857381d8aeb03024b9358aebc2d6d23e` |
| Papalexi ECCITE-seq | [GEO GSE153056](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153056) | `external_data/scperturb/PapalexiSatija2021_eccite_RNA.h5ad` | `03e9602d124c261281936717d5795445805e6e704a52b0310bcde94225aa0929` |
| Zebrafish trajectory | [CellRank datasets](https://cellrank.readthedocs.io/) | `external_data/trajectory/zebrafish_embryogenesis_axial_mesoderm.h5ad` | `991b1f060f7f413c95b36f8e9309a43c370b9aa121d5dc6bf6b21e91a6c6342c` |

Preparation entry points are `scripts/prepare_*`. Exact downstream commands
are in [REPRODUCTION.md](REPRODUCTION.md). Each generated run stores source
hashes, cell and gene ordering, split identifiers, command lines, environment
locks, and random seeds under its `provenance/` directory.

The repository does not redistribute datasets governed by their source terms.
