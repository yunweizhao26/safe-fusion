# Data used by the paper

The source data are public and are not stored in Git. The table lists each
source, the path that the preparation scripts read or write, and its
checksum.

| Study | Public source | Path | Checksum (SHA-256 unless noted) |
|---|---|---|---|
| Colon epithelium | [CELLxGENE](https://cellxgene.cziscience.com/), artifact `63ff2c52-cb63-44f0-bac3-d0b33373e312` | Source: `external_data/cellxgene/63ff2c52-cb63-44f0-bac3-d0b33373e312.h5ad`; prepared: `external_data/prepared/colon_epithelial.h5ad` | Source `612e03925bd5860e1ec81b0295af40a7f9e4111638f8f41ccb2fe976f428f053`; prepared `532261970fca1061cd00883054c86858ffdfd37e863e13a56e6ff5c2f7f6896f` |
| Pancreatic islets | [CELLxGENE](https://cellxgene.cziscience.com/), artifact `f89a618b-fe4b-404e-bd39-7c574529b1f5` | Source: `external_data/cellxgene/f89a618b-fe4b-404e-bd39-7c574529b1f5.h5ad`; prepared: `external_data/prepared/pancreas_islets.h5ad` | Source `7f0302126d35301770ba8a21eb773f80fe576b03617b441f92f7f6d1f114f0a5`; prepared `02e68751d0add44ba6e6e0f694d14fbaa4413ba6ce2d5614f414474b9ff58ede` |
| Norman CRISPRa | [GEO GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344), as the GEARS-processed file `perturb_processed.h5ad` with its raw UMI counts layer | Prepared: `external_data/prepared/norman_crispra.h5ad` | Prepared `a746950c8834037972a15b4da011be55d852af7b7f9415707db714c3c02788ec` |
| Adamson CRISPRi | [GEO GSE90546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546), harmonized by scPerturb (Zenodo record 10044268) | `external_data/scperturb/AdamsonWeissman2016_GSM2406681_10X010.h5ad` | `e70fcd49808cab8d724de8d5a332940911206e1c8ef44cc7b568d048ed795c85` |
| Dixit knockout | [GEO GSE90063](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90063), harmonized by scPerturb | `external_data/scperturb/DixitRegev2016.h5ad` | `a448af6aa250a6cdca472d26bcba5aec857381d8aeb03024b9358aebc2d6d23e` |
| Papalexi ECCITE-seq | [GEO GSE153056](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153056), harmonized by scPerturb | `external_data/scperturb/PapalexiSatija2021_eccite_RNA.h5ad` | `03e9602d124c261281936717d5795445805e6e704a52b0310bcde94225aa0929` |
| Papalexi RNA and surface protein | MuData object distributed with pertpy ([figshare file 36509460](https://ndownloader.figshare.com/files/36509460)) and antibody counts from [GEO GSE153056](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153056) | `external_data/papalexi_multimodal/papalexi.h5mu`, `external_data/papalexi_multimodal/GSE153056_RAW.tar` | MD5 `419ec1f14615b4423edd6f16b65c55a7` (MuData); SHA-256 `ded5067ee0365da6d9f611f321fcbd5c46663a5c9ab7524ec2e9f9a729d5f56c` (GEO archive) |
| Zebrafish axial mesoderm | [CellRank datasets](https://cellrank.readthedocs.io/) | `external_data/trajectory/zebrafish_embryogenesis_axial_mesoderm.h5ad` | `991b1f060f7f413c95b36f8e9309a43c370b9aa121d5dc6bf6b21e91a6c6342c` |

The preparation entry points are `scripts/prepare_*.py`,
`scripts/setup_norman_crispra_experiment.py` and
`scripts/make_pancreas_crossfit_splits.py`. Each preparation report
(`external_data/prepared/<dataset>.report.json`) records the SHA-256 of its
source. The workflow rejects a prepared colon or pancreas file whose SHA-256
differs from its config, and each workflow run stores source hashes, cell and
gene order, split identifiers, command lines and seeds under its
`provenance/` directory. `scripts/slurm_fetch_papalexi_multimodal.sh`
downloads the two RNA and protein files, and
`scripts/slurm_audit_papalexi_multimodal.sh` checks that their cells,
antibody counts and guide labels agree. `scripts/slurm_standard_imputers.sh`
downloads the colon source file for Supplementary Table S1 and checks its
SHA-256. [REPRODUCTION.md](REPRODUCTION.md), section 2, gives the commands.

The repository does not redistribute the source datasets.
`data/papalexi_crossmodal_inputs.tar.gz` (Git LFS) holds the derived inputs
of the RNA-protein analysis. [data/README.md](../data/README.md) describes
its contents and checksum.
