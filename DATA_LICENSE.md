# Dataset attribution and licensing

The analysis uses **Human Activity Recognition Using Smartphones**, created by Jorge Reyes-Ortiz, Davide Anguita, Alessandro Ghio, Luca Oneto and Xavier Parra (2013).

- Dataset DOI: https://doi.org/10.24432/C54S4K
- Provider: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- Official download: https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
- Original paper: [Anguita et al., ESANN 2013](https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf)

## Source-license discrepancy

The UCI landing page checked on **3 November 2026** lists **Creative Commons Attribution 4.0 International (CC BY 4.0)**. The historical `UCI HAR Dataset/README.txt` inside the downloaded archive includes the sentence “Any commercial use is prohibited.” These source statements differ.

This project records both statements. It does not resolve that discrepancy, redistribute the raw archive, or apply its code license to the source dataset. Anyone planning reuse beyond this educational analysis should consult the provider's applicable terms and obtain clarification when needed.

The downloader obtains the data directly from UCI. The `.gitignore` excludes the local `data/` directory. Keep the original attribution and source documentation with downloaded data. This repository contains analysis code, aggregate outputs and an indexed feature-name manifest; the dataset authors retain their data rights.

## Provenance

The analysis verifies the SHA-256 of the original inner ZIP payload:

```text
2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0
```

The downloaded UCI wrapper ZIP has SHA-256:

```text
c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031
```

The audit also records the README and feature-documentation hashes, original split participants, class counts and feature-name handling. See [dataset_audit.json](results/dataset_audit.json).

The repository's [MIT license](LICENSE) covers its original analysis code. It does not replace third-party package licenses or grant additional rights to source data.
