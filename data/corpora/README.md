# Corpora

The corpora are not redistributed in this repository. Download each one from its original source
and save it under the file name expected by the notebooks.

| Code | Corpus | Source | License / terms | Expected file |
|---|---|---|---|---|
| PT1 | HateBR (Vargas et al., 2022) | https://github.com/franciellevargas/HateBR | CC BY-NC 4.0; research use only | `HateBR.csv` |
| PT2 | OffComBR-3 (de Pelle & Moreira, 2017) | https://github.com/rogersdepelle/OffComBR | no license stated | `OffComBR3.arff` |
| PT3 | ToLD-Br (Leite et al., 2020) | https://github.com/joaoaleite/ToLD-Br | CC BY-SA 4.0 (data), MIT (code) | `ToLD-BR.csv` |
| EN | Davidson et al. (2017) | https://github.com/t-davidson/hate-speech-and-offensive-language | MIT; the authors ask users to cite the paper | `DataIngles.csv` (`labeled_data.csv` in the original repository) |
| ID | Alfina et al. (2017) | https://github.com/ialfina/id-hatespeech-detection | free use; public copies not allowed without the authors' permission | `DataIndonesio.txt` (`IDHSD_RIO_unbalanced_713_2017.txt` in the original repository) |
| KZ | Bolatbek et al. (2025) | supplemental files of https://doi.org/10.7717/peerj-cs.3027 | article under CC BY 4.0 | `DataKazake.csv` |

## Binarisation

Positive class = problematic content.

| Code | Rule |
|---|---|
| PT1 | `label_final == 1` |
| PT2 | `yes` |
| PT3 | any of the six categories marked by at least 2 of the 3 annotators; stratified subsample of 7,000 with seed 42 |
| EN | `class != 2` (hate speech or offensive language); strict ablation: `class == 0` |
| ID | `HS` |
| KZ | `label != 0` |
