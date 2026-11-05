# Language or domain? Transfer costs in multilingual detection of problematic content

Code, data partitions and predictions for the paper *Language or domain? Transfer costs in
multilingual detection of problematic content* (under review).

The study evaluates six corpora (three Portuguese, one English, one Indonesian, one Kazakh) on all
36 train/test combinations under four architectures of increasing multilingual capacity, with
shared partitions, seeds and threshold policy, and measures a domain cost and a
language-associated transfer cost as retention normalised by the trivial floor of each corpus.

## Repository layout

```
notebooks/          M1–M4 training and evaluation (Google Colab, outputs cleared)
  M1_bow.ipynb              bag-of-words + feed-forward network; also runs the NLLB-200 translations
  M2_bilstm_fasttext.ipynb  BiLSTM + frozen FastText vectors (reads the M1 translation cache)
  M3_mbert.ipynb            mBERT, full fine-tuning, plus the strict-English ablation
  M4_xlmr.ipynb             XLM-RoBERTa, full fine-tuning, plus the strict-English ablation
protocol/
  protocolo_pibic.py        logic shared by the four notebooks (partitions, ToLD-Br loader, metrics)
analysis/           re-analysis of the saved predictions (no GPU needed)
data/
  splits/                   train/validation/test indices of each corpus (no text)
  corpora/README.md         where to download each corpus and its license
results/
  predictions/              predicted probabilities of every cell and seed (.npz: probs, y)
  tables/                   per-run results written by the notebooks (JSON and CSV)
blind_validation/           answers of the blind translation validation (ids only, no text)
figures/                    output folder of the figure scripts
```

## Data

The corpora are not redistributed here. Download them from the original sources listed in
[`data/corpora/README.md`](data/corpora/README.md) and place the files in `pibic/dados/`
(or `MyDrive/pibic/dados/` on Colab) with these names: `HateBR.csv`, `OffComBR3.arff`,
`ToLD-BR.csv`, `DataIngles.csv`, `DataIndonesio.txt`, `DataKazake.csv`.

`data/splits/split_{corpus}.json` holds the indices of each partition, relative to the row order
of the original files after loading (for ToLD-Br, after the stratified subsample of 7,000 with seed 42
done by `protocolo_pibic.carregar_toldbr`). Copy the folder to `pibic/splits/`; the notebooks load
it and never regenerate it. `data/splits/SHA256SUMS` holds the SHA-256 of each partition file
(`sha256sum -c SHA256SUMS` from that folder).

## Reproducing the paper

### 1. Training (optional, GPU)

Run the notebooks in the order M1, M2, M3, M4 on Google Colab with `protocolo_pibic.py` and
`splits/` in `MyDrive/pibic/`. M1 produces the translation cache used by M2. Every cell writes
its predicted probabilities to `pibic/resultados/predicoes_{model}_{timestamp}/`.

The runs used in the paper are already in `results/predictions/`:

| Model | Run | Notes |
|---|---|---|
| M1 | `predicoes_M1_20260806_0117` | zero-shot, translate-test (`_mt`) and translate-train (`_tt`) |
| M2 | `predicoes_M2_20260806_0646` | same scenarios as M1 |
| M3 | `predicoes_M3_20260806_0327` | zero-shot; `PT1_noacc` = de-accented PT1 ablation |
| M4 | `predicoes_M4_20260806_0416` | zero-shot; `PT1_noacc` = de-accented PT1 ablation |
| M3 | `predicoes_M3_20260819_1634` | rerun on the same partitions and seeds, with `EN_ESTRITO` (strict English label) |
| M4 | `predicoes_M4_20260819_2103` | same, for M4 |

### 2. Analysis (CPU)

From `analysis/`:

```bash
B=10000 SAIDA=final_prev_B10000_v2.json python3 bootstrap_rodada1.py               # costs, differences, gamma, PT3<->EN control
B=10000 REGIME=fixo SAIDA=final_fixo_B10000_v2.json python3 bootstrap_rodada1.py   # same under the fixed 0.5 threshold
python3 figuras_bootstrap.py                                                        # per-cell bootstrap (1,500 replicates)
python3 figuras_bootstrap_sementes.py                                               # paired XLM-R vs mBERT test (10,000 replicates)
python3 figura5.py && python3 figuras.py                                            # figures, as vector PDF in ../figures/
python3 em_saerens.py                                                               # costs with EM-estimated prevalence
python3 tabelas_complementares.py                                                   # remaining point estimates
```

The bootstrap scripts take 10 to 40 minutes each; their outputs are included, so the figure
scripts can be run directly. All scripts use seed 20261030.

### Where each result comes from

| Result in the paper | Source |
|---|---|
| Domain and language-associated costs, gamma, PT3<->EN control, cost differences | `final_prev_B10000_v2.json` |
| Costs under the fixed threshold | `final_fixo_B10000_v2.json` |
| Costs with EM-estimated prevalence | `em_saerens.json` |
| XLM-R minus mBERT with Holm correction | `figura_holm.json` |
| Retention by group, Portuguese sources | `figura_grupos.json`, `figura_fontes.json`, `tabelas_complementares.json` |
| Diagonal table, cells below the floor, calibration cost, ECE, per-class performance (mean and per cell, `por_classe_celula`), translation table, alternative retention formulations, gamma sensitivity, oracle check, uncertainty decomposition | `tabelas_complementares.json` |
| Surface-shortcut table, duplicates, tokenizer fertility | `diagnosticos` field of `results/tables/M3_mBERT_resultados_20260806_0327.json` and `M4_XLMR_resultados_20260806_0416.json` |
| Strict-English ablation | `results/tables/ablacao_*.csv` |
| Blind validation of translations, agreement between annotators | `blind_validation/preservation.py`, `blind_validation/agreement.py` |

## Notes

- The notebooks are the versions that produced the runs above. For publication, comments and
  markdown were translated into English, outputs were cleared, and file paths now fall back to a
  local `./pibic` folder when Google Drive is not available. Console messages and plot labels were
  translated as well; variable names and the keys of the result files are in Portuguese.
- The Indonesian corpus is the dataset of Alfina et al. (2017). Earlier versions of the notebooks
  labelled it after Isnain et al. (2020), whose BiLSTM is the M2 architecture; the result files
  in `results/tables/` keep that label.
- An earlier M2 run (`20260806_0207`, not included) trained the three seeds identically because the
  seed was not passed to `treinar_bilstm`. The included run `20260806_0646` fixes this.
- Exported error-analysis files contain corpus texts and are not included.
- `blind_validation/` holds the answers of the two blind annotators of the translation validation
  (`annotator1_answers.csv`, `annotator2_answers.csv`), the key with the translation direction and
  gold label of each item (`key.csv`, which also holds the automatic triage label of the 80 items
  it covered), `preservation.py` (preservation table, by annotator) and
  `agreement.py` (agreement between annotators). The texts are machine translations of corpus items
  and are not distributed.

## Licenses

Code: MIT (see `LICENSE`). Predictions and partitions: CC BY 4.0. The corpora keep their own
licenses (see `data/corpora/README.md`).
