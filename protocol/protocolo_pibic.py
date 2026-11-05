# -*- coding: utf-8 -*-
"""
Protocol shared by the four models (M1, M2, M3, M4).

Keeps in one place the logic that must be identical across notebooks:

1. Partitions are LOADED from disk (split_{corpus}.json), never regenerated, and
   the test partitions are cleaned of internal duplicates and cross-corpus overlap.
2. Every cell, diagonal and transfer alike, is evaluated on the test partition.
3. Predicted probabilities are saved to .npz, so AUC and the three threshold
   regimes (R1 fixed 0.5, R2 prevalence-matched, R3 oracle) can be recomputed
   without retraining.

Usage
-----
    import sys; sys.path.insert(0, '<folder containing this file>')
    import protocolo_pibic as PP
"""

import os, re, json, unicodedata
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, precision_recall_fscore_support,
                             accuracy_score, roc_auc_score,
                             average_precision_score, confusion_matrix)

SEMENTE_SPLIT = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.10

# Canonical order of the corpora.
SIGLAS = ['PT1', 'PT2', 'PT3', 'EN', 'ID', 'KZ']
LINGUA = {'PT1': 'pt', 'PT2': 'pt', 'PT3': 'pt', 'EN': 'en', 'ID': 'id', 'KZ': 'kz'}


# ══════════════════════════════════════════════════════════════════════════════
#  1 · Normalisation and duplicate keys
# ══════════════════════════════════════════════════════════════════════════════

def sem_acento(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(s))
                   if not unicodedata.combining(c))


def chave_norm(s):
    """Key for detecting duplicates within and across corpora (same as M3/M4)."""
    s = sem_acento(str(s)).lower()
    s = re.sub(r'http\S+|www\.\S+', ' ', s)
    s = re.sub(r'[^a-z0-9\u0400-\u04FF ]', ' ', s)
    return ' '.join(s.split())


def sanitizar(t):
    """Remove lone Unicode surrogates (they break Rust tokenizers)."""
    t = str(t)
    try:
        t.encode('utf-8')
        return t
    except UnicodeEncodeError:
        return t.encode('utf-8', 'ignore').decode('utf-8')


def baseline_majoritaria(rotulos):
    """Macro-F1 of the classifier that always predicts the majority class."""
    y = np.asarray(rotulos)
    prev = float(y.mean())
    p = max(prev, 1 - prev)
    return 0.5 * (2 * p / (1 + p))


# ══════════════════════════════════════════════════════════════════════════════
#  2 · ToLD-Br (PT3)
# ══════════════════════════════════════════════════════════════════════════════
#
# Leite, Silva, Bontcheva & Scarton (2020), AACL-IJCNLP. License: CC BY-SA 4.0.
# Download:
#   https://github.com/joaoaleite/ToLD-Br  ->  ToLD-BR.csv
#
# Format: a `text` column + six category columns (homophobia, obscene,
# insult, racism, misogyny, xenophobia), each holding the NUMBER OF ANNOTATORS
# (0 to 3) who marked that category.
#
# Binarisation: problematic if any category reaches `limiar_anotadores`.
#   >= 1 annotator  -> 44.1% positives
#   >= 2 (majority) -> 19.3%  <- DEFAULT
#   >= 3 (unanimous) ->  5.6%
#
# Simple majority (>=2) is the usual aggregation for offensiveness corpora: a
# single dissenting annotator does not decide the label. It also gives 19.3%
# positives, almost the same as the 19.6% of OffComBR-3, so the PT2 x PT3 pair
# is a domain contrast without a base-rate confounder.

CATEGORIAS_TOLDBR = ['homophobia', 'obscene', 'insult',
                     'racism', 'misogyny', 'xenophobia']


def carregar_toldbr(caminho, limiar_anotadores=2, n_max=7000,
                    semente=SEMENTE_SPLIT, verboso=True):
    """
    Load ToLD-Br and return (texts, labels, metadata).

    n_max : stratified subsampling. ToLD-Br has 21,000 examples against 7,000
            in HateBR; matching the sizes removes corpus size as a competing
            explanation for domain effects. Use n_max=None for the full corpus.
    """
    import pandas as pd

    df = pd.read_csv(caminho)
    faltando = [c for c in ['text'] + CATEGORIAS_TOLDBR if c not in df.columns]
    if faltando:
        raise ValueError(f'Columns missing in ToLD-Br: {faltando}')

    y = (df[CATEGORIAS_TOLDBR].max(axis=1) >= limiar_anotadores).astype(int).values
    textos = [sanitizar(t) for t in df['text'].astype(str)]
    n_total = len(textos)

    if n_max is not None and n_total > n_max:
        idx = np.arange(n_total)
        idx_sel, _ = train_test_split(idx, train_size=n_max,
                                      random_state=semente, stratify=y)
        idx_sel = np.sort(idx_sel)
        textos = [textos[i] for i in idx_sel]
        y = y[idx_sel]

    meta = dict(n_original=n_total, n_usado=len(textos),
                limiar_anotadores=limiar_anotadores,
                prevalencia=float(y.mean()),
                subamostrado=bool(n_max is not None and n_total > n_max))
    if verboso:
        print(f"ToLD-Br: {meta['n_usado']:,} of {meta['n_original']:,} examples "
              f"| positives = {meta['prevalencia']*100:.1f}% "
              f"(threshold >= {limiar_anotadores} annotator(s))")
        if meta['subamostrado']:
            print(f"  stratified subsample to match the size of HateBR "
                  f"(seed {semente})")
    return textos, y.tolist(), meta


# ══════════════════════════════════════════════════════════════════════════════
#  3 · Shared partitions
# ══════════════════════════════════════════════════════════════════════════════

def obter_particoes(sigla, textos, rotulos, split_dir,
                    chaves_outros_corpora=None, verboso=True):
    """
    Return (idx_train, idx_val, idx_test), loading from disk when available.

    If `split_{sigla}.json` exists and the corpus size matches, the file is
    loaded and NOTHING is regenerated.
    """
    caminho = os.path.join(split_dir, f'split_{sigla}.json')
    n = len(rotulos)

    if os.path.exists(caminho):
        with open(caminho) as f:
            sp = json.load(f)
        if sp.get('n') == n:
            if verboso:
                print(f'  {sigla}: partition loaded from {os.path.basename(caminho)}')
            return sp['treino'], sp['val'], sp['teste']
        print(f'  {sigla}: size changed ({sp.get("n")} -> {n}); regenerating. '
              f'WARNING: earlier results for this corpus are no longer comparable.')

    idx = np.arange(n)
    y = np.array(rotulos)
    tr_full, te = train_test_split(idx, test_size=TEST_SIZE,
                                   random_state=SEMENTE_SPLIT, stratify=y)
    tr, va = train_test_split(tr_full, test_size=VAL_SIZE,
                              random_state=SEMENTE_SPLIT, stratify=y[tr_full])

    chaves = [chave_norm(t) for t in textos]
    vistas = set(chaves[i] for i in tr) | set(chaves[i] for i in va)
    externas = set()
    for outras in (chaves_outros_corpora or {}).values():
        externas |= set(outras)

    limpo, desc_int, desc_ext, local = [], 0, 0, set()
    for i in te:
        ch = chaves[i]
        if ch in vistas or ch in local:
            desc_int += 1; continue
        if ch and ch in externas:
            desc_ext += 1; continue
        local.add(ch); limpo.append(int(i))
    if verboso and (desc_int or desc_ext):
        print(f'  {sigla}: test cleaned — {desc_int} internal duplicates, '
              f'{desc_ext} overlaps with other corpora')
    te = np.array(limpo)

    sp = dict(n=n, treino=[int(i) for i in tr], val=[int(i) for i in va],
              teste=[int(i) for i in te], semente_split=SEMENTE_SPLIT,
              gerado_em=datetime.now().isoformat())
    os.makedirs(split_dir, exist_ok=True)
    with open(caminho, 'w') as f:
        json.dump(sp, f)
    if verboso:
        print(f'  {sigla}: partition GENERATED and saved to {os.path.basename(caminho)}')
    return sp['treino'], sp['val'], sp['teste']


def conferir_particoes(siglas, split_dir):
    """Warn if any partition did not come from disk."""
    faltando = [s for s in siglas
                if not os.path.exists(os.path.join(split_dir, f'split_{s}.json'))]
    if faltando:
        print(f'\n⚠ Partitions missing on disk: {faltando}')
        print('  If another model has already run, the comparison between models is compromised.')
        return False
    print('\n✓ All partitions came from disk — the comparison between models is valid.')
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  4 · Persistence of predicted probabilities
# ══════════════════════════════════════════════════════════════════════════════

def dir_predicoes(out_dir, modelo, tag):
    d = os.path.join(out_dir, f'predicoes_{modelo}_{tag}')
    os.makedirs(d, exist_ok=True)
    return d


def salvar_predicoes(pred_dir, src, tgt, semente, probs, y, cenario='base'):
    """
    Save predicted probabilities to .npz (same format in all four models).

    Needed to recompute thresholds, AUC, bootstrap and the R2/R3 regimes
    without retraining.
    """
    sufixo = '' if cenario == 'base' else f'_{cenario}'
    arq = os.path.join(pred_dir, f'{src}{sufixo}_to_{tgt}_s{semente}.npz')
    np.savez_compressed(arq, probs=np.asarray(probs, dtype=np.float32),
                        y=np.asarray(y, dtype=np.int8))
    return arq


# ══════════════════════════════════════════════════════════════════════════════
#  5 · Metrics and threshold regimes
# ══════════════════════════════════════════════════════════════════════════════

def grade_candidata(probs, n=512):
    """
    Threshold candidates taken from the quantiles of the probabilities.

    A fixed linear grid fails with overconfident models: the probability mass sits
    near 0 and 1 and the optimal threshold can be 0.9987, which the grid never
    visits; the oracle (an upper bound) could then fall below R2.
    """
    probs = np.asarray(probs, dtype=np.float64)
    q = np.quantile(probs, np.linspace(0.0005, 0.9995, n))
    g = np.unique(np.concatenate([q, np.array([0.5]), np.linspace(0.01, 0.99, 99)]))
    return np.clip(g - 1e-9, 0.0, 1.0)


def escolher_limiar(p_val, y_val):
    """Threshold that maximises macro-F1 on SOURCE validation. Never on test."""
    melhor, melhor_f1 = 0.5, -1.0
    for t in grade_candidata(p_val):
        f = f1_score(y_val, (p_val >= t).astype(int), average='macro', zero_division=0)
        if f > melhor_f1:
            melhor, melhor_f1 = float(t), f
    return melhor


def limiar_por_prevalencia(probs, prevalencia_alvo):
    """R2: cut that matches the predicted positive rate to the target base rate.
    Uses only the prevalence, never the individual labels."""
    return float(np.quantile(probs, 1.0 - prevalencia_alvo))


def limiar_oraculo(probs, y):
    """R3: UPPER BOUND, not a result. Maximises macro-F1 on the target labels."""
    melhor, melhor_f1 = 0.5, -1.0
    for t in grade_candidata(probs):
        f = f1_score(y, (probs >= t).astype(int), average='macro', zero_division=0)
        if f > melhor_f1:
            melhor, melhor_f1 = float(t), f
    return melhor


def metricas(y, p, limiar=0.5):
    """Metrics computed from the probabilities (same set in all four models)."""
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64)
    yhat = (p >= limiar).astype(int)
    pr, rc, f1, sup = precision_recall_fscore_support(
        y, yhat, labels=[0, 1], zero_division=0)
    out = dict(
        macro_f1=float(f1_score(y, yhat, average='macro', zero_division=0)),
        f1_neg=float(f1[0]), f1_pos=float(f1[1]),
        prec_neg=float(pr[0]), prec_pos=float(pr[1]),
        rec_neg=float(rc[0]), rec_pos=float(rc[1]),
        suporte_neg=int(sup[0]), suporte_pos=int(sup[1]),
        acuracia=float(accuracy_score(y, yhat)),
        taxa_predicao_positiva=float(yhat.mean()),
        limiar=float(limiar),
    )
    if len(np.unique(y)) > 1:
        out['auc_roc'] = float(roc_auc_score(y, p))
        out['auc_pr'] = float(average_precision_score(y, p))
    else:
        out['auc_roc'] = out['auc_pr'] = float('nan')
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out['matriz_confusao'] = dict(vn=int(tn), fp=int(fp), fn=int(fn), vp=int(tp))
    return out


def tres_regimes(y, p, prevalencia_alvo):
    """R1 (fixed 0.5), R2 (prevalence-matched) and R3 (oracle, upper bound)."""
    y = np.asarray(y); p = np.asarray(p, dtype=np.float64)
    r1 = f1_score(y, (p >= 0.5).astype(int), average='macro', zero_division=0)
    t2 = limiar_por_prevalencia(p, prevalencia_alvo)
    r2 = f1_score(y, (p >= t2).astype(int), average='macro', zero_division=0)
    t3 = limiar_oraculo(p, y)
    r3 = f1_score(y, (p >= t3).astype(int), average='macro', zero_division=0)
    return dict(R1_fixo=float(r1), R2_prevalencia=float(r2), R3_oraculo=float(r3),
                limiar_R2=float(t2), limiar_R3=float(t3),
                custo_calibracao=float(r3 - r1))


def grupo_cenario(src, tgt):
    """
    A = same language, different domain (6 cells over PT1/PT2/PT3)
    B = different language, without KZ
    C = maximum distance (involves KZ)
    """
    if src == tgt:
        return 'diagonal'
    if LINGUA[src] == LINGUA[tgt]:
        return 'A'
    if 'KZ' in (src, tgt):
        return 'C'
    return 'B'


print('protocolo_pibic loaded — corpora:', ', '.join(SIGLAS))
