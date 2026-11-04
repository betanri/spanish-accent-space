#!/usr/bin/env python3
"""All measurements for all places, laid out like an alignment (Supplement S2 of the reports).

Rows: every place in the space (places with fewer than MIN_SPK speakers last, marked *), ordered from the center of the
accent space (all places) to the edge. Columns: the measurements of the many-place space, grouped as melody, rhythm,
vowels, variants and consonant sounds. Cell: the place average (z-score, as in 05_space.py) relative to all speakers.

usage: 46_alignment_figure.py <speakers.csv> <meta.csv> <exclude,features>   (MIN_SPK env; tools as with TOOLS_ZONE=1)
writes results/c44/alignment_all.png and alignment_all_es.png
"""
import os, sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
space = import_module("05_space")
drift = import_module("42_drift_test")
COV = ROOT / "results/c44/alignment_coverage.csv"
BLOCKS = ["intonation", "timing", "vowels", "variants", "consonant acoustics"]
BNAME = {"intonation": ("Melody", "Melodía"), "timing": ("Rhythm", "Ritmo"), "vowels": ("Vowels", "Vocales"),
         "variants": ("Variants", "Variantes"), "consonant acoustics": ("Consonant sounds", "Sonido de las consonantes")}


def main():
    sp, meta, excl = sys.argv[1:4]
    min_spk = int(os.environ.get("MIN_SPK", 15))
    S, M = pd.read_csv(ROOT / sp), pd.read_csv(ROOT / meta)
    Da, Za, feats, core = space.prepare_with_extras(S, M, min_spk, 5, set(excl.split(",")), failed=set())
    VM = Za.groupby(Da.variety.values).mean()
    n = Da.variety.value_counts()
    R = pd.read_csv(ROOT / "results/c44_centres/centre_distances.csv")
    R = R[R.centre == "all places"].set_index("place")
    rows = sorted(VM.index, key=lambda v: (n[v] < min_spk, R.dist.get(v, 99)))
    cols = sorted(feats, key=lambda f: (BLOCKS.index(space.block_of(f)), feats.index(f)))
    Z = VM.loc[rows, cols].values.astype(float)
    # how many people stand behind each cell (a person has no value for a sound with too few examples)
    has = Za[core].notna()
    per_cell = has.groupby(Da.variety[core].values).sum()
    pd.DataFrame({"feature": cols, "block": [space.block_of(f) for f in cols],
                  "share_people": [has[f].mean() for f in cols],
                  "cell_min": [int(per_cell[f].min()) for f in cols], "cell_median": [float(per_cell[f].median()) for f in cols],
                  "cell_max": [int(per_cell[f].max()) for f in cols]}).to_csv(COV, index=False)
    per_cell.to_csv(ROOT / "results/c44/alignment_cell_n.csv")
    rep = es_names = import_module("place_names")     # full place names, English and Spanish
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    cmap.set_bad("#d9d8d2")
    for es in (False, True):
        space.ES = es
        fig, ax = plt.subplots(figsize=(17, 15))
        im = ax.imshow(np.ma.masked_invalid(Z), cmap=cmap, vmin=-1.5, vmax=1.5, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(rows)))
        nm = es_names.lugar if es else rep.lab
        ax.set_yticklabels([nm(v) + ("" if n[v] >= min_spk else " *") for v in rows], fontsize=9)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([drift.fname(f, es) for f in cols], fontsize=8, rotation=60, ha="right", rotation_mode="anchor")
        start = 0
        for b in BLOCKS:
            k = sum(space.block_of(f) == b for f in cols)
            if start:
                ax.axvline(start - 0.5, color="white", lw=2.5)
            ax.text(start + k / 2 - 0.5, -1.2, BNAME[b][es], ha="center", va="bottom", fontsize=12, weight="bold")
            start += k
        first_extra = sum(n[v] >= min_spk for v in rows)
        if first_extra < len(rows):
            ax.axhline(first_extra - 0.5, color="white", lw=4)
        ax.set_ylabel("del centro (arriba) a la orilla (abajo)" if es else "from the center (top) to the edge (bottom)",
                      fontsize=11)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.025, pad=0.13, aspect=60)
        cb.set_label("promedio del lugar frente a todos los lugares (desviaciones estándar; rojo = más, azul = menos)" if es
                     else "place average against all places (standard deviations; red = more, blue = less)", fontsize=10)
        note = (f"* menos de {min_spk} personas (solo se muestra)" if es else
                f"* fewer than {min_spk} people (shown only)")
        fig.text(0.01, 0.005, note, fontsize=9, color="#555555")
        fig.tight_layout()
        fig.savefig(ROOT / f"results/c44/alignment_all{'_es' if es else ''}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    print(len(rows), "places,", len(cols), "measurements")


def how_text(lang, name, example="Colombia_Cali"):
    """'How a cell is made' paragraphs for the reports (numbers from the last run)."""
    es = lang == "es"
    C = pd.read_csv(COV)
    N = pd.read_csv(ROOT / "results/c44/alignment_cell_n.csv", index_col=0)
    pct = lambda x: f"{100 * x:.0f}".replace(".", ",") + (" %" if es else "%")
    common = C[C.block.isin(["vowels", "intonation"])].share_people.median()
    rare = C.sort_values("share_people").iloc[0]
    ex = N.loc[example] if example in N.index else N.iloc[0]
    hi_f, lo_f = ex.idxmax(), ex.idxmin()
    fn = lambda f: drift.fname(f, es)[0].lower() + drift.fname(f, es)[1:]
    if es:
        return f"""<p><b>Cómo se hace una celda.</b> (1) Audio: unos 8 minutos de habla de cada persona, sin música, ruido ni la
voz del entrevistador. (2) Palabras: un reconocedor de voz (Whisper) escribe lo que dice. (3) Sonidos: un alineador (Montreal
Forced Aligner) encuentra dónde empieza y termina cada sonido. (4) Medidas: Praat mide cada sonido (el tono, los tiempos, las
vocales, y si cada /s/, j o r se debilita o cambia). (5) Un número por persona y medida: el promedio de todos sus ejemplos de
ese sonido (por ejemplo, qué parte de sus /s/ antes de consonante suena como h). (6) Se quita la diferencia entre mujeres y
hombres, y cada medida se pone en la misma escala (0 = la persona promedio; 1 = una desviación estándar). (7) La celda es el
promedio de las personas de ese lugar.</p>
<p><b>No todas las celdas tienen el mismo número de personas.</b> Los sonidos comunes (vocales, /s/, tono) dan cientos de
ejemplos por persona, y casi todas tienen un valor (mediana: {pct(common)} de las personas). Los sonidos raros (ch, ll/y, rr,
r final) dan pocos, y una persona con muy pocos ejemplos queda sin valor en esa medida: en la medida más rara que se usa
({fn(rare.feature)}), tiene valor el {pct(rare.share_people)} de las personas. Así, en {name(example)} la celda de
{fn(hi_f)} es el promedio de {int(ex[hi_f])} personas y la de {fn(lo_f)}, de {int(ex[lo_f])}. Una medida que falta en más del
35 % de las personas no se usa. Las distancias al centro corrigen por el número de personas detrás de cada valor, así que las
celdas con pocas personas no empujan un lugar hacia la orilla. Celdas grises: ninguna persona de ese lugar tiene valor.</p>"""
    return f"""<p><b>How a cell is made.</b> (1) Audio: about 8 minutes of each person's speech, without music, noise or the
interviewer's voice. (2) Words: speech recognition (Whisper) writes down what they say. (3) Sounds: an aligner (the Montreal
Forced Aligner) finds where each sound starts and ends. (4) Measurements: Praat measures each sound (the pitch, the timing,
the vowels, and whether each /s/, j or r is weakened or replaced). (5) One number per person and measurement: the average
over all their examples of that sound (for example, the share of their /s/ before a consonant that come out as h). (6) The
difference between women and men is removed, and each measurement is put on the same scale (0 = the average person; 1 = one
standard deviation). (7) A cell is the average of the people of that place.</p>
<p><b>Not every cell rests on the same number of people.</b> Common sounds (vowels, /s/, pitch) give hundreds of examples per
person, so nearly everyone has a value (median: {pct(common)} of people). Rare sounds (ch, ll/y, rr, final r) give only a
few, and a person with too few examples has no value for that measurement: for the rarest measurement used
({fn(rare.feature)}), {pct(rare.share_people)} of people have a value. So in {name(example)} the cell for {fn(hi_f)} is the
average of {int(ex[hi_f])} people, and the cell for {fn(lo_f)} of {int(ex[lo_f])}. A measurement missing for more than 35% of
people is not used at all. The distances to the center correct for the number of people behind each value, so thin cells do
not push a place toward the edge. Gray cells: no person of that place has a value.</p>"""


if __name__ == "__main__":
    main()
