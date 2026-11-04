"""Spanish orthography helpers: stress position and which letters stand for which target
segments. MFA aligns phones but does not mark stress, so stress comes from spelling
(Spanish stress is fully predictable from the written form)."""
import re
import unicodedata

STRONG = set("aeoáéó")
ACCENTED = set("áéíóú")
WEAK = set("iuü")
VOWEL_LETTERS = STRONG | WEAK | set("íú")

# Monosyllabic function words are clitics: unstressed in running speech.
CLITICS = set("""a al con de del el en la las le les lo los me mi mis nos os por que se
su sus te tu tus un una unos unas y o e u ni pero si sin tan muy cuando como donde""".split())


def norm_word(w):
    w = unicodedata.normalize("NFC", w.lower())
    return re.sub(r"[^a-záéíóúüñ]", "", w)


def ortho_nuclei(word):
    """Return the list of syllable nuclei (as letter strings) and the stressed index."""
    w0 = w = norm_word(word)
    if not w:
        return [], None
    w = re.sub(r"qu([eiéí])", r"q\1", w)          # silent u in que/qui
    w = re.sub(r"gu([eiéí])", r"g\1", w)          # silent u in gue/gui
    w = re.sub(r"y$", "i", w) if len(w) > 1 and w[-2] in VOWEL_LETTERS else w  # hoy, muy
    runs = re.findall(r"[aeiouáéíóúü]+", w)
    nuclei = []
    for run in runs:
        # split a vowel run at hiatus: two strong vowels, or an accented i/u
        cur = ""
        for ch in run:
            strongish = ch in STRONG or ch in "íú"
            if cur and strongish and any(c in STRONG or c in "íú" for c in cur):
                nuclei.append(cur)
                cur = ch
            else:
                cur += ch
        nuclei.append(cur)
    if not nuclei:
        return [], None
    acc = [i for i, n in enumerate(nuclei) if any(c in ACCENTED for c in n)]
    if acc:
        stress = acc[0]
    elif len(nuclei) == 1:
        stress = 0
    elif w0.endswith("y") and len(w0) > 1 and w0[-2] in VOWEL_LETTERS:
        stress = len(nuclei) - 1          # estoy, Paraguay, Monterrey, virrey: final -y counts as a consonant
    elif w[-1] in "aeiouns":
        stress = len(nuclei) - 2
    else:
        stress = len(nuclei) - 1
    if len(nuclei) == 1 and w0 in CLITICS:
        stress = None
    return nuclei, stress


def theta_positions(word):
    """Indices (in order of fricative letters s/z/c(e,i)) that are <z>/<ce>/<ci>,
    i.e. /θ/ in distinción varieties. Returns list of 'S' or 'Z' per sibilant letter."""
    w = norm_word(word)
    out = []
    for i, ch in enumerate(w):
        if ch == "s":
            out.append("S")
        elif ch == "z":
            out.append("Z")
        elif ch == "c" and i + 1 < len(w) and w[i + 1] in "eiéí":
            out.append("Z")
        elif ch == "x" and 0 < i < len(w) - 1:   # taxi, éxito: /ks/
            out.append("S")
    return out
