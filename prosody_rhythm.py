"""フランス語詩の韻律・リズム・プロゾディー統合分析エンジン（UI非依存）。

理論的基盤:
1. Benoît de Cornulier (1982, 1995, 1999, 2004): Métricométrie・8音節の法則・セジュール診断 (6+6, F6, S6, C6/P6, M6, 4=4=4, 8=4, 4=8)
2. Henri Meschonnic & Gérard Dessons (1982, 1998, 2000): ディスクール・リズム走査 (—, ⏓, ⏑ˈ, (⏑ˈ), ◡)、反アクセント8類型、7種のプロゾディー反響図式、グループ末音動態
3. Albert Di Cristo (1999, 2016): 4階層韻律構成素 (UT < MP < SP < UI)、双極アクセントモデル (AI / AF / AN・アクセント弧)、Eurythmy・%V・ΔC・二重nPVI
4. Jean-Michel Gouvard (1999, 2000), Jean Mazaleyrat (1974/1980), Marc Plénat (1991), Michel Murat (2008): CPMF分布タグ、e caduc 6分類、分音/合音・母音衝突、音韻的脚韻・詩節構造、アンジャンブマン/ルジェ/コントル・ルジェ
"""
import re
import math
import collections
import statistics as st
from phonetics import (
    word_to_ipa,
    extract_phonemes,
    normalize_french_orthography,
    _is_silent_3pl_ent,
    ARCHAIC_IMPERFECT_OIENT,
    AMBIGUOUS_ENT_HOMOGRAPHS,
    H_ASPIRE,
    VOWELS_ALL,
    VOWELS_ORAL,
    VOWELS_NASAL,
    GLIDES,
    CONSONANTS,
    PLOSIVES,
    FRICATIVES,
    LIQUIDS,
)

VOWEL_LETTERS = set("aeiouyàâäéèêëîïôöûùüœæ")
LENGTHENING_CONSONANTS = {"ʁ", "z", "v", "ʒ"}
OBSTRUENTS = {"p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z", "ʃ", "ʒ"}

# Gouvard & Cornulier 閉集合クリティック限定詞・前接代名詞・単音節前置詞リスト（POS誤認に依存しない厳密閉集合）
CLITIC_DETS = {
    "le", "la", "les", "l'", "l", "du", "des", "d'", "d", "au", "aux", "un",
    "mon", "ton", "son", "mes", "tes", "ses", "nos", "vos", "leurs", "leur",
    "ce", "cet", "ces", "quel", "quels", "chaque",
}
DISYLLABIC_CLITIC_DETS = {"une", "cette", "notre", "votre", "quelle", "quelles", "quelque", "quelques"}
PREVERBAL_CLITIC_PRONS = {
    "je", "j'", "j", "tu", "t'", "t", "il", "ils", "on", "ce", "c'", "c",
    "me", "m'", "m", "te", "se", "s'", "s", "le", "la", "les", "en", "y", "ne", "n'", "n",
}
DISYLLABIC_CLITIC_PRONS = {"elle", "elles"}
MONO_PREPOSITIONS = {
    "à", "chez", "dans", "de", "dès", "en", "hors", "par", "pour",
    "près", "sans", "sous", "sur", "vers", "jusqu'", "jusque",
}
WEAK_CONJ_REL = {
    "et", "ou", "ni", "mais", "car", "or", "que", "qu'", "qui", "dont", "où", "si", "comme", "quand",
    "lorsqu'", "puisqu'", "quoiqu'",
}
ELIDED_ZERO_VOWEL_PREFIXES = {"l'", "d'", "j'", "m'", "t'", "s'", "c'", "n'", "qu'"}
CARDINAL_NUMERALS = {
    "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
    "onze", "douze", "treize", "quatorze", "quinze", "seize", "vingt", "trente", "quarante",
    "cinquante", "soixante", "cent", "cents", "mille",
}

# Di Cristo / Garde 後接語（エンクリティック）3分類
ENCLITICS_TONIC_E1 = {"le", "la", "les", "moi", "toi", "lui", "nous", "vous", "leur", "en", "y"}
ENCLITICS_ATONIC_E2 = {"je", "ce", "on"}
ENCLITICS_CONDITIONAL_E3 = {"tu", "il", "ils", "elle", "elles", "t-il", "t-elle", "t-on"}

# Meschonnic 識別的・評価的形容詞（二重記号 ⏓ の候補）
EVALUATIVE_ADJS = {
    "grand", "grande", "grands", "grandes", "petit", "petite", "petits", "petites",
    "beau", "beaux", "bel", "belle", "belles", "bon", "bonne", "bons", "bonnes",
    "vieux", "vieil", "vieille", "vieilles", "jeune", "jeunes", "long", "longue", "longs", "longues",
    "haut", "haute", "hauts", "hautes", "doux", "douce", "douces", "noir", "noire", "noirs", "noires",
    "blanc", "blanche", "blancs", "blanches", "sombre", "sombres", "pâle", "pâles", "pur", "pure",
    "triste", "tristes", "vaste", "vastes", "profond", "profonde", "saint", "sainte", "divin", "divine",
    "seul", "seule", "seuls", "seules", "faux", "fausse", "vrai", "vraie", "cher", "chère",
}

# 古典詩における語内母音後 -e 縮約の許容例外 (Cornulier / Gouvard / Mazaleyrat)
POSTVOCALIC_E_EXEMPT_WORDS = {
    "soient", "aient", "vraiment", "gaiement", "gaîment", "dénouement", "dénoûment", "remerciement", "éternuement",
}

# 標準合音 (synérèse) 固定語彙（古典詩・近代定型詩を通じて原則1音節結合）
SYNERESE_FIXED = {
    "pied", "pieds", "trépied", "assied", "sied", "ciel", "cieux", "miel", "fiel", "fier", "fiers", "fière", "fières",
    "grief", "griefs", "bref", "brefs", "brièvement",
    "vierge", "vierges", "tiers", "pitié", "amitié", "moitié", "bien", "biens", "combien", "rien",
    "mien", "miens", "mienne", "miennes", "tien", "tiens", "tienne", "tiennes", "sien", "siens", "sienne", "siennes",
    "chien", "chiens", "chienne", "chrétien", "chrétiens", "chrétienne", "soutien", "entretien", "maintien",
    "vient", "tient", "souvient", "devient", "revient", "parvient", "appartient", "contient", "retient", "obtient",
    "dieu", "dieux", "adieu", "adieux", "lieu", "lieux", "milieu", "milieux", "pieu", "pieux_nom", "essieu",
    "yeux", "vieux", "mieux", "diable", "diables", "oui", "nuit", "nuits", "minuit", "bruit", "bruits",
    "fruit", "fruits", "pluie", "pluies", "ennui", "ennuis", "appui", "étui", "puits", "buis",
    "lui", "autrui", "celui", "suis", "puis", "puissant", "puissants", "puissante", "puissantes", "puissance",
    "luire", "nuire", "conduire", "instruire", "détruire", "construire", "produire", "réduire", "séduire", "traduire",
    "cuire", "suivre", "fuir", "fuis", "fuit", "fuient", "aujourd'hui",
    "roi", "rois", "foi", "loi", "lois", "soir", "soirs", "gloire", "voir", "voix", "noir", "noire", "noirs", "noires",
    "oiseau", "oiseaux", "voile", "voiles", "étoile", "étoiles", "mémoire", "histoire", "victoire", "espoir",
    "premier", "premiers", "première", "premières", "dernier", "derniers", "dernière", "dernières",
    "sentier", "sentiers", "chevalier", "chevaliers", "laurier", "lauriers", "métier", "métiers",
    "entier", "entiers", "entière", "entières", "altier", "altiers", "acier", "papier", "panier", "foyer",
    "escalier", "escaliers", "pilier", "piliers", "familier", "familiers", "familière", "familières",
    "cavalier", "cavaliers", "singulier", "régulier", "particulier", "guerrier", "guerriers", "janvier",
    "lumière", "lumières", "rivière", "rivières", "paupière", "paupières", "poussière", "pierre", "pierres",
    "lierre", "bière", "manière", "matière", "frontière", "carrière", "barrière", "chaumière", "clairière",
    "crinière", "bannière", "bannières", "cimetière", "cimetières", "lisière",
    "pièce", "pièces", "nièce", "siècle", "siècles", "fièvre", "fièvres", "lièvre", "tiède", "tièdes",
    "siège", "piège", "liège", "cortège", "fiente", "fientes",
    "deuxième", "troisième", "quatrième", "cinquième", "sixième", "septième", "huitième", "neuvième", "dixième", "vingtième", "centième", "millième",
}

# 古典・定型詩における語源的分音 (diérèse) 語彙
DIERESE_LEXICAL = {
    "niais", "niaise", "niaises", "lien", "liens", "ancien", "anciens", "ancienne", "anciennes",
    "aérien", "aériens", "aérienne", "musicien", "musiciens", "comédien", "comédiens", "magicien",
    "gardien", "gardiens", "historien", "indien", "indiens", "bohémien", "égyptien", "italien", "olympien", "quotidien",
    "ruine", "ruines", "ruiner", "bruire", "fluide", "fluides", "suicide", "cruel", "cruelle", "cruels", "cruelles",
    "poète", "poètes", "poème", "poèmes", "poésie", "inquiet", "inquiets", "inquiète", "inquiètes", "inquiétude",
    "piété", "société", "variété", "satiété", "anxiété", "sobriété",
    "hier", "lion", "lions", "pion", "espion", "million", "millions", "milliard",
    "violent", "violents", "violente", "violentes", "violence", "violet", "violets", "violette", "violettes", "violon", "violons",
    "triomphe", "triomphes", "triomphant", "triomphants", "triomphante", "triomphantes", "triompher",
    "diamant", "diamants", "diadème", "dialogue", "miasme", "miasmes", "liane", "lianes", "tiare", "fiacre",
    "science", "sciences", "conscience", "inconscience", "expérience", "audience",
    "orient", "patient", "patients", "patiente", "impatient", "patience", "client", "clients",
    "confiance", "alliance", "méfiance", "défiance", "suppliant", "souriant", "riant", "mendiant", "étudiant",
    "nuage", "nuages", "nuée", "nuées", "huée", "huées", "lueur", "lueurs", "sueur", "sueurs", "tueur",
    "muet", "muets", "muette", "muettes", "ruelle", "ruelles", "duel", "duels", "suave", "suaves", "suavité",
    "annuel", "annuelle", "spirituel", "spirituelle", "rituel", "mutuel", "sensuel", "habituel", "actuel", "éventuel", "perpétuel",
    "sinueux", "sinueuse", "somptueux", "majestueux", "voluptueux", "impétueux", "tumultueux", "tortueux", "vertueux", "monstrueux",
    "intérieur", "intérieure", "extérieur", "extérieure", "supérieur", "supérieure", "inférieur", "antérieur", "ultérieur",
    "essentiel", "essentielle", "artificiel", "superficiel", "substantiel", "providentiel", "officiel", "partiel", "matériel", "immatériel",
    "pieux", "pieuse", "pieuses", "furieux", "furieuse", "curieux", "curieuse", "glorieux", "glorieuse",
    "mystérieux", "mystérieuse", "délicieux", "délicieuse", "odieux", "odieuse", "radieux", "radieuse",
    "précieux", "précieuse", "silencieux", "silencieuse", "soucieux", "envieux", "harmonieux", "mélodieux",
    "victorieux", "laborieux", "impérieux", "sérieux", "religieux", "prodigieux", "gracieux", "spacieux", "audacieux", "pluvieux",
}

# 時代・詩人により分音/合音が揺れる語彙（Cornulier のメトリック圧力による文脈解決対象）
FLUCTUATING_DIERESE_WORDS = {
    "ancien", "anciens", "ancienne", "anciennes",
    "hier", "miasme", "miasmes", "duel", "duels", "fouet", "fouets", "moelle", "alcool", "alcools",
}


class _VerseWord:
    """アポストロフィ前接辞・ハイフン複合語を再統合しPOS誤認を補正した韻律語トークン。"""
    __slots__ = ("text", "pos_", "before_punct", "after_punct")

    def __init__(self, text, pos_, before_punct="", after_punct=""):
        self.text = text
        self.pos_ = pos_
        self.before_punct = before_punct
        self.after_punct = after_punct


def _sanitize_word_pos(word_text, raw_pos, next_word_text=None):
    """spaCy fr_core_news_md の既知の品詞誤認（例: nus->ADP, Fermentent->ADV, Choient->NOUN, Moi->NOUN）を補正する。"""
    w = normalize_french_orthography(word_text.lower())
    w_stem = w.split("'")[-1]

    if w in {"nu", "nus", "nue", "nues"}:
        return "ADJ"
    if w in {"moi", "toi", "soi", "eux"}:
        return "PRON"
    if w in {"ô", "hélas", "ah", "eh", "oh"}:
        return "INTJ"
    if w in {"et", "ou", "ni", "mais", "car", "or"}:
        return "CCONJ"
    if w in {"sans", "sous", "sur", "dans", "vers", "chez", "par", "pour", "dès", "hors", "près", "avec", "jusqu'"}:
        return "ADP"
    if w in CARDINAL_NUMERALS and raw_pos not in ("DET", "NUM"):
        return "NUM"
    if w in {"ayant", "étant"}:
        return "AUX"
    if w_stem in {
        "noir", "noirs", "noire", "noires", "blanc", "blancs", "blanche", "blanches",
        "bleu", "bleus", "bleue", "bleues", "blond", "blonds", "blonde", "blondes",
        "brun", "bruns", "brune", "brunes", "sourd", "sourds", "sourde", "sourdes",
        "épais", "épaisse", "vastes", "vaste", "amères", "amère", "amer", "hideux", "hideuse",
        "niais", "niaise", "tordus", "tordu", "clabaudeurs", "clabaudeur", "triomphants", "triomphant",
        "suprême", "suprêmes", "dévêtu", "dévêtus", "dévêtue", "dévêtues", "vêtu", "vêtus", "vêtue", "vêtues",
        "sépulcral", "sépulcrale", "sépulcraux", "furibond", "furibonde", "furibonds",
        "éployé", "éployée", "éployés", "accablant", "accablante", "vain", "vaine", "vains",
        "basse", "basses",
    } and raw_pos in ("ADP", "NOUN", "ADV", "PROPN", "X"):
        if raw_pos == "NOUN" and next_word_text is None and w_stem in {"noirs", "blancs", "bleus"}:
            return "ADJ"
        return "ADJ"
    if w_stem in {"abolit", "abolis", "aboli", "abolie"} and raw_pos in ("PROPN", "NOUN", "ADP", "X"):
        return "VERB"
    if (
        w_stem.endswith("ment")
        and len(w_stem) >= 7
        and raw_pos in ("PROPN", "ADP", "X")
        and w_stem not in {
            "tourment", "tourments", "clapotement", "clapotements", "élément", "éléments",
            "moment", "moments", "serment", "serments", "jugement", "jugements",
            "mouvement", "mouvements", "vêtement", "vêtements", "bâtiment", "bâtiments",
            "sentiment", "sentiments", "diamant", "diamants", "firmament", "monument", "monuments",
        }
    ):
        return "ADV"
    # 行頭等で ADV / NOUN と誤認された無曖昧な3人称複数動詞 -ent (例: Fermentent, Choient)
    if (
        "-" not in w_stem
        and len(w_stem) >= 5
        and w_stem.endswith("ent")
        and _is_silent_3pl_ent(w_stem, pos="VERB")
        and w_stem not in AMBIGUOUS_ENT_HOMOGRAPHS
        and not w_stem.endswith(("dent", "lent", "cent", "gent", "scient", "tient", "vient"))
        and raw_pos in ("ADV", "NOUN", "PROPN", "ADJ", "X")
    ):
        return "VERB"
    return raw_pos


def _prepare_verse_tokens(doc, fallback_ws=None):
    """spaCy Doc からアポストロフィ短縮・語内ハイフンを結合し、句読点位置と正確なPOSを持つ _VerseWord 列を返す。"""
    if doc is None:
        if not fallback_ws:
            return []
        return [_VerseWord(t.text, _sanitize_word_pos(t.text, t.pos_)) for t in fallback_ws]

    doc_tokens = list(doc)
    raw_items = []
    i = 0
    n_tok = len(doc_tokens)

    while i < n_tok:
        t = doc_tokens[i]
        if t.is_space:
            i += 1
            continue
        # spaCy が語内ハイフン '-' を NOUN 等と誤認するケースを除外し、空白なしの前後語を結合
        if t.is_punct or t.text in {"-", "–", "—"}:
            raw_items.append({"is_punct": True, "text": t.text, "ws_after": bool(t.whitespace_)})
            i += 1
            continue

        # 単語開始: 直後に空白なしで ('-' + 単語) または ('-je', '-le', '-t-il' 等のハイフン開始トークン) が続く場合は連結
        combined_text = t.text
        head_pos = t.pos_
        ws_after = bool(t.whitespace_)
        j = i + 1
        while not ws_after and j < n_tok:
            if (
                j + 1 < n_tok
                and doc_tokens[j].text == "-"
                and not doc_tokens[j].whitespace_
                and not doc_tokens[j + 1].is_space
                and not doc_tokens[j + 1].is_punct
                and doc_tokens[j + 1].text != "-"
            ):
                combined_text += "-" + doc_tokens[j + 1].text
                ws_after = bool(doc_tokens[j + 1].whitespace_)
                j += 2
            elif (
                doc_tokens[j].text.startswith("-")
                and len(doc_tokens[j].text) > 1
                and any(ch.isalpha() for ch in doc_tokens[j].text)
            ):
                combined_text += doc_tokens[j].text
                ws_after = bool(doc_tokens[j].whitespace_)
                j += 1
            else:
                break

        raw_items.append({
            "is_punct": False,
            "text": combined_text,
            "pos": head_pos,
            "ws_after": ws_after,
        })
        i = j

    # 無母音のエリジオン前接辞 (l', d', j', m', t', s', c', n', qu') を後続語へ結合 (例: l’ + oeil -> l’oeil)
    merged_items = []
    k = 0
    while k < len(raw_items):
        item = raw_items[k]
        if item["is_punct"]:
            merged_items.append(item)
            k += 1
            continue
        norm_low = normalize_french_orthography(item["text"].lower())
        if (
            norm_low in ELIDED_ZERO_VOWEL_PREFIXES
            and k + 1 < len(raw_items)
            and not raw_items[k + 1]["is_punct"]
        ):
            nxt_item = raw_items[k + 1]
            merged_items.append({
                "is_punct": False,
                "text": item["text"] + nxt_item["text"],
                "pos": nxt_item["pos"],
                "ws_after": nxt_item["ws_after"],
            })
            k += 2
            continue
        merged_items.append(item)
        k += 1

    # 前後の句読点を各 _VerseWord に付与
    verse_words = []
    pending_before = []
    for item in merged_items:
        if item["is_punct"]:
            if verse_words:
                verse_words[-1].after_punct += item["text"]
            else:
                pending_before.append(item["text"])
        else:
            b_punct = "".join(pending_before)
            pending_before = []
            verse_words.append(_VerseWord(item["text"], item["pos"], before_punct=b_punct, after_punct=""))

    # 文脈つきPOS補正
    for idx, vw in enumerate(verse_words):
        nxt_txt = verse_words[idx + 1].text if idx + 1 < len(verse_words) else None
        vw.pos_ = _sanitize_word_pos(vw.text, vw.pos_, next_word_text=nxt_txt)
        # 行末の "tu"（開き括弧の反送り "(tu" を除く）が形容詞・名詞・副詞・助動詞の直後に置かれる場合は過去分詞 (taire -> tu, VERB)
        if (
            idx == len(verse_words) - 1
            and idx > 0
            and normalize_french_orthography(vw.text.lower()) == "tu"
            and "(" not in (vw.before_punct or "")
            and verse_words[idx - 1].pos_ in ("ADJ", "NOUN", "ADV", "AUX")
        ):
            vw.pos_ = "VERB"

    return verse_words


def _starts_with_vowel_or_h_muet(word_text, word_ipa=""):
    w = normalize_french_orthography(word_text.lower()).strip("'")
    if not w:
        return False
    # l'oeil, d'or 等のようにすでに子音前接辞が付いている語は子音開始！
    if "'" in w:
        pref = w.split("'", 1)[0] + "'"
        if pref in ELIDED_ZERO_VOWEL_PREFIXES or pref in {"jusqu'", "presqu'", "quelqu'", "lorsqu'", "puisqu'", "quoiqu'"}:
            return False
    w_first = w.split("-")[0]
    if w_first in H_ASPIRE or w in H_ASPIRE:
        return False
    if word_ipa:
        phs = extract_phonemes(word_ipa)
        if phs:
            return phs[0] in VOWELS_ALL or phs[0] in GLIDES
    if w_first[0] in VOWEL_LETTERS:
        return True
    if w_first.startswith("h") and len(w_first) > 1 and w_first[1] in VOWEL_LETTERS:
        return True
    return False


def _has_final_post_tonic_e(word_text, pos=None):
    """語末が女性音節（脱落性の e / -es / 動詞3人称複数 -ent / 後接弱代名詞 -je, -ce）で終わるか判定する。"""
    w = normalize_french_orthography(word_text.lower())
    if w.endswith("'"):
        # すでにアポストロフィでエリジオン表記されている語 (jusqu', presqu') は追加の女性音節を持たない
        return False
    if "'" in w:
        w = w.split("'")[-1]
    if "-" in w:
        parts = [p for p in w.split("-") if p]
        if not parts:
            return False
        # 命令形等の強勢後接語 -le は男性強勢
        if len(parts) >= 2 and parts[-1] in ENCLITICS_TONIC_E1:
            return False
        # 倒置弱後接語 -je, -ce (disais-je, est-ce) は女性後強勢音節として振る舞う
        if len(parts) >= 2 and parts[-1] in {"je", "ce"}:
            return True
        w = parts[-1]

    # 単音節機能語は e masculin
    if w in {"le", "ce", "de", "je", "me", "te", "se", "ne", "que"}:
        return False
    if w in POSTVOCALIC_E_EXEMPT_WORDS:
        return False
    if w.endswith("aient") or w in ARCHAIC_IMPERFECT_OIENT:
        return False

    w_clean = re.sub(r"cueill?", "kœj", w)
    w_clean = re.sub(r"gueill?", "ɡœj", w_clean)
    w_clean = re.sub(r"qu", "k", w_clean)
    w_clean = re.sub(r"gu(?=[eéèêiy])", "ɡ", w_clean)
    v_groups = re.findall(r"[aeiouyàâäéèêëîïôöûùüœæ]+", w_clean)
    if len(v_groups) < 2 and not re.search(r"[aeiouyàâäéèêëîïôöûùüœæ]es?$", w_clean) and not _is_silent_3pl_ent(w, pos=pos):
        return False

    if _is_silent_3pl_ent(w, pos=pos):
        return True
    if w_clean.endswith(("es", "e")):
        return True
    return False


def _is_postvocalic_feminine(word_text, pos=None):
    """[満母音 + e/es/ent]（例: vie, épée, joie, bleues, démarrées, choient, voient）を検出する。"""
    w = normalize_french_orthography(word_text.lower()).split("'")[-1]
    if "-" in w:
        parts = [p for p in w.split("-") if p]
        if len(parts) >= 2 and parts[-1] in {"je", "ce", "le"}:
            return False
        w = parts[-1]
    if w in POSTVOCALIC_E_EXEMPT_WORDS or w.endswith("aient") or w in ARCHAIC_IMPERFECT_OIENT:
        return False
    w_clean = re.sub(r"cueill?", "kœj", w)
    w_clean = re.sub(r"gueill?", "ɡœj", w_clean)
    w_clean = re.sub(r"qu", "k", w_clean)
    w_clean = re.sub(r"gu(?=[eéèêiy])", "ɡ", w_clean)
    if _is_silent_3pl_ent(w, pos=pos) and re.search(r"[aeiouyéèêîïôûùœæ]ent$", w_clean):
        return True
    if re.search(r"[aeiouyéèêîïôûùœæ]es?$", w_clean):
        return True
    return False


def detect_synerese_dierese(word_text, pos=None, dierese_mode=False):
    """語内の半母音＋母音連続について、合音(synérèse)/分音(diérèse)の理論診断を返す。"""
    w = normalize_french_orthography(word_text.lower()).split("'")[-1]
    if "-" in w:
        # ハイフン複合語は各構成素を走査
        out = []
        for sub in w.split("-"):
            if sub:
                out.extend(detect_synerese_dierese(sub, pos=pos, dierese_mode=dierese_mode))
        return out

    w_base = re.sub(r"(es|s)$", "", w) if len(w) > 3 and not w.endswith(("ois", "ais", "uis", "eus")) else w
    if _is_silent_3pl_ent(w, pos=pos):
        w_base = w[:-3]
    results = []

    if len(w) < 3:
        return results

    w_scan = re.sub(r"cueill?", "kœj", w)
    w_scan = re.sub(r"gueill?", "ɡœj", w_scan)
    w_scan = re.sub(r"qu", "kk", w_scan)
    w_scan = re.sub(r"gu(?=[eéèêiy])", "ɡɡ", w_scan)

    # 1. 標準合音固定語彙を最優先で保護 (bruit, fruit, pluie, autrui, détruire, ciel, bien, rien, pilier, lumière 等)
    if w in SYNERESE_FIXED or w_base in SYNERESE_FIXED:
        if re.search(r"(i|u|ou|oi)[aeiouéèê]", w_scan):
            results.append({
                "word": word_text,
                "type": "synérèse_lexicale",
                "label": "標準合音 (synérèse)",
                "sequence": w,
                "counted_as_dierese": False,
                "poetic_dierese": False,
                "fluctuating": False,
            })
        return results

    # 2. Muta cum liquida (閉鎖音/摩擦音 + 流音 l/r + i/ou/u + 母音): 義務的分音（ただし ui は bruire/fluide/druide のみ）
    m_mcl = re.search(r"([pbtdkgfvcɡ][lr])(i|ou|u)([aeiouéèêàâôûùœ])", w_scan)
    if m_mcl and not (m_mcl.group(2) == "u" and m_mcl.group(3) == "i" and not re.search(r"(bruir|fluid|druid)", w_scan)):
        results.append({
            "word": word_text,
            "type": "diérèse_muta_cum_liquida",
            "label": "義務的分音 (muta cum liquida)",
            "sequence": m_mcl.group(0),
            "counted_as_dierese": True,
            "poetic_dierese": True,
            "fluctuating": False,
        })
        return results

    # 動詞活用語尾 -ions, -iez (単一子音後) -> 古典的にも合音
    if pos in ("VERB", "AUX") and re.search(r"[^aeiouyrl](ions|iez)$", w_scan):
        results.append({
            "word": word_text,
            "type": "synérèse_verbale",
            "label": "動詞語尾合音 (-ions/-iez)",
            "sequence": w[-4:],
            "counted_as_dierese": False,
            "poetic_dierese": False,
            "fluctuating": False,
        })
        return results

    # 3. 語源的・古典詩法上の分音語彙 (niais, million, passion, furieux, triomphants, échouages, nuage, ancien 等)
    is_etym_dierese = False
    reason = ""
    if w in DIERESE_LEXICAL or w_base in DIERESE_LEXICAL:
        is_etym_dierese = True
        reason = "語源的分音語彙"
    elif re.search(r"([tsxdlmnrvz]ion|llion)s?$", w_scan) and pos != "VERB":
        is_etym_dierese = True
        reason = "名詞接尾辞 (-ion)"
    elif re.search(r"ieu(x|se|ses)$", w_scan) and w not in {"vieux", "mieux", "cieux", "dieux", "lieux", "milieux", "yeux"}:
        is_etym_dierese = True
        reason = "形容詞接尾辞 (-ieux/-ieuse)"
    elif re.search(r"[^kg](uel|uelle|uels|uelles|uer|ueur|ueurs|ueux|ueuse|ueuses|ouer|oué|ouée|oués|ouées|ouage|ouages|ouir|ouï|ouïe)$", w_scan):
        is_etym_dierese = True
        reason = "u/ou + 母音連続"
    elif re.search(r"(ruin|bruir|fluid|suicid|uité|uition)", w_scan):
        is_etym_dierese = True
        reason = "語源的 u+i 分音"
    elif pos == "VERB" and re.search(r"ier$", w_scan):
        is_etym_dierese = True
        reason = "動詞 -ier"
    elif re.search(r"(ian|iant|iante|iants|iantes|iance|iances|ience|iences|ient|ients)$", w_scan) and w not in SYNERESE_FIXED and w_base not in SYNERESE_FIXED:
        is_etym_dierese = True
        reason = "語源的 i+鼻母音分音"
    elif re.search(r"(ia|io|iet|iète|ieur|ieure|ieurs|ieures|iel|ielle|iels|ielles|ien|iens|ienne|iennes)$", w_scan) and len(w_base) >= 5 and w not in SYNERESE_FIXED and w_base not in SYNERESE_FIXED:
        is_etym_dierese = True
        reason = "多音節語源的 i + 母音連続"

    if is_etym_dierese:
        is_fluct = (w in FLUCTUATING_DIERESE_WORDS) or (w_base in FLUCTUATING_DIERESE_WORDS)
        results.append({
            "word": word_text,
            "type": "diérèse_étymologique",
            "label": f"古典的分音 ({reason})",
            "sequence": w,
            "counted_as_dierese": bool(dierese_mode),
            "poetic_dierese": True,
            "fluctuating": is_fluct,
        })
    return results


def syllabify_phonemes(phonemes):
    """IPA音素列をアンシェヌマン規則に従い音節 (onset, nucleus, coda) のリストへ分割する。
    各母音音素を音節核とし、母音間子音群は最大オンセット原理（単一子音および閉鎖音/摩擦音＋流音/半母音）で後続音節頭へ配分する。
    """
    nuclei_indices = [idx for idx, p in enumerate(phonemes) if p in VOWELS_ALL]
    if not nuclei_indices:
        return []

    syllables = []
    n_vow = len(nuclei_indices)

    for s_idx, v_pos in enumerate(nuclei_indices):
        nucleus = phonemes[v_pos]
        if s_idx == 0:
            onset = list(phonemes[:v_pos])
        else:
            prev_v = nuclei_indices[s_idx - 1]
            cluster = list(phonemes[prev_v + 1:v_pos])
            onset = _split_intervocalic_cluster(cluster)[1]

        if s_idx == n_vow - 1:
            coda = list(phonemes[v_pos + 1:])
        else:
            next_v = nuclei_indices[s_idx + 1]
            cluster = list(phonemes[v_pos + 1:next_v])
            coda = _split_intervocalic_cluster(cluster)[0]

        is_open = len(coda) == 0
        has_lengthening = (
            (len(coda) == 1 and coda[0] in LENGTHENING_CONSONANTS)
            or (len(coda) == 2 and coda[0] == "v" and coda[1] == "ʁ")
        )
        syllables.append({
            "onset": onset,
            "nucleus": nucleus,
            "coda": coda,
            "is_open": is_open,
            "is_schwa": nucleus == "ə",
            "has_lengthening_coda": has_lengthening,
            "ipa": "".join(onset) + nucleus + "".join(coda),
        })
    return syllables


def _split_intervocalic_cluster(cluster):
    """母音間子音連続を (先行音節coda, 後続音節onset) に分割する。"""
    k = len(cluster)
    if k == 0:
        return [], []
    if k == 1:
        return [], cluster
    # 末尾が半母音の場合: 直前が閉鎖音/摩擦音(+流音)ならまとめて後続onsetへ
    if cluster[-1] in GLIDES:
        if k == 2 and (cluster[0] in OBSTRUENTS or cluster[0] in LIQUIDS or cluster[0] in {"m", "n"}):
            return [], cluster
        if k >= 3 and cluster[-3] in OBSTRUENTS and cluster[-2] in LIQUIDS:
            return cluster[:-3], cluster[-3:]
        return cluster[:-1], cluster[-1:]
    # 閉鎖音/摩擦音 + 流音 (muta cum liquida) -> 両方とも後続onset（開音節化）
    if cluster[-2] in OBSTRUENTS and cluster[-1] in LIQUIDS and cluster[-2] not in {"s", "z", "ʃ", "ʒ"}:
        return cluster[:-2], cluster[-2:]
    return cluster[:-1], cluster[-1:]


def build_line_syllable_stream(
    ws,
    doc,
    line_phonetics,
    dierese=False,
    is_free_verse_hint=False,
    word_dierese_overrides=None,
    ancient_postvocalic_e=False,
):
    """1詩行のトークン列から、詩法上の音節列・e caduc状態・CPMFタグ・母音衝突・分音情報を精密構築する。"""
    v_words = _prepare_verse_tokens(doc, fallback_ws=ws)
    if not v_words:
        return {
            "verse_words": [],
            "syllables": [],
            "counted_syllables": [],
            "anatonic_count": 0,
            "total_pronounced_count": 0,
            "has_extrametrical_final_e": False,
            "e_caduc_events": [],
            "hiatus_events": [],
            "synerese_dierese_events": [],
            "phonological_syllables": [],
        }

    word_dierese_overrides = word_dierese_overrides or {}
    syll_stream = []
    e_caduc_events = []
    hiatus_events = []
    syn_die_events = []
    liaison_map = {}
    for l_item in line_phonetics.get("liaisons", []):
        liaison_map[(l_item["word1"], l_item["word2"])] = l_item["consonant"]
        liaison_map[(l_item["word1"].split("'")[-1], l_item["word2"])] = l_item["consonant"]

    n_words = len(v_words)
    full_phonemes_for_syllabification = []

    for w_i, tok in enumerate(v_words):
        w_text = tok.text
        w_lower = normalize_french_orthography(w_text.lower())
        pos = tok.pos_
        nxt_tok = v_words[w_i + 1] if w_i + 1 < n_words else None
        nxt_ipa = word_to_ipa(nxt_tok.text.lower(), pos=nxt_tok.pos_)[0] if nxt_tok else ""
        nxt_starts_v = _starts_with_vowel_or_h_muet(nxt_tok.text, nxt_ipa) if nxt_tok else False

        # 分音・合音診断（文脈解決オーバーライドがあれば優先適用）
        eff_dierese = word_dierese_overrides.get(w_i, dierese)
        sd_list = detect_synerese_dierese(w_text, pos=pos, dierese_mode=eff_dierese)
        syn_die_events.extend(sd_list)

        # 単語の基本IPA取得
        base_ipa, _ = word_to_ipa(w_lower, pos=pos)
        force_dierese = any(ev["counted_as_dierese"] for ev in sd_list)
        force_synerese = (
            any(not ev["counted_as_dierese"] for ev in sd_list)
            or (w_i in word_dierese_overrides and word_dierese_overrides[w_i] is False)
        )
        if force_dierese:
            base_ipa = _apply_dierese_to_ipa(base_ipa, w_lower)
        elif force_synerese:
            base_ipa = _apply_synerese_to_ipa(base_ipa, w_lower)

        phs = extract_phonemes(base_ipa)
        has_fem_e = _has_final_post_tonic_e(w_text, pos=pos)
        is_postvoc_e = _is_postvocalic_feminine(w_text, pos=pos)

        # LEXICON 登録語 (quelque, lorsque, jusque 等) や後接語 (-je, -ce) が末尾 ə を既に持つ場合、女性語尾管理に統一するため一旦除去
        if has_fem_e and phs and phs[-1] == "ə":
            phs = phs[:-1]

        # 語内シュワー検出 (語末以外で ə を含む音節)
        internal_schwa_count = sum(1 for p in phs if p == "ə")
        w_stem_only = w_lower.split("'")[-1]
        if w_stem_only in {"le", "ce", "de", "je", "me", "te", "se", "ne", "que"}:
            pass
        elif internal_schwa_count > 0:
            e_caduc_events.append({
                "word": w_text,
                "word_idx": w_i,
                "status": "interne_maintenu",
                "label": "語内保持 e (interne maintenu)",
                "counted": True,
            })

        # 語末の脱落性 e (e féminin) の6状態判定
        realize_final_schwa = False
        extrametrical_final_schwa = False
        has_plural_guard = bool(
            w_stem_only.endswith("es") or _is_silent_3pl_ent(w_stem_only, pos=pos)
        )

        if has_fem_e:
            if w_i == n_words - 1:
                extrametrical_final_schwa = True
                e_caduc_events.append({
                    "word": w_text,
                    "word_idx": w_i,
                    "status": "surnumeraire_fin_de_vers",
                    "label": "行末余剰女性音節 (surnuméraire / (⏑ˈ))",
                    "counted": False,
                })
            else:
                if is_postvoc_e and (not nxt_starts_v or has_plural_guard):
                    if ancient_postvocalic_e:
                        realize_final_schwa = True
                        e_caduc_events.append({
                            "word": w_text,
                            "word_idx": w_i,
                            "status": "postvocalique_maintenu_ancien",
                            "label": "母音後-e古代保持・音節算入 (XVIe s. / poésie ancienne : vi-e, joi-e)",
                            "counted": True,
                        })
                    else:
                        # 母音後 -e/-es/-ent の非エリジオン: 音声上アポコープ（0音節）として処理しつつ古典詩法違反を記録 (例: Choient, des arbres... / La joie venait...)
                        realize_final_schwa = False
                        e_caduc_events.append({
                            "word": w_text,
                            "word_idx": w_i,
                            "status": "apocope_postvocalique",
                            "label": "母音後-e非エリジオン（古典詩法禁止・近代アポコープ）",
                            "counted": False,
                        })
                elif nxt_starts_v and not has_plural_guard:
                    e_caduc_events.append({
                        "word": w_text,
                        "word_idx": w_i,
                        "status": "elide_devant_V",
                        "label": "母音前エリジオン (élision métrique)",
                        "counted": False,
                    })
                else:
                    realize_final_schwa = True
                    e_caduc_events.append({
                        "word": w_text,
                        "word_idx": w_i,
                        "status": "final_maintenu_devant_C",
                        "label": "子音前保持・算入 (-e numéraire)",
                        "counted": True,
                    })

        # 母音衝突 (Hiatus métrique) の検出
        if nxt_tok and nxt_starts_v and not has_fem_e:
            has_liaison_here = (w_text, nxt_tok.text) in liaison_map or (w_text.split("'")[-1], nxt_tok.text) in liaison_map
            ends_with_phon_vowel = bool(phs and phs[-1] in VOWELS_ALL)
            has_silent_cons_guard = bool(w_lower[-1] not in VOWEL_LETTERS and w_lower != "et")
            if w_lower == "et" or (ends_with_phon_vowel and not has_liaison_here and not has_silent_cons_guard):
                nxt_phs = extract_phonemes(nxt_ipa)
                same_timbre = bool(phs and nxt_phs and phs[-1] == nxt_phs[0])
                hiatus_events.append({
                    "word1": w_text,
                    "word2": nxt_tok.text,
                    "context": f"{w_text} / {nxt_tok.text}",
                    "type": "hiatus_et" if w_lower == "et" else ("hiatus_meme_timbre" if same_timbre else "hiatus_metrique"),
                    "label": "「et + 母音」古典禁止衝突" if w_lower == "et" else (
                        "同一母音反復衝突 (Grammont強衝突)" if same_timbre else "語間母音衝突 (hiatus métrique)"
                    ),
                })

        # 単語内音節の構築
        word_phs = list(phs)
        if realize_final_schwa and not (word_phs and word_phs[-1] == "ə"):
            word_phs.append("ə")

        word_sylls = syllabify_phonemes(word_phs)
        if not word_sylls:
            continue

        # ハイフン複合語の境界インデックス特定 (例: tohu-bohus -> 第2音節末が L)
        hyphen_cut_idx = _find_hyphen_boundary_syl_idx(w_lower, len(word_sylls), realize_final_schwa)

        # Gouvard & Cornulier 厳密閉集合 CPMF タグ判定
        cpmf_tags = _compute_word_cpmf_tags(tok, nxt_tok, len(word_sylls), realize_final_schwa, hyphen_cut_idx=hyphen_cut_idx)

        masc_indices = [idx for idx in range(len(word_sylls)) if not (realize_final_schwa and idx == len(word_sylls) - 1)]
        last_masc_in_word = masc_indices[-1] if masc_indices else (len(word_sylls) - 1)
        # 余剰女性音節がある場合は語末 release を含めた音節数で正書法を近似分割する
        n_orth_syls = (len(word_sylls) + 1) if extrametrical_final_schwa else len(word_sylls)
        orth_chunks = _approximate_orth_syllables(w_text, n_orth_syls)

        prev_tok = v_words[w_i - 1] if w_i > 0 else None
        liaison_in_c = None
        if prev_tok is not None:
            liaison_in_c = (
                liaison_map.get((prev_tok.text, w_text))
                or liaison_map.get((prev_tok.text.split("'")[-1], w_text))
            )

        for s_i, s_obj in enumerate(word_sylls):
            is_word_final_fem = realize_final_schwa and (s_i == len(word_sylls) - 1)
            is_word_dvm = (s_i == last_masc_in_word) and not is_word_final_fem
            # 余剰女性音節 (extrametrical) が後続する場合、算入末尾音節は語末・行末句読点を持たない
            is_word_fin = (s_i == len(word_sylls) - 1) and not extrametrical_final_schwa
            punct_aft = tok.after_punct if is_word_fin else ""
            syll_stream.append({
                "syl_idx": len(syll_stream) + 1,
                "word_idx": w_i,
                "word_text": w_text,
                "word_lower": w_lower,
                "pos": pos,
                "orth": orth_chunks[s_i] if s_i < len(orth_chunks) else w_text,
                "onset": s_obj["onset"],
                "liaison_c": liaison_in_c if s_i == 0 else None,
                "nucleus": s_obj["nucleus"],
                "coda": s_obj["coda"],
                "ipa": s_obj["ipa"],
                "is_open": s_obj["is_open"],
                "is_schwa": s_obj["is_schwa"] or is_word_final_fem,
                "is_post_tonic_fem": is_word_final_fem,
                "is_word_initial": s_i == 0,
                "is_word_final_masc": is_word_dvm,
                "is_word_end": is_word_fin,
                "word_syl_len": len(word_sylls),
                "cpmf": cpmf_tags[s_i] if s_i < len(cpmf_tags) else "0",
                "punct_after": punct_aft,
                "punct_before": tok.before_punct if s_i == 0 else "",
                "elided_fem_after": bool(has_fem_e and not realize_final_schwa and w_i < n_words - 1 and nxt_starts_v),
            })

        full_phonemes_for_syllabification.extend(word_phs)
        if nxt_tok and (w_text, nxt_tok.text) in liaison_map:
            full_phonemes_for_syllabification.append(liaison_map[(w_text, nxt_tok.text)])

        # 行末余剰女性音節 (surnuméraire) の記録
        if extrametrical_final_schwa:
            ext_onset = _split_intervocalic_cluster(word_sylls[-1]["coda"])[1] if word_sylls and word_sylls[-1]["coda"] else []
            ext_orth = orth_chunks[-1] if len(orth_chunks) > len(word_sylls) else "(e)"
            syll_stream.append({
                "syl_idx": None,
                "word_idx": w_i,
                "word_text": w_text,
                "word_lower": w_lower,
                "pos": pos,
                "orth": ext_orth,
                "onset": ext_onset,
                "liaison_c": None,
                "nucleus": "ə",
                "coda": [],
                "ipa": ("".join(ext_onset) + "ə") if ext_onset else "ə",
                "is_open": True,
                "is_schwa": True,
                "is_post_tonic_fem": True,
                "is_extrametrical": True,
                "is_word_initial": False,
                "is_word_final_masc": False,
                "is_word_end": True,
                "word_syl_len": len(word_sylls),
                "cpmf": "F_ext",
                "punct_after": tok.after_punct,
                "punct_before": "",
                "elided_fem_after": False,
            })

    counted_sylls = [s for s in syll_stream if not s.get("is_extrametrical")]
    phon_sylls = syllabify_phonemes(full_phonemes_for_syllabification)

    return {
        "verse_words": v_words,
        "syllables": syll_stream,
        "counted_syllables": counted_sylls,
        "anatonic_count": len(counted_sylls),
        "total_pronounced_count": len(syll_stream),
        "has_extrametrical_final_e": any(s.get("is_extrametrical") for s in syll_stream),
        "e_caduc_events": e_caduc_events,
        "hiatus_events": hiatus_events,
        "synerese_dierese_events": syn_die_events,
        "phonological_syllables": phon_sylls,
    }


def _apply_dierese_to_ipa(ipa_str, w_lower):
    """分音対象語の半母音 (j, ɥ, w) を対応する狭母音 (i, y, u) に変換する。
    ただし ay/oy/uy/ill 由来のわたり音 [j] (ayant, joyeux, brillant 等) は変換しない。
    """
    w_stem = w_lower.split("'")[-1]
    res = ipa_str
    if not re.search(r"(ay|oy|uy|aill|eill|ouill|euill|œill)", w_stem):
        res = re.sub(r"j(?=[ɑ̃ɛ̃ɔ̃œ̃aeɛoɔøœu])", "i", res)
    res = re.sub(r"ɥ(?=[iɛeɑ̃aœ])", "y", res)
    if "oi" not in w_stem and "oî" not in w_stem and "oy" not in w_stem:
        res = re.sub(r"w(?=[eɛaɑ̃i])", "u", res)
    return res


def _apply_synerese_to_ipa(ipa_str, w_lower):
    """合音対象語（または文脈的合音解決語: anciens=2, alcool=2, carrière=2 等）の母音連続を1音節へ縮約する。"""
    w_stem = w_lower.split("'")[-1]
    if "alcool" in w_stem:
        return ipa_str.replace("oo", "o")
    res = re.sub(r"(?<![pbtdkɡfv][lʁ])i(?=[ɑ̃ɛ̃ɔ̃œ̃aeɛoɔøœu])", "j", ipa_str)
    if not re.search(r"[pbtdkɡfv][lʁ]y", res):
        res = re.sub(r"y(?=[iɛeɑ̃a])", "ɥ", res)
    return res


def _find_hyphen_boundary_syl_idx(w_lower, total_syls, has_realized_fem):
    """ハイフン複合語 (例: tohu-bohus) において第1構成素の最終音節インデックス (0-based) を返す。"""
    if "-" not in w_lower or total_syls < 2:
        return None
    parts = [p for p in w_lower.split("-") if p]
    if len(parts) < 2 or parts[-1] in (ENCLITICS_TONIC_E1 | ENCLITICS_ATONIC_E2 | ENCLITICS_CONDITIONAL_E3):
        return None
    p0_ipa, _ = word_to_ipa(parts[0])
    p0_vows = sum(1 for ph in extract_phonemes(p0_ipa) if ph in VOWELS_ALL)
    masc_end = total_syls - 2 if has_realized_fem else total_syls - 1
    if 1 <= p0_vows <= masc_end:
        return p0_vows - 1
    return None


def _approximate_orth_syllables(word_text, n_syl):
    """表示用に正書法文字列を音節数 n_syl 個の断片へ精密分割する。
    分音・母音後e・子音群（muta cum liquida、二重子音）の音節境界規則に完全準拠。
    """
    if n_syl <= 1 or not word_text:
        return [word_text]
    w = word_text

    # 末尾の句読点等の退避
    m_p = re.search(r"([.,;:!?—–\s]+)$", w)
    trailing_punct = m_p.group(1) if m_p else ""
    w_core = w[:-len(trailing_punct)] if trailing_punct else w

    VOWEL_LETTERS_STR = "aeiouyàâäéèêëîïôöûùüœæ"
    v_pattern = f"[{VOWEL_LETTERS_STR}]+"
    v_matches = list(re.finditer(v_pattern, w_core, flags=re.IGNORECASE))

    if len(v_matches) < n_syl:
        # 分音や母音後e等で母音塊数より音節数が多い場合、母音連続を音節核へ分離
        split_candidates = [
            # 1. 語末母音後 e/es (vie, joie, épée, bleues, vraie)
            rf"(?<=[{VOWEL_LETTERS_STR}])(?=es?$)",
            # 2. i + 語末動詞・名詞語尾 (oublier, étions, lion, violon)
            r"(?<=[iï])(?=(?:er|ez|ons|ent)$)",
            # 3. i + 母音 (diamant, passion, niais, suicide, violent)
            r"(?<=[iï])(?=[aouàâäéèêôöûùü])",
            # 4. u + 母音 (ruine, nuage, sueur, duel, cruel)
            r"(?<=[uü])(?=[aouàâäéèêëiîï])",
            # 5. é/è/ê + 母音 (création, poète)
            r"(?<=[éèêë])(?=[aiou])",
            r"(?<=[ao])(?=[éèêë])",
            # 6. その他の母音連続
            rf"(?<=[{VOWEL_LETTERS_STR}])(?=[{VOWEL_LETTERS_STR}])",
        ]
        w_expanded = w_core
        needed = n_syl - len(v_matches)
        for cand in split_candidates:
            if needed <= 0:
                break
            new_w, count = re.subn(cand, "§", w_expanded, count=needed, flags=re.IGNORECASE)
            if count > 0:
                needed -= count
                w_expanded = new_w

        parts = w_expanded.split("§")
        curr_pos = 0
        v_spans = []
        for p in parts:
            for vm in re.finditer(v_pattern, p, flags=re.IGNORECASE):
                v_spans.append((curr_pos + vm.start(), curr_pos + vm.end()))
            curr_pos += len(p)
    else:
        v_spans = [(m.start(), m.end()) for m in v_matches]
        # 母音塊が音節数より多い場合（語末無音eや合音）
        while len(v_spans) > n_syl:
            if v_spans[-1][1] == len(w_core) and w_core[v_spans[-1][0]:v_spans[-1][1]].lower() in ("e", "es", "ent"):
                v_spans.pop()
            else:
                min_dist = 999
                merge_idx = 0
                for i in range(len(v_spans) - 1):
                    d = v_spans[i + 1][0] - v_spans[i][1]
                    if d < min_dist:
                        min_dist = d
                        merge_idx = i
                v_spans[merge_idx] = (v_spans[merge_idx][0], v_spans[merge_idx + 1][1])
                v_spans.pop(merge_idx + 1)

    if len(v_spans) != n_syl:
        step = max(1, len(w_core) // n_syl)
        chunks = [w_core[i * step:(i + 1) * step] for i in range(n_syl - 1)]
        chunks.append(w_core[(n_syl - 1) * step:] + trailing_punct)
        return chunks

    cuts = [0]
    for idx in range(n_syl - 1):
        v1_end = v_spans[idx][1]
        v2_start = v_spans[idx + 1][0]
        c_len = v2_start - v1_end
        if c_len <= 0:
            cuts.append(v1_end)
        elif c_len == 1:
            cuts.append(v1_end)
        else:
            c_str = w_core[v1_end:v2_start].lower()
            if c_str[-2:] in ("pr", "br", "tr", "dr", "cr", "gr", "fr", "vr", "pl", "bl", "cl", "gl", "fl", "ch", "ph", "th", "gn", "qu"):
                cuts.append(v2_start - 2)
            else:
                cuts.append(v1_end + 1)
    cuts.append(len(w_core))
    res = [w_core[cuts[i]:cuts[i + 1]] for i in range(n_syl)]
    if trailing_punct:
        res[-1] += trailing_punct
    return res


def _compute_word_cpmf_tags(tok, nxt_tok, n_syl, has_realized_fem, hyphen_cut_idx=None):
    """Cornulier & Gouvard『Critique du vers』の分布タグ (C, CF, P, M, L, Morph, F, 0) を厳密閉集合により付与する。
    ※ spaCy の品詞誤認 (例: nus -> ADP) に引きずられないよう、pos == 'ADP' / 'DET' の単独条件は一切使わない。
    """
    if n_syl <= 0:
        return []
    w = normalize_french_orthography(tok.text.lower())
    pos = tok.pos_
    has_punct_after = bool(getattr(tok, "after_punct", ""))

    # エリジオン結合語 (l'oeil, d'or vs d'un, qu'il, l'on) の本体語を取り出す
    w_host = w.split("'")[-1] if ("'" in w and not w.endswith("'")) else w

    # 1. 二音節クリティック限定詞・代名詞 (une, cette, notre, votre, elle, quelque)
    if (
        nxt_tok is not None
        and not has_punct_after
        and (w_host in DISYLLABIC_CLITIC_DETS or w_host in DISYLLABIC_CLITIC_PRONS)
    ):
        if n_syl == 1:
            return ["C"]
        return ["C"] + ["CF"] * (n_syl - 1)

    # 2. 単音節クリティック限定詞・前接代名詞・単音節前置詞（厳密閉集合のみ！）
    if n_syl == 1:
        if nxt_tok is not None and not has_punct_after:
            if w in CLITIC_DETS or w_host in CLITIC_DETS:
                return ["C"]
            if (w in PREVERBAL_CLITIC_PRONS or w_host in PREVERBAL_CLITIC_PRONS) and "-" not in tok.text:
                return ["C"]
            if w_host in {"lui", "leur"} and pos == "PRON" and nxt_tok.pos_ in ("VERB", "AUX"):
                return ["C"]
            if w in MONO_PREPOSITIONS:
                return ["P"]
        return ["0"]

    # 3. 多音節語: 最終男性音節より前は M (複合語境界は L、接頭辞境界は Morph)、最終男性音節は 0 (二音節前置詞/接続詞は P)、後続女性音節は F
    tags = []
    masc_end = n_syl - 2 if has_realized_fem else n_syl - 1
    prefix_1syl = bool(re.match(r"^(in|im|ir|il|dé|dés|re|ré|pré|pro|sub|sur|trans|mé|mal)[a-zà-ÿ]", w_host))
    prefix_2syl = bool(re.match(r"^(anti|contre|entre|archi|ultra|demi|semi)[a-zà-ÿ]", w_host))

    for i in range(n_syl):
        if i < masc_end:
            if hyphen_cut_idx is not None and i == hyphen_cut_idx:
                tags.append("L")
            elif prefix_2syl and i == 1 and masc_end >= 2:
                tags.append("Morph")
            elif prefix_1syl and i == 0:
                tags.append("Morph")
            else:
                tags.append("M")
        elif i == masc_end:
            if (
                nxt_tok is not None
                and not has_punct_after
                and w_host in {"entre", "contre", "avant", "après", "depuis", "pendant", "selon", "lorsque", "puisque", "quoique", "jusque"}
            ):
                tags.append("P")
            else:
                tags.append("0")
        else:
            tags.append("F")
    return tags


# =============================================================================
# Pillar 1: Benoît de Cornulier (Métricométrie・8音節の法則・セジュール・v8診断)
# =============================================================================

def analyze_cornulier_line(line_stream, line_idx=1):
    """Cornulier (1982, 1995, 1999) の韻律測定法(Métricométrie)に基づき、
    主強勢領域(Domaine anatonique)・8音節の法則・セジュールコード(6+6, S6/F7回収, F6/FC6, C6/P6, L6/Morph6/M6)・v8三拍子構造を精密診断する。
    """
    counted = line_stream["counted_syllables"]
    v_words = line_stream.get("verse_words", [])
    n = len(counted)
    is_fem = line_stream["has_extrametrical_final_e"]
    genre = "F" if is_fem else "M"

    if n == 0:
        return {
            "line": line_idx,
            "anatonic_length": 0,
            "genre": genre,
            "meter_class": "空行",
            "loi_des_8": "対象外",
            "meter_code": "0",
            "caesura_code": "なし",
            "caesura_label": "対象外",
            "caesura_pos": None,
            "ternary_profile": "なし",
            "cpmf_string": "",
            "measures": [],
            "measures_str": "",
            "measure_type": "対象外",
        }

    cpmf_seq = [s["cpmf"] for s in counted]
    cpmf_str = "-".join(cpmf_seq)

    # 8音節の法則 (Loi des 8 syllabes)
    if n <= 8:
        loi_8 = f"単一詩句 (vers simple: {n}v ≤ 8)"
        meter_class = "vers_simple"
    elif n <= 12:
        loi_8 = f"複合詩句 (vers composé: {n}v > 8)"
        meter_class = "vers_compose"
    else:
        loi_8 = f"超定型・ヴェルセ/自由詩行 ({n}v > 12)"
        meter_class = "hors_metre"

    caesura_code = "なし"
    caesura_label = "単一詩句（固定セジュール不要）" if n <= 8 else "非定型（固定セジュールなし）"
    caesura_pos = None
    ternary_profile = "なし"
    meter_code = f"{n}v"

    def is_cuttable(pos_1based):
        if pos_1based < 1 or pos_1based > n:
            return False
        s_cand = counted[pos_1based - 1]
        if s_cand["cpmf"] != "0":
            return False
        if s_cand["word_lower"] in WEAK_CONJ_REL and not s_cand["punct_after"]:
            return False
        return True

    if n == 12:
        s6 = counted[5]
        s7 = counted[6]
        tag6 = s6["cpmf"]
        cut3 = is_cuttable(3)
        cut4 = is_cuttable(4)
        cut8 = is_cuttable(8)
        s8_recup = cut8 and counted[8]["is_post_tonic_fem"] and counted[8]["word_idx"] == counted[7]["word_idx"]

        # Cornulier v8 / 三拍子・準三拍子 (...4v}) プロファイル診断
        if cut4 and cut8:
            if s8_recup:
                ternary_profile = "4=4'=4 (三拍子・第9音節S8リズム回収)"
            else:
                ternary_profile = "4=4=4 (三拍子・トリメートル可能)"
        elif cut8 and not cut4:
            if cut3:
                ternary_profile = "3-5-4 / 8=4 (Cornulier ...4v} 準三拍子)"
            else:
                ternary_profile = "8=4 (後半4音節 ...4v} 構造)"
        elif cut4 and not cut8:
            ternary_profile = "4=8 (前半4音節準三拍子)"

        # 第6音節の厳密診断 (Cornulier & Gouvard コード)
        if tag6 == "0":
            if s7["is_post_tonic_fem"] and s7["word_idx"] == s6["word_idx"]:
                caesura_code = "S6 (F7: récupération rythmique)"
                caesura_label = "跨りセジュール S6/F7（第7音節に女性語尾 -e 侵入・リズム回収）"
                caesura_pos = 6
                meter_code = "6+6 (S6)"
            elif s6.get("elided_fem_after"):
                caesura_code = "6+6 (élision)"
                caesura_label = "古典的セジュール（6+6・母音前エリジオン伴う一致）"
                caesura_pos = 6
                meter_code = "6+6"
            elif s6["word_lower"] in WEAK_CONJ_REL:
                caesura_code = "6+6 (conjonction faible)"
                caesura_label = "弱接続詞・関係詞第6音節（ロマン派以降の弛緩6+6）"
                caesura_pos = 6
                meter_code = "6+6"
            else:
                caesura_code = "6+6 (concordant)"
                caesura_label = "古典的一致セジュール（6+6・男性強勢境界）"
                caesura_pos = 6
                meter_code = "6+6"
        elif tag6 in ("F", "CF"):
            code_lbl = "FC6" if tag6 == "CF" else "F6"
            caesura_code = f"{code_lbl} (césure lyrique / {tag6}6)"
            caesura_label = f"第6音節女性語尾違反 ({code_lbl}: 抒情詩的セジュールまたは三拍子化)"
            caesura_pos = 6 if not (cut4 and cut8) else None
            meter_code = "4=4=4" if (cut4 and cut8) else ("8=4" if cut8 else f"12v({code_lbl})")
        elif tag6 in ("C", "P"):
            caesura_code = f"{tag6}6 (proclitique 6e)"
            caesura_label = f"第6音節前接語違反 ({tag6}6: 冠詞・前置詞等が半句境界を跨ぐ)"
            caesura_pos = None if (cut4 and cut8) else 6
            meter_code = "4=4=4" if (cut4 and cut8) else ("8=4" if cut8 else ("4=8" if cut4 else f"6+6({tag6}6)"))
        elif tag6 in ("L", "Morph", "M"):
            caesura_code = f"{tag6}6 (mot interne 6e)"
            sub_desc = {
                "L": "ハイフン複合語境界 L6",
                "Morph": "接頭辞・形態素境界 Morph6",
                "M": "純粋語内分断 M6",
            }[tag6]
            caesura_label = f"第6音節語内前強勢違反 ({sub_desc})"
            caesura_pos = 6 if tag6 == "L" and not (cut4 and cut8) else None
            meter_code = "4=4=4" if (cut4 and cut8) else ("8=4" if cut8 else ("4=8" if cut4 else f"12v({tag6}6)"))

    elif n == 10:
        if is_cuttable(4):
            caesura_code = "4+6 (décasyllabe classique)"
            caesura_label = "古典デカシラブ（4+6 句切れ）"
            caesura_pos = 4
            meter_code = "4+6"
        elif is_cuttable(5):
            caesura_code = "5+5 (décasyllabe symétrique)"
            caesura_label = "対称デカシラブ（5+5 句切れ・ヴェルレーヌ型）"
            caesura_pos = 5
            meter_code = "5+5"
        elif is_cuttable(6):
            caesura_code = "6+4 (décasyllabe a maiori)"
            caesura_label = "逆転デカシラブ（6+4 句切れ）"
            caesura_pos = 6
            meter_code = "6+4"
        else:
            caesura_code = "10v (non césuré)"
            caesura_label = "セジュール不一致の10音節行"
            meter_code = "10v"

    elif n in (9, 11):
        possible_cuts = [p for p in (4, 5, 6) if is_cuttable(p) and (n - p) <= 8]
        if possible_cuts:
            p = possible_cuts[0]
            caesura_code = f"{p}+{n - p} (impair)"
            caesura_label = f"奇数脚複合分節 ({p}+{n - p})"
            caesura_pos = p
            meter_code = f"{p}+{n - p}"
        else:
            caesura_code = f"{n}v (impair)"
            caesura_label = f"奇数音節行 ({n}v・アンペール)"
            meter_code = f"{n}v"

    elif n <= 8:
        parity = "偶数単一脚" if n % 2 == 0 else "奇数単一脚 (vers impair)"
        caesura_label = f"{parity} ({n}音節・全体知覚可能)"
        meter_code = f"{n}v"

    # リズム分節 (Mesures rythmiques: Grammont / Mazaleyrat)
    measures_info = compute_rhythmic_measures(counted, caesura_pos, v_words=v_words)

    return {
        "line": line_idx,
        "anatonic_length": n,
        "genre": genre,
        "meter_class": meter_class,
        "loi_des_8": loi_8,
        "meter_code": meter_code,
        "caesura_code": caesura_code,
        "caesura_label": caesura_label,
        "caesura_pos": caesura_pos,
        "ternary_profile": ternary_profile,
        "cpmf_string": cpmf_str,
        "measures": measures_info["measures"],
        "measures_str": measures_info["measures_str"],
        "measure_type": measures_info["measure_type"],
    }


def compute_rhythmic_measures(counted_sylls, caesura_pos=None, v_words=None):
    """半句・詩行内の統辞的結束と強勢境界に基づき、リズム音尺 (例: 5+1 // 3+3, 2+4 // 3+3, 3+3 // 3+3) を精密算出する。"""
    n = len(counted_sylls)
    if n == 0:
        return {"accent_positions": [], "measures": [], "measures_str": "", "measure_type": "空"}

    # 各語の最終男性強勢位置（1-based）マップを構築（NOUN+ADJ等の結束長判定に使用）
    word_dvm_pos = {}
    for idx_1, s in enumerate(counted_sylls, 1):
        if s["is_word_final_masc"]:
            word_dvm_pos[s["word_idx"]] = idx_1

    def _segment_span(start_1based, end_1based):
        """区間 [start_1based, end_1based] 内のリズム音尺終端リスト（1-based）を返す。"""
        accents = []
        prev_acc = start_1based - 1
        for idx in range(start_1based, end_1based + 1):
            s = counted_sylls[idx - 1]
            if idx == end_1based:
                accents.append(idx)
                prev_acc = idx
                continue

            if s["cpmf"] != "0" or not s["is_word_final_masc"]:
                continue

            has_punct = bool(s["punct_after"])
            if has_punct:
                accents.append(idx)
                prev_acc = idx
                continue

            w_low = s["word_lower"].split("'")[-1]
            if w_low in WEAK_CONJ_REL:
                continue

            # 次語の情報を確認
            w_idx = s["word_idx"]
            nxt_w = v_words[w_idx + 1] if (v_words and w_idx + 1 < len(v_words)) else None
            nxt_dvm = word_dvm_pos.get(w_idx + 1, end_1based)

            # 1. 数詞 + 名詞/形容詞 (例: Dix nuits) は一体となって1音尺をなす
            if (s["pos"] == "NUM" or w_low in CARDINAL_NUMERALS) and nxt_w is not None:
                continue

            # 2. 助動詞 (ayant, étant, est, a, ont...) + 過去分詞/動詞/属詞形容詞 (例: Les ayant cloués = 5) は一体となる
            if s["pos"] == "AUX" and nxt_w is not None and nxt_w.pos_ in ("VERB", "ADJ", "AUX") and (nxt_dvm - prev_acc) <= 5:
                continue

            # 3. 短い副詞 (plus, très, bien, si, trop, tout, toujours...) + 形容詞/動詞/副詞 (合計4音節以下: plus sourd, toujours aimer) は一体となる
            if s["pos"] == "ADV" and s["word_syl_len"] <= 2 and nxt_w is not None and nxt_w.pos_ in ("ADJ", "VERB", "ADV") and (nxt_dvm - prev_acc) <= 4:
                continue

            # 4. 前置形容詞 + 名詞 (合計4音節以下: de noirs parfums = 4, l'autre hiver = 3) は一体となる
            if s["pos"] == "ADJ" and nxt_w is not None and nxt_w.pos_ == "NOUN" and (nxt_dvm - prev_acc) <= 4:
                continue

            # 5. 名詞 + 後置形容詞 (合計4音節以下: l'oeil niais = 3, des gouffres bruns = 4) は1音尺に結束する（5音節超の Échouages hideux = 4+2 は分割）
            if s["pos"] in ("NOUN", "PROPN") and nxt_w is not None and nxt_w.pos_ == "ADJ" and (nxt_dvm - prev_acc) <= 4:
                continue

            # 6. 直前の強勢境界から2音節以上進んでいればリズム強勢を付与（残り1音節の 5+1: Les ayant cloués (5) / nus (1) も正当に許容！）
            if idx - prev_acc >= 2:
                accents.append(idx)
                prev_acc = idx

        if not accents or accents[-1] != end_1based:
            accents.append(end_1based)
        return accents

    if caesura_pos and 1 <= caesura_pos < n and counted_sylls[caesura_pos - 1]["cpmf"] in ("0", "L"):
        acc1 = _segment_span(1, caesura_pos)
        acc2 = _segment_span(caesura_pos + 1, n)
        accent_positions = acc1 + acc2
    else:
        accent_positions = _segment_span(1, n)

    measures = []
    prev = 0
    hem1_measures = []
    hem2_measures = []
    for pos in accent_positions:
        length = pos - prev
        measures.append(length)
        if caesura_pos:
            if pos <= caesura_pos:
                hem1_measures.append(length)
            else:
                hem2_measures.append(length)
        prev = pos

    if caesura_pos and sum(hem1_measures) == caesura_pos and hem2_measures:
        m_str = "+".join(map(str, hem1_measures)) + " // " + "+".join(map(str, hem2_measures))
    else:
        m_str = "+".join(map(str, measures))

    # リズム類型判定 (Grammont / Mazaleyrat)
    m_type = "多律動・複合型"
    if n == 12:
        if measures == [3, 3, 3, 3]:
            m_type = "均衡テトラメートル (3+3 // 3+3・アナペスト型)"
        elif measures == [2, 4, 2, 4]:
            m_type = "漸増並行テトラメートル (2+4 // 2+4)"
        elif measures == [4, 2, 4, 2]:
            m_type = "漸減並行テトラメートル (4+2 // 4+2)"
        elif measures in ([2, 4, 4, 2], [4, 2, 2, 4]):
            m_type = f"交差対称テトラメートル ({m_str})"
        elif measures == [4, 4, 4]:
            m_type = "ロマン派トリメートル (4+4+4・三拍子)"
        elif len(measures) == 4 and sum(measures[:2]) == 6:
            m_type = f"古典4律動アレクサンドラン ({m_str})"
        elif len(measures) == 3 and (measures[0] == 6 or sum(measures[:2]) == 6):
            m_type = f"3律動アレクサンドラン ({m_str})"
        elif len(measures) >= 5:
            m_type = f"累積・緩徐アレクサンドラン ({len(measures)}音尺)"
    elif n <= 8:
        if len(measures) == 2 and measures[0] == measures[1]:
            m_type = f"二等分律動 ({m_str})"
        elif len(measures) == 2 and measures[0] < measures[1]:
            m_type = f"漸増律動 ({m_str})"
        elif len(measures) == 2 and measures[0] > measures[1]:
            m_type = f"漸減律動 ({m_str})"
        else:
            m_type = f"単一詩句律動 ({m_str})"

    return {
        "accent_positions": accent_positions,
        "measures": measures,
        "measures_str": m_str,
        "measure_type": m_type,
    }


def diagnose_corpus_metricality(cornulier_lines, lines_meta):
    """テクスト全体の音節分布・連構造・双子X6 (X6 jumeaux) から、等音節定型詩・周期的異音節詩・自由詩 (Cornulier / Murat) を総合診断する。"""
    lengths = [c["anatonic_length"] for c in cornulier_lines if c["anatonic_length"] > 0]
    if not lengths:
        return {
            "global_form": "解析対象なし",
            "dominant_meter": "なし",
            "isometry_ratio": 0.0,
            "vers_simples_pct": 0.0,
            "vers_composes_pct": 0.0,
            "hors_metre_pct": 0.0,
            "caesura_summary": {},
            "alexandrin_count": 0,
            "x6_jumeaux_count": 0,
        }

    cnt = collections.Counter(lengths)
    dom_len, dom_freq = cnt.most_common(1)[0]
    iso_ratio = round(100.0 * dom_freq / len(lengths), 1)

    n_total = len(lengths)
    v_simple = round(100.0 * sum(1 for x in lengths if x <= 8) / n_total, 1)
    v_comp = round(100.0 * sum(1 for x in lengths if 9 <= x <= 12) / n_total, 1)
    v_hors = round(100.0 * sum(1 for x in lengths if x > 12) / n_total, 1)

    stanza_map = collections.defaultdict(list)
    for c, m in zip(cornulier_lines, lines_meta):
        if c["anatonic_length"] > 0:
            stanza_map[m["stanza"]].append(c["anatonic_length"])
    stanza_tuples = list(stanza_map.values())
    periodic_heterometry = (
        len(stanza_tuples) >= 2
        and len(set(tuple(s) for s in stanza_tuples)) == 1
        and len(set(stanza_tuples[0])) >= 2
    )

    alex_lines = [c for c in cornulier_lines if c["anatonic_length"] == 12]
    caesura_counts = collections.Counter(c["caesura_code"] for c in alex_lines)

    # Cornulier (2004): X6 jumeaux（連続する脱古典セジュール行のペア数）
    def _is_x6(c_row):
        return c_row["anatonic_length"] == 12 and c_row["caesura_code"].startswith(("F6", "FC6", "C6", "P6", "L6", "Morph6", "M6", "S6"))

    x6_jumeaux = sum(
        1 for a, b in zip(cornulier_lines, cornulier_lines[1:]) if _is_x6(a) and _is_x6(b)
    )

    if iso_ratio >= 90.0 and dom_len <= 12:
        has_post_classical = any(
            k.startswith(("F6", "FC6", "C6", "P6", "L6", "Morph6", "M6", "S6")) for k in caesura_counts
        )
        meter_names = {12: "アレクサンドラン (12音節)", 10: "デカシラブ (10音節)", 8: "オクタシラブ (8音節)", 7: "エプタシラブ (7音節)"}
        m_name = meter_names.get(dom_len, f"{dom_len}音節等音節詩")
        if dom_len == 12 and has_post_classical:
            global_form = f"近代解放・脱古典アレクサンドラン ({m_name}・一致率 {iso_ratio}%)"
        else:
            global_form = f"等音節定型詩 (Isométrique: {m_name}・一致率 {iso_ratio}%)"
    elif periodic_heterometry:
        pat_str = "-".join(map(str, stanza_tuples[0]))
        global_form = f"周期的異音節定型詩 (Hétérométrie périodique: 連構造 {pat_str})"
    elif v_hors == 0.0 and iso_ratio >= 60.0:
        global_form = f"混成韻律詩 / 解放詩 (Vers mêlés / libérés: 主軸 {dom_len}音節 {iso_ratio}%)"
    else:
        global_form = "自由詩 / 非周期的ヴェルセ (Vers libre: 非等音節・行単位焦点化)"

    return {
        "global_form": global_form,
        "dominant_meter": f"{dom_len}音節 ({iso_ratio}%)",
        "isometry_ratio": iso_ratio,
        "vers_simples_pct": v_simple,
        "vers_composes_pct": v_comp,
        "hors_metre_pct": v_hors,
        "caesura_summary": dict(caesura_counts),
        "alexandrin_count": len(alex_lines),
        "x6_jumeaux_count": x6_jumeaux,
    }


# =============================================================================
# Pillar 4: Gouvard / Mazaleyrat / Plénat (音韻的脚韻・詩節構造・統辞不一致)
# =============================================================================

def extract_line_rhyme_signature(ws, line_stream):
    """Plénat (1991) & Gouvard に基づき、行末の最終男性強勢母音から音韻的脚韻単位を抽出する。"""
    if not ws or not line_stream["counted_syllables"]:
        return {
            "word": "",
            "word_lower": "",
            "genre": "M",
            "tonic_vowel": "",
            "appui_consonant": "",
            "coda_consonants": "",
            "rhyme_unit_ipa": "",
            "rhyme_phonemes": [],
            "core_phonemes": [],
            "full_tail_phonemes": [],
        }

    counted = line_stream["counted_syllables"]
    is_fem = line_stream["has_extrametrical_final_e"]
    last_masc = counted[-1]
    tonic_v = last_masc["nucleus"]
    appui_c = last_masc["onset"][-1] if last_masc["onset"] else ""

    # 女性韻の場合、超音節の頭子音 (例: ombre の bʁ) も脚韻子音の一部！
    coda_list = list(last_masc["coda"])
    if is_fem and line_stream["syllables"][-1].get("is_extrametrical"):
        ext_onset = line_stream["syllables"][-1].get("onset", [])
        if ext_onset and not coda_list:
            coda_list = list(ext_onset)

    rhyme_phs = ([appui_c] if appui_c else []) + [tonic_v] + coda_list
    rhyme_nucleus_and_coda = [tonic_v] + coda_list

    pre_phs = []
    if len(counted) >= 2:
        prev_s = counted[-2]
        pre_phs = prev_s["onset"] + [prev_s["nucleus"]] + prev_s["coda"] + last_masc["onset"]

    return {
        "word": ws[-1].text,
        "word_lower": ws[-1].text.lower(),
        "genre": "F" if is_fem else "M",
        "tonic_vowel": tonic_v,
        "appui_consonant": appui_c,
        "coda_consonants": "".join(coda_list),
        "rhyme_unit_ipa": "".join(rhyme_nucleus_and_coda),
        "rhyme_phonemes": rhyme_phs,
        "core_phonemes": rhyme_nucleus_and_coda,
        "full_tail_phonemes": pre_phs + [tonic_v] + coda_list,
    }


def compare_rhyme_pair(sig1, sig2):
    """2行の脚韻シグネチャを比較し、押韻成否・豊かさ (pauvre / suffisante / riche / léonine)・半諧音 (assonance / contre-assonance) を判定する。"""
    if not sig1["tonic_vowel"] or not sig2["tonic_vowel"]:
        return {"is_rhyme": False, "richness": "なし", "shared_phonemes": 0, "genre_match": False}

    # 同一語の反復は古典的脚韻ではなく同語反復 (mot-rime) だが音韻的一致として識別
    same_word = sig1["word_lower"] == sig2["word_lower"]
    genre_match = sig1["genre"] == sig2["genre"]

    # 完全脚韻条件: 最終強勢母音 + 後続子音群が完全一致
    if sig1["core_phonemes"] == sig2["core_phonemes"]:
        # 末尾から連続一致する音素数をカウント
        t1 = sig1["full_tail_phonemes"]
        t2 = sig2["full_tail_phonemes"]
        shared = 0
        for p1, p2 in zip(reversed(t1), reversed(t2)):
            if p1 == p2:
                shared += 1
            else:
                break

        # 母音が2つ以上含まれればレオニン韻 (rime léonine)
        shared_slice = t1[len(t1) - shared:] if shared > 0 else []
        shared_vowels = sum(1 for p in shared_slice if p in VOWELS_ALL)

        if same_word:
            rich_label = "同語脚韻 (mot-rime)"
        elif not genre_match:
            rich_label = f"両性脚韻 (rime androgyne: {shared}音素一致)"
        elif shared_vowels >= 2 and shared >= 4:
            rich_label = "レオニン韻・二音節韻 (rime léonine)"
        elif shared >= 3:
            rich_label = "富韻 (rime riche: 3音素以上)"
        elif shared == 2:
            rich_label = "足韻 (rime suffisante: 2音素)"
        else:
            rich_label = "貧韻 (rime pauvre: 1音素)"

        return {
            "is_rhyme": True,
            "richness": rich_label,
            "richness_code": "mot_rime" if same_word else (
                "leonine" if (shared_vowels >= 2 and shared >= 4) else (
                    "riche" if shared >= 3 else ("suffisante" if shared == 2 else "pauvre")
                )
            ),
            "shared_phonemes": shared,
            "genre_match": genre_match,
        }

    # アソナンス (母音一致・後続子音不一致)
    if sig1["tonic_vowel"] == sig2["tonic_vowel"]:
        return {
            "is_rhyme": False,
            "is_assonance": True,
            "richness": f"母音半諧音 (assonance: [{sig1['tonic_vowel']}])",
            "richness_code": "assonance",
            "shared_phonemes": 1,
            "genre_match": genre_match,
        }

    # コントル・アソナンス (母音不一致・非空の後続子音群が一致)
    if sig1["coda_consonants"] and sig1["coda_consonants"] == sig2["coda_consonants"]:
        return {
            "is_rhyme": False,
            "is_assonance": False,
            "is_contre_assonance": True,
            "richness": f"子音半諧音 (contre-assonance: [-{sig1['coda_consonants']}])",
            "richness_code": "contre_assonance",
            "shared_phonemes": len(sig1["coda_consonants"]),
            "genre_match": genre_match,
        }

    return {"is_rhyme": False, "richness": "無韻 (vers blanc)", "richness_code": "none", "shared_phonemes": 0, "genre_match": genre_match}


def analyze_stanzas_and_rhymes(lines_meta, rhyme_sigs):
    """連ごとの脚韻スキーム (AABB, ABAB, ABBA, Sonnet等)・豊かさ・男女交替規則を解析する。"""
    stanza_groups = collections.defaultdict(list)
    for idx, (m, sig) in enumerate(zip(lines_meta, rhyme_sigs)):
        stanza_groups[m["stanza"]].append((idx, sig))

    line_rhyme_info = [{} for _ in lines_meta]
    stanzas_detail = []
    richness_counter = collections.Counter()
    alternance_violations = 0
    alternance_checks = 0

    for st_num, items in stanza_groups.items():
        letters = []
        key_to_letter = {}
        next_char_ord = ord("A")
        pair_matches = {}

        # 各行の音韻キー (core_phonemes + genre または音韻単独)
        for local_i, (g_idx, sig) in enumerate(items):
            core_key = "".join(sig["core_phonemes"])
            matched_letter = None
            matched_partner_idx = None
            best_cmp = None

            # 直近6行以内で脚韻一致を探索
            for prev_local in range(local_i - 1, max(-1, local_i - 7), -1):
                prev_g_idx, prev_sig = items[prev_local]
                cmp_res = compare_rhyme_pair(prev_sig, sig)
                if cmp_res["is_rhyme"]:
                    matched_letter = letters[prev_local]
                    matched_partner_idx = prev_g_idx
                    best_cmp = cmp_res
                    break

            if matched_letter is None:
                # アソナンス探索（脚韻がない場合）
                for prev_local in range(local_i - 1, max(-1, local_i - 5), -1):
                    prev_g_idx, prev_sig = items[prev_local]
                    cmp_res = compare_rhyme_pair(prev_sig, sig)
                    if cmp_res.get("is_assonance") or cmp_res.get("is_contre_assonance"):
                        best_cmp = cmp_res
                        matched_partner_idx = prev_g_idx
                        break

                matched_letter = chr(min(next_char_ord, ord("Z")))
                key_to_letter[core_key] = matched_letter
                next_char_ord += 1

            letters.append(matched_letter)
            if best_cmp:
                pair_matches[g_idx] = (matched_partner_idx, best_cmp)
                if matched_partner_idx not in pair_matches:
                    pair_matches[matched_partner_idx] = (g_idx, best_cmp)
                richness_counter[best_cmp["richness_code"]] += 1

        # 連内の男女交替チェック (異なる脚韻文字が隣接するとき M/F が異なるべき)
        st_alt_ok = True
        for k in range(len(items) - 1):
            if letters[k] != letters[k + 1]:
                alternance_checks += 1
                if items[k][1]["genre"] == items[k + 1][1]["genre"]:
                    alternance_violations += 1
                    st_alt_ok = False

        scheme_str = "".join(letters)
        disp_type = _classify_stanza_scheme(scheme_str)
        genres_str = "".join(it[1]["genre"] for it in items)

        for local_i, (g_idx, sig) in enumerate(items):
            partner_info = pair_matches.get(g_idx)
            line_rhyme_info[g_idx] = {
                "rhyme_letter": letters[local_i],
                "rhyme_ipa": sig["rhyme_unit_ipa"],
                "rhyme_genre": sig["genre"],
                "rhyme_richness": partner_info[1]["richness"] if partner_info else "無韻 / 単独行",
                "rhyme_partner_line": (partner_info[0] + 1) if partner_info else None,
            }

        stanzas_detail.append({
            "stanza": st_num,
            "line_count": len(items),
            "scheme": scheme_str,
            "disposition": disp_type,
            "genres": genres_str,
            "alternance_respected": st_alt_ok,
        })

    # ソネット全体判定 (4-4-3-3 連構成)
    st_sizes = [s["line_count"] for s in stanzas_detail]
    poetic_fixed_form = "自由連構成 / その他"
    if st_sizes == [4, 4, 3, 3]:
        s1, s2 = stanzas_detail[0]["scheme"], stanzas_detail[1]["scheme"]
        if s1 == "ABBA" and s2 == "ABBA":
            poetic_fixed_form = "正統派フランス・ソネット (Sonnet régulier: 4-4-3-3, ABBA)"
        else:
            poetic_fixed_form = "変則・近代ソネット (Sonnet néo-classique / élisabéthain: 4-4-3-3)"
    elif all(sz == 4 for sz in st_sizes) and st_sizes:
        poetic_fixed_form = "四行連詩 (Quatrains)"
    elif all(sz == 2 for sz in st_sizes) and st_sizes:
        poetic_fixed_form = "二行連詩 (Distiques)"
    elif len(st_sizes) == 1:
        poetic_fixed_form = f"単一連詩 (Poème monostrophique: {st_sizes[0]}行)"

    alt_rate = round(100.0 * (alternance_checks - alternance_violations) / alternance_checks, 1) if alternance_checks > 0 else 100.0

    return {
        "line_rhyme_info": line_rhyme_info,
        "stanzas_detail": stanzas_detail,
        "fixed_form": poetic_fixed_form,
        "richness_counts": dict(richness_counter),
        "alternance_rate": alt_rate,
    }


def _classify_stanza_scheme(scheme):
    if len(scheme) == 4:
        if scheme == "AABB":
            return "平韻 (rimes plates: AABB)"
        if scheme == "ABAB":
            return "交差韻 (rimes croisées: ABAB)"
        if scheme == "ABBA":
            return "抱擁韻 (rimes embrassées: ABBA)"
        if scheme == "AAAA":
            return "単一韻 (monorime: AAAA)"
    if len(scheme) == 2 and scheme == "AA":
        return "平韻二行連 (distique à rimes plates: AA)"
    if len(scheme) == 3:
        if scheme == "AAA":
            return "三行単一韻 (tercet monorime: AAA)"
        if len(set(scheme)) <= 2:
            return f"押韻三行連 (tercet: {scheme})"
    if len(scheme) == 6 and scheme == "AABCCB":
        return "尾韻六行連 (sizain caudé: AABCCB)"
    # 隣接平韻ペア連続チェック
    if len(scheme) >= 4 and len(scheme) % 2 == 0 and all(scheme[k] == scheme[k + 1] for k in range(0, len(scheme), 2)):
        return f"連続平韻 (rimes suivies: {scheme})"
    if len(set(scheme)) == len(scheme):
        return "無韻 (vers blancs / non rimés)"
    return f"複合・混成押韻 ({scheme})"


def detect_syntactic_discordances(lines_meta, docs, unit_words, line_streams, cornulier_lines):
    """Mazaleyrat & Gouvard に基づき、行末およびセジュールにおける Enjambement（句またがり）・Rejet（送り）・Contre-rejet（逆送り）を検出する。"""
    discordances = []
    line_flags = [[] for _ in lines_meta]
    n_lines = len(lines_meta)
    terminal_punct_chars = set(".,;:!?…»—)")

    for i in range(n_lines):
        doc = docs[i]
        ws = unit_words[i]
        stream = line_streams[i]
        corn = cornulier_lines[i]
        if not ws:
            continue

        raw_text = lines_meta[i]["text"].strip()
        ends_with_punct = bool(raw_text and raw_text[-1] in terminal_punct_chars)

        # 1. 行間 (Externe) の統辞＝韻律不一致判定 (同一連内の次行との間)
        if i + 1 < n_lines and lines_meta[i + 1]["stanza"] == lines_meta[i]["stanza"] and unit_words[i + 1]:
            ws_next = unit_words[i + 1]
            stream_next = line_streams[i + 1]
            last_w = ws[-1]
            first_next = ws_next[0]

            # 次行冒頭4音節以内の急停止（Rejet候補）を先に計測（行頭大文字によるPROPN誤認を救済）
            counted_next = stream_next["counted_syllables"]
            early_stop_syl = None
            early_words = []
            for s_idx, s in enumerate(counted_next, 1):
                if s["is_word_end"]:
                    early_words.append(s["word_text"])
                if s["punct_after"] and any(p in ".!?;:," for p in s["punct_after"]):
                    if s_idx <= 4:
                        early_stop_syl = s_idx
                    break

            # 強い統辞的結合が行末を跨ぐ条件
            tight_syntactic_link = False
            if not ends_with_punct:
                if last_w.pos_ in ("DET", "ADP", "CCONJ", "SCONJ", "AUX", "PRON"):
                    tight_syntactic_link = True
                elif early_stop_syl is not None and first_next.text.lower() not in {"ô", "ah", "eh", "hélas", "oui", "non"}:
                    tight_syntactic_link = True
                elif last_w.pos_ in ("VERB", "NOUN", "ADJ") and first_next.pos_ in ("ADP", "ADJ", "DET", "PRON", "ADV", "VERB"):
                    if first_next.text.lower() not in {"et", "mais", "car", "or", "ni", "cependant", "pourtant"}:
                        tight_syntactic_link = True

            if tight_syntactic_link:
                # Contre-rejet 判定: 現詩行の末尾4音節以内に強い句読点があり、そこから新しい句が始まって次行へ溢れる
                counted_curr = stream["counted_syllables"]
                late_punct_syl = None
                for s_idx, s in enumerate(counted_curr, 1):
                    if s["punct_after"] and any(p in ".!?;:," for p in s["punct_after"]):
                        if len(counted_curr) - s_idx in (1, 2, 3, 4):
                            late_punct_syl = len(counted_curr) - s_idx

                if late_punct_syl is not None and early_stop_syl is None:
                    dtype = "contre-rejet_externe"
                    dlabel = f"外的逆送り (Contre-rejet: 行末{late_punct_syl}音節から次行へ突出)"
                elif early_stop_syl is not None:
                    dtype = "rejet_externe"
                    dlabel = f"外的送り語 (Rejet: 次行頭「{' '.join(early_words)}」[{early_stop_syl}音節]で急停止)"
                else:
                    dtype = "enjambement_externe"
                    dlabel = "外的句またがり (Enjambement: 行末無休止で次行へ連続)"

                discordances.append({
                    "line": i + 1,
                    "next_line": i + 2,
                    "type": dtype,
                    "label": dlabel,
                    "span": f"{last_w.text} // {first_next.text}",
                })
                line_flags[i].append(dtype)

        # 2. セジュール (Interne) の不一致判定
        if corn["anatonic_length"] == 12 and corn["caesura_code"].startswith(("C6", "P6", "M6", "Morph6", "S6")):
            discordances.append({
                "line": i + 1,
                "next_line": i + 1,
                "type": "discordance_cesure",
                "label": f"半句間不一致 ({corn['caesura_code']})",
                "span": raw_text,
            })
            line_flags[i].append("discordance_cesure")

    return {
        "events": discordances,
        "line_flags": line_flags,
        "enjambement_count": sum(1 for e in discordances if e["type"] == "enjambement_externe"),
        "rejet_count": sum(1 for e in discordances if e["type"] == "rejet_externe"),
        "contre_rejet_count": sum(1 for e in discordances if e["type"] == "contre-rejet_externe"),
        "cesure_discordance_count": sum(1 for e in discordances if e["type"] == "discordance_cesure"),
    }


# =============================================================================
# Pillar 3: Albert Di Cristo (4階層韻律句構造・双極アクセント AI/AF/AN・定量リズム指標)
# =============================================================================

def analyze_di_cristo_line(ws, line_stream, line_idx=1):
    """Di Cristo (1999, 2016) の4階層韻律構成素 (UT < MP < SP < UI) と双極アクセント弧 {AI ... AF/AN} を構築する。"""
    counted = line_stream["counted_syllables"]
    phon_sylls = line_stream["phonological_syllables"]
    n = len(counted)
    if n == 0:
        return {
            "line": line_idx,
            "phrasing_str": "",
            "mp_count": 0,
            "sp_count": 0,
            "ui_count": 0,
            "ut_lengths": [],
            "sp_lengths": [],
            "ai_count": 0,
            "af_count": 0,
            "an_count": 0,
            "arcs_count": 0,
            "syllable_marks": [],
            "open_syl_count": 0,
            "total_phon_syl": 0,
            "npvi_interne_durations": [],
            "npvi_global_durations": [],
        }

    # 1. MP (Mot Prosodique / Groupe clitique) の形成
    # 前接語 (C, CF, P) を後続の実詞ヘッドへ束ね、ハイフン後接語 (E1, E2, E3) の強勢位置を決定する
    mps = []
    cur_syl_indices = []
    for idx, s in enumerate(counted):
        cur_syl_indices.append(idx)
        is_last_syl = idx == n - 1
        # MP終端判定: 語末であり、かつ前接語(C, CF, P, M, L, Morph)でないか、句読点があるか、行末
        next_is_post_fem_same_word = (
            not is_last_syl and counted[idx + 1]["word_idx"] == s["word_idx"]
        )
        if next_is_post_fem_same_word:
            continue

        is_proclitic_word = s["cpmf"] in ("C", "CF", "P") and not s["punct_after"] and not is_last_syl
        if not is_proclitic_word:
            # MP確定: 主強勢位置 (AF候補) を特定（E2後接語 -je/-ce/-on の場合は動詞語幹へ戻す）
            w_raw = s["word_text"].lower()
            tonic_syl_idx = idx
            if s["is_post_tonic_fem"] and len(cur_syl_indices) >= 2:
                tonic_syl_idx = cur_syl_indices[-2]
            elif "-" in w_raw:
                hyph_parts = [p for p in w_raw.split("-") if p]
                if len(hyph_parts) >= 2 and hyph_parts[-1] in ENCLITICS_ATONIC_E2 and len(cur_syl_indices) >= 2:
                    tonic_syl_idx = cur_syl_indices[-2]
                elif (
                    len(hyph_parts) >= 2
                    and hyph_parts[-1] in ENCLITICS_CONDITIONAL_E3
                    and not is_last_syl
                    and counted[idx + 1]["word_syl_len"] == 1
                    and len(cur_syl_indices) >= 2
                ):
                    tonic_syl_idx = cur_syl_indices[-2]

            # MP内の最初の語彙音節 (AI付与候補)
            first_lex_idx = cur_syl_indices[0]
            for cand_i in cur_syl_indices:
                if counted[cand_i]["cpmf"] not in ("C", "CF", "P") and counted[cand_i]["word_lower"] not in WEAK_CONJ_REL:
                    first_lex_idx = cand_i
                    break

            mps.append({
                "syl_indices": list(cur_syl_indices),
                "tonic_idx": tonic_syl_idx,
                "first_lex_idx": first_lex_idx,
                "head_pos": s["pos"],
                "head_word": s["word_text"],
                "punct_after": s["punct_after"],
            })
            cur_syl_indices = []

    if cur_syl_indices:
        mps.append({
            "syl_indices": list(cur_syl_indices),
            "tonic_idx": cur_syl_indices[-1],
            "first_lex_idx": cur_syl_indices[0],
            "head_pos": counted[cur_syl_indices[-1]]["pos"],
            "head_word": counted[cur_syl_indices[-1]]["word_text"],
            "punct_after": counted[cur_syl_indices[-1]]["punct_after"],
        })

    # 2. SP (Syntagme Prosodique) と UI (Unité Intonative) への統合
    # 密接な統辞結合 (DET+ADJ+NOUN, NOUN+ADJ[短], ADV+ADJ, AUX+VERB) で合計6音節以下かつ句読点なしなら1つのSPへ統合
    sps = []
    i = 0
    while i < len(mps):
        cur_mp = mps[i]
        merged_mps = [cur_mp]
        while i + 1 < len(mps):
            prev_m = merged_mps[-1]
            nxt_m = mps[i + 1]
            if prev_m["punct_after"]:
                break
            combined_len = sum(len(m["syl_indices"]) for m in merged_mps) + len(nxt_m["syl_indices"])
            cohesive_pair = (
                (prev_m["head_pos"] == "ADJ" and nxt_m["head_pos"] == "NOUN")
                or (prev_m["head_pos"] == "NOUN" and nxt_m["head_pos"] == "ADJ" and combined_len <= 8)
                or (prev_m["head_pos"] == "ADV" and nxt_m["head_pos"] in ("ADJ", "VERB", "ADV") and combined_len <= 8)
                or (prev_m["head_pos"] == "AUX" and nxt_m["head_pos"] in ("VERB", "ADJ", "NOUN") and combined_len <= 8)
            )
            if cohesive_pair and combined_len <= 8:
                merged_mps.append(nxt_m)
                i += 1
            else:
                break

        sp_syls = [idx for m in merged_mps for idx in m["syl_indices"]]
        last_mp = merged_mps[-1]
        is_ui_end = (i == len(mps) - 1) or bool(last_mp["punct_after"] and any(p in ".,;:!?…—»)" for p in last_mp["punct_after"]))
        sps.append({
            "mps": merged_mps,
            "syl_indices": sp_syls,
            "final_tonic_idx": last_mp["tonic_idx"],
            "is_ui_end": is_ui_end,
        })
        i += 1

    # 3. 双極アクセント付与 (AN, AF, AI) と アクセント弧 {AI ... AF}
    syl_accents = ["" for _ in range(n)]
    arcs = []

    for sp in sps:
        ft_idx = sp["final_tonic_idx"]
        syl_accents[ft_idx] = "AN" if sp["is_ui_end"] else "AF"
        sp_mps = sp["mps"]
        sp_len = len(sp["syl_indices"])
        ai_assigned_in_sp = None

        # (R1) Intra-SP Clash Resolution: 同一SP内で先行MPの強勢母音と後続MPの強勢母音が隣接または1音節差で衝突するとき、先行MPの語頭へ AI 移動！（SP境界を跨ぐ移動は厳禁）
        if len(sp_mps) >= 2:
            m1 = sp_mps[0]
            m2 = sp_mps[-1]
            if m2["tonic_idx"] - m1["tonic_idx"] <= 1 and m1["first_lex_idx"] < m1["tonic_idx"]:
                ai_assigned_in_sp = m1["first_lex_idx"]
            # (R2) Cohesive Syntagm Double Marking: 4〜7音節の複数語SPで冒頭実詞に AI を置き {AI ... AF} アクセント弧を形成
            elif 4 <= sp_len <= 8 and m1["first_lex_idx"] < ft_idx - 1:
                ai_assigned_in_sp = m1["first_lex_idx"]

        # (R3) Lapse Avoidance: 単一MPでも5音節以上の長い韻律語（かつ語彙語が3音節以上）なら第1語彙音節に AI を付与
        if ai_assigned_in_sp is None and sp_len >= 5:
            m0 = sp_mps[0]
            lex_word_len = counted[m0["first_lex_idx"]]["word_syl_len"]
            if lex_word_len >= 3 and m0["first_lex_idx"] < ft_idx - 1:
                ai_assigned_in_sp = m0["first_lex_idx"]

        if ai_assigned_in_sp is not None and syl_accents[ai_assigned_in_sp] == "":
            syl_accents[ai_assigned_in_sp] = "AI"
            arcs.append((ai_assigned_in_sp, ft_idx))

    # 4. UT (Unité Tonale / Pied) の区間長と nPVI 用模擬持続時間モデル
    ut_lengths = []
    last_prom = -1
    for idx, acc in enumerate(syl_accents):
        if acc in ("AI", "AF", "AN"):
            ut_lengths.append(idx - last_prom)
            last_prom = idx
    if last_prom < n - 1:
        # 行末女性音節（算入内）がある場合は直前強勢脚へ吸収
        if ut_lengths:
            ut_lengths[-1] += (n - 1 - last_prom)
        else:
            ut_lengths.append(n)

    sp_lengths = [len(sp["syl_indices"]) for sp in sps]

    # Di Cristo 二重 nPVI 用の音節持続時間推定値（基本無強勢=1.0, AI=1.22, 長化子音閉音節=+0.18, AF=1.65, AN=2.05, シュワー=0.78）
    dur_global = []
    dur_interne = []
    for idx, s in enumerate(counted):
        acc = syl_accents[idx]
        d = 0.78 if s["is_schwa"] else 1.0
        if s["coda"] and s["coda"][-1] in LENGTHENING_CONSONANTS:
            d += 0.18
        if acc == "AI":
            d *= 1.22
        elif acc == "AF":
            d *= 1.65
        elif acc == "AN":
            d *= 2.05
        dur_global.append(d)
        if acc != "AN":
            dur_interne.append(d if acc != "AF" else 1.15)

    # 5. 表示用の韻律句構造文字列 ({AI ... AF} }, UI ])
    arc_starts = {a[0] for a in arcs}
    arc_ends = {a[1] for a in arcs}
    sp_ends = {sp["syl_indices"][-1]: sp["is_ui_end"] for sp in sps if sp["syl_indices"]}

    tokens_out = []
    for idx, s in enumerate(counted):
        prefix = "{" if idx in arc_starts else ""
        acc = syl_accents[idx]
        orth_disp = s["orth"].upper() if acc in ("AF", "AN") else (f"ˈ{s['orth']}" if acc == "AI" else s["orth"])
        tag_sup = f"[{acc}]" if acc else ""
        arc_close = "}" if idx in arc_ends else ""
        bound = ""
        if idx in sp_ends:
            bound = " ‖]" if sp_ends[idx] else " |)"
        elif acc == "AI":
            bound = " )"
        sep = " " if s["is_word_end"] else "·"
        tokens_out.append(f"{prefix}{orth_disp}{tag_sup}{arc_close}{bound}{sep}")

    phrasing_str = "".join(tokens_out).strip()

    open_cnt = sum(1 for ps in phon_sylls if ps["is_open"])

    return {
        "line": line_idx,
        "phrasing_str": phrasing_str,
        "mp_count": len(mps),
        "sp_count": len(sps),
        "ui_count": sum(1 for sp in sps if sp["is_ui_end"]),
        "ut_lengths": ut_lengths,
        "sp_lengths": sp_lengths,
        "ai_count": syl_accents.count("AI"),
        "af_count": syl_accents.count("AF"),
        "an_count": syl_accents.count("AN"),
        "arcs_count": len(arcs),
        "arcs": arcs,
        "syllable_marks": [
            {
                "syl_idx": idx + 1,
                "orth": counted[idx]["orth"],
                "ipa": counted[idx]["ipa"],
                "accent": syl_accents[idx],
                "in_arc": any(a0 <= idx <= a1 for a0, a1 in arcs),
            }
            for idx in range(n)
        ],
        "open_syl_count": open_cnt,
        "total_phon_syl": len(phon_sylls),
        "npvi_interne_durations": dur_interne,
        "npvi_global_durations": dur_global,
    }


def _compute_npvi(durations):
    if len(durations) < 2:
        return 0.0
    diffs = []
    for d1, d2 in zip(durations, durations[1:]):
        denom = (d1 + d2) / 2.0
        if denom > 0:
            diffs.append(abs(d1 - d2) / denom)
    return round(100.0 * st.mean(diffs), 1) if diffs else 0.0


def summarize_di_cristo(dc_lines, line_phonetics_list):
    """テクスト全体の Di Cristo 定量リズム指標 (EurythmyScore, BipolarRatio, ArcDensity, OpenSylRatio, %V, ΔC, 二重nPVI) を集計する。"""
    all_ut = [u for l in dc_lines for u in l["ut_lengths"]]
    all_sp = [s for l in dc_lines for s in l["sp_lengths"]]
    tot_ai = sum(l["ai_count"] for l in dc_lines)
    tot_af = sum(l["af_count"] for l in dc_lines)
    tot_an = sum(l["an_count"] for l in dc_lines)
    tot_arcs = sum(l["arcs_count"] for l in dc_lines)
    tot_sps = sum(l["sp_count"] for l in dc_lines)
    tot_open = sum(l["open_syl_count"] for l in dc_lines)
    tot_psyl = sum(l["total_phon_syl"] for l in dc_lines)

    # EurythmyScore: 理想的な2〜4音節UTの割合から、衝突(1音節)と弛緩(5音節以上)のペナルティを控除
    if all_ut:
        eurythmic_cnt = sum(1 for u in all_ut if 2 <= u <= 4)
        clash_cnt = sum(1 for u in all_ut if u == 1)
        lapse_cnt = sum(1 for u in all_ut if u >= 5)
        raw_eur = (eurythmic_cnt - 0.35 * clash_cnt - 0.35 * lapse_cnt) / len(all_ut)
        eurythmy_score = round(max(0.0, min(1.0, raw_eur)), 3)
        clash_rate = round(100.0 * clash_cnt / len(all_ut), 1)
        lapse_rate = round(100.0 * lapse_cnt / len(all_ut), 1)
        ut_mean = round(st.mean(all_ut), 2)
        ut_sd = round(st.pstdev(all_ut), 2) if len(all_ut) > 1 else 0.0
    else:
        eurythmy_score, clash_rate, lapse_rate, ut_mean, ut_sd = 0.0, 0.0, 0.0, 0.0, 0.0

    sp_mean = round(st.mean(all_sp), 2) if all_sp else 0.0
    bipolar_ratio = round(tot_ai / max(1, tot_af + tot_an), 3)
    arc_density = round(100.0 * tot_arcs / max(1, tot_sps), 1)
    open_syl_ratio = round(100.0 * tot_open / max(1, tot_psyl), 1)

    # Ramus / Di Cristo 音素配列指標 %V_phon と ΔC_phon
    all_phs = [p for lp in line_phonetics_list for p in lp["phonemes"]]
    if all_phs:
        v_cnt = sum(1 for p in all_phs if p in VOWELS_ALL)
        pct_v = round(100.0 * v_cnt / len(all_phs), 1)
        c_runs = []
        cur_c = 0
        for p in all_phs:
            if p not in VOWELS_ALL:
                cur_c += 1
            else:
                if cur_c > 0:
                    c_runs.append(cur_c)
                    cur_c = 0
        if cur_c > 0:
            c_runs.append(cur_c)
        delta_c = round(st.pstdev(c_runs), 2) if len(c_runs) > 1 else 0.0
    else:
        pct_v, delta_c = 0.0, 0.0

    # 行ごとの nPVI 平均
    npvi_int_vals = [_compute_npvi(l["npvi_interne_durations"]) for l in dc_lines if len(l["npvi_interne_durations"]) >= 2]
    npvi_glob_vals = [_compute_npvi(l["npvi_global_durations"]) for l in dc_lines if len(l["npvi_global_durations"]) >= 2]

    return {
        "eurythmy_score": eurythmy_score,
        "ut_mean": ut_mean,
        "ut_sd": ut_sd,
        "sp_mean": sp_mean,
        "clash_rate_pct": clash_rate,
        "lapse_rate_pct": lapse_rate,
        "bipolar_ratio": bipolar_ratio,
        "arc_density_pct": arc_density,
        "total_ai": tot_ai,
        "total_af": tot_af,
        "total_an": tot_an,
        "total_arcs": tot_arcs,
        "open_syl_ratio_pct": open_syl_ratio,
        "pct_v_phon": pct_v,
        "delta_c_phon": delta_c,
        "npvi_interne": round(st.mean(npvi_int_vals), 1) if npvi_int_vals else 0.0,
        "npvi_global": round(st.mean(npvi_glob_vals), 1) if npvi_glob_vals else 0.0,
        "ut_distribution": dict(collections.Counter(all_ut)),
        "sp_distribution": dict(collections.Counter(all_sp)),
    }


# =============================================================================
# Pillar 2: Henri Meschonnic & Gérard Dessons (Traité du rythme・音素連鎖条件・下部ブラケット表記・反アクセント)
# =============================================================================

# Dessons & Meschonnic (1998, pp. 130-137) グループ強勢・識別的二重記号 ⏓ 判定用の閉集合
MESCH_DETERMINATIVE_PRENOMINAL_ADJS = {
    "petit", "petits", "petite", "petites", "grand", "grands", "grande", "grandes",
    "bon", "bons", "bonne", "bonnes", "beau", "beaux", "bel", "belle", "belles",
    "vieux", "vieil", "jeune", "jeunes", "long", "longs", "longue", "longues",
    "haut", "hauts", "haute", "hautes", "même", "mêmes", "autre", "autres",
    "tel", "tels", "telle", "telles", "tout", "tous", "toute", "toutes",
    "seul", "seuls", "seule", "seules", "vrai", "vrais", "vraie", "vraies",
    "faux", "fausse", "fausses", "premier", "premiers", "première", "premières",
    "dernier", "derniers", "dernière", "dernières", "nouveau", "nouvel", "nouvelle", "nouveaux", "nouvelles",
}
MESCH_MODAL_VERBS = {
    "peux", "peut", "pouvons", "pouvez", "peuvent", "pouvait", "pouvaient", "pourra", "pourront", "pu",
    "dois", "doit", "devons", "devez", "doivent", "devait", "devaient", "devra",
    "sais", "sait", "savons", "savez", "savent", "savait", "savaient",
    "veux", "veut", "voulons", "voulez", "veulent", "voulait", "voulaient", "voudras", "voudra",
    "faut", "fallait", "faudra", "ose", "oses", "osent", "osait",
}
MESCH_CAUSATIVE_LIGHT_VERBS = {
    "fais", "fait", "faisons", "faites", "font", "faisait", "faisaient", "fit", "fera", "feront", "faire",
    "vais", "vas", "va", "allons", "allez", "vont", "allait", "allaient",
}
MESCH_TEMPORAL_COPULA_AUX = {
    "ai", "as", "a", "avons", "avez", "ont", "avais", "avait", "avaient", "aurai", "aura", "auront",
    "eus", "eut", "eûmes", "eurent", "aie", "ait", "aient", "eusse", "eût", "ayant", "avoir",
    "suis", "es", "est", "sommes", "êtes", "sont", "étais", "était", "étaient", "serai", "sera", "seront",
    "fus", "fut", "furent", "sois", "soit", "soient", "fusse", "fût", "étant", "été", "être",
}
MESCH_PRE_MODIFIER_ADVS = {
    "pas", "point", "plus", "jamais", "guère", "bien", "très", "trop", "fort", "si",
    "tout", "peu", "tant", "assez", "encor", "encore", "déjà", "toujours", "souvent", "presque", "moins", "aussi",
}
MESCH_POST_VERB_BOUND_ADVS = {
    "pas", "point", "plus", "jamais", "guère", "bien", "mal", "vite", "fort", "court", "doux",
}
MESCH_DISYLLABIC_PREPS = {
    "entre", "contre", "avant", "après", "depuis", "pendant", "selon", "parmi", "malgré", "envers", "outre", "avec", "devant", "derrière",
}


def _extract_meschonnic_opening_consonants(s):
    """Dessons & Meschonnic (1998, p. 121 n.1, pp. 138-139) に従い、音節を開く子音 (consonnes ouvrantes) のみを抽出する。
    - 子音後のわたり音 [w, ɥ, j] (pieds [pje], fruits [fʁɥi], doigts [dwa]) は母音移行部のため除外。
    - 不可分な閉鎖音/摩擦音＋流音 [ʁ, l] 結合 (pro- [pʁɔ], fleurs [flœʁ], trem-bler [ble], l'en-cre [kʁə]) は両子音とも開子音として保持。
    - 単独頭音としての [j] (yeux [jø]) は保持。
    """
    raw_onset = s.get("onset", [])
    if not raw_onset:
        return []
    if raw_onset == ["j"]:
        return ["j"]
    out = []
    for c in raw_onset:
        if c in CONSONANTS and c not in out:
            out.append(c)
    return out


def _detect_meschonnic_line_chains(counted, ext_syl, prev_stream=None, next_stream=None, prev_line_no=None, next_line_no=None):
    """『Traité du rythme』(pp. 99, 138-144, 168-182) の音素連鎖条件を厳密検証し、
    各音節のプロゾディー強勢根拠 (chain_triggers) と下部表記用ブラケット系列 (active_chains) を構築する。
    """
    n_c = len(counted)
    all_sylls = list(counted) + ([ext_syl] if ext_syl is not None else [])
    n_all = len(all_sylls)
    opening_map = [_extract_meschonnic_opening_consonants(s) for s in all_sylls]

    # 近接性上限 (Proximité des phonèmes répétés, pp. 139-141):
    # Couplage (2音節の反復) は同一リズム群・隣接群内 (最大間隔4音節) に限定。
    # 3音節以上の真の系列 (série) は各隣接間隔が4音節以内であれば詩句全体に展開可能。
    max_step = 4

    # 各算入音節 (0..n_c-1) および余剰音節 (n_c) に付与される連鎖根拠リスト
    syl_triggers = [[] for _ in range(n_all)]
    active_chains = []

    # 1. 行内開子音連鎖 (Séries / Couplages de consonnes ouvrantes)
    # ※ 定型韻律内の開子音連鎖は算入音節 (0..n_c-1) のみに限定。
    # 韻律外の余剰女性音節 (ext_syl, index == n_c) は行内音節と連鎖せず、次行への反跳のみに参加する (Traité du rythme p. 181)。
    cons_positions = collections.defaultdict(list)
    for idx in range(n_c):
        for c in opening_map[idx]:
            cons_positions[c].append(idx)

    # 出現順に子音系列を検証
    seen_cons_order = []
    for idx in range(n_c):
        for c in opening_map[idx]:
            if c not in seen_cons_order:
                seen_cons_order.append(c)

    for c in seen_cons_order:
        pos_list = cons_positions[c]
        if len(pos_list) < 2:
            continue
        # 近接性規則 (max_step=4) で連結する局所系列へ分割
        comps = []
        cur_comp = [pos_list[0]]
        for p in pos_list[1:]:
            if p - cur_comp[-1] <= max_step:
                cur_comp.append(p)
            else:
                comps.append(cur_comp)
                cur_comp = [p]
        comps.append(cur_comp)

        for comp in comps:
            if len(comp) < 2:
                continue

            # 規則1: 語間組織化の原則 (Meschonnic p. 168: "unissant des signifiants les uns aux autres")
            # 同一語内部の音節間 (例: vi-vants) のみの反復はプロゾディー連鎖とみなさない
            word_indices = {counted[p]["word_idx"] for p in comp}
            if len(word_indices) < 2:
                continue

            # 規則2: 弱勢単音節機能語 (DET, ADP, PRON, CCONJ, SCONJ) の連続リズム内協力 (Dessons & Meschonnic, pp. 139-141, 168-182)
            # 同一小辞の並列反復 (de... de..., sans... sans...) または連続リズム区間内 (≤4音節・句読点断絶なし) で実詞と協力する場合に連鎖成立
            valid_comp = []
            for p in comp:
                s = counted[p]
                is_tool = s["pos"] in ("DET", "ADP", "PRON", "CCONJ", "SCONJ") and len(s["word_text"]) <= 3
                if is_tool:
                    has_adj = any(abs(p - po) <= 1 for po in comp if po != p)
                    has_same_tool = any(counted[po]["word_lower"] == s["word_lower"] for po in comp if po != p)
                    has_continuous_lex = any(
                        counted[po]["pos"] not in ("DET", "ADP", "PRON", "CCONJ", "SCONJ")
                        and abs(p - po) <= max_step
                        and not any(
                            ch in ".,;:!?…—–()"
                            for k in range(min(p, po), max(p, po))
                            for ch in (counted[k].get("punct_after") or "")
                        )
                        for po in comp
                        if po != p
                    )
                    if has_adj or has_same_tool or has_continuous_lex:
                        valid_comp.append(p)
                else:
                    valid_comp.append(p)

            has_any_lex_in_comp = any(
                counted[p]["pos"] not in ("DET", "ADP", "PRON", "CCONJ", "SCONJ") for p in valid_comp
            )
            has_parallel_tool = len(valid_comp) >= 2 and len({counted[p]["word_lower"] for p in valid_comp}) < len(valid_comp)
            if (
                len(valid_comp) >= 2
                and len({counted[p]["word_idx"] for p in valid_comp}) >= 2
                and (has_any_lex_in_comp or has_parallel_tool)
            ):
                comp = valid_comp
            else:
                continue

            words_spanned = [f"s{p + 1}:{counted[p]['orth']}" for p in comp]

            chain_obj = {
                "phoneme": c,
                "kind": "serie_consonantique",
                "label": f"[{c}]",
                "positions": list(comp),
                "counted_positions": list(comp),
                "span_desc": " ↔ ".join(words_spanned),
            }
            active_chains.append(chain_obj)
            for p in comp:
                # 語末母音後アトーン女性音節 (e caduc post-tonique) は構造的弱音節のため強勢化させない
                if not counted[p].get("is_post_tonic_fem"):
                    if c not in syl_triggers[p]:
                        syl_triggers[p].append(c)

    # 2. 局所強化されたリエゾン子音 (p. 139 原則非強勢 / p. 182 cousent...des‿enfants のみ局所連鎖で活性化)
    for idx in range(n_c):
        c_liais = counted[idx].get("liaison_c")
        if not c_liais or opening_map[idx]:
            continue
        # 同一行内の4音節以内に同一音素 c_liais が語彙的開子音または有声語末子音として存在する場合のみ連鎖成立
        partner_positions = []
        for j in range(max(0, idx - 4), min(n_c, idx + 5)):
            if j == idx:
                continue
            if c_liais in opening_map[j] or (counted[j]["coda"] and c_liais in counted[j]["coda"]):
                partner_positions.append(j)
        if partner_positions:
            comp = sorted(set(partner_positions + [idx]))
            tag = f"{c_liais}‿"
            active_chains.append({
                "phoneme": c_liais,
                "kind": "liaison_renforcee",
                "label": f"[{c_liais}‿]",
                "positions": comp,
                "counted_positions": comp,
                "span_desc": " ↔ ".join(f"s{p + 1}:{counted[p]['orth']}" for p in comp),
            })
            if tag not in syl_triggers[idx]:
                syl_triggers[idx].append(tag)
            for p in partner_positions:
                if c_liais not in syl_triggers[p] and tag not in syl_triggers[p]:
                    syl_triggers[p].append(c_liais)

    # 3. 子音衝突・連接 (Hiatus de consonnes: 語末閉子音 C # 次語頭開子音 C, pp. 100, 172, 178: roc croulant, colossal laisse)
    for idx in range(n_c - 1):
        s1 = counted[idx]
        s2 = counted[idx + 1]
        if s1["word_idx"] != s2["word_idx"] and s1["coda"] and opening_map[idx + 1]:
            c_end = s1["coda"][-1]
            if c_end == opening_map[idx + 1][0]:
                tag_h = f"{c_end}#{c_end}"
                active_chains.append({
                    "phoneme": c_end,
                    "kind": "hiatus_de_consonnes",
                    "label": f"[{c_end}#{c_end}]",
                    "positions": [idx, idx + 1],
                    "counted_positions": [idx, idx + 1],
                    "span_desc": f"s{idx + 1}:{s1['orth']} # s{idx + 2}:{s2['orth']}",
                })
                if c_end not in syl_triggers[idx] and tag_h not in syl_triggers[idx]:
                    syl_triggers[idx].append(tag_h)
                if c_end not in syl_triggers[idx + 1] and tag_h not in syl_triggers[idx + 1]:
                    syl_triggers[idx + 1].append(c_end)

    # 4. 連内・行跨ぎ反跳連鎖 (Contre-accent / Rebond prosodique enjambant, p. 181: courbes, // Bas)
    if prev_stream is not None and n_c > 0:
        prev_all = prev_stream["syllables"]
        prev_tail = prev_all[-2:] if len(prev_all) >= 2 else prev_all
        prev_tail_cons = set()
        for ps in prev_tail:
            prev_tail_cons.update(_extract_meschonnic_opening_consonants(ps))
        for idx in range(min(2, n_c)):
            for c in opening_map[idx]:
                if c in prev_tail_cons and c not in syl_triggers[idx]:
                    tag_up = f"{c}↩L{prev_line_no}"
                    syl_triggers[idx].append(tag_up)
                    active_chains.append({
                        "phoneme": c,
                        "kind": "reprise_inter_vers",
                        "label": f"[{c}↩L{prev_line_no}]",
                        "positions": [idx],
                        "counted_positions": [idx],
                        "span_desc": f"L{prev_line_no}末 ↔ s{idx + 1}:{counted[idx]['orth']}",
                    })

    if next_stream is not None and n_c > 0:
        next_counted = next_stream["counted_syllables"]
        next_head = next_counted[:2]
        next_head_cons = set()
        for ns in next_head:
            next_head_cons.update(_extract_meschonnic_opening_consonants(ns))
        for idx in range(max(0, n_c - 2), n_all):
            for c in opening_map[idx]:
                if c in next_head_cons and c not in syl_triggers[idx]:
                    tag_dn = f"{c}↪L{next_line_no}"
                    syl_triggers[idx].append(tag_dn)
                    if idx < n_c:
                        active_chains.append({
                            "phoneme": c,
                            "kind": "reprise_inter_vers",
                            "label": f"[{c}↪L{next_line_no}]",
                            "positions": [idx],
                            "counted_positions": [idx],
                            "span_desc": f"s{idx + 1}:{counted[idx]['orth']} ↔ L{next_line_no}頭",
                        })

    # 5. 声門閉鎖アタック [ʔ] (Accent vocalique / Coup de glotte, pp. 142-143: ダッシュ・引用符直後の母音開始語)
    for idx in range(n_c):
        s = counted[idx]
        if not opening_map[idx] and s["is_word_initial"] and not syl_triggers[idx]:
            pb = s.get("punct_before", "") or ""
            prev_pa = counted[idx - 1].get("punct_after", "") if idx > 0 else ""
            has_rupture = any(ch in '—–«"“' for ch in pb) or any(ch in '—–«"“' for ch in prev_pa)
            if has_rupture:
                syl_triggers[idx].append("ʔ")
                active_chains.append({
                    "phoneme": "ʔ",
                    "kind": "coup_de_glotte",
                    "label": "[ʔ]",
                    "positions": [idx],
                    "counted_positions": [idx],
                    "span_desc": f"s{idx + 1}:{s['orth']} (声門閉鎖アタック)",
                })

    # 6. 母音反復系列 (Séries vocaliques sériantes — 非強勢・意味連鎖形成のみ, pp. 138, 173-175)
    vow_positions = collections.defaultdict(list)
    for idx, s in enumerate(counted):
        if not s["is_schwa"]:
            vow_positions[s["nucleus"]].append(idx)
    vocalic_series = []
    for v_ph, v_pos in vow_positions.items():
        if len(v_pos) >= 3 or (len(v_pos) == 2 and v_pos[1] - v_pos[0] <= 4 and v_ph in VOWELS_NASAL):
            vocalic_series.append({
                "vowel": v_ph,
                "count": len(v_pos),
                "positions": [p + 1 for p in v_pos],
                "words": [counted[p]["orth"] for p in v_pos],
            })

    return opening_map, syl_triggers, active_chains, vocalic_series


def _classify_meschonnic_word_accent(idx, s, v_words, counted, corn):
    """Dessons & Meschonnic (1998, pp. 122-137, 167, 182) に基づき、語末男性音節のグループ強勢種別を判定する。
    戻り値: 'R' (グループ強勢 —), 'D' (識別的・統辞両義二重記号 ⏓), または None (グループ内非強勢)。
    """
    if not s["is_word_final_masc"]:
        return None

    n_c = len(counted)
    w_i = s["word_idx"]
    tok = v_words[w_i]
    prev_tok = v_words[w_i - 1] if w_i > 0 else None
    nxt_tok = v_words[w_i + 1] if w_i + 1 < len(v_words) else None
    nxt2_tok = v_words[w_i + 2] if w_i + 2 < len(v_words) else None

    w_low = normalize_french_orthography(tok.text.lower())
    w_host = w_low.split("'")[-1] if ("'" in w_low and not w_low.endswith("'")) else w_low
    pos = tok.pos_

    has_punct_after = bool(s["punct_after"] and any(p in ".,;:!?…—–»)" for p in s["punct_after"]))
    at_caesura = bool(corn["caesura_pos"] and idx + 1 == corn["caesura_pos"])
    is_last_masc = idx == n_c - 1

    # 1. 直後に句読点がある語は品詞を問わずリズム群終端 (pp. 123, 124: "Tant de —", "Tu, Rrose Sélavy")
    if has_punct_after:
        return "R"

    # 2. 間投詞・呼格 Ô / Ah / Hélas (pp. 165, 177: "Ô soldats", "Ô pieds plats", "ô guerres")
    if pos == "INTJ" or w_low in {"ô", "ah", "eh", "oh", "hélas"}:
        if w_low == "fi" and nxt_tok is not None and normalize_french_orthography(nxt_tok.text.lower()) == "donc":
            return None
        return "R"

    # 3. 前接クリティック機能語 (限定詞・単音節/二音節前置詞・弱接続詞・動詞前代名詞, pp. 122-123, 136-137)
    is_proclitic = (
        s["cpmf"] in ("C", "CF", "P")
        or w_low in WEAK_CONJ_REL
        or w_host in WEAK_CONJ_REL
        or w_host in MESCH_DISYLLABIC_PREPS
        or (w_host in {"nous", "vous"} and nxt_tok is not None and nxt_tok.pos_ in ("VERB", "AUX") and "-" not in tok.text)
    )
    if is_proclitic and nxt_tok is not None:
        return None
    if is_last_masc and is_proclitic:
        # 句読点なしの行末前接語はメートル固定位置 'M' へ委譲
        return None

    # 4. 助動詞・コピュラ être・使役/準助動詞・法助動詞 (pp. 135-136, 182)
    if nxt_tok is not None and not at_caesura:
        nxt_low = normalize_french_orthography(nxt_tok.text.lower())
        nxt_host = nxt_low.split("'")[-1]
        # 時制/受動助動詞・コピュラ être -> 非強勢
        if (pos == "AUX" or w_host in MESCH_TEMPORAL_COPULA_AUX) and nxt_tok.pos_ in ("VERB", "AUX", "ADJ", "NOUN", "ADV", "DET", "PRON", "NUM", "ADP"):
            # ただし存在動詞的絶対用法 (セジュールや句末は上で処理済) 以外は非強勢
            return None
        # 使役 faire / 近接未来 aller + 不定詞 (例: fait trembler, p. 182) -> 非強勢
        if w_host in MESCH_CAUSATIVE_LIGHT_VERBS and nxt_tok.pos_ == "VERB":
            return None
        # 法助動詞 (pouvoir, devoir, vouloir, savoir, falloir) + 不定詞 (p. 136: "je peux partir ; il faut vendre") -> 二重記号 ⏓
        if w_host in MESCH_MODAL_VERBS and nxt_tok.pos_ == "VERB":
            return "D"

    # 5. 前置修飾副詞 (pp. 128, 134-135: "très fort", "pas bien", "immensément riche", "n'a pas crié", "purement passé")
    if (pos == "ADV" or w_host in MESCH_PRE_MODIFIER_ADVS) and nxt_tok is not None and not at_caesura:
        if (w_host in MESCH_PRE_MODIFIER_ADVS or w_host.endswith("ment")) and nxt_tok.pos_ in ("ADJ", "ADV", "VERB"):
            return None

    # 6. 動詞＋後置否定辞/短副詞による強勢移動 (p. 122 n.1 "je ne le vois pas", p. 134 "je mange bien ; tu n'écoutes pas")
    if pos == "VERB" and nxt_tok is not None and not at_caesura:
        nxt_low = normalize_french_orthography(nxt_tok.text.lower())
        if nxt_low in MESCH_POST_VERB_BOUND_ADVS:
            nxt_modifies_following = (
                nxt2_tok is not None
                and not bool(nxt_tok.after_punct)
                and nxt2_tok.pos_ in ("VERB", "ADJ", "ADV")
                and nxt_low in MESCH_PRE_MODIFIER_ADVS
            )
            if not nxt_modifies_following:
                return None

    # 7. 後接代名詞つき動詞＋単音節語 (p. 136: "Que voudras-tu faire ? Jette-les donc !")
    if "-" in w_low and nxt_tok is not None and not at_caesura:
        hyph_tail = w_low.split("-")[-1]
        nxt_low = normalize_french_orthography(nxt_tok.text.lower())
        if hyph_tail in (ENCLITICS_TONIC_E1 | ENCLITICS_CONDITIONAL_E3) and nxt_low in {"donc", "faire", "dire", "voir", "pas", "plus", "point", "bien", "tout", "rien"}:
            return None

    # 8. 名詞前の基数詞・前置形容詞 (pp. 130-134, 170, 182)
    if nxt_tok is not None and not at_caesura:
        # 基数詞 + 名詞 (Dix nuits, sept ans, deux yeux) -> 限定詞として非強勢
        if (pos == "NUM" or w_host in CARDINAL_NUMERALS) and nxt_tok.pos_ in ("NOUN", "PROPN", "ADJ"):
            return None
        # 前置形容詞 + 名詞 (または 連続前置形容詞 beaux grands yeux)
        if pos == "ADJ" and nxt_tok.pos_ in ("NOUN", "PROPN", "ADJ"):
            if w_host in MESCH_DETERMINATIVE_PRENOMINAL_ADJS:
                return None
            return "D"

    # 9. 名詞＋後置形容詞 (pp. 132-134, 167, 172, 177, 182)
    if pos in ("NOUN", "PROPN") and nxt_tok is not None and nxt_tok.pos_ == "ADJ" and not at_caesura:
        nxt_low = normalize_french_orthography(nxt_tok.text.lower())
        # (a) NOUN + ADJ1 + et/ou + ADJ2 (p. 167: "sur le roc noir et nu", "un cachot vaste et noir") -> 統辞両義二重記号 ⏓
        if (
            nxt2_tok is not None
            and normalize_french_orthography(nxt2_tok.text.lower()) in {"et", "ou", "ni"}
        ):
            return "D"
        # (b) ADJ1 + NOUN + ADJ2 または NOUN + même (p. 99 "mon nom même", p. 182 "leurs pauvres doigts gris") -> 二重記号 ⏓
        if (prev_tok is not None and prev_tok.pos_ == "ADJ") or nxt_low in {"même", "mêmes"}:
            return "D"
        # (c) 不定冠詞 un/une + NOUN + 多音節評価形容詞 (p. 133-134 "un film remarquable") -> 二重記号 ⏓
        if prev_tok is not None and normalize_french_orthography(prev_tok.text.lower()) in {"un", "une"} and len(nxt_low) >= 6:
            return "D"
        # (d) 通常の単一後置形容詞結合 (p. 177 "pieds plats", p. 172 "l'eau morte", "roc croulant", p. 133 "poisson rouge") -> 1リズム群に統合 (NOUNは非強勢で ⏑ˈ 受容可能)
        return None

    # 10. 不定・特徴づけ NOUN + de + NUM/ADJ + NOUN (p. 182 "des enfants de sept ans", p. 134 "une idée du tonnerre")
    if pos in ("NOUN", "PROPN") and nxt_tok is not None and not at_caesura:
        nxt_low = normalize_french_orthography(nxt_tok.text.lower())
        if nxt_low in {"de", "d'", "du"} and nxt2_tok is not None:
            nxt2_low = normalize_french_orthography(nxt2_tok.text.lower())
            if prev_tok is not None and normalize_french_orthography(prev_tok.text.lower()) in {"un", "une", "des"} and (nxt2_low in CARDINAL_NUMERALS or nxt2_tok.pos_ == "NUM"):
                return "D"

    return "R"


def analyze_meschonnic_corpus(lines_meta, docs, unit_words, line_streams, cornulier_lines):
    """Meschonnic & Dessons (1998)『Traité du rythme』に完全準拠し、音素連鎖条件を満たすプロゾディー強勢 (⏑ˈ)、
    下部音素連鎖ブラケット (notation souscrite)、グループ強勢 (— / ⏓)、反アクセント8類型、7種の反響図式を解析する。
    """
    n_lines = len(lines_meta)
    if n_lines == 0:
        return {
            "lines": [],
            "contre_accents_by_type": {},
            "total_contre_accents": 0,
            "plateaux_rythmiques": [],
            "echo_figures": [],
            "echo_counts": {},
            "superlatifs_prosodiques": [],
            "dominant_consonant_series": [],
            "dominant_vowel_series": [],
            "finale_profile": {},
            "volume_contours": {},
            "ponctuation_metrics": {},
        }

    # テクスト全体の開子音・母音系列頻度（プロゾディー主系列の特定）
    all_opening_consonants = []
    all_vowels = []
    for stream in line_streams:
        for s in stream["counted_syllables"]:
            all_opening_consonants.extend(_extract_meschonnic_opening_consonants(s))
            if not s["is_schwa"]:
                all_vowels.append(s["nucleus"])

    cons_total_counter = collections.Counter(all_opening_consonants)
    vow_total_counter = collections.Counter(all_vowels)
    top_text_consonants = {c for c, _ in cons_total_counter.most_common(5)}

    mesch_lines = []
    contre_type_counter = collections.Counter()
    plateaux_list = []
    echo_figures = []
    superlatifs = []
    finale_counter = collections.Counter()
    group_volumes_by_clause = []
    current_clause_groups = []

    punct_lines_count = 0
    dash_relance_count = 0
    lowercase_internal_starts = 0
    internal_lines_count = 0
    line_syntax_conflicts = 0

    for i in range(n_lines):
        raw_text = lines_meta[i]["text"]
        ws = unit_words[i]
        stream = line_streams[i]
        corn = cornulier_lines[i]
        sylls = stream["syllables"]
        counted = stream["counted_syllables"]
        n_c = len(counted)
        ext_syl = sylls[-1] if stream["has_extrametrical_final_e"] else None

        if any(p in raw_text for p in ".,;:!?…—–«»"):
            punct_lines_count += 1
        if "—" in raw_text or "–" in raw_text:
            dash_relance_count += 1
        if not lines_meta[i]["stanza_start"]:
            internal_lines_count += 1
            first_alpha = next((ch for ch in raw_text if ch.isalpha()), "")
            if first_alpha and first_alpha.islower():
                lowercase_internal_starts += 1

        if n_c == 0:
            mesch_lines.append({
                "line": i + 1,
                "text": raw_text,
                "scansion_unicode": "",
                "syllable_tokens": [],
                "contre_accents": [],
                "group_volumes": [],
                "echo_chains": [],
                "active_chains": [],
                "vocalic_series": [],
            })
            continue

        # 同一連内の前後詩行ストリーム（行跨ぎ反跳の厳密検出用）
        prev_st = line_streams[i - 1] if (i > 0 and lines_meta[i - 1]["stanza"] == lines_meta[i]["stanza"]) else None
        next_st = line_streams[i + 1] if (i + 1 < n_lines and lines_meta[i + 1]["stanza"] == lines_meta[i]["stanza"]) else None

        # 1. 音素連鎖条件の厳密検証（開子音連鎖・局所強化リエゾン・子音衝突・行跨ぎ反跳・声門閉鎖 [ʔ]）
        opening_map, syl_triggers, active_chains, vocalic_series = _detect_meschonnic_line_chains(
            counted,
            ext_syl,
            prev_stream=prev_st,
            next_stream=next_st,
            prev_line_no=i if prev_st is not None else None,
            next_line_no=(i + 2) if next_st is not None else None,
        )

        # 2. 階層原則に基づく各音節の記号決定 (Traité du rythme p. 144, p. 150, p. 178):
        # 第1優先: グループ強勢 'R' (—) / 識別的・両義二重記号 'D' (⏓)
        # 第2優先: 音素連鎖条件を満たすプロゾディー強勢 'P' (⏑ˈ)  ※第6位置の前接語でも連鎖があれば ⏑ˈ (p. 178 avec vos fleurs)
        # 第3優先: 連鎖を持たない定型境界音節のメートル強勢 'M' (—) (p. 155 sa | plainte, passe à | travers)
        # 第4優先: 無強勢 'U' (◡)
        is_metric_alex = corn["anatonic_length"] == 12
        is_metric_deca = corn["anatonic_length"] == 10 and corn["caesura_pos"] in (4, 5)

        marks = []
        accent_origins = []  # R (includes D for 8-type contre-accent algebra), M, P, U
        mark_subtypes = []   # R, D, M, P, U
        group_end_indices = []

        for idx, s in enumerate(counted):
            pos_1 = idx + 1
            grp_kind = _classify_meschonnic_word_accent(idx, s, ws, counted, corn)
            has_prosodic_chain = len(syl_triggers[idx]) > 0

            is_fixed_metric_pos = (
                (is_metric_alex and pos_1 in (6, 12))
                or (is_metric_deca and pos_1 in (corn["caesura_pos"], 10))
            )

            if grp_kind == "R":
                marks.append("—")
                accent_origins.append("R")
                mark_subtypes.append("R")
                group_end_indices.append(idx)
            elif grp_kind == "D":
                marks.append("⏓")
                accent_origins.append("R")
                mark_subtypes.append("D")
            elif has_prosodic_chain and not s.get("is_post_tonic_fem"):
                marks.append("⏑ˈ")
                accent_origins.append("P")
                mark_subtypes.append("P")
            elif is_fixed_metric_pos and not s.get("is_post_tonic_fem"):
                marks.append("—")
                accent_origins.append("M")
                mark_subtypes.append("M")
            else:
                marks.append("◡")
                accent_origins.append("U")
                mark_subtypes.append("U")

        # 行末が前接語・機能語で終わる場合や句読点なしで緊密結合する場合は 行＝統辞衝突をカウント
        if ws and (
            ws[-1].pos_ in ("DET", "ADP", "CCONJ", "SCONJ", "AUX")
            or (
                not raw_text.strip().endswith((".", ",", ";", ":", "!", "?", "…", "»", "—", "–"))
                and i + 1 < n_lines
                and lines_meta[i + 1]["stanza"] == lines_meta[i]["stanza"]
            )
        ):
            line_syntax_conflicts += 1

        # 3. 反アクセント (Contre-accents: 隣接する有標音節の累積番号 1, 2, ..., L と休止跨ぎリガチャー ⁀, pp. 152-155, 165)
        # 反アクセントは異なる語間 (trans-lexical) の強勢衝突であり、同一語内音節同士の衝突は計上しない
        contre_nums = [None] * n_c
        ligature_after = [False] * n_c
        line_contre_events = []

        run_start = None
        for idx in range(n_c + 1):
            is_marked = (idx < n_c) and (accent_origins[idx] in ("R", "M", "P"))
            if is_marked:
                if run_start is None:
                    run_start = idx
            else:
                if run_start is not None:
                    run_len = idx - run_start
                    if run_len >= 2:
                        for offset in range(run_len):
                            pos_k = run_start + offset
                            contre_nums[pos_k] = offset + 1
                            if offset < run_len - 1:
                                if counted[pos_k]["word_idx"] == counted[pos_k + 1]["word_idx"]:
                                    continue
                                o1 = accent_origins[pos_k]
                                o2 = accent_origins[pos_k + 1]
                                pair_code = f"{o1}-{o2}"
                                if pair_code == "M-M":
                                    pair_code = "R-M"
                                contre_type_counter[pair_code] += 1
                                has_pause_between = bool(
                                    counted[pos_k]["punct_after"]
                                    or (corn["caesura_pos"] and pos_k + 1 == corn["caesura_pos"])
                                )
                                if has_pause_between:
                                    ligature_after[pos_k] = True
                                line_contre_events.append({
                                    "line": i + 1,
                                    "type": pair_code,
                                    "ligature": has_pause_between,
                                    "span": f"{counted[pos_k]['word_text']} / {counted[pos_k + 1]['word_text']}",
                                })
                        if run_len >= 3:
                            plateaux_list.append({
                                "line": i + 1,
                                "length": run_len,
                                "words": " ".join(counted[k]["orth"] for k in range(run_start, idx)),
                            })
                    run_start = None

        # 4. リズム群ごとの末音分類 ([V]# 終止母音 vs [C]# 宙吊り子音 vs [Cː]# 長化摩擦子音) と音節量輪郭
        group_volumes = []
        prev_g_end = -1
        for g_end in group_end_indices:
            actual_end = g_end
            if g_end + 1 < n_c and counted[g_end + 1]["word_idx"] == counted[g_end]["word_idx"] and counted[g_end + 1]["is_post_tonic_fem"]:
                actual_end = g_end + 1
            vol = actual_end - prev_g_end
            group_volumes.append(vol)
            prev_g_end = actual_end

            end_s = counted[g_end]
            coda_cs = end_s["coda"]
            if g_end == n_c - 1 and ext_syl is not None and ext_syl.get("onset"):
                coda_cs = ext_syl["onset"]

            if not coda_cs:
                finale_type = "finale_vocalique"
            elif (len(coda_cs) == 1 and coda_cs[0] in LENGTHENING_CONSONANTS) or coda_cs[-2:] == ["v", "ʁ"]:
                finale_type = "consonne_allongeante"
            else:
                finale_type = "finale_consonantique"
            finale_counter[finale_type] += 1

            current_clause_groups.append(vol)
            if end_s["punct_after"] and any(p in ".!?;:" for p in end_s["punct_after"]):
                if len(current_clause_groups) >= 2:
                    group_volumes_by_clause.append(list(current_clause_groups))
                current_clause_groups = []

        # 5. 7種のプロゾディー反響図式 (Figures d'écho, pp. 170-172) & プロゾディー最上級語の検出
        active_line_cons = {ch["phoneme"] for ch in active_chains if ch["phoneme"] != "ʔ"} | top_text_consonants
        line_echoes = _detect_line_echo_figures(ws, counted, opening_map[:n_c], i + 1, active_line_cons, superlatifs)
        echo_figures.extend(line_echoes)

        # 6. Meschonnic & Dessons 音素連鎖根拠つき公式走査文字列・トークン配列の構築
        syl_tokens = []
        unicode_parts = []
        for idx, s in enumerate(counted):
            c_num = contre_nums[idx]
            sup_str = _to_superscript(c_num) if c_num is not None else ""
            lig_str = "⁀" if ligature_after[idx] else ""
            mark = marks[idx]
            triggers = syl_triggers[idx]
            # 音素連鎖根拠タグ: ⏑ˈ の場合は必須表示、— / ⏓ でも連鎖アンカーであれば [c] を添えて連鎖の対称性を明示
            chain_tag_str = f"[{','.join(triggers)}]" if triggers else ""
            caesura_bar = " │ " if (corn["caesura_pos"] and idx + 1 == corn["caesura_pos"]) else ""

            syl_tokens.append({
                "syl_idx": idx + 1,
                "orth": s["orth"],
                "word_text": s["word_text"],
                "word_idx": s.get("word_idx"),
                "word_pos": ws[s["word_idx"]].pos_ if s.get("word_idx") is not None and s["word_idx"] < len(ws) else "",
                "is_word_start": idx == 0 or counted[idx - 1].get("word_idx") != s.get("word_idx"),
                "ipa": s["ipa"],
                "mark": mark,
                "origin": accent_origins[idx],
                "subtype": mark_subtypes[idx],
                "contre_num": c_num,
                "ligature_after": ligature_after[idx],
                "caesura_after": bool(caesura_bar),
                "is_word_end": s["is_word_end"],
                "is_group_end": idx in group_end_indices,
                "punct_after": s["punct_after"],
                "onset_cons": opening_map[idx],
                "chain_triggers": triggers,
                "nucleus": s["nucleus"],
                "coda_cons": list(s["coda"]),
            })
            sep = " " if s["is_word_end"] else "·"
            unicode_parts.append(f"{sup_str}{mark}{chain_tag_str}{lig_str}({s['orth']}){caesura_bar}{sep}")

        # 行末余剰女性音節 (surnuméraire):
        # 次行頭への反跳連鎖 (reprise enjambante, Traité du rythme p. 181: courbes // Bas) がある場合のみ (⏑ˈ)[c]、
        # 反跳のない通常の女性行末では、無強勢の余韻 (e) としてアクセント記号を付与せず、語末の位取りズレを防ぐ。
        if ext_syl is not None:
            ext_triggers = syl_triggers[n_c]
            has_rebound = bool(ext_triggers)
            ext_chain_tag = f"[{','.join(ext_triggers)}]" if ext_triggers else ""
            ext_mark = "(⏑ˈ)" if has_rebound else ""
            ext_origin = "P_ext" if has_rebound else "ext"
            ext_subtype = "P_ext" if has_rebound else "ext"
            ext_orth = ext_syl.get("orth", "(e)")

            syl_tokens.append({
                "syl_idx": None,
                "orth": ext_orth,
                "word_text": ext_syl["word_text"],
                "word_idx": ext_syl.get("word_idx"),
                "word_pos": ws[ext_syl["word_idx"]].pos_ if ext_syl.get("word_idx") is not None and ext_syl["word_idx"] < len(ws) else "",
                "is_word_start": False,
                "ipa": "".join(ext_syl.get("onset", [])) + "ə",
                "mark": ext_mark,
                "origin": ext_origin,
                "subtype": ext_subtype,
                "contre_num": None,
                "ligature_after": False,
                "caesura_after": False,
                "is_word_end": True,
                "is_group_end": False,
                "punct_after": ext_syl["punct_after"],
                "onset_cons": opening_map[n_c],
                "chain_triggers": ext_triggers,
                "nucleus": "ə",
                "coda_cons": [],
            })
            if has_rebound:
                unicode_parts.append(f"↪{ext_mark}{ext_chain_tag}({ext_orth})")
            else:
                unicode_parts.append(f"({ext_orth})")

        mesch_lines.append({
            "line": i + 1,
            "text": raw_text,
            "scansion_unicode": "".join(unicode_parts).strip(),
            "syllable_tokens": syl_tokens,
            "contre_accents": line_contre_events,
            "group_volumes": group_volumes,
            "echo_chains": line_echoes,
            "active_chains": active_chains,
            "vocalic_series": vocalic_series,
        })

    if len(current_clause_groups) >= 2:
        group_volumes_by_clause.append(list(current_clause_groups))

    # 節ごとのグループ音節量輪郭 (Volume des groupes) の分類
    contour_counter = collections.Counter()
    for vols in group_volumes_by_clause:
        if all(vols[k] <= vols[k + 1] for k in range(len(vols) - 1)) and vols[0] < vols[-1]:
            contour_counter["漸増・拡大輪郭 (progressif / croissant)"] += 1
        elif all(vols[k] >= vols[k + 1] for k in range(len(vols) - 1)) and vols[0] > vols[-1]:
            contour_counter["漸減・収束輪郭 (régressif / descendant)"] += 1
        elif vols == list(reversed(vols)):
            contour_counter["対称・均衡輪郭 (symétrique)"] += 1
        else:
            contour_counter["断続・非周期輪郭 (rompu / apériodique)"] += 1

    all_g_vols = [v for l in mesch_lines for v in l["group_volumes"]]
    short_group_pct = round(100.0 * sum(1 for v in all_g_vols if v <= 2) / max(1, len(all_g_vols)), 1)

    tot_f = sum(finale_counter.values())
    finale_profile = {
        "finale_vocalique_pct": round(100.0 * finale_counter["finale_vocalique"] / max(1, tot_f), 1),
        "finale_consonantique_pct": round(100.0 * finale_counter["finale_consonantique"] / max(1, tot_f), 1),
        "consonne_allongeante_pct": round(100.0 * finale_counter["consonne_allongeante"] / max(1, tot_f), 1),
        "counts": dict(finale_counter),
    }

    echo_counts = collections.Counter(e["figure"] for e in echo_figures)

    # 重複除外したプロゾディー最上級語リスト
    seen_sup = set()
    unique_superlatifs = []
    for sp_item in superlatifs:
        key = (sp_item["word"].lower(), sp_item["consonant"])
        if key not in seen_sup:
            seen_sup.add(key)
            unique_superlatifs.append(sp_item)

    ponctuation_metrics = {
        "punctuated_lines_pct": round(100.0 * punct_lines_count / max(1, n_lines), 1),
        "ponctuation_blanche_pct": round(100.0 * (n_lines - punct_lines_count) / max(1, n_lines), 1),
        "dash_relance_lines": dash_relance_count,
        "lowercase_continuity_pct": round(100.0 * lowercase_internal_starts / max(1, internal_lines_count), 1),
        "conflit_ligne_syntaxe_pct": round(100.0 * line_syntax_conflicts / max(1, n_lines), 1),
        "short_groups_1_2_pct": short_group_pct,
    }

    return {
        "lines": mesch_lines,
        "contre_accents_by_type": dict(contre_type_counter),
        "total_contre_accents": sum(contre_type_counter.values()),
        "plateaux_rythmiques": plateaux_list,
        "echo_figures": echo_figures,
        "echo_counts": dict(echo_counts),
        "superlatifs_prosodiques": unique_superlatifs[:25],
        "dominant_consonant_series": [{"consonant": c, "count": cnt} for c, cnt in cons_total_counter.most_common(6)],
        "dominant_vowel_series": [{"vowel": v, "count": cnt} for v, cnt in vow_total_counter.most_common(6)],
        "finale_profile": finale_profile,
        "volume_contours": dict(contour_counter),
        "ponctuation_metrics": ponctuation_metrics,
    }


def _to_superscript(num):
    sup_map = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    return str(num).translate(sup_map)


def _detect_line_echo_figures(ws, counted, opening_map, line_num, active_line_cons, superlatifs_out):
    """Meschonnic & Dessons『Traité du rythme』(pp. 170-172) の7種のプロゾディー反響図式を行内から厳密検出する。"""
    figures = []
    content_words = []

    for w_i, tok in enumerate(ws):
        if len(tok.text) < 2 and tok.pos_ in ("DET", "ADP", "PRON", "CCONJ"):
            continue
        w_syll_pairs = [(idx, counted[idx]) for idx in range(len(counted)) if counted[idx]["word_idx"] == w_i]
        if not w_syll_pairs:
            continue
        w_sylls = [p[1] for p in w_syll_pairs]
        ipa_all = "".join(s["ipa"] for s in w_sylls)
        phs = extract_phonemes(ipa_all)
        cons_list = [p for p in phs if p in CONSONANTS]

        # 7. Superlatif prosodique: 活性系列子音を1語内に2回以上凝縮する実詞 (例: murmure [m,ʁ,m,ʁ], mémoire [m,m,ʁ], supplice [s,p,l,s])
        if tok.pos_ in ("NOUN", "VERB", "ADJ", "ADV") and len(cons_list) >= 2:
            c_cnt = collections.Counter(cons_list)
            for c_ph, freq in c_cnt.items():
                if freq >= 2 and c_ph in active_line_cons:
                    superlatifs_out.append({
                        "line": line_num,
                        "word": tok.text,
                        "ipa": ipa_all,
                        "consonant": c_ph,
                        "count_in_word": freq,
                    })
                    figures.append({
                        "line": line_num,
                        "figure": "superlatif_prosodique",
                        "label": f"プロゾディー最上級語 ([{c_ph}]×{freq})",
                        "example": f"{tok.text} [{ipa_all}]",
                    })
                    break

        content_words.append({
            "text": tok.text,
            "pos": tok.pos_,
            "sylls": w_sylls,
            "first_syl_idx": w_syll_pairs[0][0],
            "phs": phs,
            "cons": cons_list,
        })

    # 5. Rebond de consonnes (pp. 171-172: "lampe paisible", "flammes mortes", "laisse, sur" — 語内および語跨ぎの無強勢/ə音節を介した同一開子音反跳)
    for idx in range(len(counted) - 1):
        c1 = opening_map[idx]
        c2 = opening_map[idx + 1]
        if c1 and c2:
            shared = [c for c in c1 if c in c2]
            if shared and (counted[idx]["is_schwa"] or counted[idx + 1]["is_schwa"]):
                c_reb = shared[0]
                span_txt = (
                    counted[idx]["word_text"]
                    if counted[idx]["word_idx"] == counted[idx + 1]["word_idx"]
                    else f"{counted[idx]['word_text']} {counted[idx + 1]['word_text']}"
                )
                figures.append({
                    "line": line_num,
                    "figure": "rebond_de_consonnes",
                    "label": f"子音反跳 (Rebond: [{c_reb}ə{c_reb}])",
                    "example": span_txt,
                })

    # 6. Hiatus de consonnes (pp. 100, 172, 178: 語末閉子音 C # 次語頭開子音 C — "roc croulant", "colossal laisse")
    for idx in range(len(counted) - 1):
        s1, s2 = counted[idx], counted[idx + 1]
        if s1["word_idx"] != s2["word_idx"] and s1["coda"] and opening_map[idx + 1]:
            if s1["coda"][-1] == opening_map[idx + 1][0]:
                c_h = s1["coda"][-1]
                figures.append({
                    "line": line_num,
                    "figure": "hiatus_de_consonnes",
                    "label": f"子音衝突・連接 (Hiatus: [{c_h}#{c_h}])",
                    "example": f"{s1['word_text']} {s2['word_text']}",
                })

    # ペア比較による 1〜4 の反響図式検出 (pp. 170-171)
    for a_i in range(len(content_words)):
        for b_i in range(a_i + 1, min(len(content_words), a_i + 5)):
            wa = content_words[a_i]
            wb = content_words[b_i]
            if wa["text"].lower() == wb["text"].lower():
                continue
            sa0 = wa["sylls"][0]
            sb0 = wb["sylls"][0]
            oa = opening_map[wa["first_syl_idx"]]
            ob = opening_map[wb["first_syl_idx"]]

            # 1. Écho initial (p. 170 "profil / propylées": 語頭 C(C)V == C(C)V 完全一致)
            if oa and oa == ob and sa0["nucleus"] == sb0["nucleus"]:
                figures.append({
                    "line": line_num,
                    "figure": "echo_initial",
                    "label": f"語頭CV反響 (Écho initial: [{''.join(oa)}{sa0['nucleus']}])",
                    "example": f"{wa['text']} ↔ {wb['text']}",
                })
            # 2. Écho initial diminué (p. 170 "graine / gré": C1C2 -> C1 または C1C2VC3 -> C1C2V 縮減)
            elif (len(oa) >= 2 and len(ob) == 1 and ob[0] == oa[0]) or (
                len(oa) >= 2 and oa == ob and sa0["coda"] and not sb0["coda"]
            ):
                figures.append({
                    "line": line_num,
                    "figure": "echo_diminue",
                    "label": f"語頭縮減反響 (Écho diminué: [{''.join(oa)}] → [{''.join(ob)}])",
                    "example": f"{wa['text']} → {wb['text']}",
                })
            # 3. Écho augmenté (pp. 170-171 "fréquents / fer", "tout / tour", "éléphants / fentes": C1->C1C2, 二重子音の母音展開, 末子音付加)
            elif (
                (len(oa) == 1 and len(ob) >= 2 and oa[0] == ob[0])
                or (len(oa) == 2 and len(ob) == 1 and oa[0] == ob[0] and sb0["coda"] and sb0["coda"][0] == oa[1])
                or (oa and oa == ob and sa0["nucleus"] == sb0["nucleus"] and not sa0["coda"] and sb0["coda"])
            ):
                figures.append({
                    "line": line_num,
                    "figure": "echo_augmente",
                    "label": f"拡張反響 (Écho augmenté: [{sa0['ipa']}] → [{sb0['ipa']}])",
                    "example": f"{wa['text']} → {wb['text']}",
                })
            # 4. Écho renversé / Chiasme prosodique (p. 171 "beaux [bo] ↔ aube [ob]" CV↔VC 反転、または C1…C2 ↔ C2…C1 子音反転)
            elif (
                len(wa["phs"]) >= 2
                and len(wb["phs"]) >= 2
                and wa["phs"][0] in CONSONANTS
                and wa["phs"][1] in VOWELS_ALL
                and wb["phs"][0] == wa["phs"][1]
                and wb["phs"][1] == wa["phs"][0]
            ):
                figures.append({
                    "line": line_num,
                    "figure": "chiasme_prosodique",
                    "label": f"逆転反響 (Écho renversé: [{wa['phs'][0]}{wa['phs'][1]}] ↔ [{wb['phs'][0]}{wb['phs'][1]}])",
                    "example": f"{wa['text']} ↔ {wb['text']}",
                })
            elif len(wa["cons"]) >= 2 and len(wb["cons"]) >= 2:
                ca2 = wa["cons"][:2]
                cb2 = wb["cons"][:2]
                if ca2[0] != ca2[1] and ca2[0] == cb2[1] and ca2[1] == cb2[0]:
                    figures.append({
                        "line": line_num,
                        "figure": "chiasme_prosodique",
                        "label": f"子音キアスム反転 ([{ca2[0]}…{ca2[1]}] ↔ [{cb2[0]}…{cb2[1]}])",
                        "example": f"{wa['text']} ↔ {wb['text']}",
                    })

    return figures


def _is_valid_regular_alexandrine(corn_row):
    """12音節行が古典または脱古典アレクサンドラン（6+6, S6, C6/P6, F6, L6, 4=4=4, 8=4, 4=8）として構造的に成立するか判定する。"""
    if corn_row["anatonic_length"] != 12:
        return False
    if corn_row["caesura_code"].startswith(("6+6", "S6", "C6", "P6", "F6", "FC6", "L6")):
        return True
    if corn_row["ternary_profile"] != "なし":
        return True
    return False


def _resolve_line_stream_with_metrical_pressure(ws, doc, lp, dierese, target_meter=12, ancient_postvocalic_e=False):
    """Cornulier のメトリック圧力原理に基づき、定型詩行における語源的分音 (niais, million, passion 等) および
    揺れ語彙 (anciens, hier, miasmes, alcool 等)、母音後 -e の音節価を最適解決する。
    """
    stream_poetic = build_line_syllable_stream(ws, doc, lp, dierese=True, ancient_postvocalic_e=ancient_postvocalic_e)
    corn_poetic = analyze_cornulier_line(stream_poetic)

    if dierese:
        chosen_stream = stream_poetic
        chosen_corn = corn_poetic
    else:
        stream_colloq = build_line_syllable_stream(ws, doc, lp, dierese=False, ancient_postvocalic_e=ancient_postvocalic_e)
        corn_colloq = analyze_cornulier_line(stream_colloq)
        # 口語合音では非定型 (例: 11v) だが古典詩法分音を適用すると厳密な定型詩行 (12v 6+6 やコーパス主軸メートル) が成立する場合は定型分音を採用
        if corn_colloq["anatonic_length"] != target_meter and corn_poetic["anatonic_length"] == target_meter:
            if target_meter != 12 or _is_valid_regular_alexandrine(corn_poetic):
                chosen_stream = stream_poetic
                chosen_corn = corn_poetic
            else:
                chosen_stream = stream_colloq
                chosen_corn = corn_colloq
        else:
            chosen_stream = stream_colloq
            chosen_corn = corn_colloq

    # 母音後 -e (vie, joie, épée, bleues 等) の古代音節価による定型解決 (9v -> 10v 4+6, 11v -> 12v 6+6 等)
    if not ancient_postvocalic_e and chosen_corn["anatonic_length"] == (target_meter - 1):
        v_words = chosen_stream["verse_words"]
        has_postvoc_inside = any(
            _is_postvocalic_feminine(vw.text, vw.pos_) for vw in v_words[:-1]
        )
        if has_postvoc_inside:
            base_dierese = True if chosen_stream is stream_poetic else dierese
            cand_stream = build_line_syllable_stream(
                ws, doc, lp, dierese=base_dierese, ancient_postvocalic_e=True
            )
            cand_corn = analyze_cornulier_line(cand_stream)
            if cand_corn["anatonic_length"] == target_meter:
                if target_meter == 10 and cand_corn["caesura_code"].startswith(("4+6", "4/6", "4//6")):
                    return cand_stream
                elif target_meter == 12 and _is_valid_regular_alexandrine(cand_corn):
                    return cand_stream
                elif target_meter == 8:
                    return cand_stream

    # 揺れ語彙 (anciens, hier, miasmes, alcool 等) によって 13v (target+1) または 11v (target-1) になっている場合の文脈解決
    if chosen_corn["anatonic_length"] in (target_meter + 1, target_meter - 1):
        v_words = chosen_stream["verse_words"]
        want_dierese = chosen_corn["anatonic_length"] == (target_meter - 1)
        for w_i, vw in enumerate(v_words):
            w_stem = normalize_french_orthography(vw.text.lower()).split("'")[-1]
            if w_stem in FLUCTUATING_DIERESE_WORDS:
                base_mode = True if chosen_stream is stream_poetic else dierese
                overrides = {idx: base_mode for idx in range(len(v_words))}
                overrides[w_i] = want_dierese
                cand_stream = build_line_syllable_stream(
                    ws, doc, lp, dierese=base_mode, word_dierese_overrides=overrides, ancient_postvocalic_e=ancient_postvocalic_e
                )
                cand_corn = analyze_cornulier_line(cand_stream)
                if cand_corn["anatonic_length"] == target_meter and (target_meter != 12 or _is_valid_regular_alexandrine(cand_corn)):
                    return cand_stream
    return chosen_stream


# =============================================================================
# Top-Level Orchestrator (4理論統合エントリポイント)
# =============================================================================

def analyze_prosody_and_rhythm(lines_meta, docs, unit_words, line_phonetics_list, dierese=False, ancient_postvocalic_e=False):
    """Cornulier, Meschonnic & Dessons, Di Cristo, Gouvard/Mazaleyrat/Plénat の4大理論を統合実行し構造化辞書を返す。"""
    if not lines_meta:
        return {
            "summary": {},
            "cornulier": diagnose_corpus_metricality([], []),
            "meschonnic": analyze_meschonnic_corpus([], [], [], [], []),
            "di_cristo": summarize_di_cristo([], []),
            "gouvard": {
                "rhyme": {"stanzas_detail": [], "fixed_form": "なし", "richness_counts": {}, "alternance_rate": 0.0},
                "discordances": {"events": [], "enjambement_count": 0, "rejet_count": 0, "contre_rejet_count": 0, "cesure_discordance_count": 0},
                "e_caduc_summary": {},
                "hiatus_events": [],
                "synerese_dierese_events": [],
            },
            "lines_detail": [],
            "stanzas_detail": [],
        }

    # 事前プローブでコーパス主軸メートル（12, 10, 8音節等）を推定
    probe_streams = [
        build_line_syllable_stream(ws, doc, lp, dierese=True, ancient_postvocalic_e=ancient_postvocalic_e)
        for ws, doc, lp in zip(unit_words, docs, line_phonetics_list)
    ]
    probe_lengths = [s["anatonic_count"] for s in probe_streams if s["anatonic_count"] > 0]
    target_meter = 12
    if probe_lengths:
        dom_m, dom_c = collections.Counter(probe_lengths).most_common(1)[0]
        if dom_m in (8, 10, 12) and dom_c >= max(1, len(probe_lengths) // 2):
            target_meter = dom_m

    # 第1パス: メトリック圧力解決を伴う各行の音節ストリーム構築
    line_streams = [
        _resolve_line_stream_with_metrical_pressure(ws, doc, lp, dierese=dierese, target_meter=target_meter, ancient_postvocalic_e=ancient_postvocalic_e)
        for ws, doc, lp in zip(unit_words, docs, line_phonetics_list)
    ]
    verse_words_list = [stream["verse_words"] for stream in line_streams]

    # Pillar 1: Cornulier 定型韻律・セジュール診断
    cornulier_lines = [
        analyze_cornulier_line(stream, line_idx=idx + 1)
        for idx, stream in enumerate(line_streams)
    ]
    cornulier_summary = diagnose_corpus_metricality(cornulier_lines, lines_meta)

    # Pillar 3: Di Cristo 4階層韻律句構造・双極アクセント
    dc_lines = [
        analyze_di_cristo_line(vws, stream, line_idx=idx + 1)
        for idx, (vws, stream) in enumerate(zip(verse_words_list, line_streams))
    ]
    dc_summary = summarize_di_cristo(dc_lines, line_phonetics_list)

    # Pillar 4: Gouvard / Mazaleyrat / Plénat 脚韻・詩節・統辞不一致・e caduc・母音衝突
    rhyme_sigs = [
        extract_line_rhyme_signature(vws, stream)
        for vws, stream in zip(verse_words_list, line_streams)
    ]
    rhyme_analysis = analyze_stanzas_and_rhymes(lines_meta, rhyme_sigs)
    discordance_analysis = detect_syntactic_discordances(lines_meta, docs, verse_words_list, line_streams, cornulier_lines)

    all_e_caduc = []
    all_hiatus = []
    all_syn_die = []
    for idx, stream in enumerate(line_streams, 1):
        for ev in stream["e_caduc_events"]:
            all_e_caduc.append({"line": idx, **ev})
        for ev in stream["hiatus_events"]:
            all_hiatus.append({"line": idx, **ev})
        for ev in stream["synerese_dierese_events"]:
            all_syn_die.append({"line": idx, **ev})

    e_caduc_counts = dict(collections.Counter(ev["label"] for ev in all_e_caduc))

    # Pillar 2: Meschonnic & Dessons ディスクール走査・反アクセント・プロゾディー反響網
    meschonnic_result = analyze_meschonnic_corpus(lines_meta, docs, verse_words_list, line_streams, cornulier_lines)

    # 行ごとの統合詳細レコード作成
    lines_detail = []
    for idx in range(len(lines_meta)):
        c_row = cornulier_lines[idx]
        dc_row = dc_lines[idx]
        m_row = meschonnic_result["lines"][idx]
        r_row = rhyme_analysis["line_rhyme_info"][idx]
        d_flags = discordance_analysis["line_flags"][idx]

        lines_detail.append({
            "line": idx + 1,
            "stanza": lines_meta[idx]["stanza"],
            "text": lines_meta[idx]["text"],
            "ipa": line_phonetics_list[idx]["ipa"],
            "anatonic_syllables": c_row["anatonic_length"],
            "genre": c_row["genre"],
            "meter_code": c_row["meter_code"],
            "loi_des_8": c_row["loi_des_8"],
            "caesura_code": c_row["caesura_code"],
            "caesura_label": c_row["caesura_label"],
            "ternary_profile": c_row["ternary_profile"],
            "cpmf_string": c_row["cpmf_string"],
            "measures_str": c_row["measures_str"],
            "measure_type": c_row["measure_type"],
            "rhyme_letter": r_row.get("rhyme_letter", ""),
            "rhyme_ipa": r_row.get("rhyme_ipa", ""),
            "rhyme_richness": r_row.get("rhyme_richness", ""),
            "discordances": ", ".join(d_flags) if d_flags else "一致 (concordant)",
            "meschonnic_scansion": m_row["scansion_unicode"],
            "meschonnic_tokens": m_row["syllable_tokens"],
            "contre_accents_count": len(m_row["contre_accents"]),
            "di_cristo_phrasing": dc_row["phrasing_str"],
            "di_cristo_marks": dc_row["syllable_marks"],
            "sp_count": dc_row["sp_count"],
            "ai_count": dc_row["ai_count"],
            "arcs_count": dc_row["arcs_count"],
        })

    summary = {
        "global_poetic_form": cornulier_summary["global_form"],
        "fixed_strophic_form": rhyme_analysis["fixed_form"],
        "dominant_meter": cornulier_summary["dominant_meter"],
        "isometry_ratio": cornulier_summary["isometry_ratio"],
        "eurythmy_score": dc_summary["eurythmy_score"],
        "bipolar_ratio": dc_summary["bipolar_ratio"],
        "open_syl_ratio_pct": dc_summary["open_syl_ratio_pct"],
        "npvi_interne": dc_summary["npvi_interne"],
        "npvi_global": dc_summary["npvi_global"],
        "total_contre_accents": meschonnic_result["total_contre_accents"],
        "total_echo_figures": len(meschonnic_result["echo_figures"]),
        "alternance_rate": rhyme_analysis["alternance_rate"],
        "total_discordances": len(discordance_analysis["events"]),
    }

    return {
        "summary": summary,
        "cornulier": cornulier_summary,
        "meschonnic": meschonnic_result,
        "di_cristo": dc_summary,
        "gouvard": {
            "rhyme": rhyme_analysis,
            "discordances": discordance_analysis,
            "e_caduc_summary": e_caduc_counts,
            "e_caduc_events": all_e_caduc,
            "hiatus_events": all_hiatus,
            "synerese_dierese_events": all_syn_die,
        },
        "lines_detail": lines_detail,
        "stanzas_detail": rhyme_analysis["stanzas_detail"],
    }
