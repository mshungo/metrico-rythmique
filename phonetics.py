"""フランス語の音素分析・IPA転写（リエゾン考慮）・音素偏差分析モジュール。
UIに依存しない。
"""
import re
import math
import collections
import statistics as st

# 有音のh（h aspiré: リエゾン・エリジオンを禁止する語頭のh）
H_ASPIRE = {
    "hache", "haches", "haie", "haies", "haillon", "haillons", "haine", "haines",
    "haïr", "hair", "haï", "haïs", "haïe", "haïes", "haineux", "haineuse",
    "haleter", "haletant", "haletante", "haletants", "hâle", "hâlé", "hâlés",
    "halle", "halles", "halte", "haltes", "hameau", "hameaux", "hameçon",
    "hanche", "hanches", "hangar", "hangars", "hanneton", "hannetons", "hanter",
    "hante", "hantent", "hanté", "hantés", "hantée", "happer", "harangue", "haras",
    "harceler", "harcelé", "harcelés", "hardi", "hardie", "hardies", "hardis", "hardiment",
    "harem", "hareng", "harengs", "hargne", "hargneux", "haricot", "haricots",
    "harnais", "harpe", "harpes", "harpon", "harpons", "hasard", "hasards", "hasarder",
    "hâte", "hâter", "hâtif", "hâtive", "hausse", "hausser", "haut", "haute", "hauts",
    "hautes", "hautain", "hautaine", "hautains", "hautaines", "hauteur", "hauteurs",
    "hâve", "hâves", "havre", "heaume", "heaumes", "hennir", "hennissement", "hennissements",
    "héraut", "hérauts", "hérisser", "hérissé", "hérissés", "hérissée", "hérissées",
    "hérisson", "hérissons", "héron", "hérons", "héros", "herse", "hêtre", "hêtres",
    "heurt", "heurts", "heurter", "heurte", "heurtent", "hibou", "hiboux",
    "hideux", "hideuse", "hideuses", "hideusement", "hiérarchie", "hiéroglyphe",
    "hisser", "hisse", "hissent", "hobereau", "hochement", "hocher", "hochet",
    "holà", "hollande", "homard", "homards", "hongrie", "honnir", "honni",
    "honte", "hontes", "honteux", "honteuse", "honteusement", "hoquet",
    "horde", "hordes", "hors", "hotte", "houblon", "houe", "houille", "houle",
    "houleux", "houleuse", "houlette", "houp", "houppe", "hourra", "house",
    "housses", "housse", "houx", "hublot", "hublots", "huche", "huée", "huées",
    "huer", "huguenot", "huguenots", "huit", "huitième", "huitaines", "hulotte",
    "hululer", "humer", "humoter", "huppe", "hure", "hurlement", "hurlements",
    "hurler", "hurle", "hurlent", "hurlant", "hurlante", "hurlants", "hussard",
    "hutte", "huttes",
}

VOWEL_CHARS = "aeiouyàâäéèêëîïôöûùüœæ"
SILENT_E_MARKER = "§"

# 頻出語・機能語辞書: (単語標準IPA, リエゾン潜在子音)
LEXICON = {
    # 限定詞
    "le": ("lə", None), "la": ("la", None), "les": ("le", "z"), "l": ("l", None), "l'": ("l", None),
    "un": ("œ̃", "n"), "une": ("yn", None), "des": ("de", "z"), "du": ("dy", None), "d": ("d", None), "d'": ("d", None),
    "de": ("də", None),
    "au": ("o", None), "aux": ("o", "z"),
    "mon": ("mɔ̃", "n"), "ton": ("tɔ̃", "n"), "son": ("sɔ̃", "n"),
    "mes": ("me", "z"), "tes": ("te", "z"), "ses": ("se", "z"),
    "nos": ("no", "z"), "vos": ("vo", "z"), "leurs": ("lœʁ", "z"), "leur": ("lœʁ", None),
    "ce": ("sə", None), "cet": ("sɛt", None), "cette": ("sɛt", None), "ces": ("se", "z"), "c'": ("s", None),
    "c'est": ("sɛ", "t"),
    "tout": ("tu", "t"), "tous": ("tu", "z"), "toute": ("tut", None), "toutes": ("tut", "z"),
    "aucun": ("okœ̃", "n"), "aucune": ("okyn", None),
    "quelque": ("kɛlkə", None), "quelques": ("kɛlkə", "z"),
    "chaque": ("ʃak", None), "plusieurs": ("plyzjœʁ", "z"),

    # 代名詞
    "je": ("ʒə", None), "j": ("ʒ", None), "j'": ("ʒ", None),
    "tu": ("ty", None), "t": ("t", None), "t'": ("t", None),
    "il": ("il", None), "elle": ("ɛl", None), "ils": ("il", "z"), "elles": ("ɛl", "z"),
    "on": ("ɔ̃", "n"), "nous": ("nu", "z"), "vous": ("vu", "z"),
    "me": ("mə", None), "m": ("m", None), "m'": ("m", None),
    "te": ("tə", None),
    "se": ("sə", None), "s": ("s", None), "s'": ("s", None),
    "lui": ("lɥi", None),
    "en": ("ɑ̃", "n"), "y": ("i", None), "moi": ("mwa", None), "toi": ("twa", None), "soi": ("swa", None),
    "qui": ("ki", None), "que": ("kə", None), "qu": ("k", None), "qu'": ("k", None), "quoi": ("kwa", None),
    "dont": ("dɔ̃", "t"), "où": ("u", None),
    "celui": ("səlɥi", None), "celle": ("sɛl", None), "ceux": ("sø", "z"), "celles": ("sɛl", "z"),

    # 前置詞・接続詞
    "dans": ("dɑ̃", "z"), "sans": ("sɑ̃", "z"), "sous": ("su", "z"), "sur": ("syʁ", None),
    "par": ("paʁ", None), "pour": ("puʁ", None), "avec": ("avɛk", None), "chez": ("ʃe", "z"),
    "vers": ("vɛʁ", "z"), "avant": ("avɑ̃", "t"), "après": ("apʁɛ", "z"),
    "entre": ("ɑ̃tʁ", None), "contre": ("kɔ̃tʁ", None), "pendant": ("pɑ̃dɑ̃", "t"),
    "depuis": ("dəpɥi", "z"), "selon": ("səlɔ̃", None),
    "et": ("e", None), # 厳格な規則: 'et' は絶対にリエゾンしない
    "ou": ("u", None), "mais": ("mɛ", "z"), "donc": ("dɔ̃k", None), "car": ("kaʁ", None),
    "ni": ("ni", None), "si": ("si", None), "comme": ("kɔm", None),
    "quand": ("kɑ̃", "t"), "lorsque": ("lɔʁskə", None), "puisque": ("pɥiskə", None),
    "jusque": ("ʒyskə", None), "jusqu'": ("ʒysk", None),

    # 副詞
    "ne": ("nə", None), "n": ("n", None), "n'": ("n", None), "pas": ("pa", "z"), "plus": ("ply", "z"),
    "bien": ("bjɛ̃", "n"), "très": ("tʁɛ", "z"), "trop": ("tʁo", "p"),
    "jamais": ("ʒamɛ", "z"), "toujours": ("tuʒuʁ", "z"), "souvent": ("suvɑ̃", "t"),
    "peu": ("pø", None), "beaucoup": ("boku", "p"), "assez": ("ase", "z"),
    "tant": ("tɑ̃", "t"), "autant": ("otɑ̃", "t"), "ici": ("isi", None), "là": ("la", None),
    "hier": ("jɛʁ", None), "demain": ("dəmɛ̃", None), "maintenant": ("mɛ̃tnɑ̃", "t"),
    "rien": ("ʁjɛ̃", "n"), "non": ("nɔ̃", None), "oui": ("wi", None),
    "encore": ("ɑ̃kɔʁ", None), "aussi": ("osi", None), "ainsi": ("ɛ̃si", None), "alors": ("alɔʁ", "z"),

    # 動詞頻出形 (être / avoir / aller / faire など)
    "est": ("ɛ", "t"), "sont": ("sɔ̃", "t"), "suis": ("sɥi", "z"), "es": ("ɛ", "z"),
    "sommes": ("sɔm", "z"), "êtes": ("ɛt", "z"), "était": ("etɛ", "t"), "étaient": ("etɛ", "t"),
    "été": ("ete", None), "fus": ("fy", "z"), "fut": ("fy", "t"),
    "sera": ("səʁa", None), "seront": ("səʁɔ̃", "t"), "serait": ("səʁɛ", "t"), "seraient": ("səʁɛ", "t"),
    "soit": ("swa", "t"), "soient": ("swa", "t"),
    "a": ("a", None), "ai": ("e", None), "as": ("a", "z"), "avons": ("avɔ̃", "z"), "avez": ("ave", "z"),
    "ont": ("ɔ̃", "t"), "avait": ("avɛ", "t"), "avaient": ("avɛ", "t"),
    "aura": ("oʁa", None), "auront": ("oʁɔ̃", "t"), "aurait": ("oʁɛ", "t"), "auraient": ("oʁɛ", "t"),
    "fait": ("fɛ", "t"), "font": ("fɔ̃", "t"), "faisait": ("fəzɛ", "t"), "faisaient": ("fəzɛ", "t"),
    "va": ("va", None), "vais": ("vɛ", "z"), "vont": ("vɔ̃", "t"), "allait": ("alɛ", "t"), "allaient": ("alɛ", "t"),
    "peut": ("pø", "t"), "peuvent": ("pœv", None), "doit": ("dwa", "t"), "doivent": ("dwav", None),
    "dit": ("di", "t"), "disait": ("dizɛ", "t"), "sait": ("sɛ", "t"), "savait": ("savɛ", "t"),
    "veut": ("vø", "t"), "voulait": ("vulɛ", "t"), "voit": ("vwa", "t"), "voyait": ("vwajɛ", "t"),
    "vient": ("vjɛ̃", "t"), "viennent": ("vjɛn", None),

    # 形容詞・数詞・頻出名詞
    "deux": ("dø", "z"), "trois": ("tʁwa", "z"), "six": ("si", "z"), "dix": ("di", "z"),
    "cent": ("sɑ̃", "t"), "vingt": ("vɛ̃", "t"),
    "grand": ("ɡʁɑ̃", "t"), "grands": ("ɡʁɑ̃", "z"), "grande": ("ɡʁɑ̃d", None), "grandes": ("ɡʁɑ̃d", "z"),
    "petit": ("pəti", "t"), "petits": ("pəti", "z"), "petite": ("pətit", None), "petites": ("pətit", "z"),
    "bon": ("bɔ̃", "n"), "bons": ("bɔ̃", "z"), "bonne": ("bɔn", None), "bonnes": ("bɔn", "z"),
    "beau": ("bo", None), "beaux": ("bo", "z"), "belle": ("bɛl", None), "belles": ("bɛl", "z"),
    "nouveau": ("nuvo", None), "nouveaux": ("nuvo", "z"), "nouvelle": ("nuvɛl", None), "nouvelles": ("nuvɛl", "z"),
    "premier": ("pʁəmje", "ʁ"), "premiers": ("pʁəmje", "z"), "première": ("pʁəmjɛʁ", None), "premières": ("pʁəmjɛʁ", "z"),
    "dernier": ("dɛʁnje", "ʁ"), "derniers": ("dɛʁnje", "z"), "dernière": ("dɛʁnjɛʁ", None), "dernières": ("dɛʁnjɛʁ", "z"),
    "long": ("lɔ̃", "k"), "longs": ("lɔ̃", "z"), "longue": ("lɔ̃ɡ", None), "longues": ("lɔ̃ɡ", "z"),
    "gros": ("ɡʁo", "z"), "vieux": ("vjø", "z"), "vieille": ("vjɛj", None),
    "haut": ("o", "t"), "hauts": ("o", "z"),
    "ville": ("vil", None), "villes": ("vil", "z"),
    "mille": ("mil", None), "milles": ("mil", "z"),
    "tranquille": ("tʁɑ̃kil", None), "tranquilles": ("tʁɑ̃kil", "z"),
    "mer": ("mɛʁ", None), "mers": ("mɛʁ", "z"),
    "fer": ("fɛʁ", None), "fers": ("fɛʁ", "z"),
    "cher": ("ʃɛʁ", None), "chers": ("ʃɛʁ", "z"),
    "hiver": ("ivɛʁ", None), "hivers": ("ivɛʁ", "z"),
    "terre": ("tɛʁ", None), "terres": ("tɛʁ", "z"),
    "ciel": ("sjɛl", None), "cieux": ("sjø", "z"),
    "soleil": ("sɔlɛj", None), "soleils": ("sɔlɛj", "z"),
    "sommeil": ("sɔmɛj", None),
    "cœur": ("kœʁ", None), "cœurs": ("kœʁ", "z"),
    "œil": ("œj", None), "yeux": ("jø", "z"),
    "femme": ("fam", None), "femmes": ("fam", "z"),
    "homme": ("ɔm", None), "hommes": ("ɔm", "z"),
    "temps": ("tɑ̃", "z"),
    "jour": ("ʒuʁ", None), "jours": ("ʒuʁ", "z"),
    "nuit": ("nɥi", "t"), "nuits": ("nɥi", "z"),
    "vie": ("vi", None), "vies": ("vi", "z"),
    "mort": ("mɔʁ", "t"), "morts": ("mɔʁ", "z"),
    "amour": ("amuʁ", None), "amours": ("amuʁ", "z"),
    "arbre": ("aʁbʁ", None), "arbres": ("aʁbʁ", "z"),
    "rose": ("ʁoz", None), "roses": ("ʁoz", "z"),
    "chose": ("ʃoz", None), "choses": ("ʃoz", "z"),
    "âme": ("am", None), "âmes": ("am", "z"),
    "ombre": ("ɔ̃bʁ", None), "ombres": ("ɔ̃bʁ", "z"),
    "lumière": ("lymjɛʁ", None), "lumières": ("lymjɛʁ", "z"),
    "enfant": ("ɑ̃fɑ̃", "t"), "enfants": ("ɑ̃fɑ̃", "z"),
    "ami": ("ami", None), "amis": ("ami", "z"),
    "amie": ("ami", None), "amies": ("ami", "z"),
    "heure": ("œʁ", None), "heures": ("œʁ", "z"),
    "aujourd'hui": ("oʒuʁdɥi", None),
}

# フランス語標準音素分布リファレンス（Delattre / Wioland / コーパス統計基準値）
FRENCH_PHONEME_REFERENCE = {
    # 口母音
    "a": {"mean": 8.10, "sd": 1.8, "name": "非円唇前舌開母音 [a]", "category": "母音（口母音・開・前舌）", "type": "vowel"},
    "e": {"mean": 7.40, "sd": 1.6, "name": "非円唇前舌半閉母音 [e]", "category": "母音（口母音・半閉・前舌）", "type": "vowel"},
    "i": {"mean": 6.60, "sd": 1.5, "name": "非円唇前舌狭母音 [i]", "category": "母音（口母音・閉・前舌）", "type": "vowel"},
    "ɛ": {"mean": 5.30, "sd": 1.4, "name": "非円唇前舌半開母音 [ɛ]", "category": "母音（口母音・半開・前舌）", "type": "vowel"},
    "o": {"mean": 2.90, "sd": 1.0, "name": "円唇後舌半閉母音 [o]", "category": "母音（口母音・半閉・後舌）", "type": "vowel"},
    "u": {"mean": 2.50, "sd": 0.9, "name": "円唇後舌狭母音 [u]", "category": "母音（口母音・閉・後舌）", "type": "vowel"},
    "y": {"mean": 2.30, "sd": 0.8, "name": "円唇前舌狭母音 [y]", "category": "母音（口母音・閉・前舌）", "type": "vowel"},
    "ɔ": {"mean": 2.00, "sd": 0.8, "name": "円唇後舌半開母音 [ɔ]", "category": "母音（口母音・半開・後舌）", "type": "vowel"},
    "ə": {"mean": 1.80, "sd": 0.9, "name": "中舌中央母音（シュワー） [ə]", "category": "母音（口母音・中舌）", "type": "vowel"},
    "ø": {"mean": 0.80, "sd": 0.4, "name": "円唇前舌半閉母音 [ø]", "category": "母音（口母音・半閉・前舌）", "type": "vowel"},
    "œ": {"mean": 0.70, "sd": 0.4, "name": "円唇前舌半開母音 [œ]", "category": "母音（口母音・半開・前舌）", "type": "vowel"},
    "ɑ": {"mean": 0.50, "sd": 0.3, "name": "非円唇後舌開母音 [ɑ]", "category": "母音（口母音・開・後舌）", "type": "vowel"},
    # 鼻母音
    "ɑ̃": {"mean": 3.50, "sd": 1.1, "name": "円唇後舌鼻母音 [ɑ̃]", "category": "母音（鼻母音・開）", "type": "nasal_vowel"},
    "ɔ̃": {"mean": 2.20, "sd": 0.8, "name": "円唇後舌鼻母音 [ɔ̃]", "category": "母音（鼻母音・半閉）", "type": "nasal_vowel"},
    "ɛ̃": {"mean": 2.10, "sd": 0.8, "name": "非円唇前舌鼻母音 [ɛ̃]", "category": "母音（鼻母音・半開）", "type": "nasal_vowel"},
    "œ̃": {"mean": 0.30, "sd": 0.2, "name": "円唇前舌鼻母音 [œ̃]", "category": "母音（鼻母音・半開）", "type": "nasal_vowel"},
    # 半母音
    "j": {"mean": 1.80, "sd": 0.7, "name": "硬口蓋接近音（半母音） [j]", "category": "半母音", "type": "glide"},
    "w": {"mean": 1.00, "sd": 0.5, "name": "両唇軟口蓋接近音（半母音） [w]", "category": "半母音", "type": "glide"},
    "ɥ": {"mean": 0.40, "sd": 0.3, "name": "両唇硬口蓋接近音（半母音） [ɥ]", "category": "半母音", "type": "glide"},
    # 子音
    "ʁ": {"mean": 10.50, "sd": 2.0, "name": "口蓋垂摩擦音・流音 [ʁ]", "category": "子音（流音）", "type": "consonant"},
    "l": {"mean": 6.80, "sd": 1.5, "name": "歯茎側面接近音・流音 [l]", "category": "子音（流音）", "type": "consonant"},
    "s": {"mean": 6.20, "sd": 1.4, "name": "無声歯茎摩擦音 [s]", "category": "子音（摩擦音・無声）", "type": "consonant"},
    "t": {"mean": 6.00, "sd": 1.3, "name": "無声歯茎破裂音 [t]", "category": "子音（破裂音・無声）", "type": "consonant"},
    "d": {"mean": 4.10, "sd": 1.0, "name": "有声歯茎破裂音 [d]", "category": "子音（破裂音・有声）", "type": "consonant"},
    "k": {"mean": 3.70, "sd": 1.0, "name": "無声軟口蓋破裂音 [k]", "category": "子音（破裂音・無声）", "type": "consonant"},
    "p": {"mean": 3.30, "sd": 0.9, "name": "無声両唇破裂音 [p]", "category": "子音（破裂音・無声）", "type": "consonant"},
    "m": {"mean": 3.20, "sd": 0.9, "name": "両唇鼻音 [m]", "category": "子音（鼻音）", "type": "consonant"},
    "n": {"mean": 3.10, "sd": 0.9, "name": "歯茎鼻音 [n]", "category": "子音（鼻音）", "type": "consonant"},
    "v": {"mean": 2.50, "sd": 0.8, "name": "有声唇歯摩擦音 [v]", "category": "子音（摩擦音・有声）", "type": "consonant"},
    "z": {"mean": 2.40, "sd": 0.8, "name": "有声歯茎摩擦音 [z]", "category": "子音（摩擦音・有声）", "type": "consonant"},
    "b": {"mean": 1.80, "sd": 0.6, "name": "有声両唇破裂音 [b]", "category": "子音（破裂音・有声）", "type": "consonant"},
    "ʒ": {"mean": 1.60, "sd": 0.6, "name": "有声後部歯茎摩擦音 [ʒ]", "category": "子音（摩擦音・有声）", "type": "consonant"},
    "f": {"mean": 1.50, "sd": 0.6, "name": "無声唇歯摩擦音 [f]", "category": "子音（摩擦音・無声）", "type": "consonant"},
    "ʃ": {"mean": 1.10, "sd": 0.5, "name": "無声後部歯茎摩擦音 [ʃ]", "category": "子音（摩擦音・無声）", "type": "consonant"},
    "ɡ": {"mean": 0.90, "sd": 0.4, "name": "有声軟口蓋破裂音 [ɡ]", "category": "子音（破裂音・有声）", "type": "consonant"},
    "ɲ": {"mean": 0.30, "sd": 0.2, "name": "硬口蓋鼻音 [ɲ]", "category": "子音（鼻音）", "type": "consonant"},
    "ŋ": {"mean": 0.10, "sd": 0.1, "name": "軟口蓋鼻音 [ŋ]", "category": "子音（鼻音）", "type": "consonant"},
}

PHONEMES_ALL = list(FRENCH_PHONEME_REFERENCE.keys())
PHONEME_PATTERN = re.compile("|".join(sorted(PHONEMES_ALL, key=len, reverse=True)))

# 音素分類セット
VOWELS_ORAL = {"i", "e", "ɛ", "a", "ɑ", "ɔ", "o", "u", "y", "ø", "œ", "ə"}
VOWELS_NASAL = {"ɛ̃", "ɑ̃", "ɔ̃", "œ̃"}
VOWELS_ALL = VOWELS_ORAL | VOWELS_NASAL
GLIDES = {"j", "w", "ɥ"}
CONSONANTS = {
    "p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z", "ʃ", "ʒ", "m", "n", "ɲ", "ŋ", "l", "ʁ"
}

# 調音特徴分類
FRONT_VOWELS = {"i", "e", "ɛ", "a", "y", "ø", "œ", "ɛ̃", "œ̃"}
BACK_VOWELS = {"u", "o", "ɔ", "ɑ", "ɔ̃", "ɑ̃"}
CLOSE_VOWELS = {"i", "y", "u"}
OPEN_VOWELS = {"a", "ɑ", "ɑ̃", "ɛ", "œ", "ɔ"}
ROUNDED_VOWELS = {"y", "ø", "œ", "u", "o", "ɔ", "œ̃", "ɔ̃"}

PLOSIVES = {"p", "b", "t", "d", "k", "ɡ"}
FRICATIVES = {"f", "v", "s", "z", "ʃ", "ʒ"}
LIQUIDS = {"l", "ʁ"}
NASAL_CONSONANTS = {"m", "n", "ɲ", "ŋ"}
VOICED_CONSONANTS = {"b", "d", "ɡ", "v", "z", "ʒ", "m", "n", "ɲ", "ŋ", "l", "ʁ"}
VOICELESS_CONSONANTS = {"p", "t", "k", "f", "s", "ʃ"}


# -ent で終わるが動詞3人称複数ではない名詞・形容詞・副詞・単数動詞の保護リスト
NON_SILENT_ENT_WORDS = {
    "vent", "dent", "cent", "gent", "lent", "sent", "ment", "rend", "tend", "pend", "fend", "vend", "prend",
    "vient", "tient", "devient", "souvient", "revient", "parvient", "appartient", "contient", "retient", "obtient",
    "soutient", "maintient", "consent", "pressent", "ressent", "repent",
    "argent", "serpent", "torrent", "parent", "talent", "client", "orient", "occident", "continent",
    "accident", "incident", "président", "résident", "moment", "tourment", "serment", "froment", "jument",
    "élément", "vêtement", "bâtiment", "sentiment", "mouvement", "monument", "document", "instrument",
    "argument", "ornement", "jugement", "logement", "changement", "battement", "tremblement", "sifflement",
    "hurlement", "clapotement", "frémissement", "éblouissement", "firmament", "testament", "sacrement",
    "présent", "absent", "récent", "décent", "innocent", "adolescent", "incandescent", "phosphorescent",
    "lactescent", "convalescent", "ardent", "prudent", "impudent", "évident", "confident", "strident",
    "violent", "insolent", "indolent", "somnolent", "turbulent", "corpulent", "succulent", "opulent",
    "excellent", "équivalent", "ambivalent", "cohérent", "différent", "indifférent", "apparent", "transparent",
    "patient", "impatient", "conscient", "inconscient", "quotient", "récipient", "ingrédient", "expédient",
    "souvent", "comment",
}

# 語末子音が発音される例外語
PRONOUNCED_FINAL_S = {
    "fils", "ours", "sens", "lys", "lis", "os", "as", "hélas", "jadis", "mars", "maïs",
    "oasis", "iris", "lotus", "angélus", "vénus", "phébus", "bacchus", "blocus", "omnibus",
    "virus", "chorus", "albatros", "cosmos", "pathos", "rhinocéros", "calvados", "mérinos",
}
PRONOUNCED_FINAL_R_ER = {
    "mer", "fer", "ver", "cher", "fier", "hier", "hiver", "enfer", "amer", "cancer", "éther", "super", "pater",
}
PRONOUNCED_FINAL_T = {
    "net", "brut", "chut", "dot", "rut", "huit", "sept", "ouest", "lest", "test", "christ", "rapt",
    "abrupt", "direct", "exact", "infect", "intact", "strict", "tact", "contact", "correct",
}
ARCHAIC_IMPERFECT_OIENT = {
    "avoient", "étoient", "faisoient", "disoient", "savoient", "pouvoient",
    "vouloient", "devoient", "venoient", "tenoient", "prenoient", "aimoient",
}
AMBIGUOUS_ENT_HOMOGRAPHS = {
    "parent", "couvent", "président", "résident", "violent", "content", "ferment", "pressent", "différent", "équivalent",
}
PRONOUNCED_FINAL_C = {
    "lac", "sac", "bec", "sec", "avec", "échec", "grec", "bouc", "duc", "suc", "truc", "turc", "arc", "parc", "donc", "roc", "choc", "bloc", "froc", "troc", "soc", "coq", "cinq",
}


def normalize_french_orthography(w):
    """ASCII合字 (oe -> œ, ae -> æ) およびアポストロフィを標準フランス語正書法へ正規化する。"""
    w = w.replace("’", "'")
    w = re.sub(r"oeu", "œu", w)
    w = re.sub(r"oeil", "œil", w)
    w = re.sub(r"\b(c|s|ch|m)oeur(s?)\b", r"\1œur\2", w)
    w = re.sub(r"\b(man)?oeuvre(s?)\b", r"\1œuvre\2", w)
    w = re.sub(r"\b(v|n)oeu([dxs]?)\b", r"\1œu\2", w)
    w = re.sub(r"\b(b)?oeuf(s?)\b", r"\1œuf\2", w)
    w = re.sub(r"foetus", "fœtus", w)
    return w


def _is_silent_3pl_ent(w, pos=None):
    """語末 -ent が無音の動詞3人称複数語尾であるか厳密判定する。"""
    if not w.endswith("ent") or len(w) <= 4:
        return False
    if w in {"soient", "aient"} or w.endswith("aient") or w in ARCHAIC_IMPERFECT_OIENT:
        return False
    if w in AMBIGUOUS_ENT_HOMOGRAPHS:
        return pos in ("VERB", "AUX")
    if w in NON_SILENT_ENT_WORDS:
        return False
    # 副詞・名詞の生産的 -ment 接尾辞を保護（ただし aiment, dorment 等は動詞）
    if w.endswith("ment"):
        verb_ment = {
            "aiment", "arment", "ferment", "forment", "charment", "calment", "sèment",
            "fument", "allument", "consument", "écument", "parfument", "nomment", "assomment",
            "dorment", "endorment", "rament", "brament", "clament", "enflamment", "blâment",
            "estiment", "animent", "expriment", "oppriment", "suppriment", "impriment", "riment", "abîment",
        }
        if w not in verb_ment:
            return False
        if w == "ferment":
            return pos in ("VERB", "AUX")
        return True
    # 単数形容詞・名詞の典型接尾辞 (-dent, -lent, -cent, -gent, -rent, -vent) は POS を確認
    if w.endswith(("dent", "lent", "cent", "gent", "scient", "tient", "vient")) and pos not in ("VERB", "AUX"):
        return False
    return True


def word_to_ipa(word, pos=None):
    """単語を標準IPAと潜在リエゾン子音のタプル (ipa, latent_consonant) に高精度変換する。"""
    w = normalize_french_orthography(word.lower().strip())
    if not w:
        return "", None

    # 1. 辞書照合（完全一致）
    if w in LEXICON:
        return LEXICON[w]

    # 2. アポストロフィ短縮形（qu'il, l'arbre, l'œil, d'or, c'est 等）
    if "'" in w:
        parts = w.split("'", 1)
        pref = parts[0] + "'"
        rest = parts[1]
        elision_prefixes = {
            "l'": "l", "d'": "d", "j'": "ʒ", "m'": "m", "t'": "t",
            "s'": "s", "c'": "s", "n'": "n", "qu'": "k", "quelqu'": "kɛlk", "jusqu'": "ʒysk", "presqu'": "pʁɛsk",
        }
        if pref in elision_prefixes:
            pref_ipa = elision_prefixes[pref]
            if not rest:
                return pref_ipa, None
            rest_ipa, rest_latent = word_to_ipa(rest, pos=pos)
            return pref_ipa + rest_ipa, rest_latent

    # 3. ハイフン複合語（peut-être, avez-vous 等）
    if "-" in w and w not in LEXICON:
        subparts = [p for p in w.split("-") if p]
        sub_ipas = []
        last_latent = None
        for idx, sp in enumerate(subparts):
            s_ipa, s_latent = word_to_ipa(sp, pos=pos if idx == len(subparts) - 1 else None)
            sub_ipas.append(s_ipa)
            if idx == len(subparts) - 1:
                last_latent = s_latent
        return "".join(sub_ipas), last_latent

    # 特殊語彙例外
    if w in ("pays",):
        return "pei", "z"
    if w in ("paysan", "paysans"):
        return "peizɑ̃", ("z" if w.endswith("s") else None)
    if w in ("paysage", "paysages"):
        return "peizaʒ", ("z" if w.endswith("s") else None)
    if w in ("abbaye", "abbayes"):
        return "abei", ("z" if w.endswith("s") else None)
    if w == "fils":
        return "fis", None
    if w in ("femme", "femmes"):
        return "fam", ("z" if w.endswith("s") else None)
    if w in ("monsieur",):
        return "məsjø", None
    if w in ("messieurs",):
        return "mesjø", "z"
    if w in ("choeur", "chœur", "choeurs", "chœurs"):
        return "kœʁ", ("z" if w.endswith("s") else None)
    if w in ("écho", "échos"):
        return "eko", ("z" if w.endswith("s") else None)
    if w in ("chaos",):
        return "kao", "z"
    if w in ("orchestre", "orchestres"):
        return "ɔʁkɛstʁ", ("z" if w.endswith("s") else None)
    if w in ("maelstrom", "maelstroms", "maëlstrom", "maëlstroms"):
        return "mɛlstʁɔm", ("z" if w.endswith("s") else None)
    if w in ("paon", "paons"):
        return "pɑ̃", ("z" if w.endswith("s") else None)
    if w in ("faon", "faons"):
        return "fɑ̃", ("z" if w.endswith("s") else None)
    if w in ("taon", "taons"):
        return "tɑ̃", ("z" if w.endswith("s") else None)
    if w in ("oignon", "oignons"):
        return "ɔɲɔ̃", ("z" if w.endswith("s") else None)
    if w in ("souhait", "souhaits"):
        return "swɛ", ("z" if w.endswith("s") else "t")
    if w in ("août",):
        return "ut", None
    if w in ("saoul", "saouls", "soûl", "soûls"):
        return "su", ("z" if w.endswith("s") else None)

    # 潜在リエゾン子音の特定
    latent = None
    if w.endswith(("s", "x", "z")) and w not in PRONOUNCED_FINAL_S:
        latent = "z"
    elif w.endswith(("t", "d")) and w not in PRONOUNCED_FINAL_T:
        latent = "t"
    elif w.endswith("n") and len(w) > 1 and w[-2] in VOWEL_CHARS:
        latent = "n"
    elif w.endswith("p") and w not in {"cap", "cep"}:
        latent = "p"
    elif w.endswith("er") and w not in PRONOUNCED_FINAL_R_ER:
        latent = "ʁ"

    orig_w = w

    # --- プレースホルダー型G2P変換エンジン ---

    # 4. 語末屈折（複数 -s / -x）と無音語尾の整理（原綴りのまま先に判定）
    has_silent_fem = False
    if _is_silent_3pl_ent(w, pos=pos):
        w = w[:-3] + SILENT_E_MARKER
        has_silent_fem = True
    elif (w.endswith("aient") and len(w) > 5) or w in ARCHAIC_IMPERFECT_OIENT:
        w = w[:-5] + "@E_OPEN@"
    else:
        # 複数形 -s / -x の除去（発音される語末 -s は保護）
        if w not in PRONOUNCED_FINAL_S and len(w) >= 2:
            if w.endswith("es") and len(w) >= 3 and w not in {"les", "des", "mes", "tes", "ses", "ces"}:
                w = w[:-2] + "e"
            elif w.endswith(("s", "x")) and w not in {"ours", "mars", "fils", "lys", "lis", "os", "as", "index", "sphinx", "lynx", "onyx", "phénix", "thorax", "gaz"}:
                w = w[:-1]

        # 女性無音 -e（単音節機能語 le, ce, de 等はすでに LEXICON 処理済み）
        if w.endswith("e") and len(w) >= 2:
            w = w[:-1] + SILENT_E_MARKER
            has_silent_fem = True
        elif w.endswith("er") and len(w) >= 3 and w not in PRONOUNCED_FINAL_R_ER:
            # pilier(s), regretter, aimer -> 語末 -er は [e]
            w = w[:-2] + "@ER@"
        elif w.endswith("ez") and len(w) >= 3:
            w = w[:-2] + "@ER@"
        elif w.endswith("et") and len(w) >= 3 and w not in PRONOUNCED_FINAL_T:
            w = w[:-2] + "@E_OPEN@"
        elif w.endswith("ot") and len(w) >= 3 and w not in PRONOUNCED_FINAL_T:
            w = w[:-2] + "@O@"
        else:
            # 語末の無音子音 (-t, -d, -p, -c, -g, -z) の除去（nus, pas, falots->falot, vivants->vivant, loups->loup, blancs->blanc 等）
            if len(w) >= 2:
                if w.endswith("t") and orig_w not in PRONOUNCED_FINAL_T and not w.endswith(("st",)):
                    w = w[:-1]
                elif w.endswith("d") and orig_w not in {"sud"}:
                    w = w[:-1]
                elif w.endswith("p") and orig_w not in {"cap", "cep", "croup", "handicap"}:
                    w = w[:-1]
                elif w.endswith("c") and orig_w not in PRONOUNCED_FINAL_C and w not in PRONOUNCED_FINAL_C:
                    w = w[:-1]
                elif w.endswith("g") and orig_w not in {"grog", "zigzag", "gong"}:
                    w = w[:-1]
                elif w.endswith("z") and orig_w not in {"gaz", "fez", "quartz", "hertz"}:
                    w = w[:-1]

    # 5. cueil / gueil と ge+[aou] の軟音化処理
    w = re.sub(r"cueill?", "k@OE_OPEN@@J@", w)
    w = re.sub(r"gueill?", "ɡ@OE_OPEN@@J@", w)
    w = re.sub(r"ge(?=[aàâoôuû])", "ʒ", w)

    # 6. 原綴りで判定すべき子音規則（SILENT_E_MARKER 直前の軟音 c/g/gu も対象。軟音 c/ç は母音間 s->z 有声化から保護！）
    w = re.sub(r"(?<!s)tion$", "@S_SOFT@i@O~@", w)
    w = re.sub(r"ch", "@CH@", w)
    w = re.sub(r"ph", "f", w)
    w = re.sub(r"th", "t", w)
    w = re.sub(r"gn", "@GN@", w)
    w = re.sub(r"qu", "k", w)
    w = re.sub(rf"gu(?=[eéèêiîy{SILENT_E_MARKER}])", "ɡ", w)
    w = re.sub(r"ç", "@S_SOFT@", w)
    w = re.sub(rf"c(?=[eéèêëiîïy{SILENT_E_MARKER}])", "@S_SOFT@", w)
    w = re.sub(r"c", "k", w)
    w = re.sub(rf"g(?=[eéèêiîy{SILENT_E_MARKER}])", "ʒ", w)
    w = re.sub(r"g", "ɡ", w)
    w = re.sub(r"j", "ʒ", w)
    w = re.sub(rf"(?<=[{VOWEL_CHARS}])h(?=[{VOWEL_CHARS}])", "@HIATUS@", w)
    w = re.sub(r"h", "", w)

    # 7. 母音間 y の音韻分解 (ayant -> ɛjɑ̃, voyage -> vwajaʒ, appuyer -> apɥije)
    vow_plus_markers = VOWEL_CHARS + SILENT_E_MARKER + "@"
    w = re.sub(rf"ay(?=[{VOWEL_CHARS}@])", "@E_OPEN@j", w)
    w = re.sub(rf"oy(?=[{VOWEL_CHARS}@])", "@WA@j", w)
    w = re.sub(rf"uy(?=[{VOWEL_CHARS}@])", "@UI@j", w)

    # 8. 母音間 s の有声化 [z] と x の音素化
    w = re.sub(rf"^ex(?=[{VOWEL_CHARS}@])", "ɛɡz", w)
    w = re.sub(r"^ex", "ɛks", w)
    w = re.sub(r"x", "ks", w)
    w = re.sub(rf"(?<=[{VOWEL_CHARS}{SILENT_E_MARKER}])s(?=[{VOWEL_CHARS}{SILENT_E_MARKER}@])", "z", w)
    w = re.sub(r"ss", "s", w)

    # 9. イル（ill/il）音（鼻母音・一般母音より先に保護。ただし語頭 ill- や million の ill+i は除外）
    w = re.sub(rf"eill(?={SILENT_E_MARKER}|$)|eil$", "@E_OPEN@j", w)
    w = re.sub(r"eill", "@E_OPEN@j", w)
    w = re.sub(rf"aill(?={SILENT_E_MARKER}|$)|ail$", "aj", w)
    w = re.sub(r"aill", "aj", w)
    w = re.sub(rf"ouill(?={SILENT_E_MARKER}|$)|ouil$", "uj", w)
    w = re.sub(r"ouill", "uj", w)
    w = re.sub(rf"(?:euill|œill)(?={SILENT_E_MARKER}|$)|(?:euil|œil)$", "@OE_OPEN@j", w)
    w = re.sub(r"(?:euill|œill)", "@OE_OPEN@j", w)
    # 子音＋ill（語頭 ^ill と ville/mille/tranquille、および後続 i がある million を除外）
    if not orig_w.startswith(("ill", "mill", "vill", "tranquill")):
        w = re.sub(r"ill(?![iî])", "@IJ@", w)

    # 10. 鼻母音（e+CC 規則より必ず先に適用する：temple -> tɑ̃pl, membre -> mɑ̃bʁ）
    not_vow_or_nm = rf"(?![{VOWEL_CHARS}mn{SILENT_E_MARKER}])"
    w = re.sub(rf"oin{not_vow_or_nm}", "@WE~@", w)
    w = re.sub(rf"(?:ien|yen){not_vow_or_nm}", "@JE~@", w)
    w = re.sub(rf"(?:ain|aim|ein|eim|yn|ym|in|im){not_vow_or_nm}", "@E~@", w)
    w = re.sub(rf"(?:an|am|en|em){not_vow_or_nm}", "@A~@", w)
    w = re.sub(rf"(?:on|om){not_vow_or_nm}", "@O~@", w)
    w = re.sub(rf"(?:un|um){not_vow_or_nm}", "@OE~@", w)

    # 11. 閉音節の e -> [ɛ]（ただし muta cum liquida [pbtdkɡfv][lr] の直前は開音節なのでシュワー [ə] を保持！）
    consonant_letters = "bcdfghjklmnpqrstvwxzɡʃʒɲ"
    w = re.sub(rf"e(?!(?:[pbtdkɡfv][lr]))(?=[{consonant_letters}]{{2}})", "@E_OPEN@", w)
    w = re.sub(rf"e(?=[rlcfts]$)", "@E_OPEN@", w)

    # r の音素化
    w = re.sub(r"r+", "ʁ", w)

    # 12. 複合母音・半母音化
    w = re.sub(r"eau|au", "@O@", w)
    w = re.sub(r"ou|où|oû", "@U@", w)
    w = re.sub(r"oi|oî", "@WA@", w)
    w = re.sub(r"ui", "@UI@", w)
    # œu / eû / eu: [ʁ, v, j, f, l, p, b, k, ɡ] 等の閉音節では開母音 [œ]、語末や [z] 前では閉母音 [ø]
    w = re.sub(rf"(?:œu|eu)(?=[ʁvjflpbkɡt]{SILENT_E_MARKER}?$|[ʁvjflpbkɡ]{{2}})", "@OE_OPEN@", w)
    w = re.sub(r"œu|eû|eu", "@EU@", w)
    w = re.sub(r"œ", "@OE_OPEN@", w)
    w = re.sub(r"æ", "@E_CLOSED@", w)
    w = re.sub(r"aî|ai|eî|ei|è|ê|ë", "@E_OPEN@", w)
    w = re.sub(r"é", "@E_CLOSED@", w)

    # 単一子音後の i + 母音プレースホルダー（pilier -> pilje, ciel -> sjɛl 等）
    w = re.sub(r"(?<![pbtdkɡfv][lʁ])(?<![^a-zà-ÿ@])i(?=@(?:ER|E_OPEN|E_CLOSED|O|U|EU|OE_OPEN|A~|O~|E~)@|[aàâoôuû])", "j", w)

    # 13. 単一母音
    w = re.sub(r"[uûùü]", "@Y_VOWEL@", w)
    w = re.sub(r"[aàâä]", "a", w)
    w = re.sub(r"[oôö]", "o", w)
    w = re.sub(r"[iîïy]", "i", w)
    w = re.sub(rf"{SILENT_E_MARKER}|e$", "", w)
    w = re.sub(r"e", "@SCHWA@", w)

    # 14. プレースホルダーを正式なIPAに展開
    replacements = {
        "@O@": "o",
        "@U@": "u",
        "@WA@": "wa",
        "@WE~@": "wɛ̃",
        "@UI@": "ɥi",
        "@EU@": "ø",
        "@OE_OPEN@": "œ",
        "@E_OPEN@": "ɛ",
        "@E_CLOSED@": "e",
        "@ER@": "e",
        "@JE~@": "jɛ̃",
        "@E~@": "ɛ̃",
        "@A~@": "ɑ̃",
        "@O~@": "ɔ̃",
        "@OE~@": "œ̃",
        "@J@": "j",
        "@IJ@": "ij",
        "@CH@": "ʃ",
        "@GN@": "ɲ",
        "@S_SOFT@": "s",
        "@Y_VOWEL@": "y",
        "@SCHWA@": "ə",
        "@HIATUS@": "",
    }
    for k, v in replacements.items():
        w = w.replace(k, v)

    # 15. 重複子音の整理
    w = re.sub(r"([bdfkɡlmnpʁstvzʃʒɲ])\1+", r"\1", w)
    w = re.sub(r"jj+", "j", w)

    return w, latent


def extract_phonemes(ipa_text):
    """IPA文字列から音素リストを抽出する（スペース・記号‿は除去）。"""
    text = ipa_text.replace("g", "ɡ").replace("r", "ʁ")
    return PHONEME_PATTERN.findall(text)


def analyze_line_phonetics(tokens, line_idx=1):
    """1行の単語トークン群から、リエゾンを考慮したIPA転写と音素リスト、リエゾン情報を返す。"""
    if not tokens:
        return {
            "line": line_idx,
            "ipa": "",
            "phonemes": [],
            "liaisons": [],
            "vowel_count": 0,
            "consonant_count": 0,
            "phoneme_count": 0,
        }

    token_info = []
    for t in tokens:
        w_lower = t.text.lower()
        ipa, latent = word_to_ipa(w_lower, pos=t.pos_)
        token_info.append({
            "token": t,
            "text": t.text,
            "lower": w_lower,
            "pos": t.pos_,
            "ipa": ipa,
            "latent": latent,
        })

    vowel_ipa_starts = set("aeiouyɑ̃ɛ̃ɔ̃œ̃əøœɔoɛé")
    result_words_ipa = []
    liaisons_detected = []

    for i in range(len(token_info)):
        curr = token_info[i]
        curr_ipa = curr["ipa"]
        has_liaison = False
        liaison_cons = None

        if i + 1 < len(token_info):
            nxt = token_info[i + 1]
            nxt_w = nxt["lower"]
            nxt_ipa = nxt["ipa"]

            # 後続語が母音（または無音h）で始まるか
            starts_with_vowel = False
            if nxt_w not in H_ASPIRE:
                if nxt_ipa and nxt_ipa[0] in vowel_ipa_starts:
                    starts_with_vowel = True
                elif nxt_w and (
                    nxt_w[0] in "aeiouyàâäéèêëîïôöûùüœæ" or (
                        nxt_w.startswith("h") and len(nxt_w) > 1 and nxt_w[1] in "aeiouyàâäéèêëîïôöûùü"
                    )
                ):
                    starts_with_vowel = True

            if starts_with_vowel and curr["latent"]:
                # 「et」は絶対にリエゾンしない
                if curr["lower"] != "et":
                    # 義務的・詩的リエゾン判定
                    is_obligatory = False
                    # 1. 限定詞
                    if curr["pos"] in ("DET", "NUM") or curr["lower"] in (
                        "les", "des", "ces", "mes", "tes", "ses", "nos", "vos", "leurs",
                        "un", "aucun", "deux", "trois", "cent", "vingt", "tout", "tous", "toutes", "quelques"
                    ):
                        is_obligatory = True
                    # 2. 代名詞 + 動詞/助動詞
                    elif curr["pos"] == "PRON" and nxt["pos"] in ("VERB", "AUX"):
                        is_obligatory = True
                    # 3. 前置詞・接続詞
                    elif curr["lower"] in ("dans", "sans", "sous", "en", "chez", "quand", "vers", "après", "avant", "pendant", "dès"):
                        is_obligatory = True
                    # 4. 形容詞 + 名詞
                    elif curr["pos"] == "ADJ" and nxt["pos"] == "NOUN":
                        is_obligatory = True
                    # 5. 副詞 + 形容詞/副詞
                    elif curr["lower"] in ("très", "trop", "bien", "plus", "moins", "assez") and nxt["pos"] in ("ADJ", "ADV"):
                        is_obligatory = True
                    # 6. 繋辞動詞 (c'est, est, sont, ont, vont, etc.)
                    elif curr["lower"] in ("est", "sont", "ont", "vont", "font", "était", "étaient", "fut"):
                        is_obligatory = True

                    if is_obligatory:
                        has_liaison = True
                        liaison_cons = curr["latent"]
                        liaisons_detected.append({
                            "line": line_idx,
                            "word1": curr["text"],
                            "word2": nxt["text"],
                            "consonant": liaison_cons,
                            "context": f"{curr['text']} ‿{liaison_cons}‿ {nxt['text']}",
                        })

        if has_liaison:
            formatted_word = f"{curr_ipa}‿{liaison_cons}‿ "
        elif curr["lower"].replace("’", "'").endswith("'") and i + 1 < len(token_info):
            formatted_word = curr_ipa
        else:
            formatted_word = f"{curr_ipa} "

        result_words_ipa.append(formatted_word)

    line_ipa = "".join(result_words_ipa).strip()
    line_phonemes = extract_phonemes(line_ipa)

    v_count = sum(1 for p in line_phonemes if p in VOWELS_ALL or p in GLIDES)
    c_count = sum(1 for p in line_phonemes if p in CONSONANTS)

    return {
        "line": line_idx,
        "ipa": line_ipa,
        "phonemes": line_phonemes,
        "liaisons": liaisons_detected,
        "vowel_count": v_count,
        "consonant_count": c_count,
        "phoneme_count": len(line_phonemes),
    }


def compute_phoneme_deviations(all_phonemes):
    """テクスト全体の音素リストから、標準フランス語分布に対する偏差を計算する。"""
    total = len(all_phonemes)
    if total == 0:
        return {
            "total_phonemes": 0,
            "phoneme_stats": [],
            "category_deviations": {},
            "top_deviations_positive": [],
            "top_deviations_negative": [],
            "vc_ratio": 0.0,
        }

    counts = collections.Counter(all_phonemes)

    phoneme_stats = []
    for ph, ref_data in FRENCH_PHONEME_REFERENCE.items():
        cnt = counts.get(ph, 0)
        pct = round(100.0 * cnt / total, 2)
        ref_mean = ref_data["mean"]
        ref_sd = ref_data["sd"]

        # 偏差計算
        diff = round(pct - ref_mean, 2)
        ratio = round(pct / ref_mean, 2) if ref_mean > 0 else 1.0
        z_score = round((pct - ref_mean) / ref_sd, 2)
        t_score = round(50 + 10 * z_score, 1)

        # 状態ラベル
        if z_score >= 2.0:
            status = "極めて多い（顕著）"
        elif z_score >= 1.0:
            status = "やや多い"
        elif z_score <= -2.0:
            status = "極めて少ない（抑制）"
        elif z_score <= -1.0:
            status = "やや少ない"
        else:
            status = "標準的"

        phoneme_stats.append({
            "phoneme": ph,
            "name": ref_data["name"],
            "category": ref_data["category"],
            "type": ref_data["type"],
            "count": cnt,
            "pct": pct,
            "ref_pct": ref_mean,
            "diff": diff,
            "ratio": ratio,
            "z_score": z_score,
            "t_score": t_score,
            "status": status,
        })

    # 偏差の大きい順（過剰出現・抑制）
    sorted_by_z = sorted(phoneme_stats, key=lambda x: x["z_score"], reverse=True)
    top_pos = [p for p in sorted_by_z if p["z_score"] > 0][:5]
    top_neg = [p for p in reversed(sorted_by_z) if p["z_score"] < 0][:5]

    # カテゴリ集計（母音、鼻母音、子音各分類）
    vowels = sum(counts[p] for p in VOWELS_ALL)
    glides = sum(counts[p] for p in GLIDES)
    consonants = sum(counts[p] for p in CONSONANTS)
    nasal_v = sum(counts[p] for p in VOWELS_NASAL)

    front_v = sum(counts[p] for p in FRONT_VOWELS)
    back_v = sum(counts[p] for p in BACK_VOWELS)
    close_v = sum(counts[p] for p in CLOSE_VOWELS)
    open_v = sum(counts[p] for p in OPEN_VOWELS)
    round_v = sum(counts[p] for p in ROUNDED_VOWELS)

    plosives = sum(counts[p] for p in PLOSIVES)
    fricatives = sum(counts[p] for p in FRICATIVES)
    liquids = sum(counts[p] for p in LIQUIDS)
    nasal_c = sum(counts[p] for p in NASAL_CONSONANTS)
    voiced_c = sum(counts[p] for p in VOICED_CONSONANTS)
    voiceless_c = sum(counts[p] for p in VOICELESS_CONSONANTS)

    def per_tot(x):
        return round(100.0 * x / total, 2)

    cat_dev = {
        "vowels_pct": per_tot(vowels),
        "glides_pct": per_tot(glides),
        "consonants_pct": per_tot(consonants),
        "nasal_vowels_pct": per_tot(nasal_v),
        # 母音内訳（全音素中の%）
        "front_vowels_pct": per_tot(front_v),
        "back_vowels_pct": per_tot(back_v),
        "close_vowels_pct": per_tot(close_v),
        "open_vowels_pct": per_tot(open_v),
        "rounded_vowels_pct": per_tot(round_v),
        # 子音内訳（全音素中の%）
        "plosives_pct": per_tot(plosives),
        "fricatives_pct": per_tot(fricatives),
        "liquids_pct": per_tot(liquids),
        "nasal_consonants_pct": per_tot(nasal_c),
        "voiced_consonants_pct": per_tot(voiced_c),
        "voiceless_consonants_pct": per_tot(voiceless_c),
        # 基準値との差分比較用
        "ref_vowels_pct": 43.5,
        "ref_glides_pct": 3.2,
        "ref_consonants_pct": 53.3,
        "ref_nasal_vowels_pct": 8.1,
        "ref_plosives_pct": 19.8,
        "ref_fricatives_pct": 15.3,
        "ref_liquids_pct": 17.3,
    }

    vc_ratio = round((vowels + glides) / consonants, 2) if consonants > 0 else 0.0

    return {
        "total_phonemes": total,
        "phoneme_stats": phoneme_stats,
        "category_deviations": cat_dev,
        "top_deviations_positive": top_pos,
        "top_deviations_negative": top_neg,
        "vc_ratio": vc_ratio,
    }
