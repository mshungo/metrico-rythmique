"""フランス語詩テクストの形式分析（コア処理）。UIに依存しない。"""
import re, json, math, collections, importlib, statistics as st
from pathlib import Path
from wordfreq import zipf_frequency, word_frequency
import phonetics
import prosody_rhythm
importlib.reload(phonetics)
importlib.reload(prosody_rhythm)
from phonetics import (
    analyze_line_phonetics,
    compute_phoneme_deviations,
    normalize_french_orthography,
    _is_silent_3pl_ent,
    H_ASPIRE,
)
from prosody_rhythm import analyze_prosody_and_rhythm

_NLP = None

def get_nlp():
    """spaCy モデルの遅延ロード（UI非依存）。"""
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("fr_core_news_md")
    return _NLP

VOWELS = "aeiouyàâäéèêëîïôöûùüœæ"
VOWEL_START = set(VOWELS + "h")
LIGHT_VERBS = {"être", "avoir", "faire", "dire", "pouvoir", "aller"}
ARTICLES = {
    "le", "la", "les", "l", "un", "une", "des",
    "du", "de", "d", "au", "aux"
}

# ---------- 音節（規則ベースの近似） ----------
def _groups(w, dierese=False):
    w = normalize_french_orthography(w.lower())
    if w in ("pays", "paysan", "paysans", "paysage", "paysages"):
        w = w.replace("ay", "ai-i")
    elif w in ("abbaye", "abbayes"):
        w = w.replace("aye", "ai-ie")
    w = re.sub(r"cueill?", "keil", w)
    w = re.sub(r"gueill?", "geil", w)
    w = re.sub(r"ge(?=[aàâoôuû])", "g", w)
    w = re.sub(r"qu", "q", w)
    w = re.sub(r"gu(?=[eéèêiy])", "g", w)
    w = w.replace("œu", "œ")
    w = re.sub(r"([aou])y(?=[aeiouyàâäéèêëîïôöûùüœæ])", r"\1-j", w)
    w = re.sub(r"(?<=[aeouœ])ill", "j", w)
    w = re.sub(r"([aeiou])([ïëü])", r"\1-\2", w)
    w = re.sub(r"(é)([aiouàâéèêô])", r"\1-\2", w)
    w = re.sub(r"([aoiu])(é)", r"\1-\2", w)
    if dierese:
        # 標準合音語（ciel, bien, rien, vient, pied, dieu, nuit, bruit, lui, suis, puis 等）は分音しない
        syn_protect = {
            "pied", "pieds", "ciel", "cieux", "miel", "fiel", "fier", "fiers", "vierge", "tiers",
            "pitié", "amitié", "moitié", "bien", "biens", "combien", "rien", "mien", "tien", "sien",
            "chien", "chiens", "chrétien", "chrétiens", "soutien", "vient", "tient", "souvient", "devient",
            "dieu", "dieux", "adieu", "lieu", "lieux", "milieu", "yeux", "vieux", "mieux", "diable",
            "nuit", "nuits", "bruit", "bruits", "fruit", "fruits", "pluie", "pluies", "ennui", "appui",
            "lui", "autrui", "celui", "suis", "puis", "fuir", "suivre", "cuire", "luire", "nuire",
        }
        if w not in syn_protect:
            w = re.sub(r"(?<![aeiouyàâäéèêëîïôöûùüœæ])i(?!ll)(?=[aouàâéèêô]|er\b|ez\b|el\b|en\b|è)", "i-", w)
            if re.search(r"(ruin|bruir|fluid|suicid|uité|uition)", w):
                w = re.sub(r"(?<![aeiouyàâäéèêëîïôöûùüœæqg])u(?=[aéèêiî])", "u-", w)
            else:
                w = re.sub(r"(?<![aeiouyàâäéèêëîïôöûùüœæqg])u(?=[aéèê])", "u-", w)
    return re.findall(f"[{VOWELS}]+", w)

def mute_final(w, pos=None, dierese=False):
    """語末のeが無音か。"""
    w_norm = normalize_french_orthography(w.lower())
    w_clean = re.sub(r"qu", "q", w_norm)
    w_clean = re.sub(r"gu(?=[eéèêiy])", "g", w_clean)
    if len(_groups(w_norm, dierese=dierese)) < 2:
        return False
    if re.search(r"[^aeiouéèêy]es?$", w_clean):
        return True
    if _is_silent_3pl_ent(w_norm, pos=pos):
        return True
    return False

def word_syllables(w, pos=None, dierese=False):
    """語末の無音eを数えない音節数。"""
    n = 0
    parts = [p for p in re.split(r"['’\-]", w.lower()) if p]
    for i, p in enumerate(parts):
        g = len(_groups(p, dierese=dierese))
        if g == 0:
            continue
        if mute_final(p, pos if i == len(parts) - 1 else None, dierese=dierese):
            g -= 1
        n += max(g, 1)
    return n

def line_syllables(words, dierese=False):
    """詩法に沿った近似：語末の無音eは次の語が子音（または有音h）で始まるときだけ数え、行末では数えない。"""
    if not words:
        return 0
    n = sum(word_syllables(t.text, t.pos_, dierese=dierese) for t in words)
    for t, nxt in zip(words, words[1:]):
        nxt_low = nxt.text.lower().strip("’'")
        nxt_is_vowel = (nxt_low[:1] in VOWEL_START) and (nxt_low not in H_ASPIRE)
        if mute_final(t.text, t.pos_, dierese=dierese) and not nxt_is_vowel:
            n += 1
    return n

# ---------- 文法 ----------
def is_word(t):
    return not (t.is_punct or t.is_space)

def is_finite(t):
    return "Fin" in t.morph.get("VerbForm")

def tense_label(t):
    mood, tense = t.morph.get("Mood"), t.morph.get("Tense")
    if "Imp" in mood: lab = "命令法"
    elif "Sub" in mood: lab = "接続法"
    elif "Cnd" in mood: lab = "条件法"
    elif "Fut" in tense: lab = "単純未来"
    elif "Imp" in tense: lab = "半過去"
    elif "Past" in tense: lab = "単純過去"
    elif "Pres" in tense: lab = "現在"
    else: lab = "その他"
    if t.dep_ == "aux:tense":  # 助動詞＋過去分詞
        lab = {"現在": "複合過去", "半過去": "大過去", "単純未来": "前未来", "条件法": "条件法過去"}.get(lab, lab)
    return lab

# ---------- 入力 ----------
def parse_text(text, fold_indented=False):
    """空行=連の区切り。'#'で始まる行は無視（ページ番号などのメモ用）。
    fold_indented=True の場合、字下げで始まる行を前の行の折り返しとみなして統合する。
    """
    raw_lines = text.splitlines()
    lines, stanza, new = [], 0, True

    for raw in raw_lines:
        # 空白行のチェック
        if not raw.strip():
            new = True
            continue
        # コメント行のチェック
        if raw.strip().startswith("#"):
            continue

        # 字下げ判定（先頭がスペースまたはタブ、かつ直前の行が存在し、新しい連の先頭でない）
        is_indented = len(raw) > len(raw.lstrip())
        if fold_indented and is_indented and lines and not new:
            lines[-1]["text"] += " " + raw.strip()
            continue

        if new:
            stanza += 1
        lines.append({"text": raw.strip(), "stanza": stanza, "stanza_start": new})
        new = False

    return lines

# ---------- 共通指標計算 ----------
def compute_corpus_metrics(docs, unit_words_list=None, dierese=False):
    """docs（文または行ごとの Doc リスト）から各種指標を算出する。
    unit_words_list: 各単位の単語トークンリストのリスト（None の場合は docs から抽出）。
    """
    if unit_words_list is None:
        unit_words_list = [[t for t in d if is_word(t)] for d in docs]

    words = [t for ws in unit_words_list for t in ws]
    W = len(words)
    num_units = len(docs)

    def per100(x):
        return round(100 * x / W, 2) if W > 0 else 0.0

    finite = [t for t in words if is_finite(t)]
    pos = collections.Counter(t.pos_ for t in words)
    person = collections.Counter(
        t.morph.get("Person")[0] + ("sg" if t.morph.get("Number") == ["Sing"] else "pl")
        for t in finite if t.morph.get("Person")
    )
    tenses = collections.Counter(tense_label(t) for t in finite)
    content = [t for t in words if t.pos_ in ("NOUN", "VERB", "ADJ", "ADV") and t.text.isalpha()]
    zipfs = [zipf_frequency(t.lower_, "fr") for t in content]
    wsyl = [word_syllables(t.text, t.pos_, dierese=dierese) for t in words if any(c.isalpha() for c in t.text)]
    forms = [t.lower_ for t in words]

    # MATTR（100語以上のときのみ計算、100語未満は None）
    mattr = None
    if W >= 100:
        mattr_vals = [len(set(forms[i:i + 100])) / 100 for i in range(0, len(forms) - 100 + 1, 5)]
        if mattr_vals:
            mattr = round(st.mean(mattr_vals), 4)

    # 近接反復：同じ名詞・動詞の見出し語が同一単位内で4語以内に再出現
    taut = 0
    for ws in unit_words_list:
        for i, t in enumerate(ws):
            if t.pos_ in ("NOUN", "VERB") and t.lemma_.lower() not in LIGHT_VERBS:
                if any(u.pos_ == t.pos_ and u.lemma_.lower() == t.lemma_.lower() for u in ws[i + 1:i + 5]):
                    taut += 1

    unit_lens = [len(ws) for ws in unit_words_list] if unit_words_list else [0]
    noverb_count = sum(not any(is_finite(t) for t in ws) for ws in unit_words_list)

    punct_count = sum(1 for d in docs for t in d if t.is_punct)

    metrics = {
        "punct_per100": per100(punct_count),
        "words_per_unit_mean": round(st.mean(unit_lens), 2) if unit_lens else 0.0,
        "noverb_unit_pct": round(100 * noverb_count / num_units, 2) if num_units > 0 else 0.0,
        "finite_per100": per100(len(finite)),
        "cconj_per100": per100(pos["CCONJ"]),
        "sconj_per100": per100(pos["SCONJ"]),
        "rel_per100": per100(sum(1 for t in words if "Rel" in t.morph.get("PronType"))),
        "word_syl_mean": round(st.mean(wsyl), 3) if wsyl else 0.0,
        "mono_pct": round(100 * sum(s == 1 for s in wsyl) / len(wsyl), 2) if wsyl else 0.0,
        "zipf_content_mean": round(st.mean(zipfs), 3) if zipfs else 0.0,
        "rare_lt3_pct": round(100 * sum(z < 3 for z in zipfs) / len(zipfs), 2) if zipfs else 0.0,
        "mattr100": mattr,
        "p1sg_pct": round(100 * person["1sg"] / max(1, sum(person.values())), 2) if person else 0.0,
        "taut_per1000": round(1000 * taut / W, 2) if W > 0 else 0.0,
        **{f"pos_{k}": round(100 * pos[k] / W, 2) if W > 0 else 0.0 for k in (
            "NOUN", "VERB", "AUX", "ADJ", "ADV", "DET", "PRON", "ADP", "CCONJ", "SCONJ", "PROPN", "NUM"
        )},
    }

    tense_total = sum(tenses.values())
    tense_share = {k: round(100 * v / tense_total, 2) for k, v in tenses.most_common()} if tense_total > 0 else {}

    content_count = sum(pos[k] for k in ("NOUN", "VERB", "ADJ", "ADV", "PROPN"))
    function_count = sum(pos[k] for k in ("DET", "PRON", "ADP", "CCONJ", "SCONJ", "AUX"))
    noun_count = pos.get("NOUN", 0) + pos.get("PROPN", 0)
    verb_count = pos.get("VERB", 0) + pos.get("AUX", 0)
    nominal_ratio = round(noun_count / max(verb_count, 1), 2)
    lexical_density = round(100.0 * content_count / max(W, 1), 1)
    if nominal_ratio >= 1.5:
        style_desc = "Style nominal (descriptif / pictural)"
    elif nominal_ratio <= 0.8:
        style_desc = "Style verbal (dynamique / narratif)"
    else:
        style_desc = "Style équilibré (nominal et verbal tempéré)"

    pos_analysis = {
        "counts": dict(pos),
        "total_words": W,
        "content_words": content_count,
        "function_words": function_count,
        "lexical_density": lexical_density,
        "nominal_ratio": nominal_ratio,
        "style_desc": style_desc,
    }

    return {
        "metrics": metrics,
        "tense_share": tense_share,
        "pos_analysis": pos_analysis,
        "words": words,
        "W": W,
    }

# ---------- 定型句・行頭反復検出 ----------
def detect_patterns(lines, docs):
    """定型句・行頭反復を検出する。
    1. 「elle disait X」直後「et nous…」
    2. 「C'était hier/ailleurs」直後「il y a…」
    3. 行頭反復（冠詞を除く同一語で連続する行）
    4. 従来の「X pour ne pas dire Y」
    """
    joined = "\n".join(l["text"] for l in lines)
    pour_ne_pas_dire = [
        {"枠": a, "言った語": b.strip(), "言い直し": c.strip()}
        for a, b, c in re.findall(r"(je dis|elle dit|il dit) (.+?) pour ne pas dire (.+)", joined, re.IGNORECASE)
    ]

    disait_et_nous = []
    hier_il_y_a = []
    anaphora_groups = []

    num_lines = len(lines)
    for i in range(num_lines - 1):
        t1 = lines[i]["text"]
        t2 = lines[i + 1]["text"]

        # 1. 「elle disait X」直後「et nous…」
        m1 = re.search(r"^[Ee]lle disait\b\s*(.*)", t1)
        m2 = re.search(r"^[Ee]t nous\b\s*(.*)", t2)
        if m1 and m2:
            disait_et_nous.append({
                "line_pair": f"{i + 1}-{i + 2}",
                "line1": i + 1,
                "text1": t1,
                "line2": i + 2,
                "text2": t2,
            })

        # 2. 「C'était hier/ailleurs」直後「il y a…」
        m_hier = re.search(r"^[Cc]['’]était\s+(hier|ailleurs)\b", t1, re.IGNORECASE)
        m_ilya = re.search(r"^[Ii]l y a\b", t2, re.IGNORECASE)
        if m_hier and m_ilya:
            hier_il_y_a.append({
                "line_pair": f"{i + 1}-{i + 2}",
                "line1": i + 1,
                "text1": t1,
                "line2": i + 2,
                "text2": t2,
            })

    # 3. 同じ語で始まる連続行（行頭反復：anaphora、冠詞は除く）
    first_words = []
    for l, d in zip(lines, docs):
        ws = [t for t in d if is_word(t)]
        if ws:
            w0 = ws[0].text.lower()
            w0_clean = re.split(r"['’]", w0)[0] if any(w0.startswith(art) for art in ("l'", "l’", "d'", "d’")) else w0
            first_words.append((ws[0].text.lower(), w0_clean))
        else:
            first_words.append(("", ""))

    curr_word = None
    curr_lines = []
    for idx, (raw_w, clean_w) in enumerate(first_words, 1):
        is_article = clean_w in ARTICLES or raw_w in ARTICLES
        effective_word = raw_w if not is_article else None

        if effective_word and curr_word and effective_word == curr_word:
            curr_lines.append(idx)
        else:
            if len(curr_lines) >= 2:
                anaphora_groups.append({
                    "word": curr_word,
                    "count": len(curr_lines),
                    "lines": f"{curr_lines[0]}-{curr_lines[-1]}" if len(curr_lines) > 2 else f"{curr_lines[0]}, {curr_lines[1]}",
                    "line_list": list(curr_lines),
                })
            curr_word = effective_word
            curr_lines = [idx] if effective_word else []

    if len(curr_lines) >= 2:
        anaphora_groups.append({
            "word": curr_word,
            "count": len(curr_lines),
            "lines": f"{curr_lines[0]}-{curr_lines[-1]}" if len(curr_lines) > 2 else f"{curr_lines[0]}, {curr_lines[1]}",
            "line_list": list(curr_lines),
        })

    return {
        "pour_ne_pas_dire": pour_ne_pas_dire,
        "disait_et_nous": disait_et_nous,
        "hier_il_y_a": hier_il_y_a,
        "anaphora": anaphora_groups,
    }

# ---------- 分析メイン ----------
def analyze(text, dierese=False, fold_indented=False, nlp=None):
    """詩テクストを解析し、行データ・指標・定型句・音素分析（IPA・リエゾン・偏差）などを返す。"""
    if nlp is None:
        nlp = get_nlp()

    lines = parse_text(text, fold_indented=fold_indented)
    if not lines:
        return {
            "rows": [], "metrics": {}, "lines_summary": {},
            "tense_share": {}, "keyness": [], "formulas": [],
            "extended_patterns": {
                "pour_ne_pas_dire": [], "disait_et_nous": [],
                "hier_il_y_a": [], "anaphora": []
            },
            "pos_analysis": {
                "counts": {}, "total_words": 0, "content_words": 0,
                "function_words": 0, "lexical_density": 0.0,
                "nominal_ratio": 0.0, "style_desc": "Texte vide"
            },
            "phonetics": {
                "lines": [],
                "deviations": compute_phoneme_deviations([]),
                "all_liaisons": [],
                "total_liaisons": 0,
            },
            "prosody_rhythm": analyze_prosody_and_rhythm([], [], [], [], dierese=dierese),
        }

    docs = list(nlp.pipe([l["text"] for l in lines]))
    unit_words = [[t for t in d if is_word(t)] for d in docs]

    # 音素分析（行ごとのIPAと音素、リエゾン、偏差計算）
    line_phonetics_list = []
    all_text_phonemes = []
    all_liaisons = []

    for i, ws in enumerate(unit_words, 1):
        lp = analyze_line_phonetics(ws, line_idx=i)
        line_phonetics_list.append(lp)
        all_text_phonemes.extend(lp["phonemes"])
        all_liaisons.extend(lp["liaisons"])

    phoneme_deviations = compute_phoneme_deviations(all_text_phonemes)

    # 4大理論統合・韻律リズム分析 (Cornulier / Meschonnic / Di Cristo / Gouvard)
    pr_result = analyze_prosody_and_rhythm(
        lines, docs, unit_words, line_phonetics_list, dierese=dierese
    )
    pr_lines = pr_result["lines_detail"]

    rows = []
    for i, (l, d, ws, lp, pr_l) in enumerate(zip(lines, docs, unit_words, line_phonetics_list, pr_lines), 1):
        ws_pos = [t.pos_ for t in ws]
        pos_cnts = dict(collections.Counter(ws_pos))
        rhyme_pos = ws[-1].pos_ if ws else ""
        content_in_line = sum(pos_cnts.get(k, 0) for k in ("NOUN", "VERB", "ADJ", "ADV", "PROPN"))
        func_in_line = len(ws) - content_in_line

        rows.append({
            "line": i, "stanza": l["stanza"], "stanza_start": l["stanza_start"],
            "syllables": pr_l["anatonic_syllables"],
            "anatonic_syllables": pr_l["anatonic_syllables"],
            "words": len(ws),
            "has_finite_verb": any(is_finite(t) for t in ws),
            "feminine": mute_final(ws[-1].text, ws[-1].pos_, dierese=dierese) if ws else False,
            "capital": next((c for c in l["text"] if c.isalpha()), "").isupper(),
            "ends_function_word": bool(ws) and ws[-1].pos_ in ("DET", "ADP", "CCONJ", "SCONJ"),
            "meter_code": pr_l["meter_code"],
            "caesura_code": pr_l["caesura_code"],
            "measures": pr_l["measures_str"],
            "rhyme_letter": pr_l["rhyme_letter"],
            "rhyme_ipa": pr_l["rhyme_ipa"],
            "rhyme_richness": pr_l["rhyme_richness"],
            "rhyme_pos": rhyme_pos,
            "meschonnic_scansion": pr_l["meschonnic_scansion"],
            "di_cristo_phrasing": pr_l["di_cristo_phrasing"],
            "ipa": lp["ipa"],
            "phonemes": lp["phoneme_count"],
            "vowels": lp["vowel_count"],
            "consonants": lp["consonant_count"],
            "phoneme_density": round(lp["phoneme_count"] / max(pr_l["anatonic_syllables"], 1), 2) if pr_l["anatonic_syllables"] > 0 else 0.0,
            "vc_ratio": round(lp["vowel_count"] / max(lp["consonant_count"], 1), 2) if lp["consonant_count"] > 0 else 0.0,
            "vowel_pct": round((lp["vowel_count"] / max(lp["phoneme_count"], 1)) * 100, 1) if lp["phoneme_count"] > 0 else 0.0,
            "consonant_pct": round((lp["consonant_count"] / max(lp["phoneme_count"], 1)) * 100, 1) if lp["phoneme_count"] > 0 else 0.0,
            "liaisons": len(lp["liaisons"]),
            "text": l["text"],
            "word_tokens": [{"text": t.text, "lemma": t.lemma_, "pos": t.pos_} for t in ws],
            "pos_counts": pos_cnts,
            "content_words": content_in_line,
            "function_words": func_in_line,
        })

    computed = compute_corpus_metrics(docs, unit_words_list=unit_words, dierese=dierese)
    metrics = computed["metrics"]
    tense_share = computed["tense_share"]
    words = computed["words"]
    W = computed["W"]

    sylls = [r["syllables"] for r in rows] if rows else [0]
    stanza_starts = [r["capital"] for r in rows if r["stanza_start"]]
    non_stanza_starts = [r["capital"] for r in rows if not r["stanza_start"]]

    lines_summary = {
        "行数": len(rows),
        "連の数": len({r["stanza"] for r in rows}) if rows else 0,
        "語数": W,
        "総音素数": len(all_text_phonemes),
        "検出リエゾン数": len(all_liaisons),
        "1行の音節（平均）": round(st.mean(sylls), 2) if sylls else 0.0,
        "1行の音節（中央値）": st.median(sylls) if sylls else 0,
        "最短行（音節）": min(sylls) if sylls else 0,
        "最長行（音節）": max(sylls) if sylls else 0,
        "12音節の行（%）": round(100 * sylls.count(12) / len(sylls), 1) if sylls else 0.0,
        "機能語で終わる行": sum(r["ends_function_word"] for r in rows),
        "女性韻の行（%）": round(100 * sum(r["feminine"] for r in rows) / len(rows), 1) if rows else 0.0,
        "連頭行の大文字率（%）": round(100 * st.mean(stanza_starts), 1) if stanza_starts else 0.0,
        "連の途中の大文字率（%）": round(100 * st.mean(non_stanza_starts), 1) if non_stanza_starts else 0.0,
    }

    # 特徴語：log2（詩での出現率 ÷ 一般頻度）、4回以上
    lc = collections.Counter(t.lemma_.lower() for t in words if t.pos_ in ("NOUN", "VERB", "ADJ") and t.lemma_.isalpha())
    keyness = sorted(
        ({"lemma": l, "count": n,
          "log2_ratio": round(math.log2((n / W) / (word_frequency(l, "fr") or 1e-8)), 2),
          "zipf": zipf_frequency(l, "fr")} for l, n in lc.items() if n >= 4),
        key=lambda k: -k["log2_ratio"])[:20] if W > 0 else []

    patterns = detect_patterns(lines, docs)

    return {
        "rows": rows,
        "metrics": metrics,
        "pos_analysis": computed["pos_analysis"],
        "lines_summary": lines_summary,
        "tense_share": tense_share,
        "keyness": keyness,
        "formulas": patterns["pour_ne_pas_dire"],
        "extended_patterns": patterns,
        "phonetics": {
            "lines": line_phonetics_list,
            "deviations": phoneme_deviations,
            "all_liaisons": all_liaisons,
            "total_liaisons": len(all_liaisons),
        },
        "prosody_rhythm": pr_result,
    }

def load_reference(path=Path(__file__).with_name("reference.json")):
    return json.loads(Path(path).read_text(encoding="utf-8"))
