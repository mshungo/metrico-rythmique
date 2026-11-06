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
def _groups(w, dierese=False, ancient_postvocalic_e=False):
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
    if ancient_postvocalic_e:
        w = re.sub(r"(?<=[aeiouyàâäéèêëîïôöûùüœæ])(?=es?$)", "-", w)
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
    """語末のeが無音子音節か（table, femme等の無音e減算用）。"""
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

def word_syllables(w, pos=None, dierese=False, ancient_postvocalic_e=False):
    """語末の無音eを数えない音節数。"""
    n = 0
    parts = [p for p in re.split(r"['’\-]", w.lower()) if p]
    for i, p in enumerate(parts):
        g = len(_groups(p, dierese=dierese, ancient_postvocalic_e=ancient_postvocalic_e))
        if g == 0:
            continue
        if mute_final(p, pos if i == len(parts) - 1 else None, dierese=dierese):
            g -= 1
        n += max(g, 1)
    return n

def line_syllables(words, dierese=False, ancient_postvocalic_e=False):
    """詩法に沿った近似：語末の無音eは次の語が子音（または有音h）で始まるときだけ数え、行末では数えない。"""
    if not words:
        return 0
    n = sum(word_syllables(t.text, t.pos_, dierese=dierese, ancient_postvocalic_e=ancient_postvocalic_e) for t in words)
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
def compute_corpus_metrics(docs, unit_words_list=None, dierese=False, ancient_postvocalic_e=False):
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
    wsyl = [word_syllables(t.text, t.pos_, dierese=dierese, ancient_postvocalic_e=ancient_postvocalic_e) for t in words if any(c.isalpha() for c in t.text)]
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

# ---------- 論文・研究引用向け音素動態解析 ----------
def compute_phoneme_progression_dynamics(track_records, smoothing_window=3):
    """Calcule la trajectoire, la tendance lissée (moyenne mobile), la régression linéaire et
    la répartition strophique pour chaque phonème ou chaîne, afin de permettre une citation
    académique rigoureuse (thèses, articles de stylistique/linguistique).
    """
    if not track_records:
        return {"profiles": {}, "enriched_records": [], "stanza_boundaries": []}

    # Détection des frontières de strophes
    stanza_boundaries = []
    seen_stanzas = set()
    for r in track_records:
        st_num = r.get("stanza", 1)
        l_num = r.get("line", 1)
        if st_num not in seen_stanzas and l_num > 1:
            stanza_boundaries.append({"line": l_num, "stanza": st_num})
            seen_stanzas.add(st_num)
        seen_stanzas.add(st_num)

    # Regroupement par phonème
    ph_groups = collections.defaultdict(list)
    for r in track_records:
        ph_groups[r["phoneme"]].append(r)

    profiles = {}
    enriched_records = []

    for ph, rows in ph_groups.items():
        n = len(rows)
        if n == 0:
            continue

        rows_sorted = sorted(rows, key=lambda x: x["line"])
        xs = [float(r["line"]) for r in rows_sorted]
        ys = [float(r.get("count", 0)) for r in rows_sorted]
        y_norms = [float(r.get("norm_12", 0.0)) for r in rows_sorted]

        # Calcul de la moyenne mobile centrée
        w_rad = max(1, smoothing_window // 2)
        rolling_counts = []
        rolling_norms = []
        for i in range(n):
            i_start = max(0, i - w_rad)
            i_end = min(n, i + w_rad + 1)
            window_slice_c = ys[i_start:i_end]
            window_slice_n = y_norms[i_start:i_end]
            rolling_counts.append(round(sum(window_slice_c) / len(window_slice_c), 2))
            rolling_norms.append(round(sum(window_slice_n) / len(window_slice_n), 2))

        for r_dict, r_c, r_n in zip(rows_sorted, rolling_counts, rolling_norms):
            r_copy = dict(r_dict)
            r_copy["rolling_count"] = r_c
            r_copy["rolling_norm_12"] = r_n
            enriched_records.append(r_copy)

        tot_c = sum(ys)
        mean_c = round(tot_c / n, 2) if n > 0 else 0.0
        max_c = max(ys) if ys else 0
        peak_lines = [int(r["line"]) for r in rows_sorted if r.get("count", 0) == max_c and max_c > 0]

        mean_x = sum(xs) / n if n > 0 else 0.0
        mean_y = sum(ys) / n if n > 0 else 0.0
        var_x = sum((x - mean_x) ** 2 for x in xs)
        cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        var_y = sum((y - mean_y) ** 2 for y in ys)

        slope = round(cov_xy / var_x, 3) if var_x > 0 else 0.0
        r_val = round(cov_xy / math.sqrt(var_x * var_y), 2) if (var_x * var_y) > 0 else 0.0

        st_dict = collections.defaultdict(list)
        for r in rows_sorted:
            st_dict[r.get("stanza", 1)].append(r)

        strophes_summary = []
        for st_idx, st_rows in sorted(st_dict.items()):
            st_tot = sum(r.get("count", 0) for r in st_rows)
            st_lines = [r["line"] for r in st_rows]
            st_mean = round(st_tot / len(st_rows), 2) if st_rows else 0.0
            st_pct = round(100.0 * st_tot / tot_c, 1) if tot_c > 0 else 0.0
            span_str = f"v. {st_lines[0]}–{st_lines[-1]}" if len(st_lines) > 1 else f"v. {st_lines[0]}"
            strophes_summary.append({
                "stanza": st_idx,
                "lines_span": span_str,
                "count": int(st_tot),
                "pct": st_pct,
                "mean_per_line": st_mean,
            })

        if r_val >= 0.30 and slope > 0.03:
            dyn_code = "crescendo"
            dyn_label = "Renforcement progressif (Crescendo ↗)"
            badge_color = "#16a34a"
        elif r_val <= -0.30 and slope < -0.03:
            dyn_code = "decrescendo"
            dyn_label = "Atténuation progressive (Decrescendo ↘)"
            badge_color = "#dc2626"
        elif peak_lines and any(n * 0.20 <= pl <= n * 0.80 for pl in peak_lines) and max_c >= max(mean_c * 1.5, 2):
            dyn_code = "climax"
            dyn_label = "Apogée médian (Concentration centrale ∧)"
            badge_color = "#7c3aed"
        else:
            dyn_code = "diffuse"
            dyn_label = "Trame diffuse / Équilibrée (↔)"
            badge_color = "#2563eb"

        peak_str = ", ".join(f"v. {pl}" for pl in peak_lines) if peak_lines else "aucun"
        peak_text = next((r["text"] for r in rows_sorted if r["line"] in peak_lines), "") if peak_lines else ""

        if dyn_code == "crescendo":
            comm = (
                f"Le phonème {ph} manifeste une dynamique d'intensification progressive au fil du poème "
                f"(pente de régression linéaire β = {slope:+.3f}, coefficient de corrélation r = {r_val:+.2f}). "
                f"Avec un effectif total de {int(tot_c)} occurrences (moyenne de {mean_c:.2f} occ./vers), sa présence s'accroît "
                f"pour atteindre son point culminant au {peak_str} ({int(max_c)} occurrences, ex. « {peak_text} »). "
            )
        elif dyn_code == "decrescendo":
            comm = (
                f"Le phonème {ph} présente une dynamique d'atténuation continue au cours du poème "
                f"(pente de régression linéaire β = {slope:+.3f}, coefficient de corrélation r = {r_val:+.2f}). "
                f"Initialement saillant en tête de texte, son intensité décroît progressivement "
                f"(moyenne globale : {mean_c:.2f} occ./vers, pic initial au {peak_str} avec {int(max_c)} occurrences). "
            )
        elif dyn_code == "climax":
            comm = (
                f"Le phonème {ph} s'organise selon une structure en arche avec une forte concentration médiane "
                f"(apogée au {peak_str} culminant à {int(max_c)} occurrences, contre une moyenne générale de {mean_c:.2f} occ./vers). "
                f"Cette focalisation sonore constitue un pivot expressif au coeur de l'architecture poétique (ex. « {peak_text} »). "
            )
        else:
            comm = (
                f"Le phonème {ph} présente une distribution équilibrée et continue sur l'ensemble du poème "
                f"(pente β = {slope:+.3f}, r = {r_val:+.2f}, moyenne = {mean_c:.2f} occ./vers, pic au {peak_str} avec {int(max_c)} occ.). "
                f"Il assure un tissage phonique stable sans rupture abrupte d'intensité. "
            )

        if strophes_summary and len(strophes_summary) > 1:
            st_text_parts = [
                f"Strophe {s['stanza']} ({s['lines_span']}) : {s['count']} occ. ({s['pct']} %, moy. {s['mean_per_line']}/v.)"
                for s in strophes_summary
            ]
            comm += "Évolution strophique : " + " ｜ ".join(st_text_parts) + "."

        latex_lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{Modulation strophique du phonème {ph} (pente $\beta = {slope:+.3f}$, $r = {r_val:+.2f}$)}}",
            rf"\label{{tab:dyn_{ph.strip('[]')}}}",
            r"\begin{tabular}{lcccc}",
            r"\hline",
            r"Strophe & Vers & Occurrences & Part (\%) & Moyenne/vers \\",
            r"\hline",
        ]
        for s in strophes_summary:
            latex_lines.append(f"Strophe {s['stanza']} & {s['lines_span']} & {s['count']} & {s['pct']}\\% & {s['mean_per_line']} \\\\")
        latex_lines.extend([
            r"\hline",
            rf"\textbf{{Total}} & \textbf{{1--{n}}} & \textbf{{{int(tot_c)}}} & \textbf{{100.0\%}} & \textbf{{{mean_c}}} \\\\",
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ])
        latex_table = "\n".join(latex_lines)

        md_lines = [
            f"| Strophe | Vers | Occurrences | Part (%) | Moyenne / vers |",
            f"| :--- | :--- | :---: | :---: | :---: |",
        ]
        for s in strophes_summary:
            md_lines.append(f"| Strophe {s['stanza']} | {s['lines_span']} | {s['count']} | {s['pct']} % | {s['mean_per_line']} |")
        md_lines.append(f"| **Total** | **1–{n}** | **{int(tot_c)}** | **100.0 %** | **{mean_c}** |")
        md_table = "\n".join(md_lines)

        profiles[ph] = {
            "phoneme": ph,
            "total_count": int(tot_c),
            "mean_per_line": mean_c,
            "max_count": int(max_c),
            "peak_lines": peak_lines,
            "slope": slope,
            "correlation_r": r_val,
            "dyn_code": dyn_code,
            "dyn_label": dyn_label,
            "badge_color": badge_color,
            "academic_citation_fr": comm,
            "strophes_summary": strophes_summary,
            "latex_table": latex_table,
            "markdown_table": md_table,
        }

    return {
        "profiles": profiles,
        "enriched_records": enriched_records,
        "stanza_boundaries": stanza_boundaries,
    }

def extract_salient_phonetic_patterns(rows, lines_phonetics, mesch_data=None, min_ngram_lines=None):
    """
    詩テクスト全体から、反復音素・顕著な連鎖・特定のパターン（アリテラシオン、アソナンス、
    音素連鎖網、キアスム・反響図式、プロゾディー最上級語、反復N-gram）を自動的に拾い集めて集約・分類する。
    """
    n_lines = len(rows)
    if n_lines == 0 or len(lines_phonetics) == 0:
        return {
            "all_patterns": [],
            "summary_by_category": {},
            "categories": [],
            "recurrent_ngrams": [],
            "stats": {
                "total_patterns": 0,
                "total_alliterations": 0,
                "total_assonances": 0,
                "total_chains": 0,
                "total_echos": 0,
                "total_superlatifs": 0,
                "total_recurrent_ngrams": 0,
            },
        }

    all_patterns = []

    # 1. 行内アリテラシオン（顕著な子音反復） & アソナンス（顕著な母音反復）
    for idx, (r, lp) in enumerate(zip(rows, lines_phonetics), 1):
        line_num = r.get("line", idx)
        stanza_num = r.get("stanza", 1)
        line_text = r.get("text", "")
        line_ipa = lp.get("ipa", "")
        phs = lp.get("phonemes", [])

        # 単語とその音素を事前分解
        words_tokens = r.get("word_tokens", [])
        word_phs_map = []
        for wt in words_tokens:
            w_text = wt.get("text", "")
            if not any(c.isalnum() for c in w_text):
                continue
            ipa_w, _ = phonetics.word_to_ipa(w_text)
            w_phs = phonetics.extract_phonemes(ipa_w)
            word_phs_map.append({"text": w_text, "phonemes": w_phs, "ipa": ipa_w})

        # 子音頻度集計（半母音/渡り音 j, w, ɥ を除く）
        cons_in_line = [p for p in phs if p in phonetics.CONSONANTS and p not in ("j", "w", "ɥ")]
        cons_counts = collections.Counter(cons_in_line)
        # 短い詩句（6音節以下）は2回以上、通常詩句は3回以上
        min_cons = 2 if r.get("syllables", 12) <= 6 else 3

        for c, cnt in cons_counts.items():
            if cnt >= min_cons:
                matching_words = [wp["text"] for wp in word_phs_map if c in wp["phonemes"]]
                # 複数単語にまたがるか、または同一単語内でも3回以上
                if len(matching_words) >= 2 or cnt >= 3:
                    all_patterns.append({
                        "id": f"allit_L{line_num}_{c}",
                        "line": line_num,
                        "stanza": stanza_num,
                        "category": "alliteration",
                        "category_label": "Allitération",
                        "motif": f"[{c}]",
                        "motif_display": f"[{c}] × {cnt}",
                        "count": cnt,
                        "description": f"Allitération saillante : {cnt} occurrences de la consonne [{c}]",
                        "words_involved": matching_words,
                        "words_str": ", ".join(matching_words) if matching_words else "",
                        "line_text": line_text,
                        "ipa_text": line_ipa,
                    })

        # 母音頻度集計（シュワ [ə] を除く）
        vow_in_line = [p for p in phs if p in phonetics.VOWELS_ALL and p != "ə"]
        vow_counts = collections.Counter(vow_in_line)
        min_vow = 2 if r.get("syllables", 12) <= 6 else 3

        for v, cnt in vow_counts.items():
            if cnt >= min_vow:
                matching_words = [wp["text"] for wp in word_phs_map if v in wp["phonemes"]]
                if len(matching_words) >= 2 or cnt >= 3:
                    all_patterns.append({
                        "id": f"asson_L{line_num}_{v}",
                        "line": line_num,
                        "stanza": stanza_num,
                        "category": "assonance",
                        "category_label": "Assonance",
                        "motif": f"[{v}]",
                        "motif_display": f"[{v}] × {cnt}",
                        "count": cnt,
                        "description": f"Assonance saillante : {cnt} occurrences du timbre vocalique [{v}]",
                        "words_involved": matching_words,
                        "words_str": ", ".join(matching_words) if matching_words else "",
                        "line_text": line_text,
                        "ipa_text": line_ipa,
                    })

    # 2. Meschonnic 音素連鎖網 (active_chains)
    if mesch_data and "lines" in mesch_data:
        for idx, (m_line, r, lp) in enumerate(zip(mesch_data["lines"], rows, lines_phonetics), 1):
            line_num = r.get("line", idx)
            stanza_num = r.get("stanza", 1)
            line_text = r.get("text", "")
            line_ipa = lp.get("ipa", "")

            for ch in m_line.get("active_chains", []):
                kind = ch.get("kind", "")
                if kind in ("serie_consonantique", "couplage"):
                    ph = ch.get("phoneme", "")
                    span_desc = ch.get("span_desc", "")
                    pos_cnt = len(ch.get("positions", []))
                    words_in = [w.split(":")[-1] for w in span_desc.split(" ↔ ") if ":" in w]
                    kind_fr = "Série consonantique" if kind == "serie_consonantique" else "Couplage prosodique"
                    all_patterns.append({
                        "id": f"chain_L{line_num}_{ph}_{abs(hash(span_desc)) % 10000}",
                        "line": line_num,
                        "stanza": stanza_num,
                        "category": "chaine",
                        "category_label": "Chaîne phonématique",
                        "motif": f"[{ph}]",
                        "motif_display": f"{kind_fr} [{ph}]",
                        "count": pos_cnt,
                        "description": f"Chaîne de consonnes ouvrantes ({pos_cnt} points d'appui) : {span_desc}",
                        "words_involved": words_in,
                        "words_str": ", ".join(words_in) if words_in else span_desc,
                        "line_text": line_text,
                        "ipa_text": line_ipa,
                    })

    # 3. Meschonnic 反響図式 & キアスム (echo_figures)
    if mesch_data and "echo_figures" in mesch_data:
        figure_labels = {
            "chiasme_prosodique": "Chiasme prosodique / Écho renversé",
            "echo_initial": "Écho initial (attaques C(C)V identiques)",
            "echo_diminue": "Écho diminué",
            "echo_augmente": "Écho augmenté",
            "rebond_de_consonnes": "Rebond de consonnes (schwa intercalaire)",
            "hiatus_de_consonnes": "Hiatus de consonnes (coda-attaque)",
        }
        for fig in mesch_data["echo_figures"]:
            fig_type = fig.get("figure", "")
            if fig_type == "superlatif_prosodique":
                continue
            l_num = fig.get("line", 1)
            r_item = next((r for r in rows if r.get("line") == l_num), {})
            lp_item = next((lp for lp in lines_phonetics if lp.get("line") == l_num), {})
            st_num = r_item.get("stanza", 1)
            ex_txt = fig.get("example", "")
            words_in = [w.strip() for w in re.split(r"[↔→#]", ex_txt) if w.strip()]
            desc_label = figure_labels.get(fig_type, fig.get("label", fig_type))
            all_patterns.append({
                "id": f"echo_L{l_num}_{fig_type}_{abs(hash(ex_txt)) % 10000}",
                "line": l_num,
                "stanza": st_num,
                "category": "chiasme_echo",
                "category_label": "Chiasme & Écho",
                "motif": fig.get("label", fig_type),
                "motif_display": fig.get("label", fig_type),
                "count": 2,
                "description": f"{desc_label} : {ex_txt}",
                "words_involved": words_in,
                "words_str": ex_txt,
                "line_text": r_item.get("text", ""),
                "ipa_text": lp_item.get("ipa", ""),
            })

    # 4. Superlatifs prosodiques（高密度凝縮語）
    if mesch_data and "superlatifs_prosodiques" in mesch_data:
        for sup in mesch_data["superlatifs_prosodiques"]:
            l_num = sup.get("line", 1)
            r_item = next((r for r in rows if r.get("line") == l_num), {})
            st_num = r_item.get("stanza", 1)
            c_ph = sup.get("consonant", "")
            w_text = sup.get("word", "")
            c_cnt = sup.get("count_in_word", 2)
            all_patterns.append({
                "id": f"sup_L{l_num}_{w_text}_{c_ph}",
                "line": l_num,
                "stanza": st_num,
                "category": "superlatif",
                "category_label": "Superlatif prosodique",
                "motif": f"[{c_ph}] × {c_cnt}",
                "motif_display": f"Superlatif [{c_ph}] × {c_cnt}",
                "count": c_cnt,
                "description": f"Concentration phonématique maximale de [{c_ph}] dans le mot « {w_text} » [{sup.get('ipa','')}]",
                "words_involved": [w_text],
                "words_str": f"{w_text} [{sup.get('ipa', '')}]",
                "line_text": r_item.get("text", ""),
                "ipa_text": sup.get("ipa", ""),
            })

    # 5. N-grammes phonétiques récurrents (詩行をまたいで反復されるバイグラム・トライグラム)
    ngram_line_map = collections.defaultdict(lambda: collections.defaultdict(int))
    ngram_corpus_cnt = collections.defaultdict(int)

    for idx, lp in enumerate(lines_phonetics, 1):
        line_num = idx
        phs = [p for p in lp.get("phonemes", []) if p in phonetics.PHONEMES_ALL and p != "ə"]
        # バイグラム (2連音素)
        for i in range(len(phs) - 1):
            p1, p2 = phs[i], phs[i+1]
            if p1 in phonetics.CONSONANTS or p2 in phonetics.CONSONANTS:
                bg = f"{p1}{p2}"
                ngram_line_map[bg][line_num] += 1
                ngram_corpus_cnt[bg] += 1
        # トライグラム (3連音素)
        for i in range(len(phs) - 2):
            p1, p2, p3 = phs[i], phs[i+1], phs[i+2]
            if any(p in phonetics.CONSONANTS for p in (p1, p2, p3)):
                tg = f"{p1}{p2}{p3}"
                ngram_line_map[tg][line_num] += 1
                ngram_corpus_cnt[tg] += 1

    if min_ngram_lines is None:
        min_lines_req = 2 if n_lines <= 8 else 3
    else:
        min_lines_req = min_ngram_lines

    recurrent_ngrams = []
    for ng, l_map in ngram_line_map.items():
        total_c = ngram_corpus_cnt[ng]
        if len(l_map) >= min_lines_req or (len(l_map) >= 2 and total_c >= 3):
            recurrent_ngrams.append({
                "ngram": ng,
                "motif": f"[{ng}]",
                "lines_count": len(l_map),
                "total_count": total_c,
                "lines": sorted(list(l_map.keys())),
            })

    recurrent_ngrams.sort(key=lambda x: (x["lines_count"], x["total_count"], len(x["ngram"])), reverse=True)

    for rng in recurrent_ngrams[:12]:
        ng_str = rng["ngram"]
        for l_num in rng["lines"]:
            r_item = next((r for r in rows if r.get("line") == l_num), {})
            lp_item = next((lp for lp in lines_phonetics if lp.get("line") == l_num), {})
            st_num = r_item.get("stanza", 1)
            l_occ = ngram_line_map[ng_str][l_num]
            w_tokens = r_item.get("word_tokens", [])
            w_matches = []
            for wt in w_tokens:
                wt_ipa, _ = phonetics.word_to_ipa(wt.get("text", ""))
                if ng_str in wt_ipa:
                    w_matches.append(wt.get("text", ""))

            all_patterns.append({
                "id": f"ngram_L{l_num}_{ng_str}",
                "line": l_num,
                "stanza": st_num,
                "category": "ngram",
                "category_label": "N-gramme récurrent",
                "motif": f"[{ng_str}]",
                "motif_display": f"[{ng_str}] ({rng['lines_count']} vers)",
                "count": l_occ,
                "description": f"Motif trans-strophique [{ng_str}] répété dans {rng['lines_count']} vers ({l_occ} fois dans ce vers)",
                "words_involved": w_matches,
                "words_str": ", ".join(w_matches) if w_matches else f"[{ng_str}]",
                "line_text": r_item.get("text", ""),
                "ipa_text": lp_item.get("ipa", ""),
            })

    cat_order = {"alliteration": 1, "assonance": 2, "chaine": 3, "chiasme_echo": 4, "superlatif": 5, "ngram": 6}
    all_patterns.sort(key=lambda x: (x["line"], cat_order.get(x["category"], 99), -x["count"]))

    summary_by_category = {
        "alliteration": {"label": "Allitération", "count": sum(1 for p in all_patterns if p["category"] == "alliteration")},
        "assonance": {"label": "Assonance", "count": sum(1 for p in all_patterns if p["category"] == "assonance")},
        "chaine": {"label": "Chaîne phonématique", "count": sum(1 for p in all_patterns if p["category"] == "chaine")},
        "chiasme_echo": {"label": "Chiasme & Écho", "count": sum(1 for p in all_patterns if p["category"] == "chiasme_echo")},
        "superlatif": {"label": "Superlatif prosodique", "count": sum(1 for p in all_patterns if p["category"] == "superlatif")},
        "ngram": {"label": "N-gramme récurrent", "count": sum(1 for p in all_patterns if p["category"] == "ngram")},
    }

    categories = [
        {"key": "all", "label": f"Toutes les figures ({len(all_patterns)})"},
        {"key": "alliteration", "label": f"Allitérations ({summary_by_category['alliteration']['count']})"},
        {"key": "assonance", "label": f"Assonances ({summary_by_category['assonance']['count']})"},
        {"key": "chaine", "label": f"Chaînes phonématiques ({summary_by_category['chaine']['count']})"},
        {"key": "chiasme_echo", "label": f"Chiasmes & Échos ({summary_by_category['chiasme_echo']['count']})"},
        {"key": "superlatif", "label": f"Superlatifs prosodiques ({summary_by_category['superlatif']['count']})"},
        {"key": "ngram", "label": f"N-grammes récurrents ({summary_by_category['ngram']['count']})"},
    ]

    stats = {
        "total_patterns": len(all_patterns),
        "total_alliterations": summary_by_category["alliteration"]["count"],
        "total_assonances": summary_by_category["assonance"]["count"],
        "total_chains": summary_by_category["chaine"]["count"],
        "total_echos": summary_by_category["chiasme_echo"]["count"],
        "total_superlatifs": summary_by_category["superlatif"]["count"],
        "total_recurrent_ngrams": len(recurrent_ngrams),
    }

    return {
        "all_patterns": all_patterns,
        "summary_by_category": summary_by_category,
        "categories": categories,
        "recurrent_ngrams": recurrent_ngrams,
        "stats": stats,
    }

# ---------- 分析メイン ----------
def analyze(text, dierese=False, fold_indented=False, nlp=None, ancient_postvocalic_e=False):
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
            "prosody_rhythm": analyze_prosody_and_rhythm([], [], [], [], dierese=dierese, ancient_postvocalic_e=ancient_postvocalic_e),
            "salient_phonetic_patterns": {
                "all_patterns": [],
                "summary_by_category": {},
                "categories": [],
                "recurrent_ngrams": [],
                "stats": {
                    "total_patterns": 0,
                    "total_alliterations": 0,
                    "total_assonances": 0,
                    "total_chains": 0,
                    "total_echos": 0,
                    "total_superlatifs": 0,
                    "total_recurrent_ngrams": 0,
                },
            },
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
        lines, docs, unit_words, line_phonetics_list, dierese=dierese, ancient_postvocalic_e=ancient_postvocalic_e
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
            "feminine": prosody_rhythm._has_final_post_tonic_e(ws[-1].text, ws[-1].pos_) if ws else False,
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

    computed = compute_corpus_metrics(docs, unit_words_list=unit_words, dierese=dierese, ancient_postvocalic_e=ancient_postvocalic_e)
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

    salient_patterns = extract_salient_phonetic_patterns(
        rows, line_phonetics_list, mesch_data=pr_result.get("meschonnic")
    )

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
        "salient_phonetic_patterns": salient_patterns,
    }

def load_reference(path=Path(__file__).with_name("reference.json")):
    return json.loads(Path(path).read_text(encoding="utf-8"))
