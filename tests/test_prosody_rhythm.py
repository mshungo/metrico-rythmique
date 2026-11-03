"""4大理論統合エンジン (prosody_rhythm.py) の単体・統合テスト。"""
from pathlib import Path
from analysis import analyze, get_nlp
from prosody_rhythm import (
    detect_synerese_dierese,
    syllabify_phonemes,
    compare_rhyme_pair,
    extract_line_rhyme_signature,
    build_line_syllable_stream,
)
from phonetics import analyze_line_phonetics


def test_ui_independence():
    """コア論理モジュールが streamlit に一切依存していないことを検証する。"""
    for fname in ("phonetics.py", "prosody_rhythm.py", "analysis.py"):
        src = Path(__file__).resolve().parent.parent.joinpath(fname).read_text(encoding="utf-8")
        assert "import streamlit" not in src
        assert "from streamlit" not in src


def test_muta_cum_liquida_and_synerese_dierese():
    """Gouvard: muta cum liquida の義務的分音と語源的分音・標準合音の識別テスト。"""
    mcl = detect_synerese_dierese("sanglier", pos="NOUN", dierese_mode=False)
    assert mcl and mcl[0]["type"] == "diérèse_muta_cum_liquida"
    assert mcl[0]["counted_as_dierese"] is True

    mcl2 = detect_synerese_dierese("cruel", pos="ADJ", dierese_mode=False)
    assert mcl2 and mcl2[0]["counted_as_dierese"] is True

    syn = detect_synerese_dierese("ciel", pos="NOUN", dierese_mode=True)
    assert syn and syn[0]["counted_as_dierese"] is False

    etym_off = detect_synerese_dierese("passion", pos="NOUN", dierese_mode=False)
    etym_on = detect_synerese_dierese("passion", pos="NOUN", dierese_mode=True)
    assert etym_off and etym_off[0]["counted_as_dierese"] is False
    assert etym_on and etym_on[0]["counted_as_dierese"] is True


def test_cornulier_alexandrin_and_caesura_codes():
    """Cornulier: 8音節の法則、古典6+6一致、C6前接語違反、4=4=4三拍子の診断テスト。"""
    poem = (
        "Toujours aimer, toujours souffrir, toujours mourir,\n"
        "J'ai vu l'ombre passer sur le front des poètes ;\n"
        "Sous le pont Mirabeau"
    )
    res = analyze(poem)
    details = res["prosody_rhythm"]["lines_detail"]

    # 第1行: 12音節、6+6一致かつ 4=4=4 三拍子プロファイルも成立（両義的アレクサンドラン）
    assert details[0]["anatonic_syllables"] == 12
    assert details[0]["caesura_code"].startswith("6+6")
    assert "4=4=4" in details[0]["ternary_profile"]

    # 第2行: "J'ai vu l'om-bre pas-ser | sur le front des po-è(tes)" -> 6+6一致
    assert details[1]["anatonic_syllables"] == 12

    # 第3行: 6音節 ≤ 8 -> 単一詩句 (vers simple)
    assert details[2]["anatonic_syllables"] == 6
    assert "vers simple" in details[2]["loi_des_8"]


def test_cornulier_proclitic_c6_and_m6():
    """Cornulier & Gouvard: 第6音節の前接語 C6 / 語内 M6 の検出テスト。"""
    # "Il regarde dans la | grande forêt obscure" -> pos 6 = "la" (C6)
    res = analyze("Il regarde dans la grande forêt obscure")
    d0 = res["prosody_rhythm"]["lines_detail"][0]
    assert d0["anatonic_syllables"] == 12
    assert d0["caesura_code"].startswith(("C6", "P6"))


def test_gouvard_rhyme_schemes_and_richness():
    """Gouvard & Plénat: ABBA抱擁韻・ABAB交差韻、脚韻の豊かさ（富韻・足韻・アソナンス）テスト。"""
    quatrain_abba = (
        "La Nature est un temple où de vivants piliers\n"
        "Laissent parfois sortir de confuses paroles ;\n"
        "L'homme y passe à travers des forêts de symboles\n"
        "Qui l'observent avec des regards familiers."
    )
    res = analyze(quatrain_abba)
    stanzas = res["prosody_rhythm"]["stanzas_detail"]
    assert len(stanzas) == 1
    assert stanzas[0]["scheme"] == "ABBA"
    assert "抱擁韻" in stanzas[0]["disposition"]
    assert stanzas[0]["alternance_respected"] is True

    # paroles [paʁɔl] / symboles [sɛ̃bɔl] -> [ɔl] 2音素一致 = 足韻 (rime suffisante)
    lines_d = res["prosody_rhythm"]["lines_detail"]
    assert "足韻" in lines_d[1]["rhyme_richness"] or "富韻" in lines_d[1]["rhyme_richness"]
    # piliers [pilje] / familiers [familje] -> [ilje] 4音素・2母音一致 = レオニン韻または富韻
    assert ("レオニン" in lines_d[0]["rhyme_richness"]) or ("富韻" in lines_d[0]["rhyme_richness"])


def test_gouvard_hiatus_postvocalic_e_and_rejet():
    """Gouvard & Mazaleyrat: et+母音イアチュス、語中母音後-e違反、Rejet（送り語）検出テスト。"""
    text = (
        "Tu as vu la vie belle et il pleure dans l'ombre\n"
        "Il marche lentement vers le vieil escalier\n"
        "Dérobé, dans la nuit."
    )
    res = analyze(text)
    gouv = res["prosody_rhythm"]["gouvard"]

    # "Tu as" と "et il" でイアチュス検出
    hiatus_types = {h["type"] for h in gouv["hiatus_events"]}
    assert "hiatus_et" in hiatus_types
    assert "hiatus_metrique" in hiatus_types or "hiatus_meme_timbre" in hiatus_types

    # "vie belle" -> 母音後 -e の子音前非エリジオン検出
    ec_statuses = {e["status"] for e in gouv["e_caduc_events"]}
    assert "apocope_postvocalique" in ec_statuses

    # "escalier // Dérobé," -> 外的送り語 (rejet_externe)
    disc_types = {d["type"] for d in gouv["discordances"]["events"]}
    assert "rejet_externe" in disc_types


def test_di_cristo_bipolar_clash_and_enclitics():
    """Di Cristo: 同一韻律句(SP)内の強勢衝突による語頭AI移動、エンクリティック、優律動スコア・nPVIの検証。"""
    # "Une chemise noire, prends-le, disais-je."
    res = analyze("Une chemise noire, prends-le, disais-je.")
    dc = res["prosody_rhythm"]["di_cristo"]
    ld = res["prosody_rhythm"]["lines_detail"][0]

    # "chemise noire" は同一SP内で衝突解消または二重標示により AI が付与されアクセント弧を形成
    assert ld["ai_count"] >= 1
    assert ld["arcs_count"] >= 1
    assert 0.0 <= dc["eurythmy_score"] <= 1.0
    assert dc["open_syl_ratio_pct"] > 50.0
    assert dc["npvi_global"] >= dc["npvi_interne"]


def test_meschonnic_scansion_contre_accents_and_echoes():
    """Meschonnic & Dessons: 走査記号 (—, ⏑ˈ, (⏑ˈ), ◡)、反アクセント、7種エコー図式、最上級語の検証。"""
    poem = (
        "Le murmure des mers monte dans la mémoire,\n"
        "Triste, pâle, muet, sous la sombre lumière."
    )
    res = analyze(poem)
    mesch = res["prosody_rhythm"]["meschonnic"]

    # 走査文字列に — や (⏑ˈ) が含まれること
    scan0 = mesch["lines"][0]["scansion_unicode"]
    assert "—" in scan0
    assert "(⏑ˈ)" in scan0

    # murmure / mémoire 等がプロゾディー最上級語として検出されること
    sup_words = {s["word"].lower() for s in mesch["superlatifs_prosodiques"]}
    assert "murmure" in sup_words or "mémoire" in sup_words

    # 第2行 "Triste, pâle, muet" 等で休止跨ぎや隣接強勢による反アクセントまたはエコー図式が検出されること
    assert len(mesch["echo_figures"]) >= 1
    assert sum(mesch["finale_profile"]["counts"].values()) >= 2


def test_rimbaud_bateau_ivre_precision_and_measures():
    """ユーザー指摘のランボー『酔いどれ船』難解行および古典〜象徴派詩行の1ミリ単位の厳密検証。"""
    # 1. 単行入力・dierese=False/True 両モードで完全一致すること
    for mode in (False, True):
        r1 = analyze("Les ayant cloués nus aux poteaux de couleurs.", dierese=mode)
        d1 = r1["prosody_rhythm"]["lines_detail"][0]
        assert r1["rows"][0]["syllables"] == 12
        assert d1["anatonic_syllables"] == 12
        assert d1["caesura_code"] == "6+6 (concordant)"
        assert d1["measures_str"] == "5+1 // 3+3"

        for oeil_form in ("l’oeil", "l'œil"):
            r2 = analyze(f"Dix nuits, sans regretter {oeil_form} niais des falots !", dierese=mode)
            d2 = r2["prosody_rhythm"]["lines_detail"][0]
            assert r2["rows"][0]["syllables"] == 12
            assert d2["anatonic_syllables"] == 12
            assert d2["caesura_code"] == "6+6 (concordant)"
            assert d2["measures_str"] == "2+4 // 3+3"

    # 2. 『酔いどれ船』の脱古典セジュール (M6, L6, C6, X6 jumeaux)・揺れ語彙 (anciens)・外来語 (Maelstroms, alcool)
    bateau_excerpt = "\n".join([
        "Dans les clapotements furieux des marées,",
        "Moi, l’autre hiver, plus sourd que les cerveaux d’enfants,",
        "Je courus ! Et les Péninsules démarrées",
        "N’ont pas subi tohu-bohus plus triomphants :",
        "Échouages hideux au fond des gouffres bruns",
        "Choient, des arbres tordus, avec de noirs parfums !",
        "Et les fientes d’oiseaux clabaudeurs aux yeux blonds.",
        "Plus forts que l’alcool, plus vastes que nos lyres,",
        "Fermentent les rousseurs amères de l’amour !",
        "Le rut des Béhémots et les Maelstroms épais,",
        "Fileur éternel des immobilités bleues,",
        "Je regrette l’Europe aux anciens parapets !",
        "Million d’oiseaux d’or, ô future Vigueur ?",
    ])
    res_b = analyze(bateau_excerpt, dierese=True)
    details = res_b["prosody_rhythm"]["lines_detail"]
    assert all(d["anatonic_syllables"] == 12 for d in details)
    assert details[2]["caesura_code"].startswith("M6")
    assert "3-5-4" in details[2]["ternary_profile"]
    assert details[3]["caesura_code"].startswith("L6")
    assert "4=4=4" in details[3]["ternary_profile"]
    assert details[10]["caesura_code"].startswith("C6")
    assert details[11]["caesura_code"] == "6+6 (élision)"
    assert res_b["prosody_rhythm"]["cornulier"]["x6_jumeaux_count"] >= 1


def test_cornulier_s6_f6_and_classical_canon():
    """Cornulier S6 (F7リズム回収)・F6・古典〜近代規範詩句 (Racine, Hugo, Baudelaire, Nerval, Valéry, Verlaine) の厳密検証。"""
    s6 = analyze("Il voit cette ombre noire glisser dans la nuit", dierese=True)["prosody_rhythm"]["lines_detail"][0]
    assert s6["anatonic_syllables"] == 12
    assert s6["caesura_code"].startswith("S6")

    f6 = analyze("Il marche dans l'ombre noire de la forêt", dierese=True)["prosody_rhythm"]["lines_detail"][0]
    assert f6["anatonic_syllables"] == 12
    assert f6["caesura_code"].startswith("F6")

    canon_12 = "\n".join([
        "Le jour n’est pas plus pur que le fond de mon coeur.",
        "Sans rien voir au dehors, sans entendre aucun bruit,",
        "Comme des avirons traîner à côté d’eux.",
        "Suis-je Amour ou Phébus ? … Lusignan ou Biron ?",
        "Le vierge, le vivace et le bel aujourd’hui",
    ])
    res_c = analyze(canon_12, dierese=True)
    assert all(r["syllables"] == 12 for r in res_c["rows"])

    # Valéry 10v (4+6) & Verlaine 9v (4+5)
    res_10 = analyze("Ce toit tranquille, où marchent des colombes,", dierese=True)
    assert res_10["rows"][0]["syllables"] == 10
    assert res_10["prosody_rhythm"]["lines_detail"][0]["caesura_code"].startswith("4+6")

    res_9 = analyze("De la musique avant toute chose,", dierese=True)
    assert res_9["rows"][0]["syllables"] == 9
    assert res_9["prosody_rhythm"]["lines_detail"][0]["caesura_code"].startswith("4+5")


