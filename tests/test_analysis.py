import pytest
from analysis import analyze, parse_text

def test_empty_and_punct():
    """空行や句読点のみの入力、#で始まるコメント行で落ちないこと。"""
    res1 = analyze("")
    assert res1["rows"] == []
    assert res1["lines_summary"] == {}

    # #で始まる行は無視、他は句読点のみ
    res2 = analyze("...\n???\n!\n#p99 ページ番号のメモ")
    assert len(res2["rows"]) == 3
    assert res2["lines_summary"]["語数"] == 0
    assert res2["metrics"]["punct_per100"] == 0.0

def test_mattr_short_text():
    """100語未満のテクストでは MATTR が None となること。"""
    short_text = "Demain dès l'aube à l'heure où blanchit la campagne je partirai."
    res = analyze(short_text)
    assert res["lines_summary"]["語数"] < 100
    assert res["metrics"]["mattr100"] is None

def test_indentation_folding():
    """字下げ行の折り返し統合オプションのテスト。"""
    sample = "Premier vers long\n    suite du premier vers\nDeuxième vers"
    res_off = analyze(sample, fold_indented=False)
    assert len(res_off["rows"]) == 3

    res_on = analyze(sample, fold_indented=True)
    assert len(res_on["rows"]) == 2
    assert res_on["rows"][0]["text"] == "Premier vers long suite du premier vers"

def test_pattern_detection():
    """定型句・行頭反復検出のテスト。"""
    sample = """
Elle disait tout bas
Et nous marchions ensemble

C'était hier dans la rue
Il y a des traces de pas

Vienne la nuit sonne l'heure
Vienne le jour parte la peur

Je dis rouge pour ne pas dire sang
"""
    res = analyze(sample)
    pats = res["extended_patterns"]

    # 1. elle disait / et nous
    assert len(pats["disait_et_nous"]) == 1
    assert pats["disait_et_nous"][0]["line_pair"] == "1-2"

    # 2. C'était hier / il y a
    assert len(pats["hier_il_y_a"]) == 1
    assert pats["hier_il_y_a"][0]["line_pair"] == "3-4"

    # 3. 行頭反復（Vienne... Vienne... 冠詞除く）
    assert len(pats["anaphora"]) == 1
    assert pats["anaphora"][0]["word"] == "vienne"
    assert pats["anaphora"][0]["count"] == 2

    # 4. X pour ne pas dire Y
    assert len(res["formulas"]) == 1
    assert res["formulas"][0]["言った語"] == "rouge"
    assert res["formulas"][0]["言い直し"] == "sang"


def test_line_phoneme_dynamics():
    """詩行ごとの音素数、音素密度、V/C比、母音・子音比率の動的推移データの検証。"""
    sample = (
        "Le ciel est, par-dessus le toit,\n"
        "Si bleu, si calme !"
    )
    res = analyze(sample)
    r1 = res["rows"][0]
    r2 = res["rows"][1]

    assert r1["phonemes"] > 0
    assert r1["vowels"] > 0
    assert r1["consonants"] > 0
    assert r1["phoneme_density"] > 0.0
    assert r1["vc_ratio"] > 0.0
    assert round(r1["vowel_pct"] + r1["consonant_pct"]) == 100

    assert r2["phonemes"] > 0
    assert r2["phoneme_density"] > 0.0


def test_phoneme_length_normalization_and_deviations():
    """長短行における音素頻度歪み補正（12音節標準換算、O/E比、z-score）の検証。"""
    import math

    # 短行（4音節・8音素）と長行（12音節・24音素）で同一音素密度（音節あたり0.5回）
    cnt_short = 2
    sylls_short = 4
    phs_short = 8

    cnt_long = 6
    sylls_long = 12
    phs_long = 24

    # 1. 12音節標準換算
    norm12_short = round(cnt_short * 12.0 / sylls_short, 2)
    norm12_long = round(cnt_long * 12.0 / sylls_long, 2)
    assert norm12_short == norm12_long == 6.0  # 長短行の歪みが完全に排除される

    # 2. 標準語期待比 O/E
    p_ref = 0.05
    e_short = phs_short * p_ref
    e_long = phs_long * p_ref
    oe_short = round(cnt_short / e_short, 2)
    oe_long = round(cnt_long / e_long, 2)
    assert oe_short == oe_long == 5.0  # 濃縮倍率が同一に測定される

    # 3. 二項統計偏差 z-score
    denom_short = math.sqrt(phs_short * p_ref * (1.0 - p_ref))
    denom_long = math.sqrt(phs_long * p_ref * (1.0 - p_ref))
    z_short = round((cnt_short - e_short) / denom_short, 2)
    z_long = round((cnt_long - e_long) / denom_long, 2)
    assert z_long > z_short > 0.0  # サンプルサイズが大きい長行の方が統計的有意性が高く評価される


def test_pos_analysis():
    """品詞構成（POS分析）、語彙密度、名詞比率、各行トークンの品詞情報テスト。"""
    sample = (
        "La Nature est un temple où de vivants piliers\n"
        "Laissent parfois sortir de confuses paroles ;"
    )
    res = analyze(sample)
    pos_an = res["pos_analysis"]
    assert pos_an["total_words"] > 0
    assert pos_an["content_words"] > 0
    assert pos_an["function_words"] > 0
    assert pos_an["lexical_density"] > 0.0
    assert pos_an["nominal_ratio"] > 0.0
    assert "style_desc" in pos_an

    r1 = res["rows"][0]
    assert len(r1["word_tokens"]) > 0
    assert "pos" in r1["word_tokens"][0]
    assert r1["content_words"] > 0
    assert r1["function_words"] > 0
    assert r1["rhyme_pos"] in ("NOUN", "ADJ", "VERB", "ADV", "PROPN")


def test_compute_phoneme_progression_dynamics():
    """詩句進展に伴う特定音素の強まり・弱まり（crescendo / decrescendo）、平滑化傾向、論文引用LaTeX/Markdown生成の検証。"""
    from analysis import compute_phoneme_progression_dynamics

    # Crescendo パターン（詩句の進展に伴い徐々に音素頻度が増加）
    records_crescendo = [
        {"line": 1, "stanza": 1, "syllables": 12, "phoneme": "[ʁ]", "count": 1, "norm_12": 1.0, "text": "Vers 1"},
        {"line": 2, "stanza": 1, "syllables": 12, "phoneme": "[ʁ]", "count": 1, "norm_12": 1.0, "text": "Vers 2"},
        {"line": 3, "stanza": 1, "syllables": 12, "phoneme": "[ʁ]", "count": 2, "norm_12": 2.0, "text": "Vers 3"},
        {"line": 4, "stanza": 1, "syllables": 12, "phoneme": "[ʁ]", "count": 2, "norm_12": 2.0, "text": "Vers 4"},
        {"line": 5, "stanza": 2, "syllables": 12, "phoneme": "[ʁ]", "count": 3, "norm_12": 3.0, "text": "Vers 5"},
        {"line": 6, "stanza": 2, "syllables": 12, "phoneme": "[ʁ]", "count": 4, "norm_12": 4.0, "text": "Vers 6"},
        {"line": 7, "stanza": 2, "syllables": 12, "phoneme": "[ʁ]", "count": 5, "norm_12": 5.0, "text": "Vers 7"},
        {"line": 8, "stanza": 2, "syllables": 12, "phoneme": "[ʁ]", "count": 6, "norm_12": 6.0, "text": "Vers 8"},
    ]
    res_cresc = compute_phoneme_progression_dynamics(records_crescendo, smoothing_window=3)
    prof_c = res_cresc["profiles"]["[ʁ]"]

    assert prof_c["total_count"] == 24
    assert prof_c["slope"] > 0  # 正の回帰勾配
    assert prof_c["correlation_r"] > 0.80  # 強い正の相関
    assert prof_c["dyn_code"] == "crescendo"
    assert "Renforcement progressif" in prof_c["dyn_label"]
    assert prof_c["peak_lines"] == [8]
    assert len(prof_c["strophes_summary"]) == 2
    assert prof_c["strophes_summary"][1]["count"] > prof_c["strophes_summary"][0]["count"]

    # 論文引用向け出力フォーマットの検証
    assert "pente de régression linéaire" in prof_c["academic_citation_fr"]
    assert "\\begin{table}" in prof_c["latex_table"]
    assert "\\begin{tabular}" in prof_c["latex_table"]
    assert "| Strophe |" in prof_c["markdown_table"]

    # 平滑化（移動平均）が各行に付与されていること
    enr = res_cresc["enriched_records"]
    assert all("rolling_count" in r for r in enr)
    assert all("rolling_norm_12" in r for r in enr)

    # Decrescendo パターン（詩句の進展に伴い減衰）
    records_decrescendo = [
        {"line": 1, "stanza": 1, "syllables": 12, "phoneme": "[s]", "count": 5, "norm_12": 5.0, "text": "Vers 1"},
        {"line": 2, "stanza": 1, "syllables": 12, "phoneme": "[s]", "count": 4, "norm_12": 4.0, "text": "Vers 2"},
        {"line": 3, "stanza": 2, "syllables": 12, "phoneme": "[s]", "count": 2, "norm_12": 2.0, "text": "Vers 3"},
        {"line": 4, "stanza": 2, "syllables": 12, "phoneme": "[s]", "count": 0, "norm_12": 0.0, "text": "Vers 4"},
    ]
    res_decresc = compute_phoneme_progression_dynamics(records_decrescendo, smoothing_window=2)
    prof_d = res_decresc["profiles"]["[s]"]
    assert prof_d["slope"] < 0
    assert prof_d["correlation_r"] < -0.80
    assert prof_d["dyn_code"] == "decrescendo"
    assert "Atténuation progressive" in prof_d["dyn_label"]



