"""フランス語音素分析・IPA・リエゾン・音素偏差のユニットテスト。"""
import pytest
from phonetics import (
    word_to_ipa,
    extract_phonemes,
    analyze_line_phonetics,
    compute_phoneme_deviations,
    FRENCH_PHONEME_REFERENCE,
    H_ASPIRE
)
from analysis import analyze, get_nlp


def test_word_to_ipa_basics():
    """基本的な単語のIPA変換と潜在リエゾン子音の検証。"""
    ipa, latent = word_to_ipa("les")
    assert ipa == "le"
    assert latent == "z"

    ipa, latent = word_to_ipa("grand")
    assert ipa == "ɡʁɑ̃"
    assert latent == "t"

    ipa, latent = word_to_ipa("un")
    assert ipa == "œ̃"
    assert latent == "n"

    # 動詞3人称複数 -ent の無音
    ipa, latent = word_to_ipa("chantent", pos="VERB")
    assert "ɑ̃t" in ipa

    # 語末 -er の [e]
    ipa, latent = word_to_ipa("aimer", pos="VERB")
    assert ipa.endswith("e")


def test_word_to_ipa_elisions():
    """アポストロフィ短縮形（qu', l', d', c', j', etc.）の安全な変換テスト。"""
    # 3文字の接頭辞 qu'
    ipa, latent = word_to_ipa("qu'il")
    assert ipa == "kil"

    ipa, latent = word_to_ipa("qu'")
    assert ipa == "k"

    # Unicode アポストロフィ ’
    ipa, latent = word_to_ipa("qu’on")
    assert ipa.startswith("k")

    ipa, latent = word_to_ipa("l'arbre")
    assert ipa == "laʁbʁ"

    ipa, latent = word_to_ipa("c'est")
    assert ipa == "sɛ"
    assert latent == "t"


def test_word_to_ipa_high_precision():
    """プレースホルダー衝突回避・母音間s有声化・無音e前子音・半母音の高精度テスト。"""
    # lune は [lyn] (鼻母音や口母音誤置換がないこと)
    ipa, _ = word_to_ipa("lune")
    assert ipa == "lyn"

    # aime は [ɛm] (鼻母音 ɛ̃ にならないこと)
    ipa, _ = word_to_ipa("aime")
    assert ipa == "ɛm"

    # rose, chose は [ʁoz], [ʃoz] (母音間sが有声化すること)
    ipa, _ = word_to_ipa("rose")
    assert ipa == "ʁoz"
    ipa, _ = word_to_ipa("chose")
    assert ipa == "ʃoz"

    # suite は [sɥit] (半母音 ɥ)
    ipa, _ = word_to_ipa("suite")
    assert ipa == "sɥit"

    # terre, cette, belle は [tɛʁ], [sɛt], [bɛl] (開口母音 ɛ)
    ipa, _ = word_to_ipa("terre")
    assert ipa == "tɛʁ"
    ipa, _ = word_to_ipa("cette")
    assert ipa == "sɛt"
    ipa, _ = word_to_ipa("belle")
    assert ipa == "bɛl"

    # de は [də] (シュワー)
    ipa, _ = word_to_ipa("de")
    assert ipa == "də"

    # vient は [vjɛ̃] (半母音 j + 鼻母音 ɛ̃)
    ipa, _ = word_to_ipa("vient")
    assert ipa == "vjɛ̃"


def test_liaison_detection():
    """リエゾン判定のテスト（義務的リエゾン、et禁止、h aspiré禁止）。"""
    nlp = get_nlp()

    # 1. 義務的リエゾン: les amis
    doc1 = nlp("les amis")
    tokens1 = [t for t in doc1 if not (t.is_punct or t.is_space)]
    res1 = analyze_line_phonetics(tokens1)
    assert len(res1["liaisons"]) == 1
    assert res1["liaisons"][0]["consonant"] == "z"
    assert "le‿z‿" in res1["ipa"]

    # 2. 接続詞 et は絶対にリエゾンしない
    doc2 = nlp("et un ami")
    tokens2 = [t for t in doc2 if not (t.is_punct or t.is_space)]
    res2 = analyze_line_phonetics(tokens2)
    for lia in res2["liaisons"]:
        assert lia["word1"].lower() != "et"

    # 3. 有音のh（h aspiré）ではリエゾンしない
    assert "héros" in H_ASPIRE
    doc3 = nlp("les héros")
    tokens3 = [t for t in doc3 if not (t.is_punct or t.is_space)]
    res3 = analyze_line_phonetics(tokens3)
    assert len(res3["liaisons"]) == 0

    # 4. 形容詞 + 名詞: grand arbre
    doc4 = nlp("un grand arbre")
    tokens4 = [t for t in doc4 if not (t.is_punct or t.is_space)]
    res4 = analyze_line_phonetics(tokens4)
    lia_grand = [l for l in res4["liaisons"] if l["word1"].lower() == "grand"]
    assert len(lia_grand) == 1
    assert lia_grand[0]["consonant"] == "t"


def test_extract_phonemes():
    """IPA文字列からの音素抽出（鼻母音や特殊子音が1音素として抽出されること）。"""
    ipa = "le‿z‿ ami ɔ̃‿t‿ œ̃ ɡʁɑ̃‿t‿ aʁbʁ"
    phons = extract_phonemes(ipa)
    # リエゾン子音 z, t が含まれること
    assert "z" in phons
    assert "t" in phons
    # 鼻母音 ɔ̃, œ̃, ɑ̃ が分割されずに含まれること
    assert "ɔ̃" in phons
    assert "œ̃" in phons
    assert "ɑ̃" in phons
    # 軟口蓋破裂音 ɡ, 流音 ʁ が含まれること
    assert "ɡ" in phons
    assert "ʁ" in phons


def test_deviations_calculation():
    """音素偏差計算のテスト。"""
    sample_phonemes = ["a", "a", "a", "a", "s", "s", "t", "ʁ", "l", "i"]
    res = compute_phoneme_deviations(sample_phonemes)

    assert res["total_phonemes"] == 10
    assert len(res["phoneme_stats"]) == len(FRENCH_PHONEME_REFERENCE)

    # a は4回出現 = 40% (基準値8.10%より大幅に高い)
    stat_a = next(p for p in res["phoneme_stats"] if p["phoneme"] == "a")
    assert stat_a["count"] == 4
    assert stat_a["pct"] == 40.0
    assert stat_a["diff"] > 0
    assert stat_a["z_score"] > 2.0
    assert stat_a["t_score"] > 50

    # カテゴリ集計の存在確認
    cat = res["category_deviations"]
    assert "vowels_pct" in cat
    assert "consonants_pct" in cat
    assert "ref_vowels_pct" in cat
    assert res["vc_ratio"] > 0


def test_analyze_integration():
    """analyze() に音素分析データが正しく格納されているかのテスト。"""
    text = "Qu'il vienne tout de suite !\nLes enfants marchaient sous un grand ciel bleu."
    res = analyze(text)

    assert "phonetics" in res
    assert "rows" in res
    assert len(res["rows"]) == 2

    # rowsに音素情報が追加されている
    r1 = res["rows"][0]
    assert "ipa" in r1
    assert "phonemes" in r1
    assert "vowels" in r1
    assert "consonants" in r1
    assert "liaisons" in r1
    assert r1["phonemes"] > 0

    # phonetics辞書の内容確認
    phon = res["phonetics"]
    assert "lines" in phon
    assert "deviations" in phon
    assert "all_liaisons" in phon
    assert phon["total_liaisons"] >= 1
    assert len(phon["deviations"]["phoneme_stats"]) > 0
