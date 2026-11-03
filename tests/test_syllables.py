import pytest
from analysis import word_syllables, line_syllables

@pytest.mark.parametrize("word,pos,expected", [
    ("ortie", None, 2),
    ("orties", None, 2),
    ("épuisée", None, 3),
    ("aimaient", "VERB", 2),
    ("vieillissent", "VERB", 2),
    ("géographiques", None, 4),
    ("pays", None, 2),
    ("cimetière", None, 3),
])
def test_syllable_required_words(word, pos, expected):
    """指令書で指定された最低限確認すべき8単語の音節数テスト。"""
    assert word_syllables(word, pos=pos, dierese=False) == expected

def test_dierese_option():
    """分音（diérèse）オプションの切り替えテスト。"""
    # cimetière は標準では 3 音節、分音（ci-me-ti-è-re）では 4 音節
    assert word_syllables("cimetière", dierese=False) == 3
    assert word_syllables("cimetière", dierese=True) == 4

    # passion は標準では 2 音節（pas-sion）、分音では 3 音節（pas-si-on）
    assert word_syllables("passion", dierese=False) == 2
    assert word_syllables("passion", dierese=True) == 3

    # aimaient は標準・分音ともに 2 音節（ai は二重字であり分音の対象外）
    assert word_syllables("aimaient", pos="VERB", dierese=False) == 2
    assert word_syllables("aimaient", pos="VERB", dierese=True) == 2
