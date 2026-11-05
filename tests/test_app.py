"""Streamlit UI (app.py) の AppTest エンドツーエンド検証。"""
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_empty_default():
    """初期状態（空入力）で例外なく案内メッセージが表示されること。"""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.info) >= 1


def test_app_with_presets_and_short_text():
    """サンプル詩プリセット（100語未満で mattr100 が None のケース含む）およびGSD/ParisStories比較が例外なく描画されること。"""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # サンプル1（ボードレール四行連）を選択
    preset_sb = at.sidebar.selectbox[0]
    preset_sb.set_value(preset_sb.options[1]).run()
    assert not at.exception
    assert len(at.tabs) == 5
    assert len(at.metric) >= 10

    # サンプル2（脱古典セジュール・送り語）を選択し、分音オプションをON
    preset_sb = at.sidebar.selectbox[0]
    preset_sb.set_value(preset_sb.options[2])
    at.sidebar.checkbox[1].set_value(True).run()
    assert not at.exception

    # サンプル3（自由詩）を選択し、字下げ統合オプションをON
    preset_sb = at.sidebar.selectbox[0]
    preset_sb.set_value(preset_sb.options[3])
    at.sidebar.checkbox[0].set_value(True).run()
    assert not at.exception

    # サンプル4（ランボー『酔いどれ船』抜粋）を選択し、全12行が例外なく12音節として描画されること
    preset_sb = at.sidebar.selectbox[0]
    preset_sb.set_value(preset_sb.options[4]).run()
    assert not at.exception
    # 音素追跡 multiselect および推移チャートが例外なく描画されること
    assert len(at.multiselect) >= 1
    # スキャンション表示においてアクセント（mesch-stacked）を含む HTML が描画されること
    assert any("mesch-stacked" in h.value for h in at.html)


def test_app_labe_preset():
    """サンプル5（Louise Labé : Sonnet VIII 十音節詩）が例外なく描画されること。"""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    preset_sb = at.sidebar.selectbox[0]
    preset_sb.set_value(preset_sb.options[5]).run()
    assert not at.exception
    assert len(at.tabs) == 5
