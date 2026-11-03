"""フランス語詩テクストの韻律・リズム・プロゾディー・音素統合分析アプリ。
理論基盤: Meschonnic & Dessons / Cornulier / Di Cristo / Gouvard, Mazaleyrat & Plénat
起動: streamlit run app.py
"""
import html
import importlib
import json
import pandas as pd
import streamlit as st
import altair as alt
import math
import re

import phonetics
import prosody_rhythm
import analysis
importlib.reload(phonetics)
importlib.reload(prosody_rhythm)
importlib.reload(analysis)
from analysis import analyze, load_reference

LABELS = {
    "punct_per100": "句読点（100語あたり）",
    "words_per_unit_mean": "1単位の語数（平均）",
    "noverb_unit_pct": "定動詞を含まない単位（%）",
    "finite_per100": "定動詞（100語あたり）",
    "cconj_per100": "等位接続詞（100語あたり）",
    "sconj_per100": "従属接続詞（100語あたり）",
    "rel_per100": "関係代名詞（100語あたり）",
    "word_syl_mean": "1語の平均音節数",
    "mono_pct": "単音節語（%）",
    "zipf_content_mean": "内容語の平均頻度（Zipf）",
    "rare_lt3_pct": "低頻度語 Zipf<3（%）",
    "mattr100": "語彙の多様さ（MATTR・100語窓）",
    "p1sg_pct": "1人称単数（定動詞中%）",
    "taut_per1000": "同一語の近接反復（1000語あたり）",
    "pos_NOUN": "名詞（%）",
    "pos_VERB": "動詞（%）",
    "pos_AUX": "助動詞（%）",
    "pos_ADJ": "形容詞（%）",
    "pos_ADV": "副詞（%）",
    "pos_DET": "限定詞（%）",
    "pos_PRON": "代名詞（%）",
    "pos_ADP": "前置詞（%）",
    "pos_CCONJ": "等位接続詞率（%）",
    "pos_SCONJ": "従属接続詞率（%）",
    "pos_PROPN": "固有名詞（%）",
    "pos_NUM": "数詞（%）",
}

SAMPLE_TEXTS = {
    "（自由入力・ファイルアップロード）": "",
    "1. 古典・象徴派アレクサンドラン（Baudelaire『Correspondances』第1連）": (
        "La Nature est un temple où de vivants piliers\n"
        "Laissent parfois sortir de confuses paroles ;\n"
        "L'homme y passe à travers des forêts de symboles\n"
        "Qui l'observent avec des regards familiers."
    ),
    "2. 脱古典セジュール・三拍子・送り語（ユゴー／ヴェルレーヌ風検証連）": (
        "Toujours aimer, toujours souffrir, toujours mourir,\n"
        "J'ai vu l'ombre passer sur le front des poètes ;\n"
        "Souvenir, souvenir, que me veux-tu ? L'automne\n"
        "Faisait voler la grive à travers l'air atone."
    ),
    "3. 近代無句読点・自由詩（アポリネール風ヴェル・リーブル）": (
        "Sous le pont Mirabeau coule la Seine\n"
        "Et nos amours\n"
        "Faut-il qu'il m'en souvienne\n"
        "La joie venait toujours après la peine"
    ),
    "4. ランボー『酔いどれ船（Le Bateau ivre）』抜粋（6+6・M6/L6双子違反・分音検証）": (
        "Comme je descendais des Fleuves impassibles,\n"
        "Je ne me sentis plus guidé par les haleurs :\n"
        "Des Peaux-Rouges criards les avaient pris pour cibles,\n"
        "Les ayant cloués nus aux poteaux de couleurs.\n\n"
        "Dans les clapotements furieux des marées,\n"
        "Moi, l’autre hiver, plus sourd que les cerveaux d’enfants,\n"
        "Je courus ! Et les Péninsules démarrées\n"
        "N’ont pas subi tohu-bohus plus triomphants :\n\n"
        "La tempête a béni mes éveils maritimes.\n"
        "Plus léger qu’un bouchon j’ai dansé sur les flots\n"
        "Qu’on appelle rouleurs éternels de victimes,\n"
        "Dix nuits, sans regretter l’oeil niais des falots !"
    ),
}

CONTRE_ACCENT_LABELS = {
    "R-R": "R-R（リズム群強勢×リズム群強勢）",
    "P-P": "P-P（プロゾディー強勢×プロゾディー強勢）",
    "R-P": "R-P（リズム群強勢→プロゾディー強勢）",
    "P-R": "P-R（プロゾディー強勢→リズム群強勢）",
    "R-M": "R-M（リズム群強勢→メートル固定位置）",
    "M-R": "M-R（メートル固定位置→リズム群強勢）",
    "P-M": "P-M（プロゾディー強勢→メートル固定位置）",
    "M-P": "M-P（メートル固定位置→プロゾディー強勢）",
}


@st.cache_resource
def load_spacy_model():
    """spaCy モデルをキャッシュしてロードする。"""
    import spacy
    return spacy.load("fr_core_news_md")


def render_meschonnic_html(mesch_lines, dom_consonants):
    """Meschonnic & Dessons (1998) の多層ディスクール走査を視覚的HTMLカードとして描画する。"""
    dom_set = {d["consonant"] for d in dom_consonants[:3]}
    cards = [
        """<style>
        .mesch-box { font-family: 'Segoe UI', 'Noto Sans JP', sans-serif; margin-bottom: 10px; padding: 10px 12px; border: 1px solid #e2e8f0; border-radius: 8px; background: #ffffff; }
        .mesch-hdr { font-size: 0.80rem; color: #475569; margin-bottom: 6px; font-weight: 600; }
        .mesch-row { display: flex; flex-wrap: wrap; gap: 3px; align-items: flex-end; }
        .mesch-cell { display: inline-flex; flex-direction: column; align-items: center; padding: 3px 5px; border-radius: 5px; min-width: 28px; }
        .mesch-R { background: #eff6ff; border-bottom: 2px solid #1d4ed8; }
        .mesch-M { background: #fef3c7; border-bottom: 2px solid #d97706; }
        .mesch-P { background: #fdf2f8; border-bottom: 2px solid #be185d; }
        .mesch-U { background: #f8fafc; border-bottom: 1px solid #cbd5e1; }
        .mesch-ext { background: #f8fafc; border-bottom: 1px dashed #94a3b8; border-left: none; margin-left: -2px; }
        .mesch-ext .mesch-orth { font-style: italic; color: #64748b; }
        .mesch-ext .mesch-mark { font-size: 0.85rem; color: #94a3b8; min-height: 18px; }
        .mesch-sup { font-size: 0.70rem; font-weight: 700; color: #dc2626; min-height: 14px; }
        .mesch-mark { font-size: 1.02rem; font-weight: 700; color: #0f172a; line-height: 1.2; }
        .mesch-orth { font-size: 0.88rem; color: #1e293b; margin-top: 2px; }
        .mesch-ipa { font-size: 0.74rem; color: #64748b; }
        .mesch-echo { color: #be185d; font-weight: 700; }
        .mesch-bar { font-size: 1.25rem; font-weight: 700; color: #0284c7; padding: 0 4px; align-self: center; }
        .mesch-gap { width: 8px; }
        </style>"""
    ]
    for l_obj in mesch_lines:
        if not l_obj["syllable_tokens"]:
            continue
        cells_html = []
        for tok in l_obj["syllable_tokens"]:
            orig = tok["origin"]
            cls = (
                "mesch-R" if orig == "R"
                else ("mesch-M" if orig == "M"
                else ("mesch-P" if orig in ("P", "P_ext")
                else ("mesch-ext" if orig == "ext"
                else "mesch-U")))
            )
            sup_txt = str(tok["contre_num"]) if tok["contre_num"] is not None else ""
            if tok["ligature_after"]:
                sup_txt += "⁀"
            has_dom_echo = any(c in dom_set for c in tok.get("onset_cons", []))
            ipa_cls = "mesch-ipa mesch-echo" if has_dom_echo else "mesch-ipa"
            orth_esc = html.escape(tok["orth"] + (tok["punct_after"] or ""))
            ipa_esc = html.escape(tok["ipa"])
            mark_esc = html.escape(tok["mark"]) if tok["mark"] else "&nbsp;"
            cells_html.append(
                f'<div class="mesch-cell {cls}">'
                f'<span class="mesch-sup">{html.escape(sup_txt)}</span>'
                f'<span class="mesch-mark">{mark_esc}</span>'
                f'<span class="mesch-orth">{orth_esc}</span>'
                f'<span class="{ipa_cls}">[{ipa_esc}]</span>'
                f'</div>'
            )
            if tok["caesura_after"]:
                cells_html.append('<span class="mesch-bar" title="セジュール (Césure)">│</span>')
            elif tok["is_word_end"]:
                cells_html.append('<span class="mesch-gap"></span>')

        vol_str = " + ".join(map(str, l_obj["group_volumes"])) if l_obj["group_volumes"] else "-"
        cards.append(
            f'<div class="mesch-box">'
            f'<div class="mesch-hdr">第{l_obj["line"]}行 ｜ グループ音節量: [{vol_str}]</div>'
            f'<div class="mesch-row">{"".join(cells_html)}</div>'
            f'</div>'
        )
    return "\n".join(cards)


def render_di_cristo_html(lines_detail):
    """Di Cristo (2016) の双極アクセント弧 {AI ... AF/AN} を視覚的HTMLとして描画する。"""
    blocks = [
        """<style>
        .dc-box { font-family: 'Segoe UI', 'Noto Sans JP', sans-serif; margin-bottom: 8px; padding: 9px 12px; border: 1px solid #e2e8f0; border-radius: 8px; background: #ffffff; }
        .dc-hdr { font-size: 0.78rem; color: #475569; margin-bottom: 4px; font-weight: 600; }
        .dc-row { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
        .dc-syl { padding: 3px 6px; border-radius: 5px; font-size: 0.86rem; background: #f8fafc; color: #334155; border: 1px solid #e2e8f0; }
        .dc-arc { background: #ecfeff; border-top: 3px solid #0891b2; }
        .dc-AI { background: #cffafe; border: 1px solid #06b6d4; font-weight: 700; color: #0e7490; }
        .dc-AF { background: #dbeafe; border: 1px solid #2563eb; font-weight: 700; color: #1e40af; }
        .dc-AN { background: #fee2e2; border: 1px solid #dc2626; font-weight: 700; color: #991b1b; }
        .dc-tag { font-size: 0.68rem; margin-left: 2px; vertical-align: super; }
        </style>"""
    ]
    for row in lines_detail:
        marks = row.get("di_cristo_marks", [])
        if not marks:
            continue
        spans = []
        for m in marks:
            acc = m["accent"]
            cls_list = ["dc-syl"]
            if m["in_arc"]:
                cls_list.append("dc-arc")
            if acc in ("AI", "AF", "AN"):
                cls_list.append(f"dc-{acc}")
            tag_html = f'<span class="dc-tag">{acc}</span>' if acc else ""
            spans.append(f'<span class="{" ".join(cls_list)}">{html.escape(m["orth"])}{tag_html}</span>')
        blocks.append(
            f'<div class="dc-box">'
            f'<div class="dc-hdr">第{row["line"]}行 ｜ 韻律句数(SP): {row["sp_count"]} ｜ 双極アクセント弧: {row["arcs_count"]} ｜ 表記: {html.escape(row["di_cristo_phrasing"])}</div>'
            f'<div class="dc-row">{"".join(spans)}</div>'
            f'</div>'
        )
    return "\n".join(blocks)


st.set_page_config(page_title="フランス語詩・韻律リズム＆音素統合分析", layout="wide")
st.title("フランス語詩・韻律リズム＆音素統合分析ツール")
st.caption("Meschonnic & Dessons・Cornulier・Di Cristo・Gouvard / Mazaleyrat / Plénat の4大理論に基づく多角的解析。1行＝1詩行、空行＝連の区切り、'#'開始行は無視。")

with st.sidebar:
    st.header("入力・解析オプション")
    preset_choice = st.selectbox(
        "検証用サンプル詩テクスト",
        options=list(SAMPLE_TEXTS.keys()),
        index=0,
        help="選択すると入力欄へサンプル詩が反映される。"
    )
    fold_indented = st.checkbox(
        "字下げ行を前の行の折り返しとして統合する",
        value=False,
        help="行頭に空白やタブがある行を独立行とせず、直前行の末尾へ連結する。"
    )
    dierese = st.checkbox(
        "古典的分音（diérèse）を数える",
        value=True,
        help="フランス定型詩の語源的分音（niais, million, passion, furieux, nuage 等）を適用する。揺れ語彙（anciens 等）は Cornulier のメトリック圧力により文脈解決する。"
    )
    st.divider()
    st.markdown(
        "**理論モジュール構成**\n"
        "- **Meschonnic & Dessons**: ディスクール走査・反アクセント8類型・7種プロゾディー反響\n"
        "- **Cornulier**: 8音節の法則・セジュール厳密診断 (`6+6`, `F6`, `S6`, `C6/P6`, `M6`, `4=4=4`)\n"
        "- **Di Cristo**: 4階層韻律構成素・双極アクセント弧 `{AI…AF}`・優律動性・二重 `nPVI`\n"
        "- **Gouvard / Plénat**: `CPMF` 分布タグ・`e caduc` 6状態・音韻的脚韻・句またがり/送り"
    )

up = st.file_uploader("テキストファイル（UTF-8）", type=["txt"])
default_text = SAMPLE_TEXTS.get(preset_choice, "")
text = up.read().decode("utf-8") if up else st.text_area("またはテキストを貼り付け", value=default_text, height=190)

if not text or not text.strip():
    st.info("テキストを入力またはファイルアップロード（もしくは左サイドバーから検証用サンプルを選択）すると解析を開始する。")
    st.stop()
    raise SystemExit

nlp = load_spacy_model()
with st.spinner("韻律・リズム・プロゾディー・音素を多角解析中…"):
    res = analyze(text, dierese=dierese, fold_indented=fold_indented, nlp=nlp)
ref = load_reference()

rows_data = res["rows"]
if not rows_data:
    st.warning("有効な詩行が見つからない。")
    st.stop()
    raise SystemExit

rows_df = pd.DataFrame(rows_data)
pr = res["prosody_rhythm"]
pr_sum = pr["summary"]
corn = pr["cornulier"]
mesch = pr["meschonnic"]
dc = pr["di_cristo"]
gouv = pr["gouvard"]
lines_detail = pr["lines_detail"]
lines_detail_df = pd.DataFrame(lines_detail)

# =============================================================================
# 6タブ構成による多角的分析ビュー
# =============================================================================
tab_overview, tab_mesch, tab_corn_gouv, tab_dicristo, tab_phon, tab_corpus = st.tabs([
    "1. 総合診断・全体像",
    "2. メショニック：リズム＆プロゾディー網",
    "3. コルニュリエ＆グヴァール：定型韻律・脚韻",
    "4. ディ・クリスト：双極アクセント・定量指標",
    "5. 音声学・IPA転写と音素偏差",
    "6. 一般仏語比較（GSD / ParisStories）・出力",
])

# -----------------------------------------------------------------------------
# Tab 1: 総合診断・全体像
# -----------------------------------------------------------------------------
with tab_overview:
    st.subheader("詩形式・韻律構造の総合判定")
    st.info(
        f"**全体韻律判定（Cornulier / Murat 基準）**: {pr_sum['global_poetic_form']}  \n"
        f"**詩節・定型構成（Gouvard 基準）**: {pr_sum['fixed_strophic_form']} ｜ **主軸メートル**: {pr_sum['dominant_meter']}"
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("等音節一致率", f"{pr_sum['isometry_ratio']:.1f}%", border=True, help="最頻出音節数に一致する詩行の割合")
    k2.metric("優律動スコア (Eurythmy)", f"{pr_sum['eurythmy_score']:.2f}", border=True, help="Di Cristo 基準: 2〜4音節の音調単位(UT)が占める均衡度 (0〜1)")
    k3.metric("開音節率 (CV%)", f"{pr_sum['open_syl_ratio_pct']:.1f}%", border=True, help="アンシェヌマン・リエゾン後の開音節率（仏語標準は約75〜80%）")
    k4.metric("反アクセント総数", f"{pr_sum['total_contre_accents']} 箇所", border=True, help="Meschonnic & Dessons: 連続する有標音節の隣接衝突・緊張箇所")
    k5.metric("プロゾディー反響図式", f"{pr_sum['total_echo_figures']} 件", border=True, help="語頭CV反響・縮減/拡張・キアスム・子音反跳・最上級語の総検出数")
    k6.metric("統辞＝韻律不一致", f"{pr_sum['total_discordances']} 箇所", border=True, help="句またがり(Enjambement)・送り(Rejet)・逆送り・半句不一致の総数")

    st.subheader("詩行ごとの主強勢音節数（Longueur anatonique）推移と「8音節の法則」境界")
    st.caption("青破線 (8音節) は Cornulier『8音節の法則（単一知覚の上限）』、赤破線 (12音節) は古典アレクサンドラン基準を示す。")

    chart_line_bar = alt.Chart(lines_detail_df).mark_bar().encode(
        x=alt.X("line:O", title="詩行番号"),
        y=alt.Y("anatonic_syllables:Q", title="算入音節数（最終男性強勢母音まで）"),
        color=alt.Color("stanza:N", title="連番号", scale=alt.Scale(scheme="category10")),
        tooltip=[
            alt.Tooltip("line:Q", title="行"),
            alt.Tooltip("stanza:Q", title="連"),
            alt.Tooltip("anatonic_syllables:Q", title="算入音節数"),
            alt.Tooltip("meter_code:N", title="メートル型"),
            alt.Tooltip("caesura_code:N", title="セジュール診断"),
            alt.Tooltip("measures_str:N", title="リズム音尺"),
            alt.Tooltip("rhyme_letter:N", title="脚韻記号"),
            alt.Tooltip("text:N", title="原文"),
        ],
    ).properties(height=260)

    rule_8 = alt.Chart(pd.DataFrame([{"y": 8}])).mark_rule(color="#2563eb", strokeDash=[5, 4], strokeWidth=1.5).encode(y="y:Q")
    rule_12 = alt.Chart(pd.DataFrame([{"y": 12}])).mark_rule(color="#dc2626", strokeDash=[3, 3], strokeWidth=1.5).encode(y="y:Q")
    st.altair_chart(chart_line_bar + rule_8 + rule_12, width="stretch")

    st.subheader("全詩行・4理論統合サマリー表")
    master_cols = [
        "line", "stanza", "anatonic_syllables", "genre", "meter_code",
        "caesura_code", "measures_str", "rhyme_letter", "rhyme_richness",
        "discordances", "meschonnic_scansion", "text",
    ]
    master_rename = {
        "line": "行", "stanza": "連", "anatonic_syllables": "算入音節", "genre": "性(M/F)",
        "meter_code": "メートル型", "caesura_code": "セジュール診断", "measures_str": "リズム音尺",
        "rhyme_letter": "韻", "rhyme_richness": "脚韻の豊かさ", "discordances": "統辞一致/不一致",
        "meschonnic_scansion": "Meschonnic走査", "text": "原文",
    }
    st.dataframe(lines_detail_df[master_cols].rename(columns=master_rename), width="stretch", hide_index=True)

    with st.expander("本ツールの4大理論フレームワーク（Meschonnic / Cornulier / Di Cristo / Gouvard）解説"):
        st.markdown(
            "- **Henri Meschonnic & Gérard Dessons（『Critique du rythme』『Traité du rythme』）**: リズムを定型的な強弱拍の反復ではなく「言説（ディスクール）における主体と意味の組織化」と定義する。リズム群アクセント（`—`）、識別的二重記号（`⏓`）、子音エコー系列が開音節頭に生むプロゾディー・アクセント（`⏑ˈ`）、反アクセント（連続有標音節 `1⁀2`・8類型）、および7種のプロゾディー反響図式を可視化する。\n"
            "- **Benoît de Cornulier（『Théorie du vers』『Art Poëtique』）**: 人間の聴覚が再分割なしに等音節性を直接知覚できる上限を8音節とする「8音節の法則（Loi des 8 syllabes）」を確立した。9音節以上の複合詩句における半句境界を厳密検証し、`F6`（第6音節女性語尾）、`S6`（第7音節女性語尾跨り）、`C6/P6`（前接語跨り）、`M6`（語内分断）および三拍子代替型（`4=4=4`, `8=4`, `4=8`）を客観診断する。\n"
            "- **Albert Di Cristo（『Les musiques du français parlé』）**: フランス語を「語頭アクセント `AI` と句末アクセント `AF/AN` が織りなす双極的（bipolaire）リズム言語」としてモデル化する。4階層構成素（音調単位 `UT` ＜ 韻律語 `MP` ＜ 韻律句 `SP` ＜ イントネーション単位 `UI`）、`{AI … AF}` アクセント弧、優律動性（`EurythmyScore`）、および句末長化を含む/除いた二重 `nPVI` を算出する。\n"
            "- **Jean-Michel Gouvard / Jean Mazaleyrat / Marc Plénat / Michel Murat**: 主観的強勢付与を排した `CPMF` 分布タグ、`e caduc` の6状態分類と古典詩の「母音後 `-e` 語中禁止則」、`muta cum liquida` に基づく義務的分音、音節韻（Rime）骨格に基づく脚韻の豊かさ（貧韻・足韻・富韻・レオニン韻・半諧音）、および句またがり・送り・逆送りを体系的に検出する。"
        )

# -----------------------------------------------------------------------------
# Tab 2: メショニック＝デソン：ディスクール・リズムとプロゾディー網
# -----------------------------------------------------------------------------
with tab_mesch:
    st.subheader("Meschonnic & Dessons 多層ディスクール・リズム走査（Scansion discursive）")
    st.caption(
        "記号凡例: `—` リズム群アクセント（青枠） ／ `—` メートル固定アクセント（橙枠） ／ `⏓` 評価的二重記号 ／ "
        "`⏑ˈ` 子音系列プロゾディー・アクセント（桃枠） ／ `(⏑ˈ)` 行末女性超音節の反跳 ／ `◡` 無強勢音節 ／ "
        "赤上付き数字 `1, 2…` 反アクセント連鎖（`⁀` は休止・句切れ跨ぎ）。桃色太字のIPAはテクスト主系列子音の頭子音エコーを示す。"
    )
    st.html(render_meschonnic_html(mesch["lines"], mesch["dominant_consonant_series"]))

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.subheader("反アクセント（Contre-accents）8類型分布")
        ca_dict = mesch["contre_accents_by_type"]
        if ca_dict:
            ca_rows = [
                {"類型コード": k, "理論分類": CONTRE_ACCENT_LABELS.get(k, k), "出現数": v}
                for k, v in sorted(ca_dict.items(), key=lambda x: -x[1])
            ]
            ca_df = pd.DataFrame(ca_rows)
            ca_chart = alt.Chart(ca_df).mark_bar(color="#dc2626").encode(
                x=alt.X("出現数:Q", title="検出回数"),
                y=alt.Y("理論分類:N", sort="-x", title=None),
                tooltip=["類型コード", "理論分類", "出現数"],
            ).properties(height=220)
            st.altair_chart(ca_chart, width="stretch")
        else:
            st.info("連続する有標音節（反アクセント）は検出されなかった。")

        if mesch["plateaux_rythmiques"]:
            st.markdown("**リズム高原（Plateaux rythmiques: 3音節以上の連続強勢帯）**")
            st.dataframe(
                pd.DataFrame(mesch["plateaux_rythmiques"]).rename(
                    columns={"line": "行", "length": "連続強勢音節数", "words": "該当音節列"}
                ),
                width="stretch",
                hide_index=True,
            )

    with m_col2:
        st.subheader("リズム群末音の音韻動態と空白句読法（Blancs & Ponctuation）")
        fp = mesch["finale_profile"]
        pm = mesch["ponctuation_metrics"]
        f_col1, f_col2, f_col3 = st.columns(3)
        f_col1.metric("母音終止群 [V]#", f"{fp.get('finale_vocalique_pct', 0.0):.1f}%", border=True, help="開かれた終止的・断定的響き")
        f_col2.metric("子音終止群 [C]#", f"{fp.get('finale_consonantique_pct', 0.0):.1f}%", border=True, help="宙吊り・未完結の緊張を生む閉音節末")
        f_col3.metric("長化子音末 [ʁ,z,v,ʒ]#", f"{fp.get('consonne_allongeante_pct', 0.0):.1f}%", border=True, help="先行母音を物理的に引き延ばす持続末音")

        pm_df = pd.DataFrame([
            {"指標": "句読点を含む詩行率", "値": f"{pm.get('punctuated_lines_pct', 0.0):.1f}%"},
            {"指標": "空白句読行率（Ponctuation blanche・無句読点行）", "値": f"{pm.get('ponctuation_blanche_pct', 0.0):.1f}%"},
            {"指標": "連内小文字開始率（行境界の視覚的連続性）", "値": f"{pm.get('lowercase_continuity_pct', 0.0):.1f}%"},
            {"指標": "行＝統辞不一致率（Conflit ligne/syntaxe）", "値": f"{pm.get('conflit_ligne_syntaxe_pct', 0.0):.1f}%"},
            {"指標": "短小リズム群（1〜2音節・スタッカート密度）", "値": f"{pm.get('short_groups_1_2_pct', 0.0):.1f}%"},
            {"指標": "ダッシュ再起動（Tirets de relance）を含む行", "値": f"{pm.get('dash_relance_lines', 0)} 行"},
        ])
        st.dataframe(pm_df, width="stretch", hide_index=True)

        if mesch["volume_contours"]:
            st.markdown("**節単位のグループ音節量輪郭（Volume des groupes）**")
            vc_df = pd.DataFrame([{"輪郭タイプ": k, "節数": v} for k, v in mesch["volume_contours"].items()])
            st.dataframe(vc_df, width="stretch", hide_index=True)

    st.subheader("プロゾディー（「詩の中の詩」）：主系列と7種の反響図式（Figures d'écho）")
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.markdown("**支配的な開子音・母音プロゾディー系列**")
        dom_c_str = " ／ ".join(f"[{d['consonant']}] ({d['count']}回)" for d in mesch["dominant_consonant_series"])
        dom_v_str = " ／ ".join(f"[{d['vowel']}] ({d['count']}回)" for d in mesch["dominant_vowel_series"])
        st.markdown(f"- **主子音系列**: {dom_c_str or 'なし'}\n- **主母音系列**: {dom_v_str or 'なし'}")

        if mesch["superlatifs_prosodiques"]:
            st.markdown("**プロゾディー最上級語（Superlatifs prosodiques：主系列子音を1語内に凝縮する中核語）**")
            st.dataframe(
                pd.DataFrame(mesch["superlatifs_prosodiques"]).rename(
                    columns={"line": "行", "word": "単語", "ipa": "IPA", "consonant": "凝縮子音", "count_in_word": "語内出現数"}
                ),
                width="stretch",
                hide_index=True,
            )

    with s_col2:
        st.markdown("**検出されたプロゾディー反響図式一覧（語頭CV・縮減・拡張・キアスム・CəC反跳・C#C衝突）**")
        if mesch["echo_figures"]:
            st.dataframe(
                pd.DataFrame(mesch["echo_figures"]).rename(
                    columns={"line": "行", "label": "反響図式分類", "example": "該当箇所"}
                )[["行", "反響図式分類", "該当箇所"]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("特筆すべき反響図式は検出されなかった。")

# -----------------------------------------------------------------------------
# Tab 3: コルニュリエ＆グヴァール：定型韻律論・セジュール・脚韻詩法
# -----------------------------------------------------------------------------
with tab_corn_gouv:
    st.subheader("Cornulier「8音節の法則（Loi des 8 syllabes）」とセジュール厳密診断")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("単一詩句率 (≤8音節)", f"{corn['vers_simples_pct']:.1f}%", border=True, help="内的句切れなしに全体知覚される単一脚")
    c2.metric("複合詩句率 (9〜12音節)", f"{corn['vers_composes_pct']:.1f}%", border=True, help="半句(sous-vers ≤ 8)への分割を要する複合脚")
    c3.metric("超定型行率 (>12音節)", f"{corn['hors_metre_pct']:.1f}%", border=True, help="定型認知上限を超えるヴェルセ・自由詩行")
    c4.metric("双子X6 (X6 jumeaux)", f"{corn.get('x6_jumeaux_count', 0)} 組", border=True, help="Cornulier (2004): 脱古典セジュール行(F6/C6/P6/L6/M6/S6)が隣接連続するペア数")
    c5.metric("脚韻の男女交替遵守率", f"{pr_sum['alternance_rate']:.1f}%", border=True, help="異音脚韻隣接時に男性韻(M)と女性韻(F)が交替する割合")

    cg_left, cg_right = st.columns(2)
    with cg_left:
        st.markdown("**アレクサンドラン（12音節行）のセジュール型内訳**")
        if corn.get("caesura_summary"):
            cs_df = pd.DataFrame([
                {"セジュール型": k, "行数": v} for k, v in corn["caesura_summary"].items()
            ])
            cs_chart = alt.Chart(cs_df).mark_bar(color="#1d4ed8").encode(
                x=alt.X("行数:Q", title="行数"),
                y=alt.Y("セジュール型:N", sort="-x", title=None),
                tooltip=["セジュール型", "行数"],
            ).properties(height=200)
            st.altair_chart(cs_chart, width="stretch")
        else:
            st.info("12音節行が含まれていないため、アレクサンドラン専用セジュール集計は対象外。")

    with cg_right:
        st.markdown("**脱落性の e（e caduc / féminin）6状態の内訳**")
        ec_sum = gouv["e_caduc_summary"]
        if ec_sum:
            ec_df = pd.DataFrame([{"状態分類": k, "件数": v} for k, v in ec_sum.items()])
            st.dataframe(ec_df, width="stretch", hide_index=True)
        else:
            st.info("脱落性の e は検出されなかった。")

    st.subheader("行別の Gouvard `CPMF` 分布タグ・三拍子代替可能性・リズム音尺")
    st.caption(
        "`CPMF` 凡例: `C`=単音節前接語（冠詞・主語代名詞等）、`CF`=二音節前接語の女性音節、`P`=単音節前置詞、"
        "`M`=語内前強勢音節（`L`=複合語境界、`Morph`=接頭辞境界）、`F`=語彙語の語末女性音節、`0`=句切れ可能な強勢音節。"
    )
    cpmf_cols = [
        "line", "anatonic_syllables", "meter_code", "caesura_code", "ternary_profile",
        "measures_str", "measure_type", "cpmf_string", "text",
    ]
    st.dataframe(
        lines_detail_df[cpmf_cols].rename(columns={
            "line": "行", "anatonic_syllables": "音節", "meter_code": "メートル",
            "caesura_code": "セジュールコード", "ternary_profile": "三拍子プロファイル",
            "measures_str": "リズム音尺", "measure_type": "音尺類型",
            "cpmf_string": "CPMF音節列", "text": "原文",
        }),
        width="stretch",
        hide_index=True,
    )

    st.subheader("詩節（Strophe）ごとの脚韻スキームと音韻的豊かさ（Plénat / Gouvard）")
    st_df = pd.DataFrame(pr["stanzas_detail"])
    if not st_df.empty:
        st.dataframe(
            st_df.rename(columns={
                "stanza": "連番号", "line_count": "行数", "scheme": "脚韻配列",
                "disposition": "押韻構成型", "genres": "男女韻配列(M/F)", "alternance_respected": "男女交替遵守",
            }),
            width="stretch",
            hide_index=True,
        )

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown("**統辞＝韻律不一致：句またがり（Enjambement）・送り（Rejet）・逆送り（Contre-rejet）**")
        disc_events = gouv["discordances"]["events"]
        if disc_events:
            st.dataframe(
                pd.DataFrame(disc_events).rename(columns={
                    "line": "開始行", "next_line": "到達行", "label": "不一致類型", "span": "境界語句",
                })[["開始行", "到達行", "不一致類型", "境界語句"]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("行末・半句における顕著な統辞不一致（句またがり・送り）は検出されなかった。")

    with d_col2:
        st.markdown("**分音・合音（Diérèse / Synérèse）および母音衝突（Hiatus）検査**")
        if gouv["synerese_dierese_events"]:
            st.dataframe(
                pd.DataFrame(gouv["synerese_dierese_events"]).rename(columns={
                    "line": "行", "word": "対象語", "label": "判定区分", "counted_as_dierese": "2音節算入",
                })[["行", "対象語", "判定区分", "2音節算入"]],
                width="stretch",
                hide_index=True,
            )
        if gouv["hiatus_events"]:
            st.markdown("**検出された語間母音衝突（Hiatus métrique）**")
            st.dataframe(
                pd.DataFrame(gouv["hiatus_events"]).rename(columns={
                    "line": "行", "context": "衝突箇所", "label": "古典詩法上の分類",
                })[["行", "衝突箇所", "古典詩法上の分類"]],
                width="stretch",
                hide_index=True,
            )
        if not gouv["synerese_dierese_events"] and not gouv["hiatus_events"]:
            st.info("特記すべき分音候補や古典禁止イアチュスは検出されなかった。")

# -----------------------------------------------------------------------------
# Tab 4: ディ・クリスト：双極アクセント・韻律句構造・定量リズム指標
# -----------------------------------------------------------------------------
with tab_dicristo:
    st.subheader("Albert Di Cristo 双極アクセント・モデルと4階層構成素（UT < MP < SP < UI）")
    dc1, dc2, dc3, dc4, dc5, dc6 = st.columns(6)
    dc1.metric("優律動スコア", f"{dc['eurythmy_score']:.2f}", border=True, help="2〜4音節の音調単位(UT)比率から強勢衝突・長無強勢ペナルティを引いた均衡度")
    dc2.metric("双極アクセント比", f"{dc['bipolar_ratio']:.2f}", border=True, help="語頭強勢 AI ÷ 句末強勢 (AF+AN)")
    dc3.metric("アクセント弧密度", f"{dc['arc_density_pct']:.1f}%", border=True, help="全韻律句(SP)のうち双極弧 {AI…AF} を形成する割合")
    dc4.metric("内的 nPVI", f"{dc['npvi_interne']:.1f}", border=True, help="UI句末長化を除いた音節間変動指数（仏語等音節性の指標：低いほど均一）")
    dc5.metric("大域 nPVI", f"{dc['npvi_global']:.1f}", border=True, help="AF/AN句末長化スパイクを含めた全体リズム変動指数")
    dc6.metric("%V / ΔC", f"{dc['pct_v_phon']:.1f}% / {dc['delta_c_phon']:.2f}", border=True, help="Ramus / Di Cristo 音素配列リズム指標（母音比率 %V と子音群長標準偏差 ΔC）")

    dc_sub1, dc_sub2, dc_sub3, dc_sub4 = st.columns(4)
    dc_sub1.metric("平均音調単位長 (UT)", f"{dc['ut_mean']:.2f} 音節", f"SD {dc['ut_sd']:.2f}", border=True)
    dc_sub2.metric("平均韻律句長 (SP)", f"{dc['sp_mean']:.2f} 音節", border=True)
    dc_sub3.metric("強勢衝突率 (1音節UT)", f"{dc['clash_rate_pct']:.1f}%", border=True, help="隣接強勢衝突（Clash）の発生率")
    dc_sub4.metric("強勢希薄率 (≥5音節UT)", f"{dc['lapse_rate_pct']:.1f}%", border=True, help="5音節以上の無強勢連続（Lapse）率")

    if dc["ut_distribution"]:
        st.subheader("音調単位（UT / アクセント間隔）の音節長ヒストグラム")
        st.caption("シアン色 (2〜4音節) は Di Cristo が定めるフランス語の優律動（eurythmie）目標帯、赤色は1音節衝突または5音節以上の弛緩帯。")
        ut_rows = [
            {
                "UT音節数": k,
                "出現数": v,
                "領域": "優律動帯 (2〜4音節)" if 2 <= k <= 4 else "非均衡帯 (衝突1 / 弛緩5+)",
            }
            for k, v in sorted(dc["ut_distribution"].items())
        ]
        ut_df = pd.DataFrame(ut_rows)
        ut_chart = alt.Chart(ut_df).mark_bar().encode(
            x=alt.X("UT音節数:O", title="音調単位 (UT) の音節数"),
            y=alt.Y("出現数:Q", title="単位数"),
            color=alt.Color(
                "領域:N",
                scale=alt.Scale(
                    domain=["優律動帯 (2〜4音節)", "非均衡帯 (衝突1 / 弛緩5+)"],
                    range=["#0891b2", "#dc2626"],
                ),
            ),
            tooltip=["UT音節数", "出現数", "領域"],
        ).properties(height=220)
        st.altair_chart(ut_chart, width="stretch")

    st.subheader("行別の双極アクセント弧 `{AI … AF/AN}` と韻律句境界ビュー")
    st.caption("水色上線枠＝双極アクセント弧 `{AI … AF}`、`AI`＝語頭強勢（衝突回避・二重標示・長語分割）、`AF`＝前核韻律句末強勢、`AN`＝イントネーション単位核強勢。")
    st.html(render_di_cristo_html(lines_detail))

# -----------------------------------------------------------------------------
# Tab 5: 音声学・IPA転写（リエゾン考慮）と音素偏差
# -----------------------------------------------------------------------------
with tab_phon:
    st.subheader("音素分析・IPA転写（リエゾン考慮）と標準フランス語分布からの偏差")
    phonetics_data = res["phonetics"]
    deviations = phonetics_data["deviations"]
    all_liaisons = phonetics_data["all_liaisons"]

    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    p_col1.metric("総音素数", f"{deviations['total_phonemes']:,}", border=True)
    p_col2.metric("母音/子音比 (V/C)", f"{deviations['vc_ratio']:.2f}", border=True, help="標準フランス語の基準値は約 0.88")
    v_glide_pct = deviations["category_deviations"].get("vowels_pct", 0.0) + deviations["category_deviations"].get("glides_pct", 0.0)
    p_col3.metric("母音率 (半母音含む)", f"{v_glide_pct:.1f}%", f"{round(v_glide_pct - 46.7, 1):+.1f}%p", border=True)
    c_pct = deviations["category_deviations"].get("consonants_pct", 0.0)
    p_col4.metric("子音率", f"{c_pct:.1f}%", f"{round(c_pct - 53.3, 1):+.1f}%p", border=True)
    p_col5.metric("検出リエゾン数", f"{len(all_liaisons)} 回", border=True, help="後続母音・無音hとの義務的/詩的連音（et・有音hは除外）")

    # -------------------------------------------------------------------------
    # 各詩行における音素数・音素密度の動的変遷（推移ダイナミクス）
    # -------------------------------------------------------------------------
    st.subheader("各詩行における音素数・音素密度の動的変遷（推移ダイナミクス）")
    st.caption("全体総数・平均値のみでは不可視な、行ごとの音素量（音の密度・質量）や母音/子音比の抑揚・緊張・緩和の推移を可視化。")

    if not rows_df.empty:
        # 1. 極値・ダイナミクス要約メトリクス
        max_ph_idx = rows_df["phonemes"].idxmax()
        min_ph_idx = rows_df["phonemes"].idxmin()
        max_den_idx = rows_df["phoneme_density"].idxmax()
        max_vc_idx = rows_df["vc_ratio"].idxmax()

        row_max_ph = rows_df.loc[max_ph_idx]
        row_min_ph = rows_df.loc[min_ph_idx]
        row_max_den = rows_df.loc[max_den_idx]
        row_max_vc = rows_df.loc[max_vc_idx]

        dyn_c1, dyn_c2, dyn_c3, dyn_c4 = st.columns(4)
        dyn_c1.metric(
            "最多音素行（最濃密）",
            f"第{int(row_max_ph['line'])}行: {int(row_max_ph['phonemes'])} 音素",
            f"{int(row_max_ph['vowels'])}母音 / {int(row_max_ph['consonants'])}子音",
            border=True,
            help=f"「{row_max_ph['text']}」"
        )
        dyn_c2.metric(
            "最少音素行（最軽快）",
            f"第{int(row_min_ph['line'])}行: {int(row_min_ph['phonemes'])} 音素",
            f"{int(row_min_ph['vowels'])}母音 / {int(row_min_ph['consonants'])}子音",
            border=True,
            help=f"「{row_min_ph['text']}」"
        )
        dyn_c3.metric(
            "最高音素密度（子音集積）",
            f"第{int(row_max_den['line'])}行: {row_max_den['phoneme_density']:.2f}",
            "音素 / 音節",
            border=True,
            help=f"「{row_max_den['text']}」 (音節数: {int(row_max_den['syllables'])})"
        )
        dyn_c4.metric(
            "最高母音比率（最大開放）",
            f"第{int(row_max_vc['line'])}行: V/C {row_max_vc['vc_ratio']:.2f}",
            f"母音率 {row_max_vc['vowel_pct']:.1f}%",
            border=True,
            help=f"「{row_max_vc['text']}」"
        )

        # 2. 詩行ごとの音素数・母音数・子音数の推移チャート
        chart_col1, chart_col2 = st.columns([3, 2])
        with chart_col1:
            st.markdown("**【音素数推移】総音素数・子音数・母音数の行ごとの変遷**")
            melted_ph = pd.melt(
                rows_df,
                id_vars=["line", "stanza", "text", "ipa", "syllables"],
                value_vars=["phonemes", "consonants", "vowels"],
                var_name="音素種別",
                value_name="音素数"
            )
            melted_ph["音素種別"] = melted_ph["音素種別"].map({
                "phonemes": "総音素数",
                "consonants": "子音数",
                "vowels": "母音数"
            })
            ph_chart = alt.Chart(melted_ph).mark_line(point=True).encode(
                x=alt.X("line:O", title="詩行番号"),
                y=alt.Y("音素数:Q", title="音素数"),
                color=alt.Color(
                    "音素種別:N",
                    scale=alt.Scale(domain=["総音素数", "子音数", "母音数"], range=["#2563eb", "#6366f1", "#f59e0b"]),
                    title="音素種別"
                ),
                tooltip=[
                    alt.Tooltip("line:O", title="詩行"),
                    alt.Tooltip("stanza:O", title="連"),
                    alt.Tooltip("音素種別:N", title="種別"),
                    alt.Tooltip("音素数:Q", title="音素数"),
                    alt.Tooltip("syllables:Q", title="音節数"),
                    alt.Tooltip("text:N", title="原文"),
                    alt.Tooltip("ipa:N", title="IPA"),
                ]
            ).properties(height=320)

            mean_ph_val = float(rows_df["phonemes"].mean())
            mean_ph_rule = alt.Chart(pd.DataFrame([{"mean": mean_ph_val}])).mark_rule(
                color="#64748b", strokeDash=[4, 4], strokeWidth=1.5
            ).encode(
                y="mean:Q",
                tooltip=[alt.Tooltip("mean:Q", title="全行平均音素数", format=".1f")]
            )
            st.altair_chart(ph_chart + mean_ph_rule, width="stretch")

        with chart_col2:
            st.markdown("**【音素密度と響きの質】音節あたり音素密度と母音/子音比の変遷**")
            metric_choice = st.radio(
                "表示指標",
                options=["音素密度 (音素/音節)", "母音/子音比 (V/C ratio)", "母音率 vs 子音率 (%)"],
                horizontal=True,
                label_visibility="collapsed"
            )
            if metric_choice == "音素密度 (音素/音節)":
                den_chart = alt.Chart(rows_df).mark_line(color="#059669", point=alt.OverlayMarkDef(color="#059669", filled=True)).encode(
                    x=alt.X("line:O", title="詩行番号"),
                    y=alt.Y("phoneme_density:Q", title="音素密度（音素数 / 音節数）"),
                    tooltip=[
                        alt.Tooltip("line:O", title="詩行"),
                        alt.Tooltip("phoneme_density:Q", title="音素密度", format=".2f"),
                        alt.Tooltip("phonemes:Q", title="音素数"),
                        alt.Tooltip("syllables:Q", title="音節数"),
                        alt.Tooltip("text:N", title="原文"),
                    ]
                ).properties(height=280)
                mean_den_val = float(rows_df["phoneme_density"].mean())
                den_rule = alt.Chart(pd.DataFrame([{"mean": mean_den_val}])).mark_rule(
                    color="#64748b", strokeDash=[4, 4], strokeWidth=1.5
                ).encode(y="mean:Q", tooltip=[alt.Tooltip("mean:Q", title="平均音素密度", format=".2f")])
                st.altair_chart(den_chart + den_rule, width="stretch")
            elif metric_choice == "母音/子音比 (V/C ratio)":
                vc_chart = alt.Chart(rows_df).mark_line(color="#d97706", point=alt.OverlayMarkDef(color="#d97706", filled=True)).encode(
                    x=alt.X("line:O", title="詩行番号"),
                    y=alt.Y("vc_ratio:Q", title="母音/子音比 (V/C)"),
                    tooltip=[
                        alt.Tooltip("line:O", title="詩行"),
                        alt.Tooltip("vc_ratio:Q", title="V/C比", format=".2f"),
                        alt.Tooltip("vowels:Q", title="母音数"),
                        alt.Tooltip("consonants:Q", title="子音数"),
                        alt.Tooltip("text:N", title="原文"),
                    ]
                ).properties(height=280)
                std_vc_rule = alt.Chart(pd.DataFrame([{"std": 0.88}])).mark_rule(
                    color="#dc2626", strokeDash=[4, 3], strokeWidth=1.5
                ).encode(y="std:Q", tooltip=[alt.Tooltip("std:Q", title="標準フランス語基準値", format=".2f")])
                st.altair_chart(vc_chart + std_vc_rule, width="stretch")
            else:
                pct_melted = pd.melt(
                    rows_df,
                    id_vars=["line", "stanza", "text"],
                    value_vars=["vowel_pct", "consonant_pct"],
                    var_name="比率種別",
                    value_name="比率"
                )
                pct_melted["比率種別"] = pct_melted["比率種別"].map({"vowel_pct": "母音率 (%)", "consonant_pct": "子音率 (%)"})
                pct_chart = alt.Chart(pct_melted).mark_line(point=True).encode(
                    x=alt.X("line:O", title="詩行番号"),
                    y=alt.Y("比率:Q", title="比率 (%)"),
                    color=alt.Color("比率種別:N", scale=alt.Scale(domain=["母音率 (%)", "子音率 (%)"], range=["#f59e0b", "#6366f1"])),
                    tooltip=["line:O", "stanza:O", "比率種別:N", alt.Tooltip("比率:Q", format=".1f"), "text:N"]
                ).properties(height=280)
                fifty_rule = alt.Chart(pd.DataFrame([{"y": 50.0}])).mark_rule(color="#9ca3af", strokeDash=[3, 3]).encode(y="y:Q")
                st.altair_chart(pct_chart + fifty_rule, width="stretch")

        # 3. 連（Stanza）ごとの音素統計と変遷（複数連が存在する場合）
        if len(rows_df["stanza"].unique()) > 1:
            st.markdown("**【連（Stanza）ごとの音素推移サマリー】**")
            stanza_summary = rows_df.groupby("stanza").agg(
                詩行数=("line", "count"),
                総音素数=("phonemes", "sum"),
                行平均音素数=("phonemes", "mean"),
                行平均母音数=("vowels", "mean"),
                行平均子音数=("consonants", "mean"),
                平均音素密度=("phoneme_density", "mean"),
                平均VC比=("vc_ratio", "mean"),
                検出リエゾン=("liaisons", "sum")
            ).reset_index().rename(columns={"stanza": "連番号"})
            st.dataframe(
                stanza_summary.style.format({
                    "行平均音素数": "{:.1f}",
                    "行平均母音数": "{:.1f}",
                    "行平均子音数": "{:.1f}",
                    "平均音素密度": "{:.2f}",
                    "平均VC比": "{:.2f}",
                }),
                width="stretch",
                hide_index=True
            )

    if deviations["top_deviations_positive"] or deviations["top_deviations_negative"]:
        dev_col1, dev_col2 = st.columns(2)
        with dev_col1:
            st.markdown("**【過剰出現】標準より顕著に多い音素（上位5件）**")
            if deviations["top_deviations_positive"]:
                pos_df = pd.DataFrame(deviations["top_deviations_positive"])[
                    ["phoneme", "name", "pct", "ref_pct", "diff", "t_score", "status"]
                ].rename(columns={
                    "phoneme": "音素", "name": "名称", "pct": "テクスト(%)",
                    "ref_pct": "標準(%)", "diff": "差分(%p)", "t_score": "偏差値", "status": "判定",
                })
                st.dataframe(pos_df, width="stretch", hide_index=True)
        with dev_col2:
            st.markdown("**【出現抑制】標準より顕著に少ない音素（下位5件）**")
            if deviations["top_deviations_negative"]:
                neg_df = pd.DataFrame(deviations["top_deviations_negative"])[
                    ["phoneme", "name", "pct", "ref_pct", "diff", "t_score", "status"]
                ].rename(columns={
                    "phoneme": "音素", "name": "名称", "pct": "テクスト(%)",
                    "ref_pct": "標準(%)", "diff": "差分(%p)", "t_score": "偏差値", "status": "判定",
                })
                st.dataframe(neg_df, width="stretch", hide_index=True)

    # -------------------------------------------------------------------------
    # IPA音素・音素連鎖検索と詩行出現推移トラッカー
    # -------------------------------------------------------------------------
    st.subheader("IPA音素・音素連鎖検索と詩行出現推移トラッカー")
    st.caption("任意のIPA音素や音素連鎖（例: 「ɔʁ」、「st」、「ɑ̃」など）を入力して検索し、各詩行での出現回数や推移をグラフで追跡します（カンマ区切りで複数検索可）。")

    all_stats_raw = deviations.get("phoneme_stats", [])
    valid_ph_stats = [p for p in all_stats_raw if p["phoneme"] != "ə"]

    if valid_ph_stats and not rows_df.empty:
        ph_ref_dict = {p["phoneme"]: p.get("ref_pct", 0.0) / 100.0 for p in valid_ph_stats}

        if "ipa_search_query" not in st.session_state:
            st.session_state["ipa_search_query"] = "ɔʁ"

        def set_search_query(q_str):
            st.session_state["ipa_search_query"] = q_str

        st.markdown("**▼ 検索するIPA音素・音素連鎖（例: ɔʁ、カンマ区切りで複数指定可）:**")
        search_query = st.text_input(
            "検索するIPA音素・音素連鎖",
            key="ipa_search_query",
            placeholder="例: ɔʁ, kl, ɑ̃, s",
            label_visibility="collapsed",
            help="単一音素（例: s, k, ɑ̃）だけでなく、2音素以上の連鎖（例: ɔʁ, wa, st）も検索可能です。カンマ区切りで複数指定できます。"
        )

        st.caption("クイック設定（クリックで即時検索）:")
        preset_chips = ["ɔʁ", "wa", "st", "pʁ", "ɑ̃", "ɛ̃", "ɔ̃", "ʁ", "s", "k", "クリア"]
        cols_pre = st.columns(len(preset_chips))
        for col, pre in zip(cols_pre, preset_chips):
            v = "" if pre == "クリア" else pre
            col.button(pre, key=f"btn_pre_{pre}", on_click=set_search_query, args=(v,), use_container_width=True)

        with st.expander("全36音素一覧から追加選択（オプション）", expanded=False):
            col_f1, col_f2 = st.columns([1, 1])
            with col_f1:
                type_filter = st.radio(
                    "調音分類の絞り込み",
                    options=["全音素 (子音＋母音)", "子音のみ", "母音・半母音のみ"],
                    horizontal=True,
                    key="ph_type_filter"
                )
            with col_f2:
                sort_mode = st.radio(
                    "選択肢の並び順",
                    options=["頻出度順（過剰出現 z-score 降順）", "テクスト出現回数順", "音素記号順"],
                    horizontal=True,
                    key="ph_sort_mode"
                )

            if type_filter == "子音のみ":
                selectable_stats = [p for p in valid_ph_stats if p.get("type") == "consonant"]
            elif type_filter == "母音・半母音のみ":
                selectable_stats = [p for p in valid_ph_stats if p.get("type") in ("vowel", "glide")]
            else:
                selectable_stats = list(valid_ph_stats)

            if sort_mode == "頻出度順（過剰出現 z-score 降順）":
                selectable_stats = sorted(
                    selectable_stats,
                    key=lambda x: (x.get("count", 0) > 0, x.get("z_score", -999.0), x.get("count", 0)),
                    reverse=True
                )
            elif sort_mode == "テクスト出現回数順":
                selectable_stats = sorted(
                    selectable_stats,
                    key=lambda x: (x.get("count", 0), x.get("z_score", -999.0)),
                    reverse=True
                )
            else:
                selectable_stats = sorted(selectable_stats, key=lambda x: x["phoneme"])

            ph_options = [p["phoneme"] for p in selectable_stats]

            def format_ph_option(ph_char):
                p = next((item for item in selectable_stats if item["phoneme"] == ph_char), None)
                if not p:
                    return f"[{ph_char}]"
                cnt = p.get("count", 0)
                z = p.get("z_score", 0.0)
                diff = p.get("diff", 0.0)
                pct = p.get("pct", 0.0)
                if cnt > 0 and z >= 1.0:
                    tag = "★ 顕著頻出"
                elif cnt > 0 and z > 0:
                    tag = "▲ 頻出気味"
                elif cnt > 0:
                    tag = "・通常出現"
                else:
                    tag = "○ 出現なし"
                return f"{tag} [{p['phoneme']}] {p['name']} ｜ 出現:{cnt}回 (比率:{pct}%, 偏差:{diff:+0.2f}%p, z={z:+0.2f})"

            selected_from_list = st.multiselect(
                "全36音素から手動選択（初期値は未選択）",
                options=ph_options,
                default=[],
                format_func=format_ph_option,
                help="標準フランス語の音素を選択して検索対象に追加できます。"
            )

        # 検索対象クエリの構築
        raw_queries = [q.strip() for q in search_query.split(",") if q.strip()]
        for ph in selected_from_list:
            if ph not in raw_queries:
                raw_queries.append(ph)

        target_queries = []
        seen_labels = set()
        for q in raw_queries:
            clean_q = q.strip("[]/ ")
            if not clean_q:
                continue
            target_phs = phonetics.extract_phonemes(clean_q.replace("r", "ʁ"))
            if target_phs:
                label = f"[{''.join(target_phs)}]"
            else:
                label = f"[{clean_q}]"
            if label not in seen_labels:
                seen_labels.add(label)
                target_queries.append({
                    "query": q,
                    "clean_q": clean_q,
                    "label": label,
                    "target_phs": target_phs,
                })

        if target_queries:
            track_options = [
                "行内実出現数 (回)",
                "12音節標準換算数 (回/12音節) [長短行の音節数歪みを補正]",
                "標準語期待比 (O/E: 濃縮倍率・標準フランス語=1.0)",
                "行別統計偏差 (z-score: サンプルサイズ歪みを数理補正)",
                "行内音素占有率 (%)",
                "累積出現推移 (回)",
            ]
            track_mode = st.radio(
                "推移グラフの表示指標",
                options=track_options,
                index=0,
                horizontal=True,
                key="ph_track_mode",
                help="各行における実出現回数のほか、長短行の歪みを補正する12音節標準換算数や標準語期待比(O/E)で推移を比較できます。"
            )

            lines_data = phonetics_data.get("lines", [])
            track_records = []
            for l_idx, r in enumerate(res["rows"]):
                l_num = r["line"]
                st_num = r["stanza"]
                l_text = r["text"]
                l_ipa = r["ipa"]
                l_sylls = max(int(r.get("syllables") or r.get("anatonic_syllables") or 1), 1)

                if l_idx < len(lines_data):
                    l_phs = lines_data[l_idx].get("phonemes", [])
                else:
                    l_phs = []
                tot_l_phs = max(len(l_phs), 1)

                for q_item in target_queries:
                    target_phs = q_item["target_phs"]
                    label = q_item["label"]
                    clean_q = q_item["clean_q"]

                    if target_phs:
                        k = len(target_phs)
                        cnt = sum(1 for i in range(len(l_phs) - k + 1) if l_phs[i:i + k] == target_phs)
                    else:
                        clean_line = re.sub(r'[\s‿ˈˌ\.\-\–\—]', '', l_ipa.replace('r', 'ʁ'))
                        clean_target = re.sub(r'[\s‿ˈˌ\.\-\–\—]', '', clean_q.replace('r', 'ʁ'))
                        cnt = clean_line.count(clean_target)

                    norm_12 = round(cnt * 12.0 / l_sylls, 2)
                    pct = round(100.0 * cnt / tot_l_phs, 1)

                    if target_phs:
                        p_ref = 1.0
                        for p in target_phs:
                            p_ref *= ph_ref_dict.get(p, 0.02)
                        p_ref = max(p_ref, 0.0001)
                    else:
                        p_ref = 0.001

                    expected_e = tot_l_phs * p_ref
                    oe_ratio = round(cnt / expected_e, 2) if expected_e > 0.0001 else 0.0

                    denom = math.sqrt(tot_l_phs * p_ref * (1.0 - p_ref))
                    z_line = round((cnt - expected_e) / denom, 2) if denom > 0.0001 else 0.0

                    track_records.append({
                        "line": l_num,
                        "stanza": st_num,
                        "syllables": l_sylls,
                        "phoneme": label,
                        "raw_ph": clean_q,
                        "count": cnt,
                        "norm_12": norm_12,
                        "oe_ratio": oe_ratio,
                        "z_line": z_line,
                        "pct": pct,
                        "expected": round(expected_e, 2),
                        "text": l_text,
                        "ipa": l_ipa,
                    })

            track_df = pd.DataFrame(track_records)
            track_df["cumulative"] = track_df.groupby("phoneme")["count"].cumsum()

            if track_mode.startswith("行内実出現数"):
                val_col = "count"
                y_title = "行内実出現数 (回)"
                val_tooltip_title = "実出現数(回)"
                val_format = "d"
            elif track_mode.startswith("12音節標準換算数"):
                val_col = "norm_12"
                y_title = "12音節標準換算数 (回/12音節)"
                val_tooltip_title = "12音節換算"
                val_format = ".2f"
            elif track_mode.startswith("標準語期待比"):
                val_col = "oe_ratio"
                y_title = "標準語期待比 (O/E: 濃縮倍率)"
                val_tooltip_title = "期待比(O/E)"
                val_format = ".2f"
            elif track_mode.startswith("行別統計偏差"):
                val_col = "z_line"
                y_title = "行別統計偏差スコア (z-score)"
                val_tooltip_title = "統計偏差(z)"
                val_format = "+.2f"
            elif track_mode.startswith("行内音素占有率"):
                val_col = "pct"
                y_title = "行内音素占有率 (%)"
                val_tooltip_title = "行内占有率(%)"
                val_format = ".1f"
            else:
                val_col = "cumulative"
                y_title = "累積出現数 (回)"
                val_tooltip_title = "累積数(回)"
                val_format = "d"

            chart_tr1, chart_tr2 = st.columns([3, 2])
            with chart_tr1:
                st.markdown(f"**【各詩行における推移折れ線グラフ】({y_title})**")
                line_chart = alt.Chart(track_df).mark_line(point=True).encode(
                    x=alt.X("line:O", title="詩行番号"),
                    y=alt.Y(f"{val_col}:Q", title=y_title),
                    color=alt.Color("phoneme:N", title="音素 (IPA)", scale=alt.Scale(scheme="category10")),
                    tooltip=[
                        alt.Tooltip("line:O", title="詩行"),
                        alt.Tooltip("stanza:O", title="連"),
                        alt.Tooltip("syllables:Q", title="音節数"),
                        alt.Tooltip("phoneme:N", title="音素"),
                        alt.Tooltip("count:Q", title="実出現数(回)"),
                        alt.Tooltip("norm_12:Q", title="12音節換算(回)", format=".2f"),
                        alt.Tooltip("oe_ratio:Q", title="標準期待比(O/E)", format=".2f"),
                        alt.Tooltip("z_line:Q", title="行別偏差(z)", format="+.2f"),
                        alt.Tooltip("pct:Q", title="行内占有率(%)", format=".1f"),
                        alt.Tooltip("cumulative:Q", title="累積数(回)"),
                        alt.Tooltip("text:N", title="原文"),
                        alt.Tooltip("ipa:N", title="IPA"),
                    ]
                ).properties(height=320)

                if val_col == "oe_ratio":
                    ref_oe = alt.Chart(pd.DataFrame([{"y": 1.0}])).mark_rule(
                        color="#ef4444", strokeDash=[4, 4], strokeWidth=1.5
                    ).encode(y="y:Q")
                    line_chart = line_chart + ref_oe
                elif val_col == "z_line":
                    ref_z0 = alt.Chart(pd.DataFrame([{"y": 0.0}])).mark_rule(
                        color="#64748b", strokeWidth=1.5
                    ).encode(y="y:Q")
                    ref_zp = alt.Chart(pd.DataFrame([{"y": 1.96}])).mark_rule(
                        color="#f59e0b", strokeDash=[4, 4], strokeWidth=1.2
                    ).encode(y="y:Q")
                    ref_zm = alt.Chart(pd.DataFrame([{"y": -1.96}])).mark_rule(
                        color="#f59e0b", strokeDash=[4, 4], strokeWidth=1.2
                    ).encode(y="y:Q")
                    line_chart = line_chart + ref_z0 + ref_zp + ref_zm

                st.altair_chart(line_chart, width="stretch")

            with chart_tr2:
                st.markdown(f"**【詩行×音素 出現密度ヒートマップ】({val_tooltip_title})**")
                heatmap = alt.Chart(track_df).mark_rect().encode(
                    x=alt.X("line:O", title="詩行番号"),
                    y=alt.Y("phoneme:N", title="音素 (IPA)"),
                    color=alt.Color(f"{val_col}:Q", title=val_tooltip_title, scale=alt.Scale(scheme="blues")),
                    tooltip=[
                        alt.Tooltip("line:O", title="詩行"),
                        alt.Tooltip("syllables:Q", title="音節数"),
                        alt.Tooltip("phoneme:N", title="音素"),
                        alt.Tooltip(f"{val_col}:Q", title=val_tooltip_title, format=val_format),
                        alt.Tooltip("count:Q", title="出現回数"),
                        alt.Tooltip("norm_12:Q", title="12音節換算", format=".2f"),
                        alt.Tooltip("text:N", title="原文"),
                    ]
                ).properties(height=320)

                max_v = float(track_df[val_col].max()) if not track_df.empty else 1.0
                thresh = max_v * 0.65
                text_overlay = alt.Chart(track_df).mark_text(baseline="middle", fontSize=10).encode(
                    x=alt.X("line:O"),
                    y=alt.Y("phoneme:N"),
                    text=alt.condition(f"datum['{val_col}'] != 0", alt.Text(f"{val_col}:Q", format=val_format), alt.value("")),
                    color=alt.condition(f"datum['{val_col}'] >= {thresh} && {thresh} > 0", alt.value("#ffffff"), alt.value("#0f172a"))
                )
                st.altair_chart(heatmap + text_overlay, width="stretch")

            with st.expander("検索音素の行別詳細出現マトリクス表を表示", expanded=False):
                pivot_counts = track_df.pivot(index=["line", "stanza", "syllables", "text"], columns="phoneme", values=val_col).reset_index()
                ph_cols = [q["label"] for q in target_queries if q["label"] in pivot_counts.columns]
                pivot_counts["合計値"] = pivot_counts[ph_cols].sum(axis=1).round(2)
                line_total_map = {r["line"]: max(r.get("phonemes", 1), 1) for r in res["rows"]}
                pivot_counts["行音素総数"] = pivot_counts["line"].map(line_total_map)

                display_cols = ["line", "stanza", "syllables", "text"] + ph_cols + ["合計値", "行音素総数"]
                st.dataframe(
                    pivot_counts[display_cols].rename(columns={
                        "line": "行", "stanza": "連", "syllables": "音節数", "text": "原文", "合計値": f"合計({val_tooltip_title})"
                    }),
                    width="stretch",
                    hide_index=True
                )
        else:
            st.info("上の検索ボックスに検索したいIPA音素または音素連鎖（例: ɔʁ）を入力してください。")

    st.subheader("全36音素の標準偏差スコア（z-score）チャート")
    df_phonemes = pd.DataFrame(deviations["phoneme_stats"])
    if not df_phonemes.empty:
        chart_phonemes = alt.Chart(df_phonemes).mark_bar().encode(
            x=alt.X("z_score:Q", title="標準偏差スコア (z-score)", axis=alt.Axis(grid=True)),
            y=alt.Y("phoneme:N", sort=alt.EncodingSortField(field="z_score", order="descending"), title="音素 (IPA)"),
            color=alt.condition(alt.datum.z_score > 0, alt.value("#2563eb"), alt.value("#dc2626")),
            tooltip=[
                alt.Tooltip("phoneme:N", title="音素"),
                alt.Tooltip("name:N", title="名称"),
                alt.Tooltip("category:N", title="分類"),
                alt.Tooltip("count:Q", title="出現回数"),
                alt.Tooltip("pct:Q", title="テクスト出現率(%)", format=".2f"),
                alt.Tooltip("ref_pct:Q", title="一般標準比率(%)", format=".2f"),
                alt.Tooltip("diff:Q", title="偏差(%p差)", format="+.2f"),
                alt.Tooltip("t_score:Q", title="偏差値(T-Score)", format=".1f"),
                alt.Tooltip("status:N", title="判定"),
            ],
        ).properties(height=620)

        r0 = alt.Chart(pd.DataFrame([{"z": 0}])).mark_rule(color="#4b5563", strokeWidth=1.5).encode(x="z:Q")
        rp1 = alt.Chart(pd.DataFrame([{"z": 1.0}])).mark_rule(color="#9ca3af", strokeDash=[4, 4]).encode(x="z:Q")
        rm1 = alt.Chart(pd.DataFrame([{"z": -1.0}])).mark_rule(color="#9ca3af", strokeDash=[4, 4]).encode(x="z:Q")
        rp2 = alt.Chart(pd.DataFrame([{"z": 2.0}])).mark_rule(color="#f59e0b", strokeDash=[3, 3]).encode(x="z:Q")
        rm2 = alt.Chart(pd.DataFrame([{"z": -2.0}])).mark_rule(color="#f59e0b", strokeDash=[3, 3]).encode(x="z:Q")
        st.altair_chart(chart_phonemes + r0 + rp1 + rm1 + rp2 + rm2, width="stretch")

    cat_d = deviations["category_deviations"]
    if cat_d:
        st.subheader("調音カテゴリ別の構成比")
        cat_df = pd.DataFrame([
            {"カテゴリ": "母音全体", "テクスト(%)": cat_d["vowels_pct"], "一般標準(%)": cat_d["ref_vowels_pct"], "差分(%p)": round(cat_d["vowels_pct"] - cat_d["ref_vowels_pct"], 2)},
            {"カテゴリ": "半母音", "テクスト(%)": cat_d["glides_pct"], "一般標準(%)": cat_d["ref_glides_pct"], "差分(%p)": round(cat_d["glides_pct"] - cat_d["ref_glides_pct"], 2)},
            {"カテゴリ": "子音全体", "テクスト(%)": cat_d["consonants_pct"], "一般標準(%)": cat_d["ref_consonants_pct"], "差分(%p)": round(cat_d["consonants_pct"] - cat_d["ref_consonants_pct"], 2)},
            {"カテゴリ": "鼻母音", "テクスト(%)": cat_d["nasal_vowels_pct"], "一般標準(%)": cat_d["ref_nasal_vowels_pct"], "差分(%p)": round(cat_d["nasal_vowels_pct"] - cat_d["ref_nasal_vowels_pct"], 2)},
            {"カテゴリ": "破裂音", "テクスト(%)": cat_d["plosives_pct"], "一般標準(%)": cat_d["ref_plosives_pct"], "差分(%p)": round(cat_d["plosives_pct"] - cat_d["ref_plosives_pct"], 2)},
            {"カテゴリ": "摩擦音", "テクスト(%)": cat_d["fricatives_pct"], "一般標準(%)": cat_d["ref_fricatives_pct"], "差分(%p)": round(cat_d["fricatives_pct"] - cat_d["ref_fricatives_pct"], 2)},
            {"カテゴリ": "流音 (l, ʁ)", "テクスト(%)": cat_d["liquids_pct"], "一般標準(%)": cat_d["ref_liquids_pct"], "差分(%p)": round(cat_d["liquids_pct"] - cat_d["ref_liquids_pct"], 2)},
        ])
        cc1, cc2 = st.columns([1, 1])
        with cc1:
            st.dataframe(cat_df, width="stretch", hide_index=True)
        with cc2:
            cat_melted = pd.melt(cat_df, id_vars=["カテゴリ"], value_vars=["テクスト(%)", "一般標準(%)"], var_name="対象", value_name="比率(%)")
            cat_chart = alt.Chart(cat_melted).mark_bar().encode(
                x=alt.X("比率(%):Q", title="比率 (%)"),
                y=alt.Y("カテゴリ:N", title=None, sort=None),
                color=alt.Color("対象:N", scale=alt.Scale(domain=["テクスト(%)", "一般標準(%)"], range=["#2563eb", "#9ca3af"])),
                xOffset="対象:N",
                tooltip=["カテゴリ", "対象", alt.Tooltip("比率(%):Q", format=".2f")],
            ).properties(height=260)
            st.altair_chart(cat_chart, width="stretch")

    st.subheader("行ごとの音素推移・IPA転写・リエゾン詳細データ")
    cols_to_show = [
        "line", "stanza", "syllables", "phonemes", "phoneme_density",
        "vowels", "consonants", "vc_ratio", "vowel_pct", "liaisons", "ipa", "text"
    ]
    st.dataframe(
        rows_df[cols_to_show].rename(columns={
            "line": "行", "stanza": "連", "syllables": "音節数", "phonemes": "音素数",
            "phoneme_density": "音素密度 (音素/音節)", "vowels": "母音数", "consonants": "子音数",
            "vc_ratio": "V/C比", "vowel_pct": "母音率(%)", "liaisons": "リエゾン数",
            "ipa": "IPA転写（‿はリエゾン）", "text": "原文",
        }),
        width="stretch",
        hide_index=True,
    )
    if all_liaisons:
        st.markdown("**検出されたリエゾン箇所詳細**")
        st.dataframe(
            pd.DataFrame(all_liaisons)[["line", "context", "consonant"]].rename(
                columns={"line": "行", "context": "リエゾン文脈", "consonant": "発現子音"}
            ),
            width="stretch",
            hide_index=True,
        )

# -----------------------------------------------------------------------------
# Tab 6: 一般フランス語コーパス比較（GSD / ParisStories）・特徴語・出力
# -----------------------------------------------------------------------------
with tab_corpus:
    st.subheader("一般のフランス語との比較（文法・統語・語彙プロファイル）")
    st.caption("赤ひし形（◆ ★ このテクスト）を入力詩の値とし、現代書き言葉（UD French-GSD）および日常話し言葉（UD French-ParisStories）と対照する。")

    with st.expander("比較基準コーパス（UD French-GSD / ParisStories）の学術的出典・エビデンスと留意点"):
        st.markdown(
            "- **書き言葉基準：UD French-GSD（dev+test 1,892文）**\n"
            "  - 出典: Universal Dependencies 公式ツリーバンク (`UD_French-GSD`)。ニュース記事・百科事典・論説等の規範的現代フランス語散文。\n"
            "  - 測定方法: 本アプリと同一の `spaCy fr_core_news_md` パイプラインで再解析し算出した実測平均値。\n"
            "- **話し言葉基準：UD French-ParisStories（test 697発話単位）**\n"
            "  - 出典: パリ第10大学 Modyco 研究所・Rhapsodie 系研究チーム構築の公式音声書き起こしツリーバンク (`UD_French-ParisStories`)。\n"
            "  - 性格: 母語話者の自発的語りにおける代名詞多用・単音節語率・半過去や複合過去の分布を反映。\n"
            "- **解釈上の注意**: コーパス側は「文・発話単位」、詩テクストは「詩行単位」で集計するため、「1単位の語数」や「定動詞を含まない単位（%）」には改行形式の効果が含まれる。"
        )

    gsd_vals = ref["values"]["GSD"]
    ps_vals = ref["values"]["ParisStories"]
    metric_keys = [k for k in LABELS if k in gsd_vals and k in ps_vals]

    norm_records = []
    table_records = []
    for k in metric_keys:
        lbl = LABELS[k]
        t_val = res["metrics"].get(k)
        g_val = gsd_vals.get(k, 0.0)
        p_val = ps_vals.get(k, 0.0)

        table_records.append({
            "指標": lbl,
            "★ このテクスト": "計算不可（100語未満）" if t_val is None else f"{t_val:.3f}" if isinstance(t_val, float) and k in ("word_syl_mean", "zipf_content_mean", "mattr100") else f"{t_val}",
            "書き言葉 (GSD)": f"{g_val:.3f}" if isinstance(g_val, float) and k in ("word_syl_mean", "zipf_content_mean", "mattr100") else f"{g_val}",
            "話し言葉 (ParisStories)": f"{p_val:.3f}" if isinstance(p_val, float) and k in ("word_syl_mean", "zipf_content_mean", "mattr100") else f"{p_val}",
        })

        if t_val is not None:
            row_max = max(float(t_val), float(g_val), float(p_val), 1e-6)
            for src_name, raw_v in [
                ("書き言葉 (GSD)", float(g_val)),
                ("話し言葉 (ParisStories)", float(p_val)),
                ("★ このテクスト", float(t_val)),
            ]:
                norm_records.append({
                    "指標": lbl,
                    "資料": src_name,
                    "実測値": raw_v,
                    "相対スケール(%)": round(100.0 * raw_v / row_max, 1),
                })

    if norm_records:
        norm_df = pd.DataFrame(norm_records)
        ref_only_df = norm_df[norm_df["資料"] != "★ このテクスト"]
        this_only_df = norm_df[norm_df["資料"] == "★ このテクスト"]

        ref_points = alt.Chart(ref_only_df).mark_point(filled=True, size=95, opacity=0.85).encode(
            y=alt.Y("指標:N", sort=None, title=None),
            x=alt.X("相対スケール(%):Q", title="各指標内の最大値に対する相対位置（%・ツールチップに実測値）", scale=alt.Scale(domain=[0, 105])),
            color=alt.Color(
                "資料:N",
                scale=alt.Scale(
                    domain=["★ このテクスト", "書き言葉 (GSD)", "話し言葉 (ParisStories)"],
                    range=["#dc2626", "#2563eb", "#d97706"],
                ),
                legend=alt.Legend(title="比較対象（赤◆＝このテクスト）", orient="top"),
            ),
            shape=alt.Shape(
                "資料:N",
                scale=alt.Scale(
                    domain=["★ このテクスト", "書き言葉 (GSD)", "話し言葉 (ParisStories)"],
                    range=["diamond", "circle", "triangle-up"],
                ),
            ),
            tooltip=["指標", "資料", alt.Tooltip("実測値:Q", format=".3f"), alt.Tooltip("相対スケール(%):Q", format=".1f")],
        )

        guide_lines = alt.Chart(norm_df).mark_line(color="#e2e8f0", strokeWidth=2).encode(
            y=alt.Y("指標:N", sort=None),
            x=alt.X("相対スケール(%):Q"),
            detail="指標:N",
        )

        this_diamond = alt.Chart(this_only_df).mark_point(
            filled=True, size=260, shape="diamond", color="#dc2626", stroke="#7f1d1d", strokeWidth=1.5
        ).encode(
            y=alt.Y("指標:N", sort=None),
            x=alt.X("相対スケール(%):Q"),
            tooltip=["指標", "資料", alt.Tooltip("実測値:Q", format=".3f"), alt.Tooltip("相対スケール(%):Q", format=".1f")],
        )

        st.altair_chart((guide_lines + ref_points + this_diamond).properties(height=620), width="stretch")

    comp_df = pd.DataFrame(table_records)

    def _highlight_this_text(col):
        if col.name == "★ このテクスト":
            return ["background-color: #fef2f2; font-weight: 700; color: #991b1b;" for _ in col]
        return ["" for _ in col]

    st.dataframe(comp_df.style.apply(_highlight_this_text, axis=0), width="stretch", hide_index=True)

    st.subheader("時制の内訳（定動詞中の%）")
    all_tenses = sorted(set(
        list(res["tense_share"].keys())
        + list(gsd_vals.get("tense_share", {}).keys())
        + list(ps_vals.get("tense_share", {}).keys())
    ))
    tense_records = []
    for corpus_label, share in [
        ("★ このテクスト", res["tense_share"]),
        ("書き言葉 (GSD)", gsd_vals.get("tense_share", {})),
        ("話し言葉 (ParisStories)", ps_vals.get("tense_share", {})),
    ]:
        for t_name in all_tenses:
            pct = share.get(t_name, 0.0)
            if pct > 0:
                tense_records.append({"corpus": corpus_label, "tense": t_name, "pct": pct})

    if tense_records:
        df_tense = pd.DataFrame(tense_records)
        chart_tense = alt.Chart(df_tense).mark_bar().encode(
            y=alt.Y("corpus:N", title=None, sort=["★ このテクスト", "書き言葉 (GSD)", "話し言葉 (ParisStories)"]),
            x=alt.X("pct:Q", stack="normalize", title="割合", axis=alt.Axis(format="%")),
            color=alt.Color("tense:N", title="時制", scale=alt.Scale(scheme="tableau20")),
            tooltip=[
                alt.Tooltip("corpus:N", title="資料"),
                alt.Tooltip("tense:N", title="時制"),
                alt.Tooltip("pct:Q", title="割合(%)", format=".2f"),
            ],
        ).properties(height=180)
        st.altair_chart(chart_tense, width="stretch")
    else:
        st.info("定動詞が検出されなかった。")

    st.subheader("特徴語（Keyness：4回以上出現し一般頻度より突出する実詞）")
    if res["keyness"]:
        st.dataframe(
            pd.DataFrame(res["keyness"]).rename(
                columns={"lemma": "見出し語", "count": "出現回数", "log2_ratio": "対数比 (log2)", "zipf": "一般Zipf頻度"}
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("4回以上出現する特徴語は見つからない。")

    st.subheader("全解析データのエクスポート")
    exp_c1, exp_c2, exp_c3 = st.columns(3)
    with exp_c1:
        st.download_button(
            "行別データ（CSV：韻律・走査・IPA含む）を保存",
            rows_df.to_csv(index=False).encode("utf-8-sig"),
            "lines_prosody_phonetics.csv",
            "text/csv",
            width="stretch",
        )
    with exp_c2:
        st.download_button(
            "詩行概要テーブル（CSV）を保存",
            pd.DataFrame([{"項目": k, "値": v} for k, v in res["lines_summary"].items()]).to_csv(index=False).encode("utf-8-sig"),
            "summary.csv",
            "text/csv",
            width="stretch",
        )
    with exp_c3:
        export_json = json.dumps(res, ensure_ascii=False, indent=2)
        st.download_button(
            "4理論統合・全指標データ（JSON）を保存",
            export_json.encode("utf-8"),
            "prosody_rhythm_full.json",
            "application/json",
            width="stretch",
        )
