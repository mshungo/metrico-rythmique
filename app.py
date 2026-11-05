"""Analyse métrique, rythmique et phonétique du vers français.
Plateforme unifiée d'analyse poétique : scansion, positions du rythme, chaînes phonématiques et rimes.
Lancement : streamlit run app.py
"""
import collections
import html
import importlib
import json
import math
import re
import altair as alt
import pandas as pd
import streamlit as st

import analysis
import phonetics
import prosody_rhythm
importlib.reload(phonetics)
importlib.reload(prosody_rhythm)
importlib.reload(analysis)
from analysis import analyze, load_reference

LABELS_FR = {
    "punct_per100": "Ponctuation (pour 100 mots)",
    "words_per_unit_mean": "Mots par unité (moyenne)",
    "noverb_unit_pct": "Unités sans verbe conjugué (%)",
    "finite_per100": "Verbes conjugués (pour 100 mots)",
    "cconj_per100": "Conjonctions de coordination (pour 100 mots)",
    "sconj_per100": "Conjonctions de subordination (pour 100 mots)",
    "rel_per100": "Pronoms relatifs (pour 100 mots)",
    "word_syl_mean": "Syllabes par mot (moyenne)",
    "mono_pct": "Mots monosyllabiques (%)",
    "zipf_content_mean": "Fréquence Zipf (mots pleins)",
    "rare_lt3_pct": "Vocabulaire rare Zipf < 3 (%)",
    "mattr100": "Diversité lexicale (MATTR-100)",
    "p1sg_pct": "1re personne du singulier (verbes %)",
    "taut_per1000": "Répétitions lexicales proches (/1000 mots)",
    "pos_NOUN": "Noms (%)",
    "pos_VERB": "Verbes (%)",
    "pos_AUX": "Auxiliaires (%)",
    "pos_ADJ": "Adjectifs (%)",
    "pos_ADV": "Adverbes (%)",
    "pos_DET": "Déterminants (%)",
    "pos_PRON": "Pronoms (%)",
    "pos_ADP": "Prépositions (%)",
    "pos_CCONJ": "Conjonctions de coordination (%)",
    "pos_SCONJ": "Conjonctions de subordination (%)",
    "pos_PROPN": "Noms propres (%)",
    "pos_NUM": "Numéraux (%)",
}

POS_NAMES_FR = {
    "NOUN": "Nom",
    "VERB": "Verbe",
    "ADJ": "Adjectif",
    "ADV": "Adverbe",
    "PROPN": "Nom propre",
    "AUX": "Auxiliaire",
    "DET": "Déterminant",
    "PRON": "Pronom",
    "ADP": "Préposition",
    "CCONJ": "Coordination",
    "SCONJ": "Subordination",
    "NUM": "Numéral",
    "INTJ": "Interjection",
    "PUNCT": "Ponctuation",
    "SYM": "Symbole",
    "X": "Autre",
}

POS_COLORS = {
    "NOUN": "#1d4ed8",
    "PROPN": "#2563eb",
    "VERB": "#dc2626",
    "AUX": "#f87171",
    "ADJ": "#16a34a",
    "ADV": "#ca8a04",
    "PRON": "#9333ea",
    "DET": "#64748b",
    "ADP": "#0891b2",
    "CCONJ": "#ea580c",
    "SCONJ": "#c2410c",
    "NUM": "#0d9488",
    "INTJ": "#ec4899",
    "OTHER": "#94a3b8",
}

SAMPLE_TEXTS = {
    "(Saisie libre ou import de fichier)": "",
    "1. Baudelaire : Correspondances (Alexandrins classiques 12v)": (
        "La Nature est un temple où de vivants piliers\n"
        "Laissent parfois sortir de confuses paroles ;\n"
        "L'homme y passe à travers des forêts de symboles\n"
        "Qui l'observent avec des regards familiers."
    ),
    "2. Hugo / Verlaine : Césures romantiques, trimètre & rejets": (
        "Toujours aimer, toujours souffrir, toujours mourir,\n"
        "J'ai vu l'ombre passer sur le front des poètes ;\n"
        "Souvenir, souvenir, que me veux-tu ? L'automne\n"
        "Faisait voler la grive à travers l'air atone."
    ),
    "3. Apollinaire : Vers libre & ponctuation blanche": (
        "Sous le pont Mirabeau coule la Seine\n"
        "Et nos amours\n"
        "Faut-il qu'il m'en souvienne\n"
        "La joie venait toujours après la peine"
    ),
    "4. Rimbaud : Le Bateau ivre (12 alexandrins - rimes & échos)": (
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


@st.cache_resource
def load_spacy_model():
    """Charge le modèle linguistique spaCy."""
    import spacy
    return spacy.load("fr_core_news_md")


def format_scansion_mark_html(mark: str) -> str:
    """Rendu typographique de la scansion : place l'accent prosodique verticalement au-dessus de la brève."""
    if not mark:
        return "&nbsp;"
    if "⏑ˈ" in mark or "◡ˈ" in mark:
        is_paren = mark.startswith("(") and mark.endswith(")")
        stacked = (
            '<span class="mesch-stacked">'
            '<span class="mesch-stacked-top">ˈ</span>'
            '<span class="mesch-stacked-base">⏑</span>'
            '</span>'
        )
        return f"({stacked})" if is_paren else stacked
    return html.escape(mark)


def render_scansion_html(mesch_lines, dom_consonants):
    """Rendu visuel de la scansion syllabique et accentuelle par vers."""
    dom_set = {d["consonant"] for d in dom_consonants[:3]}
    cards = [
        """<style>
        .mesch-box { font-family: 'Segoe UI', -apple-system, sans-serif; margin-bottom: 10px; padding: 10px 12px; border: 1px solid #e2e8f0; border-radius: 8px; background: #ffffff; }
        .mesch-hdr { font-size: 0.82rem; color: #475569; margin-bottom: 6px; font-weight: 600; }
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
        .mesch-mark { font-size: 1.05rem; font-weight: 700; color: #0f172a; line-height: 1.2; min-height: 22px; display: inline-flex; align-items: center; justify-content: center; }
        .mesch-stacked { display: inline-flex; flex-direction: column; align-items: center; justify-content: center; line-height: 1; position: relative; top: -1px; }
        .mesch-stacked-top { font-size: 0.65rem; line-height: 0.45; margin-bottom: -1px; font-weight: 900; color: #0f172a; }
        .mesch-stacked-base { font-size: 1.05rem; line-height: 0.9; color: #0f172a; }
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
            mark_html = format_scansion_mark_html(tok.get("mark"))
            cells_html.append(
                f'<div class="mesch-cell {cls}">'
                f'<span class="mesch-sup">{html.escape(sup_txt)}</span>'
                f'<span class="mesch-mark">{mark_html}</span>'
                f'<span class="mesch-orth">{orth_esc}</span>'
                f'<span class="{ipa_cls}">[{ipa_esc}]</span>'
                f'</div>'
            )
            if tok["caesura_after"]:
                cells_html.append('<span class="mesch-bar" title="Césure">│</span>')
            elif tok["is_word_end"]:
                cells_html.append('<span class="mesch-gap"></span>')

        vol_str = " + ".join(map(str, l_obj["group_volumes"])) if l_obj["group_volumes"] else "-"
        cards.append(
            f'<div class="mesch-box">'
            f'<div class="mesch-hdr">Vers {l_obj["line"]} ｜ Mesures rythmiques : [{vol_str}]</div>'
            f'<div class="mesch-row">{"".join(cells_html)}</div>'
            f'</div>'
        )
    return "\n".join(cards)


def render_phrasing_html(lines_detail):
    """Rendu visuel du phrasé prosodique et des arcs accentuels."""
    blocks = [
        """<style>
        .dc-box { font-family: 'Segoe UI', -apple-system, sans-serif; margin-bottom: 8px; padding: 9px 12px; border: 1px solid #e2e8f0; border-radius: 8px; background: #ffffff; }
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
            f'<div class="dc-hdr">Vers {row["line"]} ｜ Phrases prosodiques : {row["sp_count"]} ｜ Arcs accentuels : {row["arcs_count"]} ｜ Phrasé : {html.escape(row["di_cristo_phrasing"])}</div>'
            f'<div class="dc-row">{"".join(spans)}</div>'
            f'</div>'
        )
    return "\n".join(blocks)


def render_pos_text_html(rows_data, filter_tag="TOUS"):
    """Rendu visuel du texte avec étiquetage morphosyntaxique (POS) coloré."""
    css = """<style>
    .pos-box { font-family: 'Segoe UI', -apple-system, sans-serif; margin-bottom: 8px; padding: 8px 12px; border: 1px solid #e2e8f0; border-radius: 8px; background: #ffffff; }
    .pos-hdr { font-size: 0.78rem; color: #64748b; margin-bottom: 5px; font-weight: 600; }
    .pos-line { display: flex; flex-wrap: wrap; gap: 5px; align-items: center; }
    .pos-token { display: inline-flex; align-items: baseline; padding: 2px 7px; border-radius: 4px; font-size: 0.88rem; border: 1px solid transparent; font-weight: 500; transition: opacity 0.2s; }
    .pos-lbl { font-size: 0.65rem; margin-left: 4px; font-weight: 700; opacity: 0.85; }
    .pos-dim { opacity: 0.20; filter: grayscale(90%); }
    </style>"""
    lines_html = [css]
    for r in rows_data:
        toks = r.get("word_tokens", [])
        if not toks:
            continue
        spans = []
        filter_count = 0
        for t in toks:
            tag = t.get("pos", "OTHER")
            lbl = POS_NAMES_FR.get(tag, tag)
            color = POS_COLORS.get(tag, "#94a3b8")
            dim_cls = ""
            if filter_tag != "TOUS":
                if filter_tag == tag:
                    filter_count += 1
                else:
                    dim_cls = "pos-dim"
            spans.append(
                f'<span class="pos-token {dim_cls}" style="background:{color}18; color:{color}; border-color:{color}40;">'
                f'{html.escape(t["text"])}<span class="pos-lbl">{html.escape(lbl)}</span>'
                f'</span>'
            )
        rhyme_cat = r.get("rhyme_pos", "-")
        rhyme_cat_lbl = POS_NAMES_FR.get(rhyme_cat, rhyme_cat)
        extra_hdr = ""
        if filter_tag != "TOUS":
            filt_lbl = POS_NAMES_FR.get(filter_tag, filter_tag)
            extra_hdr = f' ｜ <span style="color:#0f172a; font-weight:700;">{html.escape(filt_lbl)} : {filter_count}</span>'

        lines_html.append(
            f'<div class="pos-box">'
            f'<div class="pos-hdr">Vers {r["line"]}{extra_hdr} ｜ Mots pleins : {r.get("content_words", 0)} ｜ Mots outils : {r.get("function_words", 0)} ｜ Rime : {html.escape(rhyme_cat_lbl)} ({html.escape(rhyme_cat)})</div>'
            f'<div class="pos-line">{"".join(spans)}</div>'
            f'</div>'
        )
    return "\n".join(lines_html)


st.set_page_config(page_title="Analyse métrique, rythmique & phonétique", layout="wide")
st.title("Analyse métrique, rythmique & phonétique du vers français")
st.caption("Plateforme unifiée : Mètre & Césure ｜ Positions du rythme & Accents ｜ Chaînes phonématiques & Rimes")

with st.sidebar:
    st.header("Paramètres d'analyse")
    preset_choice = st.selectbox(
        "Textes poétiques d'exemple",
        options=list(SAMPLE_TEXTS.keys()),
        index=0,
        help="Sélectionnez un texte pour le charger instantanément dans l'analyseur."
    )
    fold_indented = st.checkbox(
        "Fusionner les vers brisés (indentation)",
        value=False,
        help="Rattache les lignes débutant par une indentation au vers précédent."
    )
    dierese = st.checkbox(
        "Diérèse poétique classique",
        value=True,
        help="Applique les diérèses étymologiques classiques (ex. niais, furieux, passion)."
    )

up = st.file_uploader("Fichier texte (UTF-8)", type=["txt"])
default_text = SAMPLE_TEXTS.get(preset_choice, "")
text = up.read().decode("utf-8") if up else st.text_area("Texte poétique à analyser", value=default_text, height=190)

if not text or not text.strip():
    st.info("Veuillez saisir un texte poétique ou sélectionner un poème d'exemple dans la barre latérale.")
    st.stop()
    raise SystemExit

nlp = load_spacy_model()
with st.spinner("Analyse métrique, rythmique et phonétique en cours..."):
    res = analyze(text, dierese=dierese, fold_indented=fold_indented, nlp=nlp)
ref = load_reference()

rows_data = res["rows"]
if not rows_data:
    st.warning("Aucun vers exploitable détecté.")
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
lines_detail_df["rhyme_pos"] = [r.get("rhyme_pos", "") for r in rows_data]

# =============================================================================
# Interface unifiée en 5 volets thématiques
# =============================================================================
tab_vers, tab_rythme, tab_phon_chain, tab_rimes, tab_phon_dist = st.tabs([
    "1. Métrique & Vers par vers",
    "2. Rythme & Positions accentuelles",
    "3. Chaînes phonématiques & Recherche IPA",
    "4. Rimes & Structure strophique",
    "5. Morphosyntaxe (POS) & Stylométrie",
])

# -----------------------------------------------------------------------------
# Onglet 1 : Métrique & Vers par vers
# -----------------------------------------------------------------------------
with tab_vers:
    st.info(
        f"**Forme poétique globale** : {pr_sum['global_poetic_form']} ｜ "
        f"**Structure strophique** : {pr_sum['fixed_strophic_form']} ｜ "
        f"**Mètre dominant** : {pr_sum['dominant_meter']}"
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Isométrie", f"{pr_sum['isometry_ratio']:.1f}%", border=True, help="Pourcentage de vers conformes au mètre dominant")
    k2.metric("Syllabes ouvertes (CV)", f"{pr_sum['open_syl_ratio_pct']:.1f}%", border=True, help="Taux d'enchaînement en syllabes ouvertes")
    k3.metric("Régularité rythmique", f"{pr_sum['eurythmy_score']:.2f}", border=True, help="Indice d'équilibre des unités rythmiques")
    k4.metric("Contre-accents", f"{pr_sum['total_contre_accents']}", border=True, help="Tensions accentuelles adjacentes")
    k5.metric("Échos phonématiques", f"{pr_sum['total_echo_figures']}", border=True, help="Figures d'écho et répétitions phoniques")
    k6.metric("Discordances", f"{pr_sum['total_discordances']}", border=True, help="Enjambements, rejets et contre-rejets")

    st.subheader("Longueur métrique des vers et seuil de l'alexandrin")
    chart_line_bar = alt.Chart(lines_detail_df).mark_bar().encode(
        x=alt.X("line:O", title="Numéro du vers"),
        y=alt.Y("anatonic_syllables:Q", title="Nombre de syllabes comptées"),
        color=alt.Color("stanza:N", title="Strophe", scale=alt.Scale(scheme="category10")),
        tooltip=[
            alt.Tooltip("line:Q", title="Vers"),
            alt.Tooltip("stanza:Q", title="Strophe"),
            alt.Tooltip("anatonic_syllables:Q", title="Syllabes"),
            alt.Tooltip("meter_code:N", title="Mètre"),
            alt.Tooltip("caesura_code:N", title="Césure"),
            alt.Tooltip("measures_str:N", title="Mesures"),
            alt.Tooltip("rhyme_letter:N", title="Rime"),
            alt.Tooltip("text:N", title="Texte"),
        ],
    ).properties(height=240)
    rule_8 = alt.Chart(pd.DataFrame([{"y": 8}])).mark_rule(color="#2563eb", strokeDash=[5, 4], strokeWidth=1.5).encode(y="y:Q")
    rule_12 = alt.Chart(pd.DataFrame([{"y": 12}])).mark_rule(color="#dc2626", strokeDash=[3, 3], strokeWidth=1.5).encode(y="y:Q")
    st.altair_chart(chart_line_bar + rule_8 + rule_12, width="stretch")

    st.subheader("Tableau métrique complet des vers")
    master_cols = [
        "line", "stanza", "anatonic_syllables", "genre", "meter_code",
        "caesura_code", "measures_str", "rhyme_letter", "rhyme_richness",
        "rhyme_pos", "discordances", "text", "ipa",
    ]
    master_rename = {
        "line": "Vers", "stanza": "Strophe", "anatonic_syllables": "Syllabes", "genre": "Genre",
        "meter_code": "Mètre", "caesura_code": "Césure / Hémistiches", "measures_str": "Mesures rythmiques",
        "rhyme_letter": "Rime", "rhyme_richness": "Richesse", "rhyme_pos": "Catégorie rime",
        "discordances": "Discordance syntaxique", "text": "Texte", "ipa": "IPA",
    }
    st.dataframe(lines_detail_df[master_cols].rename(columns=master_rename), width="stretch", hide_index=True)

    st.subheader("Scansion rythmique et découpage syllabique")
    st.html(
        '<div style="font-size:0.85rem; color:#64748b; margin-bottom:10px;">'
        '<strong>Légende :</strong> '
        '<code style="font-weight:700;">—</code> Accent tonique / métrique ｜ '
        '<code style="font-weight:700;">⏓</code> Accent expressif ｜ '
        '<span style="background:#f1f5f9; padding:1px 5px; border-radius:4px; font-family:monospace; font-weight:700; display:inline-flex; flex-direction:column; align-items:center; vertical-align:middle; line-height:1; position:relative; top:-2px;">'
        '<span style="font-size:0.65em; line-height:0.45; margin-bottom:-1px; color:#0f172a;">ˈ</span>'
        '<span style="font-size:1em; line-height:0.9; color:#0f172a;">⏑</span>'
        '</span> Accent prosodique (écho phonématique) ｜ '
        '<code style="font-weight:700;">◡</code> Syllabe atone ｜ '
        '<code style="font-weight:700;">│</code> Césure ｜ '
        '<span style="color:#be185d; font-weight:700;">Rose</span> : Écho consonantique dominant'
        '</div>'
    )
    st.html(render_scansion_html(mesch["lines"], mesch["dominant_consonant_series"]))

    with st.expander("Inspecteur détaillé d'un vers (Zoom métrique et rythmique)", expanded=False):
        selected_line_idx = st.selectbox(
            "Choisir un vers à inspecter",
            options=list(range(1, len(lines_detail) + 1)),
            format_func=lambda x: f"Vers {x} : {lines_detail[x-1]['text']}"
        )
        sel_row = lines_detail[selected_line_idx - 1]
        c_insp1, c_insp2, c_insp3, c_insp4 = st.columns(4)
        c_insp1.metric("Mètre compté", f"{sel_row['anatonic_syllables']} syllabes ({sel_row['genre']})")
        c_insp2.metric("Césure & Hémistiches", sel_row['caesura_code'])
        c_insp3.metric("Mesures rythmiques", sel_row['measures_str'])
        sel_rhyme_pos = rows_data[selected_line_idx - 1].get("rhyme_pos", "")
        sel_rhyme_pos_lbl = POS_NAMES_FR.get(sel_rhyme_pos, sel_rhyme_pos)
        c_insp4.metric("Rime", f"{sel_row['rhyme_letter']} [{sel_row['rhyme_ipa']}] ({sel_row['rhyme_richness']})")
        st.markdown(f"- **Texte** : {sel_row['text']}")
        st.markdown(f"- **Transcription IPA** : `{sel_row['ipa']}`")
        st.markdown(f"- **Catégorie grammaticale de la rime (POS)** : {sel_rhyme_pos_lbl} (`{sel_rhyme_pos}`)")
        st.markdown(f"- **Discordance syntaxique** : {sel_row['discordances']}")
        st.markdown(f"- **Phrasé prosodique** : `{sel_row['di_cristo_phrasing']}`")


# -----------------------------------------------------------------------------
# Onglet 2 : Rythme & Positions accentuelles
# -----------------------------------------------------------------------------
with tab_rythme:
    st.subheader("Positions des accents et profil rythmique du poème")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Vers simples (≤ 8 syllabes)", f"{corn['vers_simples_pct']:.1f}%", border=True)
    c2.metric("Vers composés (9 à 12 syllabes)", f"{corn['vers_composes_pct']:.1f}%", border=True)
    c3.metric("Hors mètre (> 12 syllabes)", f"{corn['hors_metre_pct']:.1f}%", border=True)
    c4.metric("Indice d'eurythmie", f"{dc['eurythmy_score']:.2f}", border=True)
    c5.metric("Discordances syntaxiques", f"{len(gouv['discordances']['events'])}", border=True)

    # Extraction et agrégation des positions accentuelles dans le vers
    pos_records = []
    for r in lines_detail:
        for s_idx, tok in enumerate(r.get("meschonnic_tokens", [])):
            s_val = tok.get("syl_idx")
            pos = (s_val + 1) if s_val is not None else (s_idx + 1)
            m = tok.get("mark", "")
            if m in ("—", "⏓", "⏑ˈ"):
                m_label = (
                    "Accent tonique / métrique (—)" if m == "—"
                    else ("Accent expressif (⏓)" if m == "⏓"
                    else "Accent prosodique (⏑ˈ)")
                )
                pos_records.append({
                    "position": pos,
                    "type_accent": m_label,
                    "mark": m,
                    "vers": r["line"],
                    "mot": tok["word_text"],
                    "syllabe": tok["orth"],
                })

    st.subheader("Distribution des positions accentuelles dans le vers (Topographie du rythme)")
    st.caption("Fréquence des accents selon la position de la syllabe dans le vers (1 à N). Permet d'observer instantanément les sommets rythmiques réguliers (ex. positions 6 et 12 dans l'alexandrin).")
    if pos_records:
        pos_df = pd.DataFrame(pos_records)
        chart_pos = alt.Chart(pos_df).mark_bar().encode(
            x=alt.X("position:O", title="Position de la syllabe dans le vers (1 à N)"),
            y=alt.Y("count():Q", title="Nombre d'accents recensés"),
            color=alt.Color(
                "type_accent:N",
                title="Type d'accent",
                scale=alt.Scale(
                    domain=["Accent tonique / métrique (—)", "Accent expressif (⏓)", "Accent prosodique (⏑ˈ)"],
                    range=["#1d4ed8", "#d97706", "#be185d"]
                )
            ),
            tooltip=[
                alt.Tooltip("position:O", title="Position syllabique"),
                alt.Tooltip("type_accent:N", title="Type d'accent"),
                alt.Tooltip("count():Q", title="Nombre d'occurrences"),
            ]
        ).properties(height=280)
        st.altair_chart(chart_pos, width="stretch")
    else:
        st.info("Aucun accent rythmique notable répertorié.")

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown("**Types de césure (pour les vers de 12 syllabes)**")
        if corn.get("caesura_summary"):
            cs_df = pd.DataFrame([{"Type de césure": k, "Nombre de vers": v} for k, v in corn["caesura_summary"].items()])
            cs_chart = alt.Chart(cs_df).mark_bar(color="#1d4ed8").encode(
                x=alt.X("Nombre de vers:Q", title="Nombre de vers"),
                y=alt.Y("Type de césure:N", sort="-x", title=None),
                tooltip=["Type de césure", "Nombre de vers"],
            ).properties(height=200)
            st.altair_chart(cs_chart, width="stretch")
        else:
            st.info("Aucun vers de 12 syllabes dans ce poème.")

    with r_col2:
        st.markdown("**Mesures rythmiques des vers (Tétramètres & Trimètres)**")
        measures_summary = collections.Counter(r["measures_str"] for r in lines_detail if r.get("measures_str"))
        if measures_summary:
            m_sum_df = pd.DataFrame([{"Mesure": k, "Vers": v} for k, v in measures_summary.most_common(8)])
            m_chart = alt.Chart(m_sum_df).mark_bar(color="#0891b2").encode(
                x=alt.X("Vers:Q", title="Nombre de vers"),
                y=alt.Y("Mesure:N", sort="-x", title=None),
                tooltip=["Mesure", "Vers"],
            ).properties(height=200)
            st.altair_chart(m_chart, width="stretch")

    st.subheader("Discordances syntaxe / mètre (Enjambements, Rejets, Contre-rejets)")
    disc_events = gouv["discordances"]["events"]
    if disc_events:
        st.dataframe(
            pd.DataFrame(disc_events).rename(columns={
                "line": "Vers début", "next_line": "Vers fin", "label": "Type de discordance", "span": "Segments textuels",
            })[["Vers début", "Vers fin", "Type de discordance", "Segments textuels"]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Aucun enjambement ou rejet marqué détecté (concordance régulière).")

    st.subheader("Phrasé prosodique et arcs accentuels")
    st.html(render_phrasing_html(lines_detail))


# -----------------------------------------------------------------------------
# Onglet 3 : Chaînes phonématiques & Recherche IPA
# -----------------------------------------------------------------------------
with tab_phon_chain:
    st.subheader("Chaînes phonématiques & Recherche d'occurrences IPA")
    st.caption("Recherchez tout phonème ou chaîne de phonèmes (ex. [ɔʁ], [wa], [st], [pʁ], [k], [s]) et suivez leur distribution vers par vers.")

    all_stats_raw = res["phonetics"]["deviations"].get("phoneme_stats", [])
    valid_ph_stats = [p for p in all_stats_raw if p["phoneme"] != "ə"]
    ph_ref_dict = {p["phoneme"]: p.get("ref_pct", 0.0) / 100.0 for p in valid_ph_stats}

    if "ipa_multiselect" not in st.session_state:
        st.session_state["ipa_multiselect"] = ["ɔʁ"]

    def add_ipa_token(token):
        curr = list(st.session_state.get("ipa_multiselect", []))
        if token not in curr:
            curr.append(token)
            st.session_state["ipa_multiselect"] = curr

    def clear_ipa_tokens():
        st.session_state["ipa_multiselect"] = []

    def add_custom_ipa():
        val = st.session_state.get("custom_ipa_input", "").strip()
        if val:
            for item in val.split(","):
                token = item.strip().strip("[]/ ")
                if token:
                    add_ipa_token(token)
            st.session_state["custom_ipa_input"] = ""

    st.markdown("**Palette & Clavier virtuel IPA (cliquez sur un symbole pour l'ajouter directement) :**")
    kb_cat = st.radio(
        "Catégorie phonétique",
        options=["Voyelles", "Consonnes", "Chaînes poétiques fréquentes", "Saisie libre"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    if kb_cat == "Voyelles":
        st.caption("Voyelles orales :")
        v_orales = ["i", "e", "ɛ", "a", "ɑ", "ɔ", "o", "u", "y", "ø", "œ"]
        cols_vo = st.columns(len(v_orales))
        for col, v in zip(cols_vo, v_orales):
            col.button(f"[{v}]", key=f"kb_vo_{v}", on_click=add_ipa_token, args=(v,), use_container_width=True)

        st.caption("Voyelles nasales & semi-voyelles (glides) :")
        v_nas_semi = ["ɑ̃", "ɛ̃", "ɔ̃", "œ̃", "j", "w", "ɥ"]
        cols_vns = st.columns(len(v_nas_semi) + 2)
        for col, v in zip(cols_vns[:len(v_nas_semi)], v_nas_semi):
            col.button(f"[{v}]", key=f"kb_vns_{v}", on_click=add_ipa_token, args=(v,), use_container_width=True)
        cols_vns[-1].button("Effacer tout", key="kb_clr_voy", on_click=clear_ipa_tokens, use_container_width=True)

    elif kb_cat == "Consonnes":
        st.caption("Occlusives & Fricatives :")
        c_occ_fric = ["p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "ʁ"]
        cols_cf = st.columns(len(c_occ_fric))
        for col, c in zip(cols_cf, c_occ_fric):
            col.button(f"[{c}]", key=f"kb_cf_{c}", on_click=add_ipa_token, args=(c,), use_container_width=True)

        st.caption("Nasales & Liquides :")
        c_nas_liq = ["m", "n", "ɲ", "ŋ", "l"]
        cols_cnl = st.columns(len(c_nas_liq) + 3)
        for col, c in zip(cols_cnl[:len(c_nas_liq)], c_nas_liq):
            col.button(f"[{c}]", key=f"kb_cnl_{c}", on_click=add_ipa_token, args=(c,), use_container_width=True)
        cols_cnl[-1].button("Effacer tout", key="kb_clr_cons", on_click=clear_ipa_tokens, use_container_width=True)

    elif kb_cat == "Chaînes poétiques fréquentes":
        st.caption("Groupes et combinaisons poétiques fréquentes :")
        freq_clusters = ["ɔʁ", "wa", "st", "pʁ", "tʁ", "kʁ", "bl", "fl", "ɑ̃s", "ɛ̃t"]
        cols_fq = st.columns(len(freq_clusters) + 2)
        for col, fq in zip(cols_fq[:len(freq_clusters)], freq_clusters):
            col.button(f"[{fq}]", key=f"kb_fq_{fq}", on_click=add_ipa_token, args=(fq,), use_container_width=True)
        cols_fq[-1].button("Effacer tout", key="kb_clr_freq", on_click=clear_ipa_tokens, use_container_width=True)

    else:
        c_in1, c_in2 = st.columns([4, 1])
        with c_in1:
            st.text_input(
                "Saisie libre (n-grammes ou phonèmes séparés par virgules)",
                key="custom_ipa_input",
                placeholder="ex. ɔʁ, kl, wa, ɑ̃s",
                label_visibility="collapsed",
            )
        with c_in2:
            st.button("+ Ajouter", key="btn_add_custom", on_click=add_custom_ipa, use_container_width=True)

    # Base options for multiselect
    std_ph_options = [p["phoneme"] for p in valid_ph_stats]
    frequent_presets = ["ɔʁ", "wa", "st", "pʁ", "tʁ", "kʁ", "bl", "fl", "ɑ̃s", "ɛ̃t"]
    all_multi_options = list(dict.fromkeys(
        list(st.session_state.get("ipa_multiselect", [])) +
        frequent_presets +
        std_ph_options
    ))

    def format_ph_opt(ph_token):
        p = next((item for item in valid_ph_stats if item["phoneme"] == ph_token), None)
        if p:
            return f"[{ph_token}] {p['name']} ({p.get('count', 0)} occ.)"
        return f"[{ph_token}] (Chaîne ciblée)"

    selected_tokens = st.multiselect(
        "Phonèmes et chaînes sélectionnés pour l'analyse :",
        options=all_multi_options,
        key="ipa_multiselect",
        format_func=format_ph_opt,
        help="Ajoutez ou supprimez des phonèmes/chaînes pour afficher leurs graphiques individuels d'évolution."
    )

    target_queries = []
    seen_labels = set()
    for q in selected_tokens:
        clean_q = q.strip("[]/ ")
        if not clean_q:
            continue
        target_phs = phonetics.extract_phonemes(clean_q.replace("r", "ʁ"))
        label = f"[{''.join(target_phs)}]" if target_phs else f"[{clean_q}]"
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
            "Occurrences réelles par vers (Nombre)",
            "Standardisé à 12 syllabes (Fréquence normalisée)",
            "Ratio Observé / Attendu (O/E - Français standard = 1.0)",
            "Score d'écart statistique (z-score)",
            "Pourcentage du vers (%)",
            "Cumul progressif",
        ]
        track_mode = st.radio(
            "Indicateur de suivi",
            options=track_options,
            index=0,
            horizontal=True,
            key="ph_track_mode_unified",
            help="Comparez les occurrences brutes ou normalisées pour neutraliser la disparité de longueur des vers."
        )

        lines_data = res["phonetics"].get("lines", [])
        track_records = []
        for l_idx, r in enumerate(res["rows"]):
            l_num = r["line"]
            st_num = r["stanza"]
            l_text = r["text"]
            l_ipa = r["ipa"]
            l_sylls = max(int(r.get("syllables") or r.get("anatonic_syllables") or 1), 1)

            l_phs = lines_data[l_idx].get("phonemes", []) if l_idx < len(lines_data) else []
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

        if track_mode.startswith("Occurrences réelles"):
            val_col = "count"
            y_title = "Occurrences réelles (Nombre)"
            val_tooltip_title = "Occurrences"
            val_format = "d"
        elif track_mode.startswith("Standardisé à 12"):
            val_col = "norm_12"
            y_title = "Fréquence standardisée (occ./12 syllabes)"
            val_tooltip_title = "Standard 12v"
            val_format = ".2f"
        elif track_mode.startswith("Ratio Observé"):
            val_col = "oe_ratio"
            y_title = "Ratio Observé / Attendu (O/E)"
            val_tooltip_title = "Ratio O/E"
            val_format = ".2f"
        elif track_mode.startswith("Score d'écart"):
            val_col = "z_line"
            y_title = "Score d'écart statistique (z-score)"
            val_tooltip_title = "Écart z"
            val_format = "+.2f"
        elif track_mode.startswith("Pourcentage"):
            val_col = "pct"
            y_title = "Part phonématique du vers (%)"
            val_tooltip_title = "Part (%)"
            val_format = ".1f"
        else:
            val_col = "cumulative"
            y_title = "Occurrences cumulées"
            val_tooltip_title = "Cumul"
            val_format = "d"

        PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#4b5563"]

        st.subheader(f"Graphiques individuels d'évolution vers par vers ({val_tooltip_title})")
        st.caption(f"Profil individuel pour chaque phonème ou chaîne sélectionné ({y_title}). La ligne pointillée indique la moyenne par vers.")

        n_targets = len(target_queries)
        cols_per_row = 2 if n_targets >= 2 else 1

        for i in range(0, n_targets, cols_per_row):
            chart_cols = st.columns(cols_per_row)
            for c_idx in range(cols_per_row):
                target_idx = i + c_idx
                if target_idx >= n_targets:
                    break
                q_item = target_queries[target_idx]
                t_label = q_item["label"]
                color = PALETTE[target_idx % len(PALETTE)]
                col_ui = chart_cols[c_idx]

                df_sub = track_df[track_df["phoneme"] == t_label]
                tot_c = int(df_sub["count"].sum())
                mean_v = float(df_sub[val_col].mean()) if not df_sub.empty else 0.0
                max_v = float(df_sub[val_col].max()) if not df_sub.empty else 0.0
                best_line = int(df_sub.loc[df_sub[val_col].idxmax()]["line"]) if not df_sub.empty and max_v > 0 else "-"
                mean_str = f"{mean_v:.2f}"
                max_str = f"{int(max_v)}" if val_format == "d" else f"{max_v:{val_format}}"

                with col_ui:
                    st.markdown(
                        f"**{t_label}** ｜ Total : **{tot_c}** occ. ｜ Moyenne : **{mean_str}** ｜ Pic : Vers **{best_line}** ({max_str})"
                    )
                    base = alt.Chart(df_sub).encode(
                        x=alt.X("line:O", title="Numéro du vers"),
                        tooltip=[
                            alt.Tooltip("line:O", title="Vers"),
                            alt.Tooltip("stanza:O", title="Strophe"),
                            alt.Tooltip("syllables:Q", title="Syllabes"),
                            alt.Tooltip("count:Q", title="Occurrences réelles"),
                            alt.Tooltip(f"{val_col}:Q", title=val_tooltip_title, format=val_format),
                            alt.Tooltip("norm_12:Q", title="Standard 12v", format=".2f"),
                            alt.Tooltip("z_line:Q", title="Score z", format="+.2f"),
                            alt.Tooltip("oe_ratio:Q", title="Ratio O/E", format=".2f"),
                            alt.Tooltip("text:N", title="Texte du vers"),
                            alt.Tooltip("ipa:N", title="IPA du vers"),
                        ]
                    )
                    bars = base.mark_bar(opacity=0.35, color=color).encode(
                        y=alt.Y(f"{val_col}:Q", title=val_tooltip_title, axis=alt.Axis(tickMinStep=1 if val_col == "count" else None))
                    )
                    lines = base.mark_line(point=True, strokeWidth=2.2, color=color).encode(
                        y=alt.Y(f"{val_col}:Q")
                    )
                    mean_rule = alt.Chart(pd.DataFrame([{"mean_val": mean_v}])).mark_rule(
                        color=color, strokeDash=[4, 4], strokeWidth=1.2
                    ).encode(y="mean_val:Q")

                    chart_obj = (bars + lines + mean_rule).properties(height=230)

                    if val_col == "oe_ratio":
                        ref_1 = alt.Chart(pd.DataFrame([{"y": 1.0}])).mark_rule(color="#ef4444", strokeDash=[3, 3], strokeWidth=1).encode(y="y:Q")
                        chart_obj = chart_obj + ref_1
                    elif val_col == "z_line":
                        ref_0 = alt.Chart(pd.DataFrame([{"y": 0.0}])).mark_rule(color="#64748b", strokeWidth=1).encode(y="y:Q")
                        chart_obj = chart_obj + ref_0

                    st.altair_chart(chart_obj, width="stretch")

        with st.expander("Vue comparative superposée & Carte thermique (Heatmap)", expanded=False):
            chart_tr1, chart_tr2 = st.columns([3, 2])
            with chart_tr1:
                st.markdown(f"**Comparaison superposée ({y_title})**")
                line_chart = alt.Chart(track_df).mark_line(point=True).encode(
                    x=alt.X("line:O", title="Numéro du vers"),
                    y=alt.Y(f"{val_col}:Q", title=y_title),
                    color=alt.Color("phoneme:N", title="Phonème / Chaîne", scale=alt.Scale(scheme="category10")),
                    tooltip=[
                        alt.Tooltip("line:O", title="Vers"),
                        alt.Tooltip("stanza:O", title="Strophe"),
                        alt.Tooltip("syllables:Q", title="Syllabes"),
                        alt.Tooltip("phoneme:N", title="Phonème"),
                        alt.Tooltip("count:Q", title="Occurrences réelles"),
                        alt.Tooltip("norm_12:Q", title="Standard 12v", format=".2f"),
                        alt.Tooltip("oe_ratio:Q", title="Ratio O/E", format=".2f"),
                        alt.Tooltip("z_line:Q", title="Score z", format="+.2f"),
                        alt.Tooltip("pct:Q", title="Part (%)", format=".1f"),
                        alt.Tooltip("cumulative:Q", title="Cumul"),
                        alt.Tooltip("text:N", title="Texte"),
                        alt.Tooltip("ipa:N", title="IPA"),
                    ]
                ).properties(height=300)
                st.altair_chart(line_chart, width="stretch")

            with chart_tr2:
                st.markdown(f"**Carte thermique (Heatmap) par vers ({val_tooltip_title})**")
                heatmap = alt.Chart(track_df).mark_rect().encode(
                    x=alt.X("line:O", title="Numéro du vers"),
                    y=alt.Y("phoneme:N", title="Phonème / Chaîne"),
                    color=alt.Color(f"{val_col}:Q", title=val_tooltip_title, scale=alt.Scale(scheme="blues")),
                    tooltip=[
                        alt.Tooltip("line:O", title="Vers"),
                        alt.Tooltip("syllables:Q", title="Syllabes"),
                        alt.Tooltip("phoneme:N", title="Phonème"),
                        alt.Tooltip(f"{val_col}:Q", title=val_tooltip_title, format=val_format),
                        alt.Tooltip("count:Q", title="Occurrences"),
                        alt.Tooltip("text:N", title="Texte"),
                    ]
                ).properties(height=300)
                st.altair_chart(heatmap, width="stretch")

            st.markdown("**Tableau détaillé des occurrences par vers**")
            pivot_counts = track_df.pivot(index=["line", "stanza", "syllables", "text"], columns="phoneme", values=val_col).reset_index()
            ph_cols = [q["label"] for q in target_queries if q["label"] in pivot_counts.columns]
            pivot_counts["Total"] = pivot_counts[ph_cols].sum(axis=1).round(2)
            display_cols = ["line", "stanza", "syllables", "text"] + ph_cols + ["Total"]
            st.dataframe(
                pivot_counts[display_cols].rename(columns={
                    "line": "Vers", "stanza": "Strophe", "syllables": "Syllabes", "text": "Texte", "Total": f"Total ({val_tooltip_title})"
                }),
                width="stretch",
                hide_index=True
            )
    else:
        st.info("Veuillez sélectionner au moins un phonème ou une chaîne IPA dans la palette ou le sélecteur ci-dessus.")

    st.subheader("Réseau des chaînes phonématiques & Échos sonores")
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        dom_c_str = " ｜ ".join(f"[{d['consonant']}] ({d['count']} fois)" for d in mesch["dominant_consonant_series"])
        dom_v_str = " ｜ ".join(f"[{d['vowel']}] ({d['count']} fois)" for d in mesch["dominant_vowel_series"])
        st.markdown(f"**Séries consonantiques majeures (Attaques)** :  \n{dom_c_str or 'Aucune'}")
        st.markdown(f"**Séries vocaliques majeures (Timbres)** :  \n{dom_v_str or 'Aucune'}")
        if mesch["superlatifs_prosodiques"]:
            st.markdown("**Mots de haute concentration phonématique**")
            st.dataframe(
                pd.DataFrame(mesch["superlatifs_prosodiques"]).rename(
                    columns={"line": "Vers", "word": "Mot", "ipa": "IPA", "consonant": "Consonne", "count_in_word": "Occurrences"}
                )[["Vers", "Mot", "IPA", "Consonne", "Occurrences"]],
                width="stretch",
                hide_index=True,
            )
    with s_col2:
        st.markdown("**Figures d'échos sonores et allitérations trans-vers**")
        if mesch["echo_figures"]:
            st.dataframe(
                pd.DataFrame(mesch["echo_figures"]).rename(
                    columns={"line": "Vers", "label": "Figure sonore", "example": "Contexte"}
                )[["Vers", "Figure sonore", "Contexte"]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Aucune figure d'écho phonématique isolée.")


# -----------------------------------------------------------------------------
# Onglet 4 : Rimes & Structure strophique
# -----------------------------------------------------------------------------
with tab_rimes:
    st.subheader("Structure strophique et disposition des rimes")
    st_df = pd.DataFrame(pr["stanzas_detail"])
    if not st_df.empty:
        st.dataframe(
            st_df.rename(columns={
                "stanza": "Strophe", "line_count": "Nombre de vers", "scheme": "Schéma des rimes",
                "disposition": "Disposition", "genres": "Genres (M/F)", "alternance_respected": "Alternance respectée",
            }),
            width="stretch",
            hide_index=True,
        )

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown("**Richesse des rimes**")
        r_counts = gouv["rhyme"].get("richness_counts", {})
        if r_counts:
            r_df = pd.DataFrame([{"Richesse": k, "Nombre": v} for k, v in r_counts.items()])
            r_chart = alt.Chart(r_df).mark_bar(color="#7c3aed").encode(
                x=alt.X("Nombre:Q", title="Nombre de rimes"),
                y=alt.Y("Richesse:N", sort="-x", title=None),
                tooltip=["Richesse", "Nombre"],
            ).properties(height=180)
            st.altair_chart(r_chart, width="stretch")

    with r_col2:
        st.markdown("**Alternance des genres masculin / féminin**")
        alt_rate = pr_sum.get("alternance_rate", 0.0)
        st.metric("Taux de respect de l'alternance M/F", f"{alt_rate:.1f}%", help="Respect de la règle classique d'alternance des rimes masculines et féminines")

    # Analyse grammaticale des rimes (rimes grammaticales vs hétérogènes)
    st.subheader("Homogénéité morphosyntaxique des rimes")
    st.caption("Une rime grammaticale associe deux mots de même catégorie morphologique (ex. Verbe-Verbe, Nom-Nom). La poétique classique valorise la variété par des rimes hétérogènes (ex. Nom-Adjectif, Verbe-Nom).")
    rhyme_pairs = []
    for st_idx, st_data in enumerate(pr.get("stanzas_detail", []), 1):
        st_lines = [r for r in rows_data if r.get("stanza") == st_idx]
        by_letter = collections.defaultdict(list)
        for l_row in st_lines:
            if l_row.get("rhyme_letter"):
                by_letter[l_row["rhyme_letter"]].append(l_row)
        for let, g_lines in by_letter.items():
            if len(g_lines) >= 2:
                for i in range(len(g_lines)):
                    for j in range(i + 1, len(g_lines)):
                        p1 = g_lines[i].get("rhyme_pos", "")
                        p2 = g_lines[j].get("rhyme_pos", "")
                        is_gram = (p1 == p2 and bool(p1))
                        w1 = g_lines[i]["text"].strip().rstrip(".,;:!?»'\"").split()[-1] if g_lines[i]["text"].strip() else ""
                        w2 = g_lines[j]["text"].strip().rstrip(".,;:!?»'\"").split()[-1] if g_lines[j]["text"].strip() else ""
                        rhyme_pairs.append({
                            "strophe": st_idx,
                            "rime": let,
                            "vers_pair": f"V{g_lines[i]['line']} - V{g_lines[j]['line']}",
                            "mots": f"{w1} ｜ {w2}",
                            "categories": f"{POS_NAMES_FR.get(p1, p1)} ｜ {POS_NAMES_FR.get(p2, p2)}",
                            "type": "Rime grammaticale (homogène)" if is_gram else "Rime hétérogène (mixte)",
                            "is_gram": is_gram,
                        })

    rg_col1, rg_col2, rg_col3 = st.columns(3)
    if rhyme_pairs:
        total_rp = len(rhyme_pairs)
        gram_rp = sum(1 for rp in rhyme_pairs if rp["is_gram"])
        hetero_rp = total_rp - gram_rp
        gram_pct = round(100.0 * gram_rp / total_rp, 1)
        hetero_pct = round(100.0 * hetero_rp / total_rp, 1)

        rg_col1.metric("Paires de rimes analysées", f"{total_rp}", border=True)
        rg_col2.metric("Rimes hétérogènes (richesse stylistique)", f"{hetero_pct}%", f"{hetero_rp} paires", border=True)
        rg_col3.metric("Rimes grammaticales (même catégorie)", f"{gram_pct}%", f"{gram_rp} paires", border=True)

        with st.expander("Détail des paires de rimes et catégories grammaticales", expanded=False):
            st.dataframe(
                pd.DataFrame(rhyme_pairs)[["strophe", "rime", "vers_pair", "mots", "categories", "type"]].rename(columns={
                    "strophe": "Strophe", "rime": "Rime", "vers_pair": "Vers associés",
                    "mots": "Mots rimants", "categories": "Catégories POS", "type": "Type morphologique"
                }),
                width="stretch",
                hide_index=True
            )
    else:
        st.info("Aucune paire de rimes consécutives ou strophiques détectée.")

    st.subheader("Traitement du e caduc, diérèses et hiatus")
    e_col1, e_col2, e_col3 = st.columns(3)
    with e_col1:
        st.markdown("**États du e caduc**")
        ec_sum = gouv["e_caduc_summary"]
        if ec_sum:
            st.dataframe(pd.DataFrame([{"État": k, "Nombre": v} for k, v in ec_sum.items()]), width="stretch", hide_index=True)
    with e_col2:
        st.markdown("**Diérèses poétiques**")
        if gouv["synerese_dierese_events"]:
            st.dataframe(pd.DataFrame(gouv["synerese_dierese_events"])[["line", "word", "label"]].rename(columns={"line": "Vers", "word": "Mot", "label": "Type"}), width="stretch", hide_index=True)
        else:
            st.info("Aucune diérèse nécessaire.")
    with e_col3:
        st.markdown("**Hiatus métriques**")
        if gouv["hiatus_events"]:
            st.dataframe(pd.DataFrame(gouv["hiatus_events"])[["line", "context", "label"]].rename(columns={"line": "Vers", "context": "Conflit", "label": "Type"}), width="stretch", hide_index=True)
        else:
            st.info("Aucun hiatus interdit détecté.")


# -----------------------------------------------------------------------------
# Onglet 5 : Morphosyntaxe (POS) & Stylométrie
# -----------------------------------------------------------------------------
with tab_phon_dist:
    st.subheader("Profil morphosyntaxique (POS) & Densité lexicale")
    st.caption("Répartition des parties du discours (mots pleins vs mots outils) et indice de nominalité stylistique.")

    pos_an = res.get("pos_analysis", {})
    pos_counts = pos_an.get("counts", {})
    W = pos_an.get("total_words", 0)

    pm1, pm2, pm3, pm4, pm5 = st.columns(5)
    pm1.metric("Mots au total (W)", f"{W:,}", border=True)
    pm2.metric("Mots pleins (Lexique)", f"{pos_an.get('content_words', 0):,}", help="Noms, verbes pleins, adjectifs, adverbes, noms propres", border=True)
    pm3.metric("Mots outils (Grammaire)", f"{pos_an.get('function_words', 0):,}", help="Déterminants, pronoms, prépositions, conjonctions, auxiliaires", border=True)
    pm4.metric("Densité lexicale", f"{pos_an.get('lexical_density', 0.0):.1f}%", help="Part des mots porteurs de sens sémantique plein", border=True)
    pm5.metric("Indice de nominalité", f"{pos_an.get('nominal_ratio', 0.0):.2f}", help="Ratio Noms / Verbes. ≥ 1.5 : style nominal, ≤ 0.8 : style verbal", border=True)

    st.info(f"**Diagnostic stylistique du texte** : {pos_an.get('style_desc', '-')}")

    pos_c1, pos_c2 = st.columns([3, 2])
    with pos_c1:
        st.markdown("**Distribution globale des parties du discours (POS)**")
        pos_list = []
        for tag, count in pos_counts.items():
            if tag in ("PUNCT", "SPACE", "SYM", "X"):
                continue
            pct = round(100.0 * count / max(W, 1), 2)
            lbl = POS_NAMES_FR.get(tag, tag)
            pos_list.append({
                "tag": tag,
                "label": f"{lbl} ({tag})",
                "count": count,
                "pct": pct,
            })
        if pos_list:
            pos_df = pd.DataFrame(pos_list).sort_values("count", ascending=False)
            chart_pos_bar = alt.Chart(pos_df).mark_bar().encode(
                x=alt.X("count:Q", title="Nombre d'occurrences"),
                y=alt.Y("label:N", sort="-x", title=None),
                color=alt.Color("tag:N", scale=alt.Scale(
                    domain=list(POS_COLORS.keys()),
                    range=list(POS_COLORS.values())
                ), legend=None),
                tooltip=[
                    alt.Tooltip("label:N", title="Catégorie"),
                    alt.Tooltip("count:Q", title="Occurrences"),
                    alt.Tooltip("pct:Q", title="Part (%)", format=".2f"),
                ]
            ).properties(height=260)
            st.altair_chart(chart_pos_bar, width="stretch")

    with pos_c2:
        st.markdown("**Composition par vers (Mots pleins vs Outils)**")
        lines_pos_records = []
        for r in rows_data:
            lines_pos_records.append({
                "line": r["line"],
                "content": r.get("content_words", 0),
                "function": r.get("function_words", 0),
            })
        if lines_pos_records:
            lp_df = pd.DataFrame(lines_pos_records)
            lp_melt = lp_df.melt(id_vars=["line"], value_vars=["content", "function"], var_name="Type", value_name="Nombre")
            lp_melt["Type"] = lp_melt["Type"].map({"content": "Mots pleins", "function": "Mots outils"})
            chart_lp = alt.Chart(lp_melt).mark_bar().encode(
                x=alt.X("line:O", title="Numéro du vers"),
                y=alt.Y("Nombre:Q", title="Nombre de mots"),
                color=alt.Color("Type:N", scale=alt.Scale(domain=["Mots pleins", "Mots outils"], range=["#2563eb", "#94a3b8"]), title=None),
                tooltip=["line:O", "Type:N", "Nombre:Q"]
            ).properties(height=260)
            st.altair_chart(chart_lp, width="stretch")

    st.subheader("Visualiseur textuel morphosyntaxique")
    st.caption("Chaque mot du poème est annoté avec sa catégorie grammaticale et coloré selon son rôle syntaxique. Utilisez le sélecteur ci-dessous pour filtrer une catégorie spécifique.")

    active_tags = ["TOUS"] + [t for t in POS_NAMES_FR.keys() if t in pos_counts]
    filter_choice = st.selectbox(
        "Filtrer par catégorie morphosyntaxique",
        options=active_tags,
        format_func=lambda x: "Afficher toutes les catégories" if x == "TOUS" else f"{POS_NAMES_FR.get(x, x)} ({x}) — {pos_counts.get(x, 0)} occurrences",
        help="Met en valeur les mots appartenant à la catégorie choisie et estompe les autres."
    )

    if filter_choice != "TOUS":
        f_lbl = POS_NAMES_FR.get(filter_choice, filter_choice)
        f_color = POS_COLORS.get(filter_choice, "#2563eb")
        v_data = []
        for r in rows_data:
            c = r.get("pos_counts", {}).get(filter_choice, 0)
            v_data.append({"line": r["line"], "count": c, "text": r["text"]})
        v_df = pd.DataFrame(v_data)
        chart_filtered = alt.Chart(v_df).mark_bar(color=f_color).encode(
            x=alt.X("line:O", title="Numéro du vers"),
            y=alt.Y("count:Q", title=f"Nombre de : {f_lbl}", axis=alt.Axis(tickMinStep=1)),
            tooltip=[
                alt.Tooltip("line:O", title="Vers"),
                alt.Tooltip("count:Q", title=f"Occurrences de {f_lbl}"),
                alt.Tooltip("text:N", title="Texte du vers"),
            ]
        ).properties(height=180)
        st.markdown(f"**Occurrences vers par vers : {f_lbl} ({filter_choice})**")
        st.altair_chart(chart_filtered, width="stretch")

    st.html(render_pos_text_html(rows_data, filter_choice))

    with st.expander("Tableau détaillé de la composition grammaticale par vers", expanded=False):
        table_lines_pos = []
        for r in rows_data:
            cnts = r.get("pos_counts", {})
            entry = {
                "Vers": r["line"],
                "Texte": r["text"],
                "Noms": cnts.get("NOUN", 0) + cnts.get("PROPN", 0),
                "Verbes": cnts.get("VERB", 0) + cnts.get("AUX", 0),
                "Adjectifs": cnts.get("ADJ", 0),
                "Adverbes": cnts.get("ADV", 0),
                "Déterminants": cnts.get("DET", 0),
                "Pronoms": cnts.get("PRON", 0),
                "Prépositions": cnts.get("ADP", 0),
                "Conjonctions": cnts.get("CCONJ", 0) + cnts.get("SCONJ", 0),
                "Mots pleins": r.get("content_words", 0),
                "Mots outils": r.get("function_words", 0),
                "Rime (POS)": f"{POS_NAMES_FR.get(r.get('rhyme_pos', ''), r.get('rhyme_pos', ''))} ({r.get('rhyme_pos', '-')})",
            }
            table_lines_pos.append(entry)
        st.dataframe(pd.DataFrame(table_lines_pos), width="stretch", hide_index=True)

    st.subheader("Profil phonologique et stylométrique du poème")

    phonetics_data = res["phonetics"]
    deviations = phonetics_data["deviations"]
    all_liaisons = phonetics_data["all_liaisons"]

    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    p_col1.metric("Total phonèmes", f"{deviations['total_phonemes']:,}", border=True)
    p_col2.metric("Ratio V/C", f"{deviations['vc_ratio']:.2f}", border=True, help="Français standard : ~0.88")
    v_glide_pct = deviations["category_deviations"].get("vowels_pct", 0.0) + deviations["category_deviations"].get("glides_pct", 0.0)
    p_col3.metric("Voyelles (+ glides)", f"{v_glide_pct:.1f}%", f"{round(v_glide_pct - 46.7, 1):+.1f}%p", border=True)
    c_pct = deviations["category_deviations"].get("consonants_pct", 0.0)
    p_col4.metric("Consonnes", f"{c_pct:.1f}%", f"{round(c_pct - 53.3, 1):+.1f}%p", border=True)
    p_col5.metric("Liaisons détectées", f"{len(all_liaisons)}", border=True)

    st.subheader("Écart statistique des 36 phonèmes par rapport au français standard (z-score)")
    df_phonemes = pd.DataFrame(deviations["phoneme_stats"])
    if not df_phonemes.empty:
        chart_phonemes = alt.Chart(df_phonemes).mark_bar().encode(
            x=alt.X("z_score:Q", title="Score d'écart normalisé (z-score)", axis=alt.Axis(grid=True)),
            y=alt.Y("phoneme:N", sort=alt.EncodingSortField(field="z_score", order="descending"), title="Phonème (IPA)"),
            color=alt.condition(alt.datum.z_score > 0, alt.value("#2563eb"), alt.value("#dc2626")),
            tooltip=[
                alt.Tooltip("phoneme:N", title="Phonème"),
                alt.Tooltip("name:N", title="Nom"),
                alt.Tooltip("count:Q", title="Occurrences"),
                alt.Tooltip("pct:Q", title="Taux texte (%)", format=".2f"),
                alt.Tooltip("ref_pct:Q", title="Taux standard (%)", format=".2f"),
                alt.Tooltip("diff:Q", title="Différence (%p)", format="+.2f"),
                alt.Tooltip("z_score:Q", title="Score z", format="+.2f"),
            ],
        ).properties(height=600)

        r0 = alt.Chart(pd.DataFrame([{"z": 0}])).mark_rule(color="#4b5563", strokeWidth=1.5).encode(x="z:Q")
        rp1 = alt.Chart(pd.DataFrame([{"z": 1.0}])).mark_rule(color="#9ca3af", strokeDash=[4, 4]).encode(x="z:Q")
        rm1 = alt.Chart(pd.DataFrame([{"z": -1.0}])).mark_rule(color="#9ca3af", strokeDash=[4, 4]).encode(x="z:Q")
        rp2 = alt.Chart(pd.DataFrame([{"z": 2.0}])).mark_rule(color="#f59e0b", strokeDash=[3, 3]).encode(x="z:Q")
        rm2 = alt.Chart(pd.DataFrame([{"z": -2.0}])).mark_rule(color="#f59e0b", strokeDash=[3, 3]).encode(x="z:Q")
        st.altair_chart(chart_phonemes + r0 + rp1 + rm1 + rp2 + rm2, width="stretch")

    st.subheader("Comparaison stylométrique avec les corpus de référence (GSD & ParisStories)")
    gsd_vals = ref["values"]["GSD"]
    ps_vals = ref["values"]["ParisStories"]
    metric_keys = [k for k in LABELS_FR if k in gsd_vals and k in ps_vals]

    table_records = []
    for k in metric_keys:
        lbl = LABELS_FR[k]
        t_val = res["metrics"].get(k)
        g_val = gsd_vals.get(k, 0.0)
        p_val = ps_vals.get(k, 0.0)

        table_records.append({
            "Indicateur": lbl,
            "Ce texte poétique": "-" if t_val is None else f"{t_val:.3f}" if isinstance(t_val, float) and k in ("word_syl_mean", "zipf_content_mean", "mattr100") else f"{t_val}",
            "Français écrit (GSD)": f"{g_val:.3f}" if isinstance(g_val, float) and k in ("word_syl_mean", "zipf_content_mean", "mattr100") else f"{g_val}",
            "Français oral (ParisStories)": f"{p_val:.3f}" if isinstance(p_val, float) and k in ("word_syl_mean", "zipf_content_mean", "mattr100") else f"{p_val}",
        })
    st.dataframe(pd.DataFrame(table_records), width="stretch", hide_index=True)

    st.subheader("Export des résultats")
    exp_c1, exp_c2, exp_c3 = st.columns(3)
    with exp_c1:
        st.download_button(
            "Données par vers (CSV)",
            rows_df.to_csv(index=False).encode("utf-8-sig"),
            "vers_metrique_phonetique.csv",
            "text/csv",
            width="stretch",
        )
    with exp_c2:
        st.download_button(
            "Synthèse poétique (CSV)",
            pd.DataFrame([{"Indicateur": k, "Valeur": v} for k, v in res["lines_summary"].items()]).to_csv(index=False).encode("utf-8-sig"),
            "synthese.csv",
            "text/csv",
            width="stretch",
        )
    with exp_c3:
        export_json = json.dumps(res, ensure_ascii=False, indent=2)
        st.download_button(
            "Analyse intégrale (JSON)",
            export_json.encode("utf-8"),
            "analyse_complete.json",
            "application/json",
            width="stretch",
        )
