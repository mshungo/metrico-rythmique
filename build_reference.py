"""UD の .conllu ファイルから参照値 (reference.json) を再生成するスクリプト。

使用例:
    python build_reference.py --gsd fr_gsd-ud-dev.conllu fr_gsd-ud-test.conllu --paris-stories fr_parisstories-ud-test.conllu --output reference.json
"""
import argparse, json, sys
from pathlib import Path
from analysis import get_nlp, compute_corpus_metrics

def extract_sentences_from_conllu(file_paths):
    """conllu ファイルから '# text =' 行の本文を抽出して文リストを返す。"""
    sentences = []
    for p in file_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("# text ="):
                    s = line[len("# text ="):].strip()
                    if s:
                        sentences.append(s)
    return sentences

def build_reference(gsd_paths=None, paris_stories_paths=None, output_path="reference.json"):
    nlp = get_nlp()
    out = Path(output_path)

    # 既存の reference.json があれば読み込んで引き継ぐ
    if out.exists():
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            data = {"_note": "spaCy fr_core_news_md 3.8 で本文のみ再解析して算出。単位は文（発話単位）。", "corpora": {}, "values": {}}
    else:
        data = {"_note": "spaCy fr_core_news_md 3.8 で本文のみ再解析して算出。単位は文（発話単位）。", "corpora": {}, "values": {}}

    if gsd_paths:
        print(f"GSD ファイルを読み込み中: {gsd_paths}")
        sents = extract_sentences_from_conllu(gsd_paths)
        print(f"GSD 文数: {len(sents)}。spaCy 解析中...")
        docs = list(nlp.pipe(sents, batch_size=200))
        res = compute_corpus_metrics(docs)
        data["corpora"]["GSD"] = f"UD French-GSD（書き言葉・{len(sents):,}文）"
        data["values"]["GSD"] = {**res["metrics"], "tense_share": res["tense_share"]}
        print("GSD 完了")

    if paris_stories_paths:
        print(f"ParisStories ファイルを読み込み中: {paris_stories_paths}")
        sents = extract_sentences_from_conllu(paris_stories_paths)
        print(f"ParisStories 単位数: {len(sents)}。spaCy 解析中...")
        docs = list(nlp.pipe(sents, batch_size=200))
        res = compute_corpus_metrics(docs)
        data["corpora"]["ParisStories"] = f"UD French-ParisStories（話し言葉の語り・{len(sents):,}単位）"
        data["values"]["ParisStories"] = {**res["metrics"], "tense_share": res["tense_share"]}
        print("ParisStories 完了")

    out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"参照値を書き出しました: {out}")

def main():
    parser = argparse.ArgumentParser(description="UD の .conllu ファイルから reference.json を生成する")
    parser.add_argument("--gsd", nargs="+", help="UD French-GSD の .conllu ファイル群（例: dev+test）")
    parser.add_argument("--paris-stories", nargs="+", dest="paris_stories", help="UD French-ParisStories の .conllu ファイル群（例: test）")
    parser.add_argument("--output", default="reference.json", help="出力ファイルパス（初期値: reference.json）")

    args = parser.parse_args()
    if not args.gsd and not args.paris_stories:
        parser.print_help()
        sys.exit(1)

    build_reference(gsd_paths=args.gsd, paris_stories_paths=args.paris_stories, output_path=args.output)

if __name__ == "__main__":
    main()
