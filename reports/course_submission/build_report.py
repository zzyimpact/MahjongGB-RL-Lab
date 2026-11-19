"""Build the course report from Markdown and existing records; never run training.

Run --check-sources and --assets with the project Python (torch/matplotlib).
Run --pdf with a Python providing reportlab and pypdf. Both can be combined
when all dependencies are installed. Chinese font paths can be overridden.
"""
from pathlib import Path
import argparse
import hashlib
import html
import json
import re
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MD = HERE / "FINAL_COURSE_REPORT.md"
RAW = ROOT / "reports/round4/raw"
TEMP = Path(tempfile.gettempdir()) / "mahjong_course_report_pdf"
FONT = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
BOLD = Path("/System/Library/Fonts/STHeiti Medium.ttc")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources():
    import gzip
    import torch
    import zipfile

    manifest = read_json(ROOT / "reports/round4/ARTIFACT_MANIFEST.json")
    checkpoint = ROOT / "artifacts/round4/PPO_legacy_best/checkpoint.pt"
    release = ROOT / "artifacts/round4/releases/PPO/mahjong_model.pt"
    deploy = ROOT / "artifacts/botzone_ready_20261119/V3_PPO/mahjong_V3_PPO.pt"
    code = deploy.with_name("botzone_V3_PPO.zip")
    ckpt = torch.load(checkpoint, map_location="cpu")
    weights = [torch.load(p, map_location="cpu") for p in (release, deploy)]
    policy = {k.removeprefix("module."): v for k, v in ckpt["policy"].items()}
    assert all(set(policy) == set(w) for w in weights)
    assert all(torch.equal(policy[k], w[k]) for w in weights for k in policy)
    assert sha(checkpoint) == manifest["models"]["PPO_legacy_best"]["sha256"]
    with gzip.open(RAW / "PPO_V3_legacy/metrics.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    selected = next(row for row in rows if row["iteration"] == 48)
    assert ckpt["iteration"] == 48 and ckpt["counters"] == selected["counters"]
    assert selected["counters"]["completed_episodes"] == 768
    assert selected["counters"]["gradient_updates"] == 422
    assert len(rows) == 64 and rows[-1]["counters"]["completed_episodes"] == 1024
    evaluations = read_json(RAW / "PPO_V3_legacy/evaluations.json")
    best = max((e for e in evaluations if e["iteration"] > 8), key=lambda e: e["score"]["mean"])
    assert best["iteration"] == 48 and best["sha256"] == sha(checkpoint)
    with zipfile.ZipFile(code) as z:
        expanded = sum(e.file_size for e in z.infolist())
    assert deploy.stat().st_size <= 268435456
    assert code.stat().st_size < 4000000 and expanded < 4000000
    assert sum(p.numel() for p in policy.values()) == 1199917

    test = read_json(RAW / "test_metrics.json")["models"]
    arena = read_json(RAW / "arena_final/results.json")
    scores = arena["aggregate"]["scores"]
    selected_models = ["D_original", "D_legacy_e06", "V3_legacy_e01", "PPO_legacy_best"]
    table = {}
    for name in selected_models:
        entry = manifest["models"][name]
        path = ROOT / entry["published_path"]
        assert sha(path) == entry["sha256"] == test[name]["identity"]["sha256"]
        state = torch.load(path, map_location="cpu")
        table[name] = {
            "parameters": sum(t.numel() for t in state["policy"].values()),
            "bc_epoch": state.get("bc", {}).get("epoch"),
            "play": test[name]["metrics"]["metrics"]["play"],
            "reward": scores[name],
            "checkpoint": entry["published_path"], "sha256": sha(path),
        }
    ds = read_json(RAW / "dataset_manifest.json")
    split_counts = {k: {key: v[key] for key in ("samples", "games", "walls", "stored_nonforced")}
                    for k, v in ds["summary"].items()}
    assert [split_counts[k]["walls"] for k in ("train", "val", "test")] == [410, 51, 51]
    assert sum(v["samples"] for v in split_counts.values()) == 2284530
    ablations = read_json(RAW / "development_three_seeds.json")["aggregate"]["paired_differences"]
    ablations = {k: v for k, v in ablations.items() if "V3_e03" in k
                 and any(t in k for t in ("no_quality", "no_context", "relu"))}
    sources = [
        "reports/round4/ARTIFACT_MANIFEST.json", "mahjong/model_v3.py",
        "mahjong/features.py", "mahjong/features_v3.py", "mahjong/train_v3.py",
        "mahjong/ppo_v3.py", "mahjong/ppo_anchor.py", "rl_framework/rlframework/algorithm/ppo.py",
        "reports/round4/raw/dataset_manifest.json", "reports/round4/raw/V3_legacy/config.json",
        "reports/round4/raw/V3_legacy/history.json", "reports/round4/raw/V3_legacy/summary.json",
        "reports/round4/raw/D_legacy/config.json", "reports/round4/raw/test_metrics.json",
        "reports/round4/raw/arena_final/results.json", "reports/round4/raw/confirmatory_statistics.json",
        "reports/round4/raw/development_three_seeds.json", "reports/round4/raw/seed_stability_legacy.json",
        "reports/round4/raw/PPO_V3_legacy/config.json", "reports/round4/raw/PPO_V3_legacy/metrics.jsonl.gz",
        "reports/round4/raw/PPO_V3_legacy/summary.json", "reports/round4/raw/PPO_V3_legacy/evaluations.json",
        "reports/round4/raw/model_benchmark_final.json", "reports/round4/raw/environment.json",
        "artifacts/botzone_ready_20261119/V3_PPO/submission_check.json",
        "artifacts/botzone_ready_20261119/V3_PPO/runtime_check.json",
    ]
    record = {
        "report_source_commit": "034b93f95e0ff0d00ba820b1c48808aacf325e11",
        "round4_development_commit": "d83016398c663cee929b05e0d66a24dc52401af7",
        "scope": "Read existing artifacts only. No new training, match, or model change.",
        "selected_model": "PPO_legacy_best", "selected_iteration": 48,
        "selected_counters": selected["counters"],
        "whole_run": read_json(RAW / "PPO_V3_legacy/summary.json"),
        "warmup_games": 128, "bc_initialization": "V3_legacy_e01",
        "tensor_identity_verified": True,
        "weights": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p), "bytes": p.stat().st_size}
                    for p in (checkpoint, release, deploy)],
        "deployment": {"zip_bytes": code.stat().st_size, "expanded_bytes": expanded,
                       "code_zip": str(code.relative_to(ROOT)), "sha256": sha(code),
                       "online_status": "Successfully submitted and run, as explicitly confirmed by the user.",
                       "online_scores_or_rank": "Not provided; no claim is made."},
        "dataset": split_counts, "core_results": table,
        "paired_statistics": read_json(RAW / "confirmatory_statistics.json"),
        "development_ablations": ablations,
        "learning_curve": [{"epoch": r["epoch"], "train_nonforced_ce": r["train_ce"],
                            "validation_nonforced_ce": r["val"]["nonforced_loss"]}
                           for r in read_json(RAW / "V3_legacy/history.json")],
        "source_files": {p: sha(ROOT / p) for p in sources},
    }
    (HERE / "SOURCE_CHECK.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print("Verified checkpoint iteration 48 / 768 games / 422 updates; deployment tensors identical.")


def make_assets(font):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from PIL import Image as PILImage

    output = HERE / "figures"
    output.mkdir(exist_ok=True)
    TEMP.mkdir(exist_ok=True)
    fp = FontProperties(fname=str(font))
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False,
                         "mathtext.fontset": "stix", "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(7.2, 2.65))
    ax.set(xlim=(0, 10), ylim=(0, 4.1))
    ax.axis("off")

    def box(x, y, w, h, label, color="#eef2f6", size=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                   edgecolor="#748397", linewidth=.8, facecolor=color))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontproperties=fp, fontsize=size)

    def arrow(a, b, color="#56677c"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color=color, linewidth=1))

    box(.10, 2.95, 1.7, .8, "20 个牌面通道\n4×9 网格")
    box(.10, 1.80, 1.7, .75, "14 个全局标量")
    box(.10, .55, 1.7, .9, "最近 16 次行动\n4 家有序副露")
    box(2.25, 2.95, 2.15, .8, "64 通道 × 4 残差块\nGroupNorm / GELU")
    box(2.25, .55, 2.15, .9, "Embedding + MLP")
    box(4.95, 1.65, 1.3, 1.05, "状态融合\n256 维", "#e6eef3")
    box(6.8, 2.95, 1.1, .65, "Actor")
    box(6.8, 1.95, 1.1, .65, "Critic")
    box(8.4, 2.95, 1.4, .65, "235 分值\n合法 Mask")
    box(8.4, 1.95, 1.4, .65, "状态价值")
    box(5.05, .1, 1.7, .8, "每种候选弃牌\n58 维质量特征")
    box(7.4, .1, 2.3, .8, "共享弃牌评分\n34 类增量分值", "#e9f0eb")
    for a,b in [((1.85,3.35),(2.2,3.35)), ((1.85,1.0),(2.2,1.0)),
                ((1.85,2.18),(4.9,2.18)), ((4.45,3.35),(4.95,2.50)),
                ((4.45,1.0),(4.95,1.85)), ((6.3,2.45),(6.8,3.27)),
                ((6.3,2.18),(6.8,2.27)), ((7.95,3.27),(8.35,3.27)),
                ((7.95,2.27),(8.35,2.27)), ((6.8,.5),(7.35,.5)),
                ((6.3,1.68),(7.4,.9))]: arrow(a,b)
    ax.plot([4.4,4.67,4.67,7.15,7.15],[3.05,2.85,.08,.08,.25],color="#56677c",linewidth=.8)
    arrow((7.15,.25),(7.4,.25))
    arrow((9.72,.5),(9.97,.5));ax.plot([9.97,9.97],[.5,3.22],color="#56677c",linewidth=.8)
    arrow((9.97,3.22),(9.82,3.22))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output / "v3_architecture.png", dpi=300)
    plt.close(fig)

    history = read_json(RAW / "V3_legacy/history.json")
    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    x = [r["epoch"] for r in history]
    ax.plot(x, [r["train_ce"] for r in history], "o-", color="#436d91", label="训练 CE", markersize=3.5, linewidth=1.6)
    ax.plot(x, [r["val"]["nonforced_loss"] for r in history], "s-", color="#a96145", label="验证 CE", markersize=3.3, linewidth=1.6)
    ax.axvline(1, color="#9aa4aa", linestyle=":", linewidth=1)
    ax.set(xlim=(.65,12.35), ylim=(0,1.3), xticks=[1,3,6,9,12])
    ax.set_xlabel("训练轮数（epoch）", fontproperties=fp)
    ax.set_ylabel("非强制决策交叉熵", fontproperties=fp)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(prop=fp, frameon=False, loc="center right")
    ax.annotate("验证最佳：epoch 1", xy=(1,.823244), xytext=(3.25,.63),
                fontproperties=fp, fontsize=9, arrowprops=dict(arrowstyle="->", color="#656565"))
    fig.tight_layout(pad=.8)
    fig.savefig(output / "v3_learning_curve.png", dpi=300)
    plt.close(fig)

    text = MD.read_text()
    expressions = re.findall(r"\$\$(.*?)\$\$", text, re.S)
    inline = re.findall(r"(?<!\$)\$([^$\n]+)\$(?!\$)", re.sub(r"\$\$.*?\$\$", "", text, flags=re.S))
    math = {}
    for expr in expressions + inline:
        clean = expr.strip()
        label = "$" + clean.replace("\\mathbb{E}", "\\mathrm{E}") + "$"
        fig = plt.figure(figsize=(8, .6))
        artist = fig.text(0, .5, label, fontsize=13, va="center")
        fig.canvas.draw()
        bbox = artist.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).expanded(1.02,1.12)
        path = TEMP / (hashlib.sha256(clean.encode()).hexdigest()[:16]+".png")
        fig.savefig(path, dpi=400, bbox_inches=bbox, transparent=True, pad_inches=.015)
        plt.close(fig)
        with PILImage.open(path) as im: w,h=im.size
        math[clean] = {"path":str(path),"width":w*72/400,"height":h*72/400}
    (TEMP / "math.json").write_text(json.dumps(math))
    print("Generated two figures from model structure / saved training history, and equation assets.")


def make_pdf(font, bold, font_index, bold_index):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether, PageBreak
    from reportlab.lib import textsplit
    from reportlab.platypus import paragraph as paragraph_module
    from pypdf import PdfReader
    from PIL import Image as PILImage

    pdfmetrics.registerFont(TTFont("CN",str(font),subfontIndex=font_index))
    pdfmetrics.registerFont(TTFont("CN-Bold",str(bold),subfontIndex=bold_index))
    pdfmetrics.registerFontFamily("CN", normal="CN", bold="CN-Bold", italic="CN", boldItalic="CN-Bold")
    pdfmetrics.registerFontFamily("CN-Bold", normal="CN-Bold", bold="CN-Bold")
    # ReportLab's default CJK list omits several Chinese full-width marks.
    textsplit.ALL_CANNOT_START += "，。；：！？）》】」』”’"
    paragraph_module.ALL_CANNOT_START = textsplit.ALL_CANNOT_START
    math = read_json(TEMP / "math.json")
    width = A4[0]-108
    common=dict(fontName="CN", fontSize=10.8, leading=18.2, wordWrap="CJK", textColor=colors.HexColor("#181818"))
    styles = {
        "body": ParagraphStyle("body", **common, firstLineIndent=21.6, alignment=TA_JUSTIFY, spaceAfter=5.5),
        "title": ParagraphStyle("title", fontName="CN-Bold", fontSize=18, leading=26, alignment=TA_CENTER, spaceAfter=11, wordWrap="CJK"),
        "meta": ParagraphStyle("meta", **{**common,"fontSize":10,"leading":17}, alignment=TA_CENTER, spaceAfter=13),
        "h2": ParagraphStyle("h2",fontName="CN-Bold",fontSize=12.2,leading=19,spaceBefore=11,spaceAfter=5,keepWithNext=True),
        "h3": ParagraphStyle("h3",fontName="CN-Bold",fontSize=11,leading=18,spaceBefore=7,spaceAfter=4,keepWithNext=True),
        "caption":ParagraphStyle("caption",fontName="CN",fontSize=9.1,leading=14,wordWrap="CJK",alignment=TA_CENTER,spaceAfter=8),
        "note":ParagraphStyle("note",fontName="CN",fontSize=9,leading=14,wordWrap="CJK",spaceAfter=4),
        "ref":ParagraphStyle("ref",fontName="CN",fontSize=8.6,leading=12.5,wordWrap="CJK",spaceAfter=4,leftIndent=17,firstLineIndent=-17),
        "cell":ParagraphStyle("cell",fontName="CN",fontSize=8.8,leading=13,wordWrap="CJK",alignment=TA_CENTER),
    }

    def rich(s):
        # Escape prose first; inject only renderer-generated math markup.
        chunks=re.split(r"(\$[^$]+\$)",s)
        out=[]
        for ch in chunks:
            if ch.startswith("$") and ch.endswith("$"):
                m=math[ch[1:-1].strip()];scale=.80
                out.append('<img src="%s" width="%.2f" height="%.2f" valign="middle"/>'%(m["path"],m["width"]*scale,m["height"]*scale))
            else:
                ch=html.escape(ch)
                ch=re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",ch)
                ch=re.sub(r"`(.+?)`",r"\1",ch)
                ch=re.sub(r"^\*(.+)\*$",r"\1",ch)
                out.append(ch)
        return "".join(out)

    text=MD.read_text(); blocks=re.split(r"\n\s*\n",text.strip())
    story=[];refs=False; i=0
    while i<len(blocks):
        b=blocks[i].strip()
        if b.startswith("# "):
            title=rich(b[2:]).replace("的国标麻将","的<br/>国标麻将")
            story.append(Paragraph(title,styles["title"]))
        elif b.startswith("作者："):
            story.append(Paragraph(rich(b),styles["meta"]))
        elif b.startswith("## "):
            refs=b[3:]=="参考文献"
            if b.startswith("## 5."):story.append(PageBreak())
            story.append(Paragraph(rich(b[3:]),styles["h2"]))
        elif b.startswith("### "):
            story.append(Paragraph(rich(b[4:]),styles["h3"]))
        elif b.startswith("$$"):
            m=math[b.strip("$").strip()]; scale=min(1.,width/m["width"])
            im=Image(m["path"],width=m["width"]*scale,height=m["height"]*scale)
            story.extend([Spacer(1,3),im,Spacer(1,7)])
        elif b.startswith("!["):
            path=HERE/re.search(r"\]\((.+)\)",b).group(1)
            with PILImage.open(path) as im:iw,ih=im.size
            w=width*.95
            group=[Image(str(path),width=w,height=w*ih/iw),Spacer(1,4)]
            if i+1<len(blocks) and blocks[i+1].startswith("*图"):
                i+=1;group.append(Paragraph(rich(blocks[i]),styles["caption"]))
            story.append(KeepTogether(group))
        elif b.startswith("|"):
            rows=[[c.strip() for c in line.strip().strip("|").split("|")] for line in b.splitlines()]
            rows=[row for row in rows if not all(re.fullmatch(r"[-: ]+",c) for c in row)]
            data=[[Paragraph(rich(c if ix else "**"+c+"**"),styles["cell"]) for c in row] for ix,row in enumerate(rows)]
            table=Table(data,colWidths=[118,67,86,78,width-349],hAlign="CENTER",repeatRows=1)
            table.setStyle(TableStyle([
                ("LINEABOVE",(0,0),(-1,0),.9,colors.HexColor("#3d4a58")),
                ("LINEBELOW",(0,0),(-1,0),.55,colors.HexColor("#3d4a58")),
                ("LINEBELOW",(0,-1),(-1,-1),.9,colors.HexColor("#3d4a58")),
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                ("TOPPADDING",(0,0),(-1,-1),7), ("BOTTOMPADDING",(0,0),(-1,-1),7),
                ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),
            ]))
            story.append(KeepTogether([table,Spacer(1,8)]))
        elif b.startswith("**表"):
            para=Paragraph(rich(b),styles["caption"]);para.keepWithNext=True;story.append(para)
        else:
            style="ref" if refs and b.startswith("[") else "note" if b.startswith("**工具") or b.startswith("**关键词") else "body"
            story.append(Paragraph(rich(b.replace("\n"," ")),styles[style]))
        i+=1

    def page(canvas,doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#6b6b6b"));canvas.setFont("CN",8)
        if doc.page>1:
            canvas.drawString(54,A4[1]-30,"国标麻将智能体设计与实现")
            canvas.setStrokeColor(colors.HexColor("#cccccc"));canvas.setLineWidth(.3)
            canvas.line(54,A4[1]-36,A4[0]-54,A4[1]-36)
        canvas.drawCentredString(A4[0]/2,28,str(doc.page));canvas.restoreState()

    output=HERE/"FINAL_COURSE_REPORT.pdf"
    doc=SimpleDocTemplate(str(output),pagesize=A4,leftMargin=54,rightMargin=54,topMargin=49,bottomMargin=45,
        title="基于行为克隆与 PPO 强化学习的国标麻将智能体设计与实现",author="【姓名】",pageCompression=1)
    doc.build(story,onFirstPage=page,onLaterPages=page)
    reader=PdfReader(output);extracted="\n".join(p.extract_text() for p in reader.pages)
    for token in ("768", "422", "1024", "68.92%", "0.2873", "【姓名】", "【学号】", "完成提交并成功运行"):
        assert token in extracted, token
    assert 4<=len(reader.pages)<=6, f"Unexpected page count: {len(reader.pages)}"
    body=text.split("## 1.",1)[1].split("## 参考文献",1)[0]
    qa={"pages":len(reader.pages),"page_size":"A4", "body_chinese_characters":len(re.findall(r"[\u4e00-\u9fff]",body)),
        "all_chinese_characters":len(re.findall(r"[\u4e00-\u9fff]",text)),
        "markdown_sha256":sha(MD),"pdf_sha256":sha(output), "pdf_bytes":output.stat().st_size,
        "expected_text_tokens_verified":True,"font":str(font),"font_subfont_index":font_index,
        "bold_font":str(bold),"bold_subfont_index":bold_index,
        "visual_review":"Required after rendering; not established by this script."}
    (TEMP/"extracted.txt").write_text(extracted)
    (HERE/"BUILD_CHECK.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(qa,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-sources",action="store_true")
    p.add_argument("--assets",action="store_true")
    p.add_argument("--pdf",action="store_true")
    p.add_argument("--font",type=Path,default=FONT)
    p.add_argument("--bold-font",type=Path,default=BOLD)
    p.add_argument("--font-index",type=int,default=6)
    p.add_argument("--bold-font-index",type=int,default=1)
    a=p.parse_args()
    if not any((a.check_sources,a.assets,a.pdf)):p.error("Select --check-sources, --assets and/or --pdf")
    if a.check_sources:check_sources()
    if a.assets:make_assets(a.font)
    if a.pdf:make_pdf(a.font,a.bold_font,a.font_index,a.bold_font_index)


if __name__ == "__main__":
    main()
