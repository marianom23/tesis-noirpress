from pathlib import Path
import re
import html
import json
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image, KeepTogether, CondPageBreak,
)
from reportlab.platypus.tableofcontents import TableOfContents
from pypdf import PdfReader
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "documento"
OUT = ROOT / "archivo/Tesina Noirpress - anterior a plantilla UGR.pdf"
FIG = ROOT / "resultados_noirpress/evaluacion_final/figuras"
OUT.parent.mkdir(parents=True, exist_ok=True)

TABLES = {int(x["number"]): (x["title"], x["widths"], x["rows"], x.get("note", "")) for x in json.loads((WORK / "tablas_definitivas.json").read_text(encoding="utf-8"))}
FIGURES = {int(x["number"]): (x["basename"], x["titulo"], x["nota"]) for x in json.loads((FIG / "catalogo_figuras.json").read_text(encoding="utf-8"))}

FONTS = Path("C:/Windows/Fonts")
for name, filename in [
    ("Georgia", "georgia.ttf"), ("Georgia-Bold", "georgiab.ttf"),
    ("Georgia-Italic", "georgiai.ttf"), ("Georgia-BoldItalic", "georgiaz.ttf"),
    ("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"), ("Arial-Italic", "ariali.ttf"),
    ("Segoe", "segoeui.ttf"), ("Segoe-Bold", "seguisb.ttf"), ("Segoe-Italic", "segoeuii.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / filename)))
pdfmetrics.registerFontFamily("Georgia", normal="Georgia", bold="Georgia-Bold",
                             italic="Georgia-Italic", boldItalic="Georgia-BoldItalic")
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")
pdfmetrics.registerFontFamily("Segoe", normal="Segoe", bold="Segoe-Bold", italic="Segoe-Italic")

PAGE_W, PAGE_H = 595.276, 841.89
LEFT, BOTTOM = 58, 58
W = PAGE_W - 2 * LEFT
INK = colors.HexColor("#243A40")
ACCENT = colors.HexColor("#8C6848")
MUTED = colors.HexColor("#667173")
PAPER = colors.HexColor("#F6F3ED")
styles = {
    "body": ParagraphStyle("body", fontName="Georgia", fontSize=10.6, leading=15.5,
                           alignment=TA_LEFT, textColor=colors.HexColor("#303638"),
                           spaceAfter=8, allowWidows=0, allowOrphans=0),
    "chapter": ParagraphStyle("chapter", fontName="Georgia-Bold", fontSize=27, leading=34,
                              textColor=INK, spaceAfter=26, keepWithNext=True),
    "section": ParagraphStyle("section", fontName="Segoe-Bold", fontSize=12.6, leading=17,
                              textColor=INK, spaceBefore=15, spaceAfter=8, keepWithNext=True),
    "caption": ParagraphStyle("caption", fontName="Segoe", fontSize=9.4, leading=13,
                              textColor=INK, spaceBefore=9, spaceAfter=5),
    "note": ParagraphStyle("note", fontName="Segoe", fontSize=8.5, leading=12,
                           textColor=MUTED, spaceAfter=12),
    "cell": ParagraphStyle("cell", fontName="Segoe", fontSize=9.1, leading=12),
    "num": ParagraphStyle("num", fontName="Segoe", fontSize=9.1, leading=12, alignment=TA_CENTER),
    "head": ParagraphStyle("head", fontName="Segoe-Bold", fontSize=8.9, leading=12,
                           textColor=colors.white),
    "item": ParagraphStyle("item", fontName="Georgia", fontSize=10.6, leading=16,
                           spaceAfter=7, leftIndent=14, firstLineIndent=-14,
                           allowWidows=0, allowOrphans=0),
    "formula": ParagraphStyle("formula", fontName="Georgia", fontSize=11, leading=18,
                              alignment=TA_CENTER, spaceBefore=13, spaceAfter=18,
                              backColor=PAPER, borderPadding=12, textColor=INK),
    "reference": ParagraphStyle("reference", fontName="Georgia", fontSize=9.8, leading=14,
                                leftIndent=16, firstLineIndent=-16, spaceAfter=9),
}

def esc(text):
    return html.escape(text).replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")

def para(text, style="body"):
    return Paragraph(esc(text), styles[style])

class Index(TableOfContents):
    def __init__(self, kind, compact=False):
        super().__init__()
        self.kind = kind
        self.levelStyles = [
            ParagraphStyle("index0"+kind, fontName="Segoe-Bold" if not compact else "Segoe",
                           fontSize=10 if not compact else 9.4, leading=12 if not compact else 12.5,
                           textColor=INK, spaceBefore=2 if not compact else 1, rightIndent=26),
            ParagraphStyle("index1"+kind, fontName="Segoe", fontSize=9.2, leading=11,
                           leftIndent=15, firstLineIndent=0, spaceBefore=0, rightIndent=26,
                           textColor=MUTED),
        ]
        self.dotsMinLevel = 0
        self.tableStyle.add("TOPPADDING",(0,0),(-1,-1),0)
        self.tableStyle.add("BOTTOMPADDING",(0,0),(-1,-1),1.5)
    def notify(self, kind, stuff):
        if kind == self.kind:
            self.addEntry(*stuff)

class ThesisDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename), pagesize=(PAGE_W,PAGE_H), leftMargin=LEFT, rightMargin=LEFT,
                         topMargin=60, bottomMargin=BOTTOM, title="Pronóstico de demanda y reposición en una microimportadora local",
                         author="Jose Pasini y Mariano Martín", subject="El caso de Noirpress",
                         pageCompression=1)
        frame = Frame(LEFT,BOTTOM,W,PAGE_H-118,id="main",leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="main",frames=frame,onPageEnd=self.decorate))
        self.outline = []
    def beforeDocument(self):
        self.outline = []
        self.running_title = "CONTENIDO"
    def decorate(self, canv, doc):
        if doc.page == 1:
            self.draw_cover(canv)
            return
        canv.saveState()
        canv.setFont("Segoe-Bold",7.5)
        canv.setFillColor(INK)
        canv.drawString(LEFT,PAGE_H-31,"NOIRPRESS  /  TESINA")
        canv.setFont("Segoe",7.5)
        canv.setFillColor(MUTED)
        canv.drawRightString(PAGE_W-LEFT,PAGE_H-31,self.running_title.upper())
        canv.setStrokeColor(colors.HexColor("#DADDD9"))
        canv.setLineWidth(.5)
        canv.line(LEFT,PAGE_H-44,PAGE_W-LEFT,PAGE_H-44)
        canv.setFont("Segoe",8)
        canv.drawString(LEFT,32,"Jose Pasini · Mariano Martín")
        canv.setFont("Segoe-Bold",9)
        canv.setFillColor(INK)
        canv.drawRightString(PAGE_W-LEFT,32,f"{doc.page:02d}")
        canv.restoreState()
    def draw_cover(self, canv):
        canv.saveState()
        canv.setFillColor(PAPER)
        canv.rect(0,0,PAGE_W,PAGE_H,fill=1,stroke=0)
        canv.setFillColor(INK)
        canv.rect(0,PAGE_H-14,PAGE_W,14,fill=1,stroke=0)
        canv.setFont("Segoe-Bold",10)
        canv.drawString(LEFT,775,"TESINA")
        canv.setFont("Segoe",9)
        canv.drawRightString(PAGE_W-LEFT,775,"OCTUBRE 2026")
        canv.setStrokeColor(ACCENT)
        canv.setLineWidth(1)
        canv.line(LEFT,735,PAGE_W-LEFT,735)
        canv.setFillColor(ACCENT)
        canv.setFont("Segoe-Bold",10)
        canv.drawString(LEFT,680,"ESTUDIO DE CASO")
        title = Paragraph("Pronóstico de demanda<br/>y reposición en una<br/>microimportadora local",
                          ParagraphStyle("cover",fontName="Georgia-Bold",fontSize=29,leading=40,textColor=INK))
        _, height = title.wrap(W,200)
        title.drawOn(canv,LEFT,640-height)
        canv.setFillColor(INK)
        canv.setFont("Georgia-Italic",17)
        canv.drawString(LEFT,475,"El caso de Noirpress")
        canv.setStrokeColor(ACCENT)
        canv.setLineWidth(2)
        canv.line(LEFT,421,LEFT+48,421)
        canv.setFont("Segoe",9)
        canv.setFillColor(MUTED)
        canv.drawString(LEFT,372,"AUTORES")
        canv.setFillColor(INK)
        canv.setFont("Georgia",21)
        canv.drawString(LEFT,342,"Jose Pasini")
        canv.drawString(LEFT,312,"Mariano Martín")
        canv.setStrokeColor(colors.HexColor("#D4CDC0"))
        canv.setLineWidth(.6)
        canv.line(LEFT,251,PAGE_W-LEFT,251)
        canv.setFillColor(INK)
        canv.setFont("Segoe-Bold",13)
        canv.drawString(LEFT,218,"Universidad del Gran Rosario")
        canv.setFillColor(MUTED)
        canv.setFont("Segoe",10)
        canv.drawString(LEFT,194,"Tesina · Octubre de 2026")
        canv.setFont("Segoe-Bold",8)
        canv.setFillColor(ACCENT)
        canv.drawString(LEFT,65,"NOIRPRESS")
        canv.restoreState()
    def afterFlowable(self, flowable):
        if hasattr(flowable,"running_title"):
            self.running_title = flowable.running_title
        entry = getattr(flowable,"index_entry",None)
        if entry:
            kind,level,title,key = entry
            self.canv.bookmarkPage(key)
            self.notify(kind,(level,esc(title),self.page,key))
            if kind == "TOCEntry":
                self.canv.addOutlineEntry(title,key,level=level,closed=level==0)
                self.outline.append((title,self.page))

def heading(title, level, index=True):
    display = esc(title)
    match = re.match(r"^(Capítulo|Anexo) ([0-9A-Z]+) (.+)$",title)
    if level == 0 and match:
        kind,number,name = match.groups()
        if number.isdigit():
            number = number.zfill(2)
        display = f'<font name="Segoe-Bold" size="9" color="#8C6848">{kind.upper()} {number}</font><br/>{esc(name)}'
    p = Paragraph(display,styles["chapter" if level == 0 else "section"])
    if level == 0:
        p.running_title = match[3] if match else title
    if index:
        key = "h" + str(len(headings))
        headings.append(title)
        p.index_entry = ("TOCEntry",level,title,key)
    return p

def make_table(n, lead=None):
    title,widths,rows,note = TABLES[n]
    cap = Paragraph(f"<b>Tabla {n}.</b> {esc(title)}",styles["caption"])
    cap.index_entry = ("TableEntry",0,f"Tabla {n}. {title}",f"t{n}")
    cap.keepWithNext = True
    numeric = n in {3,4,5,6,7,8}
    data = []
    for ri,row in enumerate(rows):
        cells = []
        for ci,v in enumerate(row):
            style = "head" if ri == 0 else ("num" if numeric and ci else "cell")
            text = esc(v)
            cells.append(Paragraph(text,styles[style]))
        data.append(cells)
    tb = Table(data,colWidths=[x*W for x in widths],repeatRows=1,hAlign="LEFT")
    tb.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),INK),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F3F4F0")]),
        ("LINEBELOW",(0,0),(-1,0),1,ACCENT),
        ("LINEBELOW",(0,1),(-1,-1),.35,colors.HexColor("#DBDFDA")),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
        ("TOPPADDING",(0,0),(-1,-1),5.5),("BOTTOMPADDING",(0,0),(-1,-1),5.5),
    ]))
    result = ([lead] if lead is not None else []) + [cap,tb,Spacer(1,5)]
    if note:
        result.append(para(note,"note"))
    else:
        result.append(Spacer(1,7))
    return [KeepTogether(result)]

def make_figure(n):
    name,title,note = FIGURES[n]
    path = FIG / (name+".png")
    with PILImage.open(path) as im:
        width,height = im.size
    target_width = W
    scale = min(target_width/width,350/height)
    img = Image(str(path),width=width*scale,height=height*scale,hAlign="CENTER")
    cap = Paragraph(f"<b>Figura {n}.</b> {esc(title)}",styles["caption"])
    cap.index_entry = ("FigureEntry",0,f"Figura {n}. {title}",f"f{n}")
    return KeepTogether([Spacer(1,8),img,cap,para(note,"note"),Spacer(1,6)])

def references():
    refs = json.loads((WORK / "referencias_revisadas.json").read_text(encoding="utf-8"))
    selected = set(json.loads((WORK / "referencias_usadas.json").read_text(encoding="utf-8")))
    assert selected <= {r["id"] for r in refs}
    return [r for r in refs if r["id"] in selected]

def reference_paragraph(ref):
    value = esc(ref["ref"])
    for key in ("url", "url2"):
        url = ref.get(key)
        if url:
            value += ' <link href="' + html.escape(url, quote=True) + '" color="#294E68">' + esc(url) + '</link>'
    return Paragraph(value, styles["reference"])

headings = []
if __name__ == "__main__":
    story = [Spacer(1,1), PageBreak()]
    story += [heading("Índice",0,False), Index("TOCEntry",False), PageBreak()]
    story += [heading("Figuras y tablas",0,False), Index("FigureEntry",True), Spacer(1,24), Index("TableEntry",True), PageBreak()]

    blocks = (WORK / "texto_definitivo.txt").read_text(encoding="utf-8").strip().split("\n\n")
    used_figs, used_tables = [], []
    first_chapter = True
    for block in blocks:
        block = block.strip()
        if block.startswith("@chapter "):
            if not first_chapter:
                story.append(PageBreak())
            first_chapter = False
            story.append(heading(block.removeprefix("@chapter "),0))
        elif block.startswith("@section "):
            story.append(heading(block.removeprefix("@section "),1))
        elif block.startswith("@figure "):
            n = int(block.split()[1]); used_figs.append(n)
            story.append(make_figure(n))
        elif block.startswith("@table "):
            n = int(block.split()[1]); used_tables.append(n)
            lead = story.pop() if isinstance(story[-1],Paragraph) and story[-1].style.name == "section" else None
            story.extend(make_table(n,lead))
        elif block.startswith("@formula "):
            story.append(para(block.removeprefix("@formula "),"formula"))
        elif block.startswith("@item "):
            for item in block.splitlines():
                assert item.startswith("@item "), item
                story.append(para(item.removeprefix("@item "),"item"))
        elif block == "@bibliography":
            story.append(PageBreak())
            story.append(heading("Bibliografía",0))
            story.extend(KeepTogether([reference_paragraph(ref)]) for ref in references())
        else:
            assert not block.startswith("@"),block
            story.append(para(block))

    assert used_figs == sorted(FIGURES)
    assert used_tables == sorted(TABLES)
    doc = ThesisDoc(OUT)
    doc.multiBuild(story)
    reader = PdfReader(OUT)
    text = "\n".join(p.extract_text() for p in reader.pages)
    for required in ["Universidad del Gran Rosario", "Jose Pasini", "Mariano Martín", "Ignacio Giordano", "391", "405"]:
        assert required in text, required
    for forbidden in ["[Completar", "[Pendiente", "Datos pendientes de confirmación", "Abstract", "base de evaluación de 413", "no se reestimaron"]:
        assert forbidden not in text, forbidden
    assert all(f"Figura {n}." in text for n in used_figs)
    assert all(f"Tabla {n}." in text for n in used_tables)
    print(f"PDF: {OUT}\nPáginas: {len(reader.pages)}\nFiguras: {len(used_figs)}\nTablas: {len(used_tables)}")

