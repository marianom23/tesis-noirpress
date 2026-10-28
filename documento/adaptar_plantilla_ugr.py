"""Adapta la tesina vigente a la referencia UGR sin recalcular sus resultados."""
from collections import Counter
from pathlib import Path
import json
import re

import build_definitiva as base
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak, KeepTogether,
)
from pypdf import PdfReader

ROOT = base.ROOT
OUT = ROOT / 'Tesina Noirpress - version final.pdf'
TEXT = base.WORK / 'texto_plantilla_ugr.txt'
LOGO = base.WORK / 'plantilla_ugr/logo_ugr.jpg'
TITLE = 'Pronóstico de demanda y reposición en una microimportadora local. El caso de Noirpress'


def adapt_text():
    original = (base.WORK / 'texto_definitivo.txt').read_text(encoding='utf-8')
    parts = {}
    for block in original.strip().split('\n\n'):
        if block.startswith(('@chapter ', '@section ')):
            key = block.split(' ', 1)[1]
            parts[key] = []
        elif block != '@bibliography':
            parts[key].append(block)
    consumed = set()

    def take(key):
        assert key not in consumed, key
        consumed.add(key)
        return parts[key]

    out = []

    def add(title, paragraphs, section=False):
        out.append(('@section ' if section else '@chapter ') + title)
        out.extend(paragraphs)

    add('Resumen', take('Resumen'))
    out.append('Temática de Interés en Ciencia de Datos: pronóstico de series temporales y análisis de decisiones de reposición.')
    question, general, intro, objectives, hypotheses, scope = take('1.3 Pregunta, objetivos e hipótesis')
    add('1. Problemática observada', take('1.1 El problema de una microimportadora local'))
    add('1.1 El caso Noirpress', take('1.2 El caso Noirpress'), True)
    add('1.2 Pregunta de investigación', [question], True)
    add('2. Hipótesis de Investigación', [hypotheses, scope])
    add('3. Objetivo General', [general.removeprefix('El objetivo general es ').capitalize()])
    add('4. Objetivos Específicos', [intro, objectives])
    add('5. Marco teórico', [])
    for n, title in enumerate([
        'Inventario, demanda durante la espera y reposición',
        'Qué representa una serie de ventas',
        'Métodos simples, estadísticos y preentrenados',
        'Evaluación temporal e interpretación del error',
    ], 1):
        add(f'5.{n} {title}', take(f'2.{n} {title}'), True)
    approach, roadmap = take('1.4 Enfoque y aporte')
    add('6. Metodología', [approach,
        'El procedimiento comprende la preparación de los registros, la construcción de la serie diaria, la comparación temporal de métodos y la simulación de compras. Las fuentes se describen en la sección 7; los resultados, en la sección 8. La discusión y las conclusiones se presentan por separado, y el Anexo A reúne los elementos de reproducibilidad.'])
    sources = take('3.1 Fuentes, período y depuración')
    add('6.1 Depuración y criterios de inclusión', sources[2:], True)
    for n, old_n, title in [
        (2, 2, 'Reconstrucción de cantidades y serie diaria'),
        (3, 3, 'Exploración y variables disponibles'),
        (4, 4, 'Métodos comparados'),
        (5, 5, 'Protocolo temporal y métricas'),
        (6, 6, 'Inventario y escenarios de compra'),
    ]:
        add(f'6.{n} {title}', take(f'3.{old_n} {title}'), True)
    add('7. Descripción de los datos', sources[:2])
    add('7.1 Registros de órdenes y recepciones', [
        'La exportación de órdenes constituye la fuente principal del historial de ventas. La planilla de recepciones aporta las entradas de mercadería utilizadas en la reconstrucción del inventario. Ambas fuentes son registros internos de Noirpress.',
        'La base resultante conserva 391 pedidos y genera 405 unidades estimadas en 179 fechas. Los pedidos se observan directamente; las unidades se reconstruyen a partir de importes. Las recepciones documentadas y los saldos calculados se mantienen diferenciados para evitar presentar una reconstrucción contable como inventario físico.'
    ], True)
    add('8. Resultados', [])
    for n, title in enumerate([
        'Comportamiento de la serie depurada', 'Canales y concentración comercial',
        'Patrón semanal y estabilidad', 'Comparación de pronósticos',
        'Sesgo y concentración del error', 'Saldo reconstruido y disponibilidad inicial',
        'Reposición con fechas de llegada explícitas', 'Síntesis de los resultados',
    ], 1):
        add(f'8.{n} {title}', take(f'4.{n} {title}'), True)
    add('9. Implementación del procedimiento', [
        'Se implementó un procedimiento de análisis que integra la depuración de registros, la estimación de cantidades, el cálculo de pronósticos y la simulación de compras. Los programas generaron las tablas y figuras presentadas en este trabajo. La evaluación se realizó sobre datos históricos; la aplicación a decisiones reales de compra queda planteada como una etapa posterior.'
    ] + take('3.7 Implementación, controles y autorización'))
    add('10. Discusión', [
        'La elección del método depende del horizonte y del momento en que se lo selecciona. La mediana de 28 días fue seleccionada mediante validación; los mejores resultados de prueba correspondieron a SARIMAX en el horizonte diario y a la media de 28 días en el acumulado. Elegir retrospectivamente al ganador de prueba como si hubiera sido conocido antes de ese tramo confundiría evaluación y selección.',
        'Las referencias simples resultaron competitivas frente a los modelos de mayor complejidad. En este caso, la incorporación de métodos más complejos no aseguró el menor error acumulado. Su conveniencia debe evaluarse para cada horizonte y con los datos disponibles, tal como se plantea en los antecedentes de pronóstico.',
        'La concentración de ventas en pocos días y las diferencias entre canales condicionan la interpretación del error. La falta de observaciones de disponibilidad impide separar con certeza una caída de demanda de una restricción de venta. Por eso, las predicciones deben interpretarse como estimaciones de la serie registrada y reconstruida, no como mediciones de demanda potencial no atendida.',
        'En los escenarios de reposición, el momento de recepción delimita el efecto de la cantidad comprada. El faltante anterior a la llegada es compartido por las reglas comparadas porque ninguna dispone todavía de las unidades pedidas. Esta relación explica por qué una mejora en precisión no basta para demostrar una mejora comercial: también deben observarse la posición de inventario, los plazos y las restricciones de compra.'
    ])
    add('11. Conclusiones', take('5.1 Respuesta y aporte del estudio') + take('5.3 Cierre'))
    implications = take('5.2 Implicaciones y límites')
    add('12. Limitaciones y trabajo futuro', [])
    add('12.1 Limitaciones', implications[1:], True)
    add('12.2 Trabajo futuro', take('A.3 Alcance de una aplicación posterior'), True)
    add('13. Recursos disponibles', [
        'El material complementario reúne las planillas originales, los programas de procesamiento, las predicciones, las métricas y los escenarios de inventario. La ubicación de estos archivos y las versiones utilizadas se detallan en el Anexo A.',
        'El archivo LEEME.txt describe las dependencias y el orden de ejecución de los programas. Los archivos se conservan localmente. Este trabajo no incluye la publicación abierta de los registros comerciales.'
    ])
    add('14. Impacto esperado', [implications[0],
        'Se espera que el procedimiento facilite revisiones de compra documentadas y comparables en el tiempo. Su utilidad potencial reside en explicitar qué ventas se esperan durante la demora de abastecimiento, qué existencias y recepciones pueden cubrirlas y qué supuestos sostienen la decisión. Ese impacto debe evaluarse con datos nuevos; no se cuantifican ahorros, ventas recuperadas ni reducciones de quiebres reales.'
    ])
    add('15. Referencias', ['@bibliography'])
    add('16. Anexos', [])
    add('Anexo A. Reproducibilidad y métricas', [], True)
    add('A.1 Trazabilidad de datos y cálculos', take('A.1 Trazabilidad de datos y cálculos'), True)
    add('A.2 Definición de las métricas', take('A.2 Definición de las métricas'), True)

    assert all(not value or key in consumed for key, value in parts.items())
    result = '\n\n'.join(out) + '\n'
    for marker in ('figure', 'table', 'formula', 'item'):
        pattern = rf'^@{marker} .+$'
        assert Counter(re.findall(pattern, original, re.M)) == Counter(re.findall(pattern, result, re.M)), marker
    # Las únicas frases originales reemplazadas describían el índice anterior y el objetivo general.
    for key in consumed:
        for paragraph in parts[key]:
            if paragraph not in (roadmap, general):
                assert paragraph in result, key
    edits = json.loads((base.WORK / 'revisiones_redaccion_ugr.json').read_text(encoding='utf-8'))
    for old, new in edits:
        assert result.count(old) == 1, old[:80]
        assert re.findall(r'\d+(?:[.,]\d+)*', old) == re.findall(r'\d+(?:[.,]\d+)*', new)
        result = result.replace(old, new)
    corrections = json.loads((base.WORK / 'correcciones_consistencia_ugr.json').read_text(encoding='utf-8'))
    for correction in corrections:
        assert result.count(correction['antes']) == 1, correction['motivo']
        result = result.replace(correction['antes'], correction['despues'])
    TEXT.write_text(result, encoding='utf-8')
    return result


class TemplateDoc(base.ThesisDoc):
    def __init__(self, filename):
        BaseDocTemplate.__init__(
            self, str(filename), pagesize=(base.PAGE_W, base.PAGE_H),
            title=TITLE, author='Jose Pasini y Mariano Martín',
            subject='Trabajo final de Ciencia de Datos - Noirpress',
            leftMargin=base.LEFT, rightMargin=base.LEFT,
            topMargin=83, bottomMargin=45,
        )
        frame = Frame(base.LEFT, 45, base.W, base.PAGE_H - 128,
                      id='main', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id='main', frames=frame, onPageEnd=self.decorate))

    def afterFlowable(self, flowable):
        entry = getattr(flowable, 'index_entry', None)
        if entry:
            kind, level, title, key = entry
            self.canv.bookmarkPage(key)
            self.notify(kind, (level, base.esc(title), self.page - 1, key))
            if kind == 'TOCEntry':
                self.canv.addOutlineEntry(title, key, level=level, closed=level == 0)

    def decorate(self, canv, doc):
        if doc.page == 1:
            self.draw_cover(canv)
            return
        canv.saveState()
        canv.drawImage(str(LOGO), base.LEFT, base.PAGE_H - 69, width=171, height=42, mask='auto')
        canv.setFillColor(colors.black)
        canv.setFont('Arial', 9)
        canv.drawRightString(base.PAGE_W - base.LEFT, base.PAGE_H - 49, 'Trabajo Final')
        canv.drawRightString(base.PAGE_W - base.LEFT, base.PAGE_H - 61, 'C.C.C Lic. en Ciencia de Datos')
        canv.setStrokeColor(colors.grey)
        canv.setLineWidth(.5)
        canv.line(base.LEFT, 32, base.PAGE_W - base.LEFT, 32)
        canv.setFont('Arial-Italic', 9)
        canv.drawString(base.LEFT, 19, 'Autores: Jose Pasini y Mariano Martín')
        canv.setFont('Arial', 9)
        canv.drawRightString(base.PAGE_W - base.LEFT, 19, str(doc.page - 1))
        canv.restoreState()

    def draw_cover(self, canv):
        canv.saveState()
        canv.drawImage(str(LOGO), (base.PAGE_W-171)/2, 690, width=171, height=42, mask='auto')

        def centered(text, y, size=12, bold=False, color=colors.black):
            style = ParagraphStyle('cover', fontName='Arial-Bold' if bold else 'Arial',
                                   fontSize=size, leading=size*1.3, alignment=TA_CENTER, textColor=color)
            p = Paragraph(text, style)
            _, height = p.wrap(base.W, 160)
            p.drawOn(canv, base.LEFT, y-height)

        centered('TRABAJO FINAL', 638, 20)
        centered('Seminario de Trabajo Final', 592, 23, True)
        centered('C.C.C Lic. en Ciencia de Datos', 525, 18, True, colors.HexColor('#A00000'))
        centered('Título: &quot;' + TITLE + '&quot;', 416, 13, True)
        centered('Autores:<br/>Jose Pasini<br/>Mariano Martín', 292, 11, True)
        centered('Octubre de 2026', 102, 11, True)
        canv.restoreState()


def build():
    text = adapt_text()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    base.LEFT = 72
    base.W = base.PAGE_W - 2*base.LEFT
    base.INK = colors.black
    base.MUTED = colors.HexColor('#333333')
    base.ACCENT = colors.grey
    title, _, rows, note = base.TABLES[5]
    base.TABLES[5] = (title, [.23, .12, .19, .15, .14, .17], rows, note)
    base.TABLES[6][2][3] = ['H3', 'Ilustrada en los escenarios',
        'La cantidad pedida no modifica los faltantes previos a la recepción. Es una propiedad del modelo; no prueba el efecto de variar plazos reales.']
    name, title, note = base.FIGURES[4]
    base.FIGURES[4] = (name, title, note.replace('referencia de precios', 'referencia de importes'))
    for name, style in base.styles.items():
        style.fontName = 'Arial-Bold' if name in ('chapter', 'section', 'head') else 'Arial'
        style.textColor = colors.white if name == 'head' else colors.black
    for name in ('body', 'item'):
        base.styles[name].fontSize = 11
        base.styles[name].leading = 14
        base.styles[name].spaceAfter = 9
        base.styles[name].alignment = TA_JUSTIFY
    for name in ('chapter', 'section'):
        base.styles[name].fontSize = 11.5 if name == 'chapter' else 11
        base.styles[name].leading = 15
        base.styles[name].spaceBefore = 17 if name == 'chapter' else 10
        base.styles[name].spaceAfter = 10
    base.styles['formula'].backColor = None
    base.styles['formula'].borderPadding = 0
    base.styles['formula'].fontSize = 10.5
    base.headings.clear()
    toc = base.Index('TOCEntry')
    for style in toc.levelStyles:
        style.fontName = 'Arial'
        style.textColor = colors.black
    story = [Spacer(1, 1), PageBreak(), base.heading('Índice', 0, False), toc, PageBreak()]
    used_figs, used_tables = [], []
    for block in text.strip().split('\n\n'):
        if block.startswith('@chapter '):
            title = block.removeprefix('@chapter ')
            if title in ('15. Referencias', '16. Anexos'):
                story.append(PageBreak())
            if title == 'Resumen':
                story.append(Paragraph('<b>TRABAJO FINAL</b><br/>' + base.esc(TITLE),
                                       ParagraphStyle('opening', parent=base.styles['body'], alignment=TA_CENTER)))
            story.append(base.heading(title, 0))
        elif block.startswith('@section '):
            story.append(base.heading(block.removeprefix('@section '), 1))
        elif block.startswith('@figure '):
            n = int(block.split()[1])
            used_figs.append(n)
            story.append(base.make_figure(n))
        elif block.startswith('@table '):
            n = int(block.split()[1])
            used_tables.append(n)
            lead = story.pop() if isinstance(story[-1], Paragraph) and story[-1].style.name == 'section' else None
            story.extend(base.make_table(n, lead))
        elif block.startswith('@item '):
            for line in block.splitlines():
                assert line.startswith('@item ')
                story.append(base.para(line.removeprefix('@item '), 'item'))
        elif block.startswith('@formula '):
            story.append(base.para(block.removeprefix('@formula '), 'formula'))
        elif block == '@bibliography':
            story.extend(KeepTogether([base.reference_paragraph(ref)]) for ref in base.references())
        else:
            assert not block.startswith('@'), block
            story.append(base.para(block))
    assert used_figs == sorted(base.FIGURES)
    assert used_tables == sorted(base.TABLES)
    TemplateDoc(OUT).multiBuild(story)
    reader = PdfReader(OUT)
    rendered = '\n'.join(page.extract_text() for page in reader.pages)
    for required in ('Jose Pasini', 'Mariano Martín', '391', '405', '179', 'Ignacio Giordano',
                     '2. Hipótesis de Investigación', '15. Referencias', '16. Anexos'):
        assert required in rendered, required
    for forbidden in ('Camila', '43.825.006', 'PrematurApp', 'Capítulo 4', '[Completar', '27/07/2026'):
        assert forbidden not in rendered, forbidden
    for forbidden in ('proyecto de trabajo final', 'borrador', '[pendiente', 'no se afirma', 'como modelo de lenguaje'):
        assert forbidden not in rendered.lower(), forbidden
    assert all(f'Figura {n}.' in rendered for n in used_figs)
    assert all(f'Tabla {n}.' in rendered for n in used_tables)
    print(f'PDF: {OUT}\nPáginas: {len(reader.pages)}\nFiguras: {len(used_figs)}\nTablas: {len(used_tables)}')


if __name__ == '__main__':
    build()
