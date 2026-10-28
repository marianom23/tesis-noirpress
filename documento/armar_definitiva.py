from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'documento'
R = ROOT / 'resultados_noirpress/evaluacion_final'
ev = json.loads((R / 'resumen_final.json').read_text(encoding='utf-8'))
desc = json.loads((R / 'resumen_descriptivo.json').read_text(encoding='utf-8'))
m = pd.read_csv(R / 'comparacion_modelos.csv')
d = pd.read_csv(R / 'serie_diaria.csv', parse_dates=['fecha']).set_index('fecha')
concentration = pd.read_csv(R / 'concentracion_error.csv').set_index('modelo')
assert ev['unidades_estimadas'] == 405 and ev['pedidos'] == 391
assert len(m) == 36 and not ev['modelos_no_convergentes']

def fmt(x, digits=1):
    return f'{float(x):.{digits}f}'.replace('.', ',')

def metric(name, horizon, stage='prueba'):
    result = m[(m.modelo == name) & (m.horizonte_dias == horizon) & (m.etapa == stage)]
    assert len(result) == 1
    return result.iloc[0]

tables = []
def table(number, title, widths, rows, note=''):
    assert all(len(row) == len(widths) for row in rows)
    assert abs(sum(widths) - 1) < .001
    tables.append(dict(number=number,title=title,widths=widths,rows=rows,note=note))

table(1,'Períodos y casos de evaluación',[.25,.35,.2,.2],[
    ['Tramo','Fechas de observación','Días','Ventanas completas de 30 días'],
    ['Historial inicial','3 de enero a 13 de marzo','70','No se puntúa'],
    ['Validación','14 de marzo a 7 de mayo','55','26'],
    ['Prueba','8 de mayo a 30 de junio','54','25']],
    'Año 2026. Las ventanas acumuladas consecutivas se superponen en 29 días.')
months = ['Enero','Febrero','Marzo','Abril','Mayo','Junio']
monthly = [['Mes','Pedidos','Unidades estimadas','Media diaria de unidades','Días sin pedidos']]
for name,row in zip(months,desc['mensuales']):
    monthly.append([name,str(row['pedidos']),str(int(row['unidades'])),fmt(row['media'],2),str(row['dias_sin_pedidos'])])
monthly.append(['Total','391','405',fmt(desc['media_diaria'],2),'34'])
table(2,'Evolución mensual de la base depurada',[.18,.16,.23,.25,.18],monthly)

ranking = m[(m.etapa == 'validacion') & (m.horizonte_dias == 30)].sort_values('MAE').modelo.tolist()
rows = [['Método','Diario validación','Diario prueba','30 días validación','30 días prueba']]
for name in ranking:
    label = 'Media de 7 días (M1)' if name == 'M1 histórico' else name
    rows.append([label]+[fmt(metric(name,h,s).MAE,2 if h == 1 else 1) for h,s in [(1,'validacion'),(1,'prueba'),(30,'validacion'),(30,'prueba')]])
table(3,'Error absoluto medio de los nueve métodos',[.32,.17,.17,.17,.17],rows,
      'MAE en unidades estimadas: por día o por ventana de 30 días. Orden según MAE acumulado de validación. Todos los métodos utilizan la misma serie depurada.')
stock = [['Stock inicial supuesto','Saldo mínimo','Días con saldo negativo','Saldo al 30 de junio']]
for row in desc['inventario']:
    if row['Stock inicial supuesto'] in [0,25,30,50,98]:
        stock.append([str(row['Stock inicial supuesto']),fmt(row['Mínimo contable'],0),str(row['Días saldo negativo']),fmt(row['Saldo 30 junio'],0)])
table(4,'Sensibilidad del saldo al inventario inicial',[.25,.25,.25,.25],stock,
      'Reconstrucción con 420 unidades recibidas y 405 unidades estimadas vendidas. Los saldos no equivalen a conteos físicos.')
scenarios = [['Regla','Compra','Faltante antes de llegada','Faltante después','Stock medio','Atención de demanda']]
for row in desc['escenarios'][:3]:
    scenarios.append([row['Método'],str(row['Pedido calculado']),fmt(row['Faltantes antes llegada'],0),fmt(row['Faltantes después llegada'],0),fmt(row['Stock medio']),fmt(row['Atendido_pct'])+' %'])
table(5,'Escenario del 27 de abril, con llegada el 27 de mayo',[.25,.1,.19,.15,.14,.17],scenarios,
      'Inventario inicial simulado: 50 unidades. Evaluación de 60 días, sin otras compras en tránsito. Las magnitudes representan unidades estimadas.')
table(6,'Respuesta a las hipótesis del estudio',[.18,.36,.46],[
    ['Hipótesis','Resultado','Alcance de la evidencia'],
    ['H1','Respaldada en la comparación','SARIMAX y TimesFM superan a la media de siete días en ambos horizontes de prueba.'],
    ['H2','Apoyo retrospectivo','La media de 28 días obtiene el menor MAE acumulado de prueba; la mediana de 28 días es la seleccionada en validación.'],
    ['H3','Verificada en los escenarios','Cambiar la cantidad pedida no modifica los faltantes anteriores a la recepción. Es una condición temporal del modelo de inventario.']])
table(7,'Definición de las métricas',[.2,.46,.34],[
    ['Métrica','Cálculo','Lectura'],
    ['MAE','Promedio de |p − y|','Error absoluto medio.'],
    ['RMSE','Raíz del promedio de (p − y)²','Da mayor peso a errores grandes.'],
    ['Sesgo','Promedio de p − y','Negativo: subestimación.'],
    ['WAPE','100 × suma de |p − y| / suma de |y|','Error agregado relativo al volumen.'],
    ['Cobertura','Casos dentro del intervalo / casos evaluados','Se interpreta junto con el ancho del intervalo.']])

sar1, sar30 = metric('SARIMAX',1), metric('SARIMAX',30)
tf1, tf30 = metric('TimesFM 3.0',1), metric('TimesFM 3.0',30)
mean30, med30 = metric('Media 28 días',30), metric('Mediana 28 días',30)
m11, m130 = metric('M1 histórico',1), metric('M1 histórico',30)
assert sar1.MAE < m11.MAE and sar30.MAE < m130.MAE
assert tf1.MAE < m11.MAE and tf30.MAE < m130.MAE
assert ev['seleccion']['nueve_exploratorios'] == {'1':'Mediana 28 días','30':'Mediana 28 días'}
assert m[(m.etapa == 'prueba') & (m.horizonte_dias == 1)].sort_values('MAE').iloc[0].modelo == 'SARIMAX'
assert m[(m.etapa == 'prueba') & (m.horizonte_dias == 30)].sort_values('MAE').iloc[0].modelo == 'Media 28 días'
top10 = float(d.demanda.nlargest(10).sum())
acf7 = d.demanda.autocorr(7)
cover = ev['cobertura']

summary = f'''@chapter Resumen

La planificación de una microimportadora exige anticipar las ventas que deberá atender mientras una nueva compra se encuentra en tránsito. Esta tesina estudia el caso de Noirpress, un emprendimiento argentino que comercializa cafeteras portátiles mediante TiendaNube y Mercado Libre. Su objetivo es evaluar un procedimiento de pronóstico y relacionarlo con la reposición bajo un plazo de 30 días, consistente con la demora habitual informada para el negocio.

La fuente contiene 409 órdenes del 3 de enero al 30 de junio de 2026. Después de excluir diez operaciones de importe cero informadas como canceladas y ocho órdenes de costo bajo, la base reúne 391 pedidos. Como no se dispone de cantidades por producto, se reconstruye una serie de 179 días y 405 unidades estimadas mediante una referencia de importes calculada con observaciones anteriores a cada fecha.

Se comparan nueve métodos para el día siguiente y el total de los próximos 30 días. La mediana de 28 días obtiene el menor error de validación en ambos horizontes. En prueba, SARIMAX registra el menor MAE diario, con {fmt(sar1.MAE,2)} unidades; la media de 28 días presenta el menor MAE acumulado, con {fmt(mean30.MAE)} unidades por ventana. Las alternativas se examinan sobre un período previamente explorado, por lo que estos resultados tienen alcance retrospectivo y no constituyen una validación operativa independiente.

Los escenarios de inventario muestran que una mayor cantidad comprada sólo modifica la disponibilidad desde su recepción. En el ejercicio con emisión el 27 de abril y llegada el 27 de mayo, las tres reglas acumulan las mismas 50 unidades de faltante simulado durante la espera, aunque difieren después de la llegada.

Se concluye que la información disponible permite organizar una planificación explícita, con referencias simples competitivas y decisiones vinculadas al plazo y al inventario. El aporte es un procedimiento reproducible para el caso. Las cantidades estimadas, el stock inicial supuesto y la ausencia de una evaluación de compras reales impiden afirmar una reducción de quiebres o una política óptima.

Palabras clave: pronóstico de demanda; gestión de inventarios; microimportadora; series de tiempo; reposición; comercio electrónico.
'''

results = f'''@chapter Capítulo 4 Resultados y discusión

@section 4.1 Comportamiento de la serie depurada

La base analítica contiene 391 pedidos y 405 unidades estimadas. La diferencia entre ambas magnitudes proviene de las órdenes cuyo importe produce una estimación superior a una unidad. La depuración retira 18 registros de la fuente, sin eliminar los picos de venta de los pedidos conservados. No se detectan claves duplicadas en el conjunto utilizado.

La Tabla 2 muestra un cambio de escala durante el semestre. Enero registra 15 pedidos y concentra 17 de los 34 días sin actividad. Entre febrero y abril, las unidades medias diarias se sitúan entre 2,04 y 2,32; en mayo y junio alcanzan 3,39 y 3,03. El promedio del período completo, 2,26, resume etapas distintas y no representa por sí solo el ritmo del último tramo.

@table 2

@figure 3

La serie combina un nivel reciente más alto con aumentos breves y pronunciados. La media de los 28 días anteriores suaviza esas variaciones, pero responde con retraso cuando cambia el nivel. Esta lectura anticipa una dificultad del pronóstico: decidir cuánto del aumento reciente continuará y cuánto corresponde a un episodio transitorio.

Los días sin pedidos se distribuyen de manera desigual. Desde febrero, la actividad es más continua que en enero, aunque subsisten jornadas sin registros. El análisis conserva esos ceros dentro del calendario observado. Su interpretación comercial depende de la cobertura y disponibilidad de los canales descritas en la metodología.

@section 4.2 Canales y concentración comercial

TiendaNube aporta 214 pedidos y Mercado Libre, 177: el 54,7 % y el 45,3 % del total. La participación cambia a lo largo de la muestra. TiendaNube representa un tercio de los pedidos de febrero y aproximadamente tres cuartas partes de los de mayo. Por lo tanto, la evolución agregada combina comportamientos de canal que no son constantes.

@figure 4

La figura presenta unidades estimadas por canal, mientras que las participaciones anteriores se calculan sobre pedidos observados. La ausencia de ventas registradas en Mercado Libre después del 19 de junio limita la interpretación del final del semestre: el descenso de actividad registrado no permite identificar por sí solo una caída de demanda.

El máximo es de 22 unidades estimadas el 11 de mayo, seguido por 17 el día 12 y diez el día 13. La ventana analítica del 11 al 17 de mayo acumula 59 unidades y la del 14 al 17 de junio, 33. En conjunto, esas once fechas reúnen el 22,7 % del total del semestre. Fuera de ambas ventanas, la media diaria es 1,86, frente a 8,43 y 8,25 dentro de ellas.

@figure 5

Los diez días de mayor volumen concentran {fmt(top10,0)} unidades, el {fmt(100*top10/405)} % del total. La concentración indica que pocos días tienen un peso importante en las necesidades de cobertura. No implica que esos días deban eliminarse como errores: forman parte del comportamiento comercial que el procedimiento debe poder describir.

Las ventanas se delimitaron al examinar las ventas. No existe una serie de inversión publicitaria, promociones o disponibilidad que permita aislar la causa de los picos. En consecuencia, las diferencias describen una asociación temporal y no un efecto causal ni un multiplicador reutilizable para otra campaña.

@section 4.3 Patrón semanal y estabilidad

El perfil por día de la semana varía al comparar períodos y al excluir las ventanas comerciales. Los promedios más altos coinciden parcialmente con fechas de picos, de modo que una media global elevada no alcanza para identificar una estacionalidad estable.

@figure 6

La autocorrelación de la serie a siete días es {fmt(acf7,3)}. Este valor describe una asociación lineal débil a ese rezago; no demuestra que toda información de calendario sea inútil. La utilidad de esa entrada depende de su contribución al pronóstico, no de la presencia de una forma semanal en un gráfico.

Los modelos ensayados incluyen configuraciones con calendario o componente semanal. El estudio compara esas especificaciones completas, sin aislar mediante un experimento separado cuánto aporta cada componente. Por ello, los resultados posteriores no se atribuyen exclusivamente a la estacionalidad.

@section 4.4 Comparación de pronósticos

La Tabla 3 compara los nueve métodos con los mismos orígenes y horizontes. Cada error diario se expresa en unidades por día; cada error acumulado, en unidades por ventana de 30 días. La diferencia de escala impide comparar directamente los valores de ambas columnas como si representaran una misma tarea.

@table 3

En validación, la mediana de 28 días obtiene el menor MAE para ambos horizontes: 1,04 unidades diarias y 5,7 acumuladas. La aplicación de la regla de selección conduce, por tanto, a ese método. El comportamiento observado después es distinto: SARIMAX registra el menor error diario de prueba y la media de 28 días, el menor error acumulado.

@figure 7

En prueba diaria, SARIMAX alcanza un MAE de {fmt(sar1.MAE,2)} unidades y TimesFM, {fmt(tf1.MAE,2)}. La media de siete días registra {fmt(m11.MAE,2)}. El WAPE de SARIMAX es del {fmt(sar1.WAPE_pct)} %, por lo que incluso el mejor resultado diario conserva un error considerable respecto del volumen atendido en ese tramo.

@figure 8

La comparación de trayectorias muestra que los métodos suavizan o reaccionan de manera diferente ante los aumentos de actividad. El error agregado combina días ordinarios con episodios mucho más difíciles. No alcanza con que una curva acompañe el nivel general: para la reposición importa cuánto volumen queda subestimado durante la espera.

En el total de 30 días, la media de 28 días obtiene un MAE de {fmt(mean30.MAE)}, frente a {fmt(med30.MAE)} de la mediana, {fmt(sar30.MAE)} de SARIMAX y {fmt(tf30.MAE)} de TimesFM. Su error es un {fmt(100*(1-mean30.MAE/sar30.MAE))} % menor que el de SARIMAX en estas ventanas. La media de siete días alcanza {fmt(m130.MAE)} unidades, afectada por la prolongación de niveles recientes que incluyen picos.

@figure 9

La media de 28 días no se declara seleccionada por haber ganado la prueba. La selección definida con la validación favorece a la mediana, y el resultado de prueba permite examinar la estabilidad de esa elección. Además, las alternativas se formularon sobre un período previamente explorado. La comparación demuestra qué ocurrió en esta muestra y no acredita que el orden se mantenga en ventas futuras.

Los ajustes estadísticos informaron convergencia en todos los orígenes evaluados. Esto indica que el procedimiento de estimación alcanzó su criterio numérico; no demuestra por sí solo que la especificación represente adecuadamente la demanda. La pertinencia se juzga junto con los errores y sus condiciones de evaluación.

@section 4.5 Sesgo y concentración del error

El sesgo permite distinguir errores por exceso y por defecto. En el horizonte de 30 días, SARIMAX subestima en promedio {fmt(-sar30.sesgo_pred_menos_real)} unidades por ventana y TimesFM, {fmt(-tf30.sesgo_pred_menos_real)}. La mediana de 28 días también subestima, mientras que la media de 28 días sobreestima en promedio {fmt(mean30.sesgo_pred_menos_real)} unidades.

@figure 10

Esta diferencia importa para interpretar la cobertura. Dos métodos con un MAE cercano pueden conducir a decisiones distintas si uno tiende a quedarse corto y el otro a excederse. Sin costos de inventario y faltantes no es posible convertir esa diferencia en una preferencia económica definitiva.

Las once fechas de las ventanas comerciales concentran el {fmt(100*concentration.loc['SARIMAX','fraccion_error_ventanas'])} % del error absoluto diario de SARIMAX y el {fmt(100*concentration.loc['TimesFM 3.0','fraccion_error_ventanas'])} % del de TimesFM. Representan aproximadamente una quinta parte de los 54 días de prueba. La dificultad se concentra, por tanto, en fechas relevantes para el abastecimiento.

El intervalo diario entre los percentiles 10 y 90 de TimesFM tiene una cobertura nominal del 80 %. En la prueba incluye {cover['cubiertos']} de {cover['n']} observaciones, equivalentes al {fmt(100*cover['cobertura_observada'])} %, con un ancho medio de {fmt(cover['ancho_medio'],2)} unidades. En esta muestra, la cobertura resulta inferior a la nominal. El intervalo no se interpreta como garantía de atención de demanda ni se traslada automáticamente a un stock de seguridad.

Los resultados acumulados se apoyan en 25 ventanas de prueba superpuestas. Una misma fecha influye en numerosas ventanas, de modo que no se dispone de 25 meses independientes para evaluar estabilidad. Las diferencias se reportan como magnitudes observadas, sin presentar un contraste de superioridad general de los métodos.

@section 4.6 Saldo reconstruido y disponibilidad inicial

Con el inventario inicial supuesto de 98 unidades, las 420 recibidas durante el período y las 405 unidades estimadas vendidas, el saldo al 30 de junio es de 113. El mínimo de la trayectoria es 68, registrado el 25 de mayo, dos días antes de una recepción de 144 unidades.

@formula Saldo al 30 de junio = 98 + 420 − 405 = 113 unidades

@figure 11

El diagnóstico cambia al modificar la existencia inicial. Sin unidades al inicio, el saldo contable alcanza un mínimo de −30; con 25 unidades, llega a −5. Se requieren al menos 30 unidades iniciales para que todos los cierres diarios de esta reconstrucción sean no negativos.

@table 4

Un saldo negativo identifica una incompatibilidad entre la existencia supuesta y los movimientos registrados. Puede relacionarse con ventas previas, recepciones omitidas, errores de estimación u otros movimientos. No demuestra por sí solo un faltante físico ni cuantifica compras que no pudieron concretarse.

Del mismo modo, que la trayectoria con 98 unidades se mantenga positiva no acredita la ausencia de quiebres reales. La agregación puede ocultar diferencias por color, ubicación o canal, y no informa la disponibilidad intradía. Esta reconstrucción sirve para evaluar la sensibilidad de una interpretación, no para reemplazar el inventario físico.

@section 4.7 Reposición con fechas de llegada explícitas

El primer escenario emite la compra al cierre del 27 de abril y la recibe al comienzo del 27 de mayo. Parte de 50 unidades y mantiene constantes esos datos para las tres reglas. La media de siete días solicita 130 unidades; la media de 28 días, 85; y la mediana de 28 días, 70.

@table 5

Las tres reglas acumulan 50 unidades de faltante simulado antes de la llegada. Después de recibir la compra, los faltantes son cero, ocho y 23, respectivamente. La mayor cantidad mejora la atención posterior, pero también eleva el inventario medio. El ejemplo muestra el intercambio entre cobertura y existencias sin asignarle un costo monetario.

@figure 12

En el segundo escenario, emitido el 1 de junio con 60 unidades iniciales, la llegada ocurre el 1 de julio. Las reglas solicitan 18, 155 y 120 unidades. Hasta el 30 de junio, las tres alternativas tienen la misma trayectoria y acumulan 30 unidades de faltante simulado: ninguna compra ha ingresado todavía. No se evalúa su efecto posterior porque no existen ventas observadas de julio.

La lectura operativa es que una compra emitida después de un aumento de ventas puede llegar cuando la necesidad ya ocurrió. Con una demora fija de 30 días, disponer de producto el 11 de mayo exigiría emitir la compra el 11 de abril, bajo la convención de fechas utilizada. La cantidad suficiente y el inventario durante la espera continúan siendo necesarios.

Estos ejercicios excluyen otras recepciones para estudiar una decisión aislada. Sus faltantes pertenecen a escenarios hipotéticos y no se presentan como pérdidas reales de Noirpress. Una política de compras sucesivas también debería incorporar pedidos en tránsito, compromisos, lotes mínimos y costos.

@section 4.8 Síntesis de los resultados

Los resultados responden a las tres hipótesis dentro del alcance del caso. H1 recibe respaldo porque varios métodos superan a la media de siete días en ambos horizontes de prueba. H2 encuentra apoyo en la competitividad de las referencias de 28 días. H3 se verifica por la mecánica temporal de los escenarios: la cantidad pedida actúa desde su llegada.

@table 6

La comparación permite separar tres decisiones que suelen confundirse: cómo representar las ventas, qué horizonte pronosticar y cuándo reponer. Un menor error diario no garantiza el mejor resultado acumulado; un menor error acumulado no determina una cantidad óptima; y una mayor cantidad no compensa la falta de disponibilidad antes de su recepción. El valor aplicado reside en analizar esas relaciones sobre una misma base de datos y bajo condiciones explícitas.
'''

conclusion = f'''@chapter Capítulo 5 Conclusiones

@section 5.1 Respuesta y aporte del estudio

El caso Noirpress muestra que los registros de una microimportadora local pueden organizarse en un procedimiento de pronóstico útil para examinar decisiones de reposición. La investigación integra depuración, reconstrucción de cantidades, evaluación temporal y simulación de compras. Su aporte es hacer verificable la relación entre los datos disponibles y la decisión que se pretende apoyar.

La comparación no identifica un método superior para todas las tareas. La mediana de 28 días obtiene el menor MAE de validación en ambos horizontes, mientras que la prueba favorece a SARIMAX para el día siguiente y a la media de 28 días para el total acumulado. El resultado respalda la inclusión de referencias simples y obliga a justificar la complejidad con evidencia específica del horizonte.

Para Noirpress, el horizonte de 30 días se vincula directamente con la espera habitual de una compra. Sin embargo, el pronóstico adquiere sentido operativo cuando se combina con existencias, unidades en tránsito y compromisos de entrega. Los escenarios muestran que la oportunidad de llegada condiciona qué ventas puede atender la compra, independientemente de que la cantidad solicitada sea mayor.

@section 5.2 Implicaciones y límites

La propuesta es utilizar una referencia de ventas acumuladas como parte de una revisión de reposición, junto con el calendario comercial y la posición de inventario. La media y la mediana de 28 días ofrecen reglas transparentes para ese seguimiento. Sus resultados retrospectivos no autorizan a automatizar compras sin observar el desempeño futuro y las restricciones reales del negocio.

El alcance se limita a 179 días y a cantidades reconstruidas por importe. La disponibilidad inicial no proviene de un conteo físico y el plazo de 30 días se fija a partir de la demora habitual informada. Tampoco se dispone de evidencia suficiente para separar ausencia de demanda y falta de disponibilidad en los días sin ventas de Mercado Libre.

La evaluación utiliza un período previamente conocido y ventanas acumuladas superpuestas. Por ello, la estabilidad del ranking y la eficacia de una política aplicada no quedan demostradas. Los escenarios permiten comprender consecuencias de decisiones bajo condiciones definidas; no acreditan reducción de quiebres reales, ventas recuperadas ni rentabilidad adicional.

@section 5.3 Cierre

Noirpress dispone así de una base depurada de 391 pedidos y 405 unidades estimadas, una comparación reproducible de pronósticos y escenarios que vinculan cantidad, inventario y fecha de llegada. El estudio cumple su objetivo al convertir los registros disponibles en una herramienta para revisar la reposición, con resultados proporcionales a la evidencia del caso. La mejora propuesta consiste en planificar con supuestos explícitos y decisiones verificables, reconociendo la incertidumbre tanto de las ventas como del abastecimiento.
'''

concept = (WORK / 'fragmento_conceptual_final.txt').read_text(encoding='utf-8')
concept, annex = concept.split('@chapter Anexo técnico Reproducibilidad y métricas')
concept = concept.replace('El stock de seguridad procura absorber parte de la incertidumbre.', '@figure 1\n\nEl stock de seguridad procura absorber parte de la incertidumbre.')
concept = concept.replace('El origen avanza diariamente y los métodos reciben la misma historia de ventas.', '@table 1\n\n@figure 2\n\nEl origen avanza diariamente y los métodos reciben la misma historia de ventas.')
concept = concept.replace('ocho órdenes de costo de producto bajo conforme al criterio adoptado para el estudio.', 'ocho órdenes con costo de producto inferior a $20.000, conforme al criterio adoptado para el estudio. En esos ocho casos, el costo registrado está entre $7.999 y $10.000.')
annex = '@chapter Anexo A Reproducibilidad y métricas' + annex
start = annex.index('@formula MAE')
end = annex.index('El WAPE sólo')
annex = annex[:start] + '@table 7\n\n' + annex[end:]
annex = annex.replace('El expediente técnico incluye las predicciones por origen,',
    'La base depurada se conserva en resultados_noirpress/cierre/ y se reproduce con scripts/cierre_noirpress.py. La evaluación se ejecuta mediante scripts/evaluacion_final_noirpress.py y guarda sus salidas en resultados_noirpress/evaluacion_final/. El manifiesto resumen_final.json registra las versiones y las huellas de entrada.\n\nEl expediente técnico incluye las predicciones por origen,')
annex = annex.replace('@section A.2 Definición de las métricas',
    f'La ejecución utiliza Python {ev["python"]}, NumPy {ev["paquetes"]["numpy"]}, pandas {ev["paquetes"]["pandas"]}, SciPy {ev["paquetes"]["scipy"]}, statsmodels {ev["paquetes"]["statsmodels"]}, timesfm {ev["paquetes"]["timesfm"]} y PyTorch {ev["paquetes"]["torch"]}. TimesFM se ejecuta en CPU con la revisión fija {ev["revision"]}, cargada de la caché local.\n\n@section A.2 Definición de las métricas')
text = '\n\n'.join([summary.strip(),concept.strip(),results.strip(),conclusion.strip(),'@bibliography',annex.strip()]) + '\n'
assert text.count('@figure ') == 12 and text.count('@table ') == 7
assert '413' not in text and 'Abstract' not in text
(WORK / 'texto_definitivo.txt').write_text(text,encoding='utf-8')
(WORK / 'tablas_definitivas.json').write_text(json.dumps(tables,ensure_ascii=False,indent=2),encoding='utf-8')
ids = ['Bergmeir','Box','CACE','Chopra','Corsten','Croston','Das','Eppen','Fildes','Gardner','GooglePesos','GoogleRepo','Holt','HyndmanLibro','HyndmanMetricas','Jain','Makridakis2018','Makridakis2020','Makridakis2022','Nahmias1994','NahmiasLibro','DatosStock','DatosOrdenes','Silver','statsmodels','SyntetosBoylan','SyntetosCroston','Tashman','Winters']
(WORK / 'referencias_usadas.json').write_text(json.dumps(ids,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Texto final: {len(text.split())} palabras, {len(tables)} tablas, 12 figuras y {len(ids)} referencias.')
