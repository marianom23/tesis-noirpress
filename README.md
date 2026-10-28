# Tesina Noirpress

Respaldo del trabajo final **Pronóstico de demanda y reposición en una microimportadora local. El caso de Noirpress**, de Jose Pasini y Mariano Martín. Octubre de 2026.

El documento de entrega es [Tesina Noirpress - version final.pdf](Tesina%20Noirpress%20-%20version%20final.pdf), de 30 páginas.

## Contenido

| Ruta | Contenido |
|---|---|
| `fuentes/` | Dos planillas originales, conservadas sin modificar. |
| `Ordenes_Noirpress_limpio.xlsx` | Entrada intermedia necesaria para reproducir la depuración final. |
| `scripts/` | Procesamiento, comparación de pronósticos y generación de figuras. |
| `resultados_noirpress/cierre/` | Órdenes retenidas y excluidas, serie diaria y criterios de depuración. |
| `resultados_noirpress/evaluacion_final/` | Predicciones guardadas, métricas, escenarios de inventario y figuras. |
| `documento/` | Texto, tablas, bibliografía, correcciones, composición y verificaciones. |

Se excluyen las versiones anteriores del PDF, la guía de defensa anterior, el plan previo, las cachés, los entornos y los temporales. Los archivos de trabajo locales permanecen en la computadora.

## Verificar el respaldo sin ejecutar modelos

Desde la raíz del repositorio, con Python 3.12:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-respaldo.txt
.venv/Scripts/python.exe documento/verificar_consistencia_ugr.py
```

La verificación contrasta las huellas SHA-256, las planillas originales, la depuración, las métricas recalculadas a partir de predicciones guardadas, las tablas y seis escenarios de inventario. La base final contiene 391 pedidos, 405 unidades estimadas y 179 días. No necesita descargar TimesFM ni volver a ejecutar los modelos.

## Reconstruir el PDF

```powershell
.venv/Scripts/python.exe documento/adaptar_plantilla_ugr.py
```

La composición utiliza las fuentes Arial, Georgia y Segoe UI instaladas en Windows, en `C:/Windows/Fonts`. El comando reemplaza el PDF de entrega con una nueva composición del texto y los resultados guardados. No se distribuyen las fuentes tipográficas.

El texto vigente se compone de `texto_definitivo.txt`, `revisiones_redaccion_ugr.json` y `correcciones_consistencia_ugr.json`. No editar únicamente `texto_plantilla_ugr.txt`: se regenera. `armar_definitiva.py` y `build_definitiva.py` son etapas anteriores; el comando de entrega es `adaptar_plantilla_ugr.py`.

## Recalcular el análisis

Para repetir el cálculo completo, instalar también `requirements.txt` en un entorno separado. Las versiones de la ejecución original y las huellas de entrada figuran en `resultados_noirpress/evaluacion_final/resumen_final.json`.

```powershell
python scripts/cierre_noirpress.py
python scripts/evaluacion_final_noirpress.py
python scripts/figuras_noirpress_final.py
```

Estos comandos reemplazan las salidas de cálculo. La evaluación requiere tener previamente en la caché de Hugging Face los pesos de `google/timesfm-3.0-pytorch`, revisión `43046b85ec22d584a13f8098c2ed39c889e129c2`: el código los carga sin conexión. Los pesos no se incluyen en este repositorio y tienen condiciones de uso no comercial. Consultar su licencia antes de utilizarlos.

Los módulos `limpiar_noirpress.py`, `auditoria_eda_noirpress.py` y `modelos_noirpress_avanzados.py` se conservan porque la evaluación final importa funciones de ellos. Sus flujos principales corresponden a etapas anteriores: no ejecutarlos como sustitutos de los tres comandos anteriores. El XLSX intermedio se conserva para evitar regenerarlo con criterios históricos.

## Alcance y acceso

Este repositorio es público y contiene importes comerciales, costos e identificadores de órdenes de Noirpress. No se incorporaron credenciales ni pesos de modelos. No se detectaron columnas con nombres, correos, teléfonos o domicilios de clientes en las planillas revisadas. Las planillas originales se conservan sin modificar para permitir la comprobación de las huellas y los resultados.

Los resultados son retrospectivos: las unidades vendidas se estiman a partir de importes, el stock inicial de 98 unidades es supuesto y los escenarios de reposición son hipotéticos. No demuestran una reducción de quiebres reales. Los stocks de 50 y 60 unidades pertenecen a las simulaciones. El escenario completo contiene 29 días antes y 31 días desde la recepción.

La revisión bibliográfica se conserva en `documento/Revision de fuentes y citas - 2026-10-28.txt`; detalla qué se cotejó y qué quedó sin acceso al texto completo. El ejemplo de formato UGR no sustituye un reglamento institucional.

No se concede una licencia abierta sobre los datos comerciales, el documento ni las marcas y recursos de terceros incluidos.
