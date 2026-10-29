# Noirpress: pronóstico de demanda y reposición

Material metodológico del trabajo final **Pronóstico de demanda y reposición en una microimportadora local. El caso de Noirpress**, de Jose Pasini y Mariano Martín. Octubre de 2026.

El repositorio reúne los datos, el código y los resultados de una evaluación retrospectiva de nueve métodos de pronóstico, con horizontes de uno y 30 días, y seis escenarios de reposición.

## Contenido

| Ruta | Contenido |
|---|---|
| `fuentes/` | Dos planillas originales, conservadas sin modificar. |
| `Ordenes_Noirpress_limpio.xlsx` | Entrada intermedia necesaria para reproducir la depuración final. |
| `scripts/` | Procesamiento, comparación de pronósticos y generación de figuras. |
| `resultados_noirpress/cierre/` | Órdenes retenidas y excluidas, serie diaria y criterios de depuración. |
| `resultados_noirpress/evaluacion_final/` | Predicciones guardadas, métricas, escenarios de inventario y figuras. |

## Verificar el respaldo sin ejecutar modelos

Desde la raíz del repositorio, con Python 3.12:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-respaldo.txt
.venv/Scripts/python.exe scripts/verificar_resultados.py
```

La verificación contrasta las huellas SHA-256, las planillas originales, la depuración, las métricas recalculadas a partir de predicciones guardadas y seis escenarios de inventario. La base final contiene 391 pedidos, 405 unidades estimadas y 179 días. No necesita descargar TimesFM ni volver a ejecutar los modelos. Los comandos indicados se ejecutan en Windows con Python 3.12.

## Recalcular el análisis

Para repetir el cálculo completo, instalar también `requirements.txt` en un entorno separado. Las versiones de la ejecución original y las huellas de entrada figuran en `resultados_noirpress/evaluacion_final/resumen_final.json`.

```powershell
python scripts/cierre_noirpress.py
python scripts/evaluacion_final_noirpress.py
python scripts/figuras_noirpress_final.py
```

Estos comandos reemplazan las salidas de cálculo. La evaluación requiere tener previamente en la caché de Hugging Face los pesos de `google/timesfm-3.0-pytorch`, revisión `43046b85ec22d584a13f8098c2ed39c889e129c2`: el código los carga sin conexión. Los pesos no se incluyen en este repositorio y tienen condiciones de uso no comercial. Consultar su licencia antes de utilizarlos.

Los módulos `limpiar_noirpress.py`, `auditoria_eda_noirpress.py` y `modelos_noirpress_avanzados.py` contienen únicamente las funciones auxiliares que utiliza el procedimiento final; no son comandos de ejecución independientes. El XLSX intermedio se conserva como entrada de la depuración.

`resumen_final.json` conserva el manifiesto de la ejecución original. `distribucion_codigo.json` registra por separado los hashes del código distribuido después de retirar los flujos e informes históricos no utilizados. Los cuerpos de las funciones de cálculo se conservaron sin cambios, comprobados mediante comparación de su estructura sintáctica (AST). Esta limpieza no equivale a una nueva ejecución de los modelos ni modifica las predicciones guardadas. El commit original queda identificado para consultar la evidencia anterior.

## Alcance y acceso

Este repositorio es público y contiene importes comerciales, costos e identificadores de órdenes de Noirpress. No se incorporaron credenciales ni pesos de modelos. No se detectaron columnas con nombres, correos, teléfonos o domicilios de clientes en las planillas revisadas. Las planillas originales se conservan sin modificar para permitir la comprobación de las huellas y los resultados.

Los resultados son retrospectivos: las unidades vendidas se estiman a partir de importes, el stock inicial de 98 unidades es supuesto y los escenarios de reposición son hipotéticos. No demuestran una reducción de quiebres reales. Los stocks de 50 y 60 unidades pertenecen a las simulaciones. El escenario completo contiene 29 días antes y 31 días desde la recepción.

No se concede una licencia abierta sobre los datos comerciales ni los recursos de terceros incluidos.
