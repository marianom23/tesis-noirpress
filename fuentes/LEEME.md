# Fuentes crudas — NO MODIFICAR

Los archivos de esta carpeta son los **datos originales tal como se recibieron**. Se conservan sin
alterar para que todo el análisis de la tesis sea reproducible y auditable desde el origen.

No editarlos, no renombrarlos, no "arreglarlos". Toda depuración se hace en Python y se escribe en
archivos nuevos en la carpeta raíz del proyecto.

| Archivo | Origen | Qué alimenta |
|---|---|---|
| `control_pedidos_ventas_quiebres_noirpress.xlsx` | Planilla manual del microimportador (Noirpress). | Ingresos de stock. No contiene ventas reales (la hoja Ventas Mensuales es una plantilla vacía). |
| `ultimas_ordenes_2026-07-06.xlsx` | Export de órdenes de TiendaNube + Mercado Libre, 409 filas, ene–jun 2026. | Única fuente de ventas reales de Noirpress. |

## Derivados (en la carpeta raíz, generados a partir de esto)

- `Ordenes_Noirpress_limpio.xlsx` — archivo de órdenes de Noirpress depurado

La depuración final se reproduce con `scripts/cierre_noirpress.py` y conserva sus
salidas en `resultados_noirpress/cierre/`. Los nueve métodos se recalcularon sobre
esa base en `resultados_noirpress/evaluacion_final/`.

`Plan de Tesis UGR - Final corregido.docx` se conserva como antecedente institucional.
El documento vigente es `Tesina Noirpress - version final.pdf`, en la carpeta raíz.
