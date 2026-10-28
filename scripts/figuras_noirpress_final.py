"""Figuras de la evaluación final: 391 pedidos y 405 unidades estimadas."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "resultados_noirpress"
E = R / "evaluacion_final"
OUT = E / "figuras"
OUT.mkdir(parents=True, exist_ok=True)
INK, ACCENT, BLUE, GREEN = "#243A40", "#8C6848", "#547B91", "#638574"
MUTED, LIGHT, GRID = "#6A7476", "#B9C8CE", "#E5E9E7"
MC = {"SARIMAX": ACCENT, "TimesFM 3.0": GREEN, "Holt-Winters": "#AD9478",
      "M1 histórico": MUTED, "Media 28 días": BLUE, "Mediana 28 días": "#54776B",
      "Media histórica": "#9C9F92", "Ingenuo último día": "#9FBDC5", "Ingenuo semanal": "#C4B29D"}
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.edgecolor": "#9AA6A7", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white", "savefig.dpi": 220,
    "lines.linewidth": 1.9, "legend.frameon": False,
})
CATALOG = []


def num(v, digits=1):
    return f"{v:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def style(ax, dates=False, grid="y"):
    ax.grid(axis=grid, color=GRID, lw=.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: num(v, 0) if v == int(v) else num(v)))
    if dates:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    else:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: num(v, 0) if v == int(v) else num(v)))


def months(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: MESES[mdates.num2date(v).month-1]))


def save(fig, basename, titulo, nota):
    fig.savefig(OUT / basename, bbox_inches="tight", pad_inches=.1)
    plt.close(fig)
    CATALOG.append(dict(number=int(basename[:2]), basename=Path(basename).stem, titulo=titulo, nota=nota))
    print(basename, flush=True)


def data():
    d = pd.read_csv(R / "cierre/serie_diaria_cierre.csv", parse_dates=["Fecha"])
    o = pd.read_csv(R / "cierre/ordenes_cierre.csv", parse_dates=["Fecha"])
    assert len(d) == 179 and d.pedidos.sum() == 391 and d.unidades_estimadas.sum() == 405
    assert len(o) == 391 and o["Unidades estimadas cierre"].sum() == 405
    return d, o


def descriptive():
    d, o = data()
    u = d.unidades_estimadas
    # Esquema conceptual: sus valores no son observaciones del caso.
    fig, ax = plt.subplots(figsize=(7.8, 3.5))
    x = [0,30,60,60,100]
    y = [200,110,20,200,80]
    ax.plot(x, y, color=INK, lw=2.4)
    ax.axhline(110, color=ACCENT, ls="--", label="Punto de reposición: 3 × 30 + 20 = 110")
    ax.axhline(20, color=GREEN, ls=":", label="Stock de seguridad: 20")
    ax.annotate("", xy=(60, 228), xytext=(30, 228), arrowprops={"arrowstyle": "<->", "color": MUTED})
    ax.text(45, 238, "Plazo de entrega: 30 días", ha="center", fontsize=9)
    ax.plot([30, 30], [110, 228], color=MUTED, ls=":", lw=1)
    ax.plot([60, 60], [200, 228], color=MUTED, ls=":", lw=1)
    ax.annotate("Emisión", xy=(30, 110), xytext=(17, 65), arrowprops={"arrowstyle": "->", "color": MUTED})
    ax.annotate("Recepción", xy=(60, 200), xytext=(72, 220), arrowprops={"arrowstyle": "->", "color": MUTED})
    ax.set(xlim=(0, 100), ylim=(0, 260), xlabel="Días", ylabel="Unidades en inventario")
    style(ax); ax.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=1)
    save(fig, "01_punto_reposicion.png", "Punto de reposición y plazo de entrega",
         "Ejemplo conceptual con demanda constante de 3 unidades diarias, plazo de 30 días y reserva de 20. No representa el stock observado de Noirpress.")

    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    start, end = pd.Timestamp("2026-01-03"), pd.Timestamp("2026-06-30")
    for a, b, label, color in [(start, pd.Timestamp("2026-03-13"), "Historial inicial\n70 días", "#E0E7E7"),
        (pd.Timestamp("2026-03-14"), pd.Timestamp("2026-05-07"), "Validación\n55 días", "#DCE9E1"),
        (pd.Timestamp("2026-05-08"), end, "Prueba\n54 días", "#ECDFD2")]:
        ax.add_patch(Rectangle((mdates.date2num(a)-.5, 4), (b-a).days+1, .8, color=color, lw=0))
        ax.text(a+(b-a)/2, 4.4, label, ha="center", va="center", fontsize=9)
    for i, origin in enumerate(pd.to_datetime(["2026-03-13", "2026-04-07", "2026-05-07", "2026-05-31"])):
        y = 3.35 - i*.7
        ax.plot([start, origin], [y,y], color=INK, lw=5)
        ax.plot([origin+pd.Timedelta(days=1), origin+pd.Timedelta(days=30)], [y,y], color=ACCENT, lw=5, alpha=.55)
        ax.scatter([origin+pd.Timedelta(days=1)], [y], color=ACCENT, s=30, zorder=3)
        ax.text(start-pd.Timedelta(days=2), y, origin.strftime("%d/%m"), ha="right", va="center", fontsize=9)
    ax.plot([], [], lw=5, color=INK, label="Historia disponible")
    ax.plot([], [], lw=5, color=ACCENT, alpha=.55, label="30 días por anticipar")
    ax.scatter([], [], color=ACCENT, s=30, label="Día siguiente")
    ax.set(xlim=(start-pd.Timedelta(days=1), end+pd.Timedelta(days=1)), ylim=(.7,5), yticks=[])
    ax.spines["left"].set_visible(False); months(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(.5,-.22), ncol=3)
    save(fig, "02_evaluacion_temporal.png", "Evaluación con origen móvil",
         "Cada origen utiliza solamente observaciones previas. Los errores a 30 días se calculan sobre ventanas completas: 26 en validación y 25 en prueba; las ventanas se superponen.")

    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    ax.bar(d.Fecha, u, width=.85, color=LIGHT, label="Unidades estimadas del día")
    ax.plot(d.Fecha, u.rolling(28, min_periods=28).mean(), color=INK, lw=2.4, label="Media móvil de 28 días")
    for start, end, label in [("2026-05-11", "2026-05-17", "Ventana de mayo"), ("2026-06-14", "2026-06-17", "Ventana de junio")]:
        ax.axvspan(pd.Timestamp(start)-pd.Timedelta(hours=12), pd.Timestamp(end)+pd.Timedelta(hours=12), color=ACCENT, alpha=.12)
    peak = d.loc[u.idxmax()]
    ax.annotate(f"{peak.Fecha:%d/%m}: {int(peak.unidades_estimadas)}", xy=(peak.Fecha, peak.unidades_estimadas),
                xytext=(pd.Timestamp("2026-04-01"), 19), arrowprops={"arrowstyle":"->", "color": MUTED}, fontsize=9)
    ax.set(ylabel="Unidades estimadas por día", ylim=(0,u.max()+2))
    style(ax, dates=True); months(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.12), ncol=2)
    save(fig, "03_serie_diaria.png", "Serie diaria y evolución del nivel de ventas",
         "Base depurada: 391 pedidos, 405 unidades estimadas y 179 días. Sombreado: ventanas analíticas del 11 al 17 de mayo y del 14 al 17 de junio; no acredita causalidad comercial ni duración oficial de los eventos.")

    p = o.pivot_table(index=o.Fecha.dt.month, columns="Origen", values="Unidades estimadas cierre", aggfunc="sum", fill_value=0).reindex(range(1,7), fill_value=0)
    fig, ax = plt.subplots(figsize=(7.8,3.3))
    bottom = np.zeros(6)
    for ch, color in [("TiendaNube",BLUE),("Mercado Libre",ACCENT)]:
        vals = p[ch].to_numpy()
        ax.bar(np.arange(6), vals, bottom=bottom, color=color, width=.6, label=ch)
        for i,v in enumerate(vals):
            if v: ax.text(i,bottom[i]+v/2,str(int(v)),ha="center",va="center",color="white",fontsize=9)
        bottom += vals
    for i,v in enumerate(bottom): ax.text(i,v+2,str(int(v)),ha="center",fontsize=10,fontweight="bold")
    style(ax); ax.set_xticks(range(6), MESES[:6]); ax.set_ylabel("Unidades estimadas por mes")
    ax.set_ylim(0,max(bottom)+15); ax.legend(ncol=2,loc="upper left")
    save(fig, "04_canales_mensuales.png", "Composición mensual de las ventas por canal",
         "Las cantidades se estiman a partir del importe de cada pedido y una referencia de precios calculada con días anteriores. Enero comienza el día 3; la ausencia de pedidos de Mercado Libre después del 19 de junio no demuestra falta de stock.")

    fig, axs = plt.subplots(1,2,figsize=(7.8,3.2),gridspec_kw={"width_ratios":[1,1.05]})
    counts = pd.cut(u,[-1,0,1,2,4,9,999],labels=["0","1","2","3–4","5–9","10 o más"]).value_counts(sort=False)
    axs[0].bar(range(len(counts)),counts.values,color=BLUE,width=.65)
    for i,v in enumerate(counts): axs[0].text(i,v+1,str(v),ha="center",fontsize=9)
    style(axs[0]); axs[0].set_xticks(range(len(counts)),counts.index,fontsize=9)
    axs[0].set(ylabel="Cantidad de días",xlabel="Unidades estimadas en el día",ylim=(0,max(counts)+8))
    ranked=np.sort(u.to_numpy())[::-1]; pct=100*np.cumsum(ranked)/ranked.sum()
    axs[1].plot(np.arange(1,len(u)+1),pct,color=ACCENT,lw=2.4)
    for n in [10,20]:
        axs[1].scatter(n,pct[n-1],color=INK,s=24)
        axs[1].annotate(f"{n} días: {num(pct[n-1])} %",(n,pct[n-1]),xytext=(n+22,pct[n-1]-3),fontsize=9,
                        arrowprops={"arrowstyle":"-","color":MUTED})
    style(axs[1]); axs[1].set(xlabel="Días ordenados de mayor a menor venta",ylabel="Porcentaje acumulado de unidades",ylim=(0,105))
    fig.tight_layout(w_pad=2)
    save(fig, "05_distribucion_concentracion.png", "Distribución y concentración de las ventas diarias",
         "El panel derecho ordena los días por volumen, sin respetar la secuencia temporal. Describe concentración, no probabilidad futura de demanda.")

    fig, ax = plt.subplots(figsize=(7.8,3.2))
    daynames=["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
    for mask,label,color,ls in [(d.Fecha.lt("2026-05-01"),"Enero a abril",BLUE,"--"),(d.Fecha.ge("2026-05-01"),"Mayo y junio",ACCENT,"-")]:
        s=d.loc[mask].groupby(d.loc[mask,"Fecha"].dt.dayofweek).unidades_estimadas.mean().reindex(range(7))
        ax.plot(range(7),s,marker="o",ms=5,color=color,ls=ls,label=label)
    style(ax); ax.set_xticks(range(7),daynames); ax.set(ylabel="Promedio de unidades estimadas",ylim=(0,None)); ax.legend(ncol=2)
    save(fig, "06_perfil_semanal.png", "El perfil semanal cambia entre tramos del semestre",
         "Promedios descriptivos de la base depurada. El tramo de mayo y junio incluye las ventanas de campaña; la diferencia no demuestra una estacionalidad semanal estable.")

    receipts=pd.read_csv(E/"serie_diaria.csv",parse_dates=["fecha"])[["fecha","ingresos_unidades"]]
    stock=d.merge(receipts,left_on="Fecha",right_on="fecha",validate="one_to_one")
    assert stock.ingresos_unidades.sum()==420
    flow=(stock.ingresos_unidades-stock.unidades_estimadas).cumsum()
    fig,ax=plt.subplots(figsize=(7.8,3.5))
    for s0,color,ls in [(98,INK,"-"),(32,GREEN,"--"),(0,ACCENT,":")]:
        ax.plot(stock.Fecha,s0+flow,color=color,ls=ls,label=f"Inicial supuesto: {s0}")
    ax.axhline(0,color=MUTED,lw=1); ax.fill_between(stock.Fecha,flow,0,where=flow.lt(0),color=ACCENT,alpha=.1)
    style(ax,dates=True);months(ax);ax.set_ylabel("Saldo teórico de unidades")
    ax.legend(ncol=3,loc="upper center",bbox_to_anchor=(.5,1.16),fontsize=9)
    save(fig,"11_saldo_reconstruido.png","Saldo de inventario bajo tres supuestos iniciales",
         "Se combinan 420 unidades recibidas durante el período con 405 unidades de venta estimadas. Los saldos negativos representan incompatibilidad del supuesto con el flujo registrado, no faltantes comprobados. No existe conteo físico inicial.")


def forecasts():
    metrics=pd.read_csv(E/"comparacion_modelos.csv")
    pred=pd.read_csv(E/"pronosticos_por_origen.csv",parse_dates=["fecha","origen"])
    totals=pd.read_csv(E/"demanda_acumulada_30_dias.csv",parse_dates=["origen"])
    daily, _ = data()
    assert len(metrics)==36 and metrics.modelo.nunique()==9
    assert pred.groupby(["etapa","modelo"]).apply(lambda g: (g.horizonte==1).sum(),include_groups=False).isin([54,55]).all()
    names=metrics[(metrics.etapa=="prueba") & (metrics.horizonte_dias==30)].sort_values("MAE").modelo.tolist()
    fig,axs=plt.subplots(1,2,figsize=(8,4.4),sharey=True)
    for ax,horizon,title in zip(axs,[1,30],["Día siguiente","Total de 30 días"]):
        m=metrics[metrics.horizonte_dias==horizon].pivot(index="modelo",columns="etapa",values="MAE").loc[names]
        for i,(_,row) in enumerate(m.iterrows()):
            ax.plot([row.validacion,row.prueba],[i,i],color=LIGHT,lw=2)
        ax.scatter(m.validacion,range(9),color=GREEN,s=40,label="Validación",zorder=3)
        ax.scatter(m.prueba,range(9),color=ACCENT,s=40,marker="s",label="Prueba",zorder=3)
        ax.set_title(title); ax.set_xlabel("Error absoluto medio (unidades)")
        style(ax,grid="x"); ax.set_xlim(left=0)
    axs[0].set_yticks(range(9),names);axs[0].invert_yaxis()
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc="lower center",bbox_to_anchor=(.63,-.02))
    fig.tight_layout(w_pad=2,rect=(0,.06,1,1))
    save(fig,"07_comparacion_mae.png","Error en validación y prueba para los nueve métodos",
         "Menor MAE indica mayor precisión. Orden: error de prueba a 30 días. Las cinco referencias adicionales y esta reevaluación se realizan con conocimiento previo del período, por lo que la comparación es retrospectiva y exploratoria.")

    one=pred[(pred.etapa=="prueba") & (pred.horizonte==1)]
    real=daily[daily.Fecha.ge("2026-05-08")]
    fig,axs=plt.subplots(2,2,figsize=(7.8,4.8),sharex=True,sharey=True)
    for ax,name in zip(axs.flat,["SARIMAX","TimesFM 3.0","Media 28 días","Mediana 28 días"]):
        p=one[one.modelo==name].sort_values("fecha")
        assert len(p)==54
        ax.bar(real.Fecha,real.unidades_estimadas,color=LIGHT,alpha=.7,width=.85)
        ax.plot(p.fecha,p.pronostico,color=MC[name],lw=1.9)
        mae=metrics[(metrics.modelo==name)&(metrics.etapa=="prueba")&(metrics.horizonte_dias==1)].MAE.iloc[0]
        ax.set_title(f"{name} · MAE {num(mae,2)}",fontsize=10)
        style(ax,dates=True);ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1,15]))
        ax.set_ylim(0,real.unidades_estimadas.max()+1)
    axs[0,0].set_ylabel("Unidades estimadas");axs[1,0].set_ylabel("Unidades estimadas")
    fig.supxlabel("Barras: serie depurada. Línea: pronóstico emitido el día anterior.",fontsize=9)
    fig.tight_layout(h_pad=1.5,w_pad=1.5)
    save(fig,"08_pronosticos_diarios.png","Pronósticos diarios y picos de la serie de prueba",
         "Se muestran dos modelos complejos y dos referencias simples para comparar su reacción a los picos. Cada curva corresponde a 54 pronósticos de un día; las barras son cantidades inferidas de los pedidos.")

    test=totals[totals.etapa=="prueba"]
    observed=test[test.modelo=="SARIMAX"].sort_values("origen")
    assert len(observed)==25
    fig,ax=plt.subplots(figsize=(7.8,3.6))
    ax.plot(observed.origen,observed.real,color=INK,lw=2.8,label="Total de la serie depurada")
    for name,ls in [("SARIMAX","-"),("TimesFM 3.0","-."),("Media 28 días","--")]:
        p=test[test.modelo==name].sort_values("origen")
        assert len(p)==25
        ax.plot(p.origen,p.pronostico,color=MC[name],ls=ls,label=name)
    style(ax,dates=True);ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.set(ylabel="Total de 30 días (unidades estimadas)",xlabel="Fecha de emisión del pronóstico")
    ax.legend(ncol=2,loc="upper center",bbox_to_anchor=(.5,-.23),fontsize=9)
    save(fig,"09_totales_30_dias.png","Pronóstico acumulado a 30 días desde 25 orígenes de prueba",
         "Cada punto compara el pronóstico con el total posterior de la serie depurada. Las ventanas se superponen: no son 25 meses independientes. Se muestran tres métodos para facilitar la lectura; la tabla de resultados incluye los nueve.")

    m=metrics[metrics.horizonte_dias==30].pivot(index="modelo",columns="etapa",values="sesgo_pred_menos_real").loc[names]
    fig,ax=plt.subplots(figsize=(7.8,4.2))
    for i,(_,row) in enumerate(m.iterrows()):
        ax.plot([row.validacion,row.prueba],[i,i],color=LIGHT,lw=2)
    ax.scatter(m.validacion,range(9),color=GREEN,s=40,label="Validación",zorder=3)
    ax.scatter(m.prueba,range(9),color=ACCENT,s=40,marker="s",label="Prueba",zorder=3)
    ax.axvline(0,color=INK,lw=1);style(ax,grid="x")
    ax.set_yticks(range(9),names);ax.invert_yaxis()
    ax.set_xlabel("Sesgo medio: pronóstico menos serie depurada (unidades)")
    ax.legend(ncol=2,loc="upper center",bbox_to_anchor=(.55,-.15))
    save(fig,"10_sesgo_30_dias.png","Dirección del error en los totales de 30 días",
         "Un sesgo negativo indica subestimación; un valor positivo indica sobreestimación. El sesgo puede compensar errores de signo contrario y debe interpretarse junto con el MAE.")

    trajectories=pd.read_csv(E/"reposicion_dia_a_dia.csv",parse_dates=["fecha","emision","llegada"])
    summary=pd.read_csv(E/"reposicion_resumen.csv",parse_dates=["Emisión","Llegada"])
    fig,axs=plt.subplots(2,1,figsize=(7.8,5.2))
    colors={"Media 7 días":ACCENT,"Media 28 días":BLUE,"Mediana 28 días":GREEN}
    for ax,issued in zip(axs,pd.to_datetime(["2026-04-27","2026-06-01"])):
        rows=summary[summary["Emisión"]==issued]
        arrival=rows["Llegada"].iloc[0]
        stock0=int(rows["Stock inicial simulado"].iloc[0])
        missed=rows["Faltantes antes llegada"].iloc[0]
        for name,ls in [("Media 7 días","-"),("Media 28 días","--"),("Mediana 28 días",":")]:
            p=trajectories[(trajectories.emision==issued)&(trajectories.metodo==name)]
            quantity=int(rows.loc[rows["Método"]==name,"Pedido calculado"].iloc[0])
            ax.plot(p.fecha,p.stock_cierre,color=colors[name],ls=ls,label=f"{name} · pedido {quantity}")
        ax.axvline(arrival,color=MUTED,ls=":",lw=1.3)
        ax.axhline(0,color=MUTED,lw=.8)
        ax.set_title(f"Emisión {issued:%d/%m} · stock inicial simulado: {stock0}",fontsize=10)
        ax.text(.02 if issued.month==4 else .4,.88,f"{int(missed)} unidades sin atender antes de la llegada",transform=ax.transAxes,fontsize=9,color=INK)
        ax.set_ylabel("Stock simulado")
        style(ax,dates=True);ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
        ax.legend(ncol=3,loc="upper center",bbox_to_anchor=(.5,-.16),fontsize=8.5)
        if issued.month==4:
            ax.text(arrival+pd.Timedelta(days=1),ax.get_ylim()[1]*.75,"Llegada\n27/05",fontsize=9,color=MUTED)
        else:
            ax.set_xlim(issued,pd.Timestamp("2026-07-03"))
            ax.text(arrival-pd.Timedelta(days=1),ax.get_ylim()[1]*.63,"Llegada: 01/07\nfuera del período",ha="right",fontsize=9,color=MUTED)
    fig.tight_layout(h_pad=2.1)
    save(fig,"12_escenarios_reposicion.png","La fecha de llegada limita el efecto de aumentar el pedido",
         "Escenarios hipotéticos: plazo fijo de 30 días, stock inicial reducido al 30 % del saldo contable y pedidos para cubrir 60 días. No se incluyen otras recepciones. En junio, las tres reglas producen la misma trayectoria hasta el cierre porque la compra llega el 1 de julio; su efecto posterior no puede evaluarse.")


def main():
    descriptive()
    forecasts()
    assert len(CATALOG) == 12
    CATALOG.sort(key=lambda r: r["basename"])
    (OUT / "catalogo_figuras.json").write_text(json.dumps(CATALOG,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
