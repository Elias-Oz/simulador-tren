
import streamlit as st
from streamlit import components
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt

# ============================================================
# SIMULADOR TREN DE CERCANÍAS
# CAPEX + tasas de deuda + inflación + estructura OPEX
# ============================================================

# -----------------------------
# DATOS BASE DEL MODELO
# -----------------------------
BASE_CAPEX = 479_481_861.0

CAPEX_SHARES = np.array([
    0.0315,
    0.3874,
    0.3874,
    0.1937
])

PDI_ORIGINAL = 62_803_694.0
PDI_YEARS = 15

# OPEX
OPEX_INITIAL = 16_000_000.0
INFLATION_US = 0.02
PLUS_OPEX = 0.10

# Construcción / operación
CONSTRUCTION_YEARS = 4
OPERATION_YEARS = 30
N_YEARS = CONSTRUCTION_YEARS + OPERATION_YEARS

# TIR objetivo
TARGET_IRR = 0.09

# -----------------------------
# FINANCIAMIENTO
# -----------------------------
DEBT1_AMOUNT = 100_000_000.0
DEBT2_AMOUNT = 139_740_930.5

DEBT1_TERM = 12

DEBT1_FEE = 0.003
DEBT2_FEE = 0.003
DEBT2_COMMITMENT_FEE = 0.015


# ============================================================
# FUNCIONES FINANCIERAS
# ============================================================

def irr(cashflows):
    return float(npf.irr(np.asarray(cashflows, dtype=float)))


def ipmt(rate, per, nper, pv):
    if rate == 0:
        return 0.0

    pmt = npf.pmt(rate, nper, pv)

    balance_before = (
        pv * (1 + rate) ** (per - 1)
        + pmt * ((1 + rate) ** (per - 1) - 1) / rate
    )

    return -balance_before * rate


def ppmt(rate, per, nper, pv):
    pmt = npf.pmt(rate, nper, pv)

    if rate == 0:
        return pmt

    interest = ipmt(rate, per, nper, pv)
    return pmt - interest


# ============================================================
# OPEX
# ============================================================

def calculate_opex(
    inflation_py,
    gasto_gs_share,
    gasto_usd_share
):
    """
    OPEX en USD.

    Componente nacional:
        OPEX_Gs,t = OPEX_Gs,0 * (1 + inflación PY)^t

    Componente internacional:
        OPEX_USD,t = OPEX_USD,0 *
                     [(1 + inflación PY) /
                      (1 + inflación EEUU)]^t

    Importante:
    El factor PPP utilizado para proyectar el componente
    internacional es el ÍNDICE, por ejemplo 1.0196078,
    no 0.0196078.
    """

    inflation_index = np.array([
        (1 + inflation_py) ** t
        for t in range(N_YEARS)
    ])

    ppp_index_annual = (
        (1 + inflation_py) /
        (1 + INFLATION_US)
    )

    ppp_index = np.array([
        ppp_index_annual ** t
        for t in range(N_YEARS)
    ])

    opex_gs_initial = OPEX_INITIAL * gasto_gs_share
    opex_usd_initial = OPEX_INITIAL * gasto_usd_share

    opex_gs = opex_gs_initial * inflation_index
    opex_usd = opex_usd_initial * ppp_index

    opex_total = opex_gs + opex_usd

    return {
        "opex": opex_total,
        "opex_gs": opex_gs,
        "opex_usd": opex_usd,
        "inflation_index": inflation_index,
        "ppp_index": ppp_index,
        "ppp_index_annual": ppp_index_annual,
        "ppp_devaluation_rate": ppp_index_annual - 1,
    }


# ============================================================
# MODELO
# ============================================================

def build_model(
    pdi,
    capex_variation,
    debt1_rate,
    debt2_rate,
    inflation_py,
    gasto_gs_share
):

    gasto_usd_share = 1 - gasto_gs_share

    capex = BASE_CAPEX * (1 + capex_variation)

    # -----------------------------
    # CAPEX
    # -----------------------------
    capex_by_year = np.zeros(N_YEARS)
    capex_by_year[:CONSTRUCTION_YEARS] = (
        capex * CAPEX_SHARES
    )

    # -----------------------------
    # OPEX + PDI
    # -----------------------------
    opex_data = calculate_opex(
        inflation_py,
        gasto_gs_share,
        gasto_usd_share
    )

    opex = opex_data["opex"]

    income = np.zeros(N_YEARS)

    # 15 años con PDI
    for t in range(
        CONSTRUCTION_YEARS,
        CONSTRUCTION_YEARS + PDI_YEARS
    ):
        income[t] = pdi + PLUS_OPEX * opex[t]

    # Después del período PDI queda solamente 10% OPEX
    for t in range(
        CONSTRUCTION_YEARS + PDI_YEARS,
        N_YEARS
    ):
        income[t] = PLUS_OPEX * opex[t]

    # ========================================================
    # DEUDA 1
    # ========================================================

    debt1_schedule = (
        DEBT1_AMOUNT / BASE_CAPEX
    ) * capex

    debt1_draw = np.zeros(N_YEARS)
    debt1_draw[:4] = (
        debt1_schedule * CAPEX_SHARES
    )

    debt1_balance = np.zeros(N_YEARS)
    debt1_construction_interest = np.zeros(N_YEARS)

    for t in range(4):
        previous = (
            debt1_balance[t - 1]
            if t > 0 else 0
        )

        debt1_construction_interest[t] = (
            debt1_rate *
            (debt1_draw[t] / 2 + previous)
        )

        debt1_balance[t] = (
            debt1_draw[t]
            + previous
            + debt1_construction_interest[t]
        )

    debt1_interest = np.zeros(N_YEARS)
    debt1_principal = np.zeros(N_YEARS)

    for t in range(4, N_YEARS):
        per = t - 3

        if per <= DEBT1_TERM:
            debt1_interest[t] = -ipmt(
                debt1_rate,
                per,
                DEBT1_TERM,
                debt1_balance[3]
            )

            debt1_principal[t] = -ppmt(
                debt1_rate,
                per,
                DEBT1_TERM,
                debt1_balance[3]
            )

        debt1_balance[t] = max(
            debt1_balance[t - 1]
            - debt1_principal[t],
            0
        )

    # ========================================================
    # DEUDA 2
    # ========================================================

    debt2_schedule = (
        DEBT2_AMOUNT / BASE_CAPEX
    ) * capex

    debt2_draw = np.zeros(N_YEARS)
    debt2_draw[:4] = (
        debt2_schedule * CAPEX_SHARES
    )

    debt2_balance = np.zeros(N_YEARS)
    debt2_construction_interest = np.zeros(N_YEARS)
    commitment_fee = np.zeros(N_YEARS)

    for t in range(4):
        previous = (
            debt2_balance[t - 1]
            if t > 0 else 0
        )

        debt2_construction_interest[t] = (
            debt2_rate *
            (debt2_draw[t] / 2 + previous)
        )

        commitment_fee[t] = max(
            (
                debt2_schedule
                - debt2_draw[t]
                - previous
            ) * DEBT2_COMMITMENT_FEE,
            0
        )

        debt2_balance[t] = (
            debt2_draw[t]
            + previous
            + debt2_construction_interest[t]
        )

    debt2_interest = np.zeros(N_YEARS)
    debt2_principal = np.zeros(N_YEARS)

    for t in range(4, N_YEARS):
        debt2_interest[t] = (
            debt2_balance[t - 1]
            * debt2_rate
        )

        # En el Excel, la amortización de Deuda 2 NO usa un DSCR fijo.
        # Se paga el máximo disponible después de intereses:
        # Principal = MIN(Disponible para servicio - Intereses, Saldo)
        # donde Disponible para servicio = PDI + 10% OPEX.
        debt2_principal[t] = min(
            max(
                income[t] - debt2_interest[t],
                0
            ),
            debt2_balance[t - 1]
        )

        debt2_balance[t] = max(
            debt2_balance[t - 1]
            - debt2_principal[t],
            0
        )

    # ========================================================
    # EQUITY
    # ========================================================

    fee1 = DEBT1_FEE * DEBT1_AMOUNT
    fee2 = DEBT2_FEE * debt2_schedule

    equity_drawdown = np.zeros(N_YEARS)

    for t in range(4):
        fees = 0

        if t == 0:
            fees += fee1
            fees += fee2

        fees += commitment_fee[t]

        equity_drawdown[t] = (
            capex_by_year[t]
            + fees
            - DEBT1_AMOUNT * CAPEX_SHARES[t]
            - DEBT2_AMOUNT * CAPEX_SHARES[t]
        )

    # ========================================================
    # CASH FLOW
    # ========================================================

    debt_service = (
        debt1_interest
        + debt1_principal
        + debt2_interest
        + debt2_principal
    )

    cash_available_for_distribution = (
        income - debt_service
    )

    project_cf = (
        -capex_by_year
        + income
    )

    equity_cf = (
        -equity_drawdown
        + cash_available_for_distribution
    )

    return {
        "capex": capex,
        "opex": opex,
        "opex_gs": opex_data["opex_gs"],
        "opex_usd": opex_data["opex_usd"],
        "ppp_index_annual": opex_data["ppp_index_annual"],
        "ppp_devaluation_rate": opex_data["ppp_devaluation_rate"],
        "project_cf": project_cf,
        "equity_cf": equity_cf,
        "project_irr": irr(project_cf),
        "equity_irr": irr(equity_cf),
    }


# ============================================================
# SOLVER DE PDI
# ============================================================

def solve_pdi(
    capex_variation,
    target,
    mode,
    debt1_rate,
    debt2_rate,
    inflation_py,
    gasto_gs_share
):

    low = 0.0
    high = 150_000_000.0

    def objective(pdi):
        model = build_model(
            pdi,
            capex_variation,
            debt1_rate,
            debt2_rate,
            inflation_py,
            gasto_gs_share
        )

        if mode == "project":
            return model["project_irr"] - target

        return model["equity_irr"] - target

    f_high = objective(high)

    while f_high < 0:
        high *= 2
        f_high = objective(high)

        if high > 1_000_000_000:
            raise ValueError(
                "No se encontró solución para PDI."
            )

    for _ in range(120):
        mid = (low + high) / 2
        f_mid = objective(mid)

        if f_mid > 0:
            high = mid
        else:
            low = mid

    return (low + high) / 2


# ============================================================
# INTERFAZ
# ============================================================

# Simulación visual Asunción → Luque: se conserva la animación del recorrido y se oculta el panel textual del HTML original.
ROUTE_SIMULATION_HTML = '<!DOCTYPE html>\n<html lang="es">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>Simulación Tren Asunción–Luque</title>\n<style>\n  :root{color-scheme:dark}\n  *{box-sizing:border-box}\n  html,body{margin:0;height:100%;background:#07090c;color:#e8edf5;font-family:ui-sans-serif,system-ui,sans-serif}\n  .layout{display:grid;grid-template-columns:290px minmax(0,1fr);height:100%}\n  aside{padding:18px 16px;background:#0b0f14;border-right:1px solid #1c2430;overflow:auto}\n  .stage{position:relative;overflow:hidden}\n  canvas{display:block;width:100%;height:100%;background:#07090c}\n  h1{font-size:1.05rem;margin:0 0 6px}\n  p{color:#9aa6b8;font-size:.82rem;line-height:1.45}\n  .tag{font-size:.68rem;letter-spacing:.06em;text-transform:uppercase;color:#7d8b9c;margin-bottom:12px}\n  .row{display:flex;justify-content:space-between;gap:8px;font-size:.78rem;color:#c5d0dc;margin:5px 0}\n  .row b{font-variant-numeric:tabular-nums}\n  hr{border:0;border-top:1px solid #1c2430;margin:14px 0}\n  .note{font-size:.73rem;color:#6e7b8c}\n  button{width:100%;padding:9px 10px;border:1px solid #263444;border-radius:7px;background:#111821;color:#dce7f2;cursor:pointer}\n  button:hover{background:#17222e}\n  @media(max-width:820px){\n    .layout{grid-template-columns:1fr;grid-template-rows:auto 1fr}\n    aside{border-right:0;border-bottom:1px solid #1c2430}\n  }\n</style>\n\n<style>\nhtml, body {\n  margin: 0 !important;\n  padding: 0 !important;\n  width: 100%;\n  height: 100%;\n  overflow: hidden !important;\n  background: #07090c !important;\n}\n.layout {\n  display: block !important;\n  width: 100%;\n  height: 100%;\n}\naside {\n  display: none !important;\n}\n.stage {\n  position: relative !important;\n  width: 100% !important;\n  height: 100% !important;\n  overflow: hidden !important;\n}\ncanvas {\n  display: block !important;\n  width: 100% !important;\n  height: 100% !important;\n}\n</style>\n\n</head>\n<body>\n<div class="layout">\n  <aside>\n    <div class="tag">simulación ferroviaria · tramo 18 km</div>\n    <h1>Tren Asunción → Luque</h1>\n    <p>Una sola clase de tren y una sola simbología de estación. La intensidad horaria reproduce la estructura de demanda/frecuencia solicitada: pico fuerte de mañana, valle, pico de mediodía y pico fuerte de tarde.</p>\n\n    <div class="row"><span>hora simulada</span><b id="clock">05:00</b></div>\n    <div class="row"><span>período</span><b id="period">Offpeak</b></div>\n    <div class="row"><span>trenes/hora</span><b id="tph">5</b></div>\n    <div class="row"><span>headway</span><b id="headway">11,2 min</b></div>\n    <div class="row"><span>trenes en línea</span><b id="active">0</b></div>\n    <div class="row"><span>ciclos/día</span><b id="cycles">116</b></div>\n    <div class="row"><span>capacidad/h/dirección</span><b id="capacity">999</b></div>\n\n    <hr>\n    <div class="row"><span>distancia Central–Luque 3</span><b>18 km</b></div>\n    <div class="row"><span>distancia RT</span><b>36 km</b></div>\n\n    <hr>\n    <button id="pause" type="button">Pausar</button>\n    <p class="note">La animación comprime las 24 horas en 36 segundos. El movimiento del tren es visual; no se interpreta como una velocidad operacional real.</p>\n  </aside>\n\n  <div class="stage">\n    <canvas id="c"></canvas>\n  </div>\n</div>\n\n<script>\nconst ESTACIONES = [\n  {id:1,nombre:"Central",dm:0.000,lat:-25.28377,lon:-57.62983},\n  {id:2,nombre:"Av. Perú",dm:1.650,lat:-25.2795,lon:-57.6155},\n  {id:3,nombre:"Virgen de Luján",dm:2.700,lat:-25.2748,lon:-57.6060},\n  {id:4,nombre:"La Perseverancia",dm:4.100,lat:-25.2700,lon:-57.5935},\n  {id:5,nombre:"Santísima Trinidad",dm:6.050,lat:-25.2572,lon:-57.5776},\n  {id:6,nombre:"Botánico",dm:7.100,lat:-25.25654,lon:-57.56761},\n  {id:7,nombre:"Universidad de Taiwán",dm:10.250,lat:-25.2585,lon:-57.5350},\n  {id:8,nombre:"Conmebol",dm:12.500,lat:-25.2613,lon:-57.5147},\n  {id:9,nombre:"Luque 2",dm:14.100,lat:-25.2670,lon:-57.4990},\n  {id:10,nombre:"Luque Histórico",dm:15.200,lat:-25.2629,lon:-57.4904},\n  {id:11,nombre:"Luque 3",dm:18.000,lat:-25.2700,lon:-57.4720}\n];\n\n// 05:00–21:30 = 16,5 h de operación.\n// Se ubican los cuatro períodos de la tabla para que sumen exactamente 16,5 h.\nconst PERIODOS = [\n  {nombre:"Offpeak", inicio:5.0, fin:6.0, tph:5, headway:11.2, cap:999},\n  {nombre:"Morning", inicio:6.0, fin:8.0, tph:11, headway:5.3, cap:2089},\n  {nombre:"Offpeak", inicio:8.0, fin:11.0, tph:5, headway:11.2, cap:999},\n  {nombre:"Midday", inicio:11.0, fin:13.0, tph:9, headway:6.5, cap:1726},\n  {nombre:"Offpeak", inicio:13.0, fin:17.0, tph:5, headway:11.2, cap:999},\n  {nombre:"Evening", inicio:17.0, fin:19.0, tph:9, headway:6.5, cap:1726},\n  {nombre:"Offpeak", inicio:19.0, fin:21.5, tph:5, headway:11.2, cap:999}\n];\n\nconst SIM_START_HOUR = 5.0;\nconst SIM_END_HOUR = 21.5;\nconst DAY_MS = 36000; // 05:00–21:30 en 36 s\nconst TRAIN_VISUAL_RT_MS = 22000; // duración visual de un ciclo RT\nconst TRAIN_COUNT = 28;\nlet paused = false;\n\nconst canvas=document.getElementById("c");\nconst ctx=canvas.getContext("2d");\n\nfunction getPeriod(h){\n  for(const p of PERIODOS) if(h>=p.inicio && h<p.fin) return p;\n  return {nombre:"Sin operación",tph:0,headway:null,cap:0,inicio:21.5,fin:24};\n}\nfunction dayProgress(now){\n  return (now % DAY_MS) / DAY_MS;\n}\nfunction hourFromProgress(p){\n  return SIM_START_HOUR + p * (SIM_END_HOUR - SIM_START_HOUR);\n}\nfunction clock(h){\n  const m=Math.floor((h%24)*60);\n  return String(Math.floor(h%24)).padStart(2,"0")+":"+String(m%60).padStart(2,"0");\n}\nfunction project(lat,lon,w,h){\n  const xs=ESTACIONES.map(e=>e.lon), ys=ESTACIONES.map(e=>e.lat);\n  const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);\n  const pad=70;\n  const sx=(w-2*pad)/(maxX-minX);\n  const sy=(h-2*pad)/(maxY-minY);\n  const s=Math.min(sx,sy);\n  const usedW=(maxX-minX)*s, usedH=(maxY-minY)*s;\n  const ox=(w-usedW)/2, oy=(h-usedH)/2;\n  return {x:ox+(lon-minX)*s,y:oy+(maxY-lat)*s};\n}\nfunction routePoint(u,w,h){\n  u=Math.max(0,Math.min(1,u));\n  const scaled=u*(ESTACIONES.length-1);\n  const i=Math.min(ESTACIONES.length-2,Math.floor(scaled));\n  const t=scaled-i;\n  const a=project(ESTACIONES[i].lat,ESTACIONES[i].lon,w,h);\n  const b=project(ESTACIONES[i+1].lat,ESTACIONES[i+1].lon,w,h);\n  return {x:a.x+(b.x-a.x)*t,y:a.y+(b.y-a.y)*t};\n}\nfunction size(){\n  const dpr=devicePixelRatio||1;\n  const w=canvas.clientWidth,h=canvas.clientHeight;\n  canvas.width=Math.round(w*dpr); canvas.height=Math.round(h*dpr);\n  ctx.setTransform(dpr,0,0,dpr,0,0);\n  return {w,h};\n}\nfunction drawRoute(w,h){\n  ctx.lineCap="round";ctx.lineJoin="round";\n  ctx.strokeStyle="#1b3448";ctx.lineWidth=14;\n  ctx.beginPath();\n  ESTACIONES.forEach((e,i)=>{const p=project(e.lat,e.lon,w,h);i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y)});\n  ctx.stroke();\n  ctx.strokeStyle="#42657d";ctx.lineWidth=2;\n  ctx.beginPath();\n  ESTACIONES.forEach((e,i)=>{const p=project(e.lat,e.lon,w,h);i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y)});\n  ctx.stroke();\n}\nfunction drawStations(w,h){\n  ctx.textBaseline="middle";\n  ESTACIONES.forEach((e,i)=>{\n    const p=project(e.lat,e.lon,w,h);\n    ctx.fillStyle="#d8e6ef";\n    ctx.beginPath();ctx.arc(p.x,p.y,6.5,0,Math.PI*2);ctx.fill();\n    ctx.fillStyle="#0b0f14";\n    ctx.beginPath();ctx.arc(p.x,p.y,2.3,0,Math.PI*2);ctx.fill();\n\n    const align=i===0?"left":(i===ESTACIONES.length-1?"right":"center");\n    ctx.textAlign=align;\n    ctx.font="600 11px ui-sans-serif,system-ui";\n    ctx.fillStyle="#c9d6e4";\n    ctx.fillText(e.nombre,p.x+(i===0?10:(i===ESTACIONES.length-1?-10:0)),p.y-16);\n    ctx.font="10px ui-sans-serif,system-ui";\n    ctx.fillStyle="#708092";\n    ctx.fillText("Dm "+e.dm.toFixed(3),p.x+(i===0?10:(i===ESTACIONES.length-1?-10:0)),p.y+15);\n  });\n}\nfunction demandShape(h){\n  // Curva visual inspirada en el gráfico de VMT: madrugada casi nula,\n  // pico fuerte AM, valle, pico de mediodía, pico fuerte PM y caída nocturna.\n  const g=(x,mu,s,a)=>a*Math.exp(-0.5*((x-mu)/s)**2);\n  if(h<4 || h>23.5) return 0.03;\n  return Math.min(1,\n    0.10 +\n    g(h,6.8,0.75,0.90) +\n    g(h,12.3,1.45,0.34) +\n    g(h,17.7,0.95,0.82) +\n    g(h,20.8,1.25,0.10)\n  );\n}\nfunction drawDemand(w,h,p){\n  const left=38, bottom=h-28, width=Math.min(w-70,420), height=70;\n  ctx.fillStyle="#0e151d";ctx.fillRect(left,bottom-height,width,height);\n  ctx.strokeStyle="#263746";ctx.strokeRect(left,bottom-height,width,height);\n  const bins=96;\n  for(let i=0;i<bins;i++){\n    const hh=i*24/bins;\n    const v=demandShape(hh);\n    const bw=width/bins;\n    ctx.fillStyle="#42657d";\n    ctx.globalAlpha=.72;\n    ctx.fillRect(left+i*bw,bottom-1-v*(height-5),Math.max(1,bw-.5),v*(height-5));\n  }\n  ctx.globalAlpha=1;\n  ctx.strokeStyle="#e8edf5";ctx.lineWidth=2;\n  const currentHour=hourFromProgress(p); const x=left+(currentHour/24)*width;\n  ctx.beginPath();ctx.moveTo(x,bottom-height-2);ctx.lineTo(x,bottom+2);ctx.stroke();\n  ctx.font="10px ui-sans-serif,system-ui";ctx.fillStyle="#8b98a8";ctx.textAlign="left";\n  ctx.fillText("comportamiento horario",left,bottom-height-7);\n}\nfunction drawTrains(w,h,now){\n  const hour=hourFromProgress(dayProgress(now));\n  const p=getPeriod(hour);\n  let active=0;\n  if(p.tph>0){\n    for(let i=0;i<TRAIN_COUNT;i++){\n      const phase=(i/TRAIN_COUNT + now/TRAIN_VISUAL_RT_MS*0.5)%1;\n      const direction=i%2===0?1:-1;\n      // Más trenes durante períodos de mayor frecuencia.\n      const visibleScore=(i%11)/10;\n      const intensity=p.tph/11;\n      if(visibleScore>intensity && p.tph<11) continue;\n      const u=direction===1?phase:1-phase;\n      const pos=routePoint(u,w,h);\n      active++;\n      ctx.save();\n      ctx.globalCompositeOperation="lighter";\n      const glow=ctx.createRadialGradient(pos.x,pos.y,2,pos.x,pos.y,14);\n      glow.addColorStop(0,"rgba(95,174,255,.38)");\n      glow.addColorStop(1,"rgba(95,174,255,0)");\n      ctx.fillStyle=glow;ctx.beginPath();ctx.arc(pos.x,pos.y,14,0,Math.PI*2);ctx.fill();\n      ctx.restore();\n\n      ctx.fillStyle="#6bb8ff";\n      ctx.beginPath();ctx.arc(pos.x,pos.y,5.2,0,Math.PI*2);ctx.fill();\n      ctx.fillStyle="#dcefff";\n      ctx.beginPath();ctx.arc(pos.x,pos.y,2,0,Math.PI*2);ctx.fill();\n    }\n  }\n  document.getElementById("active").textContent=active;\n  document.getElementById("tph").textContent=p.tph;\n  document.getElementById("headway").textContent=p.headway?p.headway.toFixed(1).replace(".",",")+" min":"—";\n  document.getElementById("capacity").textContent=p.cap.toLocaleString("es-PY");\n  document.getElementById("period").textContent=p.nombre;\n}\nfunction frame(now){\n  if(!paused){\n    const {w,h}=size();\n    const p=dayProgress(now);\n    const hour=hourFromProgress(p);\n\n    ctx.clearRect(0,0,w,h);\n    const g=ctx.createRadialGradient(w*.55,h*.42,30,w*.5,h*.5,Math.max(w,h));\n    g.addColorStop(0,"#0d131a");g.addColorStop(1,"#07090c");\n    ctx.fillStyle=g;ctx.fillRect(0,0,w,h);\n\n    drawRoute(w,h);\n    drawTrains(w,h,now);\n    drawStations(w,h);\n    drawDemand(w,h,p);\n\n    ctx.fillStyle="#e8edf5";ctx.textAlign="right";ctx.font="600 28px ui-sans-serif,system-ui";\n    ctx.fillText(clock(hour),w-24,36);\n    ctx.font="12px ui-sans-serif,system-ui";ctx.fillStyle="#8b98a8";\n    ctx.fillText("Asunción → Luque · 18 km · 05:00–21:30",w-24,54);\n\n    document.getElementById("clock").textContent=clock(hour);\n  }\n  requestAnimationFrame(frame);\n}\ndocument.getElementById("pause").addEventListener("click",()=>{\n  paused=!paused;\n  document.getElementById("pause").textContent=paused?"Continuar":"Pausar";\n});\ndocument.getElementById("cycles").textContent="116";\nrequestAnimationFrame(frame);\n</script>\n</body>\n</html>\n'


st.set_page_config(
    page_title="Simulador Tren de Cercanías",
    layout="centered"
)

st.title("Simulador Tren de Cercanías")

st.caption(
    "Modelo financiero del Tren de Cercanías."
)


# ============================================================
# SIMULACIÓN OPERATIVA ASUNCIÓN → LUQUE
# ============================================================

st.subheader("Simulación operativa — Asunción → Luque")

components.v1.html(
    ROUTE_SIMULATION_HTML,
    height=400,
    scrolling=False
)

# -----------------------------
# CAPEX
# -----------------------------

with st.expander("Simulador CAPEX", expanded=False):
    capex_variation_pct = st.slider(
        "Variación del CAPEX",
        min_value=-30,
        max_value=50,
        value=0,
        step=1,
        format="%+d%%"
    )

capex_variation = capex_variation_pct / 100

# -----------------------------
# TASAS DE INTERÉS
# -----------------------------

with st.expander("Tasas de interés", expanded=False):
    col_rate1, col_rate2 = st.columns(2)

    with col_rate1:
        debt1_rate_pct = st.slider(
            "Tasa de interés Deuda 1",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.1,
            format="%.1f%%"
        )

    with col_rate2:
        debt2_rate_pct = st.slider(
            "Tasa de interés Deuda 2",
            min_value=0.0,
            max_value=15.0,
            value=7.0,
            step=0.1,
            format="%.1f%%"
        )

debt1_rate = debt1_rate_pct / 100
debt2_rate = debt2_rate_pct / 100

# -----------------------------
# INFLACIÓN
# -----------------------------

with st.expander("Inflación", expanded=False):
    inflation_py_pct = st.slider(
        "Inflación Paraguay",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.1,
        format="%.1f%%"
    )

    inflation_py = inflation_py_pct / 100

    ppp_index_annual = (
        (1 + inflation_py)
        / (1 + INFLATION_US)
    )

    ppp_devaluation_rate = ppp_index_annual - 1

    st.caption(
        f"Inflación EEUU: {INFLATION_US:.1%}  |  "
        f"Índice PPP: {ppp_index_annual:.4f}  |  "
        f"Devaluación equivalente: {ppp_devaluation_rate:.2%}"
    )

# -----------------------------
# ESTRUCTURA DE COSTOS DEL OPEX
# -----------------------------

with st.expander("Estructura de Costos del OPEX", expanded=False):
    gasto_gs_pct = st.slider(
        "Gasto en Gs",
        min_value=0,
        max_value=100,
        value=100,
        step=1,
        format="%d%%"
    )

    gasto_gs_share = gasto_gs_pct / 100
    gasto_usd_pct = 100 - gasto_gs_pct

    st.caption(
        f"Gasto en Gs: {gasto_gs_pct}%  |  "
        f"Gasto en USD: {gasto_usd_pct}%"
    )

# ============================================================
# BOTONES — SIEMPRE VISIBLES
# ============================================================

col1, col2 = st.columns(2)

with col1:
    simulate_project = st.button(
        "Simular TIR Proyecto = 9%",
        type="primary",
        use_container_width=True
    )

with col2:
    simulate_equity = st.button(
        "Simular TIR Accionista = 9%",
        use_container_width=True
    )


# ============================================================
# RESULTADOS
# ============================================================

def show_result(mode):
    new_pdi = solve_pdi(
        capex_variation,
        TARGET_IRR,
        mode,
        debt1_rate,
        debt2_rate,
        inflation_py,
        gasto_gs_share
    )

    model = build_model(
        PDI_ORIGINAL,
        capex_variation,
        debt1_rate,
        debt2_rate,
        inflation_py,
        gasto_gs_share
    )

    st.subheader("Resultado")

    pdi_original_mm = round(PDI_ORIGINAL / 1_000_000)
    nuevo_pdi_mm = round(new_pdi / 1_000_000)

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "PDI original",
            f"USD {pdi_original_mm} MM/año"
        )

    with c2:
        st.metric(
            "Nuevo PDI",
            f"USD {nuevo_pdi_mm} MM/año"
        )

    fig, ax = plt.subplots(figsize=(8, 3.8))

    valores = [pdi_original_mm, nuevo_pdi_mm]
    etiquetas = ["PDI original", "Nuevo PDI"]

    bars = ax.bar(
        etiquetas,
        valores,
        color=["#B8B8B8", "#FF4D4D"]
    )

    ax.set_ylabel("USD MM/año")
    ax.set_ylim(0, max(valores) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, valor in zip(bars, valores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(valores) * 0.025,
            f"{valor} MM",
            ha="center",
            va="bottom",
            fontsize=14
        )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


if simulate_project:
    show_result("project")

if simulate_equity:
    show_result("equity")
