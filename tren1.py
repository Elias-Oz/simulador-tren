import streamlit as st
import numpy as np

# ============================================================
# SIMULADOR TREN DE CERCANÍAS
# CAPEX -> PDI requerido para mantener Project IRR objetivo
# ============================================================

st.set_page_config(
    page_title="Simulador Tren de Cercanías",
    page_icon="🚆",
    layout="wide"
)

# -----------------------------
# Parámetros base del modelo
# -----------------------------
BASE_CAPEX = 479_481_861.0
BASE_PDI = 55_972_365.9769
TARGET_IRR_DEFAULT = 0.09

# Distribución del CAPEX: años 1 a 4
CAPEX_SHARES = np.array([
    0.0315,
    0.3874,
    0.3874,
    0.1937
])

# OPEX inicial y crecimiento anual
OPEX_INITIAL = 16_000_000.0
OPEX_GROWTH = 0.04
OPEX_SHARE_TO_REVENUE = 0.10

# PDI durante 20 años de operación
PDI_YEARS = 20

# Horizonte: 4 años construcción + 30 años operación
TOTAL_YEARS = 34


def build_cash_flow(capex, pdi):
    """
    Flujo de caja del proyecto, sin financiamiento.

    Años 1-4:
        CAPEX negativo.

    Años 5-24:
        PDI + 10% OPEX.

    Años 25-34:
        10% OPEX.
    """
    cf = np.zeros(TOTAL_YEARS)

    # Construcción
    for i in range(4):
        cf[i] = -capex * CAPEX_SHARES[i]

    # Operación
    for year in range(5, TOTAL_YEARS + 1):
        opex = OPEX_INITIAL * ((1 + OPEX_GROWTH) ** (year - 5))
        revenue = OPEX_SHARE_TO_REVENUE * opex

        if year <= 24:
            revenue += pdi

        cf[year - 1] = revenue

    return cf


def npv(rate, cash_flows):
    return sum(
        cf / ((1 + rate) ** (i + 1))
        for i, cf in enumerate(cash_flows)
    )


def calculate_irr(cash_flows):
    """
    Calcula la TIR mediante búsqueda binaria.
    """
    low = -0.99
    high = 1.0

    # Expandir el límite superior si fuera necesario
    while npv(high, cash_flows) > 0 and high < 100:
        high *= 2

    if npv(low, cash_flows) * npv(high, cash_flows) > 0:
        return None

    for _ in range(200):
        mid = (low + high) / 2
        value = npv(mid, cash_flows)

        if value > 0:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def solve_pdi(capex, target_irr):
    """
    Encuentra el PDI que hace que la Project IRR sea igual
    al objetivo.
    """
    low = 0.0
    high = 200_000_000.0

    def irr_for_pdi(pdi):
        return calculate_irr(build_cash_flow(capex, pdi))

    # Asegurar que el límite superior alcance la TIR objetivo
    while irr_for_pdi(high) is not None and irr_for_pdi(high) < target_irr:
        high *= 2
        if high > 2_000_000_000:
            return None

    for _ in range(150):
        mid = (low + high) / 2
        irr = irr_for_pdi(mid)

        if irr is None:
            return None

        if irr < target_irr:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def mm(value):
    return f"USD {value / 1_000_000:,.2f} MM"


# ============================================================
# INTERFAZ
# ============================================================

st.title("🚆 Simulador Financiero — Tren de Cercanías")

st.caption(
    "Simulación del PDI requerido para mantener una Project IRR objetivo "
    "ante variaciones del CAPEX."
)

st.divider()

# -----------------------------
# Inputs
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    capex_variation = st.slider(
        "Variación del CAPEX",
        min_value=-20.0,
        max_value=50.0,
        value=0.0,
        step=1.0,
        format="%.0f%%"
    )

with col2:
    target_irr_percent = st.number_input(
        "Project IRR objetivo",
        min_value=0.0,
        max_value=30.0,
        value=TARGET_IRR_DEFAULT * 100,
        step=0.1,
        format="%.1f%%"
    )

target_irr = target_irr_percent / 100
simulated_capex = BASE_CAPEX * (1 + capex_variation / 100)

# -----------------------------
# Simulación
# -----------------------------
if st.button("SIMULAR", type="primary", use_container_width=True):

    required_pdi = solve_pdi(simulated_capex, target_irr)

    if required_pdi is None:
        st.error("No fue posible encontrar un PDI compatible con el objetivo.")
    else:
        resulting_cf = build_cash_flow(simulated_capex, required_pdi)
        resulting_irr = calculate_irr(resulting_cf)

        delta_pdi = required_pdi - BASE_PDI
        delta_pdi_pct = (delta_pdi / BASE_PDI) * 100

        st.subheader("Resultado")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "CAPEX simulado",
                mm(simulated_capex)
            )

        with c2:
            st.metric(
                "PDI requerido",
                mm(required_pdi),
                f"{delta_pdi_pct:+.2f}% vs. PDI base"
            )

        with c3:
            st.metric(
                "Project IRR",
                f"{resulting_irr * 100:.2f}%"
            )

        st.divider()

        # Tabla resumen
        data = {
            "Variable": [
                "CAPEX base",
                "Variación CAPEX",
                "CAPEX simulado",
                "PDI base",
                "PDI requerido",
                "Variación PDI",
                "Project IRR objetivo",
                "Project IRR resultante",
            ],
            "Valor": [
                mm(BASE_CAPEX),
                f"{capex_variation:+.1f}%",
                mm(simulated_capex),
                mm(BASE_PDI),
                mm(required_pdi),
                f"{mm(delta_pdi)} ({delta_pdi_pct:+.2f}%)",
                f"{target_irr * 100:.2f}%",
                f"{resulting_irr * 100:.2f}%",
            ]
        }

        st.table(data)

else:
    # Mostrar valores iniciales antes de simular
    st.subheader("Escenario base")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("CAPEX base", mm(BASE_CAPEX))

    with c2:
        st.metric("PDI base", mm(BASE_PDI))

    with c3:
        base_irr = calculate_irr(build_cash_flow(BASE_CAPEX, BASE_PDI))
        st.metric("Project IRR base", f"{base_irr * 100:.2f}%")

    st.info("Modificá el CAPEX y presioná **SIMULAR**.")
