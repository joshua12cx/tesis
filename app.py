import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ================================
st.set_page_config(
    page_title="Riesgo de preeclampsia",
    page_icon="🩺",
    layout="wide",
)

# Estilos propios (solo para la tarjeta de resultado)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');
    :root { --teal: #2dd4bf; --muted: #6b93bc; --line: rgba(45,212,191,.18); --card: #0a1628; }
    .stApp { font-family: 'Inter', sans-serif; }
    h1, h2, h3, .titulo, .nivel, .card-t { font-family: 'Outfit', sans-serif; }
    .mono { font-family: 'JetBrains Mono', monospace; }
    footer, #MainMenu { visibility: hidden; }
    .block-container { padding-top: 2rem; max-width: 1150px; }
    button[role="tab"] { font-weight: 600; }
    [data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: .8rem 1rem; }

    .cabecera { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
    .cabecera .titulo { font-size: 2rem; font-weight: 700; color: #e8f0fe; line-height: 1.15; }
    .cabecera .subtitulo { color: var(--muted); margin-top: .4rem; max-width: 42rem; }
    .cabecera .sello { border: 1px solid var(--line); color: var(--teal); border-radius: 999px; padding: .25rem .85rem; font-size: .8rem; white-space: nowrap; }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.card-t):not(:has(div[data-testid="stVerticalBlockBorderWrapper"])) {
        background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: .4rem .6rem; }
    .card-t { font-size: 1.05rem; font-weight: 600; color: var(--teal); margin-bottom: .2rem; }

    [data-testid="stHorizontalBlock"]:has(.panel-ancla) { align-items: flex-start; }
    [data-testid="stColumn"]:has(.panel-ancla), [data-testid="column"]:has(.panel-ancla) { position: sticky; top: 3.5rem; }
    .panel { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 1.5rem 1.4rem; text-align: center; animation: entra .5s ease both; }
    .gauge { position: relative; width: 200px; height: 200px; margin: 0 auto .8rem; border-radius: 50%; }
    .gauge.alto { animation: pulso 2s infinite; }
    .gauge .arco { animation: llenar 1s ease-out both; }
    .gauge-n { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
    .gauge-p { font-size: 2.3rem; font-weight: 500; color: #e8f0fe; }
    .gauge-l { font-size: .8rem; color: var(--muted); }
    .panel .nivel { font-size: 1.5rem; font-weight: 700; }
    .panel .chip { display: inline-block; margin: .5rem 0 .2rem; border: 1px solid var(--c); color: var(--c); border-radius: 999px; padding: .2rem .8rem; font-size: .85rem; font-weight: 600; }
    .panel .sub { font-size: .8rem; color: var(--muted); }
    .panel .sec { text-align: left; font-size: .8rem; color: var(--muted); border-top: 1px solid var(--line); margin-top: 1.1rem; padding-top: .9rem; }
    .fac { display: flex; justify-content: space-between; padding: .3rem 0; font-size: .92rem; color: #e8f0fe; }
    .fac-vacio { text-align: left; font-size: .9rem; color: var(--muted); padding: .3rem 0; }
    .panel .nota { text-align: left; font-size: .88rem; color: #b9cde4; margin-top: .9rem; background: #0f2040; border-radius: 10px; padding: .7rem .9rem; }
    .aviso { border-top: 1px solid var(--line); margin-top: 1.5rem; padding-top: .8rem; font-size: .85rem; color: var(--muted); }

    @keyframes llenar { from { stroke-dashoffset: 427.26; } }
    @keyframes entra { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
    @keyframes pulso { 0% { box-shadow: 0 0 0 0 rgba(251,113,133,.4); } 70% { box-shadow: 0 0 0 16px rgba(251,113,133,0); } 100% { box-shadow: 0 0 0 0 rgba(251,113,133,0); } }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# CARGA DE ARTEFACTOS
# ==========================================
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(__file__)
    art_dir = os.path.join(base_dir, "artefactos", "v1")

    with open(os.path.join(art_dir, "input_schema.json"), "r", encoding="utf-8") as f:
        input_schema = json.load(f)
    with open(os.path.join(art_dir, "label_map.json"), "r", encoding="utf-8") as f:
        label_map = json.load(f)
    with open(os.path.join(art_dir, "decision_policy.json"), "r", encoding="utf-8") as f:
        policy = json.load(f)
    with open(os.path.join(art_dir, "sample_inputs.json"), "r", encoding="utf-8") as f:
        samples = json.load(f)

    pipe = joblib.load(os.path.join(art_dir, "pipeline_NNM.joblib"))
    rev_label = {v: k for k, v in label_map.items()}
    thr = float(policy["threshold"])
    return pipe, input_schema, label_map, rev_label, thr, policy, samples


PIPE, INPUT_SCHEMA, LABEL_MAP, REV_LABEL, THRESHOLD, POLICY, SAMPLES = load_artifacts()
FEATURES = list(INPUT_SCHEMA.keys())

# Nombres legibles y rangos permitidos (se usan en formulario, validación y tablas)
ETIQUETAS = {
    "edad": "Edad",
    "imc": "IMC",
    "p_a_sistolica": "Presión sistólica",
    "p_a_diastolica": "Presión diastólica",
    "hipertension": "Hipertensión previa",
    "diabetes": "Diabetes",
    "creatinina": "Creatinina",
    "ant_fam_hiper": "Antecedentes familiares de hipertensión",
    "tec_repro_asistida": "Reproducción asistida",
}
RANGOS = {
    "edad": (10, 60),
    "imc": (10.0, 60.0),
    "p_a_sistolica": (80, 200),
    "p_a_diastolica": (40, 130),
    "creatinina": (0.1, 10.0),
}

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================
def _coerce_and_align(df: pd.DataFrame) -> pd.DataFrame:
    """Alinea columnas y fuerza tipos según input_schema.json."""
    df = df.copy()
    for col, tipo in INPUT_SCHEMA.items():
        if col not in df.columns:
            df[col] = np.nan
        if "int" in tipo or "float" in tipo:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype("string").str.strip().str.upper()
    return df[FEATURES]


def predict_batch(records, thr=None):
    """Devuelve probabilidad y clase textual."""
    thr = THRESHOLD if thr is None else float(thr)

    if isinstance(records, dict):
        records = [records]

    df = _coerce_and_align(pd.DataFrame(records))
    proba = PIPE.predict_proba(df)[:, 1]
    y_int = (proba >= thr).astype(int)

    resultados = []
    for p, y in zip(proba, y_int):
        resultados.append({
            "proba": float(p),
            "pred_int": int(y),
            "pred_label": REV_LABEL[int(y)],
            "threshold": thr,
        })
    return resultados


def nivel_riesgo(p: float):
    """Nivel de lectura y color. El corte 'Alto' coincide con el umbral del modelo."""
    if p < THRESHOLD * 0.6:
        return "Riesgo bajo", "#2dd4bf"
    if p < THRESHOLD:
        return "Riesgo intermedio", "#fbbf24"
    return "Riesgo alto", "#fb7185"


def sensibilidad(registro: dict, proba_base: float) -> pd.DataFrame:
    """Cambia un dato a la vez y mide cuánto se mueve la probabilidad."""
    filas = []
    for col in FEATURES:
        alt = dict(registro)
        valor = registro[col]
        if col in RANGOS:
            lo, hi = RANGOS[col]
            nuevo = valor * 1.10
            if nuevo > hi:
                nuevo = valor * 0.90
            nuevo = min(max(nuevo, lo), hi)
            nuevo = int(round(nuevo)) if isinstance(valor, (int, np.integer)) else round(float(nuevo), 2)
            texto = f"{valor} → {nuevo}"
        else:
            nuevo = "NO" if valor == "SI" else "SI"
            texto = f"{valor} → {nuevo}"
        alt[col] = nuevo
        delta = (predict_batch(alt)[0]["proba"] - proba_base) * 100
        filas.append({
            "Dato": ETIQUETAS[col],
            "Cambio simulado": texto,
            "Efecto (puntos %)": round(delta, 1),
        })
    df = pd.DataFrame(filas)
    return df.reindex(df["Efecto (puntos %)"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def panel_resultado(res: dict, sens: pd.DataFrame):
    p = res["proba"]
    nivel, color = nivel_riesgo(p)
    C = 427.26
    top = sens[sens["Efecto (puntos %)"].abs() >= 0.5].head(4)
    if top.empty:
        filas = '<div class="fac-vacio">Ningún dato, modificado por separado, cambia la estimación.</div>'
    else:
        filas = "".join(
            f'<div class="fac"><span>{r["Dato"]}</span><span class="mono">{r["Efecto (puntos %)"]:+.1f} pts</span></div>'
            for _, r in top.iterrows()
        )
    nota = {
        "Riesgo bajo": "Con estos datos el modelo no detecta señales de riesgo.",
        "Riesgo intermedio": "Estimación cercana al umbral. Conviene revisar los datos ingresados y considerar seguimiento.",
        "Riesgo alto": "El modelo detecta un patrón asociado a riesgo. Se sugiere valoración clínica.",
    }[nivel]
    clase = "alto" if nivel == "Riesgo alto" else ""
    st.markdown(
        f"""
<div class="panel" style="--c:{color}">
<div class="gauge {clase}"><svg viewBox="0 0 160 160" width="200" height="200">
<circle cx="80" cy="80" r="68" fill="none" stroke="#162d57" stroke-width="12"/>
<circle class="arco" cx="80" cy="80" r="68" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round" stroke-dasharray="{C}" stroke-dashoffset="{C * (1 - p):.1f}" transform="rotate(-90 80 80)"/>
</svg><div class="gauge-n"><div class="gauge-p mono">{p:.0%}</div><div class="gauge-l">probabilidad</div></div></div>
<div class="nivel" style="color:{color}">{nivel}</div>
<div class="chip">El modelo clasifica: {res["pred_label"]}</div>
<div class="sub">Umbral de decisión: {THRESHOLD:.2f}</div>
<div class="sec">Datos que más mueven la estimación (uno a la vez, +10 % o el valor contrario)</div>
{filas}
<div class="nota">{nota}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def si_no(etiqueta: str, clave: str) -> str:
    valor = st.segmented_control(etiqueta, ["NO", "SI"], default="NO", key=clave)
    return valor or "NO"


# ==========================================
# ENCABEZADO
# ==========================================
st.markdown(
    """
    <div class="cabecera">
      <div>
        <div class="titulo">Riesgo de preeclampsia en gestantes</div>
        <div class="subtitulo">Estimación en tiempo real con un modelo de red neuronal a partir de nueve datos clínicos. Mueve cualquier dato y el resultado se actualiza.</div>
      </div>
      <div class="sello">Prototipo académico</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# PESTAÑAS
# ==========================================
tab_pred, tab_lote, tab_modelo = st.tabs(["Predicción", "Carga por lote", "Acerca del modelo"])

# ---------- Pestaña 1: predicción en vivo ----------
with tab_pred:
    col_form, col_res = st.columns([1.45, 1], gap="large")

    with col_form:
        with st.container(border=True):
            st.markdown('<div class="card-t">Datos maternos</div>', unsafe_allow_html=True)
            edad = st.slider("Edad (años)", 10, 60, 30)
            imc = st.slider("IMC (kg/m²)", 10.0, 60.0, 25.0, step=0.1)
        with st.container(border=True):
            st.markdown('<div class="card-t">Presión arterial</div>', unsafe_allow_html=True)
            p_a_sistolica = st.slider("Presión sistólica (mmHg)", 80, 200, 120)
            p_a_diastolica = st.slider("Presión diastólica (mmHg)", 40, 130, 80)
        with st.container(border=True):
            st.markdown('<div class="card-t">Laboratorio</div>', unsafe_allow_html=True)
            creatinina = st.slider("Creatinina (mg/dL)", 0.3, 5.0, 1.0, step=0.01)
        with st.container(border=True):
            st.markdown('<div class="card-t">Antecedentes</div>', unsafe_allow_html=True)
            a1, a2 = st.columns(2)
            with a1:
                hipertension = si_no("Hipertensión previa", "hta")
                ant_fam_hiper = si_no("Antecedentes familiares de hipertensión", "fam")
            with a2:
                diabetes = si_no("Diabetes", "dm")
                tec_repro_asistida = si_no("Reproducción asistida", "tra")

    registro = {
        "edad": int(edad),
        "imc": float(imc),
        "p_a_sistolica": int(p_a_sistolica),
        "p_a_diastolica": int(p_a_diastolica),
        "hipertension": hipertension,
        "diabetes": diabetes,
        "creatinina": float(creatinina),
        "ant_fam_hiper": ant_fam_hiper,
        "tec_repro_asistida": tec_repro_asistida,
    }

    with col_res:
        st.markdown('<div class="panel-ancla"></div>', unsafe_allow_html=True)
        if p_a_sistolica <= p_a_diastolica:
            st.error("La presión sistólica debe ser mayor que la diastólica.")
        else:
            res = predict_batch(registro)[0]
            panel_resultado(res, sensibilidad(registro, res["proba"]))
            if hipertension == "NO" and (p_a_sistolica >= 140 or p_a_diastolica >= 90):
                st.warning(
                    "Indicaste que no hay hipertensión previa, pero la presión está en rango hipertensivo "
                    "(≥140/90 mmHg). Revisa que los datos sean correctos."
                )

    st.markdown('<div class="aviso">Este sistema es solo apoyo a la decisión clínica y no reemplaza el criterio médico.</div>', unsafe_allow_html=True)

# ---------- Pestaña 2: carga por lote ----------
with tab_lote:
    st.subheader("Predicción para varias pacientes")
    st.write(
        "Sube un archivo CSV con una fila por paciente y estas columnas: "
        + ", ".join(f"`{c}`" for c in FEATURES)
        + ". Los campos de SI/NO deben decir `SI` o `NO`."
    )

    plantilla = pd.DataFrame(SAMPLES)[FEATURES].to_csv(index=False).encode("utf-8")
    st.download_button("Descargar plantilla de ejemplo", plantilla, "plantilla_pacientes.csv", "text/csv")

    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        try:
            df_in = pd.read_csv(archivo)
        except Exception:
            st.error("No se pudo leer el archivo. Verifica que sea un CSV válido.")
            df_in = None

        if df_in is not None:
            faltan = [c for c in FEATURES if c not in df_in.columns]
            if faltan:
                st.error("Faltan estas columnas: " + ", ".join(f"`{c}`" for c in faltan))
            else:
                limpio = _coerce_and_align(df_in)
                invalidas = limpio.isna().any(axis=1) | limpio[
                    ["hipertension", "diabetes", "ant_fam_hiper", "tec_repro_asistida"]
                ].isin(["SI", "NO"]).eq(False).any(axis=1)
                validas = limpio[~invalidas]

                if invalidas.any():
                    filas_malas = ", ".join(str(i + 2) for i in df_in.index[invalidas][:15])
                    st.warning(
                        f"{int(invalidas.sum())} fila(s) con datos vacíos o inválidos se omitieron "
                        f"(filas del archivo: {filas_malas}{'…' if invalidas.sum() > 15 else ''})."
                    )

                if validas.empty:
                    st.error("No quedaron filas válidas para predecir.")
                else:
                    resultados = predict_batch(validas.to_dict("records"))
                    salida = validas.copy()
                    salida["probabilidad"] = [round(r["proba"], 4) for r in resultados]
                    salida["resultado"] = [r["pred_label"] for r in resultados]

                    n = len(salida)
                    n_riesgo = int((salida["resultado"] == "RIESGO").sum())
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Pacientes evaluadas", n)
                    c2.metric("Con RIESGO", n_riesgo)
                    c3.metric("Sin riesgo", n - n_riesgo)

                    st.dataframe(
                        salida,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "probabilidad": st.column_config.ProgressColumn(
                                "probabilidad", min_value=0.0, max_value=1.0, format="%.2f"
                            )
                        },
                    )
                    st.download_button(
                        "Descargar resultados (CSV)",
                        salida.to_csv(index=False).encode("utf-8"),
                        "resultados_preeclampsia.csv",
                        "text/csv",
                        type="primary",
                    )

# ---------- Pestaña 3: acerca del modelo ----------
with tab_modelo:
    m = POLICY["test_metrics"]

    st.subheader("Desempeño en el conjunto de prueba")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("F1", f"{m['f1']:.3f}")
    c2.metric("Precisión", f"{m['precision']:.3f}")
    c3.metric("Sensibilidad", f"{m['recall']:.3f}")
    c4.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c5.metric("PR-AUC", f"{m['pr_auc']:.3f}")

    st.write("**Matriz de confusión** (filas: valor real, columnas: predicción)")
    cm = pd.DataFrame(
        m["confusion_matrix"],
        index=["Real: SIN RIESGO", "Real: RIESGO"],
        columns=["Pred: SIN RIESGO", "Pred: RIESGO"],
    )
    st.dataframe(cm, width="content")

    st.subheader("Cómo se construyó")
    st.markdown(
        f"""
- **Modelo:** {POLICY['winner']} (red neuronal MLP con una capa oculta de 64 neuronas).
- **Datos:** 1 800 registros, con 45.4 % de casos de riesgo.
- **Partición:** 80 % entrenamiento y 20 % prueba, estratificada.
- **Preprocesamiento:** estandarización de variables numéricas, codificación one-hot de las categóricas y SMOTE para balancear clases.
- **Umbral de decisión:** {THRESHOLD:.2f}.
- **Variables:** {", ".join(ETIQUETAS[c].lower() for c in FEATURES)}.
        """
    )

    st.subheader("Limitaciones")
    st.warning(
        "Las métricas son muy altas y provienen de un único conjunto de datos, sin validación externa. "
        "Antes de cualquier uso clínico, el modelo debe probarse con datos clínicos reales y "
        "revisarse por posible sobreajuste. Por ahora es un prototipo académico."
    )