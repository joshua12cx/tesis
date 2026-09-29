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
    footer, #MainMenu { visibility: hidden; }
    .block-container { padding-top: 1.6rem; max-width: 1100px; }
    h2, h3 { color: #0F3D4C; letter-spacing: -0.01em; }
    [data-testid="stSidebar"] { border-right: 1px solid #D5E0E4; }
    [data-testid="stSidebar"] h2 { font-size: 1.15rem; }
    [data-testid="stSidebar"] h3 { font-size: .95rem; color: #1F6F82; border-bottom: 1px solid #D5E0E4; padding-bottom: .3rem; margin-top: .8rem; }
    [data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #D5E0E4; border-radius: 10px; padding: .8rem 1rem; }
    button[role="tab"] { font-weight: 600; }

    .cabecera { background: #0F3D4C; border-radius: 12px; padding: 1.4rem 1.8rem; margin-bottom: 1.3rem;
                display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; }
    .cabecera .titulo { color: #FFFFFF; font-size: 1.7rem; font-weight: 700; line-height: 1.2; }
    .cabecera .subtitulo { color: #CFE3E8; font-size: .95rem; margin-top: .35rem; max-width: 46rem; }
    .cabecera .sello { border: 1px solid #6FA8B5; color: #DCEEF2; border-radius: 999px; padding: .25rem .85rem; font-size: .8rem; white-space: nowrap; }

    .resultado { background: #FFFFFF; border: 1px solid #D5E0E4; border-left: 6px solid var(--c); border-radius: 10px; padding: 1.3rem 1.6rem; }
    .resultado .fila { display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; }
    .resultado .nivel { font-size: 1.7rem; font-weight: 700; }
    .resultado .chip { color: #FFFFFF; font-weight: 600; font-size: .85rem; border-radius: 999px; padding: .3rem .9rem; }
    .resultado .prob { color: #3F5158; margin: .5rem 0 1.4rem; }
    .escala { position: relative; display: flex; height: 14px; }
    .escala .zona:first-child { border-radius: 7px 0 0 7px; }
    .escala .zona:nth-child(3) { border-radius: 0 7px 7px 0; }
    .escala .punto { position: absolute; top: 50%; width: 20px; height: 20px; border-radius: 50%; background: #0F3D4C;
                     border: 3px solid #FFFFFF; box-shadow: 0 0 0 1px #0F3D4C; transform: translate(-50%, -50%); }
    .leyenda { display: flex; margin-top: .55rem; font-size: .8rem; color: #3F5158; }
    .leyenda span { text-align: center; }

    .aviso { border-top: 1px solid #D5E0E4; margin-top: 1.5rem; padding-top: .8rem; font-size: .85rem; color: #3F5158; }
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
        return "Riesgo bajo", "#2E7D5B"
    if p < THRESHOLD:
        return "Riesgo intermedio", "#9A6412"
    return "Riesgo alto", "#B23A48"


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


def tarjeta_resultado(p: float, label: str):
    nivel, color = nivel_riesgo(p)
    a = THRESHOLD * 0.6 * 100
    b = THRESHOLD * 100
    st.markdown(
        f"""
<div class="resultado" style="--c:{color}">
<div class="fila"><div class="nivel" style="color:{color}">{nivel}</div><div class="chip" style="background:{color}">{label}</div></div>
<div class="prob">Probabilidad estimada: <b>{p:.1%}</b>. Umbral de decisión del modelo: {THRESHOLD:.2f}.</div>
<div class="escala">
<div class="zona" style="width:{a:.1f}%; background:#CFE8DC"></div>
<div class="zona" style="width:{b - a:.1f}%; background:#F3E2BD"></div>
<div class="zona" style="width:{100 - b:.1f}%; background:#F2CDD1"></div>
<div class="punto" style="left:{p * 100:.1f}%"></div>
</div>
<div class="leyenda"><span style="width:{a:.1f}%">Bajo</span><span style="width:{b - a:.1f}%">Intermedio</span><span style="width:{100 - b:.1f}%">Alto</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


# ==========================================
# ENCABEZADO
# ==========================================
st.markdown(
    """
    <div class="cabecera">
      <div>
        <div class="titulo">Riesgo de preeclampsia en gestantes</div>
        <div class="subtitulo">Estimación con un modelo de red neuronal a partir de nueve datos clínicos. Herramienta de apoyo a la decisión.</div>
      </div>
      <div class="sello">Prototipo académico</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# BARRA LATERAL: DATOS DE LA PACIENTE
# ==========================================
with st.sidebar:
    st.header("Datos de la paciente")
    with st.form("form_paciente"):
        st.subheader("Datos generales")
        edad = st.number_input("Edad (años)", 10, 60, 30, help="Entre 10 y 60 años.")
        imc = st.number_input("IMC (kg/m²)", 10.0, 60.0, 25.0, step=0.1, help="Índice de masa corporal.")
        creatinina = st.number_input("Creatinina (mg/dL)", 0.1, 10.0, 1.0, step=0.01, help="Entre 0.1 y 10 mg/dL.")

        st.subheader("Presión arterial")
        p_a_sistolica = st.number_input("Sistólica (mmHg)", 80, 200, 120, help="Entre 80 y 200 mmHg.")
        p_a_diastolica = st.number_input("Diastólica (mmHg)", 40, 130, 80, help="Entre 40 y 130 mmHg.")

        st.subheader("Antecedentes")
        hipertension = st.radio("Hipertensión previa", ["NO", "SI"], horizontal=True)
        diabetes = st.radio("Diabetes", ["NO", "SI"], horizontal=True)
        ant_fam_hiper = st.radio("Antecedentes familiares de hipertensión", ["NO", "SI"], horizontal=True)
        tec_repro_asistida = st.radio("Reproducción asistida", ["NO", "SI"], horizontal=True)

        enviado = st.form_submit_button("Calcular riesgo", type="primary", width="stretch")

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

# ==========================================
# VALIDACIÓN Y CÁLCULO (al enviar el formulario)
# ==========================================
if enviado:
    errores, avisos = [], []
    if p_a_sistolica <= p_a_diastolica:
        errores.append("La presión sistólica debe ser mayor que la diastólica.")
    if hipertension == "NO" and (p_a_sistolica >= 140 or p_a_diastolica >= 90):
        avisos.append(
            "Indicaste que no hay hipertensión previa, pero la presión ingresada está en rango hipertensivo "
            "(≥140/90 mmHg). Revisa que los datos sean correctos."
        )
    st.session_state["errores"] = errores
    st.session_state["avisos"] = avisos
    if errores:
        st.session_state.pop("resultado", None)
    else:
        res = predict_batch(registro)[0]
        st.session_state["resultado"] = res
        st.session_state["registro"] = dict(registro)
        st.session_state["sens"] = sensibilidad(registro, res["proba"])

# ==========================================
# PESTAÑAS
# ==========================================
tab_pred, tab_lote, tab_modelo = st.tabs(["Predicción", "Carga por lote", "Acerca del modelo"])

# ---------- Pestaña 1: predicción individual ----------
with tab_pred:
    for e in st.session_state.get("errores", []):
        st.error(e)
    for a in st.session_state.get("avisos", []):
        st.warning(a)

    if "resultado" not in st.session_state:
        if not st.session_state.get("errores"):
            st.info("Completa los datos en la barra lateral y presiona **Calcular riesgo**.")
    else:
        res = st.session_state["resultado"]
        tarjeta_resultado(res["proba"], res["pred_label"])

        st.write("")
        st.subheader("¿Qué datos mueven más la estimación?")
        st.caption(
            "Se cambia un dato a la vez (los numéricos +10 %, los de SI/NO al valor contrario) "
            "y se mide cuánto varía la probabilidad para esta paciente. Es una simulación, no prueba causalidad."
        )
        sens = st.session_state["sens"]
        st.dataframe(
            sens,
            hide_index=True,
            width="stretch",
            column_config={
                "Efecto (puntos %)": st.column_config.NumberColumn(format="%+.1f"),
            },
        )

        with st.expander("Datos ingresados"):
            r = st.session_state["registro"]
            st.dataframe(
                pd.DataFrame({"Dato": [ETIQUETAS[k] for k in r], "Valor": [str(v) for v in r.values()]}),
                hide_index=True,
                width="stretch",
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