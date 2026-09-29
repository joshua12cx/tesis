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
    .resultado { border: 1px solid #C9D6DA; border-radius: 10px; padding: 1.2rem 1.4rem; background: #FFFFFF; }
    .resultado .nivel { font-size: 1.6rem; font-weight: 700; }
    .resultado .prob { font-size: 1rem; color: #4A5B61; margin-bottom: .9rem; }
    .pista { position: relative; height: 16px; background: #E3EAEC; border-radius: 8px; overflow: visible; }
    .relleno { height: 100%; border-radius: 8px; }
    .umbral { position: absolute; top: -5px; width: 2px; height: 26px; background: #1B2A30; }
    .escala { position: relative; height: 22px; margin-top: 8px; font-size: .8rem; color: #4A5B61; }
    .escala span { position: absolute; transform: translateX(-50%); white-space: nowrap; }
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
        return "Riesgo intermedio", "#B7791F"
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
    umbral_pct = THRESHOLD * 100
    st.markdown(
        f"""
        <div class="resultado">
          <div class="nivel" style="color:{color}">{nivel}</div>
          <div class="prob">Probabilidad estimada: <b>{p:.1%}</b> · El modelo clasifica: <b>{label}</b></div>
          <div class="pista">
            <div class="relleno" style="width:{p*100:.1f}%; background:{color}"></div>
            <div class="umbral" style="left:{umbral_pct:.1f}%"></div>
          </div>
          <div class="escala">
            <span style="left:0%">0%</span>
            <span style="left:{umbral_pct:.1f}%">umbral {umbral_pct:.0f}%</span>
            <span style="left:100%">100%</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# ENCABEZADO
# ==========================================
st.title("Riesgo de preeclampsia en gestantes")
st.write(
    "Estimación del riesgo con un modelo de red neuronal (MLP) a partir de nueve datos clínicos. "
    "Es un prototipo académico de apoyo a la decisión."
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

    st.caption("Este sistema es solo apoyo a la decisión clínica y no reemplaza el criterio médico.")

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
