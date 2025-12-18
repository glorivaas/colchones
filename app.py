import json, re, ast
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sentence_transformers import SentenceTransformer

CLUSTER_TITLES = {
    0: "🟦 Confort y firmeza (generalista)",
    1: "🟥 No válido / ruido ",
    2: "🟩 Muelles + visco (tecnología núcleo)",
    3: "🟨 Descanso ideal / adaptabilidad (marketing)",
    4: "🟪 Premium / lujo (claims en inglés)",
}

CLUSTER_DESC = {
    0: "Claims sobre calidad, soporte, firmeza y confort.",
    1: "Textos vacíos, incoherentes o de otro idioma.",
    2: "Énfasis en muelles ensacados, viscoelástica, zonas, núcleo.",
    3: "Claims genéricos de descanso, adaptabilidad y bienestar.",
    4: "Claims de gama alta (a menudo en inglés).",
}


def _to_python_obj(item):
    if isinstance(item, dict):
        return item

    if isinstance(item, str):
        s = item.strip()

        try:
            return json.loads(s)
        except Exception:
            pass

        s2 = re.sub(r"\bNone\b", "null", s)
        s2 = re.sub(r"\bTrue\b", "true", s2)
        s2 = re.sub(r"\bFalse\b", "false", s2)

        try:
            return json.loads(s2)
        except Exception:
            pass

        try:
            obj = ast.literal_eval(s)
            return obj
        except Exception as e:
            raise ValueError(f"Texto no es JSON válido: {e}")

    raise TypeError("item debe ser dict o str(JSON)")

def _binarize_salud_higiene(d, max_k=8):
    """
    Convierte 'salud e higiene': [3,4] -> salud_higiene_1..8 = 0/1
    """
    salud = d.get("salud e higiene") or []

    if isinstance(salud, str):
        try:
            salud = json.loads(salud)
        except Exception:
            salud = []

    if not isinstance(salud, (list, tuple, set)):
        salud = []

    salud_set = set()
    for x in salud:
        try:
            salud_set.add(int(x))
        except Exception:
            pass

    for k in range(1, max_k + 1):
        d[f"salud_higiene_{k}"] = 1 if k in salud_set else 0

    d.pop("salud e higiene", None)
    return d

def predict_cluster(d, embedder, kmeans, text_fields=("claim",)):
    texts = []
    for f in text_fields:
        val = d.get(f)
        if val is not None and str(val).strip() != "":
            texts.append(str(val))

    text = " ".join(texts).strip()
    if text == "":
        return None

    emb = embedder.encode([text])        
    cl = int(kmeans.predict(emb)[0])
    return cl

def build_features_df(item, features, embedder=None, kmeans=None):
    d = _to_python_obj(item)

    if embedder is not None and kmeans is not None:
        cl = predict_cluster(d, embedder=embedder, kmeans=kmeans, text_fields=("claim",))
        if cl is not None:
            d["cluster"] = int(cl)

    d = _binarize_salud_higiene(d, max_k=8)

    df = pd.DataFrame([d]).replace({"": np.nan})
    X_one = df.reindex(columns=features, fill_value=np.nan)

    # asegurar cluster int si existe
    if "cluster" in X_one.columns and pd.notna(X_one.loc[0, "cluster"]):
        try:
            X_one.loc[0, "cluster"] = int(float(X_one.loc[0, "cluster"]))
        except Exception:
            pass

    return X_one

def get_cluster_from_Xone(X_one):
    if "cluster" not in X_one.columns:
        return None
    try:
        v = X_one["cluster"].iloc[0]
        if pd.isna(v):
            return None
        return int(float(v))
    except Exception:
        return None


def predict_price(item, model, features, trained_on_log=True, clip_to_train=None,
                  embedder=None, kmeans=None):
    """
    Predice precio (y calcula cluster si hay vectorizer+kmeans).
    """
    X_one = build_features_df(
        item=item,
        features=features,
        embedder = embedder,
        kmeans=kmeans,
    )

    pred_val = float(model.predict(X_one)[0])

    if trained_on_log:
        if clip_to_train is not None:
            low, high = clip_to_train
            pred_val = float(np.clip(pred_val, low, high))
        price_eur = float(np.expm1(pred_val))
        return price_eur, pred_val, X_one
    else:
        return pred_val, None, X_one

def predict_with_interval(item, model, features, embedder, kmeans, calibrator):
    # Predicción puntual
    price_eur, pred_log, X_one = predict_price(
        item=item,
        model=model,
        features=features,
        trained_on_log=True,  
        clip_to_train=None,   
        embedder=embedder,
        kmeans=kmeans
    )

    # Intervalo en LOG = pred_log + quantiles de residuo
    low_log = pred_log + calibrator["q_low"]
    high_log = pred_log + calibrator["q_high"]

    # Pasar a euros
    low_eur = float(np.expm1(low_log))
    high_eur = float(np.expm1(high_log))

    return price_eur, low_eur, high_eur, pred_log, X_one

import plotly.graph_objects as go


def interval_plot(price, low, high):
    fig = go.Figure()

    # barra del rango
    fig.add_trace(go.Scatter(
        x=[low, high],
        y=[0, 0],
        mode="lines",
        line=dict(width=12),
        name="Rango"
    ))

    # punto de predicción
    fig.add_trace(go.Scatter(
        x=[price],
        y=[0],
        mode="markers",
        marker=dict(size=14),
        name="Estimación"
    ))

    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=140,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis_title="Precio (€)"
    )
    return fig

def get_similars(text, embedder, comp_meta, comp_E, top_k=5, require_price=True, diversify_by="marca comercial"):
    # 1) embedding query
    v = embedder.encode([text]).astype("float32")[0]
    v = v / (np.linalg.norm(v) + 1e-12)

    # 2) cosine similarity
    sims = comp_E @ v  # (N,)

    # 3) máscara para filtrar antes de ordenar (evita out-of-bounds)
    mask = np.ones(len(comp_meta), dtype=bool)
    if require_price and ("precio de venta público" in comp_meta.columns):
        mask &= comp_meta["precio de venta público"].notna().values

    sims_f = sims[mask]
    meta_f = comp_meta.loc[mask].reset_index(drop=True)

    if len(meta_f) == 0:
        return meta_f  # vacío

    # 4) ordenar por similitud
    order = np.argsort(-sims_f)
    meta_f = meta_f.iloc[order].copy()
    meta_f["similarity"] = sims_f[order]

    # 5) Diversificación: evita “todo igual” (por marca, o por cluster)
    if diversify_by is not None and diversify_by in meta_f.columns:
        meta_f = meta_f.drop_duplicates(subset=[diversify_by], keep="first")

    return meta_f.head(top_k)


def explain_price_ablation(item, model, features, embedder, kmeans, trained_on_log=True):
    """
    Explicación local: impacto aproximado por variable/grupo
    = pred(base) - pred(sin esa variable)
    """

    base_price, base_log, X_base = predict_price(
        item=item, model=model, features=features, trained_on_log=trained_on_log,
        clip_to_train=None, embedder=embedder, kmeans=kmeans
    )

    d = _to_python_obj(item)

    groups = {
        "Marca": ["marca comercial"],
        "Firmeza": ["firmeza"],
        "Transpiración": ["transpiracion"],
        "Materiales principales": ["materiales principales"],
        "Tipo de núcleo": ["tipo de núcleo"],
        "Dimensiones": ["dimensiones"],
        "Fabricante": ["fabricante"],
        "Salud e higiene (flags)": ["salud e higiene"], 
        "Regulación térmica": ["regulación térmica"],
        "Capas confort": ["capas de confort y acolchados"],
        "Plazo entrega": ["plazo_de_entrega", "plazo de entrega"],  
        "Días de prueba": ["días de prueba"],
        "Garantía": ["garantia", "garantía"],
    }

    rows = []

    for gname, keys in groups.items():
        d2 = dict(d)  
        for k in keys:
            if k in d2:
                d2[k] = None

        try:
            p2, log2, _ = predict_price(
                item=d2, model=model, features=features, trained_on_log=trained_on_log,
                clip_to_train=None, embedder=embedder, kmeans=kmeans
            )
            delta = base_price - p2  # >0 => esta variable sube precio
            rows.append((gname, delta, p2))
        except Exception:
            pass

    exp = pd.DataFrame(rows, columns=["variable", "impacto_eur", "precio_sin_variable"])
    exp = exp.sort_values("impacto_eur", ascending=False)
    return base_price, exp

# Streamlit App

st.set_page_config(page_title="Demo Colchones | Predicción de precio", layout="centered")
st.title("Demo interactiva: estimación de precio de colchones")
st.write("Pega un elemento del dataset en JSON (puede tener `null`) y la app calcula cluster + precio.")

@st.cache_resource
def load_artifacts():
    model = joblib.load("modelo_gbr.joblib")
    features = joblib.load("features.joblib")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    kmeans = joblib.load("kmeans.joblib")
    calibrator = joblib.load("calibrator_interval.joblib")
    return model, features, kmeans, embedder, calibrator

def load_comparables():
    meta = pd.read_parquet("comparables_meta.parquet")
    data = np.load("comparables_embeddings.npz")
    E = data["E"]  # (N, D)
    # normaliza para cosine similarity rápida
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return meta, E

comp_meta, comp_E = load_comparables()

model, features, kmeans, embedder, calibrator = load_artifacts()

example_json = """{
  "nombre":"Junior Visco",
  "coleccion":null,
  "marca comercial":"Star",
  "claim":"Colchón juvenil ideal para camas convertibles, nidos, etc.",
  "materiales principales":8.0,
  "tipo de núcleo":"Active Air",
  "firmeza":2.0,
  "transpiracion":4.0,
  "regulación térmica":"Fiber Therm que permite regular la temperatura",
  "precio de venta público":null,
  "salud e higiene":[3,4],
  "capas de confort y acolchados":null,
  "dimensiones":"Individual",
  "fabricante":"Colchon Star",
  "plazo_de_entrega":2.0
}"""

json_text = st.text_area("JSON del colchón", value=example_json, height=260)

trained_on_log = True

use_clip = True
clip_low = 0.0
clip_high = 16.5
show_interval = st.checkbox("Mostrar intervalo de precio (rango de confianza)", value=True)
show_similars = st.checkbox("Mostrar comparables", value=True)
show_explain = st.checkbox("Mostrar explicación del precio", value=True)


if st.button("Predecir"):
    try:
        # 1) Construir features UNA sola vez
        X_one = build_features_df(
            json_text,
            features,
            embedder=embedder,
            kmeans=kmeans
        )

        cluster_val = get_cluster_from_Xone(X_one)

        # 2) Predecir (SIEMPRE usando X_one que es 2D)
        pred_log = float(model.predict(X_one)[0])  # model entrenado en log

        precio = float(np.expm1(pred_log))
        st.success(f"💰 Precio estimado: {precio:,.0f} €")

        # 3) Mostrar cluster
        if cluster_val is not None:
            titulo = CLUSTER_TITLES.get(cluster_val, f"Cluster {cluster_val}")
            desc = CLUSTER_DESC.get(cluster_val, "")
            st.subheader(f"🧠 Cluster semántico: {titulo}")
            if desc:
                st.caption(desc)
        else:
            st.warning("No se pudo calcular el cluster (claim vacío).")

        st.caption(f"Predicción interna (log): {pred_log:.4f}")

        # 4) Intervalo (si lo quieres) -> que use X_one también
        if show_interval:
            precio, low, high, pred_log, X_one = predict_with_interval(
                item=json_text,
                model=model,
                features=features,
                embedder=embedder,
                kmeans=kmeans,
                calibrator=calibrator
            )
            st.write(f"📍 Intervalo {calibrator['level']}: **{low:,.0f} € – {high:,.0f} €**")
            st.plotly_chart(interval_plot(precio, low, high), width='stretch')

        # 5) Explicación (ideal: que acepte X_one para no recalcular)
        if show_explain:
            base_price, exp = explain_price_ablation(
                item=json_text,
                model=model,
                features=features,
                embedder=embedder,
                kmeans=kmeans
            )
            exp = exp.dropna(subset=["impacto_eur"]).copy()
            up = exp.sort_values("impacto_eur", ascending=False).head(5)
            down = exp.sort_values("impacto_eur", ascending=True).head(5)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**⬆️ Lo que más SUBE el precio (según el modelo):**")
                st.dataframe(up[["variable", "impacto_eur"]])
            with c2:
                st.markdown("**⬇️ Lo que más BAJA el precio (según el modelo):**")
                st.dataframe(down[["variable", "impacto_eur"]])

        with st.expander("Ver features utilizadas (1 fila)"):
            st.dataframe(X_one)

        # 6) Similares (esto no toca X_one)
        if show_similars:
            claim_txt = _to_python_obj(json_text).get("claim")
            if claim_txt and str(claim_txt).strip():
                st.subheader("Productos comparables (por similitud semántica)")
                sims_df = get_similars(
                    str(claim_txt),
                    embedder=embedder,
                    comp_meta=comp_meta,
                    comp_E=comp_E,
                    top_k=8,
                    require_price=True,
                    diversify_by="marca comercial",
                )
                if sims_df.empty:
                    st.info("No encontré comparables con precio disponible.")
                else:
                    show_cols = ["nombre", "marca comercial", "precio de venta público", "cluster", "similarity", "url"]
                    show_cols = [c for c in show_cols if c in sims_df.columns]
                    if "precio de venta público" in sims_df.columns:
                        sims_df = sims_df.copy()
                        sims_df["precio de venta público"] = sims_df["precio de venta público"].map(
                            lambda x: f"{x:,.0f} €" if pd.notna(x) else ""
                        )
                    st.dataframe(sims_df[show_cols], use_container_width=True)
            else:
                st.info("No hay `claim` para buscar comparables.")

    except Exception as e:
        st.error(f"Error procesando el JSON o prediciendo: {e}")
