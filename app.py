import streamlit as st
import pandas as pd
import joblib

# ===============================
# CONFIGURAÇÃO INICIAL
# ===============================
st.set_page_config(
    page_title="Predição de Evasão Acadêmica",
    layout="wide"
)

st.title("📊 Predição de Risco de Evasão Acadêmica")
st.write(
    "Este aplicativo aplica um modelo preditivo para estimar o risco "
    "de evasão de estudantes matriculados."
)

# ===============================
# CARREGAR ARTEFATOS
# ===============================
@st.cache_resource
def carregar_artefatos():
    pipeline = joblib.load("pipeline_cenario170.pkl")
    features = joblib.load("features_originais_cenario170.pkl")
    dicionarios = joblib.load("mapeamentos_categoricos.pkl")
    return pipeline, features, dicionarios

pipeline, features, dicionarios = carregar_artefatos()

# ===============================
# UPLOAD DOS DADOS
# ===============================
st.header("📂 Upload dos dados dos alunos")

arquivo = st.file_uploader(
    "Envie um arquivo CSV ou Excel com os dados dos alunos matriculados",
    type=["csv", "xlsx"]
)

if arquivo:
    if arquivo.name.endswith(".csv"):
        df = pd.read_csv(arquivo)
    else:
        df = pd.read_excel(arquivo)

    st.success(f"Arquivo carregado com sucesso! ({df.shape[0]} registros)")

    # ===============================
    # PRESERVAR IDENTIFICADORES
    # ===============================
    id_cols = [c for c in ["Cod_Pessoa", "cod_matricula"] if c in df.columns]
    df_ids = df[id_cols].copy()

    # ===============================
    # APLICAR MAPEAMENTO CATEGÓRICO
    # ===============================
    for col, mapping in dicionarios.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # ===============================
    # ALINHAR FEATURES
    # ===============================
    X = df.reindex(columns=features, fill_value=0)

    # ===============================
    # PREDIÇÃO
    # ===============================
    probs = pipeline.predict_proba(X)[:, 1]
    preds = pipeline.predict(X)

    # ===============================
    # RESULTADO FINAL
    # ===============================
    df_resultado = df_ids.copy()
    df_resultado["prob_evasao"] = probs
    df_resultado["pred_evasao"] = preds

    def classificar_risco(p):
        if p >= 0.75:
            return "Alto risco"
        elif p >= 0.50:
            return "Risco médio"
        else:
            return "Baixo risco"

    df_resultado["nivel_risco"] = df_resultado["prob_evasao"].apply(classificar_risco)

    # ===============================
    # RESULTADOS
    # ===============================
    st.header("📊 Painel de Resultados")

    # KPIs
    total = len(df_resultado)
    alto = (df_resultado["nivel_risco"] == "Alto risco").sum()
    medio = (df_resultado["nivel_risco"] == "Risco médio").sum()
    baixo = (df_resultado["nivel_risco"] == "Baixo risco").sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total de estudantes", total)
    col2.metric("🔴 Alto risco", alto)
    col3.metric("🟡 Médio risco", medio)
    col4.metric("🟢 Baixo risco", baixo)

    st.divider()

    # ===============================
    # FILTRO
    # ===============================
    st.subheader("🔎 Filtros")

    filtro = st.multiselect(
        "Selecione o nível de risco",
        options=["Alto risco", "Risco médio", "Baixo risco"],
        default=["Alto risco", "Risco médio", "Baixo risco"]
    )

    df_filtrado = df_resultado[df_resultado["nivel_risco"].isin(filtro)]

    # ===============================
    # GRÁFICO
    # ===============================
    st.subheader("📈 Distribuição de risco")
    st.bar_chart(df_resultado["nivel_risco"].value_counts())

    # ===============================
    # TOP RISCO
    # ===============================
    st.subheader("🚨 Estudantes com maior risco")

    top_n = st.slider("Quantidade de estudantes", 5, 50, 10)

    st.dataframe(
        df_resultado.sort_values("prob_evasao", ascending=False).head(top_n)
    )

    # ===============================
    # TABELA COMPLETA COM ESTILO
    # ===============================
    st.subheader("📋 Tabela completa")

    def color_risco(val):
        if val == "Alto risco":
            return "background-color: #ff4d4d"
        elif val == "Risco médio":
            return "background-color: #ffd966"
        else:
            return "background-color: #66ff66"

    styled_df = df_filtrado.sort_values(
        "prob_evasao", ascending=False
    ).style.applymap(color_risco, subset=["nivel_risco"])

    st.dataframe(styled_df)

    # ===============================
    # DOWNLOAD
    # ===============================
    csv = df_filtrado.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "⬇️ Baixar dados filtrados",
        data=csv,
        file_name="resultado_predicao_evasao.csv",
        mime="text/csv"
    )

    # # ===============================
    # # VISUALIZAÇÃO
    # # ===============================
    # st.header("📈 Resultados")

    # col1, col2 = st.columns(2)

    # with col1:
    #     st.subheader("Distribuição por nível de risco")
    #     st.bar_chart(df_resultado["nivel_risco"].value_counts())

    # with col2:
    #     st.subheader("Estatísticas das probabilidades")
    #     st.write(df_resultado["prob_evasao"].describe())

    # st.subheader("🔎 Visualização dos estudantes")
    # st.dataframe(df_resultado.sort_values("prob_evasao", ascending=False))

    # # ===============================
    # # DOWNLOAD
    # # ===============================
    # csv = df_resultado.to_csv(index=False).encode("utf-8-sig")
    # st.download_button(
    #     "⬇️ Baixar resultado em CSV",
    #     data=csv,
    #     file_name="resultado_predicao_evasao.csv",
    #     mime="text/csv"
    # )
