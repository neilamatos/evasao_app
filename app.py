import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import json


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
        df = pd.read_csv(arquivo,
                         sep=';',
                         encoding='cp1252',
                         encoding_errors='replace')
    else:
        df = pd.read_excel(arquivo)

    st.success(f"Arquivo carregado com sucesso! ({df.shape[0]} registros)")

    # Remove as colunas desejadas
    df = df.drop(columns=[
        'bairro',
        'Bairro_Pais',
        'Bairro_1',
        'Cod_Distrito',
        'Necessidade_Fisica',
        'Necessidade_Visual',
        'Necessidade_Auditiva',
        'Necessidade_Mental',
        'Necessidade_Multipla',
        'Outras_Necessidades',
        'Superdotado',
        'Condutas_Tipicas',
        'Sindrome_Down',
        'necessidades_especiais',
        'clNecessidades',
        'Cod_Responsavel',
        'Id_Grupo_Etnico',
        'Desc_Grupo_Etnico',
        'Documento_Estrangeiro',
        'Numero_Cns',
        '__Observacoes',
        '__Tratamento',
        'Renda_Per_Capita_Inep',
        'Desc_Renda_Per_Capita_INEP',
        'BOLSA_FAMILIA',
        'BOLSA_ESCOLA',
        'Dt_Limite_Interesse_Emprego',
        'DESC_EAD_POLO',
        'DT_COLACAO_GRAU',
        'clApoios_Sociais'
    ])

    # Criar coluna alvo: 'Status' conforme a regra:
    # 1 → Concluído
    # 0 → Abandono, Cancelado Compulsoriamente, Cancelado Voluntariamente
    df["Status"] = df["Desc_Sit_Matricula"].apply(
        lambda x: 1 if str(x).strip().lower() == "concluído" else
                (0 if str(x).strip().lower() in [
                    "abandono",
                    "cancelado compulsoriamente",
                    "cancelado voluntariamente"
                ] else None)
    )

    #### Primeiro criar as novas features ####

    # Converte a coluna de data de nascimento para datetime
    df["Dt_Nascimento"] = pd.to_datetime(df["Dt_Nascimento"], errors="coerce", dayfirst=True)

    # Garante que ano_letivo_ini é numérico (caso tenha vindo como texto)
    df["ano_letivo_ini"] = pd.to_numeric(df["ano_letivo_ini"], errors="coerce")

    # Cria a nova coluna "Idade" (idade da pessoa no início do ano letivo)
    df["Idade"] = df["ano_letivo_ini"] - df["Dt_Nascimento"].dt.year

    # Garante que as colunas estão como números (caso estejam como texto)
    df["ano_letivo_ini"] = pd.to_numeric(df["ano_letivo_ini"], errors="coerce")

    df["ano_letivo_ini"] = pd.to_numeric(df["ano_letivo_ini"], errors="coerce")

    # Cria nova coluna com o cálculo
    df["Anos_Apos_Graduacao"] = df["ano_letivo_ini"] - df["Ano_Conclusao_Graduacao"]

    # --- Criar novas features baseadas em cod_pessoa ---

    # 1️⃣ Quantidade de matrículas por pessoa
    contagem = df.groupby("Cod_Pessoa")["cod_matricula"].nunique().reset_index()
    contagem.rename(columns={"cod_matricula": "n_matriculas_pessoa"}, inplace=True)

    # 2️⃣ Mesclar com o dataset original
    df = df.merge(contagem, on="Cod_Pessoa", how="left")

    # 3️⃣ Criar variações derivadas
    df["primeira_matricula"] = df.groupby("Cod_Pessoa")["cod_matricula"].transform(
        lambda x: (x == x.min()).astype(int)
    )
    df["reingresso"] = (df["n_matriculas_pessoa"] > 1).astype(int)
    df["n_matriculas_maior_que_1"] = df["n_matriculas_pessoa"].apply(lambda x: 1 if x > 1 else 0)

    ### Antes de remover as colunas preciso criar o df_novos
    df_novos = df.loc[pd.isna(df["Status"]), :].copy()

    # colunas de identificação (não entram no modelo)
    id_cols = ["Cod_Pessoa", "cod_matricula"]

    # guardar identificadores
    ids = df_novos[id_cols].copy()

    df_novos.to_csv("novos_alunos.csv", index=False, encoding="utf-8-sig")

    df = df_novos.copy()

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
    probs = pipeline.predict_proba(X)[:, 0]
    preds = pipeline.predict(X)

    with open("thresholds_xgb.json", "r") as f:
        thresholds = json.load(f)

    THRESHOLD_MEDIO = thresholds["threshold_medio"]
    THRESHOLD_ALTO = thresholds["threshold_alto"]

    # ===============================
    # RESULTADO FINAL
    # ===============================
    df_resultado = df_ids.copy()
    df_resultado["prob_evasao"] = probs
    df_resultado["pred_evasao"] = preds

    def classificar_risco(p):
        if p >= THRESHOLD_ALTO:
            return "Alto risco"
        elif p >= THRESHOLD_MEDIO:
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
