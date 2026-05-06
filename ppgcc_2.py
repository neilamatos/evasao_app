### Carrega bibliotecas e lê arquivo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv("dados_originais.csv")

# Exibe informações básicas
print("\n--- Informações do arquivo ---")
print(f"Quantidade de linhas: {df.shape[0]}")
print(f"Quantidade de colunas: {df.shape[1]}")

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

print("\n--- Informações do arquivo ---")
print(f"Quantidade de linhas: {df.shape[0]}")
print(f"Quantidade de colunas: {df.shape[1]}")

#### Primeiro criar as novas features ####

# Converte a coluna de data de nascimento para datetime
df["Dt_Nascimento"] = pd.to_datetime(df["Dt_Nascimento"], errors="coerce", dayfirst=True)

# Garante que ano_letivo_ini é numérico (caso tenha vindo como texto)
df["ano_letivo_ini"] = pd.to_numeric(df["ano_letivo_ini"], errors="coerce")

# Cria a nova coluna "Idade" (idade da pessoa no início do ano letivo)
df["Idade"] = df["ano_letivo_ini"] - df["Dt_Nascimento"].dt.year

# Mostra as primeiras linhas para conferir
print(df[["Dt_Nascimento", "ano_letivo_ini", "Idade"]])

# Garante que as colunas estão como números (caso estejam como texto)
df["ano_letivo_ini"] = pd.to_numeric(df["ano_letivo_ini"], errors="coerce")

df["ano_letivo_ini"] = pd.to_numeric(df["ano_letivo_ini"], errors="coerce")

# Cria nova coluna com o cálculo
df["Anos_Apos_Graduacao"] = df["ano_letivo_ini"] - df["Ano_Conclusao_Graduacao"]

# Mostra as primeiras linhas para conferir
print(df[["Ano_Conclusao_Graduacao", "ano_letivo_ini", "Anos_Apos_Graduacao"]].head())

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


print(f"Colunas até aqui: {df.shape[1]}")
print('Novas colunas:', list(df.columns[97:]))

print(df[["Cod_Pessoa", 'n_matriculas_pessoa', "primeira_matricula", "reingresso", "n_matriculas_maior_que_1"]])

### Antes de remover as colunas preciso criar o df_novos

df_novos = df.loc[pd.isna(df["Status"]), :].copy()

# colunas de identificação (não entram no modelo)
id_cols = ["Cod_Pessoa", "cod_matricula"]

# guardar identificadores
ids = df_novos[id_cols].copy()


df_novos.to_csv("novos_alunos.csv", index=False, encoding="utf-8-sig")

#### Remoção de features

# Remove colunas com mesmo valor em todas as linhas
colunas_constantes = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
df = df.drop(columns=colunas_constantes)

# Remove colunas diretamente relacionadas ao alvo (não podem ir para X)
colunas_target_like = [
    "Desc_Sit_Matricula", "Desc_Matricula", "Sit_Matricula",
    "Situacao_Ultimo_Periodo_Letivo", "DESC_SIT_MATRICULA_PERIODO",
    "__DIPLOMA_PASTA", "__DIPLOMA_N_REGISTRO",
    "__DIPLOMA_N_LIVRO_N_FOLHA", "N_PASTA",
    "__Dt_Conclusao_Curso", "Ultimo_Evento_Matricula"
]
df = df.drop(columns=[c for c in colunas_target_like if c in df.columns])

# Remover colunas com menos de 50% preenchidas
percentual_preenchimento = df.notnull().mean() * 100
colunas_validas = percentual_preenchimento[percentual_preenchimento >= 50].index
df = df[colunas_validas]

import pandas as pd

def encontrar_colunas_duplicadas(df):
    """
    Encontra grupos de colunas que possuem exatamente os mesmos valores,
    mesmo que com nomes diferentes.
    Retorna uma lista de listas com os nomes das colunas duplicadas.
    """
    duplicadas = []
    colunas = df.columns.tolist()
    visitadas = set()

    for i in range(len(colunas)):
        col_i = colunas[i]
        if col_i in visitadas:
            continue

        grupo = [col_i]  # sempre começa com a própria coluna

        for j in range(i+1, len(colunas)):
            col_j = colunas[j]

            # compara valores linha a linha
            if df[col_i].equals(df[col_j]):
                grupo.append(col_j)
                visitadas.add(col_j)

        if len(grupo) > 1:
            duplicadas.append(grupo)
            visitadas.add(col_i)

    return duplicadas

duplicadas = encontrar_colunas_duplicadas(df)

print("Grupos de colunas duplicadas:")
for grupo in duplicadas:
    print(grupo)


#Há pares simples de duplicação
#Como:
#Cod_Pessoa ↔ Cod_Pessoa_1
#sexo ↔ Sexo_1
#Dt_Nascimento ↔ Dt_Nascimento_1
#Nesses casos, manter apenas uma coluna.

# Remove colunas diretamente relacionadas ao alvo (não podem ir para X)
colunas_duplicadas = [
    "Cod_Pessoa_1", "Ano_Conclusao_2_Grau", "Dt_Nascimento_1", "Tipo_Escola_Origem_1",
    "Sexo_1", "Cod_Cidade_1", "DT_Rematricula"
]
df = df.drop(columns=[c for c in colunas_duplicadas if c in df.columns])

#Remover Desc_Forma_Ingresso_Matricula, coluna deveria ser preenchida apenas com Seleção Pós-Graduação
coluna_forma_ingresso = ["Desc_Forma_Ingresso_Matricula"]
df = df.drop(columns=[c for c in coluna_forma_ingresso if c in df.columns])

df.info()

def coluna_entrega_status(df, col):
    """
    Retorna True se a coluna 'col' determina completamente o Status.
    Ou seja, se para cada valor de col sempre existe apenas um Status.
    """
    tab = df.groupby(col)["Status"].nunique()
    return (tab <= 1).all()  # nunca aparece mais de 1 status por categoria

# Rodar a função coluna_entrega_status para tirar mais vazadoras
vazando = []
for col in df.columns:
    if col != "Status" and coluna_entrega_status(df, col):
        vazando.append(col)

df = df.drop(columns=vazando)
print("Colunas removidas por vazamento direto:", vazando)

colunas_pos_ingresso = ['Ultimo_Periodo_Letivo_Presente', 'Ultima_Aula_Presente', 'Ultimo_Periodo_Letivo_Presente', 'Ano_Let_Atual', 'Periodo_Let_Atual', 'Periodo_Atual', 'Coeficiente_Rendimento']
colunas_cod = ['Cod_Pessoa', 'Cod_Aluno', 'Cod_Escola_2_Grau', 'Cod_cidade', 'Cod_Nacionalidade', 'Cod_Naturalidade', 'Cod_Turno', 'Cod_Estado_Civil']
datas_ingresso = ['ano_letivo_ini', 'dt_matricula', 'Dt_Cadastro', 'clPeriodo_Let_ini']

df = df.drop(columns=[c for c in colunas_pos_ingresso + colunas_cod + datas_ingresso if c in df.columns])
df = df.drop(columns=[c for c in colunas_cod if c in df.columns])

colunas = df.columns
print(len(colunas))
print(colunas)

df.info()

##RAQUEL: Retirar as linhas com status NaN
df = df[pd.notna(df['Status'])]

# # Verificar resultado
print(df["Status"].value_counts(dropna=False))

#Realizar esse código antes de dividir, aplicar ao df, conjunto completo
# --- Identificar colunas não numéricas ---
colunas_nao_numericas = df.select_dtypes(exclude=["number"]).columns
print("\nColunas não numéricas encontradas:")
print(list(colunas_nao_numericas))

##RAQUEL: Fazer o mapeamento antes de dividir os dados de treino e teste
# --- Passo 4: Criar dicionários de mapeamento e transformar ---
dicionarios = {}

for col in colunas_nao_numericas:
    categorias_unicas = df[col].dropna().unique()
    mapeamento = {cat: idx for idx, cat in enumerate(categorias_unicas)}
    dicionarios[col] = mapeamento
    df[col] = df[col].map(mapeamento)

# --- Passo 5: Mostrar resultados ---
print("\nDicionário de dados criado (coluna -> mapeamento):")
for col, mapping in dicionarios.items():
    print(f"\n{col}:")
    print(mapping)

print("\nPrévia do DataFrame transformado:")
print(df.head())

# --- Passo 6 (opcional): salvar novo CSV ---
df.to_csv("dados_ppgcc_convertidos.csv", index=False)

# Salvando os dicionários
import joblib

joblib.dump(dicionarios, "mapeamentos_categoricos.pkl")

import seaborn as sb
import math

bloco = 10
features = list(df.columns)
num_blocos = math.ceil(len(features) / bloco)

for i in range(num_blocos):
    inicio = i * bloco
    fim = min((i+1) * bloco, len(features))
    subset = list(dict.fromkeys(features[inicio:fim] + ["Status"]))
    print(inicio, '-', fim, '->', subset)

    sb.pairplot(df[subset], hue="Status", height=2)
    plt.show()

##RAQUEL: Melhor usar uma divisão stratificada dos dados, conforme o status. Mas pode deixar para fazer a divisão de treino e teste somente depois que todo o pré-processamento dos dados for feito
from sklearn.model_selection import train_test_split

X = df.drop(columns=["Status"])
y = df["Status"]

print('Qtde de colunas: ', len(list(df.columns)), df.columns)
print('Qtde de linhas: ', len(df))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#PREPARO COMUM (rodar uma vez)
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np

#Cenários de Imputação

# CENÁRIO 1 – NaN -> ZERO
X_train_zero = X_train.fillna(0)
X_test_zero  = X_test.fillna(0)

# CENÁRIO 2 – NaN -> MENOS UM
X_train_menos_um = X_train.copy()
X_test_menos_um = X_test.copy()

X_train_menos_um = X_train.fillna(-1)
X_test_menos_um = X_test.fillna(-1)

# CENÁRIO 2 – NaN -> MÉDIA
mean_vals = X_train.mean(numeric_only=True)
X_train_mean = X_train.fillna(mean_vals).fillna(0)
X_test_mean  = X_test.fillna(mean_vals).fillna(0)

# CENÁRIO 3 – NaN -> MODA
mode_vals = X_train.mode().iloc[0]
X_train_mode = X_train.fillna(mode_vals)
X_test_mode  = X_test.fillna(mode_vals)

cenarios = {
    "zero":  (X_train_zero,  X_test_zero),
    "menos_um": (X_train_menos_um,  X_test_menos_um),
    "media": (X_train_mean,  X_test_mean),
    "moda":  (X_train_mode,  X_test_mode),
}

# Balanceadores
balanceadores = {
    "RandomOverSampler": RandomOverSampler(random_state=42),
    "SMOTE": SMOTE(random_state=42),
    "BorderlineSMOTE": BorderlineSMOTE(random_state=42),
}

#Lista de profundidades para DecisionTree
lista_max_depth = [None, 2, 3, 4, 5, 6, 7, 10, 15, 20]

#BLOCO 1 – MODELOS SEM SELEÇÃO DE FEATURES

def treinar_modelos_sem_fs(Xtr_imp, Xte_imp, ytr, yte,
                           nome_cenario, nome_balanceador, balanceador):
    resultados = []

    # 1) oversampling no treino
    Xtr_bal, ytr_bal = balanceador.fit_resample(Xtr_imp, ytr)

    # -------------- Decision Tree (vários max_depth) --------------
    for depth in lista_max_depth:
        dt = DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=depth,
            min_samples_leaf=2,
            min_impurity_decrease=0.001
        )
        dt.fit(Xtr_bal, ytr_bal)
        y_pred = dt.predict(Xte_imp)

        resultados.append({
            "cenario": nome_cenario,
            "balanceador": nome_balanceador,
            "feature_selection": "nenhuma",
            "modelo": "DecisionTree",
            "param": f"max_depth={depth}",
            "accuracy": accuracy_score(yte, y_pred),
            "precision": precision_score(yte, y_pred, zero_division=0),
            "recall": recall_score(yte, y_pred, zero_division=0),
            "f1_score": f1_score(yte, y_pred, zero_division=0),
        })

    # -------------- RandomForest (fixo) --------------
    for depth in lista_max_depth:
      rf = RandomForestClassifier(
          n_estimators=300,
          max_depth=depth,
          random_state=42,
          n_jobs=-1
      )
      rf.fit(Xtr_bal, ytr_bal)
      y_pred_rf = rf.predict(Xte_imp)

      resultados.append({
          "cenario": nome_cenario,
          "balanceador": nome_balanceador,
          "feature_selection": "nenhuma",
          "modelo": "RandomForest",
          "param": f"n_estimators=300,max_depth={depth}",
          "accuracy": accuracy_score(yte, y_pred_rf),
          "precision": precision_score(yte, y_pred_rf, zero_division=0),
          "recall": recall_score(yte, y_pred_rf, zero_division=0),
          "f1_score": f1_score(yte, y_pred_rf, zero_division=0),
      })

    # -------------- XGBoost (fixo) --------------
    for depth in lista_max_depth:
      xgb = XGBClassifier(
          n_estimators=300,
          max_depth=depth,
          learning_rate=0.05,
          subsample=0.8,
          colsample_bytree=0.8,
          random_state=42,
          n_jobs=-1,
          eval_metric="logloss"
      )

      xgb.fit(Xtr_bal, ytr_bal)
      y_pred_xgb = xgb.predict(Xte_imp)

      resultados.append({
          "cenario": nome_cenario,
          "balanceador": nome_balanceador,
          "feature_selection": "nenhuma",
          "modelo": "XGBoost",
          "param": f"n_estimators=300,max_depth={depth}",
          "accuracy": accuracy_score(yte, y_pred_xgb),
          "precision": precision_score(yte, y_pred_xgb, zero_division=0),
          "recall": recall_score(yte, y_pred_xgb, zero_division=0),
          "f1_score": f1_score(yte, y_pred_xgb, zero_division=0),
      })

    return resultados

# Loop geral SEM FS
resultados_sem_fs = []

for nome_cenario, (Xtr_imp, Xte_imp) in cenarios.items():
    print(f"\n=== SEM FS – CENÁRIO: {nome_cenario} ===")
    for nome_bal, bal in balanceadores.items():
        try:
            res = treinar_modelos_sem_fs(
                Xtr_imp, Xte_imp,
                y_train, y_test,
                nome_cenario, nome_bal, bal
            )
            resultados_sem_fs.extend(res)
        except ValueError as e:
            # ADASYN pode falhar em alguns cenários
            print(f"Pulando {nome_cenario} | {nome_bal}: {e}")
            continue

df_resultados_sem_fs = pd.DataFrame(resultados_sem_fs)
df_resultados_sem_fs.sort_values("f1_score", ascending=False).head(20)

df_resultados_sem_fs[(df_resultados_sem_fs.modelo == 'XGBoost') & (df_resultados_sem_fs.cenario == 'moda')]

#BLOCO 2 – MODELOS COM SELEÇÃO DE FEATURES ((PCA / SelectKBest / Correlação + imputação, oversampling, modelo))

K = 15  # número de atributos/componentes que você quer manter

def selecionar_pca(Xtr, ytr, Xte, k=K):
    pca = PCA(n_components=min(k, Xtr.shape[1]), random_state=42)
    Xtr_pca = pca.fit_transform(Xtr)
    Xte_pca = pca.transform(Xte)
    info = {"n_componentes": pca.n_components_}
    return Xtr_pca, Xte_pca, info

def selecionar_selectkbest(Xtr, ytr, Xte, k=K):
    skb = SelectKBest(score_func=f_classif, k=min(k, Xtr.shape[1]))
    Xtr_sel = skb.fit_transform(Xtr, ytr)
    Xte_sel = skb.transform(Xte)
    colunas_sel = Xtr.columns[skb.get_support()]
    info = {"features": list(colunas_sel)}
    return Xtr_sel, Xte_sel, info

def selecionar_por_correlacao(Xtr, ytr, Xte, k=K):
    df_temp = Xtr.copy()
    df_temp["Status"] = ytr.values
    corrs = df_temp.corr(numeric_only=True)["Status"].drop(labels=["Status"])
    corrs_abs = corrs.abs().sort_values(ascending=False)
    top_features = list(corrs_abs.head(min(k, len(corrs_abs))).index)
    Xtr_sel = Xtr[top_features].values
    Xte_sel = Xte[top_features].values
    info = {"features": top_features}
    return Xtr_sel, Xte_sel, info

selecao_features = {
    "PCA": selecionar_pca,
    "SelectKBest": selecionar_selectkbest,
    "Correlacao": selecionar_por_correlacao,
}

# Função para treinar modelos COM FS
def treinar_modelos_com_fs(Xtr_imp, Xte_imp, ytr, yte,
                           nome_cenario, nome_fs, func_fs,
                           nome_balanceador, balanceador, qtde_features):
    resultados = []

    # 1) seleção de features
    Xtr_fs, Xte_fs, info_fs = func_fs(Xtr_imp, ytr, Xte_imp, k=qtde_features)
    print(info_fs)

    # 2) oversampling no treino
    Xtr_bal, ytr_bal = balanceador.fit_resample(Xtr_fs, ytr)

    # -------- Decision Tree (vários max_depth) --------
    for depth in lista_max_depth:
        dt = DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=depth,
            min_samples_leaf=2,
            min_impurity_decrease=0.001
        )
        dt.fit(Xtr_bal, ytr_bal)
        y_pred = dt.predict(Xte_fs)

        resultados.append({
            "cenario": nome_cenario,
            "balanceador": nome_balanceador,
            "feature_selection": nome_fs,
            "fs_info": info_fs,
            "modelo": "DecisionTree",
            "param": f"max_depth={depth}",
            "accuracy": accuracy_score(yte, y_pred),
            "precision": precision_score(yte, y_pred, zero_division=0),
            "recall": recall_score(yte, y_pred, zero_division=0),
            "f1_score": f1_score(yte, y_pred, zero_division=0),
        })

    # -------- RandomForest --------
    for depth in lista_max_depth:
      rf = RandomForestClassifier(
          n_estimators=300,
          max_depth=depth,
          random_state=42,
          n_jobs=-1
      )
      rf.fit(Xtr_bal, ytr_bal)
      y_pred_rf = rf.predict(Xte_fs)

      resultados.append({
          "cenario": nome_cenario,
          "balanceador": nome_balanceador,
          "feature_selection": nome_fs,
          "fs_info": info_fs,
          "modelo": "RandomForest",
          "param": f"n_estimators=300,max_depth={depth}",
          "accuracy": accuracy_score(yte, y_pred_rf),
          "precision": precision_score(yte, y_pred_rf, zero_division=0),
          "recall": recall_score(yte, y_pred_rf, zero_division=0),
          "f1_score": f1_score(yte, y_pred_rf, zero_division=0),
      })

    # -------- XGBoost --------
    for depth in lista_max_depth:
      xgb = XGBClassifier(
          n_estimators=300,
          max_depth=depth,
          learning_rate=0.05,
          subsample=0.8,
          colsample_bytree=0.8,
          random_state=42,
          n_jobs=-1,
          eval_metric="logloss"
      )
      xgb.fit(Xtr_bal, ytr_bal)
      y_pred_xgb = xgb.predict(Xte_fs)

      resultados.append({
          "cenario": nome_cenario,
          "balanceador": nome_balanceador,
          "feature_selection": nome_fs,
          "k": qtde_features,
          "fs_info": info_fs,
          "modelo": "XGBoost",
          "param": f"n_estimators=300,max_depth={depth}",
          "accuracy": accuracy_score(yte, y_pred_xgb),
          "precision": precision_score(yte, y_pred_xgb, zero_division=0),
          "recall": recall_score(yte, y_pred_xgb, zero_division=0),
          "f1_score": f1_score(yte, y_pred_xgb, zero_division=0),
      })

    return resultados

#Loop geral COM FS
# Raquel até aqui..fiquei com receio de apagar os códigos que já tinham sido feitos (Y)
resultados_com_fs = []

for nome_cenario, (Xtr_imp, Xte_imp) in cenarios.items():
    print(f"\n=== COM FS – CENÁRIO: {nome_cenario} ===")
    for nome_fs, func_fs in selecao_features.items():
        print(f"  -> FS: {nome_fs}")
        for nome_bal, bal in balanceadores.items():
          for k in [10, 15, 20]:
            try:
                res = treinar_modelos_com_fs(
                    Xtr_imp, Xte_imp,
                    y_train, y_test,
                    nome_cenario, nome_fs, func_fs,
                    nome_bal, bal, k
                )
                resultados_com_fs.extend(res)
            except ValueError as e:
                # ADASYN pode falhar em alguns cenários; PCA também pode se k>n_features
                print(f"Pulando {nome_cenario} | {nome_fs} | {nome_bal}: {e}")
                continue

df_resultados_com_fs = pd.DataFrame(resultados_com_fs)
df_resultados_com_fs.sort_values("f1_score", ascending=False).head(25)

# ordenar pelos critérios principais
melhor_cenario = (
    df_resultados_com_fs
    .sort_values(
        by=["f1_score", "recall", "accuracy"],
        ascending=[False, False, False]
    )
    .iloc[0]
)

melhor_cenario

#### Aqui começa trabalho das predições

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import joblib

pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("pca", PCA(n_components=20, random_state=42)),
    ("model", XGBClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        eval_metric="logloss"
    ))
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "pipeline_cenario170.pkl")

features = [f for f in features if f != "Status"]
joblib.dump(features, "features_originais_cenario170.pkl")
dicionarios = joblib.load("mapeamentos_categoricos.pkl")

pipeline = joblib.load("pipeline_cenario170.pkl")
features = joblib.load("features_originais_cenario170.pkl")

df_novos = pd.read_csv("novos_alunos.csv")

# Aplicar o mapeamento
for col, mapping in dicionarios.items():
    if col in df_novos.columns:
        df_novos[col] = df_novos[col].map(mapping)

# Imputação zero do melhor cenário
df_novos = df_novos.fillna(0)

# alinhar colunas
X_novos = df_novos.reindex(columns=features, fill_value=0)

# predição
# Rodar o modelo
probs = pipeline.predict_proba(X_novos)[:, 1]
preds = pipeline.predict(X_novos)


df_novos["prob_evasao"] = probs
df_novos["pred_evasao"] = preds

def classificar_risco(p):
    if p >= 0.75:
        return "Alto risco"
    elif p >= 0.50:
        return "Risco médio"
    else:
        return "Baixo risco"

df_novos["nivel_risco"] = df_novos["prob_evasao"].apply(classificar_risco)

df_novos[[
    "Cod_Pessoa",
    "cod_matricula",
    "prob_evasao",
    "pred_evasao",
    "nivel_risco"
]]

#### Encerra aqui