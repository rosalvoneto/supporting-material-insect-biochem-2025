
# Instalar bibliotecas necessárias no Colab
!pip install scikit-learn pandas matplotlib --quiet

# Upload manual do arquivo .csv no Colab
from google.colab import files
uploaded = files.upload()  # O usuário seleciona o arquivo .csv

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Capturar o nome do arquivo CSV carregado
csv_filename = list(uploaded.keys())[0]

# Tentar ler usando o separador padrão, se der erro tenta com o outro
try:
    df = pd.read_csv(csv_filename)
except Exception:
    df = pd.read_csv(csv_filename, sep=';')  # para CSVs com separador ponto-e-vírgula

# Conferir leitura
print("Colunas detectadas:", list(df.columns))

# Separar X e y
y = df.iloc[:, 1].astype(int).values     # Classe (2ª coluna)
X_full = df.iloc[:, 2:].astype(float).values  # Descritores (do 3º em diante)
feature_names = df.columns[2:]

# --- Hiperparâmetros do algoritmo ABC + Random Forest ---
ITERATIONS = 500
N_BEES = 50
NO_IMPROVEMENT_LIMIT = 50
INIT_VARS = 50
BFS_Q2_LIMIT = 0.99
NO_IMPROVEMENT_BFS = 30
VARS_TO_ADD = 1
OFFSPRING_PER_PARENT = 2
N_TREES = 50
MAX_FEATURES = 5

from random import sample, randint

def evaluate_subset(X, y, features_idx):
    if len(features_idx) == 0:
        return 0
    model = RandomForestClassifier(n_estimators=N_TREES, max_features=MAX_FEATURES, random_state=42)
    scores = cross_val_score(model, X[:, features_idx], y, cv=5, scoring='roc_auc')
    return scores.mean()

num_features = X_full.shape[1]
population = [sample(range(num_features), INIT_VARS) for _ in range(N_BEES)]
fitness = [evaluate_subset(X_full, y, p) for p in population]
best_solution = population[np.argmax(fitness)]
best_score = max(fitness)
best_scores = [best_score]
no_improvement_counter = 0

for iteration in range(ITERATIONS):
    new_population = []
    for i in range(N_BEES):
        parent = population[i]
        new_child = parent[:]
        for _ in range(VARS_TO_ADD):
            new_var = randint(0, num_features - 1)
            if new_var not in new_child:
                new_child.append(new_var)
        new_child = list(set(new_child))
        new_fitness = evaluate_subset(X_full, y, new_child)
        if new_fitness > fitness[i]:
            population[i] = new_child
            fitness[i] = new_fitness

    current_best = population[np.argmax(fitness)]
    current_score = max(fitness)
    best_scores.append(current_score)

    if current_score > best_score:
        best_solution = current_best
        best_score = current_score
        no_improvement_counter = 0
    else:
        no_improvement_counter += 1

    if no_improvement_counter >= NO_IMPROVEMENT_LIMIT:
        break

# Resultados
print(f"Melhor AUC: {best_score:.4f}")
print(f"Melhores variáveis (indices): {sorted(best_solution)}")
print(f"Melhores variáveis (nomes): {[feature_names[i] for i in sorted(best_solution)]}")

plt.plot(best_scores)
plt.xlabel("Iteração")
plt.ylabel("Melhor AUC")
plt.title("Convergência do ABC com Random Forest")
plt.grid(True)
plt.show()

# Gerar arquivo com variáveis selecionadas (ABC)
final_df = df.iloc[:, :2].copy()
selected_columns = [feature_names[i] for i in sorted(best_solution)]
final_df[selected_columns] = df[selected_columns]
final_df.to_csv("subset_selected_variables.csv", index=False)
print("✅ Arquivo salvo: subset_selected_variables.csv")

# Download do arquivo do ABC
from google.colab import files
files.download("subset_selected_variables.csv")

# -------------------------------
# Filtro Best-First-like (BFS) pós-ABC
# -------------------------------

# 1. Carrega o CSV gerado pelo algoritmo ABC
df_bfs = pd.read_csv("subset_selected_variables.csv")

# 2. Separa colunas
X = df_bfs.iloc[:, 2:].copy()  # descritores
y = df_bfs.iloc[:, 1].values   # classe binária

# 3. Avalia AUC de cada variável sozinha
scores = {}
for col in X.columns:
    model = RandomForestClassifier(n_estimators=50, max_features=5, random_state=42)
    auc = cross_val_score(model, X[[col]], y, cv=5, scoring='roc_auc').mean()
    scores[col] = auc

# 4. Ordena variáveis por performance individual
sorted_vars = sorted(scores.items(), key=lambda x: x[1], reverse=True)

# 5. Best-First-like: adiciona variáveis apenas se melhorarem AUC
selected = []
best_auc = 0

for var, score in sorted_vars:
    candidate = selected + [var]
    model = RandomForestClassifier(n_estimators=50, max_features=5, random_state=42)
    auc = cross_val_score(model, X[candidate], y, cv=5, scoring='roc_auc').mean()
    if auc > best_auc:
        selected.append(var)
        best_auc = auc

print(f"✅ Melhor subconjunto: {selected}")
print(f"🔢 AUC final após BFS: {best_auc:.4f}")

# 6. Gera novo DataFrame e salva
df_filtered = df_bfs.iloc[:, :2].copy()  # No + Effect
df_filtered[selected] = X[selected]
output_name = "subset_selected_filtered.csv"
df_filtered.to_csv(output_name, index=False)
print(f"📁 Arquivo salvo: {output_name}")

# 7. Download no Google Colab
files.download(output_name)


    2. Script for Google Colab - XGBoost classifier generation

# Instalação e importações
!pip install xgboost scikit-learn pandas matplotlib seaborn --quiet

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from google.colab import files
import pickle

# Upload do arquivo
print("📁 Faça upload do CSV contendo os dados (ex: 'subset_selected_filtered bfs.csv'):")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# Leitura
df = pd.read_csv(file_name)

# Identificação automática das colunas de ID e Classe
id_column_candidates = ['No,', 'No', 'ID', 'id']
effect_column_candidates = ['Effect', 'Classe', 'class', 'Target', 'target']

# Acha colunas presentes
id_col = next((col for col in id_column_candidates if col in df.columns), None)
effect_col = next((col for col in effect_column_candidates if col in df.columns), None)

if not effect_col:
    raise ValueError("Coluna de classe (Effect/Classe/Target) não encontrada no arquivo.")

# Seleção dinâmica dos descritores (todas exceto ID e classe)
X = df.drop(columns=[c for c in [id_col, effect_col] if c is not None])
y = df[effect_col]
ids = df[id_col] if id_col is not None else np.arange(len(df))

# Modelo XGBoost
model = xgb.XGBClassifier(
    n_estimators=10,
    max_depth=2,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=1.0,
    reg_alpha=0.1,
    reg_lambda=1.0,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Validação cruzada k=5
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
predictions = []
probas = []
true_labels = []
ids_all = []
folds = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
tprs = []

for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    id_test = ids.iloc[test_idx] if isinstance(ids, pd.Series) else ids[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    aucs.append(auc)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)

    predictions.extend(y_pred)
    probas.extend(y_proba)
    true_labels.extend(y_test)
    ids_all.extend(id_test)
    folds.extend([i+1]*len(y_test))

# Tabela de resultados
df_results = pd.DataFrame({
    'ID': ids_all,
    'Fold': folds,
    'Real': true_labels,
    'Previsto': predictions,
    'Probabilidade': probas
})
df_auc = pd.DataFrame({'Fold': list(range(1, 6)), 'AUC': aucs})
df_auc.loc[len(df_auc)] = ['Média', np.mean(aucs)]

# Importância dos descritores
model.fit(X, y)
xgb.plot_importance(model, max_num_features=20)
plt.title("Importância dos descritores")
plt.tight_layout()
plt.show()

# Curva ROC média
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(aucs)

plt.plot(mean_fpr, mean_tpr, label=f'Média ROC (AUC = {mean_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curva ROC média (k=5)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Exportações
df_results.to_csv("resultados_xgb.csv", index=False)
df_auc.to_csv("auc_xgb.csv", index=False)

with open("xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Downloads automáticos
files.download("resultados_xgb.csv")
files.download("auc_xgb.csv")
files.download("xgb_model.pkl")

*********************************************************************

    3. Script for Google Colab - model application

# Etapa 1: Instalação de pacotes e importações
!pip install xgboost pandas scikit-learn matplotlib openpyxl --quiet

import pandas as pd
import numpy as np
import pickle
from google.colab import files

# Etapa 2: Upload do modelo .pkl
print("📦 Envie o arquivo do modelo treinado (.pkl):")
uploaded_model = files.upload()
model_file = list(uploaded_model.keys())[0]

with open(model_file, "rb") as f:
    model = pickle.load(f)

# Etapa 3: Upload do XLSX com novas amostras (NuBBE)
print("📁 Agora envie o arquivo XLSX com as amostras a serem previstas (NuBBE):")
uploaded_xlsx = files.upload()
xlsx_file = list(uploaded_xlsx.keys())[0]

# Etapa 4: Leitura e preparação dos dados
# Lê Excel e converte separador decimal vírgula para ponto (internamente)
df_new = pd.read_excel(xlsx_file, decimal=',')

# Seleciona apenas as colunas desejadas, se existirem no arquivo
required_cols = ['SpMax_EA', 'SsCH3', 'SM13_EA(ri)', 'VR2_X', 'Eig12_AEA(bo)', 'G2e', 'SpMin5_Bh(s)']
missing = [col for col in required_cols if col not in df_new.columns]
if missing:
    raise ValueError(f"❌ Faltam colunas necessárias no arquivo: {', '.join(missing)}")

X_new = df_new[required_cols].copy()

# Remove linhas com qualquer valor ausente (NaN, na, nam, etc)
for na_val in ['na', 'Na', 'NA', 'nan', 'Nam', 'NAM', 'NAN', 'N/a', 'n/a']:
    X_new = X_new.replace(na_val, np.nan)
X_new = X_new.apply(pd.to_numeric, errors='coerce')  # força conversão, converte 'na' em np.nan
before = len(X_new)
X_new_clean = X_new.dropna(axis=0)
after = len(X_new_clean)
if after < before:
    print(f"⚠️ {before-after} linhas removidas por conter valores ausentes nos descritores selecionados.")

# Mantém apenas linhas válidas para nome/ID
df_clean = df_new.loc[X_new_clean.index].copy()
names = df_clean['NAME'] if 'NAME' in df_clean.columns else np.arange(len(df_clean))

# Etapa 5: Previsão com o modelo treinado
y_pred = model.predict(X_new_clean)
y_proba = model.predict_proba(X_new_clean)[:, 1]

# Etapa 6: Resultados em DataFrame
df_pred = pd.DataFrame({
    'NAME': names.values,
    'Classe_prevista': y_pred,
    'Probabilidade': y_proba
})

# Etapa 7: Exportar resultado como CSV
output_path = "previsoes_xgb_nubbe.csv"
df_pred.to_csv(output_path, index=False)
files.download(output_path)
