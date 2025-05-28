# %% [markdown]
# ## Features engeneering
# 
# Testo senza rimuove latitude e longitude

# %%
import pandas as pd

# %%
df = pd.read_csv("startupdata.csv")

# %%
df

# %%
df.size

# %%
df.shape

# %%
df["age_first_milestone_year_missing"] = df["age_first_milestone_year"].isnull().astype(int) # 0 ha raggiunto la milestone
df["age_last_milestone_year_missing"] = df["age_last_milestone_year"].isnull().astype(int) # 1 non ha raggiunto la milestone
df["age_first_milestone_year"].fillna(-1, inplace=True)
df["age_last_milestone_year"].fillna(-1, inplace=True)

# %%
df["Unnamed: 6"] = df["city"] + " " + df["state_code"] + " " + df["zip_code"]

# %%
df.rename(columns={"Unnamed: 6": "city_state_zip"}, inplace=True) # rinomino la colonna Unnamed: 6 in city_state_zip

# %%
freq = df["city_state_zip"].value_counts()
df["city_state_zip_freq"] = df["city_state_zip"].map(freq)

# %%
df.drop(["Unnamed: 0", "city_state_zip", "state_code", "state_code.1", "zip_code", "id", "city", "name", "founded_at", "closed_at", "first_funding_at", "last_funding_at", "category_code", "object_id", "status"], axis = 1, inplace= True) # status = 1 acquired, 0 fallito

# %%
df.columns

# %%
df

# %%
df.isnull().sum()

# %%
df.duplicated().sum()

# %% [markdown]
# ## Esplorazione del dataset

# %%
import matplotlib.pyplot as plt

# %%
# Calcola la frequenza delle colonne richieste
freq = df[["is_CA","is_NY", "is_MA", "is_TX", "is_otherstate"]].sum()

# Crea il grafico a barre
freq.plot(kind='bar')
plt.title("Frequenza di NY, MA, TX")
plt.ylabel("Frequenza")
plt.xticks(rotation=0)
plt.show()

# %%
df['labels'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Acquisito', 'Chiuso'], colors=['Red','Blue'])
plt.title('Distribuzione Labels (Acquisito vs Chiuso)')
plt.show()

# %%
freq = df[
    ["is_software", 
     "is_web", 
     "is_mobile", 
     "is_enterprise", 
     "is_advertising", 
     "is_gamesvideo", 
     "is_ecommerce", 
     "is_biotech", 
     "is_consulting", 
     "is_othercategory"]].sum()

freq.plot(kind='bar')

# %%
round_cols = ['has_roundA', 'has_roundB', 'has_roundC', "has_VC"]
round_counts = df[round_cols].sum()
plt.figure(figsize=(6,6))
plt.pie(round_counts, labels=round_cols, autopct='%1.1f%%', startangle=90, colors=['Red','Green','Blue', "Purple"])
plt.title('Percentuale aziende con Round A, B, C, VC')
plt.show()

# %% [markdown]
# ## Preprocessing

# %%
from sklearn.model_selection import train_test_split
from sklearn import set_config

# %%
set_config(transform_output="pandas")

# %%
X = df.drop("labels", axis = 1)
y = df["labels"]

# %%
X

# %%
y

# %%
y.value_counts(normalize=True)

# %% [markdown]
# ## Divisione in train test split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

# %% [markdown]
# ## Baseline: Regressione Logistica
# Creiamo una baseline semplice usando la regressione logistica per valutare le performance iniziali del modello.
# 
# - Per la baseline creo una pipeline con standardizzazione e modello

# %%
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# %%
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter= 500, verbose= 1, n_jobs=-1, random_state=42))
])

# %%
pipe.fit(X_train, y_train)

# %%
y_pred_train = pipe.predict(X_train)

# %%
y_pred = pipe.predict(X_test)

# %%
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# %%
accuracy_score(y_test, y_pred)

# %%
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

# %%
cr = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(cr)

# %% [markdown]
# ## Metriche Logistic Regression

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

# %%


# %%
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# %%
accuracy_train = accuracy_score(y_train, y_pred_train)
precision_train = precision_score(y_train, y_pred_train)
recall_train = recall_score(y_train, y_pred_train)
f1_train = f1_score(y_train, y_pred_train)
roc_auc_train = roc_auc_score(y_train, y_pred_train)


# %%
print(f"""
TRAIN METRICS - Logistic Regression:
Accuracy: {accuracy_train},
Precision: {precision_train}, 
Recall: {recall_train},
F1-score: {f1_train}, 
ROC-AUC: {roc_auc_train}

----------------------------

TEST METRICS - Logistic Regression:
Accuracy: {accuracy}, 
Precision: {precision}, 
Recall: {recall},
F1-score: {f1}, 
ROC-AUC: {roc_auc}
""")


# %%
metricslr = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-score': f1,
    'ROC-AUC': roc_auc
}

plt.figure(figsize=(8, 5))
plt.bar(metricslr.keys(), metricslr.values(), color='skyblue')
plt.title('Model Performance: Logistic Regression')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
y_proba = pipe.predict_proba(X_test)[:, 1] # prendo solo la classe positiva
log_losslr = log_loss(y_test, y_proba) 
log_losslr

# %% [markdown]
# ## Random Forest Classifier

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
X = df.drop("labels", axis= 1)
y = df["labels"]

# %%
X

# %%
y

# %%
y.value_counts(normalize=True) * 100

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

# %%
from sklearn.model_selection import StratifiedKFold, cross_val_score

# %%
pipe2 = Pipeline([
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=5,                # Limita la profondità degli alberi
        min_samples_split=10,        # Aumenta i campioni minimi per split
        min_samples_leaf=8,    
        max_features="sqrt",     # Aumenta i campioni minimi per foglia
        verbose=1,
        n_jobs=-1,
        random_state=42
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(pipe2, X_train, y_train, cv=cv, scoring='accuracy')

# %%
print(f"Cross-validation scores: {scores}")
print(f"Mean CV accuracy: {scores.mean()}")

# %%
pipe2.fit(X_train, y_train)

# %%
y_pred_train = pipe2.predict(X_train)

# %%
y_pred = pipe2.predict(X_test)

# %% [markdown]
# ## Metriche Random Forest Classifier

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

# %%
class_report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(class_report)

# %%
accuracyrf_train = accuracy_score(y_train, y_pred_train)
precisionrf_train = precision_score(y_train, y_pred_train)
recallrf_train = recall_score(y_train, y_pred_train)
f1rf_train = f1_score(y_train, y_pred_train)
roc_aucrf_train = roc_auc_score(y_train, y_pred_train)


# %%
accuracyrf = accuracy_score(y_test, y_pred)
precisionrf = precision_score(y_test, y_pred)
recallrf = recall_score(y_test, y_pred)
f1rf = f1_score(y_test, y_pred)
roc_aucrf = roc_auc_score(y_test, y_pred)

# %%
print(f""" 
TRAIN METRICS - Random Forest:
Accuracy: {accuracyrf_train},
Precision: {precisionrf_train}, 
Recall: {recallrf_train},
F1-score: {f1rf_train}, 
ROC-AUC: {roc_aucrf_train}

----------------------------

TEST METRICS - Random Forest:
Accuracy: {accuracyrf},
Precision: {precisionrf}, 
Recall: {recallrf},
F1-score: {f1rf}, 
ROC-AUC: {roc_aucrf}
""")

# %%
metricsrf = {
    'Accuracy': accuracyrf,
    'Precision': precisionrf,
    'Recall': recallrf,
    'F1-score': f1rf,
    'ROC-AUC': roc_aucrf
}

plt.figure(figsize=(8, 5))
plt.bar(metricsrf.keys(), metricsrf.values(), color='skyblue')
plt.title('Model Performance: Random Forest')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
y_proba = pipe2.predict_proba(X_test)[:, 1] # prendo solo la classe positiva
log_lossrf = log_loss(y_test, y_proba) 
log_lossrf

# %%
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

# %% [markdown]
# ## Confronto tra Logistic Regression e Random Forest Classifier

# %%
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'Log Loss']
random_forest = [accuracyrf, precisionrf, recallrf, f1rf, roc_aucrf, log_lossrf]
logistic_regression = [accuracy, precision, recall, f1, roc_auc, log_losslr]

# %%
df_long0 = pd.DataFrame({
    'Metric': metrics,
    'Random Forest': random_forest,
    'Logistic Regression': logistic_regression
}).melt(id_vars='Metric', var_name='Model', value_name='Score')

# %%
import seaborn as sns

# %%
plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Metric', hue='Model', data=df_long0, palette='Set1', orient='h')

plt.title('Confronto tra Random Forest e Logistic Regression')
plt.xlabel('Score')
plt.ylabel('Metriche')
plt.legend(title='Modello')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Model Optimization - Grid Search

# %%
pipe2.get_params()

# %% [markdown]
# Parametri con cui decido di lavorare: 
# - n estimators
# - max depth
# - criterion
# - min samples split
# - min samples leaf
# - max features

# %%
params = {
    "classifier__n_estimators": [100, 200],  # Numero di alberi, evita valori troppo alti per efficienza
    "classifier__max_depth": [3, 5, 10, 20],  # Alberi meno profondi per ridurre overfitting
    "classifier__criterion": ["gini", "entropy"],
    "classifier__min_samples_split": [5, 10, 20],  # Più campioni per split = meno overfitting
    "classifier__min_samples_leaf": [4, 8, 12],    # Più campioni per foglia = meno overfitting
    "classifier__max_features": ["sqrt", "log2"],  # Limita il numero di feature considerate
    "classifier__class_weight": [None, "balanced"] # Prova anche il bilanciamento automatico
}

# %% [markdown]
# Decido di usare un k-fold stratificato per evitare che i dati di test vengano inclusi nel set di addestramento, garantendo così una valutazione imparziale del modello su dati mai visti prima.

# %%
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# %%
grid_search = GridSearchCV(
    estimator=pipe2,
    param_grid=params,
    scoring="f1",
    n_jobs=-1,
    refit=True,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    verbose=1
)

# %%
grid_search.fit(X_train, y_train)

# %%
grid_search.best_score_

# %%
grid_search.best_params_

# %%
random_migliorato = grid_search.best_estimator_

# %%
df_grid = pd.DataFrame(grid_search.cv_results_)

# %%
df_grid.sort_values(by="mean_test_score", ascending=True)

# %% [markdown]
# ## Valutazione del modello RFC con i parametri della Grid Search

# %%
y_pred_train = random_migliorato.predict(X_train)

# %%
y_pred = random_migliorato.predict(X_test)

# %%
accuracygrid_train = accuracy_score(y_train, y_pred_train)
precisiongrid_train = precision_score(y_train, y_pred_train)
recallgrid_train = recall_score(y_train, y_pred_train)
f1grid_train = f1_score(y_train, y_pred_train)
roc_aucgrid_train = roc_auc_score(y_train, y_pred_train)


# %%
accuracygrid = accuracy_score(y_test, y_pred)
precisiongrid = precision_score(y_test, y_pred)
recallgrid = recall_score(y_test, y_pred)
f1grid = f1_score(y_test, y_pred)
roc_aucgrid = roc_auc_score(y_test, y_pred)

# %%
print(f""" 
TRAIN METRICS - Grid Search - Random Forest:
Accuracy: {accuracygrid_train},
Precision: {precisiongrid_train}, 
Recall: {recallgrid_train},
F1-score: {f1grid_train}, 
ROC-AUC: {roc_aucgrid_train}

----------------------------

TEST METRICS - Grid Search - Random Forest:
Accuracy: {accuracygrid},
Precision: {precisiongrid}, 
Recall: {recallgrid},
F1-score: {f1grid}, 
ROC-AUC: {roc_aucgrid}
""")

# %%
report_migliorato = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(report_migliorato)

# %%
y_prob = random_migliorato.predict_proba(X_test)[:, 1]  # Probabilità per la classe positiva
logloss_grid = log_loss(y_test, y_prob)

# %%
logloss_grid

# %% [markdown]
# Confronto nuovo modello vs vecchio

# %%
metrics_new = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'Log Loss']
random_forest = [accuracyrf, precisionrf, recallrf, f1rf, roc_aucrf, log_lossrf]
random_forest_gridsearch = [accuracygrid, precisiongrid, recallgrid, f1grid, roc_aucgrid, logloss_grid]

# %%
df_long = pd.DataFrame({
    'Metric': metrics_new,
    'Random Forest': random_forest,
    'RF Grid Search': random_forest_gridsearch
}).melt(id_vars='Metric', var_name='Model', value_name='Score')

# %%
# Plot comparativo
plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Metric', hue='Model', data=df_long, palette='Set2', orient='h')

plt.title('Confronto tra Random Forest e RF Grid Search')
plt.xlabel('Score')
plt.ylabel('Metriche')
plt.legend(title='Modello')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Optuna Model Selection
# 
# Dato che con la grid search, i parametri da me selezionati, non migliorano il modello, decido di optare per il bayesian search

# %%
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, log_loss

# %%
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'random_state': 42,
        'n_jobs': -1
    }
    clf = RandomForestClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    return score.mean()

# %%
study6 = optuna.create_study(storage="sqlite:///study6.db", direction='minimize', study_name="optuna_study6", load_if_exists=True)

# %%
study6.optimize(objective, n_trials=10, show_progress_bar=False)

# %%
print('Best trial:')
print(study6.best_trial)
print('Best params:')
print(study6.best_params)

best_rf = RandomForestClassifier(**study6.best_params)
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)
y_pred_train = best_rf.predict(X_train)
y_prob = best_rf.predict_proba(X_test)[:, 1]
y_prob_train = best_rf.predict_proba(X_train)[:, 1]
print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1-score:', f1_score(y_test, y_pred))
print('Log Loss:', log_loss(y_test, y_prob))

# %%
study6.trials_dataframe()

# %%
accuracyopt_train = accuracy_score(y_train, y_pred_train)
precisionopt_train = precision_score(y_train, y_pred_train)
recallopt_train = recall_score(y_train, y_pred_train)
f1opt_train = f1_score(y_train, y_pred_train)
roc_aucopt_train = roc_auc_score(y_train, y_prob_train)
loglossopt_train = log_loss(y_train, y_prob_train)

# %%
accuracyopt = accuracy_score(y_test, y_pred)
precisionopt = precision_score(y_test, y_pred)
recallopt = recall_score(y_test, y_pred)
f1opt = f1_score(y_test, y_pred)
roc_aucopt = roc_auc_score(y_test, y_prob)
loglossopt = log_loss(y_test, y_prob)

# %%
print(f""" 
TRAIN METRICS - Optuna Model:
Accuracy:   {accuracyopt_train}
Precision:  {precisionopt_train}
Recall:     {recallopt_train}
F1-score:   {f1opt_train}
ROC-AUC:    {roc_aucopt_train}
Log Loss:   {loglossopt_train}

----------------------------

TEST METRICS - Optuna Model:
Accuracy:   {accuracyopt}
Precision:  {precisionopt}
Recall:     {recallopt}
F1-score:   {f1opt}
ROC-AUC:    {roc_aucopt}
Log Loss:   {loglossopt}
""")


# %%
metrics_new_opt = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'Log Loss']
optuna = [accuracyopt, precisionopt, recallopt, f1opt, roc_aucopt, loglossopt]
random_forest_gridsearch = [accuracygrid, precisiongrid, recallgrid, f1grid, roc_aucgrid, logloss_grid]

# %%
df_long1 = pd.DataFrame({
    'Metric': metrics_new_opt,
    'RF Optuna': optuna,
    'RF Grid Search': random_forest_gridsearch
}).melt(id_vars='Metric', var_name='Model', value_name='Score')

# %%
plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Metric', hue='Model', data=df_long1, palette='Set3', orient='h')

plt.title('Confronto tra RF Optuna e RF Grid Search')
plt.xlabel('Score')
plt.ylabel('Metriche')
plt.legend(title='Modello')
plt.tight_layout()
plt.show()

# %% [markdown]
# Dal grafico possiamo notare che i parametri trovati dal bayesian search, performano meglio del gridsearch

# %%
metrics_new_rfopt = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'Log Loss']
random_forest = [accuracyrf, precisionrf, recallrf, f1rf, roc_aucrf, log_lossrf]
optuna = [accuracyopt, precisionopt, recallopt, f1opt, roc_aucopt, loglossopt]

# %%
df_long2 = pd.DataFrame({
    'Metric': metrics_new_rfopt,
    'Random Forest': random_forest,
    'RF Optuna': optuna
}).melt(id_vars='Metric', var_name='Model', value_name='Score')

# %%
plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Metric', hue='Model', data=df_long2, palette='Set2', orient='h')

plt.title('Confronto tra Random Forest e RF Optuna')
plt.xlabel('Score')
plt.ylabel('Metriche')
plt.legend(title='Modello')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Esplorazione: Optuna Optimization

# %%
import optuna.visualization as opt

# %%
opt.plot_optimization_history(study6)

# %%
opt.plot_param_importances(study6)

# %%
opt.plot_parallel_coordinate(study6)

# %%
opt.plot_slice(study6)

# %%
opt.plot_edf(study6)


