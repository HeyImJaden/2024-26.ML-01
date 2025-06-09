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
pipe_fittato = pipe.fit(X_train, y_train)

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


# %%
import joblib

joblib.dump(pipe_fittato, "logistic_model.joblib")
