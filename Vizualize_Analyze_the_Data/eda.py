# %%
import pandas as pd # type: ignore
import seaborn as sns # type: ignore

import matplotlib.pyplot as plt #pacote de plotting # type: ignore
#%matplotlib inline
import matplotlib as mpl # type: ignore
import seaborn as sns  # type: ignore
mpl.rcParams['figure.dpi'] = 200 #Alta Resolução

# DISCLAIMER: Muito da limpeza dos dados já foi feita no módulo "Transform_the_Data". Por favor verificar.



# %%
df_laranja = pd.read_excel(r'C:\Users\garbu\OneDrive\Área de Trabalho\laranja\transform_the_data\df_laranja_sp.xlsx')
df_laranja = df_laranja.drop(columns='Unnamed: 0') #correção por que foi exportado para excel no último modo

# %%
# Características do df_laranja
df_laranja.shape
df_laranja.columns
df_laranja.info()
# %%
df_laranja.describe()

# %%
# List of numeric columns
numeric_columns = ['Laranja - Posta - Indústria', 'Laranja Lima - Árvore - Mercado',
                   'Laranja Pêra - Árvore - Mercado', 'producao', 'rendimento_medio',
                   'area_plantada', 'area_colhida',
                   'NUMERO DE DIAS COM PRECIP. PLUV, MENSAL (AUT)(número)',
                   'PRECIPITACAO TOTAL, MENSAL (AUT)(mm)',
                   'TEMPERATURA MEDIA, MENSAL (AUT)(°C)']

numeric_df = df_laranja[numeric_columns]

# %%
df_laranja.columns

# %%
# variaveis climáticas
clim_var = df_laranja[['NUMERO DE DIAS COM PRECIP. PLUV, MENSAL (AUT)(número)',
       'PRECIPITACAO TOTAL, MENSAL (AUT)(mm)',
       'TEMPERATURA MEDIA, MENSAL (AUT)(°C)']]

# variaveis producao
prod_var = df_laranja[['producao', 'rendimento_medio', 'area_plantada', 'area_colhida']]

# variaeis de preço
prec_var = df_laranja[['Laranja - Posta - Indústria',
       'Laranja Lima - Árvore - Mercado', 'Laranja Pêra - Árvore - Mercado']]

# %%
#mpl.rcParams['font.size'] = 4
clim_var.hist(layout=(1,3));

# %%
prec_var.hist(layout=(1,3));

# %%
prod_var.hist(layout=(2,2));

# Aparentemente área plantada, rendimento_medio e area_colhida estão com dados faltantes
# %%
features_response = df_laranja.columns.tolist()
features_to_remove = ['Laranja - Posta - Indústria','Laranja Lima - Árvore - Mercado','rendimento_medio', 'area_plantada', 'area_colhida', 'Estado']
features_to_remove_d = ['Laranja - Posta - Indústria','Laranja Lima - Árvore - Mercado','rendimento_medio', 'area_plantada', 'area_colhida', 'Estado', 'date']
# Remover Estado por que só temos um estado
# Remover Laranja - Posta - industria por que é votlado ao setor industrial
# Remover Rendimento_medio, área_plantada e area_colhida por que estão com dados faltantes (v1)

features_response = [item for item in features_response if item not in features_to_remove]
features_response_d = [item for item in features_response if item not in features_to_remove_d]
features_response
# %%
df_laranja_clean = df_laranja[features_response]
df_laranja_clean_d = df_laranja[features_response_d]
df_laranja_clean.head(2)

# %%
# Correlação entre variáveis e Teste F
corr = df_laranja_clean.corr()
corr_d = df_laranja_clean_d.corr()
corr.iloc[0:5, 0:5] 

# %%
mpl.rcParams['font.size'] = 6
plt.figure(figsize=(10, 8))
hm = sns.heatmap(corr.iloc[0:5, 0:5], cmap='Blues', cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=True, xticklabels=True)
plt.title('Correlações entre as Variáveis', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
# %%
mpl.rcParams['font.size'] = 6
plt.figure(figsize=(10, 8))
hm = sns.heatmap(corr_d.iloc[0:5, 0:5], cmap='Blues', cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=True, xticklabels=True)
plt.title('Correlações entre as Variáveis', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()

# %% 
# Aparentemente, O preço da laranja está fortemente conectado com a temperatura média mensal e também com a produção
df_laranja_clean_d.columns
# %%
# teste F para seleção de variáveis:

#X = df_laranja_clean[[ 'producao',
       #'NUMERO DE DIAS COM PRECIP. PLUV, MENSAL (AUT)(número)',
       #'PRECIPITACAO TOTAL, MENSAL (AUT)(mm)',
       #'TEMPERATURA MEDIA, MENSAL (AUT)(°C)']]

X = df_laranja[['producao', 'rendimento_medio',
                   'area_plantada', 'area_colhida',
                   'NUMERO DE DIAS COM PRECIP. PLUV, MENSAL (AUT)(número)',
                   'PRECIPITACAO TOTAL, MENSAL (AUT)(mm)',
                   'TEMPERATURA MEDIA, MENSAL (AUT)(°C)']]

#y = df_laranja_clean[['Laranja Pêra - Árvore - Mercado']]
y = df_laranja[['Laranja Pêra - Árvore - Mercado']]
# %%
from scipy import stats # type: ignore
from sklearn.feature_selection import f_classif # type: ignore
from sklearn.feature_selection import SelectPercentile # type: ignore

# %%
[f_stat, f_p_value] = f_classif(X, y)

# %%
f_test_df = pd.DataFrame({'Feature': X.columns.tolist(),
                          'F statistic': f_stat,
                          'p value': f_p_value})
f_test_df.sort_values('p value')
# %%
selector = SelectPercentile(f_classif, percentile=20)
selector.fit(X, y)
# %%
best_feature_ix = selector.get_support()
best_feature_ix
features = X.columns.tolist()
best_features = [features[counter] for counter in range(len(features))
                 if best_feature_ix[counter]]
best_features
# %%
# v1. Nesse teste, concluimos que nenhuma das características tem uma relação significativamente estatística com o preço da laranja pera.
# v2. No teste 2, com todas as características, vimos que Área_Plantada e Area_colhida influenciam significativamente no preço da Laranja


# %%
# Filter the DataFrame for the desired period
import matplotlib.pyplot as plt # type: ignore

# Filter the DataFrame for the desired period
mask = (df_laranja['date'] >= '2023-01-01') & (df_laranja['date'] <= '2023-12-29')
df_filtered = df_laranja.loc[mask]

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting the monthly production data on the primary y-axis (Y1)
df_filtered.groupby(df_filtered['date'].dt.strftime('%Y-%m'))['producao'].sum().plot(kind='bar', ax=ax1, color='blue', position=0, width=0.4)
ax1.set_ylabel('Produção', color='blue')

# Creating a secondary y-axis (Y2) for 'Laranja Pêra - Árvore - Mercado' data
ax2 = ax1.twinx()
df_filtered.groupby(df_filtered['date'].dt.strftime('%Y-%m'))['Laranja Pêra - Árvore - Mercado'].sum().plot(kind='line', ax=ax2, color='red')
ax2.set_ylabel('Laranja Pêra - Árvore - Mercado', color='red')

# Set x-axis label and title
ax1.set_xlabel('Mês')
plt.title('Produção e Preço de Laranja Pêra de Outubro de 2023 a Fevereiro de 2024')

# Show plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
# Filter the DataFrame for the desired period
mask = (df_laranja['date'] >= '2023-01-01') & (df_laranja['date'] <= '2023-12-29')
df_filtered = df_laranja.loc[mask]

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting the monthly area_plantada data on the primary y-axis (Y1)
df_filtered.groupby(df_filtered['date'].dt.strftime('%Y-%m'))['area_plantada'].sum().plot(kind='bar', ax=ax1, color='blue', position=0, width=0.4)
ax1.set_ylabel('Área Plantada', color='blue')

# Creating a secondary y-axis (Y2) for 'Laranja Pêra - Árvore - Mercado' data
ax2 = ax1.twinx()
df_filtered.groupby(df_filtered['date'].dt.strftime('%Y-%m'))['Laranja Pêra - Árvore - Mercado'].sum().plot(kind='line', ax=ax2, color='red')
ax2.set_ylabel('Laranja Pêra - Árvore - Mercado', color='red')

# Set x-axis label and title
ax1.set_xlabel('Mês')
plt.title('Área Plantada e Preço de Laranja Pêra de Outubro de 2023 a Fevereiro de 2024')

# Show plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
from sklearn.model_selection import train_test_split # type: ignore
X_train, X_test, y_train, y_test = \
       train_test_split(X.values, \
              y.values, test_size=0.2, random_state=24)

# %%
from sklearn.model_selection import GridSearchCV # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
import numpy as np # type: ignore

rf = RandomForestRegressor(n_estimators=10, criterion='squared_error', max_depth=3,
                           min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                           bootstrap=True, oob_score=False, n_jobs=None, random_state=4, verbose=0,
                           warm_start=False)

# %%
df_params_ex = {
    'max_depth': [1, 2, 4, 6, 8,10, 12],
    'n_estimators': list(range(10, 100, 10))
}

cv_rf_ex = GridSearchCV(rf, param_grid=df_params_ex, scoring='neg_mean_squared_error',
                        n_jobs=None, refit=True, cv=4, verbose=1,
                        error_score=np.nan, return_train_score=True)
# %%
cv_rf_ex.fit(X_train, y_train.ravel())
# %%
print("Best Parameters:", cv_rf_ex.best_params_)
best_rf = cv_rf_ex.best_estimator_
print("Feature Importances:", best_rf.feature_importances_)

# %%
#cv_rf_results_df= pd.DataFrame(cv_rf_ex.cv_results_)
#cv_rf_results_df.sort_values('mean_test_score', ascending=False)
# %%

feature_importances = best_rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

# %%
plt.figure(figsize=(10, 6))
plt.bar(importance_df['Feature'][:3], importance_df['Importance'][:3], width=0.4)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.xticks(rotation=45)  # Rotate feature names for better readability
plt.tight_layout()
plt.show()