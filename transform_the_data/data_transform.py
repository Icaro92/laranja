# %% 
import pandas as pd
import json 
import numpy as np
#_ibge data
    # Cód 109 = Área plantada
    # Cód 216 = Área colhida
    # Cód 35 = Produção
    # Cód 36 = Rendimento médio
    
# %%
file_path1 = r'C:\Users\garbu\OneDrive\Área de Trabalho\laranja\get_the_data\ibge_data\get_the_data\ibge_data\2024-04-12 13_23_54.053629_35.json'
file_path2 = r'C:\Users\garbu\OneDrive\Área de Trabalho\laranja\get_the_data\ibge_data\get_the_data\ibge_data\2024-04-12 13_23_54.053629_36.json'
file_path3 = r'C:\Users\garbu\OneDrive\Área de Trabalho\laranja\get_the_data\ibge_data\get_the_data\ibge_data\2024-04-12 13_23_54.053629_109.json'
file_path4 = r'C:\Users\garbu\OneDrive\Área de Trabalho\laranja\get_the_data\ibge_data\get_the_data\ibge_data\2024-04-12 13_23_54.053629_216.json'

path_inmet_1 = r'C:\Users\garbu\OneDrive\Área de Trabalho\laranja\get_the_data\time_data\inmet\dados_inmet\dados_A002_M_2007-01-01_2024-03-01.csv'
path_inmet_2 = r'C:\Users\garbu\OneDrive\Área de Trabalho\laranja\get_the_data\time_data\inmet\dados_inmet\dados_A214_M_2008-04-16_2024-03-01.csv'
path_inmet_3 = r'C:\Users\garbu\OneDrive\Área de Trabalho\laranja\get_the_data\time_data\inmet\dados_inmet\dados_A441_M_2008-05-08_2024-03-01.csv'
path_inmet_4 = r'C:\Users\garbu\OneDrive\Área de Trabalho\laranja\get_the_data\time_data\inmet\dados_inmet\dados_A711_M_2008-01-01_2024-04-17.csv'
path_inmet_5 = r'C:\Users\garbu\OneDrive\Área de Trabalho\laranja\get_the_data\time_data\inmet\dados_inmet\dados_A860_M_2008-02-27_2024-03-01.csv'


with open(file_path1, 'r') as file:
    data1 = json.load(file)

with open(file_path2, 'r') as file:
    data2 = json.load(file)

with open(file_path3, 'r') as file:
    data3 = json.load(file)

with open(file_path4, 'r') as file:
    data4 = json.load(file)

    
dados = [data1, data2, data3, data4]
paths_inmet = [path_inmet_1, path_inmet_2, path_inmet_3, path_inmet_4, path_inmet_5]


# %% 

def transform_ibge(dados):
    dfs = {}
    for i, data in enumerate(dados, start=1):
        table_data = []
        for entry in data:
            localidade_nome = entry["localidade_nome"]
            serie_data = entry["data"] 

            for month, producao in serie_data.items():
                table_data.append({
                    "localidade_nome": localidade_nome,
                    "month": month,
                    "producao": producao
                })
        df = pd.DataFrame(table_data)
        df = df[['localidade_nome', 'month', 'producao']]
        df['localidade_nome'] = df['localidade_nome'].astype(str)
        df['month'] = df['month'].apply(lambda x: x.zfill(6))
        df['month'] = pd.to_datetime(df['month'], format='%Y%m')
        df.rename(columns={'producao': f'producao{i}'}, inplace=True)
        df[f'producao{i}'] = df[f'producao{i}'].replace('-', np.nan)
        df[f'producao{i}'] = df[f'producao{i}'].astype(float)
        #df[f'producao{i}'] = df[f'producao{i}'].replace('-', np.nan)
        #df[f'producao{i}'] = df[f'producao{i}'].astype(float)

        dfs[f'df_data{i}'] = df

    return dfs

def inmet_data(paths_inmet):
    for i, path_inmet in enumerate(paths_inmet, start=1):
        with open(path_inmet) as f:
            first_line = f.readline()
            # Extract the "Estado" value from the first line
            estado = first_line.split(": ")[1].strip()

        # Read the CSV file specifying the delimiter and skipping the first 10 rows
        df_inmet = pd.read_csv(path_inmet, delimiter=';', skiprows=10)

        # Drop the extra column 'Unnamed: 4' if it exists
        df_inmet.drop(columns=['Unnamed: 4'], inplace=True, errors='ignore')

        # Convert columns to appropriate data types
        df_inmet['PRECIPITACAO TOTAL, MENSAL (AUT)(mm)'] = df_inmet['PRECIPITACAO TOTAL, MENSAL (AUT)(mm)'].str.replace(',', '.').astype(float)
        df_inmet['TEMPERATURA MEDIA, MENSAL (AUT)(°C)'] = df_inmet['TEMPERATURA MEDIA, MENSAL (AUT)(°C)'].str.replace(',', '.').astype(float)

        # tratamento dos dados de tempo
        df_inmet['Data Medicao'] = pd.to_datetime(df_inmet['Data Medicao'], format='%Y-%m-%d')
        df_inmet = df_inmet[df_inmet['Data Medicao'] >= '2008-05-31']
        # Add a new column 'Estado' with the value extracted from the first line
        #df_inmet[f'estado{i}'] = estado
        df_inmet['Estado'] = estado
        # Reset the index to create a unified index
        df_inmet.reset_index(drop=True, inplace=True)

        # Assign the DataFrame to a variable with a meaningful name
        globals()[f'df_inmet{i}'] = df_inmet

# %%
dfs = transform_ibge(dados)
# %%
# Todos os dfs criados foram:
df_data1 = dfs['df_data1']
df_data2 = dfs['df_data2']
df_data3 = dfs['df_data3']
df_data4 = dfs['df_data4']

# %%
# Unindo todos os dfs criados:
dfs = {
    'df_data1': df_data1,
    'df_data2': df_data2,
    'df_data3': df_data3,
    'df_data4': df_data4
}

df_ibge = dfs['df_data1']

for i in range(2, len(dfs) + 1):
    df_ibge = df_ibge.merge(dfs[f'df_data{i}'], on=['localidade_nome', 'month'], how='left')
# %%
# renomeando as variáveis adequadamente:
df_ibge = df_ibge.rename(columns={
    'producao1': 'producao',
    'producao2': 'rendimento_medio',
    'producao3': 'area_plantada',
    'producao4': 'area_colhida'
})
# %%
inmet_data(paths_inmet)
# %%
# Juntando todos os os df_inmet em um único

dfs_inmet = {}
for i in range(1, 6):
    dfs_inmet[f'df_inmet{i}'] = globals()[f'df_inmet{i}']
    
    
df_inmets = dfs_inmet['df_inmet1']
for i in range(2, 6):
    suffix = f'_df_inmet{i}'
    df_inmets = df_inmets.merge(dfs_inmet[f'df_inmet{i}'], on=['Data Medicao'], how='outer', suffixes=('', suffix))


df_inmets = pd.concat([dfs_inmet[f'df_inmet{i}'] for i in range(1, 6)], ignore_index=True)

df_inmets.sort_values(by=['Data Medicao', 'Estado'], inplace=True)

df_inmets.reset_index(drop=True, inplace=True)

df_inmets = df_inmets.sort_values(['Estado'])

# %%
#-----------------------------------------------------
# importar dados CEPEA = ESSE DATASET SÓ POSSUI DADOS PARA O ESTADO DE SÃO PAULO
df_cepea = pd.read_excel(r'C:\Users\garbu\OneDrive\Área de Trabalho\laranja\get_the_data\cepea_esalq_conab\laranja_precos_cepea_esalq.xlsx')

# %%
# transformar brevemente os dados
df_cepea['Sub-Produto'] = df_cepea['Produto']
df_cepea['Mês'] = df_cepea['Mês'].fillna(12).astype(int)
df_cepea['Ano'] = df_cepea['Ano'].fillna(2054).astype(int)
df_cepea['data'] = pd.to_datetime(df_cepea['Ano'] * 10000 + df_cepea['Mês'] * 100 + 1, format='%Y%m%d') + pd.offsets.MonthEnd()
df_cepea = df_cepea[df_cepea['data'] < '2054-12-31' ]
df_cepea.drop(columns=['Mês', 'Ano'], inplace=True)
df_cepea.drop(columns=['Moeda'], inplace=True)
df_cepea['Produto'] = 'Laranja'
df_cepea['Estado'] = 'São Paulo'
 
# %%

# Vamos unir todos os dataframes.

## Para unirmos todos os dataframes, precisaremos considerar apenas a região de SÃO PAULO pois, só temos preços para a região de SÃO PAULO.

# filtando somente SP em IBGE
df_ibge_sp = df_ibge[df_ibge['localidade_nome'] =='São Paulo']
df_ibge_sp.value_counts(['localidade_nome'])


# filtrando somenter SP em INMETS
df_inmets_sp = df_inmets[df_inmets['Estado'] == 'SAO CARLOS'] #unidade de mensuração no estado de são paulo
df_inmets_sp.value_counts(['Estado'])

# %%
# Para unificar os valores, precisamos padronizar as datas
# CEPEA e INMETS estão como o último dia de cada mês
# IBGE está como o primeiro dia de cada mês, vamos alterar o ibge
df_ibge_sp['month'] = pd.to_datetime(df_ibge_sp['month'] + pd.offsets.MonthEnd(1)) 
# %%
# para unificar os datasets, vamos manter as datas em mesma ordem e a coluna com o mesmo nome
df_ibge_sp.rename(columns ={'month': 'date'}, inplace = True) 
df_inmets_sp.rename(columns={'Data Medicao': 'date'}, inplace = True)
df_cepea.rename(columns={'data': 'date'}, inplace = True)

# %%
df_ibge_sp.rename(columns = {'localidade_nome': 'Estado'}, inplace=True)
df_inmets_sp['Estado'] = 'São Paulo'
# %%
df_cepea['Unidade (kg)'] = df_cepea['Unidade'].str.extract(r'(\d+\.?\d*)').astype(float)
df_cepea = df_cepea.drop(columns=['Unidade','Região'])

# %%
# esse passo é necessário para eliminarmos a variavel 'unidade' para Caixa 40.8kg
df_cepea['preco_ajustado'] = df_cepea.apply(lambda row: (40.8 * row['Preço']) / 27 if row['Unidade (kg)'] == 27 else row['Preço'], axis=1)
df_cepea = df_cepea.drop(columns=['Preço', 'Unidade (kg)'])

# %%
df_cepea_pivot = df_cepea.pivot_table(index=['Produto', 'date', 'Estado'], columns='Sub-Produto', values='preco_ajustado').reset_index()
df_cepea_pivot.columns.name = None
# %%
start_date = pd.to_datetime('2008-05-31')
end_date = pd.to_datetime('2024-02-29')

# Filtrar os dataframes por data para garantir o número de linhas
df_inmets_sp = df_inmets_sp[(df_inmets_sp['date'] >= start_date) & (df_inmets_sp['date'] <= end_date)]
df_cepea_pivot = df_cepea_pivot[(df_cepea_pivot['date'] >= start_date) & (df_cepea_pivot['date'] <= end_date)]
df_ibge_sp = df_ibge_sp[(df_ibge_sp['date'] >= start_date) & (df_ibge_sp['date'] <= end_date)]

# %%
# Prova de que os datasets tem as mesmas datas e também o mesmo número de linhas
print(f'INMET {df_inmets_sp["date"].min()}, IBGE: {df_ibge_sp["date"].min()}, CEPEA : {df_cepea_pivot["date"].min()}')

print(f'INMET {df_inmets_sp["date"].max()}, IBGE: {df_ibge_sp["date"].max()}, CEPEA : {df_cepea_pivot["date"].max()}')

print(f'INMET {df_inmets_sp.shape}, IBGE: {df_ibge_sp.shape}, CEPEA : {df_cepea_pivot.shape}')

# %%
# fazer o merge dos datasets
merge_cepea_ibge = pd.merge(df_cepea_pivot, df_ibge_sp, on=['date', 'Estado'])

df_laranja = pd.merge(merge_cepea_ibge, df_inmets_sp, on=['date', 'Estado'])
# %%
# Retirar as colunas com muitos valores faltantes e também interpolar as colunas com poucos valores faltantes
# Lima ácida = limão (Sim, acabo de descobrir isso e exclui a coluna por que queremos apenas laranjas)
df_laranja = df_laranja.drop(columns=['Produto','Preço da Laranja - Posta - Indústria (Precoce)','Tangerina Ponkan - Árvore - Mercado','Tangor Murcote - Árvore - Mercado','Laranja Natal/Valência (Mercado Árvore)','Laranja Baia - Árvore - Mercado','Lima Ácida Tahiti - Colhida - Mercado'])

# %%
columns_to_interpolate = ['Laranja - Posta - Indústria',
                          'NUMERO DE DIAS COM PRECIP. PLUV, MENSAL (AUT)(número)',
                          'PRECIPITACAO TOTAL, MENSAL (AUT)(mm)',
                          'TEMPERATURA MEDIA, MENSAL (AUT)(°C)']

df_laranja[columns_to_interpolate] = df_laranja[columns_to_interpolate].interpolate(method='linear')

# %%
#exportar para excel
df_laranja.to_excel('df_laranja_sp.xlsx')
