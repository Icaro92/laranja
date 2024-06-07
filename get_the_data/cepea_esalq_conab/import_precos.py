# %%
import pandas as pd

# %%

# Base de dados importada do institudo CEPEA/ESALQ - Centro de Estudos Avançados em Economia Aplicada - CEPEA Esalq/USP
df_cepea = pd.read_excel(r'C:\Users\garbu\OneDrive\Área de Trabalho\laranja\get_the_data\cepea_esalq_conab\laranja_precos_cepea_esalq.xlsx')
df_cepea.head()

# %%
# dados importados da CONAB: Companhia Nacional de Abastecimento

df_conab = pd.read_excel('C:\\Users\\garbu\\OneDrive\\Área de Trabalho\\laranja\\get_the_data\\cepea_esalq_conab\\Laranja_precos_conab.xlsx')
df_conab.head(1)
# %%
