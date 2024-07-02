import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Previsão de Renda",
    page_icon="$",
)

st.markdown('# Projeto Previsão de Renda')
st.markdown('Este é um projeto com objetivo de criar um modelo preditivo para os valores de renda com base nos demais dados, a renda será declarada no projeto como a variável resposta ou target, as demais informações serão chamadas como variáveis explicativas. Como objetivo secundário, esse projeto vai ser usado como um estudo do modelo CRISP e suas fases, sendo assim os comentários realizados tem não somente a função de ajudar na manutenção e entendimento do código, mas também como uma fonte de aprendizado.Ao decorrer do projeto todas as informações serão análisadas separadamente, os dados vão ser trabalhados e editados para que fiquem em um formato útil e possam ser aplicados a etapa de modelagem, nessa etapa serão aplicados conceitos estatisticos e de aprendizagem de máquina como Árvore de Decisão (decisionTree) e Árvores Aleatórias (randomForest), além de usar modelos diferentes para o mesmo propósito para comparação de resultado.')


df = pd.read_csv("C:\\Users\\Bill_\\Desktop\\Projetos\\EBAC\\DataScience\\Modulo 16 - Pandas e Projeto\\previsao_de_renda.csv") # Carregando o quadro de dados para previsão de rendas
df = df.drop('sexo', axis = 1) # Removendo colunas de não interesse (devido a LGPD)
df = df.drop(['Unnamed: 0', 'data_ref', 'id_cliente'], axis=1) # Colunas que não vão ajudar em nada nas análises
#
filtro_nulo = df['tempo_emprego'].isnull() # Salvando os valores nulos como True/False
df_nulo_renda = df[filtro_nulo][['tempo_emprego', 'tipo_renda','renda']] # Pegando os valores True(nulos) junto com as demais colunas
#
def gerar_intervalos_e_rotulos(margem, limite):
    if limite % margem != 0: # Verifica resto da divisão
        limite_c_margem = (limite - (limite % margem)) + margem # Altera para o próximo valor divisivel por margem
    else:
        limite_c_margem = limite # Mantem o valor caso a divisão seja inteira
    i = 0 
    intervalos = []
    rotulos = []
    while i <= limite_c_margem:
        intervalos.append(i) # Adiciona os valores de intervalo de acordo com a margem
        if i < limite_c_margem:
            rotulos.append(f'{i}-{i + margem}') # Adiciona os rotulos, exemplo 500-1000 ou 300-600
        else:
            rotulos.append(f'{i}+') # Adiciona o ultimo elemento do rotulo e o simbolo +, define sem limite superior
        i = i + margem
    intervalos.append(float('inf')) # Adiciona por ultimo 'inf' que é infinito, sem limite superior
    
    return intervalos, rotulos # Retorna as listas
#
df_clone = df.copy() # Gerando um DF novo pra poder mexer sem alterar o original
intervalos, rotulos = gerar_intervalos_e_rotulos(1000, 9000) # Passando os valores pra função

df_clone['renda_intervalo'] = pd.cut(df_clone['renda'], bins=intervalos, labels=rotulos, right=False)


# INFORMAÇÕES DA POSSE DE VEICULO
st.markdown("# Posse de Veículo")
st.markdown("#### Tabela de faixa de valores de 1000 em 1000 das pessoas que tem carro(True) e das pessoas que não tem carro (False).")
info_cross = pd.crosstab(
    index=df_clone['posse_de_veiculo'],
    columns=df_clone['renda_intervalo'], 
    margins=True)
info_cross
st.markdown("#### Valores proporcionais.")
#
info_cross_proporcional = pd.crosstab(
    index=df_clone['posse_de_veiculo'],
    columns=df_clone['renda_intervalo'], 
    margins=True,
    normalize='columns')# Divide os valores pelo total
info_cross_proporcional = info_cross_proporcional.applymap(lambda x: f'{x*100:.2f}%') # Vezes 100, deixa 2 casas decimais e %
info_cross_proporcional
#
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
info_cross.iloc[:-1,:-1].plot(kind='bar', stacked=True, ax=ax)
ax.set_ylabel('Frequência')
ax.set_xlabel('Posse de Veículo')
ax.set_title('Distribuição de Posse de Veículo por Faixa de Renda')
ax.legend(title='Renda', bbox_to_anchor=(1, 1), loc='upper left')
st.pyplot(fig)


# INFORMAÇÕES DA POSSE DE IMÓVEL
st.markdown("# Posse de Imóvel")
st.markdown("#### Tabela de faixa de valores de 1000 em 1000 das pessoas que tem imóvel(True) e das pessoas que não tem imóvel (False).")
info_cross = pd.crosstab(
    index=df_clone['posse_de_imovel'],
    columns=df_clone['renda_intervalo'], 
    margins=True)
info_cross
#
st.markdown("#### Valores proporcionais.")
info_cross_proporcional = pd.crosstab(
    index=df_clone['posse_de_imovel'],
    columns=df_clone['renda_intervalo'], 
    margins=True,
    normalize='columns')# Divide os valores pelo total
info_cross_proporcional = info_cross_proporcional.applymap(lambda x: f'{x*100:.2f}%') # Vezes 100, deixa 2 casas decimais e %
info_cross_proporcional
#
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
info_cross.iloc[:-1,:-1].plot(kind='bar', stacked=True, ax=ax)
ax.set_ylabel('Frequência')
ax.set_xlabel('Posse de Imóvel')
ax.set_title('Distribuição de Posse de Imóvel por Faixa de Renda')
ax.legend(title='Renda', bbox_to_anchor=(1, 1), loc='upper left')
st.pyplot(fig)


# INFORMAÇÕES DO TIPO DE RENDA
st.markdown("# Tipo de Renda")
st.markdown("#### Tabela de faixa de valores de 1000 em 1000 dos tipos de renda e a quantidade de pessoas do respectivo tipo de renda e faixa de valor.")
info_cross = pd.crosstab(
    index=df_clone['tipo_renda'],
    columns=df_clone['renda_intervalo'], 
    margins=True)
info_cross
#
st.markdown("#### Valores proporcionais.")
info_cross_proporcional = pd.crosstab(
    index=df_clone['tipo_renda'],
    columns=df_clone['renda_intervalo'], 
    margins=True,
    normalize='columns')# Divide os valores pelo total
info_cross_proporcional = info_cross_proporcional.applymap(lambda x: f'{x*100:.2f}%') # Vezes 100, deixa 2 casas decimais e %
info_cross_proporcional
#
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
info_cross.iloc[:-1,:-1].plot(kind='bar', stacked=True, ax=ax)
ax.set_ylabel('Frequência')
ax.set_xlabel('Tipo de Renda')
ax.set_title('Distribuição de Tipo de Renda por Faixa de Renda')
ax.legend(title='Renda', bbox_to_anchor=(1, 1), loc='upper left')
st.pyplot(fig)


# INFORMAÇÕES DO GRAU DE EDUCAÇÃO
st.markdown("# Educação")
st.markdown("#### Tabela de faixa de valores de 1000 em 1000 dos tipos de educação e a quantidade de pessoas do respectivo tipo de educação e faixa de valor.")
info_cross = pd.crosstab(
    index=df_clone['educacao'],
    columns=df_clone['renda_intervalo'], 
    margins=True)
info_cross
#
st.markdown("#### Valores proporcionais.")
info_cross_proporcional = pd.crosstab(
    index=df_clone['educacao'],
    columns=df_clone['renda_intervalo'], 
    margins=True,
    normalize='columns')# Divide os valores pelo total
info_cross_proporcional = info_cross_proporcional.applymap(lambda x: f'{x*100:.2f}%') # Vezes 100, deixa 2 casas decimais e %
info_cross_proporcional
#
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
info_cross.iloc[:-1,:-1].plot(kind='bar', stacked=True, ax=ax)
ax.set_ylabel('Frequência')
ax.set_xlabel('Educação')
ax.set_title('Distribuição de Tipo de Educação por Faixa de Renda')
ax.legend(title='Renda', bbox_to_anchor=(1, 1), loc='upper left')
st.pyplot(fig)


# INFORMAÇÕES DE ESTADO CIVIL
st.markdown("# Estado Civil")
st.markdown("#### Tabela de faixa de valores de 1000 em 1000 dos tipos de estado civil e a quantidade de pessoas do respectivo tipo de estado civil e faixa de valor.")
info_cross = pd.crosstab(
    index=df_clone['estado_civil'],
    columns=df_clone['renda_intervalo'], 
    margins=True)
info_cross
#
st.markdown("#### Valores proporcionais.")
info_cross_proporcional = pd.crosstab(
    index=df_clone['estado_civil'],
    columns=df_clone['renda_intervalo'], 
    margins=True,
    normalize='columns')# Divide os valores pelo total
info_cross_proporcional = info_cross_proporcional.applymap(lambda x: f'{x*100:.2f}%') # Vezes 100, deixa 2 casas decimais e %
info_cross_proporcional
#
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
info_cross.iloc[:-1,:-1].plot(kind='bar', stacked=True, ax=ax)
ax.set_ylabel('Frequência')
ax.set_xlabel('Estado Civil')
ax.set_title('Distribuição de Estado Civil por Faixa de Renda')
ax.legend(title='Renda', bbox_to_anchor=(1, 1), loc='upper left')
st.pyplot(fig)



# INFORMAÇÕES DO TIPO DE RESIDÊNCIA
st.markdown("# Tipo de Residência")
st.markdown("#### Tabela de faixa de valores de 1000 em 1000 dos tipos de residência e a quantidade de pessoas do respectivo tipo de residência e faixa de valor.")
info_cross = pd.crosstab(
    index=df_clone['tipo_residencia'],
    columns=df_clone['renda_intervalo'], 
    margins=True)
info_cross
#
st.markdown("#### Valores proporcionais.")
info_cross_proporcional = pd.crosstab(
    index=df_clone['tipo_residencia'],
    columns=df_clone['renda_intervalo'], 
    margins=True,
    normalize='columns')# Divide os valores pelo total
info_cross_proporcional = info_cross_proporcional.applymap(lambda x: f'{x*100:.2f}%') # Vezes 100, deixa 2 casas decimais e %
info_cross_proporcional
#
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
info_cross.iloc[:-1,:-1].plot(kind='bar', stacked=True, ax=ax)
ax.set_ylabel('Frequência')
ax.set_xlabel('Tipo de Residência')
ax.set_title('Distribuição de Tipo de Residência por Faixa de Renda')
ax.legend(title='Renda', bbox_to_anchor=(1, 1), loc='upper left')
st.pyplot(fig)



# INFORMAÇÕES DA QUANTIDADE DE FILHOS
st.markdown("# Quantidade de Filhos")
st.markdown("#### Tabela de faixa de valores de 1000 em 1000 da quantidade de filhos e a quantidade de filhos da respectiva faixa de valor.")
#
info_cross = pd.crosstab(
    index=df_clone['qtd_filhos'],
    columns=df_clone['renda_intervalo'], 
    margins=True)
info_cross
#
st.markdown("#### Valores proporcionais.")
info_cross_proporcional = pd.crosstab(
    index=df_clone['qtd_filhos'],
    columns=df_clone['renda_intervalo'], 
    margins=True,
    normalize='columns')# Divide os valores pelo total
info_cross_proporcional = info_cross_proporcional.applymap(lambda x: f'{x*100:.2f}%') # Vezes 100, deixa 2 casas decimais e %
info_cross_proporcional
#
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x='qtd_filhos', y='renda', data=df)
plt.xlabel('Quantidade de Filhos')
plt.ylabel('Renda Média')
plt.title('Relação entre Quantidade de Filhos e Renda Média')
st.pyplot()



# INFORMAÇÕES SOBRE QUANTIDADE DE PESSOAS RESIDÊNCIA
st.markdown("# Quantidade de Pessoas Residência")
st.markdown("#### Tabela de faixa de valores de 1000 em 1000 da quantidade de pessoas na residência.")
info_cross = pd.crosstab(
    index=df_clone['qt_pessoas_residencia'],
    columns=df_clone['renda_intervalo'], 
    margins=True)
info_cross
#
st.markdown("#### Valores proporcionais.")
info_cross_proporcional = pd.crosstab(
    index=df_clone['qt_pessoas_residencia'],
    columns=df_clone['renda_intervalo'], 
    margins=True,
    normalize='columns')# Divide os valores pelo total
info_cross_proporcional = info_cross_proporcional.applymap(lambda x: f'{x*100:.2f}%') # Vezes 100, deixa 2 casas decimais e %
info_cross_proporcional
#
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
info_cross.iloc[:-1,:-1].plot(kind='bar', stacked=True, ax=ax)
ax.set_ylabel('Frequência')
ax.set_xlabel('Tipo de Residência')
ax.set_title('Distribuição de Tipo de Residência por Faixa de Renda')
ax.legend(title='Renda', bbox_to_anchor=(1, 1), loc='upper left')
st.pyplot(fig)


# INFORMAÇÕES SOBRE IDADE
st.markdown("# Idade")
st.markdown("#### Tabela de faixa de valores de 1000 em 1000 da idade.")
df_clone = df.copy() # Gerando um DF novo pra poder mexer sem alterar o original
intervalos, rotulos = gerar_intervalos_e_rotulos(1000, 9000) # Passando os valores de renda para função
intervalos_idade, rotulos_idade = gerar_intervalos_e_rotulos(5, 60) # Valores de idade para função
df_clone['renda_intervalo'] = pd.cut(df_clone['renda'], bins=intervalos, labels=rotulos, right=False)
df_clone['idade_intervalo'] = pd.cut(df_clone['idade'], bins=intervalos_idade, labels=rotulos_idade, right=False)
# Crie uma tabela de informações cruzadas usando pd.crosstab(), margins=True inclui os totais
info_cross = pd.crosstab(
    index=df_clone['idade_intervalo'],
    columns=df_clone['renda_intervalo'], 
    margins=True)
info_cross
#
st.markdown("#### Valores proporcionais.")
info_cross_proporcional = pd.crosstab(
    index=df_clone['idade_intervalo'],
    columns=df_clone['renda_intervalo'], 
    margins=True,
    normalize='columns')# Divide os valores pelo total
info_cross_proporcional = info_cross_proporcional.applymap(lambda x: f'{x*100:.2f}%') # Vezes 100, deixa 2 casas decimais e %
info_cross_proporcional
#
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
info_cross.iloc[:-1,:-1].plot(kind='bar', stacked=True, ax=ax)
ax.set_ylabel('Frequência')
ax.set_xlabel('Idade')
ax.set_title('Distribuição de Renda por Idade')
ax.legend(title='Renda', bbox_to_anchor=(1, 1), loc='upper left')
st.pyplot(fig)


# INFORMAÇÕES DE TEMPO EMPREGO
st.markdown("# Tempo Emprego")
st.markdown("#### Tabela de faixa de valores de 1000 em 1000 de acordo com o tempo de emprego.")
maximo = df['tempo_emprego'].max()
df_clone = df.copy() # Gerando um DF novo pra poder mexer sem alterar o original
# Passando os valores de renda para função
intervalos, rotulos = gerar_intervalos_e_rotulos(1000, 9000) 
# Valores de tempo de emprego para função
intervalos_t_emprego, rotulos_t_emprego = gerar_intervalos_e_rotulos(5, maximo)
df_clone['renda_intervalo'] = pd.cut(df_clone['renda'], bins=intervalos, labels=rotulos, right=False)
df_clone['t_emprego_intervalo'] = pd.cut(df_clone['tempo_emprego'], bins=intervalos_t_emprego, labels=rotulos_t_emprego, right=False)
info_cross = pd.crosstab(
    index=df_clone['t_emprego_intervalo'],
    columns=df_clone['renda_intervalo'], 
    margins=True)
info_cross
#
st.markdown("#### Valores proporcionais.")
info_cross_proporcional = pd.crosstab(
    index=df_clone['t_emprego_intervalo'],
    columns=df_clone['renda_intervalo'], 
    margins=True,
    normalize='columns')# Divide os valores pelo total
info_cross_proporcional = info_cross_proporcional.applymap(lambda x: f'{x*100:.2f}%') # Vezes 100, deixa 2 casas decimais e %
info_cross_proporcional
#
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
info_cross.iloc[:-1,:-1].plot(kind='bar', stacked=True, ax=ax)
ax.set_ylabel('Frequência')
ax.set_xlabel('Tempo de Emprego')
ax.set_title('Distribuição de Renda por Tempo de Emprego')
ax.legend(title='Renda', bbox_to_anchor=(1, 1), loc='upper left')
st.pyplot(fig)




