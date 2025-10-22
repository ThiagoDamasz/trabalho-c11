import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_path = r"C:\Users\Thiag\OneDrive\Documentos\Projeto_c11\trabalho-c11\data\data_science_job_posts_2025.csv"

df = pd.read_csv(dataset_path,
delimiter = ',')

print(df.columns)

# Formatação geral do DataFrame
pd.options.display.float_format = '{:.2f}'.format

# Formatação da coluna salary

# tirando os caracteres e espacos da coluna salary
df['salary_clean'] = df['salary'].str.replace('€', '').str.replace(' ', '')

# criando duas colunas para o salário mínimo e máximo
df['salary_min'] = df['salary_clean'].str.split('-').str[0]
df['salary_max'] = df['salary_clean'].str.split('-').str[1]

# Remover vírgulas e converter para float
df['salary_min'] = df['salary_min'].str.replace(',', '').astype(float)
df['salary_max'] = df['salary_max'].str.replace(',', '').astype(float)

# Criar coluna de salário médio
df['salary_avg'] = df[['salary_min', 'salary_max']].mean(axis=1)

# Formatação da coluna location
df['country'] = df['location'].str.split(',').str[2].str.strip()

#1 Qual o salário médio por cargo?
print("\n=== Qual o salário médio por cargo(anual)? ===")
salario_cargo = df.groupby('job_title')['salary_avg'].mean().sort_values()
print(salario_cargo)

plt.figure(figsize=(10,6))
plt.bar(salario_cargo.index[::1], salario_cargo.values[::1], color='palegreen', width=0.8)
plt.title('Cargos com maior salário médio anual (€)', fontsize = 14)
plt.xlabel('Cargo', fontsize = 10)
plt.ylabel('Salário médio anual(€)', fontsize = 12)
plt.grid(axis='y', alpha=0.6)
plt.show()

#2 Qual a diferença de salário médio por nível de senioridade (junior, pleno, senior) ? (Em numpy)
seniority = df['seniority_level'].fillna('Unknown').to_numpy()
salary = df['salary_avg'].to_numpy()
unique_levels, inverse_idx = np.unique(seniority, return_inverse=True) # niveis unicos e indices inversos
salary_mean_by_level = np.bincount(inverse_idx, weights=salary) / np.bincount(inverse_idx)
print("\n=== Qual o salário médio por nível de senioridade? ===")
for level, avg in zip(unique_levels, salary_mean_by_level):
    print(f"{level}: {avg:.2f}")


#3 Qual o salário médio por regime de trabalho (hibrído, presencial ou remoto) ?
salario_medio_por_regime = df.groupby('status')['salary_avg'].mean().to_string()
print("\n===Qual o salário médio por regime de trabalho? ===")
print(salario_medio_por_regime)

#4 Qual país paga os maiores salários para profissionais de Data Science? 
print("\n===Quais são os países que possuem os melhores salários? ===")
salario_localizacao = df.groupby('country')['salary_avg'].mean().to_string()
print(salario_localizacao)
#Melhorar a resposta que so esta mostrando em ordem alfabetica

#5 Qual empresa contrata mais profissionais de Data Science?
print("\n=== Quais empresas mais contratam profissionais de Data Science? ===")
empresas_cont = df['company'].value_counts().head(5)
print(empresas_cont.to_string())

plt.figure(figsize=(10,6))
plt.bar(empresas_cont.index[::1], empresas_cont.values[::1], color='paleturquoise', width=0.6)
plt.title('Quantidade de posts de vagas de emprego', fontsize=14)
plt.xlabel('Numero da empresa', fontsize = 12)
plt.ylabel('Quantidade de posts', fontsize = 11)
plt.grid(axis='y', alpha=0.6)
plt.show()

#6 Qual o nivel de experiencia mais requisitado pelas empresas?
print("\n=== Qual o nivel de experiencia mais requisitado pelas empresas? ===")
nivel_exp = df['seniority_level'].value_counts().to_string()
print(nivel_exp)

#7 Quais sao as 5 empresas com mais receita no mercado? (numpy)
print("\n=== Quais as 5 empresas com mais receita no mercado? ===")
company = df['company'].to_numpy()
size = df['company_size'].to_numpy()

size_str = np.array(size, dtype=str)
size_str[size_str == 'nan'] = '0'

# Limpeza dos caracteres
size_clean = np.char.replace(size_str, '"', '')
size_clean = np.char.replace(size_clean, '€', '')
size_clean = np.char.replace(size_clean, ',', '')
size_clean = np.char.replace(size_clean, 'B', '')
size_clean = np.char.replace(size_clean, '.', '')
size_clean = np.char.replace(size_clean, 'Private', '')

# Colocar 0 onde não for número
size_clean = np.where(np.char.isdigit(size_clean), size_clean, '0')

# Converter para inteiro
size_num = size_clean.astype(int)

# Remover repetições de empresas (mantendo o primeiro registro)
unique_companies, unique_indices = np.unique(company, return_index=True)
unique_sizes = size_num[unique_indices]

# Pegar os índices das 5 maiores empresas únicas
top_idx = np.argsort(unique_sizes)[-5:][::-1]

# Exibir resultado com for, linha por linha
print("As 5 empresas com mais receita no mercado são:")
for i in top_idx:
    print(f"{unique_companies[i]} ({unique_sizes[i]})")


#alguem arruma essa pergunta ai porfavor pois nao consegui

#8 Quais sao as habilidades mais requisitadas pelas empresas?
print("\n=== Quais sao as habilidades mais requisitadas pelas empresas? ===")
df['skills_clean'] = (
    df['skills']
    .astype(str)
    .str.replace('[\[\]\']', '', regex=True)
)
df['skills_list'] = df['skills_clean'].str.split(', ')

df_exploded = df.explode('skills_list')
df_exploded['skills_list'] = df_exploded['skills_list'].str.strip()
df_exploded = df_exploded[df_exploded['skills_list'].astype(bool)]

habilidades = df_exploded['skills_list'].value_counts().head(10)
print(habilidades)

plt.figure(figsize=(8, 8))
plt.pie(
    habilidades.values,
    labels=habilidades.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.tab10.colors,
    wedgeprops={'edgecolor': 'white'}
)

plt.title('As 10 habilidades mais requisitadas pelas empresas', fontsize=14)
plt.tight_layout()
plt.show()

#9 Qual tipo de empresa (Publica ou Privada) que melhor paga os funcionarios?
print("\n=== Qual o tipo de empresa (Public ou Private) está pagando mais? ===")
df = df.dropna(subset=["ownership", "salary_avg"])
tipo_emp = df.groupby("ownership")["salary_avg"].mean().sort_values(ascending=False)
print(tipo_emp)

#10 Quais setores da Industria que menos utilizam data science?
print("\n=== Qual setor da Industria contrata menos profissionais de data science? ===")
industria = df['industry'].value_counts().sort_values()
print(industria.head(3))