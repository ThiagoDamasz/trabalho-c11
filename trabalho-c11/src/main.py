import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_path = r"C:\Users\Thiag\OneDrive\Documentos\Projeto_c11\trabalho-c11\data\data_science_job_posts_2025.csv"

df = pd.read_csv(dataset_path,
delimiter = ',')

print(df.columns)

pd.options.display.float_format = '{:.2f}'.format

df['salary_clean'] = df['salary'].str.replace('€', '').str.replace(' ', '')

df['salary_min'] = df['salary_clean'].str.split('-').str[0]
df['salary_max'] = df['salary_clean'].str.split('-').str[1]

df['salary_min'] = df['salary_min'].str.replace(',', '').astype(float)
df['salary_max'] = df['salary_max'].str.replace(',', '').astype(float)

df['salary_avg'] = df[['salary_min', 'salary_max']].mean(axis=1)

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

#4 Quais localizacoes mais pagam salários para profissionais de Data Science? 
print("\n===Quais são as 5 localizações das empresas que pagam os maiores salários anuais? ===")
salario_localizacao = df.groupby('country')['salary_avg'].mean().sort_values(ascending=False)
localizacao = salario_localizacao.head(5)
print(localizacao)

plt.figure(figsize=(10,6))
plt.barh(localizacao.index[::-1], localizacao.values[::-1], color='violet', height=0.6)
plt.title('5 lugares com maiores pagamentos', fontsize=14)
plt.ylabel('Localização', fontsize=12)
plt.xlabel('Salário médio anual (€)', fontsize=11)
plt.grid(axis='x', alpha=0.6)
plt.tight_layout()
plt.show()

#5 Quais empresas contrata mais profissionais de Data Science?
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
nivel_exp = df['seniority_level'].value_counts()
print(nivel_exp)

plt.figure(figsize=(8, 8))
plt.pie(
    nivel_exp.values,
    labels=nivel_exp.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.tab10.colors,
    wedgeprops={'edgecolor': 'white'}
)

plt.title('Os níveis de experiencia mais requisitados pelas empresas', fontsize=14)
plt.tight_layout()
plt.show()

#7 Quais sao as 5 empresas com mais receita no mercado? (numpy)
print("\n=== Quais as 5 empresas com mais receita no mercado? ===")
company = df['company'].to_numpy()
size = df['company_size'].to_numpy()

size_str = np.array(size, dtype=str)
size_str[size_str == 'nan'] = '0'

size_clean = np.char.replace(size_str, '"', '')
size_clean = np.char.replace(size_clean, '€', '')
size_clean = np.char.replace(size_clean, ',', '')
size_clean = np.char.replace(size_clean, 'B', '')
size_clean = np.char.replace(size_clean, '.', '')
size_clean = np.char.replace(size_clean, 'Private', '')

size_clean = np.where(np.char.isdigit(size_clean), size_clean, '0')

size_num = size_clean.astype(int)

unique_companies, unique_indices = np.unique(company, return_index=True)
unique_sizes = size_num[unique_indices]

top_idx = np.argsort(unique_sizes)[-5:][::-1]

print("As 5 empresas com mais receita no mercado são:")
for i in top_idx:
    print(f"{unique_companies[i]} ({unique_sizes[i]})")

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