import pandas as pd
import numpy as np

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
salario_cargo = df.groupby('job_title')['salary_avg'].mean().to_string()
print(salario_cargo)

#2 Qual a diferença de salário médio por nível de senioridade (junior, pleno, senior) ?
print("\n=== Qual o salário médio por nível de senioridade? ===")
salario_senioridade = df.groupby('seniority_level')['salary_avg'].mean().to_string()
print(salario_senioridade)

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
empresas_cont = df['company'].value_counts().head(10)
print(empresas_cont.to_string())

#6 Qual o nivel de experiencia mais requisitado pelas empresas?
print("\n=== Qual o nivel de experiencia mais requisitado pelas empresas? ===")
nivel_exp = df['seniority_level'].value_counts().to_string()
print(nivel_exp)

#7 Quais sao as 5 empresas com mais receita no mercado?
print("\n=== Quais as 5 empresas com mais receita no mercado? ===")

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

#9 Qual tipo de empresa (Publica ou Privada) que melhor paga os funcionarios?
print("\n=== Qual o tipo de empresa (Public ou Private) está pagando mais? ===")
df = df.dropna(subset=["ownership", "salary_avg"])
tipo_emp = df.groupby("ownership")["salary_avg"].mean().sort_values(ascending=False)
print(tipo_emp)

#10 Quais setores da Industria que menos utilizam data science?
print("\n=== Qual setor da Industria contrata menos profissionais de data science? ===")
industria = df['industry'].value_counts().sort_values()
print("\nIndustrias que menos utilizam:")
print(industria.head(3))