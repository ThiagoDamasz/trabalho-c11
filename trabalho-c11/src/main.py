import pandas as pd
import numpy as np

dataset_path = r"C:\Users\Thiag\OneDrive\Documentos\Projeto_c11\trabalho-c11\data\data_science_job_posts_2025.csv"

df = pd.read_csv(dataset_path,
delimiter = ',')

print(df.columns)


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

#1 Qual o salário médio por cargo?
salario_medio_por_cargo = df.groupby('job_title')['salary_avg'].mean()
print("Salário médio por cargo:")
print(salario_medio_por_cargo)

#2 Qual a diferença de salário médio por nível de senioridade (junior, pleno, senior) ?
salario_medio_por_senioridade = df.groupby('seniority_level')['salary_avg'].mean()
print("Salário médio por nível de senioridade:")
print(salario_medio_por_senioridade)

#3 Qual o salário médio por regime de trabalho (hibrído, presencial ou remoto) ?
salario_medio_por_regime = df.groupby('status')['salary_avg']