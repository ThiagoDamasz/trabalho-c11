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



#Perguntas:

#1 Qual o salário médio por cargo?
salario_medio_por_cargo = df.groupby('job_title')['salary_avg'].mean().to_string()
print("\n=== Qual o salário médio por cargo? ===")
print(salario_medio_por_cargo)

#2 Qual a diferença de salário médio por nível de senioridade (junior, pleno, senior) ?
salario_medio_por_senioridade = df.groupby('seniority_level')['salary_avg'].mean().to_string()
print("\n=== Qual o salário médio por nível de senioridade? ===")
print(salario_medio_por_senioridade)

#3 Qual o salário médio por regime de trabalho (hibrído, presencial ou remoto) ?
salario_medio_por_regime = df.groupby('status')['salary_avg'].mean().to_string()
print("\n===Qual o salário médio por regime de trabalho? ===")
print(salario_medio_por_regime)

#4 Qual país paga os maiores salários para profissionais de Data Science? 
salario_medio_por_localizacao = df.groupby('country')['salary_avg'].mean().to_string()
print("\n===Quais são os países que possuem os melhores salários? ===")
print(salario_medio_por_localizacao)
#TODO: CONSERTAR A FORMATAÇÃO DESSE DATASET

#5 Qual empresa contrata mais profissionais de Data Science?
print("\n=== Quais empresas mais contratam profissionais de Data Science? ===")
empresas_mais_ativas = df['company'].value_counts().head(10)
print(empresas_mais_ativas.to_string())
#TODO: MELHORAR ESSA PERGUNTA 

#6 Qual o nivel de experiencia mais requisitado pelas empresas?
print("\n=== Qual o nivel de experiencia mais requisitado pelas empresas? ===")
nivel_exp = df['seniority_level'].value_counts()
print("\nNiveis de experiencia mais requisitados pelas empresas:")
print(nivel_exp)

#7 Quais sao as 5 empresas com mais receita no mercado?
print("\n=== Quais as 5 empresas com mais receita no mercado? ===")

#8 Quais sao as habilidades mais requisitadas pelas empresas?

#print("\n=== Quais sao as habilidades mais requisitadas pelas empresas? ===")
#df['skills'] = df['skills'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
#todas_habilidades = [hab for sublista in df['habilidades'] for hab in sublista]
#contagem = Counter(todas_habilidades)
#top3 = contagem.most_common(3)
#print("As 3 habilidades mais comuns são:")
#for skills, freq in top3:#print(f"{skills}: {freq}"}

#9 Qual tipo de empresa (Publica ou Privada) que melhor paga os funcionarios?
df["salario_medio"] = df["salary"].mean()
df = df.dropna(subset=["salario_medio", "ownership"])
media_por_tipo = df.groupby("ownership")["salario_medio"].mean().sort_values(ascending=False)
print("Salário médio por tipo de empresa (€):")
print(media_por_tipo)

#10 Qual setor da Industria mais paga os funcionarios?