import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit 설정
st.set_page_config(layout="wide")

# 파일 경로 설정
file_path = 'C:/Users/user/Desktop/a_dashboard/data/Foreign_travel.xlsx'

# 데이터 불러오기
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 'Q5_1a99' 열에서 'del' 문자열 제거
df['Q5_1a99'] = pd.to_numeric(df['Q5_1a99'], errors='coerce')

# 제외할 변수들 리스트
exclude_vars = ['Q9_1_1', 'Q9_1_2', 'Q9_1_3', 'Q9_1_4', 'Q9_1_5'] + \
               [f'Q12a{str(i).zfill(2)}' for i in range(1, 27)] + \
               ['Q13', 'Q14']

# 제외할 변수들 제거
df_filtered = df.drop(columns=exclude_vars, errors='ignore')

# 'Q11'과 다른 변수들 간의 상관관계 계산
quality_corr = df_filtered.corr()['Q11']

# 자기 자신과의 상관관계 제거
quality_corr.drop('Q11', axis=0, inplace=True)

# 상관관계가 높은 순서대로 20개의 변수 추출
top_20_corr = quality_corr.sort_values(ascending=False).head(20)

# 결과 요약 출력
st.write("Top 20 variables most correlated with Q11:")
st.write(top_20_corr)

most_correlated_var = top_20_corr.index[0]
least_correlated_var = top_20_corr.index[-1]
summed = round(top_20_corr.iloc[0] + top_20_corr.iloc[-1], 2)

st.write(f'The most correlated variable with Q11 is: {most_correlated_var}')
st.write(f'The least correlated variable with Q11 in the top 20 is: {least_correlated_var}')
st.write(f'And the sum of their correlations is: {summed}')

# 히트맵을 그리기 위한 상관관계 행렬 생성
top_20_vars = top_20_corr.index.tolist() + ['Q11']
corr_matrix = df_filtered[top_20_vars].corr()

# 히트맵 출력
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Heatmap of Top 20 Variables with Q11')
plt.xticks(rotation=45, ha='right')
plt.yticks()
st.pyplot(plt)
