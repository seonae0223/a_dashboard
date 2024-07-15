import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit 애플리케이션 제목
st.title("무신사 리뷰 데이터 클러스터링")

# 업로드된 Excel 파일 경로
file_path = './data/keyword.xlsx'

# Excel 파일을 읽기
df = pd.read_excel(file_path)
st.write("업로드된 데이터:")
st.write(df)

# 클러스터링에 사용할 카테고리 선택
category_col = st.selectbox("카테고리 열을 선택하세요", df.columns)

# 클러스터 수 선택
num_clusters = st.slider("클러스터 수를 선택하세요", min_value=2, max_value=10, value=3)

# 클러스터링 수행
if st.button("클러스터링 실행"):
    # 카테고리 데이터를 숫자로 변환
    df['category_numeric'] = pd.factorize(df[category_col])[0]

    # K-means 모델 생성 및 학습
    kmeans = KMeans(n_clusters=num_clusters)
    df['cluster'] = kmeans.fit_predict(df[['category_numeric']])

    st.write("클러스터링 결과:")
    st.write(df)

    # 클러스터 시각화
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df.index, y='category_numeric', hue='cluster', data=df, palette='viridis')
    plt.title('카테고리별 클러스터링 결과')
    plt.xlabel('Index')
    plt.ylabel(category_col)
    st.pyplot(plt)

# Streamlit 애플리케이션 실행
if __name__ == '__main__':
    st.run()
