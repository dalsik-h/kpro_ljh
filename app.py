import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.font_manager as fm

st.set_page_config(page_title="KMeans Cluster 분석", layout="wide")
st.title("KMeans 기반 시계열 군집 분석")

# 한글 폰트 설정 (NanumGothic 설치된 경우 우선 사용)
font_path = './NanumGothic-Regular.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로딩 및 전처리 캐싱
@st.cache_data
def load_and_process():
    df = pd.read_csv('./pip_dataset_pro.csv', index_col='date_time', parse_dates=True)
    df.drop(['thj_vv_open_3'], axis=1, inplace=True)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df, pd.DataFrame(df_scaled, columns=df.columns)

st.subheader("1. 데이터 불러오기")
df, df_scaled = load_and_process()
st.success("데이터 불러오기 성공")
st.write(df.head())

# Elbow Method: 버튼 클릭 시 실행
st.subheader("2. 최적 군집 수 찾기 (Elbow Method)")
if st.button("Elbow Method 실행"):
    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)

    fig1 = plt.figure(figsize=(5, 3))
    plt.plot(K_range, inertia, marker='o')
    plt.xlabel('k (Number of Clusters)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.grid()
    st.pyplot(fig1)

# 클러스터 평가 함수
@st.cache_data
def evaluate_k(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return {
        "silhouette": silhouette_score(X, labels),
        "calinski": calinski_harabasz_score(X, labels),
        "davies": davies_bouldin_score(X, labels)
    }

st.subheader("3. 클러스터 품질 평가 (k=3, 4)")
if st.button("품질 지표 보기"):
    for k in [3, 4]:
        result = evaluate_k(df_scaled, k)
        st.markdown(f"**▶ K = {k}**")
        st.markdown(f"- Silhouette Score: {result['silhouette']:.4f}")
        st.markdown(f"- Calinski-Harabasz Index: {result['calinski']:.2f}")
        st.markdown(f"- Davies-Bouldin Index: {result['davies']:.4f}")

# 클러스터링 실행
st.subheader("4. 클러스터링 실행")
clusters = None
if st.button("클러스터링 수행"):
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)
    df['cluster'] = clusters
    st.success("클러스터링 완료!")

    st.subheader("5. 클러스터별 평균값")
    st.dataframe(df.groupby('cluster').mean().round(2))

    # 시각화 탭 구분
    st.subheader("6. 클러스터 시각화")
    tab1, tab2 = st.tabs(["PCA", "t-SNE"])

    with tab1:
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(df_scaled)
        fig2 = plt.figure(figsize=(4, 2))
        plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('PCA Method')
        plt.colorbar(label='Cluster')
        plt.grid()
        st.pyplot(fig2)

    with tab2:
        if st.button("t-SNE 시각화 실행"):
            tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
            tsne_result = tsne.fit_transform(df_scaled)
            df['tsne_1'] = tsne_result[:, 0]
            df['tsne_2'] = tsne_result[:, 1]

            fig3 = plt.figure(figsize=(5, 3))
            plt.scatter(df['tsne_1'], df['tsne_2'], c=df['cluster'], cmap='tab10', s=10, alpha=0.7)
            plt.title("t-SNE Method")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.colorbar(label="Cluster")
            plt.grid()
            st.pyplot(fig3)

    # 클러스터 0 통계 요약
    st.subheader("7. 클러스터 0 통계 요약")
    cluster_0 = df[df['cluster'] == 0]
    total_len = len(df)
    cluster_0_len = len(cluster_0)
    percentage = cluster_0_len / total_len * 100

    st.markdown(f"**▣ 클러스터 '0'은 전체 {total_len:,}개 중 {cluster_0_len:,}개를 차지합니다. ({percentage:.2f}%)**")

    numeric_cols = df.select_dtypes(include='number').columns.drop('cluster')
    summary = pd.DataFrame({
        'Mean': cluster_0[numeric_cols].mean(),
        'Mode': cluster_0[numeric_cols].mode().iloc[0],
        'Min': cluster_0[numeric_cols].min(),
        'Max': cluster_0[numeric_cols].max(),
        'Q1 (25%)': cluster_0[numeric_cols].quantile(0.25),
        'Q2 (Median)': cluster_0[numeric_cols].quantile(0.5),
        'Q3 (75%)': cluster_0[numeric_cols].quantile(0.75),
    })

    st.dataframe(summary.round(2))
