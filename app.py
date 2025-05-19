import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.font_manager as fm
import matplotlib.patches as patches
from sklearn.metrics import pairwise_distances_argmin_min

# 페이지 설정
st.set_page_config(page_title="KMeans Cluster 분석", layout="wide")
st.title("KMeans 기반 시계열 군집 분석")

# 한글 폰트 설정
# font_path = './NanumGothic-Regular.ttf'  # 파일을 프로젝트 폴더에 포함시켜야 함
# font_prop = fm.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로딩 및 전처리
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

# Elbow Method
st.subheader("2. 최적 군집 수 찾기 (Elbow Method)")
if st.button("Elbow Method 실행"):
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)

    fig1 = plt.figure(figsize=(5, 3))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.xlabel("k (Number of Clusters)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.grid()
    st.pyplot(fig1)

# 클러스터 품질 평가
@st.cache_data
def evaluate_k(X, k):
    model = KMeans(n_clusters=k, random_state=42)
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

    st.subheader("6. 클러스터 시각화")
    tab1, tab2 = st.tabs(["PCA", "t-SNE"])

    with tab1:
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(df_scaled)
        fig2 = plt.figure(figsize=(5, 3))
        plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.title("PCA Method")
        plt.colorbar(label="Cluster")
        plt.grid()
        st.pyplot(fig2)

    with tab2:
        if st.button("t-SNE 시각화 실행"):
            st.info("⚠️ 약간의 시간이 걸릴 수 있어요. 잠시만 기다려 주세요.")

            df_sample = df_scaled.copy()
            if len(df_scaled) > 1000:
                df_sample = df_scaled.sample(n=1000, random_state=42)
                cluster_sample = df['cluster'].iloc[df_sample.index]
            else:
                cluster_sample = df['cluster']

            with st.spinner("t-SNE 계산 중입니다... 시간이 걸릴 수 있어요."):
                tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
                tsne_result = tsne.fit_transform(df_sample)

            fig3 = plt.figure(figsize=(5, 3))
            plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_sample, cmap='tab10', s=10, alpha=0.7)
            plt.title("t-SNE 기반 클러스터 시각화")
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
        'Q1': cluster_0[numeric_cols].quantile(0.25),
        'Q2': cluster_0[numeric_cols].quantile(0.5),
        'Q3': cluster_0[numeric_cols].quantile(0.75),
    })

    st.dataframe(summary.round(2))



    st.subheader("8. 클러스터 대표 샘플 조회")

    if 'cluster' in df.columns:
        cluster_labels = sorted(df['cluster'].unique())
        cluster_choice = st.slider("대표 샘플을 볼 클러스터 번호를 선택하세요", min_value=min(cluster_labels), max_value=max(cluster_labels), step=1)

        # 클러스터 중심과 가장 가까운 샘플 구하기
        centers = kmeans.cluster_centers_
        indices, distances = pairwise_distances_argmin_min(centers, df_scaled)

        rep_index = indices[cluster_choice]  # 선택된 클러스터의 대표 인덱스
        rep_row = df.iloc[rep_index]

        st.markdown(f"**클러스터 {cluster_choice}의 대표 시간대: {rep_row.name}**")
        st.dataframe(rep_row.to_frame(name='Value'))
    else:
        st.warning("먼저 클러스터링을 수행해주세요.")




    st.subheader("9. 관망도 상 대표 샘플 표시")
    image = Image.open("./back_img2.jpg")  # 업로드한 배경 이미지

    # ngt_flow_5 컬럼 요약
    col = "ngt_flow_5"
    col_title = "남계터널 출구 유량"
    stats = summary.loc[col].round(2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)
    ax.axis('off')

    # 오버레이 텍스트 출력 (위치: 유량(1) 기준 수동 지정)
    base_x, base_y = 410, 450
    line_height = 40

    max_width = 160   # 박스 고정 너비
    box_height = 40   # 한 줄 높이

        # 0. 제목 줄
    ax.add_patch(patches.Rectangle(
        (base_x, base_y - 15),
        max_width,
        box_height,
        linewidth=1,
        edgecolor='blue',
        facecolor='lightgray',
        alpha=0.9
    ))
    ax.text(
        base_x + 5, base_y,
        col_title,  # 컬럼명
        fontsize=7,
        weight='bold',
        color='black',
        verticalalignment='top'
    )

    # 1. 통계값 줄들
    for i, (label, value) in enumerate(stats.items()):
        y = base_y + (i + 1) * line_height
        text = f"{label}: {value}"

        ax.add_patch(patches.Rectangle(
            (base_x, y - 15),
            max_width,
            box_height,
            linewidth=1,
            edgecolor='black',
            facecolor='white',
            alpha=0.9
        ))
        ax.text(
            base_x + 5, y, text,
            fontsize=5,
            color='black',
            verticalalignment='top'
        )

    st.pyplot(fig)
