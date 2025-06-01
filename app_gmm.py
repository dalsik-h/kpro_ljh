# 전체 앱 코드에서 session_state 기반 상태 저장 및 유지 적용
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.font_manager as fm
import matplotlib.patches as patches
from sklearn.metrics import pairwise_distances_argmin_min
import os

# 1. 경로 지정
FONT_PATH = os.path.join(os.path.dirname(__file__), "NanumGothic.ttf")

# 2. fontprop 객체 생성
font_prop = fm.FontProperties(fname=FONT_PATH)

# 3. rcParams에 반영
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="GMM Cluster 분석", layout="wide")
st.title("GMM 기반 시계열 군집 분석")
plt.rcParams['axes.unicode_minus'] = False

@st.cache_data
def load_and_process():
    df = pd.read_csv('./pip_dataset_pro.csv', index_col='date_time', parse_dates=True)
    df.drop(['thj_vv_open_3'], axis=1, inplace=True)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df, pd.DataFrame(df_scaled, columns=df.columns)

st.subheader("1. 데이터 불러오기")
if "df" not in st.session_state:
    df, df_scaled = load_and_process()
    st.session_state.df = df
    st.session_state.df_scaled = df_scaled
else:
    df = st.session_state.df
    df_scaled = st.session_state.df_scaled
st.success("데이터 불러오기 성공")
st.write(df.head())

st.subheader("2. GMM 기반 최적 클러스터 수 평가 (BIC/AIC)")

if st.button("GMM 클러스터 수 평가 실행"):
    max_k = 10
    bic = []
    aic = []
    k_list = list(range(1, max_k + 1))

    with st.spinner("GMM 모델 학습 중..."):
        for k in k_list:
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
            gmm.fit(df_scaled)
            bic.append(gmm.bic(df_scaled))
            aic.append(gmm.aic(df_scaled))

    fig_gmm = plt.figure(figsize=(6, 4))
    plt.plot(k_list, bic, marker='o', label='BIC')
    plt.plot(k_list, aic, marker='s', label='AIC')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Score")
    plt.title("GMM 최적 클러스터 수 평가 (BIC & AIC)")
    plt.legend()
    plt.grid()
    st.pyplot(fig_gmm)

@st.cache_data
def evaluate_k_gmm(X, k):
    model = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    labels = model.fit_predict(X)  # 하드 할당
    return {
        "silhouette": silhouette_score(X, labels),
        "calinski": calinski_harabasz_score(X, labels),
        "davies": davies_bouldin_score(X, labels)
    }

st.subheader("3. 클러스터 품질 평가 (GMM, k=4, 5, 6)")
if st.button("GMM 품질 지표 보기"):
    for k in [4, 5, 6]:
        result = evaluate_k_gmm(df_scaled, k)
        st.markdown(f"**▶ K = {k}**")
        st.markdown(f"- Silhouette Score: {result['silhouette']:.4f}")
        st.markdown(f"- Calinski-Harabasz Index: {result['calinski']:.2f}")
        st.markdown(f"- Davies-Bouldin Index: {result['davies']:.4f}")

st.subheader("4. 클러스터링 실행")
if st.button("클러스터링 수행"):
    gmm = GaussianMixture(n_components=6, covariance_type='full', random_state=42)
    clusters = gmm.fit_predict(df_scaled)
    df['cluster'] = clusters
    
    st.session_state.df = df
    st.session_state.df_scaled = df_scaled
    st.session_state.gmm = gmm

    cluster_0 = df[df['cluster'] == 0]
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
    st.session_state.summary = summary
    st.success("클러스터링 완료!")

if 'df' in st.session_state and 'kmeans' in st.session_state:
    df = st.session_state.df
    df_scaled = st.session_state.df_scaled
    kmeans = st.session_state.kmeans
    summary = st.session_state.summary

    st.subheader("5. 클러스터별 최빈값")
    st.dataframe(df.groupby('cluster').agg(lambda x: x.mode().iloc[0]).round(2))

    st.subheader("6. 클러스터 시각화")
    tab1, tab2 = st.tabs(["PCA", "t-SNE"])

    with tab1:
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(df_scaled)
        fig2 = plt.figure(figsize=(5, 3))
        plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['cluster'], cmap='tab10', alpha=0.7)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.title("PCA Method")
        plt.colorbar(label="Cluster")
        plt.grid()
        st.pyplot(fig2)

    with tab2:
        if st.button("t-SNE 시각화 실행"):
            st.info("잠시만 기다려 주세요.")
            df_sample = df_scaled.copy()
            if len(df_scaled) > 1000:
                df_sample = df_scaled.sample(n=1000, random_state=42)
                cluster_sample = df['cluster'].iloc[df_sample.index]
            else:
                cluster_sample = df['cluster']

            with st.spinner("t-SNE 계산 중입니다..."):
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

    st.subheader("7. 클러스터 대표 샘플 조회")
    if "cluster_choice" not in st.session_state:
        st.session_state.cluster_choice = 0

    cluster_labels = sorted(df['cluster'].unique())
    st.session_state.cluster_choice = st.slider(
        "대표 샘플을 볼 클러스터 번호를 선택하세요",
        min_value=min(cluster_labels),
        max_value=max(cluster_labels),
        value=st.session_state.cluster_choice,
        step=1
    )

    if st.button("대표 샘플 보기"):
        st.subheader(f"**(1)클러스터 {st.session_state.cluster_choice}의 통계 요약**")
        cluster_chk = df[df['cluster'] == st.session_state.cluster_choice]
        total_len = len(df)
        cluster_chk_len = len(cluster_chk)
        percentage = cluster_chk_len / total_len * 100
        st.markdown(f"**▣ 클러스터 '{st.session_state.cluster_choice}'은 전체 {total_len:,}개 중 {cluster_chk_len:,}개를 차지합니다. ({percentage:.2f}%)**")
        st.dataframe(summary.round(2))

       
        centers = gmm.means_
        indices, distances = pairwise_distances_argmin_min(centers, df_scaled)

        rep_index = indices[st.session_state.cluster_choice]
        rep_row = df.iloc[rep_index]
        st.markdown(f"**(2)클러스터 {st.session_state.cluster_choice}의 대표 시간대: {rep_row.name}**")
        st.dataframe(rep_row.to_frame(name='Value'))

        st.subheader("**(3)관망도 상 대표 샘플 표시**")
        image = Image.open("./back_img2.jpg")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image)
        ax.axis('off')

        base_x, base_y = 350, 190
        line_height = 41
        max_width = 150
        box_height = 40

        # 접합정 출구부 유량
        col_a = "jhj_flow_1"
        col_a_title = "접합정 출구 유량"
        ax.add_patch(patches.Rectangle(
            (base_x, base_y - 15), max_width, box_height,
            linewidth=1, edgecolor='red', facecolor='lightpink', alpha=0.9
        ))
        ax.text(
            base_x + 5, base_y, col_a_title,
            fontsize=5, weight='bold', color='black', verticalalignment='top', fontproperties=font_prop
    
        )
        y = base_y + 1 * line_height
        text = f"{rep_row[col_a].round(2)}"
        ax.add_patch(patches.Rectangle(
            (base_x, y - 15), max_width, box_height,
            linewidth=1, edgecolor='black', facecolor='white', alpha=0.9
        ))
        ax.text(
            base_x + 5, y, text,
            fontsize=5, color='black', verticalalignment='top'
        )

        # 통합제수변실 유량
        col_b = "tjv_flow_2"
        col_b_title = "통합제수변실 유량"
        base_x2, base_y2 = 1070, 200
        ax.add_patch(patches.Rectangle(
            (base_x2, base_y2 - 15), max_width, box_height,
            linewidth=1, edgecolor='red', facecolor='lightpink', alpha=0.9
        ))
        ax.text(
            base_x2 + 5, base_y2, col_b_title,
            fontsize=5, weight='bold', color='black', verticalalignment='top', fontproperties=font_prop
    
        )
        y2 = base_y2 + 1 * line_height
        text = f"{rep_row[col_b]}"
        ax.add_patch(patches.Rectangle(
            (base_x2, y2 - 15), max_width, box_height,
            linewidth=1, edgecolor='black', facecolor='white', alpha=0.9
        ))
        ax.text(
            base_x2 + 5, y2, text,
            fontsize=5, color='black', verticalalignment='top'
        )

        # 포항시(분) 유량
        col_c = "phb_flow_3"
        col_c_title = "포항시(분) 유량"
        base_x3, base_y3 = 1400, 100
        ax.add_patch(patches.Rectangle(
            (base_x3, base_y3 - 15), max_width, box_height,
            linewidth=1, edgecolor='red', facecolor='lightpink', alpha=0.9
        ))
        ax.text(
            base_x3 + 5, base_y3, col_c_title,
            fontsize=5, weight='bold', color='black', verticalalignment='top', fontproperties=font_prop
    
        )
        y3 = base_y3 + 1 * line_height
        text = f"{rep_row[col_c]}"
        ax.add_patch(patches.Rectangle(
            (base_x3, y3 - 15), max_width, box_height,
            linewidth=1, edgecolor='black', facecolor='white', alpha=0.9
        ))
        ax.text(
            base_x3 + 5, y3, text,
            fontsize=5, color='black', verticalalignment='top'
        )

        # 학야(정) 유입유량
        col_d = "hyj_flow_4"
        col_d_title = "학야(정) 유입유량"
        base_x4, base_y4 = 1225, 550
        ax.add_patch(patches.Rectangle(
            (base_x4, base_y4 - 15), max_width, box_height,
            linewidth=1, edgecolor='red', facecolor='lightpink', alpha=0.9
        ))
        ax.text(
            base_x4 + 5, base_y4, col_d_title,
            fontsize=5, weight='bold', color='black', verticalalignment='top', fontproperties=font_prop
    
        )
        y4 = base_y4 + 1 * line_height
        text = f"{rep_row[col_d]}"
        ax.add_patch(patches.Rectangle(
            (base_x4, y4 - 15), max_width, box_height,
            linewidth=1, edgecolor='black', facecolor='white', alpha=0.9
        ))
        ax.text(
            base_x4 + 5, y4, text,
            fontsize=5, color='black', verticalalignment='top'
        )

        # 남계터널출구 유량
        col_e = "ngt_flow_5"
        col_e_title = "남계터널 출구유량"
        base_x5, base_y5 = 600, 340
        ax.add_patch(patches.Rectangle(
            (base_x5, base_y5 - 15), max_width, box_height,
            linewidth=1, edgecolor='red', facecolor='lightpink', alpha=0.9
        ))
        ax.text(
            base_x5 + 5, base_y5, col_e_title,
            fontsize=5, weight='bold', color='black', verticalalignment='top', fontproperties=font_prop
    
        )
        y5 = base_y5 + 1 * line_height
        text = f"{rep_row[col_e]}"
        ax.add_patch(patches.Rectangle(
            (base_x5, y5 - 15), max_width, box_height,
            linewidth=1, edgecolor='black', facecolor='white', alpha=0.9
        ))
        ax.text(
            base_x5 + 5, y5, text,
            fontsize=5, color='black', verticalalignment='top'
        )

        # 남계터널출구 신관 유량
        col_f = "ngt_sub_flow_6"
        col_f_title = "남계터널 신관유량"
        base_x6, base_y6 = 620, 510
        ax.add_patch(patches.Rectangle(
            (base_x6, base_y6 - 15), max_width, box_height,
            linewidth=1, edgecolor='red', facecolor='lightpink', alpha=0.9
        ))
        ax.text(
            base_x6 + 5, base_y6, col_f_title,
            fontsize=5, weight='bold', color='black', verticalalignment='top', fontproperties=font_prop
    
        )
        y6 = base_y6 + 1 * line_height
        text = f"{rep_row[col_f]}"
        ax.add_patch(patches.Rectangle(
            (base_x6, y6 - 15), max_width, box_height,
            linewidth=1, edgecolor='black', facecolor='white', alpha=0.9
        ))
        ax.text(
            base_x6 + 5, y6, text,
            fontsize=5, color='black', verticalalignment='top'
        )

        # 안계댐 유입 유량
        col_g = "agd_flow_7"
        col_g_title = "안계댐 유입유량"
        base_x7, base_y7 = 1580, 230
        ax.add_patch(patches.Rectangle(
            (base_x7, base_y7 - 15), max_width, box_height,
            linewidth=1, edgecolor='red', facecolor='lightpink', alpha=0.9
        ))
        ax.text(
            base_x7 + 5, base_y7, col_g_title,
            fontsize=5, weight='bold', color='black', verticalalignment='top', fontproperties=font_prop
    
        )
        y7 = base_y7 + 1 * line_height
        text = f"{rep_row[col_g]}"
        ax.add_patch(patches.Rectangle(
            (base_x7, y7 - 15), max_width, box_height,
            linewidth=1, edgecolor='black', facecolor='white', alpha=0.9
        ))
        ax.text(
            base_x7 + 5, y7, text,
            fontsize=5, color='black', verticalalignment='top'
        )

        # 통합제수변실 밸브 #1 개도
        col_h = "thj_vv_open_1"
        base_x8, base_y8 = 1010, 200
        text = f"{rep_row[col_h]}"
        ax.add_patch(patches.Rectangle(
            (base_x8, base_y8 - 15), 50, box_height,
            linewidth=1, edgecolor='black', facecolor='lightgray', alpha=0.9
        ))
        ax.text(
            base_x8 + 5, base_y8, text,
            fontsize=5, weight='bold', color='black', verticalalignment='top', fontproperties=font_prop
    
        )

        # 통합제수변실 밸브 #2 개도
        col_i = "thj_vv_open_2"
        base_x9, base_y9 = 970, 290
        text = f"{rep_row[col_i]}"
        ax.add_patch(patches.Rectangle(
            (base_x9, base_y9 - 15), 50, box_height,
            linewidth=1, edgecolor='black', facecolor='lightgray', alpha=0.9
        ))
        ax.text(
            base_x9 + 5, base_y9, text,
            fontsize=5, weight='bold', color='black', verticalalignment='top', fontproperties=font_prop
    
        )

        # 통합제수변실 #1 압력
        col_j = "thj_pre_1"
        base_x10, base_y10 = 1100, 150
        text = f"{rep_row[col_j]}kgf/cm²"
        ax.add_patch(patches.Rectangle(
            (base_x10, base_y10 - 15), 115, box_height,
            linewidth=1, edgecolor='blue', facecolor='skyblue', alpha=0.9
        ))
        ax.text(
            base_x10 + 5, base_y10, text,
            fontsize=5, weight='bold', color='blue', verticalalignment='top', fontproperties=font_prop
    
        )

        # 포항시 분기 압력
        col_k = "phb_pre_2"
        base_x11, base_y11 = 1400, 55
        text = f"{rep_row[col_k]}kgf/cm²"
        ax.add_patch(patches.Rectangle(
            (base_x11, base_y11 - 15), 115, box_height,
            linewidth=1, edgecolor='blue', facecolor='skyblue', alpha=0.9
        ))
        ax.text(
            base_x11 + 5, base_y11, text,
            fontsize=5, weight='bold', color='blue', verticalalignment='top', fontproperties=font_prop
    
        )

        # 학야(정) 유입 압력
        col_l = "hyj_pre_3"
        base_x12, base_y12 = 1100, 550
        text = f"{rep_row[col_l]}kgf/cm²"
        ax.add_patch(patches.Rectangle(
            (base_x12, base_y12 - 15), 115, box_height,
            linewidth=1, edgecolor='blue', facecolor='skyblue', alpha=0.9
        ))
        ax.text(
            base_x12 + 5, base_y12, text,
            fontsize=5, weight='bold', color='blue', verticalalignment='top', fontproperties=font_prop
    
        )

        # 통합제수변실 #2 압력
        col_m = "thj_pre_4"
        base_x13, base_y13 = 1040, 400
        text = f"{rep_row[col_m]}kgf/cm²"
        ax.add_patch(patches.Rectangle(
            (base_x13, base_y13 - 15), 115, box_height,
            linewidth=1, edgecolor='blue', facecolor='skyblue', alpha=0.9
        ))
        ax.text(
            base_x13 + 5, base_y13, text,
            fontsize=5, weight='bold', color='blue', verticalalignment='top', fontproperties=font_prop
    
        )

        # 안계댐 유입 압력
        col_n = "agd_pre_5"
        base_x14, base_y14 = 1580, 320
        text = f"{rep_row[col_n]}kgf/cm²"
        ax.add_patch(patches.Rectangle(
            (base_x14, base_y14 - 15), 115, box_height,
            linewidth=1, edgecolor='blue', facecolor='skyblue', alpha=0.9
        ))
        ax.text(
            base_x14 + 5, base_y14, text,
            fontsize=5, weight='bold', color='blue', verticalalignment='top', fontproperties=font_prop
    
        )

        # 남계터널출구 신관 압력
        col_o = "ngt_sub_pre_6"
        base_x15, base_y15 = 780, 540
        text = f"{rep_row[col_o]}kgf/cm²"
        ax.add_patch(patches.Rectangle(
            (base_x15, base_y15 - 15), 115, box_height,
            linewidth=1, edgecolor='blue', facecolor='skyblue', alpha=0.9
        ))
        ax.text(
            base_x15 + 5, base_y15, text,
            fontsize=5, weight='bold', color='blue', verticalalignment='top', fontproperties=font_prop
    
        )

        # 안계댐 유입 신관 압력
        col_p = "agd_new_pre_7"
        base_x16, base_y16 = 1750, 340
        text = f"{rep_row[col_p]}kgf/cm²"
        ax.add_patch(patches.Rectangle(
            (base_x16, base_y16 - 15), 115, box_height,
            linewidth=1, edgecolor='blue', facecolor='skyblue', alpha=0.9
        ))
        ax.text(
            base_x16 + 5, base_y16, text,
            fontsize=5, weight='bold', color='blue', verticalalignment='top', fontproperties=font_prop
    
        )

        # for i, (label, value) in enumerate(stats.items()):
        #     y = base_y + (i + 1) * line_height
        #     text = f"{label}: {value}"
        #     ax.add_patch(patches.Rectangle(
        #         (base_x, y - 15), max_width, box_height,
        #         linewidth=1, edgecolor='black', facecolor='white', alpha=0.9
        #     ))
        #     ax.text(
        #         base_x + 5, y, text,
        #         fontsize=5, color='black', verticalalignment='top'
        #     )

        st.pyplot(fig)
