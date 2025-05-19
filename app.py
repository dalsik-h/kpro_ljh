import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score



plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_csv('./pip_dataset_pro.csv', index_col='date_time', parse_dates=True)

df.drop(['thj_vv_open_3'], axis=1, inplace=True)

# 데이터 스케일링
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# 군집 수 결정
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('k (Number of Clusters)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal k')
plt.grid()
plt.show()

# 엘보우 method로 시각화 결과 3또는 4

# 클러스터링 대상은 scaled_df (스케일링된 데이터)

def evaluate_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    davies = davies_bouldin_score(X, labels)
    
    print(f"=== K = {k} ===")
    print(f"Silhouette Score       : {silhouette:.4f}")
    print(f"Calinski-Harabasz Index: {calinski:.2f}")
    print(f"Davies-Bouldin Index   : {davies:.4f}")
    print()

# k = 3과 4 모두 평가
evaluate_kmeans(df_scaled, 3)
evaluate_kmeans(df_scaled, 4)

# k-means 실행
optimal_k = 4  # elbow에서 확인한 값(3 또는 4) 적용
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

# 결과를 원본에 추가
df['cluster'] = clusters

# 각 군집의 평균값 보기
cluster_summary = df.groupby('cluster').mean()
print(cluster_summary)

# 2차원 축소를 통한 시각화
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('KMeans Clustering Result (PCA 2D)')
plt.colorbar(label='Cluster')
plt.grid()
plt.show()

# t-sne 분석

# t-SNE 모델 생성 및 변환 (시간이 조금 걸릴 수 있음)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
tsne_result = tsne.fit_transform(df_scaled)  # scaled_df는 표준화된 데이터

# t-SNE 결과를 데이터프레임에 저장
df['tsne_1'] = tsne_result[:, 0]
df['tsne_2'] = tsne_result[:, 1]

# 시각화
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['tsne_1'], df['tsne_2'], c=df['cluster'], cmap='tab10', s=10, alpha=0.7)
plt.title("t-SNE Clustering Visualization")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.show()

# 클러스터별 데이터 통계치 분석

# 클러스터 0만 필터링
cluster_0 = df[df['cluster'] == 0]

# 전체 중 몇 개인지, 몇 퍼센트인지 계산
total_len = len(df)
cluster_0_len = len(cluster_0)
percentage = cluster_0_len / total_len * 100

# 메시지 출력
message = (
    f"▣ 클러스터 '0'은 전체 {total_len:,}개 중 {cluster_0_len:,}개를 차지합니다. ({percentage:.2f}%)<br>"
    f"  ☞ 클러스터 0 요약"
)
# 수치형 컬럼만 선택 (예: date_time 제외)
numeric_cols = df.select_dtypes(include='number').columns.drop('cluster')

# 평균
mean_values = cluster_0[numeric_cols].mean().to_frame(name='Mean')

# 최빈값 (mode는 여러 개일 수 있어 첫 번째만 사용)
mode_values = cluster_0[numeric_cols].mode().iloc[0].to_frame(name='Mode')

# 최소/최대
min_values = cluster_0[numeric_cols].min().to_frame(name='Min')
max_values = cluster_0[numeric_cols].max().to_frame(name='Max')

# 분위수
quantiles = cluster_0[numeric_cols].quantile([0.25, 0.5, 0.75])
quantiles.index = ['Q1 (25%)', 'Q2 (Median)', 'Q3 (75%)']
quantiles = quantiles.T

# 모든 통계 결합
summary = pd.concat([mean_values, mode_values, min_values, max_values, quantiles], axis=1)

# 출력
# print("📊 클러스터 0의 통계 요약:\n")
# print(summary.round(2))  # 보기 좋게 반올림
