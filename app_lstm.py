import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
# from keras.models import load_model
from tensorflow.keras.models import load_model
from scipy.signal import savgol_filter
from datetime import timedelta

# =============================
# Streamlit UI
# =============================
st.header("📈 댐 수위 예측 (LSTM+GRU 모델)")

uploaded_future = st.file_uploader("미래 입력 파일 (예: future_input2.csv)", type=["csv"])
uploaded_history = st.file_uploader("과거 입력 파일 (예: last_168_input.csv)", type=["csv"])

if uploaded_future and uploaded_history:
    # =============================
    # 데이터 불러오기
    # =============================
    raw_future = pd.read_csv(uploaded_future, parse_dates=['date_time'], index_col='date_time')
    history_df = pd.read_csv(uploaded_history, parse_dates=['date_time'], index_col='date_time')
    st.success("CSV 파일 업로드 완료!")

    raw_future.index = pd.to_datetime(raw_future.index)
    history_df.index = pd.to_datetime(history_df.index)
    raw_future.sort_index(inplace=True)
    history_df.sort_index(inplace=True)

    # =============================
    # 모델 및 스케일러 불러오기
    # =============================
    # model = load_model("recursive_lstm_model.keras", compile=False)
    model = load_model("recursive_lstm_model.h5")
    model.compile(optimizer='adam', loss='mae')
    scaler_input = joblib.load('scaler_input.pkl')
    scaler_target = joblib.load('scaler_target.pkl')

    # =============================
    # 전처리 함수
    # =============================
    def prepare_initial_input(future_df, is_future=True):
        df = future_df.copy()
        if is_future:
            df['ycd_level'] = np.nan
        df['month'] = df.index.month
        df['season'] = df['month'] % 12 // 3 + 1
        season_onehot = pd.get_dummies(df['season'], prefix='season').astype(int)
        df = pd.concat([df, season_onehot], axis=1)
        for s in ['season_1', 'season_2', 'season_3', 'season_4']:
            if s not in df.columns:
                df[s] = 0
        df.drop(['season', 'month'], axis=1, inplace=True)
        df['ycd_gflow_out1'] = np.log1p(df['ycd_gflow_out1'])
        df['over_flow_out5'] = np.log1p(df['over_flow_out5'])
        for lag in [1, 2, 3, 4, 5, 6, 24]:
            df[f'ycd_level_lag{lag}'] = np.nan
        df['flow_out1_rolling_12h'] = np.nan
        df['flow_out1_rolling_12h_diff'] = np.nan
        df['rain_count_12h'] = np.nan
        return df

    # =============================
    # 재귀 예측 함수
    # =============================
    def recursive_forecast(future_df, history_df, model, scaler_input, scaler_target, lookback=96):
        df = prepare_initial_input(future_df, is_future=True)
        df = df[scaler_input.feature_names_in_]
        hdf = prepare_initial_input(history_df, is_future=False)
        for t in range(24, len(hdf)):
            current_time = hdf.index[t]

            for lag in [1, 2, 3, 4, 5, 6, 24]:
                lag_time = current_time - pd.Timedelta(hours=lag)
                if lag_time in hdf.index:
                    val = hdf.at[lag_time, 'ycd_level']
                    hdf.at[current_time, f'ycd_level_lag{lag}'] = val

            past_flow = hdf.loc[current_time - pd.Timedelta(hours=11):current_time, 'ycd_gflow_out1']
            past_flow2 = hdf.loc[current_time - pd.Timedelta(hours=12):current_time - pd.Timedelta(hours=1), 'ycd_gflow_out1']
            if len(past_flow) == 12:
                mean1 = past_flow.mean()
                mean2 = past_flow2.mean()
                hdf.at[current_time, 'flow_out1_rolling_12h'] = mean1
                hdf.at[current_time, 'flow_out1_rolling_12h_diff'] = mean1 - mean2

            past_rain = hdf.loc[current_time - pd.Timedelta(hours=11):current_time, 'rain_total']
            if len(past_rain) == 12:
                hdf.at[current_time, 'rain_count_12h'] = (past_rain > 0).sum()

        full_data = pd.concat([hdf, df], axis=0)
        preds = []
        forecast_index = []

        bias = 0  # 초기화
        initial_slope = 0
        initial_level = history_df['ycd_level'].iloc[-1]  # 마지막 실제값

        for t in range(len(df)):
            current_time = df.index[t]

            for lag in [1,2,3,4,5,6,24]:
                lag_time = current_time - pd.Timedelta(hours=lag)
                if lag_time in full_data.index:
                    val = full_data.at[lag_time, 'ycd_level']
                    df.at[current_time, f'ycd_level_lag{lag}'] = val
                    full_data.at[current_time, f'ycd_level_lag{lag}'] = val
            past_flow = full_data.loc[current_time - pd.Timedelta(hours=11):current_time, 'ycd_gflow_out1']
            past_flow2 = full_data.loc[current_time - pd.Timedelta(hours=12):current_time - pd.Timedelta(hours=1), 'ycd_gflow_out1']
            if len(past_flow) == 12:
                df.at[current_time, 'flow_out1_rolling_12h'] = past_flow.mean()
                df.at[current_time, 'flow_out1_rolling_12h_diff'] = past_flow.mean() - past_flow2.mean()
                full_data.at[current_time, 'flow_out1_rolling_12h'] = past_flow.mean()
                full_data.at[current_time, 'flow_out1_rolling_12h_diff'] = past_flow.mean() - past_flow2.mean()
            past_rain = full_data.loc[current_time - pd.Timedelta(hours=11):current_time, 'rain_total']
            if len(past_rain) == 12:
                df.at[current_time, 'rain_count_12h'] = (past_rain > 0).sum()
                full_data.at[current_time, 'rain_count_12h'] = (past_rain > 0).sum()
            input_row = df.loc[[current_time]].copy()
            input_row_scaled = pd.DataFrame(
                scaler_input.transform(input_row[scaler_input.feature_names_in_]),
                columns=scaler_input.feature_names_in_,
                index=input_row.index
            )
            seq_start_time = current_time - pd.Timedelta(hours=lookback)
            seq_end_time = current_time - pd.Timedelta(hours=1)
            sequence = full_data.loc[seq_start_time:seq_end_time, scaler_input.feature_names_in_]
            if len(sequence) < lookback:
                continue
            sequence_scaled = scaler_input.transform(sequence)
            sequence_scaled = sequence_scaled.reshape(1, lookback, -1)
            pred_scaled = model.predict(sequence_scaled, verbose=0)
            pred = scaler_target.inverse_transform(pred_scaled)[0][0]

            ##########################################################################################
            if t == 0:
                bias = initial_level - pred
                pred += bias
                initial_slope = 0
            else:
                pred += bias
                slope = pred - preds[-1]
                ratio = min(t / len(df), 1)
                if slope > 0:
                    # pred = preds[-1] + slope
                    if t < int(len(df) * 0.4):  # 예: 40% 시점 이전엔 완만화 없음
                        pred = preds[-1] + slope  # slope 그대로 반영
                    else:
                        weaken_ratio = np.exp(-7 * (ratio - 0.4) / 0.6)
                        pred = preds[-1] + slope * weaken_ratio
                    initial_slope = slope
                else:
                    # ✅ 하강이라면 완만하게 조정 (후반 갈수록 완만화)
                    # weaken_ratio = max(0.5, 1 - math.log1p(ratio * 9) / math.log1p(10))
                    weaken_ratio = max(0.3, np.exp(-2 * ratio))
                    pred = preds[-1] + slope * weaken_ratio
                    initial_slope = slope * weaken_ratio
            ##########################################################################################

            preds.append(pred)
            forecast_index.append(current_time)
            df.at[current_time, 'ycd_level'] = pred
            full_data.at[current_time, 'ycd_level'] = pred
        return forecast_index, preds

    # =============================
    # 예측 실행
    # =============================
    forecast_index, forecast_preds = recursive_forecast(raw_future, history_df, model, scaler_input, scaler_target)

    # 결과 정리 및 시각화
    forecast_df = pd.DataFrame({'date_time': forecast_index, 'predicted_ycd_level': forecast_preds})
    forecast_df['predicted_ycd_level'] = forecast_df['predicted_ycd_level'].round(2)
    forecast_df.set_index('date_time', inplace=True)

    st.success("예측 완료!")

        # 예측 결과 표로 먼저 출력
    st.subheader("📋 예측 결과 표")
    st.dataframe(forecast_df)  # 또는 st.table() 사용 가능


    st.subheader("📈 예측 결과 그래프")

    plt.figure(figsize=(12, 4))
    plt.plot(forecast_df.index, forecast_df['predicted_ycd_level'], label='예측 수위')
    plt.xlabel('시간')
    plt.ylabel('수위')
    plt.title('예측된 댐 수위')

    # 🔽 y축 자동 조정
    ymin = forecast_df['predicted_ycd_level'].min()
    ymax = forecast_df['predicted_ycd_level'].max()
    padding = (ymax - ymin) * 0.1
    plt.ylim(ymin - padding, ymax + padding)

    plt.grid(True)
    plt.legend()
    st.pyplot(plt)