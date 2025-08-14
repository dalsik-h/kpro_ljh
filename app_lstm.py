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

uploaded_future = st.file_uploader(" ➡️ 예측기간에 대한 독립변수 데이터 (파일명: future_input2.csv)", type=["csv"])
uploaded_history = st.file_uploader(" ➡️ 과거(4일) 전체(독립+종속)변수 데이터 (파일명: last_168_input.csv)", type=["csv"])

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
    st.subheader("📋 영천댐 수위 예측 결과 표")
    st.dataframe(forecast_df)  


    st.subheader("📈 영천댐 수위 예측 결과 그래프")

    plt.figure(figsize=(12, 4))
    plt.plot(forecast_df.index, forecast_df['predicted_ycd_level'], label='Predicted Level')
    plt.xlabel('Date/Time')
    plt.ylabel('Dam Level')
    plt.title('Predicted Dam Water Level')

    # 🔽 y축 자동 조정
    ymin = forecast_df['predicted_ycd_level'].min()
    ymax = forecast_df['predicted_ycd_level'].max()
    padding = (ymax - ymin) * 0.1
    plt.ylim(ymin - padding, ymax + padding)

    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

    # =============================
    # 안계소수력 발전전력 예측 부분
    # =============================
    st.header("🔋 안계소수력 발전전력 예측 (XGBoost 모델)")

    hpower_future = st.file_uploader(" ➡️ 예측기간에 대한 독립변수 데이터 (파일명: future_input3.csv)", type=["csv"])


    if hpower_future is not None:

        hpower_df = pd.read_csv(hpower_future, parse_dates=['date_time'])
        hpower_df.sort_values('date_time', inplace=True)

        # ========================
        # 모델 및 스케일러 불러오기
        # ========================
        model2 = joblib.load('final_xgb_model.pkl')
        scaler2 = joblib.load('step3_standard_scaler.pkl')

        # ========================
        # 1차로 예측된 댐수위 활용
        # ========================
        forecast_df = forecast_df.rename(columns={'predicted_ycd_level': 'ycd_level'})

        # ========================
        # 데이터 병합 및 정렬
        # ========================
        merged_df = pd.merge(hpower_df, forecast_df, on='date_time', how='inner')

        # 열 순서 맞추기 (모델 학습 시 사용한 순서)
        expected_features = ['agp_bp_vv', 'ycd_level', 'thj_vv', 'agp_inflow']
        model_input = merged_df[expected_features]

        # ========================
        # 스케일링 및 예측
        # ========================
        scaled2_input = scaler2.transform(model_input)
        predictions = model2.predict(scaled2_input)
        merged_df['predicted_agp_power'] = np.floor(predictions).astype(int)
        # ========================
        # 결과 출력
        # ========================
        st.success("예측 완료!")

        # 예측 결과 표로 먼저 출력
        st.subheader("📋 안계(소) 발전전력 예측 결과 표")
        st.dataframe(merged_df)

        # ========================
        # 시각화
        # ========================
        st.subheader("📈 안계(소) 발전전력 예측 결과 그래프")

        plt.figure(figsize=(12, 5))
        plt.plot(merged_df['date_time'], merged_df['predicted_agp_power'], marker='o', linestyle='-', label='predicted agp_power')
        plt.xlabel('Date / Time')
        plt.ylabel('Predicted agp_power')
        plt.title('XGBoost agp_power Prediction')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        st.pyplot(plt)


        # =========================================
        # 최종 요약 텍스트 생성 (댐 수위 & 발전전력)
        # =========================================

        def _dir_word(a, b):
            if b > a: return "상승"
            if b < a: return "하강"
            return "변화 없음"

        def _pick_points_from_index(idx_like):
            n = len(idx_like)
            if n == 0:
                return None
            if n == 1:
                i0 = i1 = i2 = 0
            elif n == 2:
                i0, i1, i2 = 0, 0, 1
            else:
                i0, i1, i2 = 0, n // 2, n - 1

            s_ts, m_ts, e_ts = idx_like[i0], idx_like[i1], idx_like[i2]
            s_str = pd.to_datetime(s_ts).strftime("%Y-%m-%d %H:%M")
            m_str = pd.to_datetime(m_ts).strftime("%Y-%m-%d %H:%M")
            e_str = pd.to_datetime(e_ts).strftime("%Y-%m-%d %H:%M")
            return (s_str, m_str, e_str, s_ts, m_ts, e_ts)

        # 박스 렌더 함수
        def _summary_box(level_text_html, power_text_html=None, total_text_html=None):
            box_open = """
            <div style="
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 16px 18px;
                margin-top: 12px;
                background: #f9fff9;
            ">
            """
            parts = [box_open, level_text_html]
            if power_text_html:
                parts.append(power_text_html)
            if total_text_html:
                parts.append(total_text_html)
            parts.append("</div>")
            st.markdown("\n".join(parts), unsafe_allow_html=True)

        # 2) 댐 수위 요약 (rename 전/후 모두 대응)
        st.subheader("📝 최종 요약")
        if not forecast_df.empty:
            level_col = 'predicted_ycd_level' if 'predicted_ycd_level' in forecast_df.columns else (
                'ycd_level' if 'ycd_level' in forecast_df.columns else None
            )

            if level_col is None:
                st.info("예측 수위 컬럼을 찾을 수 없습니다. (predicted_ycd_level 또는 ycd_level)")
            else:
                pick = _pick_points_from_index(forecast_df.index)
                if pick is not None:
                    s_str, m_str, e_str, s_ts, m_ts, e_ts = pick

                    # 값
                    s_level = float(forecast_df.loc[s_ts, level_col])
                    m_level = float(forecast_df.loc[m_ts, level_col])
                    e_level = float(forecast_df.loc[e_ts, level_col])

                    # 방향
                    dir_level_1 = _dir_word(s_level, m_level)
                    dir_level_2 = _dir_word(m_level, e_level)

                    # 수위 요약 HTML
                    level_text_html = f"""
<div style="font-size:18px; font-weight:600; line-height:1.5; margin-bottom:8px;">
    <span style="font-weight:800;">댐 수위</span>는 <span style="font-weight:800;">{s_str}</span> 기준
    <span style="font-weight:800;">{s_level:.2f}</span>으로 시작해
    <span style="font-weight:800;">중간시점({m_str})</span>에는
    <span style="font-weight:800;">{m_level:.2f}</span>까지
    <span style="font-weight:800;">{dir_level_1}</span>할 예정이며,
    <span style="font-weight:800;">종료시점({e_str})</span>에는
    <span style="font-weight:800;">{e_level:.2f}</span>로
    <span style="font-weight:800;">{dir_level_2}</span>할 예정입니다.
</div>
                    """

                    # 3) 발전전력 요약/총합 (이번 블록은 hpower_future 업로드된 경우에만 실행되는 위치라 merged_df 존재)
                    power_text_html = None
                    total_text_html = None

                    if ("predicted_agp_power" in merged_df.columns) and (len(merged_df) > 0):
                        pick2 = _pick_points_from_index(merged_df["date_time"].values)
                        if pick2 is not None:
                            s2_str, m2_str, e2_str, s2_ts, m2_ts, e2_ts = pick2

                            s_pow = int(merged_df.loc[merged_df["date_time"] == s2_ts, "predicted_agp_power"].iloc[0])
                            m_pow = int(merged_df.loc[merged_df["date_time"] == m2_ts, "predicted_agp_power"].iloc[0])
                            e_pow = int(merged_df.loc[merged_df["date_time"] == e2_ts, "predicted_agp_power"].iloc[0])

                            dir_pow_1 = _dir_word(s_pow, m_pow)
                            dir_pow_2 = _dir_word(m_pow, e_pow)

                            power_text_html = f"""
<div style="font-size:18px; font-weight:600; line-height:1.5; margin-bottom:8px;">
    <span style="font-weight:800;">발전전력</span>은 <span style="font-weight:800;">{s2_str}</span> 기준
    <span style="font-weight:800;">{s_pow}</span>로 시작해
    <span style="font-weight:800;">중간시점({m2_str})</span>에는
    <span style="font-weight:800;">{m_pow}</span>까지
    <span style="font-weight:800;">{dir_pow_1}</span>하며,
    <span style="font-weight:800;">종료시점({e2_str})</span>에는
    <span style="font-weight:800;">{e_pow}</span>로
    <span style="font-weight:800;">{dir_pow_2}</span>할 것으로 예측됩니다.
</div>
                            """

                            total_power = int(merged_df['predicted_agp_power'].sum())
                            total_text_html = f"""
<div style="font-size:18px; font-weight:700; margin-top:4px;">
    총 발전생산전력 (3일): <span style="font-weight:900;">{total_power} kW</span>
</div>
                            """

                    # 🔳 한 박스에 출력 (제목은 box 내부에 넣지 않음)
                    _summary_box(level_text_html, power_text_html, total_text_html)