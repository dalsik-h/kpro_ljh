# menu2_predict.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model
from datetime import timedelta

# 1. 타이틀
st.header("📈 댐 수위 예측 (LSTM 모델)")

# 2. CSV 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요 (예: future_input.csv)", type=["csv"])

if uploaded_file is not None:
    # 3. 데이터 불러오기
    df_input = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    st.success("CSV 파일 업로드 완료!")
    st.write("입력 데이터 미리보기", df_input.head())

    # 4. 모델 및 스케일러 불러오기
    model = load_model("recursive_lstm_model.h5")
    with open("scaler_input.pkl", "rb") as f:
        scaler_input = pickle.load(f)
    with open("scaler_target.pkl", "rb") as f:
        scaler_target = pickle.load(f)

    # 5. lookback 설정
    lookback = 168

    # 6. 예측 함수 정의
    def recursive_forecast(model, input_df, scaler_input, scaler_target, lookback, n_steps):
        input_scaled = scaler_input.transform(input_df)
        X = input_scaled[-lookback:].reshape(1, lookback, -1)
        predictions = []

        for _ in range(n_steps):
            pred_scaled = model.predict(X, verbose=0)
            pred = scaler_target.inverse_transform(pred_scaled)[0, 0]
            predictions.append(pred)

            # 새로운 입력 구성
            new_input = np.concatenate([X[0, 1:], pred_scaled], axis=0).reshape(1, lookback, -1)
            X = new_input

        return predictions

    # 7. 예측 수행
    n_hours = st.slider("몇 시간 예측할까요?", 24, 168, 72)
    if st.button("예측 실행"):
        forecast = recursive_forecast(model, df_input, scaler_input, scaler_target, lookback, n_hours)

        # 8. 결과 시각화
        last_time = df_input.index[-1]
        future_index = [last_time + timedelta(hours=i+1) for i in range(n_hours)]

        forecast_df = pd.DataFrame({
            "datetime": future_index,
            "predicted_ycd_level": forecast
        }).set_index("datetime")

        st.success("예측 완료!")
        st.line_chart(forecast_df)

        # 9. 다운로드 기능 (선택)
        csv = forecast_df.to_csv().encode("utf-8-sig")
        st.download_button("예측 결과 CSV 다운로드", csv, "predicted_ycd_level.csv", "text/csv")

