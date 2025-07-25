import streamlit as st
import importlib.util

if "selected" not in st.session_state or st.session_state.selected == "page1.py":
    st.session_state.selected = None

st.set_page_config(layout="wide")

# 좌측 사이드바 버튼
st.sidebar.markdown("## ✅ 포항권 공업용수<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AI분석시스템", unsafe_allow_html=True)

if st.sidebar.button("📈 영천댐 수위 예측  &nbsp;&nbsp;&nbsp;→&nbsp;&nbsp;&nbsp;  안계소수력 발전전력 예측"):
    st.session_state.selected  = "app_lstm.py"

if st.sidebar.button("📊 공업용수 관망정보 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;클러스터링"):
    st.session_state.selected  = "app_gmm.py"

selected = st.session_state.selected

# ------------------- 오른쪽 메인 화면 ---------------------
if selected is None:
    # 포항권지사 타이틀 이미지 표시
    
    cols = st.columns([1, 10])
    with cols[0]:
        st.image("water-tap.png", width=48)
    with cols[1]:
        st.markdown(
            "<div style='padding-top: 10px; align-items: center; font-size: 40px; font-weight: bold;'>포항권 공업용수 AI 분석 시스템</div>",
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)  # 약간의 여백

    st.image("title.png", use_container_width=True)
else:
    # 동적으로 py 파일을 import 후 실행
    spec = importlib.util.spec_from_file_location("page_module", selected)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 반드시 run() 함수가 있어야 함
    if hasattr(module, "run"):
        module.run()
