import streamlit as st
import importlib.util

st.set_page_config(layout="wide")

# 좌측 사이드바 버튼
st.sidebar.markdown("## ✅ 포항권 공업용수<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AI분석시스템", unsafe_allow_html=True)
show_page = None

if st.sidebar.button("📈 영천댐 수위 예측  &nbsp;&nbsp;&nbsp;→&nbsp;&nbsp;&nbsp;  안계소수력 발전전력 예측"):
    show_page = "page1.py"

if st.sidebar.button("📊 공업용수 관망정보 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;클러스터링"):
    show_page = "app_gmm.py"

# ------------------- 오른쪽 메인 화면 ---------------------
if selected is None:
    # 예시 이미지 표시 (센터 정렬 + 사이즈 조절 가능)
    st.markdown("<br><br>", unsafe_allow_html=True)  # 약간의 상단 여백

    st.image("title.png", use_column_width=False, width=400, caption="")  # 이미지 사이즈/경로 조절

    # 또는 중앙 정렬 CSS 추가
    st.markdown("""
    <div style='text-align: center;'>
        <img src='your_image_path.png' width='400'/>
    </div>
    """, unsafe_allow_html=True)
else:
    # 동적으로 py 파일을 import 후 실행
    spec = importlib.util.spec_from_file_location("page_module", selected)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 반드시 run() 함수가 있어야 함
    if hasattr(module, "run"):
        module.run()
    else:
        st.error("선택된 파일에 run() 함수가 없습니다.")
