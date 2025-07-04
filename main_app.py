import streamlit as st
import importlib.util

st.set_page_config(layout="wide")
st.title("🔘 버튼 클릭에 따라 우측 화면이 바뀌는 앱")

# 좌측 사이드바 버튼
st.sidebar.markdown("## ✅ 포항권 공업용수<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AI분석시스템", unsafe_allow_html=True)
show_page = None

if st.sidebar.button("📈 영천댐 수위 예측  &nbsp;&nbsp;&nbsp;→&nbsp;&nbsp;&nbsp;  안계소수력 발전전력 예측"):
    show_page = "page1.py"

if st.sidebar.button("📊 공업용수 관망정보 &nbsp;&nbsp;&nbsp; 클러스터링"):
    show_page = "app_gmm.py"

# 화면 분할: 좌측 - 제어 / 우측 - 페이지 출력
col1, col2 = st.columns([1, 2])

with col1:
    st.info("⬅ 왼쪽 메뉴에서 버튼을 클릭하세요.")

with col2:
    if show_page:
        file_path = show_page
        module_name = file_path.replace(".py", "")

        # 모듈 동적 import
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 반드시 run() 함수가 있어야 함
        if hasattr(module, "run"):
            module.run()
        else:
            st.error(f"`{file_path}`에는 `run()` 함수가 필요합니다.")
    else:
        st.write("오른쪽 영역에 페이지 내용이 여기에 표시됩니다.")
