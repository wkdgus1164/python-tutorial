import streamlit as st

from dotenv import load_dotenv

from chat_llm import get_ai_response

st.set_page_config(page_title="Chatbot", layout="wide", page_icon="🔍")
st.title("유레카 챗봇")
st.caption('GPT-4o 기반 RAG 가 포함된 챗봇입니다.')

load_dotenv()

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder='무엇이 궁금하세요?'):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner('답변을 생성하고 있어요.'):
        ai_response = get_ai_response(user_question)

        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})
