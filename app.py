import streamlit as st
from src.agent import answer
from src.config import USE_LLM, OLLAMA_MODEL

st.set_page_config(page_title="Emporio da Musica", page_icon="🎵")
st.title("🎵 Empório da Música")
st.caption("Assistente virtual — protótipo de atendimento")
st.caption(f"Modo: {'LLM local (' + OLLAMA_MODEL + ')' if USE_LLM else 'baseline determinístico'}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Como posso ajudar?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Consultando o assistente..."):
            response = answer(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
