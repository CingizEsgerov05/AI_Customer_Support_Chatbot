# app.py - Professional Streamlit Interface
import streamlit as st
import time
from backend import load_system
from datetime import datetime

# Səhifə konfiqurasiyası
st.set_page_config(
    page_title="AI Müştəri Xidməti",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-stats {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Başlıq
st.markdown('<div class="main-header">🤖 AI Müştəri Xidməti Köməkçisi</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">BERT əsaslı ağıllı söhbət sistemi</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Customer Service AI Chatbot")
    
    try:
        import pickle
        with open('chatbot_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
    except:
        st.error("⚠️ Model yüklənməyib")
        st.info("Zəhmət olmasa `train.py`-ni işə salın")
    
    st.divider()
    
    st.header("💡 Tövsiyələr")
    st.markdown("""
    **Sual nümunələri:**
    - Qiymətlər haqqında
    - Çatdırılma şərtləri
    - Ödəniş üsulları
    - Məhsul kataloqu
    - Qaytarma qaydaları
    """)
    
    st.divider()
    
    if st.button("🗑️ Söhbəti təmizlə"):
        st.session_state.messages = []
        st.session_state.message_count = 0
        st.rerun()

# Model yüklə (cache istifadə et)
@st.cache_resource
def get_chatbot():
    try:
        bot = load_system()
        return bot
    except FileNotFoundError as e:
        st.error(f"❌ Xəta: {e}")
        st.info("**Addımlar:**\n1. `pip install -r requirements.txt`\n2. `python train.py`\n3. `streamlit run app.py`")
        return None
    except Exception as e:
        st.error(f"❌ Gözlənilməz xəta: {e}")
        return None

chatbot = get_chatbot()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.message_count = 0
    # Xoş gəldin mesajı
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Salam! 👋 Mən sizin AI köməkçinizəm. Məhsullar, qiymətlər, çatdırılma və s. haqqında suallarınızı cavablandıra bilərəm. Necə kömək edə bilərəm?"
    })

# Chat tarixçəsi
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Sualınızı bura yazın... (məs: 'qiymətlər haqqında')"):
    if not chatbot:
        st.error("⚠️ Model yüklənməyib. Zəhmət olmasa səhifəni yeniləyin və ya modeli təlim verin.")
    else:
        # İstifadəçi mesajı
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.message_count += 1
        
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)
        
        # Bot cavabı
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            
            # Loading animation
            with st.spinner("Düşünürəm..."):
                response_text = chatbot.get_response(prompt)
            
            # Typing effect
            full_response = ""
            words = response_text.split()
            
            for i, word in enumerate(words):
                full_response += word + " "
                if i % 3 == 0:  # Hər 3 sözdən bir yenilə
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.03)
            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})



# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "Powered by BERT + PyTorch"
    "</div>",
    unsafe_allow_html=True
)