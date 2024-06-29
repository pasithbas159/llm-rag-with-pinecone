import streamlit as st
from pinecone_rag import *

st.title('📧 LangTai: Central-Thai to Southern-Thai Assistant App')

# PINECONE_API_KEY = st.sidebar.text_input('Pinecone API Key')
# GOOGLE_API_KEY = st.sidebar.text_input('Google API Key')
# HUGGINGFACE_API_KEY = st.sidebar.text_input('HuggingFaceHub API Key')

os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]
os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]

embeddings = HuggingFaceHubEmbeddings(
            huggingfacehub_api_token=HUGGINGFACE_API_KEY, model="intfloat/multilingual-e5-base"
        )

with st.form('my_form'):

    text = st.text_area('กรุณาใส่ประโยคที่ต้องการแปล (ภาษาไทยกลาง-ภาษาไทยใต้ หรือ ภาษาไทยใต้-ภาษาไทยกลาง)', 'หมาขึ้นมาขี้ที่วัดทุกเช้า เฝ้ากุฏิให้ดีนะ')

    submitted = st.form_submit_button('Submit')

    if submitted:
        vector = embeddings.embed_query(text)
        results = pinecone_query(index_name=idx_name, embeddings=embeddings, query=text, output_num=5)

        texts = []
        for res in results: 
                texts.append(res[0].page_content)

        # LLM with RAG
        reply_message = extract_product(text_1=texts[0], text_2=texts[1], text_3=texts[2], text_4=texts[3], text_5=texts[4], message=text)

        st.info(reply_message)