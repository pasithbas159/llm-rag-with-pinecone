import openai
import os
import langchain
from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.llms import OpenAI
# from config import *

idx_name = "llm-demo-usecases"

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', "value does not exist")
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', "value does not exist")
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', "value does not exist")

pc = Pinecone(api_key=PINECONE_API_KEY)

## Lets Read the document
def read_doc_from_csv(directory):
    file_loader = CSVLoader(directory, encoding='utf-8')
    documents = file_loader.load()
    return documents

## Divide the docs into chunks
### https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.RecursiveCharacterTextSplitter.html#
def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return doc

def create_pinecone_index(index_name, dimension): 

    # pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(index_name)

    return "Created pinecone index!!!"

def insert_overwrite_pinecone(documents, embeddings, index_name): 

    # get pinecone index
    index = pc.Index(index_name)

    # delete records
    index.delete(delete_all=True)

    # insert
    docsearch = PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)

    print("Insert overwrite done!!!")

    return docsearch

def pinecone_query(index_name, embeddings, query, output_num): 

    # get pinecone index
    index = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # query based-on index
    res = index.similarity_search_with_score(
        query=query, 
        k=output_num
    )

    return res

def extract_product(text_1, text_2, text_3, text_4, text_5, message):

        llm = ChatGoogleGenerativeAI(model="gemini-pro")

        prompt = ChatPromptTemplate.from_template(
                """
                คุณคือล่ามที่มีหน้าที่แปลภาษาใต้ให้กับลูกค้าที่เข้ามาสอบถามเป็นภาษากลาง และยังสามารถแปลภาษากลางเป็นภาษาใต้ได้อีกด้วย ส่งมาเฉพาะคำตอบเท่านั้น

                โดยข้อมูลภาษาไทยกลาง-ไทยใต้ที่เกี่ยวข้องของคุณมี

                เอกสารที่ 1: {doc_1}
                เอกสารที่ 2: {doc_2}
                เอกสารที่ 3: {doc_3}
                เอกสารที่ 4: {doc_4}
                เอกสารที่ 5: {doc_5}

                คำถาม: {question}
                คำตอบ: 
            """
            )
        
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        reply_message = chain.invoke({"doc_1": text_1, 
                                    "doc_2": text_2, 
                                    "doc_3": text_3, 
                                    "doc_4": text_4,
                                    "doc_5": text_5,
                                    "question": message})
        
        return reply_message


def extract_product_without_rag(message):

        llm = ChatGoogleGenerativeAI(model="gemini-pro")

        prompt = ChatPromptTemplate.from_template(
                """
                คุณคือล่ามที่มีหน้าที่แปลภาษาใต้ให้กับลูกค้าที่เข้ามาสอบถามเป็นภาษากลาง และยังสามารถแปลภาษากลางเป็นภาษาใต้ได้อีกด้วย ส่งมาเฉพาะคำตอบเท่านั้น

                คำถาม: {question}
                คำตอบ: 
            """
            )
        
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        reply_message = chain.invoke({"question": message})
        
        return reply_message


if __name__ == "__main__":

    # doc = read_doc_from_csv('documents/thai-central-to-south.csv')
    # documents=chunk_data(docs=doc)

    ## Embedding Technique Of Google
    # embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

    ## Embedding Technique Of HuggingFace
    embeddings = HuggingFaceHubEmbeddings(
        huggingfacehub_api_token=HUGGINGFACE_API_KEY, model="intfloat/multilingual-e5-base"
    )

    ## (One-time execution)!!!
    ## Create Pinecone Index 
    # create_pinecone_index(idx_name, 768)
    ## Insert vector database
    ## https://thaiarc.tu.ac.th/folktales/southern/
    # docsearch = insert_overwrite_pinecone(documents, embeddings, index_name=idx_name)

    query = "หมาขึ้นมาขี้ที่วัดทุกเช้า เฝ้ากุฏิให้ดีนะ"
    vector = embeddings.embed_query(query)
    results = pinecone_query(index_name=idx_name, embeddings=embeddings, query=query, output_num=5)

    texts = []
    for res in results: 
         texts.append(res[0].page_content)

    # LLM with RAG
    reply_message = extract_product(text_1=texts[0], text_2=texts[1], text_3=texts[2], text_4=texts[3], text_5=texts[4], message=query)

    # # LLM without RAG!!!
    # reply_message = extract_product_without_rag(message=query)

    print(reply_message)