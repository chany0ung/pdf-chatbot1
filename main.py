__import__('pysqlite3')
import sys
sys.modeules['sqlite3'] = sys.modules.pop('pyslite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
import tempfile
import os

#제목
st.title("ChatPDF")
st.write("---")

#파일업로드
uploaded_file = st.file_uploader("Choose a Pdf file", type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    #Text splitters
    from langchain_text_splitters import RecursiveCharacterTextSplitter 
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)


    #VECTOR DB
    embeddings_model = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings_model)

    #Question
    st.header("PDF에게 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')


    if st.button("질문하기"):
     
        #Retrieve
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        #LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.)

        #prompt 
        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        result = rag_chain.invoke(question)
        st.write(result)
