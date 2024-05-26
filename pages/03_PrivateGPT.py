from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OllamaEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
import streamlit as st

st.set_page_config(
    page_title="PrivateGPT",
    page_icon="📃",
)


@st.cache_resource
def init_llm(chat_callback: bool):
    if chat_callback == True:

        class ChatCallbackHandler(BaseCallbackHandler):
            def __init__(self, *args, **kwargs):
                self.message = ""

            def on_llm_start(self, *args, **kwargs):
                self.message_box = st.empty()
                # 아래 코드가 없으면 첫번째로 들어온 질문에 대한 답만 반복하게 됨.
                # @st.cache로 인해서 위에 있는 self.message = "" 가 다시 실행되지 않기 때문.
                # def __init__ 의 유무랑은 상관 없는듯.
                # on_llm_start 안에 있는 건 llm이 응답 받으면 실행되는 것이기 때문에 @st.cache가 있어도 다시 실행됨.
                self.message = ""

            def on_llm_end(self, *args, **kwargs):
                save_message(self.message, "ai")

            def on_llm_new_token(self, token, *args, **kwargs):
                self.message += token
                self.message_box.markdown(self.message)

        callbacks = [ChatCallbackHandler()]
    else:
        callbacks = []

    return ChatOllama(
        model="mistral:latest",
        temperature=0.1,
        streaming=True,
        callbacks=callbacks,
    )


llm = init_llm(chat_callback=True)

llm_for_memory = init_llm(chat_callback=False)


@st.cache_resource
def init_memory(_llm):
    return ConversationSummaryBufferMemory(
        llm=_llm, max_token_limit=60, return_messages=True, memory_key="history"
    )


memory = init_memory(llm_for_memory)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/private_embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OllamaEmbeddings(
        model="mistral:latest",
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


#########
# 메모리 #
#########
# input을 받아줘야하기 때문에 무시하기 위해서 _라는 argument 입력
def load_memory(_):
    return memory.load_memory_variables({})["history"]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")

        #########
        # 메모리 #
        #########
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "chat_history": load_memory,
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)

        memory.save_context(
            {"input": message},
            {"output": response.content},
        )

else:
    st.session_state["messages"] = []
