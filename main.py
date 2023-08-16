# Import Statements.
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write('User: ' + message.content)
        else:
            st.write('PDF: ' + message.content)

def main():
    # Use a breakpoint in the code line below to debug your script.
    load_dotenv()
    st.set_page_config(page_title='Chat with multiple PDFs',
                       page_icon=':books:')

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    st.header('Chat with multiple PDFs :books:')
    user_question = st.text_input('Ask a question about your documents:')
    if user_question:
        handle_userinput(user_question)
    with st.sidebar:
        st.subheader('Your documents')
        pdf_docs = st.file_uploader('Upload your PDFs here and click "Process"', accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing'):
                # Get PDF Text
                raw_text = get_pdf_text(pdf_docs)

                # Put Text into Chunks
                text_chunks = get_text_chunks(raw_text)

                # Create Vector Store
                vectorstore = get_vectorstore(text_chunks)

                # Create Conversation Chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
