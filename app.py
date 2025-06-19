import streamlit as st
from langchain_community.llms import Ollama
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from htmlTemplate import css,bot_template,user_template




def get_pdf_text(pdf_docs):
  text = ""
  for pdf in pdf_docs:
    pdf_reader =PdfReader(pdf)
    for page in pdf_reader.pages:
      text += page.extract_text()
  return text
  
def get_text_chunks(text,chunk_size=800,chunk_overlap=50):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
  chunks = text_splitter.split_text(text)
  return chunks

def get_vectorstore(text_chunks):
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
  return vectorstore

def get_conversation_chain(vector_store):
  llm = Ollama(model="gemma3:1b", temperature=0.1)
  memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
  conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory)
  return conversation_chain

def handle_userinput(user_question):
  response = st.session_state.conversation({'question':user_question})
  st.session_state.chat_history = response['chat_history']

  for i, message in enumerate(st.session_state.chat_history):
    if i % 2 ==0:
      st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
      st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



def main():
  st.set_page_config(page_title="Chat with multiple pdfs",page_icon=":books:")
  st.write(css,unsafe_allow_html=True)

  if "conversation" not in st.session_state:
    st.session_state.conversation = None
  if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

  st.header("Chat with multiple pdfs :books:")
  user_question = st.text_input("Ask a question about your documents")
  if user_question:
    handle_userinput(user_question)

  with st.sidebar:
    st.subheader("Your documnets")
    pdf_docs = st.file_uploader("Upload your pdfs here and click on 'process'",accept_multiple_files=True)
    if st.button("Process"):
      with st.spinner("Processing"):
        #get pdf text
        raw_text = get_pdf_text(pdf_docs)
        #st.write(raw_text)

        #get the text chunks
        text_chunks = get_text_chunks(raw_text)
        #st.write(text_chunks)

        # create vector store
        vector_store = get_vectorstore(text_chunks)

        #create conversation chain
        st.session_state.conversation = get_conversation_chain(vector_store)
        
  #st.session_state.conversation


if __name__ =='__main__':
  main()