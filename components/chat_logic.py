import gradio as gr
import logging
from services.search_service import SearchLoader
from langchain_deepseek import ChatDeepSeek
from langchain_ollama.llms import OllamaLLM

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from components.ranking_modes import reciprocal_rank_fusion, get_unique_union
from utils.config import Config

# Token limits
MAX_TOKENS_CHAT = 5000
MAX_TOKENS_CONTEXT = 3000
MAX_TOKENS_HISTORY = 500
MAX_HISTORY_MESSAGES = 3

# Initialize logging
logging.basicConfig(level=logging.INFO)


def truncate_chat_history(chat_history):
    """Truncates chat history to the last MAX_HISTORY_MESSAGES messages."""
    return chat_history[-MAX_HISTORY_MESSAGES:]


def truncate_documents(documents):
    """Truncates document content to MAX_TOKENS_CONTEXT words."""
    for doc in documents:
        doc.page_content = " ".join(doc.page_content.split()[:MAX_TOKENS_CONTEXT])
    return documents


def server(user_message, history, selected_model="gpt-4o", temperature=0.7, n_results=5, search_type="Search", rag_type="RAG Fusion"):
    if not user_message.strip():
        return history  # Don't process empty messages

    # Ensure history is initialized correctly
    if history is None or not isinstance(history, list):
        history = []

    print("Received user input:", user_message)

    # Initialize language models
    if(selected_model == "gpt-4o"):
        llm = ChatOpenAI(model=selected_model, temperature=temperature, max_tokens=None, max_retries=2)
    else:
        # llm = OllamaLLM(model=selected_model, temperature=temperature, max_tokens=None, max_retries=2)
        llm = ChatDeepSeek(model=selected_model, temperature=temperature, max_tokens=None, max_retries=2)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Load and preprocess documents
    documents, src_info = SearchLoader(user_message, n_results, llm)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Vector store setup
    vectorstore = Qdrant.from_documents(
        documents=splits, 
        embedding=embeddings, 
        location=Config.QDRANT_HOST,
        collection_name=Config.COLLECTION_NAME
    )

    retriever = vectorstore.as_retriever()

    # Convert chat history to expected format
    chat_history = [user_message.content for user_message in truncate_chat_history(history)] if search_type == "Search" else []

    # RAG Prompt
    system_template = """Answer the following question based on this context:
    Make sure the output is in markdown format.
    {context}
    Question: {question}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", "{question}")
        ]
    )

    if rag_type == "RAG Fusion": # RAG Fusion using reciprocal rank fusion
        template = """Generate multiple search queries related to: {question}"""
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_rag_fusion
            | llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion
        print(retrieval_chain)
        final_rag_chain = (
            {"context": retrieval_chain,
            "question": itemgetter("question")}
            | qa_prompt
            | llm
            | StrOutputParser()
        )

    else:
        template = """Generate multiple related search queries for {question}"""
        prompt_perspectives = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_perspectives
            | llm
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        retrieval_chain = generate_queries | retriever.map() | get_unique_union

        final_rag_chain = (
            {"context": retrieval_chain, "question": itemgetter("question")}
            | qa_prompt
            | llm
            | StrOutputParser()
        )

    # Get response
    bot_response = final_rag_chain.invoke({"question": user_message})
    chat = []
    # Append messages in the new Gradio "messages" format
    chat.append({"role": "user", "content": user_message})
    chat.append({"role": "assistant", "content": bot_response})

    return chat
