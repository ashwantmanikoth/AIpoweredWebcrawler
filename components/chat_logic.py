from shiny import ui
from services.search_service import SearchLoader
# from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from components.reciprocal_rank_fusion import reciprocal_rank_fusion, get_unique_union
load_dotenv()

messages = [
    {
        "type": "markdown",
        "content": """
    ##### **AI powered Crawler**
    - This is a AI powered crawler that can search the web for 
    information based on your input.
    Utilizing Retrieval Augmented Generation (RAG) using Chroma.
    This application can be customized to any use-case.
    """,
    }
]

history = []


def server(input, output, session):
    chat = ui.Chat(id="chat", messages=messages)

    @chat.on_user_submit
    async def _():
        user_prompt = f"{chat.user_input()}\n\nMake the output as detailed as possible and output should be in markdown format"
        selected_model = input.models()
        temperature = input.temp()
        n_results = input.n_results()
        search_type = input.rag_type()



        # OpenAI API using langchain
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
            # base_url="...",
            # organization="...",
            # other params...
        )

        # Deepseek API using langchain
        # llm = ChatDeepSeek(
        #     model=selected_model,
        #     temperature=0,
        #     max_tokens=None,
        #     timeout=None,
        #     max_retries=2,
        #     # other params...
        # )

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

                # Load and preproccess documents
        documents, src_info = SearchLoader(user_prompt, n_results, llm)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        chat_history = []

        # RAG
        system_template = """Answer the following question based on this context:
        Make sure the output is in mardown format.
        {context}

        Also Take into consideration the Chat History:
        {chat_history}

        Question: {question}
        """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )

        if search_type == "RAG Fusion":
            # RAG-Fusion
            template = """"You are a helpful assistant designed to generate multiple search queries based on a single input query. 
            Please generate four different search queries related to the following question: {question}"""

            prompt_rag_fusion = ChatPromptTemplate.from_template(template)

            generate_queries = (
                prompt_rag_fusion
                | llm
                | StrOutputParser() 
                | (lambda x: x.split("\n"))
            )

            retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
            
            final_rag_chain = (
                {"context": retrieval_chain_rag_fusion, 
                "question": itemgetter("question"),
                "chat_history" : itemgetter("chat_history")} 
                | qa_prompt
                | llm
                | StrOutputParser()
            )

        else:
            # Multi Query
            template = """You are an AI language model assistant. 
            Your task is to create five different variations of the given user question to retrieve relevant 
            documents from a vector database. By offering multiple perspectives on the user question, your goal 
            is to help the user mitigate the limitations of distance-based similarity searches. 
            Present these alternative questions separated by newlines. Original 
            question: {question}"""

            prompt_perspectives = ChatPromptTemplate.from_template(template)
            
            generate_queries = (
                prompt_perspectives
                | llm
                | StrOutputParser() 
                | (lambda x: x.split("\n"))
            )

            retrieval_chain = generate_queries | retriever.map() | get_unique_union
            
            final_rag_chain = (
                {"context": retrieval_chain, 
                "question": itemgetter("question"),
                "chat_history" : itemgetter("chat_history")} 
                | qa_prompt
                | llm
                | StrOutputParser()
            )


        res = final_rag_chain.invoke({"question":user_prompt, "chat_history" : chat_history})

        history.append(HumanMessage(content=user_prompt))
        history.append(AIMessage(content=res))

        # Append a response to the chat
        await chat.append_message({"type": "text", "content": res})

