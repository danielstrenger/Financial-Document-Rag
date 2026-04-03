import os
import chainlit as cl
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

db_path = "./chroma_db"

@cl.on_chat_start
async def on_chat_start():
    """
    Initializes the RAG pipeline upon a new user session.

    This function connects to the local ChromaDB vector store, instantiates the 
    HuggingFace embedding model, configures the ChatOpenAI generative engine, 
    and compiles the LCEL retrieval chain. The compiled chain is then stored 
    in the user's session state to maintain conversational context.

    Side Effects:
        - Reads from the local './chroma_db' directory.
        - Mutates cl.user_session by setting the 'rag_chain' key.
        - Emits an asynchronous welcome message to the UI.
    """

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load the embedding database
    if not os.path.exists(db_path):
        await cl.Message(content="Error: Database not found. Please run ingest.py first.").send()
        return

    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model
    )

    retriever=vector_store.as_retriever(search_kwargs={"k":4})

    # Define the LLM for the actual conversation.
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # System prompt template
    template="""You are a financial data assistent.
    Use the following pieces of retrieved context to answer the question.
    If you cannot find the answer in the provided context, state explicitly that you do not know. 
    Do not hallucinate external information. Keep the answer concise, objective, and professional.

    Context: {context}

    Question: {question}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Helper function to join retrieved documents into a single string
    def join_documents(docs: list[Document])->str:
        """
        Extracts and concatenates text content from retrieved Document objects.

        Args:
            docs (list[Document]): The top-k documents retrieved from ChromaDB.

        Returns:
            str: A single string containing all context separated by double newlines, 
                 formatted for injection into the LLM prompt.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    #Construct LCEL chain
    rag_chain=(
        {"context": retriever | join_documents, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    cl.user_session.set("rag_chain", rag_chain)

    await cl.Message("Welcome. I am the Financial RAG assistant. I am connected to the documents you processed. What would you like to know?").send()

# Handle user messages
@cl.on_message
async def on_message(message: cl.Message):
    """
    Processes incoming user queries and streams the LLM response to the UI.

    Retrieves the compiled LCEL retrieval chain from the current user's session 
    state. Executes the chain asynchronously using the user's input, and iterates 
    over the resulting token stream.

    Args:
        message (cl.Message): The Chainlit message object containing the raw 
                              string input from the user.

    Side Effects:
        - Reads 'rag_chain' from cl.user_session.
        - Emits and updates asynchronous UI messages.
    """
    rag_chain=cl.user_session.get("rag_chain")

    if not rag_chain:
        await cl.Message(content="System not initialized properly.").send()

    msg=cl.Message(content="")
    await msg.send()

    async for chunk in rag_chain.astream(message.content):
        await msg.stream_token(chunk)

    await msg.update()