# chat.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from rich.console import Console
from rich.markdown import Markdown

load_dotenv()

# Configuration
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_chatbot"
TOP_K = 5

console = Console()

def get_vector_store():
    """Connect to existing ChromaDB vector store."""
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

def format_docs(docs):
    """Format retrieved documents into a single context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(
            f"[Source {i}: {source}, Page {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)

# System prompt with retrieval augmented generation instructions
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions
based on the provided context. Follow these rules strictly:

1. Only answer based on the provided context
2. If the context does not contain enough information, say so
3. Cite your sources using [Source N] notation
4. Be concise but thorough
5. If asked about something outside the context, explain that
   your knowledge is limited to the provided documents

Context:
{context}"""),
    ("human", "{question}")
])

def build_rag_chain():
    """Build the complete RAG chain."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 20}
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        max_tokens=2048
    )

    chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

def main():
    """Run the interactive chatbot."""
    console.print("\n[bold green]RAG Chatbot Ready[/bold green]")
    console.print("Type your questions below. Type 'quit' to exit.\n")

    chain = build_rag_chain()

    while True:
        question = console.input("[bold cyan]You:[/bold cyan] ")
        if question.lower() in ("quit", "exit", "q"):
            console.print("[yellow]Goodbye![/yellow]")
            break

        console.print("\n[bold green]Assistant:[/bold green]")
        response = chain.invoke(question)
        console.print(Markdown(response))
        console.print()

if __name__ == "__main__":
    main()