from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLM and embeddings
llm = ChatGroq(api_key=GROQ_API_KEY, model="Llama3-8b-8192")

CHROMA_PATH = "Chroma"

# Define the system prompt for the assistant
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n{context}"
)



def query_rag():
    embedding_function=get_embedding_function()
    db=Chroma(embedding_function=embedding_function,persist_directory=CHROMA_PATH)
    retriever=db.as_retriever()

    # Create the chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create the RAG chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response=rag_chain.invoke({"input":"What is the full form of DBMS"})
    response['answer']
    return response['answer']


if __name__=="__main__":
    response=query_rag()
    print(response)
