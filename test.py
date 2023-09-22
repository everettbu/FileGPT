import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# PDF file path
pdf_path = "./paper.pdf"

# Initialize the PDF loader
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# Create embeddings
embeddings = OpenAIEmbeddings()

# Create a vector store
vectordb = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")
vectordb.persist()

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the Conversational Retrieval Chain
pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.8), vectordb.as_retriever(), memory=memory)

# Start a continuous conversation
while True:
    # Prompt the user for a question
    query = input("Ask a question (or 'exit' to end): ")

    # Check if the user wants to exit
    if query.lower() == "exit":
        print("Goodbye!")
        break

    elif query.lower() == "load":
        """ Insert funtionallity here """
        break

    # Get the answer
    result = pdf_qa({"question": query})
    answer = result["answer"]

    # Print the answer
    print("Answer:", answer)

