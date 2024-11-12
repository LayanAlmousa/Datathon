from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import joblib
import nest_asyncio  # noqa: E402

# Your imports here
from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader

nest_asyncio.apply()

# Load environment variables
load_dotenv()
# llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
llamaparse_api_key = "llx-21ibGvKJj5cwkEYTOrDBEEXpt6pUa5q3X4kEKh8bBPD6A0CY"
# groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_key = "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"

app = Flask(__name__)
    
# Define functions as in your original code
def load_or_parse_data():
    data_file = "data/parsed_data.pkl"

    if os.path.exists(data_file):
        # Load the parsed data from the file
        parsed_data = joblib.load(data_file)
    else:
        # Perform the parsing step and store the result in llama_parse_documents
        parsingInstructionUber10k = """
            The provided document contains financial statements reports about Elm Company in the Saudi Exchange website for a specific year or a specific quarter.
            Extract information as it is in the document no more no less.
            """
        parser = LlamaParse(api_key=llamaparse_api_key,
                            result_type="markdown",
                            parsing_instruction=parsingInstructionUber10k,
                            max_timeout=5000,)
        llama_parse_documents = parser.load_data("data/All-elm.pdf")


        # Save the parsed data to a file
        print("Saving the parse results in .pkl format ..........")
        joblib.dump(llama_parse_documents, data_file)

        # Set the parsed data to the variable
        parsed_data = llama_parse_documents

    return parsed_data


def retrieve_vs():
    # Define the path for the vector database directory
    vector_db_path = "chroma_db_llamaparse1"

    # Check if the vector database exists
    if os.path.exists(vector_db_path):
        # If the database exists, load it
        print("Loading the existing vector database...")
        vs = Chroma(persist_directory=vector_db_path, embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-m3"), collection_name="rag")
    else:
        # Call the function to either load or parse the data
        llama_parse_documents = load_or_parse_data()
        print(llama_parse_documents[0].text[:300])

        with open('data/output.md', 'a') as f:
            for doc in llama_parse_documents:
                f.write(doc.text + '\n')

        markdown_path = "data/output.md"
        loader = UnstructuredMarkdownLoader(markdown_path)
        documents = loader.load()

        # Split loaded documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # Initialize Embeddings
        embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

        # Create and persist a Chroma vector database from the chunked documents
        print(f"length of documents loaded: {len(documents)}")
        print(f"total number of document chunks generated: {len(docs)}")

        vs = Chroma.from_documents(
            documents=docs,
            embedding=embed_model,
            persist_directory=vector_db_path,  # Local mode with persistent storage
            collection_name="rag"
        )

        print('Vector DB created and saved successfully!')

    return vs


def set_custom_prompt():
    custom_prompt_template = """Only extract the information from the vector database.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Please respond in Arabic if the query is in Arabic.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

chat_model = ChatGroq(temperature=0, model_name="llama3-groq-70b-8192-tool-use-preview", api_key=groq_api_key)
vectorstore = retrieve_vs()
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
prompt = set_custom_prompt()

qa = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Define route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for handling the query
@app.route('/query', methods=['POST'])
def query():
    user_query = request.form['query']
    response = qa.invoke({"query": user_query})
    result = response['result']
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
