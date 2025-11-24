from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # disable static file caching during development


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



embeddings = download_embeddings()

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=384,  # Dimension of the embeddings
        metric= "cosine",  # Cosine similarity
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)



docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})




chatModel = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Prompt template
prompt = ChatPromptTemplate.from_template("{context}\n\nQ: {question}\nA:")
# LLM chain
llm_chain = LLMChain(prompt=prompt, llm=chatModel)
# StuffDocumentsChain
question_answer_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context" 
)
# RAG chain
rag_chain = RetrievalQA(
    retriever=retriever,
    combine_documents_chain=question_answer_chain
)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"query": msg})
    print("Response : ", response["result"])
    return str(response["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)