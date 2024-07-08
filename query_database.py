import os
import sys
import logging
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

# Suppress specific loggers
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Set environment variables to suppress specific warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

CHROMA_PATH = "chroma"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def main():
    try:
        query = get_query_from_user()
        print(f"🔍 Query: {query}")

        print("🔧 Initializing...")
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        retriever = db.as_retriever()
        llm = ChatAnthropic(model_name="claude-3-sonnet-20240229", anthropic_api_key=ANTHROPIC_API_KEY)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        print("🤖 Querying the database...")
        response = qa_chain.invoke(query)
        print("💡 Response:")
        print(response['result'])
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

def get_query_from_user():
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    else:
        return input("🖊️ Enter your query: ")

if __name__ == "__main__":
    print("🚀 Starting query process...")
    if not os.path.exists(CHROMA_PATH):
        print(f"❌ Error: Chroma database not found at {CHROMA_PATH}")
        sys.exit(1)
    main()