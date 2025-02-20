import os
import sys
import ssl
import warnings
import logging
import math
import queue
import threading
import time
import textwrap
from typing import List, Set, Tuple
from urllib.error import URLError
from functools import lru_cache
from io import BytesIO
from collections import Counter

# Third-party imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from PyPDF2 import PdfWriter, PdfReader
from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for cross-platform compatibility
import matplotlib.pyplot as plt
import shutil
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Suppress warnings and configure logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_HUB_OFFLINE'] = '1'

# Create unverified HTTPS context for NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Suppress print output
class SuppressPrintOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

# Download NLTK data silently
with SuppressPrintOutput():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "data/notes"

# Load environment variables
load_dotenv()

# Disable HuggingFace's internet connection check
def dummy_urlopen(*args, **kwargs):
    raise URLError("No internet")

import urllib.request
urllib.request.urlopen = dummy_urlopen

def print_welcome_message():
    welcome_text = """
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║   🚀 Welcome to the College Notes RAG System! 📚           ║
    ║                                                            ║
    ║   This project allows you to create a searchable database  ║
    ║   of your PDF documents using advanced AI techniques.      ║
    ║                                                            ║
    ║   Type 'python college_notes_rag.py instructions' for help.║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """
    print(welcome_text)

def print_instructions():
    instructions = """
    🚀 College Notes RAG System - Available Commands 📚

    1. 🏗️  create
       Create a new database from PDF documents in the data/notes directory.
       Usage: python college_notes_rag.py create

    2. 📋 list
       List all documents currently in the database.
       Usage: python college_notes_rag.py list

    3. ➕ add <file>
       Add a new PDF file to the database.
       Usage: python college_notes_rag.py add /path/to/your/file.pdf

    4. 🗑️  delete <file>
       Remove a specific file from the database.
       Usage: python college_notes_rag.py delete filename.pdf

    5. 🔄 update
       Update the database with new or removed files from the data/notes directory.
       Usage: python college_notes_rag.py update

    6. 🧹 clear
       Clear all contents from the database.
       Usage: python college_notes_rag.py clear

    7. 🔍 search <term>
       Search for documents by name in the database.
       Usage: python college_notes_rag.py search lecture

    8. 💬 converse <document>
       Start an AI-powered conversation about a specific document.
       Usage: python college_notes_rag.py converse document_name.pdf

    9. 🔎 semantic_search <query>
       Perform a semantic search across all documents and visualize key terms.
       Usage: python college_notes_rag.py semantic_search "quantum physics"

    10. ℹ️  instructions
        Show this help message.
        Usage: python college_notes_rag.py instructions

    📌 Note: 
    - Ensure your PDF documents are in the 'data/notes' directory before creating or updating the database.
    - The 'converse' and 'semantic_search' commands use AI to analyze document content.
    - Use Control+C to cancel long-running operations like semantic search.

    For more detailed information about each command, refer to the project documentation.
    """
    print(instructions)

class CollegeNotesRAG:
    def __init__(self):
        self.embedding_function = self.get_embedding_function()
        self.db = self.load_database()
        self.llm = ChatAnthropic(
            model_name="claude-3-sonnet-20240229",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.cancel_search = False
        self.search_results = None
        self.search_completed = None
        self.cancel_event = threading.Event()
        self.input_queue = queue.Queue()
    
    @staticmethod
    def get_embedding_function():
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def create_database(self):
        print("🚀 Creating database...")
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
            print(f"📁 Created directory: {DATA_PATH}")

        documents = DirectoryLoader(DATA_PATH, glob="*.pdf").load()
        if not documents:
            print("❌ No PDF documents found in the data/notes directory.")
            return

        # Refined chunking strategy
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)

        print(f"Processing {len(chunks)} chunks...")

        self.db = Chroma.from_documents(chunks, self.embedding_function, persist_directory=CHROMA_PATH)
        self.db.persist()
        print(f"✅ Database created with {len(chunks)} chunks from {len(documents)} documents.")

    def load_database(self):
        if not os.path.exists(CHROMA_PATH):
            return None
        try:
            return Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embedding_function)
        except Exception as e:
            print(f"❌ Error loading database: {str(e)}")
            return None
    
    def database_exists(self):
        exists = os.path.exists(CHROMA_PATH) and os.path.isdir(CHROMA_PATH)
        print(f"Checking if database exists: {exists}")
        return exists

    def semantic_search(self, query, top_k=5):
        if self.db is None:
            print("🚫 Database not loaded. Please run 'create' command first.")
            return [], []
        results = self.db.similarity_search_with_score(query, k=top_k)
        return [(doc.metadata, doc.page_content) for doc, score in results], [doc.page_content for doc, score in results]
    
    def extract_key_terms(self, text, top_n=10):
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]
        return Counter(words).most_common(top_n)
    
    def semantic_search_and_visualize(self, query):
        print(f"🔍 Performing semantic search for: '{query}'")
        print("ℹ️  Press Control+C at any time to cancel the search.")
        
        try:
            metadata, documents = self.semantic_search(query)
            all_text = " ".join(documents)
            key_terms = self.extract_key_terms(all_text)
            
            print("📊 Visualizing results...")
            terms, counts = zip(*key_terms)
            plt.figure(figsize=(10, 6))
            plt.bar(terms, counts)
            plt.title(f"Key Terms for: '{query}'")
            plt.xlabel("Terms")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

            print("\n📝 Top related passages:")
            for i, (meta, doc) in enumerate(zip(metadata, documents), 1):
                print(f"\n{i}. From: {os.path.basename(meta['source'])}")
                print(f"   {doc[:200]}...")
        except KeyboardInterrupt:
            print("\n🛑 Search cancelled.")
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")
            
    def _check_for_cancel(self):
        while not self.cancel_event.is_set():
            try:
                user_input = self.input_queue.get(timeout=0.1)
                if user_input.lower() == 'c':
                    self.cancel_event.set()
                    print("Cancelling search...")
                    break
            except queue.Empty:
                continue
    
    def _perform_search(self, query):
        try:
            print("Starting semantic search in thread...")
            if self.cancel_event.is_set():
                return
            
            metadata, documents = self.semantic_search(query)
            
            if self.cancel_event.is_set():
                return

            print("Combining documents...")
            all_text = " ".join(documents)
            
            if self.cancel_event.is_set():
                return
            
            print("Extracting key terms...")
            key_terms = self.extract_key_terms(all_text)
            
            if self.cancel_event.is_set():
                return

            print("Storing search results...")
            self.search_results = {
                'metadata': metadata,
                'documents': documents,
                'key_terms': key_terms
            }
            print("Search completed in thread.")
        except Exception as e:
            print(f"An error occurred during the search in thread: {str(e)}")
            
    def _visualize_results(self, query, results):
        print("Starting visualization...")
        terms, counts = zip(*results['key_terms'])
        plt.figure(figsize=(10, 6))
        plt.bar(terms, counts)
        plt.title(f"Key Terms for: '{query}'")
        plt.xlabel("Terms")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        print("Displaying plot...")
        plt.show()
        print("Plot displayed.")

        print("\nTop related passages:")
        for i, (meta, doc) in enumerate(zip(results['metadata'], results['documents']), 1):
            print(f"\n{i}. From: {os.path.basename(meta['source'])}")
            print(f"   {doc[:200]}...")
    
    def start_conversation(self, document_name: str) -> List[Tuple[str, str]]:
        chat_history = []
        print(f"🚀 Starting conversation about 📄 {document_name}")
        print("💡 Type 'exit' to end the conversation")

        retriever = self.db.as_retriever(search_kwargs={"k": 5})
        qa_chain = self.create_qa_chain(retriever)

        while True:
            query = input("🙋 You: ")

            if query.lower() == 'exit':
                print("👋 Ending conversation...")
                break

            result = qa_chain({"question": query, "chat_history": chat_history})
            answer = result['answer']
            print(f"🤖 AI: {answer}")

            chat_history.append((query, answer))

        return chat_history
    
    def create_qa_chain(self, retriever):
        prompt_template = """You are an AI assistant helping with questions about a PDF document. 
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        When answering, please follow these guidelines:
        1. Provide specific examples from the text when relevant, including any case studies or scenarios mentioned.
        2. If the question asks about a particular section or concept, focus on that specific information.
        3. Include key points and main ideas from the relevant sections, using the exact wording from the text where appropriate.
        4. If there are numbered lists, steps, or rules in the text, include them in your answer.
        5. Use quotation marks for direct quotes from the text.
        6. Highlight any unique ideas or approaches mentioned in the text, especially those that might be counterintuitive or go against common assumptions.
        7. If the text mentions specific examples or scenarios, be sure to include them in your answer.

        {context}

        Based on the above context, provide a comprehensive and specific answer to the following question, making sure to include any relevant examples, unique ideas, and specific scenarios mentioned in the text:
        Question: {question}
        Answer: """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )


    def converse_and_export(self, document_name):
        if not self.db:
            print("❌ Database not initialized. Please create or load a database first.")
            return

        chat_history = []
        print(f"🚀 Starting conversation about 📄 {document_name}")
        print("💡 Type 'exit' to end the conversation")

        retriever = self.db.as_retriever(search_kwargs={
            "k": 7  # Retrieve 7 most relevant chunks
        })
        qa_chain = self.create_qa_chain(retriever)

        while True:
            query = input("🙋 You: ")

            if query.lower() == 'exit':
                print("👋 Ending conversation...")
                break

            result = qa_chain({"question": query, "chat_history": chat_history})
            answer = result['answer']
            source_docs = result['source_documents']

            print(f"🤖 AI: {answer}\n")
            print("📚 Sources used:")
            for i, doc in enumerate(source_docs, 1):
                print(f"  {i}. {doc.metadata['source']} (Page: {doc.metadata.get('page', 'N/A')})")
                print(f"     Excerpt: {doc.page_content[:100]}...")
            print()

            chat_history.append((query, answer))
        if chat_history:
            export_choice = input("💾 Do you want to export the conversation? (yes/no): ").lower()
            if export_choice == 'yes':
                format = input("📁 Export format (pdf/md/docx): ").lower()
                if format in ['pdf', 'md', 'docx']:
                    content = self.prepare_export_content(chat_history, document_name)
                    desktop_path = os.path.expanduser("~/Desktop")
                    filename = os.path.join(desktop_path, f"conversation_export_{document_name}_{len(chat_history)}_exchanges")
                    try:
                        if format == 'pdf':
                            self.export_to_pdf(content, f"{filename}.pdf")
                        elif format == 'md':
                            with open(f"{filename}.md", 'w') as f:
                                f.write(content)
                        elif format == 'docx':
                            from docx import Document
                            doc = Document()
                            doc.add_paragraph(content)
                            doc.save(f"{filename}.docx")
                        print(f"✅ Exported conversation to {filename}.{format}")
                    except Exception as e:
                        print(f"❌ Error exporting conversation: {str(e)}")
                else:
                    print("❌ Invalid format. Conversation not exported.")
            else:
                print("👋 Conversation ended without exporting.")

                
                
    def print_document_content(self, document_name):
        if self.db is None:
            print("Database not initialized.")
            return
        
        print(f"📄 Printing content for {document_name}:")
        results = self.db.similarity_search("", filter={"source": os.path.join(DATA_PATH, document_name)}, k=10)
        for i, doc in enumerate(results):
            print(f"Chunk {i + 1}:")
            print(f"Content: {doc.page_content}")
            print("---")

    @staticmethod
    def prepare_export_content(chat_history, document_name):
        content = f"🗣️ Conversation Log for document: {document_name}\n\n"
        for i, (question, answer) in enumerate(chat_history, 1):
            content += f"❓ Q{i}: {question}\n"
            content += f"💡 A{i}: {answer}\n\n"
        return content

    def process_document(self, file_path):
        print(f"📖 Loading document: {os.path.basename(file_path)}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"📄 Document loaded. Pages: {len(documents)}")
    
        print("✂️ Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        print(f"🧩 Document split into {len(chunks)} chunks.")
    
        return chunks

    def search_documents(self, search_term):
        print(f"🔄 Searching for '{search_term}'")
        documents = self.db.get()
        matching_docs = [
            doc['source'] for doc in documents['metadatas']
            if search_term.lower() in os.path.basename(doc['source']).lower()
        ]
        
        if matching_docs:
            print(f"📚 Documents matching '{search_term}':")
            for doc in set(matching_docs):  # Use set to remove duplicates
                print(f"  📄 {os.path.basename(doc)}")
        else:
            print(f"❌ No documents found matching '{search_term}'")

    def list_documents(self):
        if self.db is None:
            print("📢 No database found. Add documents to create a database.")
            return
    
        try:
            documents = self.db.get()
            if not documents['metadatas']:
                print("📢 Database is empty. No documents found.")
            else:
                print("📋 Documents in the database:")
                unique_sources = set(doc['source'] for doc in documents['metadatas'])
                for source in unique_sources:
                    print(f"  📄 {os.path.basename(source)}")
        except Exception as e:
            print(f"❌ Error listing documents: {str(e)}")
    

    def delete_document(self, filename):
        if self.db is None:
            print("🚫 Database not loaded. Please run 'create' command first.")
            return
        try:
            documents = self.db.get()
            to_delete = [i for i, meta in enumerate(documents['metadatas']) if os.path.basename(meta['source']) == filename]
            if not to_delete:
                print(f"❌ Document '{filename}' not found in the database.")
                return
            self.db._collection.delete(ids=[documents['ids'][i] for i in to_delete])
            print(f"🗑️ Deleted '{filename}' from the database.")
        
        # Delete the file from DATA_PATH
            file_path = os.path.join(DATA_PATH, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Deleted '{filename}' from {DATA_PATH}")
            else:
                print(f"⚠️ File '{filename}' not found in {DATA_PATH}")
        except Exception as e:
            print(f"❌ Error deleting document: {str(e)}")

    def add_document(self, filepath):
        if not os.path.exists(filepath):
            print(f"❌ File '{filepath}' not found.")
            return
        filename = os.path.basename(filepath)
        destination = os.path.join(DATA_PATH, filename)
        os.makedirs(DATA_PATH, exist_ok=True)
        shutil.copy2(filepath, destination)
        print(f"📁 Copied '{filename}' to {DATA_PATH}")
        
        print(f"📄 Processing file: {filename}")
        try:
            print("🔍 Analyzing document content...")
            chunks = self.process_document(destination)
            print(f"✅ Document analyzed. Found {len(chunks)} chunks.")

            if not chunks:
                print("⚠️ No content extracted from the document. It might be empty or unreadable.")
                return

            print(f"💾 Adding document to database...")
            if self.db is None:
                print("📢 Creating new database...")
                batch_size = 100  # Adjust this value if needed
                initial_batch = chunks[:batch_size]
                self.db = Chroma.from_documents(initial_batch, self.embedding_function, persist_directory=CHROMA_PATH)
                chunks = chunks[batch_size:]  # Remaining chunks

            if chunks:  # If there are remaining chunks or if the database already existed
                batch_size = 100  # Adjust this value if needed
                total_batches = math.ceil(len(chunks) / batch_size)

                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                try:
                    self.db.add_documents(batch)
                    print(f"✅ Added batch {math.floor(i/batch_size) + 1} of {total_batches}")
                except Exception as e:
                    print(f"❌ Error adding batch {math.floor(i/batch_size) + 1}: {str(e)}")
                    return
            
            self.db.persist()
            print(f"🎉 Successfully added '{filename}' to the database.")
            print(f"📊 Total chunks added: {len(chunks)}")

        except KeyboardInterrupt:
            print("\n⚠️ Operation cancelled by user.")
        except Exception as e:
            print(f"❌ An error occurred while processing '{filename}': {str(e)}")
    # Optionally, you can add a step to verify the document was added
        self.verify_document_added(filename)

    def verify_document_added(self, filename):
        print(f"🔍 Verifying '{filename}' was added to the database...")
        if self.db is None:
            print("❌ Database is not initialized.")
            return
        try:
            documents = self.db.get()
            if any(filename == os.path.basename(doc['source']) for doc in documents['metadatas']):
                print(f"✅ Verified: '{filename}' is in the database.")
            else:
                print(f"⚠️ Warning: '{filename}' was not found in the database after adding.")
        except Exception as e:
            print(f"❌ An error occurred while verifying: {str(e)}")
        
        
    def update_database(self):
        print("🔄 Updating database...")
        try:
            db_files = set(os.path.basename(doc['source']) for doc in self.db.get()['metadatas'])
        except Exception as e:
            print(f"❌ Error accessing database: {str(e)}")
            return

        disk_files = set(f for f in os.listdir(DATA_PATH) if f.endswith('.pdf'))

        for file in db_files - disk_files:
            self.delete_document(file)

        for file in disk_files - db_files:
            print(f"Adding new file: {file}")
            self.add_document(os.path.join(DATA_PATH, file))

        print("✅ Database update process completed.")

    def clear_database(self):
        if not os.path.exists(CHROMA_PATH):
            print("🚫 Database not found. Nothing to clear.")
            return

        confirm = input("🚨 Are you sure you want to clear all contents from the database? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return

        try:
            shutil.rmtree(CHROMA_PATH)
            print("🧹 Database cleared successfully.")
            self.db = None  # Reset the database object
        except Exception as e:
            print(f"❌ Error clearing database: {str(e)}")

    @staticmethod
    def export_to_pdf(content, filename):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Register a default font (you can change this to a different font if needed)
        pdfmetrics.registerFont(TTFont('Arial', 'Arial.ttf'))
        c.setFont('Arial', 10)
        
        lines = content.split('\n')
        y_position = height - 50  # Start from the top with a small margin
        
        for line in lines:
            # Wrap the text to fit within the page width
            wrapped_lines = textwrap.wrap(line, width=100)  # Adjust the width as needed
            
            for wrapped_line in wrapped_lines:
                if y_position < 50:  # If we're near the bottom of the page
                    c.showPage()  # Start a new page
                    y_position = height - 50  # Reset y_position for the new page
                
                c.drawString(40, y_position, wrapped_line)
                y_position -= 15  # Move down for next line
            
            y_position -= 5  # Add a small gap between original lines
        
        c.save()
        
        # Move to the beginning of the BytesIO buffer
        buffer.seek(0)
        
        # Create a new PDF with reportlab's output
        new_pdf = PdfReader(buffer)
        
        # Create the final PDF
        output = PdfWriter()
        for page in new_pdf.pages:
            output.add_page(page)
        
        # Write the final PDF to a file
        with open(filename, "wb") as output_file:
            output.write(output_file)


def main():
    if len(sys.argv) < 2:
        print_welcome_message()
        print_instructions()
        return

    command = sys.argv[1]
    print(f"🎯 Command: {command}")

    if command == "instructions":
        print_instructions()
        return

    rag = CollegeNotesRAG()

    commands = {
        "create": rag.create_database,
        "list": rag.list_documents,
        "delete": rag.delete_document,
        "update": rag.update_database,
        "add": rag.add_document,
        "clear": rag.clear_database,
        "search": rag.search_documents,
        "converse": lambda doc_name: rag.converse_and_export(doc_name),
        "semantic_search": rag.semantic_search_and_visualize,
        "print_content": rag.print_document_content  # Add this line
    }

    if command in commands:
        if command in ["delete", "add", "search", "converse", "semantic_search", "print_content"] and len(sys.argv) < 3:
            print(f"ℹ️ Usage: python college_notes_rag.py {command} <filename, or search term>")
            return

        print(f"🏃 Executing command: {command}")
        try:
            if len(sys.argv) > 2:
                commands[command](" ".join(sys.argv[2:]))
            else:
                commands[command]()
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")
    else:
        print("❌ Invalid command. Use 'instructions' for help.")

if __name__ == "__main__":
    main()