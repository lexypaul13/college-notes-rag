

**College Notes RAG System**

**Overview**

The College Notes RAG (Retrieval-Augmented Generation) system is an advanced tool designed to help students and researchers efficiently manage, search, and interact with their PDF documents, primarily focusing on college notes and academic papers. This system leverages AI technologies to provide semantic search capabilities and enable conversational interactions with document content.

**Features**

- **Document Management:** Add, list, and delete PDF documents in the system.
- **Semantic Search:** Perform content-based searches across all added documents.
- **Conversational AI:** Engage in AI-powered conversations about specific documents.
- **Database Management:** Create, update, and clear the document database.
- **Export Functionality:** Export conversations for future reference.

**Installation**

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/college-notes-rag.git
   cd college-notes-rag
   ```

2. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   - Copy `.env.sample` to `.env`
   - Open `.env` and replace `your_api_key_here` with your actual Anthropic API key

**Usage**

Run the script with a command and any necessary arguments:

```
python college_notes_rag.py <command> [arguments]
```

**Available Commands:**

- `create`: Create a new database
- `list`: List all documents in the database
- `add <file>`: Add a new PDF file to the database
- `delete <file>`: Remove a file from the database
- `update`: Update the database with new or removed files
- `clear`: Clear all contents from the database
- `search <term>`: Search for documents by name
- `converse <document>`: Start a conversation about a document
- `semantic_search <query>`: Perform a semantic search across all documents

**Examples**

1. Add a document:
   ```
   python college_notes_rag.py add path/to/your/document.pdf
   ```

2. Search for documents:
   ```
   python college_notes_rag.py search quantum
   ```

3. Start a conversation about a document:
   ```
   python college_notes_rag.py converse document_name.pdf
   ```

**Project Structure**

- `college_notes_rag.py`: Main script containing the CollegeNotesRAG class and command-line interface.
- `data/notes/`: Directory where added PDF documents are stored.
- `chroma/`: Directory for the Chroma vector database.
- `.env`: Configuration file for API keys (not tracked by Git).
- `.env.sample`: Template for the `.env` file.
- `requirements.txt`: List of Python package dependencies.

**Contributing**

Contributions to improve the College Notes RAG system are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature.
3. Commit your changes.
4. Push to your fork and submit a pull request.

**License**

[Specify your license here, e.g., MIT, GPL, etc.]

**Disclaimer**

This project is for educational purposes only. Ensure you have the right to use and process any documents you add to the system.

---
