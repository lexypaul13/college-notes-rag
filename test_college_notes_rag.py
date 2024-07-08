import os
import shutil
from college_notes_rag import CollegeNotesRAG, DATA_PATH, CHROMA_PATH

rag = CollegeNotesRAG()

def create_test_pdf():
    from reportlab.pdfgen import canvas
    test_pdf_path = "test.pdf"
    c = canvas.Canvas(test_pdf_path)
    c.drawString(100, 100, "This is a test PDF for the College Notes RAG system.")
    c.save()
    return test_pdf_path

def test_add_document(test_pdf_path):
    print("Testing add_document...")
    rag.add_document(test_pdf_path)
    assert os.path.exists(os.path.join(DATA_PATH, "test.pdf")), "File not copied to DATA_PATH"

def test_create_database():
    print("Testing create_database...")
    rag.create_database()
    assert os.path.exists(CHROMA_PATH), "Database not created"

def test_list_documents():
    print("Testing list_documents...")
    rag.list_documents()

def test_search_documents():
    print("Testing search_documents...")
    rag.search_documents("test")

def test_delete_document():
    print("Testing delete_document...")
    rag.delete_document("test.pdf")
    assert not os.path.exists(os.path.join(DATA_PATH, "test.pdf")), "File not deleted from DATA_PATH"

def test_clear_database():
    print("Testing clear_database...")
    rag.clear_database()
    assert not os.path.exists(CHROMA_PATH), "Database not cleared"

def test_converse():
    print("Testing converse...")
    print("This test requires manual interaction. Please respond to the prompts.")
    rag.converse_and_export("test.pdf")

def run_tests():
    test_pdf_path = create_test_pdf()
    try:
        test_add_document(test_pdf_path)
        test_create_database()
        test_list_documents()
        test_search_documents()
        test_converse()
        test_delete_document()
        test_clear_database()
        print("All tests completed successfully!")
    except AssertionError as e:
        print(f"Test failed: {str(e)}")
    except Exception as e:
        print(f"An error occurred during testing: {str(e)}")
    finally:
        # Clean up
        if os.path.exists(DATA_PATH):
            shutil.rmtree(DATA_PATH)
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        if os.path.exists(test_pdf_path):
            os.remove(test_pdf_path)

if __name__ == "__main__":
    run_tests()