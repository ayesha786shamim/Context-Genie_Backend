import fitz  # PyMuPDF is used for reading PDF files

def load_pdf_with_page_numbers(pdf_path):
    """
    Loads a PDF file and extracts text from each page along with its page number.
    Returns a list of dictionaries with 'text' and 'page' keys.
    """
    doc = fitz.open(pdf_path)  # Open the PDF
    chunks = []  # This will hold our page-wise text chunks

    for page_number in range(len(doc)):
        # Extract text from each page
        text = doc[page_number].get_text()
        
        if text.strip():  # Ignore empty or whitespace-only pages
            chunks.append({
                "text": text,           # Page content
                "page": page_number + 1  # Store page number (1-based indexing)
            })

    return chunks  # List of dicts like [{text: "...", page: 1}, ...]
