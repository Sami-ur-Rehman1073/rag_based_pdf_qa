import os
from pypdf import PdfReader

# -----------------------------------------------
# extract_text_from_pdf()
# 
# Takes a file path to a PDF
# Reads every page and joins all text together
# Returns one big string of the full PDF content
# -----------------------------------------------
def extract_text_from_pdf(file_path: str) -> str:
    
    # Check if file actually exists before reading
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")
    
    reader = PdfReader(file_path)
    
    all_text = []
    
    # Loop through every page in the PDF
    for page_number, page in enumerate(reader.pages):
        
        # Extract text from this page
        text = page.extract_text()
        
        # Some pages may be empty or image-only
        # Only add if there is actual text content
        if text and text.strip():
            all_text.append(text)
    
    # Join all pages with a newline separator
    full_text = "\n".join(all_text)
    
    # If we got nothing at all, the PDF might be scanned/image-based
    if not full_text.strip():
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned image-based PDF which is not supported."
        )
    
    return full_text