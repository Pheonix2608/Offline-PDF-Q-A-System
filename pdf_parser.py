# import fitz

# def extract_text_from_multiple_pdfs(file_paths):
#     texts = {}
#     for path in file_paths:
#         doc = fitz.open(path)
#         text = ""
#         for page in doc:
#             text += page.get_text()
#         texts[path] = text
#     return texts

# def chunk_text(text, chunk_size=500, overlap=50):
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunk = text[start:end]
#         chunks.append(chunk)
#         start += chunk_size - overlap
#     return chunks


import fitz  # PyMuPDF
import docx
import pandas as pd

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "".join(page.get_text() for page in doc)

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def extract_text_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string()

def extract_text_from_multiple_files(file_paths):
    ext_map = {
        ".pdf": extract_text_from_pdf,
        ".docx": extract_text_from_docx,
        ".txt": extract_text_from_txt,
        ".csv": extract_text_from_csv,
    }
    texts = {}
    for path in file_paths:
        ext = path.lower().split(".")[-1]
        extractor = ext_map.get(f".{ext}")
        if extractor:
            texts[path] = extractor(path)
    return texts

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
