
from pdf_parser import extract_text_from_multiple_pdfs, chunk_text
from embedder import get_embeddings
from retriever import FAISSRetriever
from qa_engine import summarize_text, chat_with_memory, generate_answer
import os
from fpdf import FPDF


# Load multiple PDFs from folder
folder = "docs"
pdf_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pdf")]
texts_by_file = extract_text_from_multiple_pdfs(pdf_paths)

# Combine all texts for summarization + chunking
combined_text = "\n".join(texts_by_file.values())
summary = summarize_text(combined_text)
print("\n📚 Document Summary:\n", summary)

# Chunk + Embed
all_chunks = []
for text in texts_by_file.values():
    all_chunks.extend(chunk_text(text))

embeddings = get_embeddings(all_chunks)
retriever = FAISSRetriever(dim=len(embeddings[0]))
retriever.add(embeddings, all_chunks)

# Query input
question = input("Ask a question about the PDF: ")
query_embedding = get_embeddings([question])[0]
matched_chunks = retriever.search(query_embedding)

# Generate answer
answer = generate_answer(matched_chunks, question)
print("\n🤖 Answer:\n", answer)


def export_to_pdf(summary, qa_list, out_file="qa_summary.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.multi_cell(0, 10, "📄 PDF Summary:\n" + summary + "\n\n")
    pdf.multi_cell(0, 10, "🤖 Q&A Log:\n")
    for qa in qa_list:
        pdf.multi_cell(0, 10, qa + "\n")

    pdf.output(out_file)

def export_to_txt(summary, qa_list, out_file="qa_summary.txt"):
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("📄 PDF Summary:\n")
        f.write(summary + "\n\n")
        f.write("🤖 Q&A Log:\n")
        for qa in qa_list:
            f.write(qa + "\n")

# Summarize document
summary = summarize_text(text)
print("\n📚 Document Summary:\n", summary)

qa_log = []
while True:
    question = input("\nAsk a question (or type 'exit'): ")
    if question.lower() == 'exit':
        save = input("Do you want to export this session? (yes/no): ").lower()
        if save == "yes":
            export_to_txt(summary, qa_log)
            export_to_pdf(summary, qa_log)
            print("✅ Exported as `qa_summary.txt` and `qa_summary.pdf`")

        break

    query_embedding = get_embeddings([question])[0]
    top_chunks = retriever.search(query_embedding)
    answer = chat_with_memory(top_chunks, question)

    print("\n🤖 Answer:\n", answer)
    
    qa_log.append({"question": question, "answer": answer})