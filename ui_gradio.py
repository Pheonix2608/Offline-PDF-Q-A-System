# import gradio as gr
# import os
# import tempfile
# from pdf2image import convert_from_path
# from pdf_parser import extract_text_from_multiple_pdfs, chunk_text

# from embedder import get_embeddings
# from retriever import FAISSRetriever
# from qa_engine import generate_answer_with_memory, reset_memory, summarize_text

# retriever = None
# summaries = {}
# all_chunks = []

# def process_pdfs(files):
#     global retriever, summaries, all_chunks
#     file_paths = [f.name for f in files]
    
#     texts_by_file = extract_text_from_multiple_pdfs(file_paths)
#     all_chunks = []
#     summaries = {}

#     for file, text in texts_by_file.items():
#         summaries[file] = summarize_text(text)
#         chunks = chunk_text(text)
#         all_chunks.extend(chunks)

#     embeddings = get_embeddings(all_chunks)
#     retriever = FAISSRetriever(dim=len(embeddings[0]))
#     retriever.add(embeddings, all_chunks)

#     summary_output = "\n\n".join([f"📄 {os.path.basename(f)}:\n{summaries[f]}" for f in summaries])

#     previews = []
#     for file_path in file_paths:
#         with tempfile.TemporaryDirectory() as temp_dir:
#             images = convert_from_path(file_path, first_page=1, last_page=1, fmt="jpeg", output_folder=temp_dir)
#             if images:
#                 previews.append(images[0])

#     return summary_output, previews

# def export_qa_log(summary, chat_log):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as f:
#         f.write("📄 Summary:\n" + summary + "\n\n")
#         f.write("🤖 Q&A Log:\n")
#         for msg in chat_log:
#             if msg["role"] == "user":
#                 f.write(f"Q: {msg['content']}\n")
#             elif msg["role"] == "assistant":
#                 f.write(f"A: {msg['content']}\n\n")
#         return f.name

# def chat(user_input, chat_history):
#     if not retriever:
#         return "⚠️ Please upload and process PDFs first.", chat_history
    
#     query_embedding = get_embeddings([user_input])[0]
#     top_chunks = retriever.search(query_embedding)
#     answer = generate_answer_with_memory(top_chunks, user_input)
#     if not answer:
#         answer = "🤖 No relevant information found in the documents."

#     chunk_context = "\n---\n".join(top_chunks)
#     chat_history.append({"role": "user", "content": user_input})
#     chat_history.append({"role": "assistant", "content": f"{answer}\n\n🔍 *Context used:*\n{chunk_context}"})

#     return "", chat_history

# def clear_chat():
#     reset_memory()
#     return [], "Memory cleared."

# def download_log(summary, chat_log):
#     path = export_qa_log(summary, chat_log)
#     return gr.File.update(value=path, visible=True)

# # ---------------- Gradio App ----------------

# with gr.Blocks() as demo:
#     gr.Markdown("# 🧠 Offline AI PDF Assistant")

#     with gr.Row():
#         pdf_input = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDFs")
#         summary_box = gr.Textbox(label="📄 Document Summaries", lines=20)
#         load_btn = gr.Button("Process PDFs")

#     with gr.Row():
#         preview_gallery = gr.Gallery(label="📷 PDF Previews", columns=2, height="auto")

#     with gr.Row():
#         chatbot = gr.Chatbot(label="Chat", type="messages")

#     with gr.Row():
#         user_input = gr.Textbox(placeholder="Ask something about the uploaded documents…")
#         send_btn = gr.Button("Ask")
#         clear_btn = gr.Button("Reset Memory")

#     with gr.Row():
#         download_btn = gr.Button("⬇️ Download Q&A Log")
#         download_file = gr.File(label="Download Q&A Session", visible=False)

#     load_btn.click(fn=process_pdfs, inputs=[pdf_input], outputs=[summary_box, preview_gallery])
#     send_btn.click(fn=chat, inputs=[user_input, chatbot], outputs=[user_input, chatbot])
#     clear_btn.click(fn=clear_chat, outputs=[chatbot, summary_box])
#     download_btn.click(fn=download_log, inputs=[summary_box, chatbot], outputs=[download_file])

# demo.launch()

import gradio as gr
import os
import tempfile
import json
from pdf2image import convert_from_path
from pdf_parser import extract_text_from_multiple_files, chunk_text
from embedder import get_embeddings
from retriever import FAISSRetriever
from qa_engine import (
    generate_answer_with_memory,
    reset_memory,
    summarize_text,
    generate_followups
)

retriever = None
summaries = {}
all_chunks = []
session_id = "default_session"

def process_docs(files):
    global retriever, summaries, all_chunks
    file_paths = [f.name for f in files]

    texts_by_file = extract_text_from_multiple_files(file_paths)
    all_chunks = []
    summaries = {}

    for file, text in texts_by_file.items():
        summaries[file] = summarize_text(text)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    embeddings = get_embeddings(all_chunks)
    retriever = FAISSRetriever(dim=len(embeddings[0]))
    retriever.add(embeddings, all_chunks)

    summary_output = "\n\n".join(
        [f"📄 {os.path.basename(f)}:\n{summaries[f]}" for f in summaries]
    )

    previews = []
    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            with tempfile.TemporaryDirectory() as temp_dir:
                images = convert_from_path(file_path, first_page=1, last_page=1, fmt="jpeg", output_folder=temp_dir)
                if images:
                    previews.append(images[0])

    return summary_output, previews

def chat(user_input, chat_history):
    if not retriever:
        yield "⚠️ Please upload and process files first.", chat_history, "", "", ""

    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": "🤔 Thinking..."})
    yield "", chat_history, "", "", ""

    query_embedding = get_embeddings([user_input])[0]
    top_chunks = retriever.search(query_embedding)

    answer = generate_answer_with_memory(top_chunks, user_input, session_id=session_id)
    followups = generate_followups(user_input, answer)
    context = "\n---\n".join(top_chunks)

    suggestions = "\n\n💡 *Follow-up Suggestions:*"
    for i, q in enumerate(followups[:3]):
        suggestions += f"\n`#{i+1}` {q}"

    full_response = f"{answer}\n\n🔍 *Context used:*\n{context}{suggestions}"

    chat_history = chat_history[:-1]
    chat_history.append({"role": "assistant", "content": full_response})

    btn_1 = followups[0] if len(followups) > 0 else ""
    btn_2 = followups[1] if len(followups) > 1 else ""
    btn_3 = followups[2] if len(followups) > 2 else ""

    yield "", chat_history, btn_1, btn_2, btn_3

def clear_chat():
    reset_memory(session_id=session_id)
    return [], "Memory cleared."

def export_qa_log(summary, chat_log):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as f:
        f.write("📄 Summary:\n" + summary + "\n\n")
        f.write("🤖 Q&A Log:\n")
        for msg in chat_log:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                f.write(f"Q: {content}\n")
            elif role == "assistant":
                f.write(f"A: {content}\n\n")
        return f.name

def export_qa_json(summary, chat_log):
    data = {
        "summary": summary,
        "chat": chat_log
    }
    path = os.path.join(tempfile.gettempdir(), "qa_session.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path

def download_log(summary, chat_log):
    return gr.File.update(value=export_qa_log(summary, chat_log), visible=True)

def download_json(summary, chat_log):
    return gr.File.update(value=export_qa_json(summary, chat_log), visible=True)

# ---------------- Gradio UI ----------------

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Offline AI Document Assistant")

    with gr.Row():
        doc_input = gr.File(file_types=[".pdf", ".docx", ".txt", ".csv"], file_count="multiple", label="Upload Documents")
        summary_box = gr.Textbox(label="📄 Document Summaries", lines=20)
        load_btn = gr.Button("Process Documents")

    with gr.Row():
        preview_gallery = gr.Gallery(label="📷 Previews (PDF only)", columns=2, height="auto")

    with gr.Row():
        chatbot = gr.Chatbot(label="Chat", type="messages")

    with gr.Row():
        user_input = gr.Textbox(placeholder="Ask something about the uploaded documents…")
        send_btn = gr.Button("Ask")
        clear_btn = gr.Button("Reset Memory")

    with gr.Row():
        followup_1 = gr.Button(visible=False)
        followup_2 = gr.Button(visible=False)
        followup_3 = gr.Button(visible=False)

    with gr.Row():
        download_btn = gr.Button("⬇️ Export Q&A Log (.txt)")
        download_file = gr.File(label="Download Q&A", visible=False)

        download_json_btn = gr.Button("⬇️ Export as JSON")
        download_json_file = gr.File(label="Download .json", visible=False)

    load_btn.click(fn=process_docs, inputs=[doc_input], outputs=[summary_box, preview_gallery])

    send_btn.click(
        fn=chat,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot, followup_1, followup_2, followup_3],
    )

    clear_btn.click(fn=clear_chat, outputs=[chatbot, summary_box])

    followup_1.click(lambda q: (q,), inputs=[followup_1], outputs=[user_input])
    followup_2.click(lambda q: (q,), inputs=[followup_2], outputs=[user_input])
    followup_3.click(lambda q: (q,), inputs=[followup_3], outputs=[user_input])

    download_btn.click(fn=download_log, inputs=[summary_box, chatbot], outputs=[download_file])
    download_json_btn.click(fn=download_json, inputs=[summary_box, chatbot], outputs=[download_json_file])

demo.launch()
# demo.launch(share=True)  # Uncomment to enable public sharing