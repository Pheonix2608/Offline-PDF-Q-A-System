import gradio as gr
import os
import tempfile
import json
import traceback
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

# Global variables
retriever = None
summaries = {}
all_chunks = []
session_id = "default_session"

def process_docs(files):
    """Process uploaded documents and prepare for Q&A"""
    global retriever, summaries, all_chunks
    
    if not files:
        return "⚠️ No files uploaded.", []
    
    try:
        file_paths = [f.name for f in files]
        print(f"Processing files: {file_paths}")
        
        # Extract text from all files
        texts_by_file = extract_text_from_multiple_files(file_paths)
        
        if not texts_by_file:
            return "⚠️ No text extracted from files.", []
        
        all_chunks = []
        summaries = {}
        
        # Process each file
        for file_path, text in texts_by_file.items():
            if text.strip():  # Only process non-empty files
                print(f"Summarizing: {os.path.basename(file_path)}")
                summaries[file_path] = summarize_text(text)
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
        
        if not all_chunks:
            return "⚠️ No content found in uploaded files.", []
        
        # Create embeddings and retriever
        print(f"Creating embeddings for {len(all_chunks)} chunks...")
        embeddings = get_embeddings(all_chunks)
        retriever = FAISSRetriever(dim=len(embeddings[0]))
        retriever.add(embeddings, all_chunks)
        
        # Generate summary output
        summary_output = "\n\n".join([
            f"📄 {os.path.basename(f)}:\n{summaries[f]}" 
            for f in summaries
        ])
        
        # Generate previews for PDF files
        previews = []
        for file_path in file_paths:
            if file_path.lower().endswith(".pdf"):
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        images = convert_from_path(
                            file_path, 
                            first_page=1, 
                            last_page=1, 
                            fmt="jpeg", 
                            output_folder=temp_dir
                        )
                        if images:
                            previews.append(images[0])
                except Exception as e:
                    print(f"Preview generation failed for {file_path}: {e}")
        
        print(f"✅ Processed {len(file_paths)} files successfully")
        return summary_output, previews
        
    except Exception as e:
        error_msg = f"❌ Error processing files: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg, []

def chat(user_input, chat_history):
    """Handle chat interaction with streaming response"""
    if not user_input.strip():
        yield "", chat_history, "", "", ""
        return
    
    if not retriever:
        chat_history.append({
            "role": "assistant", 
            "content": "⚠️ Please upload and process documents first."
        })
        yield "", chat_history, "", "", ""
        return
    
    try:
        # Add user message
        chat_history.append({"role": "user", "content": user_input})
        
        # Add thinking message
        chat_history.append({"role": "assistant", "content": "🤔 Thinking..."})
        yield "", chat_history, "", "", ""
        
        # Get relevant chunks
        query_embedding = get_embeddings([user_input])[0]
        top_chunks = retriever.search(query_embedding)
        
        # Generate answer
        answer = generate_answer_with_memory(top_chunks, user_input, session_id=session_id)
        
        # Generate follow-up questions
        followups = generate_followups(user_input, answer)
        
        # Format context
        context = "\n---\n".join(top_chunks)
        
        # Format suggestions
        suggestions = "\n\n💡 *Follow-up Suggestions:*"
        for i, q in enumerate(followups[:3]):
            if q.strip():
                suggestions += f"\n`#{i+1}` {q}"
        
        # Create full response
        full_response = f"{answer}\n\n🔍 *Context used:*\n{context}{suggestions}"
        
        # Replace thinking message with actual response
        chat_history = chat_history[:-1]
        chat_history.append({"role": "assistant", "content": full_response})
        
        # Extract follow-up buttons
        btn_1 = followups[0] if len(followups) > 0 and followups[0].strip() else ""
        btn_2 = followups[1] if len(followups) > 1 and followups[1].strip() else ""
        btn_3 = followups[2] if len(followups) > 2 and followups[2].strip() else ""
        
        yield "", chat_history, btn_1, btn_2, btn_3
        
    except Exception as e:
        error_msg = f"❌ Error generating response: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        # Remove thinking message and add error
        if chat_history and chat_history[-1]["content"] == "🤔 Thinking...":
            chat_history = chat_history[:-1]
        chat_history.append({"role": "assistant", "content": error_msg})
        
        yield "", chat_history, "", "", ""

def clear_chat():
    """Clear chat history and memory"""
    try:
        reset_memory(session_id=session_id)
        return [], "✅ Memory cleared successfully."
    except Exception as e:
        return [], f"❌ Error clearing memory: {str(e)}"

def export_qa_log(summary, chat_log):
    """Export Q&A session as text file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as f:
            f.write("📄 Document Summary:\n")
            f.write(summary + "\n\n")
            f.write("🤖 Q&A Session:\n")
            f.write("=" * 50 + "\n\n")
            
            for msg in chat_log:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    f.write(f"👤 User: {content}\n\n")
                elif role == "assistant":
                    f.write(f"🤖 Assistant: {content}\n\n")
                    f.write("-" * 30 + "\n\n")
        
        return f.name
    except Exception as e:
        print(f"Export error: {e}")
        return None

def export_qa_json(summary, chat_log):
    """Export Q&A session as JSON file"""
    try:
        data = {
            "summary": summary,
            "chat_history": chat_log,
            "timestamp": str(pd.Timestamp.now()),
            "session_id": session_id
        }
        
        path = os.path.join(tempfile.gettempdir(), "qa_session.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return path
    except Exception as e:
        print(f"JSON export error: {e}")
        return None

def download_log(summary, chat_log):
    """Prepare text log for download"""
    file_path = export_qa_log(summary, chat_log)
    if file_path:
        return gr.File.update(value=file_path, visible=True)
    return gr.File.update(visible=False)

def download_json(summary, chat_log):
    """Prepare JSON log for download"""
    file_path = export_qa_json(summary, chat_log)
    if file_path:
        return gr.File.update(value=file_path, visible=True)
    return gr.File.update(visible=False)

def use_suggestion(suggestion):
    """Use follow-up suggestion as input"""
    return suggestion

# ---------------- Gradio UI ----------------

with gr.Blocks(title="AI Document Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 Offline AI Document Assistant
    
    Upload documents (PDF, DOCX, TXT, CSV) and ask questions about their content.
    Powered by local LLMs via Ollama - completely offline and private.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            doc_input = gr.File(
                file_types=[".pdf", ".docx", ".txt", ".csv"], 
                file_count="multiple", 
                label="📁 Upload Documents"
            )
            load_btn = gr.Button("🔄 Process Documents", variant="primary")
            
        with gr.Column(scale=2):
            summary_box = gr.Textbox(
                label="📄 Document Summaries", 
                lines=15,
                placeholder="Document summaries will appear here after processing..."
            )
    
    with gr.Row():
        preview_gallery = gr.Gallery(
            label="📷 PDF Previews", 
            columns=3, 
            height="300px",
            object_fit="contain"
        )
    
    with gr.Row():
        chatbot = gr.Chatbot(
            label="💬 Chat with Your Documents", 
            type="messages",
            height=400
        )
    
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Ask something about your documents...",
            label="Your Question",
            scale=4
        )
        send_btn = gr.Button("📤 Send", variant="primary")
        clear_btn = gr.Button("🗑️ Clear Chat")
    
    with gr.Row():
        followup_1 = gr.Button("", visible=False, scale=1)
        followup_2 = gr.Button("", visible=False, scale=1)
        followup_3 = gr.Button("", visible=False, scale=1)
    
    with gr.Row():
        download_btn = gr.Button("💾 Export as Text")
        download_file = gr.File(label="📄 Download Q&A Log", visible=False)
        
        download_json_btn = gr.Button("💾 Export as JSON")
        download_json_file = gr.File(label="📄 Download JSON", visible=False)
    
    # Event handlers
    load_btn.click(
        fn=process_docs, 
        inputs=[doc_input], 
        outputs=[summary_box, preview_gallery]
    )
    
    send_btn.click(
        fn=chat,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot, followup_1, followup_2, followup_3],
    )
    
    user_input.submit(
        fn=chat,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot, followup_1, followup_2, followup_3],
    )
    
    clear_btn.click(
        fn=clear_chat, 
        outputs=[chatbot, summary_box]
    )
    
    # Follow-up button handlers
    followup_1.click(
        fn=use_suggestion,
        inputs=[followup_1],
        outputs=[user_input]
    )
    followup_2.click(
        fn=use_suggestion,
        inputs=[followup_2],
        outputs=[user_input]
    )
    followup_3.click(
        fn=use_suggestion,
        inputs=[followup_3],
        outputs=[user_input]
    )
    
    # Export handlers
    download_btn.click(
        fn=download_log, 
        inputs=[summary_box, chatbot], 
        outputs=[download_file]
    )
    download_json_btn.click(
        fn=download_json, 
        inputs=[summary_box, chatbot], 
        outputs=[download_json_file]
    )

if __name__ == "__main__":
    print("🚀 Starting AI Document Assistant...")
    print("📝 Make sure Ollama is running with a model (e.g., 'ollama run mistral')")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )