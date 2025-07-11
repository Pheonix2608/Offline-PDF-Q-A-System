from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import traceback

# Initialize LLM with error handling
try:
    llm = OllamaLLM(model="mistral")
    print("✅ Connected to Ollama with mistral model")
except Exception as e:
    print(f"❌ Failed to connect to Ollama: {e}")
    print("Make sure Ollama is running and 'mistral' model is available")
    print("Run: ollama run mistral")
    raise

# Improved prompts
contextual_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful AI assistant analyzing documents. Use the provided context to answer questions accurately.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer based only on the provided context
- If the context doesn't contain relevant information, say "I don't have enough information in the documents to answer this question"
- Be concise but thorough
- Use specific details from the context when possible

Answer:"""
)

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""Summarize the following document content in a clear, structured way:

Document Content:
{text}

Create a summary that includes:
- Main topics covered
- Key points and findings
- Important details

Summary:"""
)

followup_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="""Based on this Q&A exchange, suggest 3 relevant follow-up questions that would help the user explore the topic further.

Question: {question}
Answer: {answer}

Generate 3 specific, actionable follow-up questions that:
1. Dive deeper into the topic
2. Explore related aspects
3. Clarify or expand on the answer

Follow-up Questions:
1."""
)

# Create chains
qa_chain = contextual_prompt | llm
summary_chain = summary_prompt | llm
followup_chain = followup_prompt | llm

# Session-based memory management using ChatMessageHistory
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Memory-enabled chain
qa_chain_with_memory = RunnableWithMessageHistory(
    qa_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

def generate_answer_with_memory(context_chunks, question, session_id="default"):
    """Generate answer using context and conversation memory"""
    try:
        if not context_chunks:
            return "No relevant context found in the documents."
        
        context = "\n\n".join(context_chunks)
        
        # Limit context length to avoid token limits
        if len(context) > 4000:
            context = context[:4000] + "..."
        
        response = qa_chain_with_memory.invoke(
            {"context": context, "question": question},
            config={"configurable": {"session_id": session_id}}
        )
        
        return response.strip()
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        traceback.print_exc()
        return f"Error generating response: {str(e)}"

def generate_answer(context_chunks, question):
    """Generate answer without memory (stateless)"""
    try:
        if not context_chunks:
            return "No relevant context found in the documents."
        
        context = "\n\n".join(context_chunks)
        
        # Limit context length
        if len(context) > 4000:
            context = context[:4000] + "..."
        
        response = qa_chain.invoke({"context": context, "question": question})
        return response.strip()
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Error generating response: {str(e)}"

def summarize_text(text, max_chars=3000):
    """Summarize document text"""
    try:
        if not text or not text.strip():
            return "No content to summarize."
        
        # Limit text length for faster processing
        short_text = text[:max_chars]
        if len(text) > max_chars:
            short_text += "..."
        
        summary = summary_chain.invoke({"text": short_text})
        return summary.strip()
        
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return f"Error creating summary: {str(e)}"

def generate_followups(question, answer):
    """Generate follow-up question suggestions"""
    try:
        if not question or not answer:
            return []
        
        # Limit input length
        short_answer = answer[:1000] if len(answer) > 1000 else answer
        
        response = followup_chain.invoke({
            "question": question,
            "answer": short_answer
        })
        
        # Parse response into individual questions
        lines = response.strip().split('\n')
        questions = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Follow-up'):
                # Remove numbering and clean up
                clean_line = line.replace('1.', '').replace('2.', '').replace('3.', '')
                clean_line = clean_line.replace('-', '').strip()
                if clean_line and len(clean_line) > 10:
                    questions.append(clean_line)
        
        return questions[:3]  # Return max 3 questions
        
    except Exception as e:
        print(f"Error generating follow-ups: {e}")
        return []

def reset_memory(session_id="default"):
    """Clear memory for a specific session"""
    try:
        if session_id in store:
            store[session_id].clear()
            print(f"✅ Memory cleared for session: {session_id}")
        else:
            print(f"No memory found for session: {session_id}")
    except Exception as e:
        print(f"Error clearing memory: {e}")

def list_sessions():
    """List all active memory sessions"""
    return list(store.keys())

# Test connection on import
def test_ollama_connection():
    """Test if Ollama is responding"""
    try:
        response = llm.invoke("Hello, are you working?")
        print(f"✅ Ollama test successful: {response[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False

# Test connection when module is imported
if __name__ == "__main__":
    print("Testing Ollama connection...")
    test_ollama_connection()
else:
    # Quick test on import
    try:
        llm.invoke("test")
        print("✅ Ollama connection verified")
    except Exception as e:
        print(f"⚠️ Ollama connection issue: {e}")
        print("Make sure Ollama is running: ollama serve")
        print("And model is available: ollama run mistral")