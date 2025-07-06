# from langchain_ollama import OllamaLLM
# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnableSequence
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain.memory import ConversationBufferMemory

# # 🧠 Load Local Model (Ollama must be running!)
# llm = OllamaLLM(model="mistral")  # llama2, phi3, gemma etc. also work

# # 📋 Prompt Template
# contextual_prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are a helpful assistant answering based on the following document context.
# Only respond based on the context. If unsure, say "I don't know".

# Context:
# {context}

# Question:
# {question}

# Answer:"""
# )

# # 🧠 Summary Prompt
# summary_prompt = PromptTemplate(
#     input_variables=["text"],
#     template="""
# You are an AI assistant. Summarize the following document clearly and concisely in bullet points.

# Document:
# {text}

# Summary:"""
# )

# # ✅ Summary Chain
# summarize_chain = summary_prompt | llm

# def summarize_text(text, max_chars=2000):
#     short_text = text[:max_chars]  # keep inference fast
#     return summarize_chain.invoke({"text": short_text})


# # 🧠 No-Memory Q&A Chain
# qa_chain = contextual_prompt | llm

# def generate_answer(context_chunks, question):
#     context = "\n".join(context_chunks)
#     return qa_chain.invoke({"context": context, "question": question})


# # 🧠 Memory Support
# memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)

# # 🔁 Memory-Enabled Chain
# qa_chain_with_memory = RunnableWithMessageHistory(
#     qa_chain,
#     memory,
#     input_messages_key="question",
#     history_messages_key="chat_history"
# )

# def chat_with_memory(context_chunks, question):
#     context = "\n".join(context_chunks)
#     return qa_chain_with_memory.invoke(
#         {"context": context, "question": question},
#         config={"configurable": {"session_id": "pdf_session"}}
#     )

# def generate_answer_with_memory(context_chunks, question):
#     return chat_with_memory(context_chunks, question)

# def reset_memory():
#     memory.clear()


from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory

# Load model
llm = OllamaLLM(model="mistral")

# Prompt
contextual_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant answering based on the following document context.
Only respond based on the context. If unsure, say "I don't know".

Context:
{context}

Question:
{question}

Answer:"""
)

qa_chain = contextual_prompt | llm

# Session-based memory cache
memory_sessions = {}

def get_memory(session_id: str):
    if session_id not in memory_sessions:
        memory_sessions[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            return_messages=True,
        )
    return memory_sessions[session_id]

qa_chain_with_memory = RunnableWithMessageHistory(
    qa_chain,
    lambda session_id: get_memory(session_id),
    input_messages_key="question",
    history_messages_key="chat_history"
)

def generate_answer_with_memory(context_chunks, question, session_id="default"):
    context = "\n".join(context_chunks)
    return qa_chain_with_memory.invoke(
        {"context": context, "question": question},
        config={"configurable": {"session_id": session_id}}
    )

def summarize_text(text, max_chars=2000):
    short_text = text[:max_chars]
    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
Summarize the following document clearly and concisely in bullet points.

Document:
{text}

Summary:
"""
    )
    return (summary_prompt | llm).invoke({"text": short_text})

def reset_memory(session_id="default"):
    if session_id in memory_sessions:
        memory_sessions[session_id].clear()


# 🔮 Generate follow-up suggestions
followup_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="""
Based on the following Q&A, suggest 2–3 helpful follow-up questions the user might ask.

Question: {question}
Answer: {answer}

Follow-up Suggestions:
"""
)

followup_chain = followup_prompt | llm

def generate_followups(question, answer):
    result = followup_chain.invoke({"question": question, "answer": answer})
    return [line.strip("-• \n") for line in result.strip().splitlines() if line.strip()]
