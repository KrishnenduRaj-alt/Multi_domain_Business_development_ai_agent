import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# 1. Load embeddings and FAISS DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("medical_store_faiss", embeddings, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 2. Setup web search tool
web_search = DuckDuckGoSearchRun()

# 3. Connect to Groq LLM (replace with your Groq API key)
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0)

# 4. Create a prompt for RAG
prompt_template = """
You are a helpful AI assistant for a medical store.
If the question is about the store, use the provided store data to answer.
If you can't find the answer in the store data, search the web and answer.

Question: {question}
Context from store data: {context}
Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["question", "context"])

# 5. Function to answer questions
def answer_question(query):
    # Search store data
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    if context.strip():
        # Use store data
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT}
        )
        return chain.run(query)
    else:
        # Use web search
        web_result = web_search.run(query)
        return web_result

# 6. Chat loop
if __name__ == "__main__":
    print("🤖 Business Assistant AI Chatbot (type 'exit' to quit)")
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break
        answer = answer_question(question)
        print("AI:", answer)
