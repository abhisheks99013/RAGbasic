import os
from dotenv import load_dotenv
import chromadb
import google.generativeai as genai

# Custom Gemini embedding function for ChromaDB with the expected signature
class GeminiEmbeddingFunction:
    def __init__(self, api_key, model_name="models/text-embedding-004", dimensions=768):
        self.api_key = api_key
        self.model_name = model_name
        self.dimensions = dimensions
        # Configure Gemini API
        genai.configure(api_key=api_key)

    def __call__(self, input):
        # Handle if input is a list of strings
        if isinstance(input, list):
            return [self.__call__(x) for x in input]
        
        print("==== Generating embeddings with Gemini... ====")
        # Ensure input is a string
        text_str = str(input)
        response = genai.embed_content(
            model=self.model_name,
            content={"parts": [{"text": text_str}]}
        )
        embedding = response["embedding"]
        # Ensure the embedding has the correct dimension (768)
        if len(embedding) != self.dimensions:
            print(f"Adjusting embedding size: got {len(embedding)} dims, expected {self.dimensions}.")
            if len(embedding) < self.dimensions:
                embedding = embedding + [0] * (self.dimensions - len(embedding))
            else:
                embedding = embedding[:self.dimensions]
        return embedding

# Load environment variables
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

# Use custom Gemini embedding method with dimensions set to 768
gemini_ef = GeminiEmbeddingFunction(api_key=gemini_key, model_name="models/text-embedding-004", dimensions=768)

chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=gemini_ef
)

# (Re)configure Gemini API (already configured in our custom class, but included for clarity)
genai.configure(api_key=gemini_key)

def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

directory_path = "./articles"
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents")

chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

print(f"Split documents into {len(chunked_documents)} chunks")

# A standalone Gemini embedding function (optional, for manual embedding generation)
def get_gemini_embedding(text):
    print("==== Generating embeddings... ====")
    response = genai.embed_content(
        model="models/text-embedding-004",
        content={"parts": [{"text": str(text)}]}
    )
    embedding = response["embedding"]
    # Adjust to 768 dims if needed
    if len(embedding) != 768:
        print(f"Adjusting standalone embedding size: got {len(embedding)} dims, expected 768.")
        if len(embedding) < 768:
            embedding = embedding + [0] * (768 - len(embedding))
        else:
            embedding = embedding[:768]
    return embedding

for doc in chunked_documents:
    print("==== Generating embeddings for document chunks... ====")
    doc["embedding"] = get_gemini_embedding(doc["text"])
    
for doc in chunked_documents:
    print("==== Inserting chunks into db... ====")
    collection.upsert(
        ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]]
    )
    
def query_documents(question, n_results=2):
    results = collection.query(query_texts=question, n_results=n_results)
    # Extract the relevant chunks
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks

def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    answer = response.text  # or adjust according to your response object's attribute
    return answer

question = "tell me about hardware products"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)
print(answer)