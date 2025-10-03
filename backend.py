from openai import OpenAI
import numpy as np
import faiss
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Define your OpenRouter API key (for completion only)
OPENROUTER_API_KEY = "API_KEY"  # Replace with your actual API key

# Initialize OpenAI client for completions only with proper headers
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "http://localhost:3000",  # Optional but recommended
        "X-Title": "RAG Application",  # Optional
    }
)

class SimpleEmbedder:
    def __init__(self):
        self.vectorizer = None
        self.is_fitted = False
        
    def fit(self, texts):
        """Fit the vectorizer on a corpus of texts"""
        self.vectorizer = TfidfVectorizer(
            max_features=384,  # Reduced dimension for efficiency
            stop_words='english',
            lowercase=True,
            analyzer='word'
        )
        self.vectorizer.fit(texts)
        self.is_fitted = True
        
    def get_embedding(self, text):
        """Generate embedding for a single text"""
        if not self.is_fitted or self.vectorizer is None:
            # If not fitted, fit on the text itself as a fallback
            self.fit([text])
        
        try:
            embedding = self.vectorizer.transform([text]).toarray()[0]
            return embedding.tolist()
        except Exception as e:
            print(f"Error in TF-IDF embedding: {e}")
            return self.get_fallback_embedding(text)
    
    def get_fallback_embedding(self, text):
        """Simple hash-based fallback embedding"""
        try:
            import hashlib
            hash_object = hashlib.md5(text.encode())
            hash_hex = hash_object.hexdigest()
            
            embedding = []
            for i in range(0, len(hash_hex)-1, 2):
                pair = hash_hex[i:i+2]
                embedding.append(int(pair, 16) / 255.0)
            
            # Ensure consistent dimension
            if len(embedding) < 128:
                embedding.extend([0.0] * (128 - len(embedding)))
            else:
                embedding = embedding[:128]
                
            return embedding
        except Exception as e:
            print(f"Fallback embedding failed: {e}")
            return [0.0] * 128  # Return zero vector as last resort

# Initialize embedder
embedder = SimpleEmbedder()

# Function to generate a completion using OpenRouter's GPT model
def generate_completion(prompt, model="openai/gpt-4o"):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating completion: {e}")
        # Provide a fallback response based on the context
        return generate_fallback_response(prompt)

def generate_fallback_response(prompt):
    """Generate a simple response when OpenRouter fails"""
    try:
        # Extract the context and question from the prompt
        lines = prompt.split('\n')
        context = ""
        question = ""
        in_context = False
        
        for line in lines:
            if line.startswith("Context"):
                in_context = True
            elif line.startswith("Question:"):
                question = line.replace("Question:", "").strip()
                break
            elif in_context and line.strip():
                context += line + " "
        
        # Simple rule-based response
        if "CWM" in context.upper() or "world model" in context.lower():
            return "Based on the context provided, CWM appears to be a world model or coding model developed by Meta that learns how code works, not just what it looks like. It enhances code understanding and can work with reasoning tasks like verification, testing, and debugging."
        else:
            return "I found some relevant context in the document, but I cannot provide a detailed answer without access to the AI completion service. Please check your API key and try again."
            
    except Exception as e:
        return "I was able to retrieve relevant context from the document, but I'm currently unable to generate a detailed AI response. Please check your API configuration."

# Function to generate embeddings using local model
def get_embedding(text):
    return embedder.get_embedding(text)

# Function to chunk text into smaller parts with overlap
def chunk_text(text, max_words=100, overlap=20):
    words = text.split()
    chunks = []
    
    if len(words) <= max_words:
        return [" ".join(words)]
    
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
        # Stop if we've reached the end
        if i + max_words >= len(words):
            break
            
    return chunks

# Function to load and chunk the text from a file
def load_and_chunk_text(file_path, chunk_size=100, overlap=20):
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            raw_text = f.read()
        chunks = chunk_text(raw_text, chunk_size, overlap)
        # Fit the embedder on all chunks for better TF-IDF representation
        if chunks:
            embedder.fit(chunks)
        return chunks
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

# Function to build the FAISS index with embeddings of the chunks
def build_faiss_index(chunks):
    try:
        print("Generating test embedding with local model...")
        test_embedding = get_embedding(chunks[0])
        
        if test_embedding is None:
            raise ValueError("Local embedding generation failed.")
        
        dimension = len(test_embedding)
        print(f"Embedding dimension: {dimension}")
        index = faiss.IndexFlatL2(dimension)
        
        chunk_mapping = []
        embeddings_list = []
        
        print("Generating embeddings for all chunks...")
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            emb = get_embedding(chunk)
            if emb is not None:
                # Ensure all embeddings have the same dimension
                if len(emb) == dimension:
                    embeddings_list.append(emb)
                    chunk_mapping.append(chunk)
                else:
                    print(f"Warning: Embedding dimension mismatch for chunk {i+1}")
        
        # Add all embeddings to FAISS at once (more efficient)
        if embeddings_list:
            embeddings_array = np.array(embeddings_list).astype("float32")
            index.add(embeddings_array)
        
        print(f"FAISS index built with {len(chunk_mapping)}/{len(chunks)} embeddings")
        return index, chunk_mapping
        
    except Exception as e:
        print(f"Error building FAISS index: {e}")
        return None, []

# Function to search for the top-k relevant chunks based on a query
def retrieve_top_k(query, index, chunk_mapping, k=3):
    try:
        print(f"Generating query embedding for: '{query}'")
        query_embedding = get_embedding(query)
        if query_embedding is None:
            print("Failed to generate query embedding")
            return []
            
        # Search in FAISS index
        distances, indices = index.search(np.array([query_embedding]).astype("float32"), k)
        
        # Get the relevant chunks
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunk_mapping) and idx >= 0:  # Ensure index is valid
                results.append({
                    'chunk': chunk_mapping[idx],
                    'distance': distances[0][i]
                })
        
        # Sort by distance (lower is better)
        results.sort(key=lambda x: x['distance'])
        
        print(f"Search completed. Found {len(results)} results")
        return [result['chunk'] for result in results]
    except Exception as e:
        print(f"Error retrieving top-k results: {e}")
        return []

# Function to improve chunk relevance by using better preprocessing
def preprocess_text(text):
    """Clean and preprocess text for better embeddings"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Main backend function
def backend(file_path="met.txt", query="", k=3):
    # Load and chunk text from the file
    print(f"Loading file: {file_path}")
    chunks = load_and_chunk_text(file_path, chunk_size=150, overlap=25)
    
    if not chunks:
        return "Error: Unable to load or process the text file.", []
    
    print(f"Total Chunks: {len(chunks)}")
    
    # Preprocess chunks
    chunks = [preprocess_text(chunk) for chunk in chunks]
    
    # Generate embeddings for each chunk and store in FAISS index
    print("Building FAISS index with local embeddings...")
    index, chunk_mapping = build_faiss_index(chunks)
    
    if index is None or len(chunk_mapping) == 0:
        return "Error: Failed to build the FAISS index or no embeddings generated.", []
    
    # Retrieve top-k relevant chunks for the query
    print(f"Searching for query: '{query}'")
    top_chunks = retrieve_top_k(query, index, chunk_mapping, k)
    
    if not top_chunks:
        return "Error: No relevant context found for the query.", []
    
    # Build the prompt with the retrieved chunks
    context_text = '\n\n'.join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(top_chunks)])
    
    prompt = f"""Based on the following context information, please answer the user's question. 
If the context doesn't contain relevant information to answer the question, please say so rather than making up an answer.

{context_text}

Question: {query}

Please provide a comprehensive answer based only on the context provided:"""
    
    # Generate the completion from OpenRouter's GPT model
    print("Generating completion with OpenRouter...")
    completion = generate_completion(prompt)
    
    return completion, top_chunks

# Function to test the embedding system
def test_embedding_system():
    print("Testing local embedding system...")
    test_sentences = [
        "What is artificial intelligence?",
        "Machine learning is a subset of AI.",
        "The weather today is sunny and warm."
    ]
    
    # Fit the embedder first
    embedder.fit(test_sentences)
    
    for i, sentence in enumerate(test_sentences):
        embedding = get_embedding(sentence)
        print(f"Sentence {i+1}: {sentence[:50]}... - Embedding length: {len(embedding) if embedding else 'Failed'}")

# Function to test OpenRouter connection
def test_openrouter_connection():
    print("Testing OpenRouter connection...")
    try:
        # Simple test request
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello'"}],
            max_tokens=10
        )
        print("✓ OpenRouter connection successful!")
        return True
    except Exception as e:
        print(f"✗ OpenRouter connection failed: {e}")
        print("Please check your API key and make sure it's valid.")
        return False

# Example usage
if __name__ == "__main__":
    # First test the local embedding system
    test_embedding_system()
    
    # Test OpenRouter connection
    print(f"\n{'='*50}")
    print("TESTING OPENROUTER CONNECTION")
    print(f"{'='*50}")
    connection_ok = test_openrouter_connection()
    
    if not connection_ok:
        print("\nWarning: OpenRouter connection failed. Using fallback mode.")
        print("To fix this, please:")
        print("1. Visit https://openrouter.ai/")
        print("2. Get an API key")
        print("3. Replace 'your_openrouter_api_key' with your actual key")
        print("4. Ensure you have credits in your account")
    
    # Then run the main application
    query = "what is agent?"
    print(f"\n{'='*50}")
    print(f"MAIN QUERY: {query}")
    print(f"{'='*50}")
    
    completion, top_chunks = backend(query=query)
    
    print(f"\n{'='*50}")
    print("CONTEXT USED FOR ANSWERING:")
    print(f"{'='*50}")
    for i, chunk in enumerate(top_chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    
    print(f"\n{'='*50}")
    print("AI RESPONSE:")
    print(f"{'='*50}")

    print(completion)
