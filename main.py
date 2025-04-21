from flask import Flask, request, jsonify
import cohere
import json
import numpy as np

# Initialize Cohere and Flask
co = cohere.Client("FhOJJb3c6jOhFjfFueVbenEYxRxFqkr8utVdhBn9")  # REPLACE with your key
app = Flask(__name__)

# Load your precomputed vectors
with open("vectors.json") as f:
    data = json.load(f)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route("/ask", methods=["POST"])
def ask():
    query = request.json["question"]
    q_vector = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    ).embeddings[0]

    # Find top 3 closest chunks
    top_chunks = sorted(data, key=lambda d: -cosine_similarity(q_vector, d["embedding"]))[:3]
    context = "\n\n".join(chunk["text"] for chunk in top_chunks)

    # Final AI prompt
    prompt = f"Use the context below to answer the question:\n{context}\n\nQuestion: {query}"

    response = co.generate(
        prompt=prompt,
        model="command-r-plus",
        max_tokens=400,
        temperature=0.3
    )

    return jsonify({"answer": response.generations[0].text.strip()})

# 🔥 THIS PART IS MISSING IN YOUR CODE 🔥
if __name__ == "__main__":
    app.run(debug=True)
