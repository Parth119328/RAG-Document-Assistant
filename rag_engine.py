import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from pypdf import PdfReader
import numpy as np
import json
from sentence_transformers import SentenceTransformer

import groq_implem

MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    local_files_only=True
)

def loadPdf(file_path):
    reader = PdfReader(file_path)
    extracted_text = ""

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            extracted_text += f"\n{text}"

    return extracted_text

def splitText(text, chunk_size = 500, overlap = 100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def generateEmbedding(chunks):
    embedding = MODEL.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding

def saveVectorStore(chunks, embeddings):
    os.makedirs("vector store", exist_ok=True)

    np.save("vector store/embedding.npy", embeddings)

    with open("vector store/chunks.json", "w", encoding='utf-8') as file:
        json.dump(chunks, file, ensure_ascii=False, indent=4)

    return "vector store saved..."

def loadVector():
    embeddings = np.load("vector store/embedding.npy")

    with open("vector store/chunks.json", "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks, embeddings

def retrieveChunks(question, chunks, embeddings, top_k=3):
    question_embedding = MODEL.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities = np.dot(embeddings, question_embedding)

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for i in top_indices:
        results.append({
            "chunk": chunks[i],
            "score": float(similarities[i]),
            "index": int(i)
        })

    return results

def ragPipeline(question):
    chunks, embeddings = loadVector()

    retrieved_chunks = retrieveChunks(
        question,
        chunks,
        embeddings,
        top_k=3
    )

    answer = groq_implem.generateAnswer(
        question,
        retrieved_chunks
    )

    return answer

def run_1():
    text = loadPdf("data/sample.pdf")
    chunks = splitText(text)

    print("Total characters:", len(text))
    print("Total chunks:", len(chunks))

    embeddings = generateEmbedding(chunks)

    print("Embedding shape:", embeddings.shape)

    print("Saving vector store...")

    saveVectorStore(chunks, embeddings)

    print("\nFirst embedding:")
    print(embeddings[0])

    print("\nRAG preprocessing completed!")

def run_2():
    chunks, embeddings = loadVector()

    print("Num of loaded chunks: ", len(chunks))
    print("Embeddings Shape: ", embeddings.shape)

    ques = input()

    results = retrieveChunks(
        ques,
        chunks,
        embeddings,
        top_k=3
    )

    for result in results:
        print("Chunk Index:", result["index"])
        print("Similarity Score:", round(result["score"], 4))
        print("Content:", result["chunk"])
        print("\n\n")

def run_3():
    question = input("Enter: ")

    answer = ragPipeline(question)

    print("\nGenerated Answer:")
    print(answer)

if __name__ == "__main__":
    """
    DEVELOPMENT RUNS
    run_1() IS COMPULSORY TO RUN IF YOU ARE
    TESTING THE CODE FOR THE FIRST TIME
    MODEL USED - 'all-MiniLM-L6-v2'
    - IT IS RESPONSIBLE TO GENERATE AN EMBEDDING FOR THE GIVEN PDF FILE

    run_2() IS RESPONSIBLE TO GENERATE A DOT PRODUCT 
    OR COSINE SIMILARITY SCORE FOR THE PROVIDED PDF EMBEDDINGS
    AND QUESTION ENTERED BY THE USER

    run_3() IS RESPONSIBLE FOR IMPORTING generateAnswer() FUNCTION
    FROM groq_implem.py AND PROVIDE GROQ AI USING MODEL 'openai/gpt-oss-120b'
    THE TOP 3 CHUNKS WHICH ARE BASED ON THE HIGHEST 3 SIMILARITY SCORED IN run_2
    """
    # run_1()
    # run_2()
    run_3()