import os
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI


@dataclass
class RetrievedChunk:
    content: str
    source: str
    score: float


class RAGPipeline:
    def __init__(
        self,
        persist_dir: str,
        embedding_model: str,
        llm_model: str,
        openai_api_key: str,
        openai_base_url: str,
    ):
        self.llm_model = llm_model
        self.client = OpenAI(api_key=openai_api_key or "not-set", base_url=openai_base_url)
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            api_base=openai_base_url,
            model_name=embedding_model,
        )
        self.chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma.get_or_create_collection(
            name="documents",
            embedding_function=self.ef,
        )

    @property
    def collection_count(self) -> int:
        return self.collection.count()

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end].strip())
            start = end - overlap
        return [c for c in chunks if c]

    def ingest_text(self, text: str, source: str) -> int:
        chunks = self._chunk_text(text)
        if not chunks:
            return 0
        ids = [f"{source}-{i}-{hash(c) % 10**8}" for i, c in enumerate(chunks)]
        self.collection.add(
            documents=chunks,
            metadatas=[{"source": source}] * len(chunks),
            ids=ids,
        )
        return len(chunks)

    def query(self, question: str, top_k: int) -> tuple[str, list[dict]]:
        results = self.collection.query(query_texts=[question], n_results=top_k)
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        sources = [
            RetrievedChunk(
                content=doc,
                source=meta.get("source", "unknown"),
                score=round(1 - dist, 4),
            )
            for doc, meta, dist in zip(docs, metas, distances)
        ]

        context = "\n\n".join(f"[{s.source}] {s.content}" for s in sources)
        prompt = f"""Answer the question using only the context below. Cite sources when possible.

Context:
{context}

Question: {question}

Answer:"""

        if not os.getenv("OPENAI_API_KEY"):
            return "Configure OPENAI_API_KEY to generate answers.", [
                {"content": s.content, "source": s.source, "score": s.score} for s in sources
            ]

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = response.choices[0].message.content or ""
        return answer, [
            {"content": s.content, "source": s.source, "score": s.score} for s in sources
        ]
