import datetime
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import OpenAIEmbeddings
import numpy as np

glossary_engine = create_engine("sqlite:///health.db")

def setup_glossary_table():
    create_table_query = """
    CREATE TABLE IF NOT EXISTS glossary (
        term TEXT PRIMARY KEY,
        definition TEXT,
        last_updated DATE,
        quality_score FLOAT
    );
    """
    with glossary_engine.connect() as conn:
        conn.execute(text(create_table_query))
        conn.commit()

def preload_glossary_terms(terms):
    insert_query = """
        INSERT OR IGNORE INTO glossary (term, definition, last_updated, quality_score)
        VALUES (:term, :definition, :last_updated, :quality_score)
    """
    with glossary_engine.connect() as conn:
        for term, definition in terms:
            conn.execute(
                text(insert_query),
                {
                    "term": term,
                    "definition": definition,
                    "last_updated": datetime.date.today(),
                    "quality_score": 1.0,
                },
            )
        conn.commit()

def fetch_glossary_definition(term):
    query = "SELECT definition FROM glossary WHERE term = :term"
    with glossary_engine.connect() as conn:
        result = conn.execute(text(query), {"term": term}).fetchone()
        return result[0] if result else None

def generate_definition_from_context(context_sentences):
    if not context_sentences or all(not s.strip() for s in context_sentences):
        return "No context available to generate definition."

    excerpts = [s.strip() for s in context_sentences if s.strip()][:3]  # first 3 sentences/snippets
    combined = " ".join(excerpts)
    if len(combined) > 500:
        combined = combined[:500] + "..."
    return combined

def validate_definition_with_context(term, definition, context_sentences=None):
    # Basic validation: Does definition contain the term?
    if term.lower() in definition.lower():
        return "Definition contains the term and appears valid."
    else:
        return "Warning: Definition does not explicitly mention the term."



# Load embeddings once (if not already)
embeddings = OpenAIEmbeddings()

def score_definition_quality(definition, context_sentences):
    # Join context into a single string
    context_text = " ".join(context_sentences).strip()

    # If no context is available, return lowest score
    if not context_text:
        return 0

    try:
        # Get embeddings
        def_embedding = embeddings.embed_query(definition)
        context_embedding = embeddings.embed_query(context_text)

        # Calculate cosine similarity
        dot_product = np.dot(def_embedding, context_embedding)
        norm_def = np.linalg.norm(def_embedding)
        norm_ctx = np.linalg.norm(context_embedding)
        similarity = dot_product / (norm_def * norm_ctx)

        # Convert similarity (0–1) to score (0–10)
        score = round(similarity * 10)
        return score

    except Exception as e:
        print(f"[Scoring Error] {e}")
        return 0  # Fail-safe: low score on any error

