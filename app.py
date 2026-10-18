from flask import Flask, render_template, request, jsonify, make_response
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import re

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = BASE_DIR / "reference_documents"
ALLOWED_EXTENSIONS = {".txt"}

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embedding_model = None
SIMILARITY_THRESHOLD = 0.62


def get_embedding_model():
    """Load the semantic model once and reuse it."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(MODEL_NAME)
    return _embedding_model


def normalize(text):
    """Normalize whitespace and case."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= 20]


def load_references():
    refs = []
    REFERENCE_DIR.mkdir(exist_ok=True)

    for path in sorted(REFERENCE_DIR.glob("*.txt")):
        try:
            content = path.read_text(
                encoding="utf-8", errors="ignore"
            ).strip()
            if content:
                refs.append({"name": path.name, "text": content})
        except OSError:
            continue

    return refs


def analyze_document(text):
    references = load_references()
    sentences = split_sentences(text)
    findings = []

    if references and sentences:
        # Split references into passages so long documents can be compared
        # beyond just their opening text.
        reference_passages = []

        for ref in references:
            parts = split_sentences(ref["text"])
            if not parts:
                parts = [ref["text"]]

            for part in parts:
                reference_passages.append({
                    "name": ref["name"],
                    "text": part,
                    "full_text": ref["text"],
                })

        try:
            model = get_embedding_model()

            submitted_vectors = model.encode(
                sentences,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            reference_vectors = model.encode(
                [item["text"] for item in reference_passages],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            similarities = cosine_similarity(
                submitted_vectors, reference_vectors
            )

            for i, sentence in enumerate(sentences):
                best_idx = int(similarities[i].argmax())
                score = float(similarities[i, best_idx])
                ref = reference_passages[best_idx]

                exact = normalize(sentence) in normalize(
                    ref["full_text"]
                )

                if exact or score >= SIMILARITY_THRESHOLD:
                    findings.append({
                        "passage": sentence,
                        "source": ref["name"],
                        "similarity": round(
                            (1.0 if exact else score) * 100, 1
                        ),
                        "match_type": (
                            "Exact text match"
                            if exact else "Semantic similarity"
                        ),
                        "evidence": (
                            "The submitted passage appears verbatim "
                            "in this local reference."
                            if exact else
                            "The model found a passage with a similar "
                            "meaning in this local reference. Review "
                            "the source and context manually."
                        ),
                    })

        except Exception as exc:
            app.logger.exception("Semantic analysis failed")
            raise RuntimeError(
                "The semantic model could not analyze this document. "
                "Check that the model is installed and available."
            ) from exc

    flagged = len(findings)
    flagged_percent = round(
        flagged / max(len(sentences), 1) * 100, 1
    )

    citation_markers = len(re.findall(
        r"\([A-Z][A-Za-z-]+(?:\s+et al\.)?,?\s*(?:19|20)\d{2}\)"
        r"|\[\d+\]",
        text,
    ))

    return {
        "document_sentences": len(sentences),
        "reference_count": len(references),
        "findings": findings,
        "flagged_sentence_percent": flagged_percent,
        "citation_markers_detected": citation_markers,
        "comparison_method": "Sentence Transformers semantic similarity",
        "notice": (
            "This prototype compares text only with local reference "
            "documents. Similarity is not proof of plagiarism."
        ),
    }


def build_review(result):
    notes = []

    if result["reference_count"] == 0:
        notes.append(
            "No local reference documents were found. Add .txt files "
            "to reference_documents/ before scanning."
        )

    if result["findings"]:
        notes.append(
            f"{len(result['findings'])} passage(s) need human review "
            "against the listed local sources."
        )
    else:
        notes.append(
            "No passages crossed the matching threshold. This does "
            "not establish that the paper is original."
        )

    if result["citation_markers_detected"] == 0:
        notes.append(
            "No common author-year or numeric citation markers were "
            "detected. Check citations manually."
        )
    else:
        notes.append(
            f"Detected {result['citation_markers_detected']} "
            "citation-like marker(s); this does not verify citation accuracy."
        )

    return notes


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/analyze")
def analyze():
    text = request.form.get("text", "").strip()
    uploaded = request.files.get("file")

    if uploaded and uploaded.filename:
        if Path(uploaded.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            return jsonify({
                "error": "For this MVP, upload a .txt file. "
                         "PDF/DOCX support is not enabled yet."
            }), 400

        file_text = uploaded.read(2 * 1024 * 1024 + 1)

        if len(file_text) > 2 * 1024 * 1024:
            return jsonify({"error": "File exceeds the 2 MB limit."}), 413

        text = file_text.decode("utf-8", errors="ignore").strip()

    if len(text) < 20:
        return jsonify({
            "error": "Enter or upload at least 20 characters of research text."
        }), 400

    try:
        result = analyze_document(text)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    result["agent_review"] = build_review(result)
    return jsonify(result)


@app.post("/api/report")
def report():
    data = request.get_json(silent=True) or {}
    title = str(
        data.get("title", "PaperProof AI Analysis Report")
    )[:160]
    findings = data.get("findings", [])

    lines = [
        title,
        "=" * len(title),
        "",
        "Prototype report: matches are review signals, not a plagiarism verdict.",
        "",
    ]

    for idx, item in enumerate(findings, 1):
        lines += [
            f"{idx}. {item.get('match_type', 'Similarity finding')}",
            f"Source: {item.get('source', 'Unknown')}",
            f"Similarity: {item.get('similarity', 'N/A')}%",
            f"Submitted passage: {item.get('passage', '')}",
            f"Evidence note: {item.get('evidence', '')}",
            "",
        ]

    lines.append(
        "Scope: local reference documents only. No internet-wide search is performed."
    )

    response = make_response("\n".join(lines))
    response.headers["Content-Type"] = "text/plain; charset=utf-8"
    response.headers["Content-Disposition"] = (
        'attachment; filename="paperproof-report.txt"'
    )
    return response


if __name__ == "__main__":
    app.run(debug=True)