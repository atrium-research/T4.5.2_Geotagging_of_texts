from __future__ import annotations

import gc
import io
import json
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

import fitz
import spacy
from lxml import etree as ET

from .config import CHAT_MODEL, EMBEDDING_MODEL, NER_MODEL_PATH, OCR_MODEL, RERANKER_MODEL, device
from .gazetteer import embedding_candidates, load_gazetteer, lookup_place_by_id
from .llm import llama
from .tei import write_notebook_tei
from .text_utils import split_text

Progress = Callable[[int, str, str], None]
_pipeline_lock = threading.Lock()
_nlp = None
_nlp_error: str | None = None
_embedding_model = None

SYSTEM_RECONTEXT = """You are an expert historian and archaeologist specializing in identifying historical and archaeological entities.

## Input

You will receive:
- **Mention:** A referenced entity, such as a building, temple, city, region, island, country, monument, harbor, person, or object.
- **Context:** A short text snippet used only to disambiguate the Mention.

## Task

Identify the most likely real-world entity referred to by the Mention using the Context for disambiguation. Produce its canonical title followed by a concise, retrieval-optimized description.

## Mandatory Output Format

Return exactly one line in this format:

**Canonical Title: Description.**

The output MUST contain both parts:
1. **Canonical Title:** The standard, most recognizable name of the identified entity.
2. **Description:** A concise factual description of the entity.

The colon (`:`) separating the canonical title and description is mandatory.

**Never output the description alone. Never omit the canonical title.**

## Canonical Title Rules

- Always place the canonical title at the very beginning of the output.
- Use the historically or conventionally recognized name of the identified entity.
- Preserve the appropriate spelling, capitalization, and diacritics.
- Do not replace the canonical title with a generic entity type or a descriptive phrase.
- Include alternative names in the title only when they are essential for identifying the entity.

## Description Rules

- Describe the identified entity directly and factually.
- Include the entity type, geographical location where applicable, and key identifying feature or function.
- Keep the description compact and optimized for entity retrieval.
- Do not repeat the canonical title unnecessarily in the description.
- Avoid additional historical narrative unless essential for identification.

## Context Restrictions

- Use the Context ONLY to disambiguate the Mention and identify the correct entity.
- Do NOT mention, summarize, quote, or describe the Context or source text.
- Do NOT explain your reasoning or how the entity was identified.
- Do NOT include citations, references, notes, confidence scores, or disclaimers.

## Output Restrictions

- Output exactly one line containing the canonical title, a colon, a space, and the description.
- Do not include headings, labels, quotation marks, bullet points, numbering, or Markdown formatting.
- Do not output multiple candidate entities.
- Do not output anything before or after the required line.
- Ensure every output follows the same structure, regardless of entity type.

## Examples

Kantharos: The main ancient harbor of Piraeus in Athens, Greece.

Athens: A major city in Greece and the historical center of ancient Greek civilization.

Pompeion: A public building in ancient Athens used for organizing religious processions.

Acropolis of Athens: An ancient fortified citadel in Athens, Greece, containing major religious and monumental structures.

Alexander the Great: A king of Macedon who created one of the largest empires of the ancient world.

## Final Instruction

Before responding, verify that the output begins with the canonical title, immediately followed by a colon and a space, and then the entity description. If the canonical title is missing or the colon separator is absent, the output is invalid and must be corrected.

Return only the formatted output line.
"""


def _progress(callback: Progress, percent: int, stage: str, message: str) -> None:
    callback(percent, stage, message)


def _load_nlp() -> tuple[Any | None, str | None]:
    global _nlp, _nlp_error
    if _nlp is not None:
        return _nlp, None
    if _nlp_error is not None:
        return None, _nlp_error
    try:
        if device() == "cuda":
            spacy.prefer_gpu()
        _nlp = spacy.load(NER_MODEL_PATH)
        return _nlp, None
    except Exception as exc:
        _nlp_error = str(exc)
        return None, _nlp_error


def preload_pipeline_models(on_status=None) -> None:
    """Load and warm the NER, embedding and gazetteer assets before accepting jobs."""
    say = on_status or (lambda _msg: None)
    nlp, error = _load_nlp()
    if nlp is None:
        raise RuntimeError(f"Could not preload the historical-place NER model '{NER_MODEL_PATH}': {error}")
    nlp("Athens and Corinth.")
    say(f"Loaded NER model on {device().upper()}: {NER_MODEL_PATH}")
    _embed_query("Athens: a city in Greece.", on_status=say)
    say(f"Loaded embedding model on {device().upper()}: {EMBEDDING_MODEL}")
    _, records, warning = load_gazetteer()
    if warning:
        raise RuntimeError(f"ToposText gazetteer loading failed: {warning}")
    say(f"Loaded ToposText gazetteer: {len(records)} records.")


def release_pipeline_models() -> None:
    """Release Python model memory at application shutdown."""
    global _nlp, _nlp_error
    _release_embedding_model()
    _nlp = None
    _nlp_error = None


def pipeline_model_status() -> dict[str, bool]:
    return {"ner": _nlp is not None, "embedding": _embedding_model is not None}


def extract_paragraphs(filename: str, content: bytes, progress: Progress) -> tuple[list[dict[str, Any]], list[str]]:
    suffix = Path(filename).suffix.lower()
    warnings: list[str] = []
    paragraphs: list[str] = []
    if suffix == ".txt":
        paragraphs = split_text(content.decode("utf-8-sig", errors="replace"))
    elif suffix in {".xml", ".tei"}:
        parser = ET.XMLParser(resolve_entities=False, no_network=True, recover=False, huge_tree=False)
        tree = ET.parse(io.BytesIO(content), parser=parser)
        root = tree.getroot()
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}
        nodes = root.findall(".//tei:p", namespaces=ns)
        paragraphs = ["".join(node.itertext()).strip() for node in nodes]
    elif suffix == ".pdf":
        with fitz.open(stream=content, filetype="pdf") as document:
            page_text: list[str] = []
            for page in document:
                page_text.append(page.get_text("text").strip())
            extracted_count = sum(len(t) for t in page_text)
            use_ocr = extracted_count < max(80, len(document) * 35)
            if use_ocr:
                _progress(progress, 12, "extract", "Little embedded PDF text was found; using local GLM-OCR through llama.cpp.")
            for index, (page, existing) in enumerate(zip(document, page_text), start=1):
                current = existing
                if use_ocr and len(existing) < 50:
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False)
                    try:
                        current = llama.ocr_page(OCR_MODEL, pix.tobytes("png"), page_number=index,
                                                 on_status=lambda msg: _progress(progress, 12, "extract", msg))
                    except Exception as exc:
                        warnings.append(f"OCR failed on PDF page {index}: {exc}")
                paragraphs.extend(split_text(current))
    else:
        raise ValueError("Unsupported file type. Upload one .pdf, .txt, or .xml/.tei file.")

    if not paragraphs:
        raise ValueError("No readable text was found in this file. For a scanned PDF, check that the GLM-OCR model can be downloaded and loaded.")
    rows = [{"index": i, "text": p} for i, p in enumerate(paragraphs)]
    return rows, warnings


def recognize_places(paragraphs: list[dict[str, Any]], records: list[dict[str, Any]], progress: Progress) -> tuple[list[dict[str, Any]], str | None]:
    """Run the notebook's pretrained spaCy NER model; fail rather than substitute a heuristic."""
    nlp, error = _load_nlp()
    if nlp is None:
        raise RuntimeError(f"Could not load the historical-place NER model '{NER_MODEL_PATH}': {error}")
    if device() == "cuda":
        spacy.prefer_gpu()  # thinc keeps its GPU setup per thread, and each job runs on its own worker thread

    found: list[dict[str, Any]] = []
    total = max(1, len(paragraphs))
    for i, row in enumerate(paragraphs):
        doc = nlp(row["text"])
        for span in doc.ents:
            if not span.text.strip():
                continue
            found.append({
                "id": f"m-{row['index']}-{span.start_char}-{uuid.uuid4().hex[:5]}",
                "paragraph_index": row["index"],
                "start": span.start_char,
                "end": span.end_char,
                "mention": span.text,
                "label": "PLACE",
            })
        progress(20 + int(20 * (i + 1) / total), "ner",
                 f"Recognizing place names in paragraph {i + 1} of {total}.")
    found.sort(key=lambda item: (item["paragraph_index"], item["start"]))
    return found, None


def _get_embedding_model(on_status=None):
    """Load the baked SentenceTransformer on the GPU when one is present, otherwise the CPU."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    from sentence_transformers import SentenceTransformer

    if on_status:
        on_status(f"Loading {EMBEDDING_MODEL} embedding model on {device().upper()}.")
    _embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device(),
        model_kwargs={"local_files_only": True},
        tokenizer_kwargs={"local_files_only": True},
    )
    return _embedding_model


def _embed_query(query: str, on_status=None) -> list[float]:
    model = _get_embedding_model(on_status=on_status)
    vector = model.encode(query, prompt_name="SearchQuery", normalize_embeddings=True)
    return vector.tolist()


def _release_embedding_model() -> None:
    global _embedding_model
    _embedding_model = None
    gc.collect()


def _call_recontext(mention: dict[str, Any], paragraph_text: str, progress: Progress) -> str:
    """Call the local Qwen model with the notebook's context/mention input format."""
    user_input = f"Context: {paragraph_text}\n\nMention: {mention['mention']}"
    return llama.chat(
        CHAT_MODEL,
        [
            {"role": "system", "content": SYSTEM_RECONTEXT},
            {"role": "user", "content": user_input},
        ],
        reasoning="off",
        on_status=lambda msg: progress(45, "context", msg),
        max_tokens=256,
        temperature=0.0,
        timeout=240,
    )


def _rerank(mention: dict[str, Any], context: str, candidates: list[dict[str, Any]], progress: Progress) -> tuple[dict[str, Any] | None, float, str]:
    if not candidates:
        raise RuntimeError(f"FAISS/metadata retrieval returned no candidates for mention: {mention.get('mention', '')}")

    documents = [
        f"{str(candidate.get('title') or '').strip()}: {str(candidate.get('description') or '').strip()}"
        for candidate in candidates
    ]
    query = str(mention.get("recontext") or "").strip()
    if not query:
        raise RuntimeError(f"The contextualized query is empty for mention: {mention.get('mention', '')}")
    if not documents or any(not document.strip() for document in documents):
        raise RuntimeError("The ToposText metadata produced an empty reranker document.")

    response = llama.rerank(
        RERANKER_MODEL,
        query=query,
        documents=documents,
        top_n=len(documents),
        on_status=lambda msg: progress(72, "rerank", msg),
        timeout=240,
    )
    results = response.get("results") or []
    if not results:
        raise RuntimeError("The Qwen3 reranker returned no results.")
    selected_index = int(results[0].get("index", -1))
    if selected_index < 0 or selected_index >= len(candidates):
        raise RuntimeError(f"The Qwen3 reranker returned an invalid candidate index: {selected_index}.")
    chosen = dict(candidates[selected_index])
    # Notebook's reranker_results value is ids[index_number].
    chosen["toposid"] = str(chosen.get("toposid") or "")
    chosen["id"] = chosen["toposid"]
    chosen["url"] = f"https://topostext.org/place/{chosen['toposid']}" if chosen["toposid"] else ""
    score = results[0].get("relevance_score", results[0].get("score", chosen.get("retrieval_score", 0.0)))
    return chosen, max(0.0, min(1.0, float(score))), "qwen3-reranker"


def run_pipeline(job_id: str, filename: str, content: bytes, job_dir: Path, progress: Progress) -> dict[str, Any]:
    with _pipeline_lock:
        warnings: list[str] = []
        _progress(progress, 5, "extract", "Reading and normalizing the uploaded document.")
        paragraphs, extraction_warnings = extract_paragraphs(filename, content, progress)
        warnings.extend(extraction_warnings)
        _progress(progress, 16, "gazetteer", "Loading the required ToposText v2 metadata and FAISS index.")
        index, records, gazetteer_warning = load_gazetteer(on_status=lambda msg: progress(17, "gazetteer", msg))
        if gazetteer_warning:
            raise RuntimeError(f"ToposText gazetteer loading failed: {gazetteer_warning}")
        if index is None:
            raise RuntimeError("The required ToposText FAISS index is unavailable; semantic retrieval cannot continue.")
        if not records:
            raise RuntimeError("The ToposText JSONL metadata is empty. Semantic retrieval is required.")
        if int(index.ntotal) != len(records):
            raise RuntimeError(
                f"ToposText FAISS/metadata row mismatch: index has {index.ntotal} vectors, "
                f"but JSONL has {len(records)} rows. Use the matching v2 index and metadata files."
            )

        _progress(progress, 20, "ner", "Running historical place-name recognition.")
        mentions, ner_warning = recognize_places(paragraphs, records, progress)
        if ner_warning:
            warnings.append(ner_warning)
        if not mentions:
            warnings.append("No place mentions were detected in the document.")

        para_map = {row["index"]: row["text"] for row in paragraphs}
        total = max(1, len(mentions))
        for i, mention in enumerate(mentions):
            pct = 41 + int(15 * (i + 1) / total)
            _progress(progress, pct, "context", f"Disambiguation context {i + 1} of {len(mentions)}.")
            mention["recontext"] = _call_recontext(mention, para_map[mention["paragraph_index"]], progress)
        retrieved: list[list[dict[str, Any]]] = []
        total = max(1, len(mentions))
        for i, mention in enumerate(mentions):
            pct = 57 + int(14 * (i + 1) / total)
            _progress(progress, pct, "retrieve", f"Creating EmbeddingGemma query and searching FAISS for mention {i + 1} of {len(mentions)}.")
            # Match notebook: query_text=mention['recontext']; SearchQuery prompt;
            # normalized embeddings; FAISS top-10 retrieval; no fuzzy fallback.
            candidates = embedding_candidates(
                mention["mention"], mention.get("recontext", ""), index, records,
                lambda query, on_status=None: _embed_query(query, on_status=on_status),
                on_status=lambda msg: progress(pct, "retrieve", msg), limit=10,
            )
            if not candidates:
                raise RuntimeError(f"EmbeddingGemma/FAISS returned no candidates for mention: {mention['mention']}")
            mention["candidates"] = candidates
            retrieved.append(candidates)
        locations_by_key: dict[str, dict[str, Any]] = {}
        selected_mentions: list[dict[str, Any]] = []
        total = max(1, len(mentions))
        for i, (mention, candidates) in enumerate(zip(mentions, retrieved)):
            pct = 72 + int(10 * (i + 1) / total)
            _progress(progress, pct, "rerank", f"Selecting the best match for mention {i + 1} of {len(mentions)}.")
            chosen, match_score, method = _rerank(mention, para_map[mention["paragraph_index"]], candidates,
                                                  lambda p, s, m: progress(max(pct, p), s, m))
            if chosen:
                chosen = dict(chosen)
                # Resolve via the authoritative JSONL toposid, never a generic internal/index ID.
                lookup_id = chosen.get("toposid") or chosen.get("id")
                metadata_record = lookup_place_by_id(lookup_id, records)
                if metadata_record:
                    authoritative_id = str(metadata_record.get("toposid") or "").strip()
                    for field in ("name", "title", "description", "type", "latitude", "longitude", "region", "country",
                                  "department", "coordinate_confidence", "modern_location", "wikipedia_url",
                                  "wikipedia_id", "pleiades_id", "wikidata_id", "toposid"):
                        if field in metadata_record:
                            # Treat explicitly missing coordinates as authoritative too; don't
                            # retain potentially mismatched coordinates from another candidate.
                            if field in ("latitude", "longitude") or metadata_record.get(field) not in (None, ""):
                                chosen[field] = metadata_record.get(field)
                    if authoritative_id:
                        chosen["id"] = authoritative_id
                        chosen["toposid"] = authoritative_id
                        chosen["url"] = f"https://topostext.org/place/{authoritative_id}"
                    else:
                        chosen["url"] = str(metadata_record.get("url") or "")
                else:
                    authoritative_id = str(chosen.get("toposid") or "").strip()
                    if authoritative_id and not authoritative_id.startswith("metadata-"):
                        chosen["id"] = authoritative_id
                        chosen["url"] = f"https://topostext.org/place/{authoritative_id}"
                    else:
                        chosen["url"] = ""
                if not chosen.get("name") or str(chosen.get("name", "")).startswith("ToposText place "):
                    chosen["name"] = chosen.get("modern_location") or mention["mention"]
                key = str(chosen.get("url") or chosen.get("id") or chosen.get("name"))
                normalized_location = locations_by_key.setdefault(key, {
                    "id": f"place-{len(locations_by_key) + 1}",
                    "kb_id": str(chosen.get("id") or ""),
                    "name": str(chosen.get("name") or mention["mention"]),
                    "description": str(chosen.get("description") or ""),
                    "type": str(chosen.get("type") or "Unclassified place"),
                    "latitude": chosen.get("latitude"), "longitude": chosen.get("longitude"),
                    "region": str(chosen.get("region") or ""), "country": str(chosen.get("country") or ""),
                    "department": str(chosen.get("department") or ""),
                    "coordinate_confidence": str(chosen.get("coordinate_confidence") or ""),
                    "modern_location": str(chosen.get("modern_location") or ""),
                    "wikipedia_url": str(chosen.get("wikipedia_url") or ""),
                    "wikipedia_id": str(chosen.get("wikipedia_id") or ""),
                    "pleiades_id": str(chosen.get("pleiades_id") or ""),
                    "wikidata_id": str(chosen.get("wikidata_id") or ""),
                    "toposid": str(chosen.get("toposid") or chosen.get("id") or ""),
                    "url": str(chosen.get("url") or ""),
                    "match_method": method, "match_score": match_score, "mention_count": 0,
                })
                normalized_location["mention_count"] += 1
                mention["location"] = normalized_location
            else:
                mention["location"] = {
                    "id": f"unresolved-{i+1}", "name": mention["mention"], "description": "No matching ToposText record found.",
                    "type": "Unresolved", "latitude": None, "longitude": None, "url": "", "mention_count": 1,
                }
            mention["match_score"] = match_score
            mention["detection_method"] = "Historical place NER"
            mention["match_method"] = method
            mention["context"] = para_map[mention["paragraph_index"]]
            mention["snippet"] = para_map[mention["paragraph_index"]][max(0, mention["start"] - 80):min(len(para_map[mention["paragraph_index"]]), mention["end"] + 80)]
            selected_mentions.append(mention)

        locations = list(locations_by_key.values())
        # The notebook uses the v2 JSONL record directly: no online enrichment,
        # no alternate geocoder, and no description replacement from Wikipedia.
        _progress(progress, 90, "tei", "Generating notebook-compatible stand-off TEI annotations.")
        # Keep selected mention records compact and avoid embedding the full candidate lists in API results.
        output_mentions = []
        for mention in selected_mentions:
            output_mentions.append({k: mention.get(k) for k in (
                "id", "paragraph_index", "start", "end", "mention", "label", "recontext", "match_score", "detection_method", "match_method", "context", "snippet", "location"
            )})
        # Make the stand-off place list consistent with enriched coordinates and mention counts.
        by_location = {loc["id"]: loc for loc in locations}
        for item in output_mentions:
            loc = item.get("location") or {}
            if loc.get("id") in by_location:
                item["location"] = by_location[loc["id"]]
        tei_name = write_notebook_tei(
            job_dir, filename, paragraphs, output_mentions,
            locations + [m["location"] for m in output_mentions if m["location"].get("type") == "Unresolved"],
        )
        result = {
            "job_id": job_id,
            "filename": filename,
            "title": Path(filename).stem,
            "paragraphs": paragraphs,
            "mentions": output_mentions,
            "locations": locations,
            "stats": {
                "paragraphs": len(paragraphs), "mentions": len(output_mentions),
                "disambiguated": sum(1 for m in output_mentions if m.get("location", {}).get("url")),
                "mapped": sum(1 for loc in locations if loc.get("latitude") is not None and loc.get("longitude") is not None),
                "unresolved": sum(1 for m in output_mentions if m.get("location", {}).get("type") == "Unresolved"),
            },
            "warnings": list(dict.fromkeys(warnings)),
            "tei_filename": tei_name,
        }
        (job_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        _progress(progress, 100, "complete", "Notebook-compatible TEI export is ready. Review the map and annotations, then download the file.")
        return result
