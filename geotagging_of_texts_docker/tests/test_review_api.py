from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
pytest.importorskip("openai")

from app import main
from app.tei import TEI_NS, make_tei
from lxml import etree as ET


def _fixture(tmp_path: Path):
    job_id = "a1b2c3d4e5f6"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    source = "The traveller reached Corinth before dusk."
    start = source.index("Corinth")
    location = {
        "id": "place-1", "kb_id": "123", "name": "Corinth", "description": "Ancient city",
        "type": "City", "latitude": 37.9, "longitude": 22.9, "url": "https://topostext.org/place/123",
        "region": "Corinthia", "country": "Greece", "department": "", "modern_location": "Corinth",
        "coordinate_confidence": "", "wikipedia_url": "", "mention_count": 1,
    }
    mention = {
        "id": "m-1", "paragraph_index": 0, "start": start, "end": start + len("Corinth"),
        "mention": "Corinth", "label": "PLACE", "recontext": "Ancient city", "match_score": 0.8,
        "detection_method": "Test", "match_method": "test", "context": source, "snippet": source,
        "location": location,
    }
    result = {
        "job_id": job_id, "filename": "sample.txt", "title": "sample",
        "paragraphs": [{"index": 0, "text": source}], "mentions": [mention], "locations": [location],
        "stats": {"paragraphs": 1, "mentions": 1, "disambiguated": 1, "mapped": 1, "unresolved": 0},
        "warnings": [], "tei_filename": "sample_geotagged.tei.xml",
    }
    (job_dir / result["tei_filename"]).write_bytes(make_tei(result["filename"], result["paragraphs"], result["mentions"], result["locations"]))
    (job_dir / "results.json").write_text(json.dumps(result), encoding="utf-8")
    return job_id, result, location, mention


def test_save_review_updates_json_and_tei(tmp_path, monkeypatch):
    job_id, result, location, mention = _fixture(tmp_path)
    monkeypatch.setattr(main, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(main, "_jobs", {})
    edited_location = {**location, "name": "Ancient Corinth", "description": "Edited description"}
    response = asyncio.run(main.save_review(job_id, {"mentions": [{**mention, "location": edited_location}], "locations": [edited_location]}))
    assert response["status"] == "saved"
    saved = json.loads((tmp_path / job_id / "results.json").read_text(encoding="utf-8"))
    assert saved["mentions"][0]["location"]["name"] == "Ancient Corinth"
    xml_tree = ET.parse(str(tmp_path / job_id / saved["tei_filename"]))
    ns = {"tei": TEI_NS}
    annotation = xml_tree.find(".//tei:listAnnotation/tei:annotation", namespaces=ns)
    assert annotation is not None
    assert annotation.get("target") == f"/TEI[1]/text[1]/body[1]/p[1]::{mention['start']} /TEI[1]/text[1]/body[1]/p[1]::{mention['end']}"
    assert annotation.get("ana") == "https://topostext.org/place/123"
    assert saved["stats"]["mentions"] == 1


def test_deleting_review_mention_keeps_source_paragraph(tmp_path, monkeypatch):
    job_id, result, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(main, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(main, "_jobs", {})
    response = asyncio.run(main.save_review(job_id, {"mentions": [], "locations": []}))
    saved = response["result"]
    assert saved["mentions"] == []
    assert saved["locations"] == []
    assert saved["paragraphs"][0]["text"] == result["paragraphs"][0]["text"]
    assert saved["stats"]["mentions"] == 0
