from __future__ import annotations

from pathlib import Path
from typing import Any
import zipfile

from lxml import etree as ET

TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NSMAP = {None: TEI_NS}
MAX_PER_FILE = 50


def make_tei(filename: str, paragraphs: list[dict[str, Any]], mentions: list[dict[str, Any]],
             locations: list[dict[str, Any]], uid_start: int = 0) -> bytes:
    """Produce one TEI file using the notebook's stand-off annotation structure."""
    tei = ET.Element("TEI", nsmap=NSMAP, version="3.3.0")
    standoff = ET.SubElement(tei, "standOff", type="recogito_studio_annotations")
    listannotation = ET.SubElement(standoff, "listAnnotation")
    text = ET.SubElement(tei, "text")
    body = ET.SubElement(text, "body")

    mentions_by_paragraph: dict[int, list[dict[str, Any]]] = {}
    for mention in mentions:
        mentions_by_paragraph.setdefault(int(mention.get("paragraph_index", 0)), []).append(mention)

    uid_counter = int(uid_start)
    for paragraph_counter, paragraph in enumerate(paragraphs, start=1):
        p = ET.SubElement(body, "p")
        p.text = str(paragraph.get("text") or "")
        paragraph_index = int(paragraph.get("index", paragraph_counter - 1))
        for mention in sorted(
            mentions_by_paragraph.get(paragraph_index, []),
            key=lambda item: (int(item.get("start", 0)), int(item.get("end", 0))),
        ):
            start = int(mention.get("start", 0))
            end = int(mention.get("end", 0))
            if start < 0 or end < start or end > len(p.text or ""):
                raise ValueError(
                    f"Mention offsets {start}:{end} are outside paragraph {paragraph_counter} "
                    f"(length {len(p.text or '')})."
                )

            # This target format deliberately matches the notebook, including the space.
            location = mention.get("location") if isinstance(mention.get("location"), dict) else {}
            topos_id = location.get("toposid") or location.get("kb_id")
            annotation = ET.SubElement(
                listannotation,
                "annotation",
                ana=f"https://topostext.org/place/{topos_id}",
                target=(
                    f"/TEI[1]/text[1]/body[1]/p[{paragraph_counter}]::{start} "
                    f"/TEI[1]/text[1]/body[1]/p[{paragraph_counter}]::{end}"
                ),
            )
            annotation.set(f"{{{XML_NS}}}id", f"UID-FAKE-{uid_counter}")

            uid_counter += 1

    tree = ET.ElementTree(tei)
    return ET.tostring(tree, xml_declaration=True, encoding="utf-8", pretty_print=True)


def make_tei_documents(filename: str, paragraphs: list[dict[str, Any]], mentions: list[dict[str, Any]],
                       locations: list[dict[str, Any]], max_per_file: int = MAX_PER_FILE) -> list[tuple[str, bytes]]:
    """Build the notebook's max-50-paragraph annotation_output_N.xml files."""
    if max_per_file < 1:
        raise ValueError("max_per_file must be at least 1")
    chunks = [paragraphs[i:i + max_per_file] for i in range(0, len(paragraphs), max_per_file)]
    documents: list[tuple[str, bytes]] = []
    uid_counter = 0
    for file_index, chunk in enumerate(chunks, start=1):
        paragraph_indexes = {int(row.get("index", i)) for i, row in enumerate(chunk)}
        chunk_mentions = [m for m in mentions if int(m.get("paragraph_index", 0)) in paragraph_indexes]
        documents.append((f"annotation_output_{file_index}.xml",
                          make_tei(filename, chunk, chunk_mentions, locations, uid_start=uid_counter)))
        uid_counter += len(chunk_mentions)
    return documents


def write_notebook_tei(output_dir: Path, filename: str, paragraphs: list[dict[str, Any]],
                       mentions: list[dict[str, Any]], locations: list[dict[str, Any]]) -> str:
    """Write TEI files as in the notebook; return the direct XML or multi-file ZIP path."""
    output_dir = Path(output_dir)
    annotation_dir = output_dir / "annotation_env"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    for old_file in annotation_dir.glob("annotation_output_*.xml"):
        old_file.unlink()
    for old_archive in output_dir.glob("*_geotagged.tei.zip"):
        old_archive.unlink()

    documents = make_tei_documents(filename, paragraphs, mentions, locations)
    if not documents:
        raise ValueError("The notebook TEI export has no paragraph chunks to write.")
    for document_name, content in documents:
        (annotation_dir / document_name).write_bytes(content)

    safe_stem = re_safe_stem(Path(filename).stem)
    if len(documents) == 1:
        return f"annotation_env/{documents[0][0]}"

    archive_name = f"{safe_stem}_geotagged.tei.zip"
    archive_path = output_dir / archive_name
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for document_name, _ in documents:
            archive.write(annotation_dir / document_name, arcname=document_name)
    return archive_name


def re_safe_stem(value: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")[:80] or "document"
