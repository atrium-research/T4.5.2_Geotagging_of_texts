import unittest
from lxml import etree as ET

from app.tei import make_tei, TEI_NS, XML_NS
from app.text_utils import split_text
from app.gazetteer import normalize_metadata, normalize_jsonl_metadata, lookup_place_by_id


class TextParsingTests(unittest.TestCase):
    def test_blank_line_paragraphs(self):
        self.assertEqual(split_text("Athens\n\n  Corinth \n\nSparta"), ["Athens", "Corinth", "Sparta"])

    @unittest.skipIf(normalize_metadata is None, "application dependencies are installed in the Docker image")
    def test_columnar_topostext_metadata(self):
        raw = {
            "ids": [["https://topostext.org/place/123"]],
            "meta": ["https://topostext.org/place/123"],
            "desc": ["Athens: Athens is a major city in Greece."],
        }
        records = normalize_metadata(raw)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], "123")
        self.assertEqual(records[0]["name"], "Athens")
        self.assertEqual(records[0]["url"], "https://topostext.org/place/123")
        self.assertEqual(lookup_place_by_id("123", records)["name"], "Athens")

    @unittest.skipIf(normalize_jsonl_metadata is None, "application dependencies are installed in the Docker image")
    def test_v2_jsonl_metadata_uses_toposid_title_description_and_preserves_row_order(self):
        rows = [
            {"id": "index-row-0", "toposid": 123, "title": "Athens", "description": "A major city in Greece.",
             "latitude": 37.98, "longitude": 23.72, "settlement_type": "city"},
            {"toposid": 123, "title": "Athens alternate context", "description": "A separate indexed passage."},
        ]
        records = normalize_jsonl_metadata(rows)
        self.assertEqual(len(records), 2, "metadata positions must stay aligned with FAISS rows")
        self.assertEqual(records[0]["id"], "123")
        self.assertEqual(records[0]["toposid"], "123")
        self.assertEqual(records[0]["name"], "Athens")
        self.assertEqual(records[0]["title"], "Athens")
        self.assertEqual(records[0]["description"], "A major city in Greece.")
        self.assertAlmostEqual(records[0]["latitude"], 37.98)
        self.assertEqual(records[1]["title"], "Athens alternate context")
        self.assertEqual(records[1]["description"], "A separate indexed passage.")

    @unittest.skipIf(normalize_metadata is None, "application dependencies are installed in the Docker image")
    def test_wikipedia_field_is_not_misread_as_place_name(self):
        raw = [{
            "id": "777",
            "desc": "Wikipedia: https://en.wikipedia.org/wiki/Phoenicus_(Cythera) Wikidata ID: Q60792041 DARE: 46291",
        }]
        records = normalize_metadata(raw)
        self.assertEqual(records[0]["name"], "Phoenicus (Cythera)")
        self.assertEqual(records[0]["wikipedia_url"], "https://en.wikipedia.org/wiki/Phoenicus_(Cythera)")

    def test_toposid_beats_generic_id_and_extracts_external_identifiers(self):
        row = {
            "id": "540888", "toposid": 1234, "title": "Example", 
            "description": "Example: site. Wikipedia: https://en.wikipedia.org/wiki/Example_Place Pleiades: 991234 Wikidata ID: Q12345",
            "coordinates": {"lat": 37.4, "lon": 22.1},
        }
        record = normalize_jsonl_metadata([row])[0]
        self.assertEqual(record["id"], "1234")
        self.assertEqual(record["url"], "https://topostext.org/place/1234")
        self.assertEqual(record["wikipedia_id"], "Example Place")
        self.assertEqual(record["pleiades_id"], "991234")
        self.assertEqual(record["wikidata_id"], "Q12345")
        self.assertAlmostEqual(record["latitude"], 37.4)
        self.assertAlmostEqual(record["longitude"], 22.1)


    def test_nested_toposid_and_identifiers_are_read_but_generic_id_is_not_a_place_id(self):
        rows = [
            {
                "id": "internal-index-row-8",
                "metadata": {
                    "toposid": "379236HPei",
                    "title": "Nested Place",
                    "description": "Nested Place: an ancient settlement.",
                    "identifiers": {
                        "wikipedia_url": "https://en.wikipedia.org/wiki/Nested_Place",
                        "pleiades": "https://pleiades.stoa.org/places/876543",
                        "wikidata": "https://www.wikidata.org/entity/Q54321",
                    },
                },
            },
            {"id": "internal-only-99", "title": "No public ID", "description": "No ToposText identifier."},
            {"id": "internal-url-row", "url": "https://topostext.org/place/7654", "title": "URL identified"},
        ]
        records = normalize_jsonl_metadata(rows)
        self.assertEqual(records[0]["toposid"], "379236HPei")
        self.assertEqual(records[0]["url"], "https://topostext.org/place/379236HPei")
        self.assertEqual(records[0]["wikipedia_id"], "Nested Place")
        self.assertEqual(records[0]["pleiades_id"], "876543")
        self.assertEqual(records[0]["wikidata_id"], "Q54321")
        self.assertEqual(records[1]["toposid"], "")
        self.assertEqual(records[1]["url"], "")
        self.assertTrue(records[1]["id"].startswith("metadata-"))
        self.assertEqual(records[2]["toposid"], "7654")
        self.assertEqual(records[2]["url"], "https://topostext.org/place/7654")

    def test_embedding_retrieval_uses_faiss_ids_and_surfaces_dimension_errors(self):
        import numpy as np
        import app.gazetteer as gazetteer

        class FakeIndex:
            ntotal = 3
            d = 2
            def search(self, vector, k):
                self.searched_vector = vector.copy()
                self.k = k
                return np.zeros((1, k), dtype=np.float32), np.array([[1, 0, 2, 1, 0, 2, 1, 0, 2, 1]], dtype=np.int64)

        index = FakeIndex()
        rows = [
            {"id": "540888", "toposid": "10", "title": "Ten", "description": "first row"},
            {"id": "wrong-id", "toposid": "20", "title": "Twenty", "description": "second row"},
            {"id": "other-index-key", "toposid": "10", "title": "Ten alternate", "description": "other row"},
        ]
        old_faiss = gazetteer.faiss
        gazetteer.faiss = object()
        try:
            query_seen = []
            found = gazetteer.embedding_candidates(
                "Corinth", "Corinth: ancient city", index, rows,
                lambda query, on_status=None: (query_seen.append(query) or [0.1, 0.2]),
            )
            self.assertEqual(query_seen, ["Corinth: ancient city"])
            self.assertEqual(index.k, 10)
            self.assertEqual([item["toposid"] for item in found[:3]], ["20", "10", "10"])
            self.assertEqual(found[1]["title"], "Ten")  # first row for duplicate ToposText ID, as notebook does
            with self.assertRaisesRegex(ValueError, "3 dimensions"):
                gazetteer.embedding_candidates(
                    "Corinth", "query", index, rows,
                    lambda query, on_status=None: [0.1, 0.2, 0.3],
                )
        finally:
            gazetteer.faiss = old_faiss



class TeiTests(unittest.TestCase):
    def test_emits_notebook_standoff_annotations_and_exact_offsets(self):
        paragraphs = [{"index": 0, "text": "I travelled to Corinth."}]
        location = {"id": "place-1", "kb_id": "102", "toposid": "102", "name": "Corinth",
                    "url": "https://topostext.org/place/102"}
        mentions = [{"id": "m-1", "paragraph_index": 0, "start": 15, "end": 22,
                     "mention": "Corinth", "location": location}]
        document = make_tei("sample.txt", paragraphs, mentions, [location])
        root = ET.fromstring(document)
        ns = {"tei": TEI_NS}
        self.assertEqual(root.get("version"), "3.3.0")
        self.assertEqual(root.find("tei:standOff", namespaces=ns).get("type"), "recogito_studio_annotations")
        self.assertEqual(root.findtext("tei:text/tei:body/tei:p", namespaces=ns), "I travelled to Corinth.")
        ann = root.find("tei:standOff/tei:listAnnotation/tei:annotation", namespaces=ns)
        self.assertEqual(ann.get("target"), "/TEI[1]/text[1]/body[1]/p[1]::15 /TEI[1]/text[1]/body[1]/p[1]::22")
        self.assertEqual(ann.get(f"{{{XML_NS}}}id"), "UID-FAKE-0")
        self.assertEqual(ann.get("ana"), "https://topostext.org/place/102")
        self.assertIsNone(root.find(".//tei:listPlace", namespaces=ns))

    def test_notebook_export_splits_at_50_paragraphs_and_keeps_uid_counter(self):
        from app.tei import make_tei_documents
        paragraphs = [{"index": i, "text": f"Paragraph {i} Corinth."} for i in range(51)]
        mentions = []
        for i, paragraph in enumerate(paragraphs):
            start = paragraph["text"].index("Corinth")
            mentions.append({"paragraph_index": i, "start": start, "end": start + len("Corinth"),
                             "mention": "Corinth", "location": {"toposid": "102"}})
        docs = make_tei_documents("sample.txt", paragraphs, mentions, [], max_per_file=50)
        self.assertEqual([name for name, _ in docs], ["annotation_output_1.xml", "annotation_output_2.xml"])
        first = ET.fromstring(docs[0][1])
        second = ET.fromstring(docs[1][1])
        ns = {"tei": TEI_NS}
        self.assertEqual(len(first.findall("tei:text/tei:body/tei:p", namespaces=ns)), 50)
        self.assertEqual(len(second.findall("tei:text/tei:body/tei:p", namespaces=ns)), 1)
        anns1 = first.findall("tei:standOff/tei:listAnnotation/tei:annotation", namespaces=ns)
        anns2 = second.findall("tei:standOff/tei:listAnnotation/tei:annotation", namespaces=ns)
        self.assertEqual(anns1[-1].get(f"{{{XML_NS}}}id"), "UID-FAKE-49")
        self.assertEqual(anns2[0].get(f"{{{XML_NS}}}id"), "UID-FAKE-50")
        self.assertIn("p[1]::", anns2[0].get("target"))

    def test_writer_returns_notebook_xml_path_or_zip_for_multiple_chunks(self):
        import zipfile
        from app.tei import write_notebook_tei
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            paragraphs = [{"index": 0, "text": "Corinth"}]
            mention = {"paragraph_index": 0, "start": 0, "end": 7,
                       "location": {"toposid": "102"}}
            path = write_notebook_tei(Path(temp), "sample.txt", paragraphs, [mention], [])
            self.assertEqual(path, "annotation_env/annotation_output_1.xml")
            self.assertTrue((Path(temp) / path).exists())
        with tempfile.TemporaryDirectory() as temp:
            paragraphs = [{"index": i, "text": f"Corinth {i}"} for i in range(51)]
            mentions = [{"paragraph_index": i, "start": 0, "end": 7,
                         "location": {"toposid": "102"}} for i in range(51)]
            path = write_notebook_tei(Path(temp), "sample.txt", paragraphs, mentions, [])
            self.assertTrue(path.endswith(".tei.zip"))
            with zipfile.ZipFile(Path(temp) / path) as archive:
                self.assertEqual(archive.namelist(), ["annotation_output_1.xml", "annotation_output_2.xml"])


if __name__ == "__main__":
    unittest.main()
