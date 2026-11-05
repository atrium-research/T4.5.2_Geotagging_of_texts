from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

try:
    import faiss
except ImportError:
    faiss = None
import numpy as np
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets" / "topostext"
INDEX_PATH = Path(os.getenv("TOPOS_INDEX_PATH", str(ASSETS_DIR / "topostext.index")))
METADATA_PATH = Path(os.getenv("TOPOS_METADATA_PATH", str(ASSETS_DIR / "topostext_metadata.jsonl")))
_WIKI_URL_RE = re.compile(r"https?://(?:[a-z0-9-]+\.)?wikipedia\.org/wiki/[^\s<>\"'\]]+", flags=re.I)

_lock = threading.RLock()
_loaded_key: tuple[str, str, float, float] | None = None
_index = None
_records: list[dict[str, Any]] = []
_raw_metadata: Any = None


def _scalar(value: Any) -> str:
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value).strip()
    return ""


def _strings(value: Any, limit: int = 100) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for child in value.values():
            found.extend(_strings(child, limit))
            if len(found) >= limit:
                break
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            found.extend(_strings(child, limit))
            if len(found) >= limit:
                break
    return found[:limit]


def _find_key(mapping: dict[str, Any], possible: tuple[str, ...]) -> Any:
    for key, value in mapping.items():
        normalized = re.sub(r"[^a-z]", "", str(key).lower())
        if normalized in possible and value not in (None, ""):
            return value
    return None


def _preferred_value(mapping: dict[str, Any], possible: tuple[str, ...]) -> Any:
    """Choose aliases in declared priority order rather than JSON field order."""
    normalized = {re.sub(r"[^a-z]", "", str(key).lower()): value for key, value in mapping.items()}
    for name in possible:
        value = normalized.get(re.sub(r"[^a-z]", "", name.lower()))
        if value not in (None, ""):
            return value
    return None


def _walk_key_values(value: Any):
    """Yield nested object fields without confusing generic IDs with ToposText IDs."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield re.sub(r"[^a-z0-9]", "", str(key).lower()), child
            yield from _walk_key_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_key_values(child)


def _deep_preferred_value(mapping: dict[str, Any], possible: tuple[str, ...], depth: int = 4) -> Any:
    """Prefer an explicitly named field, then search nested metadata objects.

    The ToposText JSONL can wrap values in nested structures. Search aliases in
    declared preference order and deliberately never substitute a generic `id`
    for `toposid`.
    """
    direct = _preferred_value(mapping, possible)
    if direct not in (None, "", [], {}) or depth <= 0:
        return direct
    for child in mapping.values():
        children = child if isinstance(child, (list, tuple)) else (child,)
        for nested in children:
            if isinstance(nested, dict):
                found = _deep_preferred_value(nested, possible, depth - 1)
                if found not in (None, "", [], {}):
                    return found
    return None


def _deep_value(value: Any, aliases: tuple[str, ...]) -> Any:
    alias_set = {re.sub(r"[^a-z0-9]", "", alias.lower()) for alias in aliases}
    for key, child in _walk_key_values(value):
        if key in alias_set and child not in (None, "", [], {}):
            return child
    return None


def _external_identifiers(value: dict[str, Any]) -> dict[str, str]:
    strings = " ".join(_strings(value, limit=5000))
    wikipedia_raw = _deep_value(value, ("wikipedia_id", "wikipediaid", "wikipedia", "wikipedia_url", "wikipediaurl"))
    wiki_url = _wikipedia_url_from_value(wikipedia_raw) or _wikipedia_url_from_value(value)
    wikipedia_id = _scalar(wikipedia_raw)
    if not wikipedia_id and wikipedia_raw is not None:
        wikipedia_id = next((candidate.strip() for candidate in _strings(wikipedia_raw) if candidate.strip()), "")
    if wiki_url:
        path = urlparse(wiki_url).path.split("/wiki/", 1)
        if len(path) == 2:
            wikipedia_id = unquote(path[1].split("#", 1)[0]).replace("_", " ")
    if not wikipedia_id:
        match = re.search(r"(?:wikipedia(?:\s+(?:page|id|article))?\s*[:=]\s*)([^,;|]+)", strings, re.I)
        if match:
            wikipedia_id = match.group(1).strip()

    pleiades_raw = _deep_value(value, ("pleiades_id", "pleiadesid", "pleiades", "pleiades_url", "pleiadesurl", "pleiades_uri"))
    pleiades_id = _scalar(pleiades_raw)
    if not pleiades_id and pleiades_raw is not None:
        for candidate in _strings(pleiades_raw):
            nested_match = re.search(r"(?:pleiades\.stoa\.org/places/)?([0-9]+)", candidate, re.I)
            if nested_match:
                pleiades_id = nested_match.group(1)
                break
    match = re.search(r"pleiades(?:\s*(?:id|uri|url))?\s*[:=]\s*(?:https?://pleiades\.stoa\.org/places/)?([0-9]+)", strings, re.I)
    if match:
        pleiades_id = match.group(1)
    else:
        match_url = re.search(r"https?://pleiades\.stoa\.org/places/([0-9]+)", pleiades_id, re.I)
        if match_url:
            pleiades_id = match_url.group(1)

    wikidata_raw = _deep_value(value, ("wikidata_id", "wikidataid", "wikidata", "wikidata_url", "wikidataurl", "wikidata_uri"))
    wikidata_id = _scalar(wikidata_raw)
    if not wikidata_id and wikidata_raw is not None:
        for candidate in _strings(wikidata_raw):
            nested_match = re.search(r"(?:wikidata\.org/(?:entity|wiki)/)?(Q[0-9]+)", candidate, re.I)
            if nested_match:
                wikidata_id = nested_match.group(1).upper()
                break
    match = re.search(r"wikidata(?:\s*(?:id|uri|url))?\s*[:=]\s*(?:https?://(?:www\.)?wikidata\.org/(?:entity|wiki)/)?(Q[0-9]+)", strings, re.I)
    if match:
        wikidata_id = match.group(1).upper()
    else:
        match_url = re.search(r"https?://(?:www\.)?wikidata\.org/(?:entity|wiki)/(Q[0-9]+)", wikidata_id, re.I)
        if match_url:
            wikidata_id = match_url.group(1).upper()

    return {"wikipedia_id": wikipedia_id, "wikipedia_url": wiki_url,
            "pleiades_id": pleiades_id, "wikidata_id": wikidata_id}


def _place_id_from(value: Any) -> str:
    for candidate in _strings(value, limit=200):
        match = re.search(r"topostext\.org/place/([A-Za-z0-9_-]+)", candidate, re.I)
        if match:
            return match.group(1)
        match = re.fullmatch(r"(?:place[:/])?([0-9]{1,12})", candidate.strip(), re.I)
        if match:
            return match.group(1)
    return ""


def _clean_wikipedia_url(value: str) -> str:
    """Trim citation punctuation without removing parentheses in article titles."""
    value = value.rstrip(".,;:")
    while value.endswith(")") and value.count(")") > value.count("("):
        value = value[:-1]
    return value


def _wikipedia_url_from_value(value: Any) -> str:
    for candidate in _strings(value, limit=1000):
        match = _WIKI_URL_RE.search(candidate)
        if match:
            return _clean_wikipedia_url(match.group(0))
    return ""


def _record_from(value: Any, ordinal: int = 0) -> dict[str, Any] | None:
    if isinstance(value, dict):
        # `toposid` is the canonical public identifier. A generic `id` can be
        # a metadata-row/vector-row key and must never generate a ToposText URL.
        raw_toposid = _deep_preferred_value(value, ("toposid", "topostextid", "topostextplaceid"))
        place_id = _place_id_from(raw_toposid) or _scalar(raw_toposid)
        url_value = _deep_preferred_value(value, ("topostexturl", "url", "uri", "link", "ref"))
        if not place_id:
            place_id = _place_id_from(url_value)
        if not url_value:
            url_value = _deep_preferred_value(value, ("meta", "metadataurl", "sourceurl"))

        name_value = _deep_preferred_value(value, ("title", "titles", "name", "names", "placename", "label", "labels", "preferredname", "toponym"))
        desc_value = _deep_preferred_value(value, ("description", "descriptions", "desc", "summary", "summaries", "text", "content", "abstract", "snippet"))
        type_value = _deep_preferred_value(value, ("settlementtype", "placetype", "typeofsettlement", "featuretype", "type", "types", "category", "categories", "kind"))
        lat_value = _deep_preferred_value(value, ("latitude", "lat", "latitudes", "geolatitude", "decimallatitude", "latitude_dd", "y"))
        lon_value = _deep_preferred_value(value, ("longitude", "lon", "lng", "long", "longitudes", "geolongitude", "decimallongitude", "longitude_dd", "x"))
        region_value = _deep_preferred_value(value, ("region", "regions"))
        country_value = _deep_preferred_value(value, ("country", "countries"))
        department_value = _deep_preferred_value(value, ("department", "district", "province"))
        period_value = _deep_preferred_value(value, ("timeperiod", "period", "dateperiod"))
        confidence_value = _deep_preferred_value(value, ("confidence", "coordinateconfidence"))
        modern_value = _deep_preferred_value(value, ("modernlocation", "modernplace", "mod"))

        name = _scalar(name_value)
        description = _scalar(desc_value)
        place_type = _scalar(type_value)
        latitude = _number(lat_value)
        longitude = _number(lon_value)
        supplied_url = _scalar(url_value)
        url = f"https://topostext.org/place/{place_id}" if place_id else supplied_url
        if not place_id:
            place_id = _place_id_from(url)

        if not name:
            # Some metadata dumps contain only a retrieval description instead
            # of a dedicated name. Avoid treating bibliography labels as names.
            match = re.match(r"^\s*([^:]{2,100}):\s+", description)
            if match and match.group(1).strip().casefold() not in {
                "wikipedia", "wikidata id", "pleiades", "trismegistos geo", "dare", "info"
            }:
                name = match.group(1).strip()
            else:
                match = re.match(r"^\s*([A-Z][\w’' -]{1,70}?)\s+(?:is|was|lies|stands|are)\b", description)
                if match:
                    name = match.group(1).strip()
        if not name:
            name = _scalar(modern_value)
        ids = _external_identifiers(value)
        if not name and ids.get("wikipedia_url"):
            title_path = urlparse(ids["wikipedia_url"]).path.split("/wiki/", 1)
            if len(title_path) == 2:
                name = unquote(title_path[1].split("#", 1)[0]).replace("_", " ").strip()
        if not name:
            strings = _strings(value, 30)
            name = next((item for item in strings if item and "topostext.org/place/" not in item
                         and len(item) < 100 and not item.startswith("http")
                         and not re.fullmatch(r"[0-9]+", item.strip()) and "/" not in item
                         and not re.match(r"^(?:Wikipedia|Wikidata\s*ID|Pleiades|Trismegistos|DARE|Info)\s*:", item, re.I)), "")
        if not description:
            description = name
        if name or place_id:
            return {
                "id": place_id or f"metadata-{ordinal}", "toposid": place_id,
                "name": name or f"ToposText place {place_id or ordinal}",
                "title": name or f"ToposText place {place_id or ordinal}",
                "description": description, "type": _scalar(place_type) or "Unclassified place",
                "latitude": latitude, "longitude": longitude,
                "url": f"https://topostext.org/place/{place_id}" if place_id else "",
                "region": _scalar(region_value), "country": _scalar(country_value),
                "department": _scalar(department_value), "time_period": _scalar(period_value),
                "coordinate_confidence": _scalar(confidence_value), "modern_location": _scalar(modern_value),
                **ids, "metadata": value,
            }
    elif isinstance(value, (list, tuple)):
        # Handle per-place sequences without using a generic numeric token as ID.
        place_id = _place_id_from(value)
        strings = [_scalar(item) for item in value if _scalar(item)]
        url = next((item for item in strings if "topostext.org/place/" in item), "")
        generic_prefix = re.compile(r"^(?:Wikipedia|Wikidata\s*ID|Pleiades|Trismegistos(?:\s+Geo)?|DARE|Info)\s*:", re.I)
        name = next((item for item in strings if item != url and not item.startswith("http")
                     and len(item) < 180 and not generic_prefix.match(item)), "")
        ids = _external_identifiers({"metadata": value})
        if not name and ids.get("wikipedia_url"):
            title_path = urlparse(ids["wikipedia_url"]).path.split("/wiki/", 1)
            if len(title_path) == 2:
                name = unquote(title_path[1].split("#", 1)[0]).replace("_", " ").strip()
        numbers = [_number(item) for item in value if _number(item) is not None]
        if name or place_id:
            return {
                "id": place_id or f"metadata-{ordinal}", "toposid": place_id,
                "name": name or f"ToposText place {place_id or ordinal}",
                "title": name or f"ToposText place {place_id or ordinal}",
                "description": name, "type": "Unclassified place",
                "latitude": numbers[0] if len(numbers) >= 2 else None,
                "longitude": numbers[1] if len(numbers) >= 2 else None,
                "url": f"https://topostext.org/place/{place_id}" if place_id else "",
                "region": "", "country": "", "department": "", "time_period": "",
                "coordinate_confidence": "", "modern_location": "", **ids, "metadata": value,
            }
    return None


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            m = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", "."))
            if not m:
                return None
            value = m.group(0)
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def normalize_metadata(raw: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(value: Any, ordinal: int) -> None:
        rec = _record_from(value, ordinal)
        if rec:
            key = rec["url"] or rec["id"]
            if key not in seen:
                seen.add(key)
                records.append(rec)

    if isinstance(raw, dict) and isinstance(raw.get("ids"), (list, tuple)):
        ids = raw.get("ids") or []
        count = len(ids)
        for i in range(count):
            row: dict[str, Any] = {"ids": ids[i]}
            for key, value in raw.items():
                if key == "ids":
                    continue
                if isinstance(value, (list, tuple)) and len(value) == count:
                    row[key] = value[i]
            meta_url = _scalar(row.get("meta"))
            if "topostext.org/place/" in meta_url:
                row.setdefault("url", meta_url)
            if not _find_key(row, ("name", "names", "placename", "title", "titles", "label", "labels", "preferredname", "toponym")):
                row.setdefault("description", _scalar(row.get("desc")))
            add(row, i)
        # Include nested dict records too if the column-oriented layout yielded too little.
        if len(records) < max(1, count // 4):
            for key, value in raw.items():
                if key in ("ids",) or not isinstance(value, (list, tuple)):
                    continue
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        add(item, i)
    elif isinstance(raw, (list, tuple)):
        for i, item in enumerate(raw):
            add(item, i)
    elif isinstance(raw, dict):
        # A common shape is {place_url: {name: ..., latitude: ..., ...}}.
        for i, (key, value) in enumerate(raw.items()):
            if isinstance(value, dict):
                obj = dict(value)
                obj.setdefault("url", key)
                add(obj, i)
            elif isinstance(value, (list, tuple)):
                add({"url": key, "metadata": value, "name": next((s for s in _strings(value) if not s.startswith("http")), "")}, i)

    return records


def normalize_jsonl_metadata(rows: list[Any]) -> list[dict[str, Any]]:
    """Normalize the notebook's v2 JSONL, preserving exact FAISS row alignment."""
    records: list[dict[str, Any]] = []
    for ordinal, value in enumerate(rows):
        if not isinstance(value, dict):
            raise ValueError(f"ToposText metadata row {ordinal + 1} is not a JSON object.")

        # The notebook's metadata key is specifically `toposid`; never use a
        # generic `id` (which may identify a vector/index row) as the place ID.
        raw_toposid = _deep_preferred_value(value, ("toposid", "topostextid", "topostextplaceid"))
        place_id = _place_id_from(raw_toposid) or _scalar(raw_toposid)
        if place_id and not re.fullmatch(r"[A-Za-z0-9_-]+", place_id):
            place_id = _place_id_from(place_id)
        if not place_id:
            # A real ToposText place URL is an acceptable identifier source;
            # a generic metadata `id` is not.
            explicit_url = _deep_preferred_value(value, ("topostexturl", "url", "uri", "link", "ref", "meta", "metadataurl", "sourceurl"))
            place_id = _place_id_from(explicit_url)
        title = _scalar(_deep_preferred_value(value, ("title", "titles", "name", "names", "placename", "preferredname")))
        description = _scalar(_deep_preferred_value(value, ("description", "descriptions", "desc", "summary", "abstract", "snippet")))

        # Preserve all raw data while normalizing fields used by the app.
        lat_raw = _deep_value(value, ("latitude", "lat", "geolatitude"))
        lon_raw = _deep_value(value, ("longitude", "lon", "lng", "long", "geolongitude"))
        latitude = _number(lat_raw)
        longitude = _number(lon_raw)
        # Support GeoJSON-style [longitude, latitude] if explicit fields aren't present.
        if latitude is None or longitude is None:
            coords = _deep_value(value, ("coordinates", "coordinate"))
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                if longitude is None:
                    longitude = _number(coords[0])
                if latitude is None:
                    latitude = _number(coords[1])
        place_type = _scalar(_deep_preferred_value(value, ("settlement_type", "settlementtype", "place_type", "placetype", "featuretype", "type", "category")))
        region = _scalar(_deep_preferred_value(value, ("region", "regions")))
        country = _scalar(_deep_preferred_value(value, ("country", "countries")))
        department = _scalar(_deep_preferred_value(value, ("department", "district", "province")))
        modern_location = _scalar(_deep_preferred_value(value, ("modern_location", "modernlocation", "modernplace")))
        confidence = _scalar(_deep_preferred_value(value, ("coordinate_confidence", "coordinateconfidence")))
        ids = _external_identifiers(value)
        url = f"https://topostext.org/place/{place_id}" if place_id else ""

        records.append({
            "id": place_id or f"metadata-{ordinal}",
            "toposid": place_id,
            "name": title or f"ToposText place {place_id or ordinal}",
            "title": title or f"ToposText place {place_id or ordinal}",
            "description": description,
            "type": place_type or "Unclassified place",
            "latitude": latitude,
            "longitude": longitude,
            "url": url,
            "region": region,
            "country": country,
            "department": department,
            "time_period": _scalar(_deep_value(value, ("time_period", "timeperiod", "period"))),
            "coordinate_confidence": confidence,
            "modern_location": modern_location,
            **ids,
            "metadata": value,
        })
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in ToposText metadata line {line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"ToposText metadata line {line_number} is not a JSON object.")
            rows.append(value)
    return rows


def load_gazetteer(on_status=None) -> tuple[Any, list[dict[str, Any]], str | None]:
    global _loaded_key, _index, _records, _raw_metadata
    with _lock:
        key = (str(INDEX_PATH), str(METADATA_PATH), INDEX_PATH.stat().st_mtime if INDEX_PATH.exists() else 0.0,
               METADATA_PATH.stat().st_mtime if METADATA_PATH.exists() else 0.0)
        if _loaded_key == key:
            return _index, _records, None
        warning = None
        if not INDEX_PATH.is_file() or not METADATA_PATH.is_file():
            missing = [str(path) for path in (INDEX_PATH, METADATA_PATH) if not path.is_file()]
            warning = "Required local ToposText asset(s) are missing: " + ", ".join(missing)
        idx = None
        raw = None
        if INDEX_PATH.exists() and faiss is not None:
            try:
                idx = faiss.read_index(str(INDEX_PATH))
            except Exception as exc:
                warning = f"Could not read FAISS index: {exc}. Semantic retrieval cannot continue."
        if INDEX_PATH.exists() and faiss is None:
            warning = "FAISS is not installed in this Python environment. Semantic retrieval cannot continue."
        if METADATA_PATH.exists():
            try:
                raw = _read_jsonl(METADATA_PATH)
            except Exception as exc:
                warning = f"Could not read ToposText JSONL metadata: {exc}. Semantic retrieval cannot continue."
        records = normalize_jsonl_metadata(raw) if raw is not None else []
        _index, _records, _raw_metadata = idx, records, raw
        _loaded_key = key if warning is None else None
        return idx, records, warning



def embedding_candidates(mention: str, context: str, index: Any, records: list[dict[str, Any]],
                         embedder, on_status=None, limit: int = 10) -> list[dict[str, Any]]:
    """Follow the notebook's EmbeddingGemma SearchQuery -> FAISS top-10 stage."""
    if faiss is None:
        raise RuntimeError("faiss-cpu is not installed; semantic retrieval cannot run.")
    if index is None:
        raise RuntimeError("The ToposText FAISS index is unavailable.")
    if not records:
        raise RuntimeError("The ToposText JSONL metadata is empty.")
    if int(index.ntotal) != len(records):
        raise RuntimeError(
            f"FAISS index has {index.ntotal} rows, but metadata has {len(records)} rows. "
            "The notebook requires index/metadata row alignment."
        )

    query_text = str(context or mention).strip()
    if not query_text:
        raise RuntimeError(f"Empty embedding query for mention: {mention}")
    query_emb = np.asarray(embedder(query_text, on_status=on_status), dtype=np.float32).reshape(1, -1)
    if query_emb.shape[1] != int(index.d):
        raise ValueError(
            f"EmbeddingGemma produced {query_emb.shape[1]} dimensions, but the ToposText FAISS index expects {index.d}."
        )

    # Exactly the notebook's `_, indices = index.search(query_emb, k=10)`.
    _, indices = index.search(query_emb, 10)
    result_ids = []
    for idx in indices[0].tolist():
        if idx < 0 or idx >= len(records):
            raise RuntimeError(f"FAISS returned invalid metadata row index {idx}.")
        result_ids.append(records[idx].get("toposid"))

    # Exactly the notebook's reranker preparation: for every ToposText ID,
    # scan metadata from the start and take the first row with that ID.
    candidates: list[dict[str, Any]] = []
    for topos_id in result_ids:
        if topos_id in (None, ""):
            raise RuntimeError("A retrieved metadata row is missing its `toposid` field.")
        found = None
        for record in records:
            if str(record.get("toposid")) == str(topos_id):
                found = dict(record)
                break
        if found is None:
            raise RuntimeError(f"FAISS returned ToposText ID {topos_id}, but it was not found in metadata.")
        found["id"] = str(topos_id)
        found["toposid"] = str(topos_id)
        found["url"] = f"https://topostext.org/place/{topos_id}"
        candidates.append(found)
    return candidates


def lookup_place_by_id(place_id: Any, records: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """Resolve the disambiguated ToposText ID against the authoritative metadata rows."""
    if place_id is None:
        return None
    raw_id = str(place_id).strip()
    target = _place_id_from(raw_id) or raw_id
    if not target:
        return None
    if records is None:
        _, records, _ = load_gazetteer()
    for record in records:
        record_id = str(record.get("toposid") or "")
        if record_id == target:
            return dict(record)
    return None


