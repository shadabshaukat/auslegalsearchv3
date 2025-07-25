"""
Loader module for the 'ingest' component of auslegalsearchv2/v3.

- Walks directories/files for ingestion
- Detects and parses legal document formats (.txt, .html), legislation via /au/legis/ path, and law journals
- Chunks documents for embedding
- Legal-aware chunking splits on sections, clauses, parts/divisions; legislation splits on legislative sections; law journals split on headings
- Designed for extensibility to PDF, DOCX, etc.
- ENHANCED: Extracts YAML-style metadata block from top of .txt files if present.
- NEW: Extracts and chunks legislation files with deep legislative metadata/sections
- NEW: Extracts and chunks law journal files with metadata and section headings
"""

import os
import re
import json
from pathlib import Path
from typing import Iterator, Dict, Any, List

from bs4 import BeautifulSoup
import numpy as np

SUPPORTED_EXTS = {'.txt', '.html'}

def walk_legal_files(root_dirs: List[str]) -> Iterator[str]:
    """Yield supported legal document filepaths (recursively)."""
    for root_dir in root_dirs:
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext in SUPPORTED_EXTS:
                    yield os.path.join(dirpath, fname)

def extract_metadata_block(text: str) -> (dict, str):
    """
    Extract a metadata block in the format:
    -----------------------------------
    key: value
    key2: value2
    -----------------------------------
    Returns (metadata_dict, body_text)
    """
    lines = text.lstrip().splitlines()
    chunk_metadata = {}
    if len(lines) >= 3 and lines[0].startswith("-") and lines[1].find(":") != -1:
        # Find header start and end (lines of dashes)
        try:
            start_idx = 0
            while start_idx < len(lines) and not lines[start_idx].startswith("-"):
                start_idx += 1
            end_idx = start_idx + 1
            while end_idx < len(lines) and not lines[end_idx].startswith("-"):
                end_idx += 1
            # Pull key: value pairs
            for l in lines[start_idx+1:end_idx]:
                if ":" in l:
                    k, v = l.split(":", 1)
                    k, v = k.strip(), v.strip()
                    try:
                        v_eval = eval(v)
                        chunk_metadata[k] = v_eval
                    except:
                        chunk_metadata[k] = v
            # Put remaining text lines after end_idx
            body_text = "\n".join(lines[end_idx+1:]).strip()
            return chunk_metadata, body_text
        except Exception:
            return {}, text
    return {}, text

def is_legislation_file(filepath: str) -> bool:
    # Use POSIX path for robust matching on all OS
    relpath = filepath.replace("\\", "/")
    # Legislation are any .txt files under /au/legis/
    # Adjust this match as necessary for your structure.
    return "/au/legis/" in relpath or "/legis/" in relpath

def is_journal_file(filepath: str) -> bool:
    # Law journals are typically under /journals/ and have .txt extension
    relpath = filepath.replace("\\", "/")
    return "/journals/" in relpath and filepath.endswith(".txt")

def to_title_case(s):
    # Split the string into words
    words = s.split()
    words = [word if not word else
             (f"({word[1].upper()}{word[2:].lower()}" if word.startswith('(') and len(word) > 2
              else word[0].upper() + word[1:].lower())
             for word in words]
    return ' '.join(words)

def generate_doc_header(metadata):
    res = "-----------------------------------\n"
    for key, value in metadata.items():
        if isinstance(value, str):
            res += f"{key}: {value}\n"
        elif isinstance(value, list):
            res += f"{key}: {json.dumps(value)}\n"
    res += "-----------------------------------\n"
    return res

def parse_alt_legislation(filepath, legislation_text):
    legislation_text = legislation_text.replace('\u00A0', ' ')
    lines = legislation_text.split('\n')
    line_num = 3
    title = lines[line_num].strip()
    line_num += 1
    while line_num < len(lines) and lines[line_num]:
        title += " " + lines[line_num].strip()
        line_num += 1

    year = re.search(r'\d{4}$', title).group(0)
    sections = []
    started_sections = False
    i = line_num
    while i < len(lines):
        if re.match(r'^\d', lines[i]) and (i == 0 or not lines[i-1]):
            started_sections = True
            sec = {}
            first_space = lines[i].index(" ") if " " in lines[i] else len(lines[i])
            sec["section"] = lines[i][:first_space].strip()
            sec["title"] = lines[i][first_space:].strip()
            i += 1
            while i < len(lines) and lines[i]:
                sec["title"] += " " + lines[i].strip()
                i += 1
            i += 1
            sec["text"] = ""
            while i < len(lines) and (not lines[i] or re.match(r'^\s', lines[i])):
                if not lines[i]:
                    sec["text"] += "\n"
                else:
                    sec["text"] += " " + lines[i].strip()
                i += 1
            sections.append(sec)
            i -= 1
        if lines[i].startswith(f"Notes to the {title}") or (lines[i].startswith("Schedule") and started_sections):
            break
        i += 1

    parsed = {
        'name': to_title_case(title),
        'year': year,
        'date': f"{year}-01-01 00:00:00",
        'type': 'legislation',
        'jurisdiction': filepath.split(os.sep)[0] if os.sep in filepath else '',
        'database': filepath.split(os.sep)[1] if os.sep in filepath else ''
    }
    relative_url = filepath.replace('\\', '/')
    relative_url = relative_url[:-4] + '/'  # Remove .txt and add slash
    BASE_URL = os.environ.get("BASE_URL", "/")
    parsed['url'] = BASE_URL + relative_url

    return {"meta": parsed, "sections": sections}

def parse_legislation(filepath, legislation_text):
    if legislation_text.startswith('\n[pic]'):
        return parse_alt_legislation(filepath, legislation_text)
    title = legislation_text.split('\n')[0].strip()
    year = re.search(r'\d{4}$', title).group(0)
    section_break = f"{title}\n-"
    legislation_sections = legislation_text.split(section_break)
    sections = []
    for section in legislation_sections:
        section = section.strip()
        if section.startswith("SECT"):
            sec = {}
            end_line_one = section.index('\n')
            sec["section"] = section[5:end_line_one].strip()
            next_newline = section.index('\n', end_line_one + 1)
            sec["title"] = section[end_line_one + 1:next_newline].strip()
            sec["text"] = section[next_newline + 1:].strip()
            sections.append(sec)

    parsed = {
        'name': to_title_case(title),
        'year': year,
        'date': f"{year}-01-01 00:00:00",
        'type': 'legislation',
        'jurisdiction': filepath.split(os.sep)[0] if os.sep in filepath else '',
        'database': filepath.split(os.sep)[1] if os.sep in filepath else ''
    }
    relative_url = filepath.replace('\\', '/')
    relative_url = relative_url[:-4] + '/'  # Remove .txt and add slash
    BASE_URL = os.environ.get("BASE_URL", "/")
    parsed['url'] = BASE_URL + relative_url
    return {"meta": parsed, "sections": sections}

def generate_section_text(section):
    return f"""-----------------------------------
section: {section['section']}
title: {section['title']}
-----------------------------------
{section['text']}
"""

def generate_output_text(parsed):
    res = generate_doc_header(parsed['meta'])
    res += '\n\n'
    for section in parsed['sections']:
        res += generate_section_text(section)
    return res

def parse_legislation_file(filepath: str) -> Dict[str, Any]:
    """
    Reads and parses a legislation .txt file, extracting legislative metadata and sections.
    Returns a dict: {"text": full_text, "source": filepath, "format": "legislation", "chunk_metadata": {meta dict}, "sections": [per-section dicts]}
    Handles catch-all logic for special acts (e.g., Constitution) that do not follow normal section patterns.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            legislation_text = f.read()
        parsed = parse_legislation(filepath, legislation_text)
        meta = parsed.get('meta', {})
        sections = parsed.get('sections', [])
        # Always generate AustLII URL for metadata if possible
        relpath = filepath.replace("\\", "/")
        austlii_url = None
        m = re.search(r"(/au/legis/.+?)/([^/]+)\.txt$", relpath)
        if m:
            rel_url_part = f"{m.group(1)}/{m.group(2)}"
            austlii_url = f"https://www.austlii.edu.au/cgi-bin/viewdb{rel_url_part}/"
        if austlii_url:
            meta['url'] = austlii_url
        # Handling for edge/special acts
        if not sections or all(not sec.get("text") for sec in sections):
            # Use entire text as single chunk if no parsable sections
            try:
                # Try to extract name/title and year from top lines
                lines = legislation_text.strip().split("\n")
                name = ""
                year = ""
                for line in lines:
                    m = re.match(r".*Act\s+(\d{4})", line)
                    if m:
                        name = line.strip()
                        year = m.group(1)
                        break
                if not name and lines:
                    name = lines[0].strip()
                if not year:
                    m = re.search(r"\b(\d{4})\b", legislation_text)
                    if m:
                        year = m.group(1)
                meta.update({
                    "name": name,
                    "year": year,
                    "type": "legislation",
                })
            except Exception:
                pass
            meta['url'] = austlii_url
            return {
                "text": legislation_text,
                "source": filepath,
                "format": "legislation",
                "chunk_metadata": meta,
                "sections": []
            }
        # Collate all section texts for a single 'text' field (for DB compatibility)
        full_text = generate_output_text(parsed)
        return {
            "text": full_text,
            "source": filepath,
            "format": "legislation",
            "chunk_metadata": meta,
            "sections": sections
        }
    except Exception as e:
        print(f"Error reading legislation {filepath}: {e}")
        return {}

def parse_journal_file(filepath: str) -> Dict[str, Any]:
    """Parse law journal .txt file with metadata header."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        meta, body = extract_metadata_block(text)
        return {"text": body, "source": filepath, "format": "journal", "chunk_metadata": meta or None}
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

def parse_txt(filepath: str) -> Dict[str, Any]:
    """Parse plain text legal file with optional metadata. If legislation, parses as legislation and returns full metadata."""
    # Delegate to legislation parser if this is a legislation file
    if is_legislation_file(filepath):
        return parse_legislation_file(filepath)
    if is_journal_file(filepath):
        return parse_journal_file(filepath)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        meta, body = extract_metadata_block(text)
        return {"text": body, "source": filepath, "format": "txt", "chunk_metadata": meta or None}
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

def parse_html(filepath: str) -> Dict[str, Any]:
    """Extract visible text from legal HTML file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        raw_text = soup.get_text(separator='\n', strip=True)
        return {"text": raw_text, "source": filepath, "format": "html", "chunk_metadata": None}
    except Exception as e:
        print(f"Error parsing HTML {filepath}: {e}")
        return {}

def chunk_legal_by_section(doc: Dict[str, Any], min_chunk_length: int = 150) -> List[Dict[str, Any]]:
    """
    Chunk legal docs by sections/clauses/parts, with fallback to overlap if needed.

    Splits on patterns like:
    Section 12, Sec. 13, Clause 5, PART III, Division 1, Article 7, etc
    For legislation, splits by provided sections if present.
    For law journals, splits by major headings (I, II, A, B, etc.).
    """
    # Prioritize chunking by explicit legislation sections if available
    if doc.get("format") == "legislation" and doc.get("sections"):
        meta = doc.get("chunk_metadata", {})
        srcpath = doc.get("source", "")
        relpath = srcpath.replace("\\", "/")
        austlii_url = None
        # Extract /au/legis/.../<file_root> as group(1)/group(2)
        m = re.search(r"(/au/legis/.+?)/([^/]+)\.txt$", relpath)
        if m:
            rel_url_part = f"{m.group(1)}/{m.group(2)}"
            austlii_url = f"https://www.austlii.edu.au/cgi-bin/viewdb{rel_url_part}/"
        chunks = []
        for idx, section in enumerate(doc["sections"]):
            # Attach AustLII url if this is legislation
            chunk_url = austlii_url if austlii_url else None
            cm = {
                **meta,
                "section": section.get("section"),
                "section_title": section.get("title"),
                "section_idx": idx
            }
            if chunk_url:
                cm["url"] = chunk_url
            chunk_info = {
                "text": section["text"].strip(),
                "source": doc.get("source"),
                "format": "legislation",
                "chunk_metadata": cm,
                "legal_section": section.get("title", section.get("section", f"Section_{idx}"))
            }
            chunks.append(chunk_info)
        return chunks

    if doc.get("format") == "journal":
        meta = doc.get("chunk_metadata", {})
        chunks = []
        lines = doc.get("text", "").splitlines()
        buffer = ""
        last_heading = ""
        for line in lines:
            line = line.strip()
            if re.match(r"^[IVXLCDM]+\.?\s|^[A-Z]+\.?$", line):
                if buffer.strip() and len(buffer.strip()) >= min_chunk_length:
                    chunks.append({"text": buffer.strip(), "source": doc.get("source"), "format": "journal", "chunk_metadata": meta, "legal_section": last_heading})
                last_heading = line
                buffer = last_heading + " "
            else:
                buffer += line + " "
        if buffer.strip() and len(buffer.strip()) >= min_chunk_length:
            chunks.append({"text": buffer.strip(), "source": doc.get("source"), "format": "journal", "chunk_metadata": meta, "legal_section": last_heading})
        return chunks

    text = doc.get("text", "")
    meta = {k: v for k, v in doc.items() if k != "text"}
    chunk_metadata = doc.get("chunk_metadata")

    pattern = re.compile(
        r"(?=\n?(Section|Sec\.?|Clause|Part|Division|DIVISION|ARTICLE|Article|Regulation|CHAPTER|Chapter)\s+[\w\d\-IVXLCDM]+[\.\: ]?)",
        re.IGNORECASE,
    )
    parts = [s.strip() for s in pattern.split(text) if s.strip()]

    chunks = []
    buffer = ""
    last_heading = ""
    for p in parts:
        if re.match(r"^(Section|Sec\.?|Clause|Part|Division|ARTICLE|Regulation|CHAPTER)\b", p, re.IGNORECASE):
            if buffer.strip():
                if len(buffer.strip()) >= min_chunk_length:
                    chunks.append({"text": buffer.strip(), **meta, "legal_section": last_heading or "Preamble", "chunk_metadata": chunk_metadata})
            last_heading = p.strip()
            buffer = last_heading + " "
        else:
            buffer += p.strip() + " "
    if buffer.strip() and len(buffer.strip()) >= min_chunk_length:
        chunks.append({"text": buffer.strip(), **meta, "legal_section": last_heading or "End", "chunk_metadata": chunk_metadata})
    return chunks

def chunk_document(doc: Dict[str, Any], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split large documents into intelligently chunked segments:
    - If legal section headers are detected, chunk by section
    - Else, fallback to fixed-size overlapping text chunks.
    Returns: List of chunk dicts (text, meta)
    """
    by_section = chunk_legal_by_section(doc)
    if by_section and len(by_section) > 2:
        return by_section

    text = doc.get("text", "")
    meta = {k: v for k, v in doc.items() if k != "text"}
    chunk_metadata = doc.get("chunk_metadata")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append({"text": chunk_text, **meta, "start_pos": start, "end_pos": end, "chunk_metadata": chunk_metadata})
        if end == len(text):
            break
        start = end - overlap
    return chunks

def embed_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Embed a single chunk of text using the Embedder."""
    try:
        embedding = Embedder().embed([chunk["text"]])[0]
        # Convert numpy float32 array to list for JSON compatibility
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()  # list[float]
        chunk["embedding"] = embedding
        return chunk
    except Exception as e:
        print(f"Embedding error for chunk {chunk.get('source', '(unknown)')} : {e}")
        return {**chunk, "embedding": None}

# TODO: Extend with parse_pdf, parse_docx as needed
