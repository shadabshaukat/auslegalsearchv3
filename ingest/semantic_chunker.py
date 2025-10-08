"""
Semantic, token-aware chunking utilities for AUSLegalSearch v3 (beta dataset).

Design goals:
- Modern, tokenizer-agnostic approximation of token counts for fast, dependency-light operation.
- Preserve semantic boundaries: headings -> paragraphs -> sentences, before hard-splitting.
- Configurable target token budget and overlap; defaults aligned with current industry practice (e.g., ~512 tokens, 10–20% overlap).
- Produce rich per-chunk metadata (e.g., section titles, indices) to improve retrieval and citation-grounded answers.
- Strict guarantees: chunk size never exceeds max_tokens after final splitting.

This module is standalone (no DB), focused purely on chunking logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional

# Defaults tuned to common LLM context windows and RAG best-practices
DEFAULT_TARGET_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 64
DEFAULT_MAX_TOKENS = 640  # safety upper bound


# ---- Tokenization and basic text utilities ----

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize_rough(text: str) -> List[str]:
    """
    Lightweight tokenization: split into words and punctuation.
    Good proxy for transformer/BPE token counts without heavy dependencies.
    """
    return _WORD_RE.findall(text or "")

def count_tokens(text: str) -> int:
    return len(tokenize_rough(text))


# ---- Structural segmentation helpers ----

# Headings: Roman numerals / numbered / uppercase words / legal "Section" markers
_HEADING_LINE_RE = re.compile(
    r"""
    ^\s*(
        (?:[IVXLCDM]+\.?)                                  # Roman numerals
        | (?:\d+(?:\.\d+)*\.?)                             # 1. or 1.2.3.
        | (?:[A-Z][A-Z0-9 ]{2,})                           # UPPERCASE headings
        | (?:Section\s+\d+[A-Za-z\-]*|s\.\s*\d+[A-Za-z\-]*) # Section 12 / s. 12
    )\b
    """,
    re.VERBOSE,
)

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

def split_into_sentences(text: str) -> List[str]:
    """
    Conservative sentence splitter. Avoids over-splitting abbreviations by requiring next token capital/opening bracket.
    """
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out

def split_into_paragraphs(text: str) -> List[str]:
    """
    Split on blank lines and normalize whitespace.
    """
    paras = [p.strip() for p in re.split(r"(?:\r?\n){2,}", text or "") if p.strip()]
    return paras

def split_into_blocks(text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Extract semantic blocks from text: a heading (if present) and its associated paragraph(s).
    Returns a list of (block_text, heading_title_or_none).
    """
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    blocks: List[Tuple[str, Optional[str]]] = []

    current_heading: Optional[str] = None
    current_accum: List[str] = []

    def flush_block():
        nonlocal blocks, current_accum, current_heading
        if current_accum:
            block_text = "\n".join(current_accum).strip()
            if block_text:
                blocks.append((block_text, current_heading))
        current_accum = []

    for ln in lines:
        if ln.strip() == "":
            # paragraph boundary
            if current_accum:
                current_accum.append("")  # keep a blank to mark paragraph split
            continue

        if _HEADING_LINE_RE.match(ln.strip()):
            # heading encountered -> flush previous
            flush_block()
            current_heading = ln.strip()
            # Do not add heading itself to the block text; keep it as metadata
            continue

        current_accum.append(ln)

    flush_block()
    # Post-process: split any block on consecutive blanks to form tighter paragraph blocks
    refined: List[Tuple[str, Optional[str]]] = []
    for text_block, heading in blocks:
        parts = [p.strip() for p in re.split(r"(?:\r?\n){2,}", text_block) if p.strip()]
        for p in parts:
            refined.append((p, heading))
    return refined


# ---- Chunking core ----

@dataclass
class ChunkingConfig:
    target_tokens: int = DEFAULT_TARGET_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    max_tokens: int = DEFAULT_MAX_TOKENS
    min_sentence_tokens: int = 6   # drop ultra-short sentences unless necessary
    min_chunk_tokens: int = 40     # drop tiny chunks that don't add value

def _merge_sentences_to_chunks(
    sentences: List[str],
    cfg: ChunkingConfig
) -> List[List[str]]:
    """
    Merge sentences into chunks, aiming for target_tokens and allowing overlap.
    Always respects max_tokens by backoff-splitting long sentences if needed.
    Returns list of chunks as lists of sentences.
    """
    chunks: List[List[str]] = []
    current: List[str] = []
    current_tokens = 0

    def sentence_tokens(s: str) -> int:
        return count_tokens(s)

    i = 0
    while i < len(sentences):
        sent = sentences[i].strip()
        if not sent:
            i += 1
            continue
        stoks = sentence_tokens(sent)
        # If a single sentence exceeds max_tokens, hard-split it
        if stoks > cfg.max_tokens:
            # Hard split by approximate token size using whitespace slicing
            words = sent.split()
            piece_size = max(cfg.max_tokens - 5, cfg.min_chunk_tokens)
            acc = []
            w_acc = []
            for w in words:
                w_acc.append(w)
                if count_tokens(" ".join(w_acc)) >= piece_size:
                    acc.append(" ".join(w_acc))
                    w_acc = []
            if w_acc:
                acc.append(" ".join(w_acc))
            # Flush current first
            if current:
                chunks.append(current)
                current = []
                current_tokens = 0
            # Each hard piece is its own chunk (no overlap)
            for piece in acc:
                chunks.append([piece])
            i += 1
            continue

        if current_tokens + stoks <= cfg.target_tokens or not current:
            current.append(sent)
            current_tokens += stoks
            i += 1
        else:
            # Emit current as a chunk
            chunks.append(current)
            # Prepare overlap for next chunk
            if cfg.overlap_tokens > 0:
                overlap: List[str] = []
                acc = 0
                # Walk backwards adding sentences until reach overlap_tokens
                for s in reversed(current):
                    t = sentence_tokens(s)
                    if acc + t <= cfg.overlap_tokens or not overlap:
                        overlap.append(s)
                        acc += t
                    else:
                        break
                overlap = list(reversed(overlap))
            else:
                overlap = []
            current = overlap.copy()
            current_tokens = sum(sentence_tokens(s) for s in current)

    if current:
        chunks.append(current)

    # Filter tiny chunks
    filtered: List[List[str]] = []
    for ch in chunks:
        toks = sum(count_tokens(s) for s in ch)
        if toks >= cfg.min_chunk_tokens or (len(chunks) == 1 and toks > 0):
            filtered.append(ch)
    return filtered

def chunk_text_semantic(
    text: str,
    cfg: Optional[ChunkingConfig] = None,
    section_title: Optional[str] = None,
    section_idx: Optional[int] = None
) -> List[Dict]:
    """
    Chunk a single block of text into token-aware, semantically bounded chunks.
    Returns a list of dicts with 'text' and 'chunk_metadata' keys.
    """
    cfg = cfg or ChunkingConfig()
    sentences = split_into_sentences(text)
    if not sentences:
        # Fallback: treat entire text as one sentence
        sentences = [text.strip()] if text and text.strip() else []

    # Remove ultra-short sentences (except if that would empty the set)
    if len(sentences) > 1:
        sentences = [s for s in sentences if count_tokens(s) >= cfg.min_sentence_tokens] or sentences

    sent_chunks = _merge_sentences_to_chunks(sentences, cfg)
    out: List[Dict] = []
    for idx, ch_sents in enumerate(sent_chunks):
        ch_text = " ".join(ch_sents).strip()
        if not ch_text:
            continue
        out.append({
            "text": ch_text,
            "chunk_metadata": {
                "section_title": section_title,
                "section_idx": section_idx,
                "chunk_idx": idx,
                "tokens_est": count_tokens(ch_text),
            }
        })
    return out


# ---- Document-level chunking ----

def detect_doc_type(meta: Optional[Dict], text: str) -> str:
    """
    Heuristic doc type detection: 'case', 'legislation', 'journal', or 'txt'.
    """
    meta = meta or {}
    t = (meta.get("type") or "").lower()
    if t in ("case", "legislation", "journal"):
        return t
    sample = (text or "")[:1000].lower()
    if " v " in sample or " appellant" in sample or " respondent" in sample:
        return "case"
    if " section " in sample or re.search(r"\bs\.\s*\d+", sample):
        return "legislation"
    if re.search(r"^(?:[ivxlcdm]+\.)|^\d+\.", (text or "").strip().lower(), re.M):
        return "journal"
    return "txt"

def chunk_document_semantic(
    doc_text: str,
    base_meta: Optional[Dict] = None,
    cfg: Optional[ChunkingConfig] = None
) -> List[Dict]:
    """
    Full semantic chunking pipeline:
    - Split into blocks keyed by headings (if any)
    - Chunk each block with token-aware logic
    - Attach per-chunk metadata with section titles and indices
    """
    cfg = cfg or ChunkingConfig()
    blocks = split_into_blocks(doc_text)
    chunks: List[Dict] = []
    sec_counter = 0
    for block_text, heading in blocks:
        block_chunks = chunk_text_semantic(block_text, cfg, section_title=heading, section_idx=sec_counter)
        for bc in block_chunks:
            md = dict(base_meta or {})
            cm = dict(bc.get("chunk_metadata") or {})
            md.update(cm)
            chunks.append({
                "text": bc["text"],
                "chunk_metadata": md
            })
        sec_counter += 1

    # Fallback: if no blocks produced anything, chunk whole doc
    if not chunks:
        for bc in chunk_text_semantic(doc_text, cfg, section_title=None, section_idx=None):
            md = dict(base_meta or {})
            cm = dict(bc.get("chunk_metadata") or {})
            md.update(cm)
            chunks.append({
                "text": bc["text"],
                "chunk_metadata": md
            })
    return chunks
