import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(end - overlap, start + 1)

    return chunks


def hashed_vector(text: str, dimensions: int) -> List[float]:
    vec = [0.0] * dimensions
    tokens = TOKEN_RE.findall(text.lower())

    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [round(v / norm, 6) for v in vec]

    return vec


def vectorize_pdf(pdf_path: Path, output_root: Path, chunk_size: int, overlap: int, dimensions: int) -> Dict[str, int]:
    reader = PdfReader(str(pdf_path))
    doc_dir = output_root / pdf_path.stem
    doc_dir.mkdir(parents=True, exist_ok=True)

    extracted_pages: List[str] = []
    chunks_path = doc_dir / "chunks.jsonl"
    text_path = doc_dir / "document.txt"

    chunk_count = 0

    with chunks_path.open("w", encoding="utf-8") as chunk_file:
        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = normalize_text(page.extract_text() or "")
            extracted_pages.append(f"--- PAGE {page_idx} ---\n{page_text}\n")

            page_chunks = split_text(page_text, chunk_size=chunk_size, overlap=overlap)
            for local_chunk_idx, chunk_text in enumerate(page_chunks, start=1):
                chunk_count += 1
                record = {
                    "id": f"{pdf_path.stem}_p{page_idx}_c{local_chunk_idx}",
                    "source_file": pdf_path.name,
                    "page": page_idx,
                    "chunk_in_page": local_chunk_idx,
                    "text": chunk_text,
                    "vector": hashed_vector(chunk_text, dimensions=dimensions),
                }
                chunk_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    text_path.write_text("\n".join(extracted_pages), encoding="utf-8")

    return {
        "pages": len(reader.pages),
        "chunks": chunk_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorize reference PDFs into IA-friendly JSONL chunks.")
    parser.add_argument("--input-dir", type=Path, default=Path("../referencia"), help="Directory with source PDFs.")
    parser.add_argument("--output-dir", type=Path, default=Path("../referencias_vetorizadas"), help="Directory for vectorized output.")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=200, help="Chunk overlap in characters.")
    parser.add_argument("--dimensions", type=int, default=512, help="Vector dimension size.")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {input_dir}")

    summary = []
    aggregate_path = output_dir / "all_chunks.jsonl"

    with aggregate_path.open("w", encoding="utf-8") as aggregate_file:
        for pdf_path in pdf_files:
            stats = vectorize_pdf(
                pdf_path=pdf_path,
                output_root=output_dir,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                dimensions=args.dimensions,
            )
            summary.append({"file": pdf_path.name, **stats})

            doc_chunks = output_dir / pdf_path.stem / "chunks.jsonl"
            aggregate_file.write(doc_chunks.read_text(encoding="utf-8"))

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = (
        "# Referencias Vetorizadas\n\n"
        "Este diretório contém os PDFs da pasta 'referencia' em um formato amigável para consumo por IA.\n\n"
        "## Estrutura\n"
        "- all_chunks.jsonl: todos os chunks vetorizados em um único arquivo.\n"
        "- summary.json: resumo por PDF (páginas e chunks).\n"
        "- <nome-do-pdf>/document.txt: texto extraído por página.\n"
        "- <nome-do-pdf>/chunks.jsonl: chunks com metadados e vetor numérico.\n\n"
        "## Formato de cada linha em chunks.jsonl\n"
        "- id\n"
        "- source_file\n"
        "- page\n"
        "- chunk_in_page\n"
        "- text\n"
        "- vector (embedding por hashing, dimensão configurável)\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Processed {len(pdf_files)} PDFs into: {output_dir}")


if __name__ == "__main__":
    main()
