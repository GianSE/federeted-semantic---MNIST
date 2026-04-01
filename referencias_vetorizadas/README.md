# Referencias Vetorizadas

Este diretório contém os PDFs da pasta 'referencia' em um formato amigável para consumo por IA.

## Estrutura
- all_chunks.jsonl: todos os chunks vetorizados em um único arquivo.
- summary.json: resumo por PDF (páginas e chunks).
- <nome-do-pdf>/document.txt: texto extraído por página.
- <nome-do-pdf>/chunks.jsonl: chunks com metadados e vetor numérico.

## Formato de cada linha em chunks.jsonl
- id
- source_file
- page
- chunk_in_page
- text
- vector (embedding por hashing, dimensão configurável)
