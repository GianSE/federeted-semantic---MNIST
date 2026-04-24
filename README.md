# 🛰️ Comunicação Semântica Federada IoT/6G — Testbed de Pesquisa

> Protótipo de pesquisa que demonstra como **Nós Baseados em IA Generativa** (Autoencoders Variacionais/AE)
> reduzem de forma radical o volume de dados transmitidos na borda, extraindo e trafegando apenas a informação semântica utilitária. 
> Rigorosamente avaliado contra degradação matemática nos datasets MNIST, Fashion-MNIST e CIFAR-10.

[![Python](https://img.shields.io/badge/Python-3.11+-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)](https://docs.docker.com/compose/)

---

## 📋 Hipótese e O "Trade-off Triplo"

O Testbed foi projetado para provar que a compressão extrema através de um Extrator de Características Semânticas na transmissão, acoplado a um Nó GenAI receptor, é a arquitetura ideal na futura conectividade IoT/6G.

A análise metodológica foca no Ponto de Equilíbrio entre 3 vetores matemáticos:
1. **Redução de Exigência em Canal (Compressão Otimizada)**
2. **Robustez ante o Ruído Quântico de Transmissão Físico (AWGN/Perdas)**
3. **Preservação de Significado (Acurácia Oculta pelo Classificador Juiz)**

---

## 🏗️ Arquitetura do Sistema e Pipeline Linear

O sistema foi refatorado para garantir **reprodutibilidade acadêmica total** através de um pipeline estritamente linear (Módulos 1 a 7). O fluxo garante que o "Storytelling" do experimento seja respeitado, acumulando conhecimento entre as etapas de treino e avaliação.

```text
┌─────────────────────────────────────────────────────────────────┐
│  PLATAFORMA DE EXPERIMENTAÇÃO (Jupyter Notebook)                │
│  experimento_federado.ipynb                                     │
│  Orquestração Linear (1 → 7)                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP REST Requests
┌──────────────────────────▼──────────────────────────────────────┐
│  ORQUESTRADOR E API DE IA (ml-service)                          │
│  FastAPI + PyTorch 2.5 — Validações da MobileNetV2 / CNNs       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Gerente Disparador
┌──────────────────────────▼──────────────────────────────────────┐
│  REDE COLETIVA FEDERADA (FL-Nodes)                              │
│  [ fl-server ] <--> [ fl-client-n ]                             │
│  Simulação de Antenas Rádio Base (FedAvg Autoencoders)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Como Executar o Experimento

### Passo 1: Ligar a Infraestrutura
Abra o terminal na pasta raiz e instancie os serviços Docker:
```bash
docker-compose up --build -d
```

### Passo 2: Execução do Notebook (1 a 7)
Abra o arquivo principal **`experimento_federado.ipynb`** e execute as células em ordem estrita:

1.  **📘 Módulo 1 (Setup):** Configura parâmetros globais e baixa os datasets necessários.
2.  **🧠 Módulo 2 (Cérebro Semântico - Treino):** 
    *   **Célula 4**: Treina Autoencoders para a lista de dimensões (`LATENT_DIMS`).
    *   **Célula 5**: Treina Classificadores Juízes sincronizados com as dimensões latentes.
3.  **🔬 Módulo 3 (Laboratório Visual):**
    *   **Célula 6**: Gera mosaicos qualitativos para inspeção visual do canal semântico.
4.  **🏭 Módulo 4 (Fábrica de Trade-off):**
    *   **Célula 7**: Benchmark automatizado final que gera o `CSV` e os plots `PNG` para o artigo.

---

## 📂 Estrutura de Diretórios Sanitizada

```text
federeted-semantic/
├── docker-compose.yml           # Infraestrutura Docker
├── experimento_federado.ipynb   # Interface Master de execução
│
├── ml-service/                  # Microserviço de IA e Orquestração
├── fl-server/ / fl-client/      # Componentes de Aprendizado Federado
│
├── shared_data/                 # Volume persistente compartilhado
│   ├── ml-data/                 # Datasets e logs de treino
│   ├── fl-weights/              # Pesos .pth (Modelos treinados)
│   └── resultados/              # Destino final de plots e métricas
│
└── docs/ / paper/               # Documentação e manuscrito LaTeX
```

---
*Pesquisa Operacionalizada Acadêmica — Testbed IoT/6G | 2026*