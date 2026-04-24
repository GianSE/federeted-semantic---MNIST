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

## 🏗️ Arquitetura do Sistema Científico e Redes Juízas

A orquestração paralela segue dois corações de Inteligência Artificial para emulação do canal:
- **Rede Preditiva Dinâmica:** Utilizamos CNNs purificadas com *BatchNormalization* para processamento semântico leve (MNIST) e um motor agressivo de **Transfer Learning empregando pesos da MobileNetV2** (pré-treino no grande *ImageNet*) redimensionado dinamicamente via PyTorch para inferir datasets tridimensionais severos sem estourar o limite processual da Borda (Ex: CIFAR-10 em ambiente CPU/Edge).

O sistema inteiro é encapsulado por um Backend REST (Microserviço FastApi) e governado metodologicamente via **Jupyter Notebook**:

```text
┌─────────────────────────────────────────────────────────────────┐
│  PLATAFORMA DE EXPERIMENTAÇÃO (Jupyter Notebook)                │
│  experimento_federado.ipynb                                     │
│  Lança chamadas aos atuadores e renderiza vetorialmente os dados│
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP REST Requests Livres de UI
┌──────────────────────────▼──────────────────────────────────────┐
│  ORQUESTRADOR E API DE IA (ml-service)                          │
│  FastAPI + PyTorch 2.5 — Validações da MobileNetV2 / CNNs       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Gerente Disparador
┌──────────────────────────▼──────────────────────────────────────┐
│  REDE COLETIVA FEDERADA (FL-Nodes)                              │
│  [ fl-server ] <--> [ fl-client-1 ] , [ fl-client-2 ]           │
│  Simulação de Antenas Rádio Base (FedAvg Autoencoder Isolado)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Como Executar o Experimento

### Pré-requisitos
- **Docker Desktop** ativo na máquina base.
- IDE Visual (ex: *VS Code* com a extensão para Kernels Jupyter).

### Passo 1: Ligar a Infraestrutura Restrita 6G
Abra o terminal na pasta raiz e instancie a fábrica central enxuta do projeto:

```bash
docker-compose up --build -d
```

### Passo 2: O Laboratório de Pesquisa Exploratória (Arquitetura do Jupyter)
Pressione `Play` no arquivo principal **`experimento_federado.ipynb`** baseando-se estritamente na progressão acadêmica de Módulos (Storytelling implantado de Top/Down):

- **📘 Módulo 1 (A Fundação):** Levanta bibliotecas locais e ingere vetores de Datasets pela porta REST.
- **🧠 Módulo 2 (O Cérebro Semântico):** Operação Paralela em blocos — Primeiro treina os Autoencoders em FedAvg simulando o Meio Criptográfico Rádio; em sequência, treina as Redes Neurais Preditoras localmente (O nosso Juiz Semântico).
- **🔬 Módulo 3 (Laboratório Visual):** Executa o processador gráfico e injeta caos estático nas imagens vetoriais, documentando a reação e as reconstruções semânticas da MobileNet num mosaico em Tela de "Inspeção".
- **🏭 Módulo 4 (A Fábrica Automotriz de Trade-off):** Loop paramétrico violento iterando todos os graus de Compressão (Bit-Width) vs SNR. Célula responsável final que emite todo e qualquer `CSV` empírico ou plot quantitativo `PNG` lido pelo tcc `.tex`.

---

## 📂 Estrutura de Diretórios Sanitizada

*A engenharia do laboratório foi polida removendo endpoints Web e artefatos velhos para concentrar os HDs do docker estritamente na computação de métricas.*

```text
federeted-semantic/
├── docker-compose.yml           # Infraestrura das três máquinas lógicas
├── experimento_federado.ipynb   # O Painel Jupyter Magno
│
├── docs/                        # Gaveta de Teorias 
│   ├── literature/              # Arquivo de Referências Acadêmicas Inertes
│   └── *.md                     # Roteiros secundários
│
├── ml-service/                  # O Corpo e o Cérebro de IA
│   └── app/
│       ├── main.py              # Backend REST limpo (Start/Stop/Benchmark Iterativo)
│       └── classifier_utils.py  # Interceptor MobileNetV2
│
├── fl-server/                   # Nó Agregador Global (Federated Learning)
├── fl-client/                   # Antenas Edge Pessoais Simétricas
│
├── shared_data/                 # Cache, Volumes Montados Permanentes (Sem UI ghosting)
│   ├── ml-data/                 # /datasets limpos, /weights .pth salvos e /runs
│   └── resultados/              # Destino final automático dos desenhos de paper (Galeria/Tradeoff)
│
└── paper/                       # Redação de Compilação Oficial em LaTeX (IEEEtran)
```

---
*Pesquisa Operacionalizada Acadêmica | 2026*