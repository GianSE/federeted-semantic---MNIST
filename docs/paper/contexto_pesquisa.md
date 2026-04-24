# Contexto da Pesquisa — Comunicação Semântica com Aprendizado Federado

## Autores
- **Gian Pedro Rodrigues** — Aluno, UTFPR Cornélio Procópio
- **Herman L. dos Santos** — Orientador, Prof. UTFPR Cornélio Procópio
- Email: gian.2000@alunos.utfpr.edu.br | hermansantos@utfpr.edu.br

## Hipótese Central
> Um autoencoder (CNN-VAE) treinado via Federated Learning pode comprimir imagens
> para um espaço latente compacto, reduzindo drasticamente a largura de banda,
> enquanto preserva a integridade semântica (queda de acurácia ≤ 5%).

## Resultados Experimentais Obtidos (Experimento Real)

### Identificação do Experimento
- **ID**: `experimento_20260422_024134`
- **Dataset**: MNIST (28×28, 1 canal, 10 classes)
- **Modelo**: CNN-VAE (Variational Autoencoder Convolucional)
- **Clientes FL**: 3 clientes | 5 rounds × 5 épocas/round
- **Loss final FL**: `0.010379`

### Compressão de Banda
| Métrica | Valor |
|---------|-------|
| Bytes por imagem (raw float32) | 3.136 B |
| Bytes transmitidos (latente int8) | 36 B |
| **Razão de compressão** | **87,1×** |
| **Redução de largura de banda** | **98,9%** |
| Dimensão do espaço latente | 32 valores |

### Qualidade de Reconstrução (50 amostras)
| Métrica | Média | Desvio Padrão |
|---------|-------|----------------|
| MSE | 0,00975 | ± 0,00437 |
| PSNR | **20,61 dB** | ± 2,22 dB |
| SSIM | **0,9111** | ± 0,0436 |

### Preservação Semântica (Classificador CNN)
| Métrica | Valor |
|---------|-------|
| Acurácia na imagem original | 90,0% |
| Acurácia na imagem reconstruída | 86,0% |
| **Queda semântica** | **4,0%** |
| Critério ≤ 5% | ✅ **APROVADO** |

### Escalabilidade (Economia por número de dispositivos)
| Dispositivos | Dados brutos (KB) | Dados latentes (KB) | Economia (KB) |
|-------------|-------------------|----------------------|---------------|
| 1  | 3,06   | 0,04  | 3,02   |
| 5  | 15,31  | 0,18  | 15,13  |
| 10 | 30,62  | 0,35  | 30,27  |
| 50 | 153,12 | 1,76  | 151,36 |
| 100| 306,25 | 3,52  | 302,73 |

## Pipeline do Canal Semântico

```
TRANSMISSOR                    CANAL AWGN           RECEPTOR
Imagem (3136B) → [Encoder] → z (32 floats) → quantiza int8 → z' (36B) → [Decoder] → Imagem reconstruída
                                   ↓
                          Banda economizada: 98,9%
```

1. **Encoder**: CNN com 2 camadas convolucionais + 2 lineares → µ e log(σ²) (VAE)
2. **Quantização**: float32 → int8 (reparameterization trick + uniform min-max)
3. **Canal AWGN**: SNR = 15 dB no espaço latente
4. **Masking**: 10% de drop rate (pixels aleatórios zerados)
5. **Decoder**: 2 lineares + ConvTranspose → imagem 28×28

## Arquitetura CNN-VAE

### Encoder
| Camada | Tipo | Saída |
|--------|------|-------|
| Entrada | Imagem grayscale | 1×28×28 |
| Conv2d + ReLU | 32 filtros 3×3 | 32×28×28 |
| MaxPool2d 2×2 | — | 32×14×14 |
| Conv2d + ReLU | 64 filtros 3×3 | 64×14×14 |
| MaxPool2d 2×2 | — | 64×7×7 |
| Conv2d + ReLU | 128 filtros 3×3 | 128×7×7 |
| MaxPool2d 2×2 | — | 128×3×3 |
| Flatten + Linear | → 256 | 256 |
| Linear (µ) | → 32 | **32** |
| Linear (log σ²) | → 32 | **32** |

### Decoder (espelhado)
- Linear 32 → 256 → 3136 → Reshape → ConvTranspose → Sigmoid

### Função de Perda VAE
```
L_VAE = L_MSE + β · D_KL(q(z|x) || N(0, I))
D_KL = -½ Σ(1 + log σ²_j - µ²_j - σ²_j)
β = 1 (VAE padrão)
```

## Protocolo FedAvg

- **Rounds**: 5
- **Épocas locais/round**: 5
- **Clientes**: 3 (simulados em containers Docker)
- **Batch size**: 64
- **Otimizador**: Adam, lr = 0,001
- **Agregação**: média simples dos state_dicts (FedAvg)
- **Volume compartilhado**: `/app/data/ml-data/` (Docker bind mount)

## Infraestrutura Docker

| Container | Função | Porta |
|-----------|--------|-------|
| `ml-service` | FastAPI — orquestração, benchmark, canal semântico | 8000 |
| `fl-server` | FastAPI — FedAvg, agregação, distribuição de pesos | 8100 |
| `fl-client-1` | Cliente FL #1 — treino local | — |
| `fl-client-2` | Cliente FL #2 — treino local | — |
| `fl-client-3` | Cliente FL #3 — treino local | — |

Volume compartilhado: `./shared_data:/app/data`

## Figuras Disponíveis para o Paper

| Arquivo | Conteúdo |
|---------|---------|
| `figures/fig_convergence.png` | Curva de convergência loss por round |
| `figures/fig_reconstruction.png` | Grade original × reconstruída |
| `figures/fig_semantic_compressions.png` | Gráfico 3 painéis: bytes, acurácia, escalabilidade |
| `figures/fig_canal_semantico.png` | Original → recebida+ruído → reconstruída (1 imagem) |
| `figures/fig_galeria_analise.png` | Histograma PSNR + pizza de resultados semânticos |
| `figures/fig_chaos.png` | Impacto do caos na convergência |
| `figures/fig_completion.png` | Completação de imagens parciais |

## Conclusão / Validação da Hipótese

A hipótese **foi confirmada**:
- ✅ **98,9% de redução de largura de banda** (3136 B → 36 B, razão 87,1×)
- ✅ **Queda semântica de apenas 4,0%** (90,0% → 86,0%), dentro do limiar de 5%
- ✅ **PSNR 20,6 dB e SSIM 0,911** — qualidade de reconstrução excelente
- ✅ **Loss FL convergiu para 0,0104** em 5 rounds × 5 épocas

O CNN-VAE treinado via Federated Learning efetivamente aprende a comprimir imagens
mantendo sua semântica, demonstrando a viabilidade da comunicação semântica
federada para cenários IoT/6G com restrição de banda.
