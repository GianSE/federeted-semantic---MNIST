import { useEffect, useRef, useState } from "react";

// ─── Dataset metadata (must match DATASET_META in image_utils.py) ────────────

const DATASET_META = {
  fashion: { channels: 1, height: 28, width: 28, rawBytes: 3136,  label: "Fashion-MNIST" },
  mnist:   { channels: 1, height: 28, width: 28, rawBytes: 3136,  label: "MNIST Clássico" },
  cifar10: { channels: 3, height: 32, width: 32, rawBytes: 12288, label: "CIFAR-10 Colorido (32×32)" },
  cifar100:{ channels: 3, height: 32, width: 32, rawBytes: 12288, label: "CIFAR-100 (32×32)" },
};

// ─── Canvas-based image renderer ─────────────────────────────────────────────

/**
 * Render a tensor returned by the Python backend onto an HTML canvas.
 *
 * The backend returns tensors squeezed of the batch dim:
 *   - Grayscale: [H, W]         (28×28 or 32×32)
 *   - RGB:       [3, H, W]      (CIFAR-10)
 *
 * @param {Array} tensorData  - Nested JavaScript array from JSON response.
 * @param {number} displaySize - Canvas render width/height in px (default 140).
 */
function TensorImage({ tensorData, label, displaySize = 140 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!tensorData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const isRGB = (
      tensorData.length === 3 &&
      Array.isArray(tensorData[0]) &&
      Array.isArray(tensorData[0][0])
    );

    const height = isRGB ? tensorData[0].length : tensorData.length;
    const width  = isRGB ? tensorData[0][0].length : (tensorData[0]?.length || height);

    canvas.width  = width;
    canvas.height = height;

    const imageData = ctx.createImageData(width, height);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const offset = (y * width + x) * 4;
        let r, g, b;

        if (isRGB) {
          r = Math.round(Math.max(0, Math.min(1, tensorData[0][y][x])) * 255);
          g = Math.round(Math.max(0, Math.min(1, tensorData[1][y][x])) * 255);
          b = Math.round(Math.max(0, Math.min(1, tensorData[2][y][x])) * 255);
        } else {
          const v = Math.round(Math.max(0, Math.min(1, tensorData[y][x])) * 255);
          r = g = b = v;
        }

        imageData.data[offset]     = r;
        imageData.data[offset + 1] = g;
        imageData.data[offset + 2] = b;
        imageData.data[offset + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [tensorData]);

  if (!tensorData) return null;

  return (
    <div className="flex flex-col items-center gap-2">
      <canvas
        ref={canvasRef}
        style={{
          width: displaySize,
          height: displaySize,
          imageRendering: "pixelated",
          borderRadius: "4px",
          border: "1px solid #1d2a3d",
        }}
      />
      {label !== undefined && (
        <span className="text-xs text-slate-400 font-mono">Classe: {label}</span>
      )}
    </div>
  );
}

// ─── Metric badge helper ──────────────────────────────────────────────────────

function MetricRow({ label, value, color = "text-slate-200", unit = "" }) {
  return (
    <div className="flex justify-between items-center py-1 border-b border-[#121c2e] last:border-0">
      <span className="text-xs text-slate-500 font-mono">{label}</span>
      <span className={`text-xs font-bold font-mono ${color}`}>
        {value}{unit && <span className="text-slate-400 ml-1">{unit}</span>}
      </span>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

/**
 * SemanticCommsPage
 *
 * Demonstrates the semantic compression pipeline: raw image → latent vector → reconstruction.
 * Metrics shown: MSE, PSNR (dB), SSIM, compression ratio, bandwidth reduction.
 */
export default function SemanticCommsPage() {
  const [dataset,    setDataset]    = useState("fashion");
  const [modelType,  setModelType]  = useState("cnn_vae");
  const [bits,       setBits]       = useState(8);
  const [awgn,       setAwgn]       = useState({ enabled: false, snr_db: 10 });
  const [masking,    setMasking]    = useState({ enabled: false, drop_rate: 0.25, fill_value: 0 });
  const [classifier, setClassifier] = useState({ enabled: true, min_confidence: 0.5, top_k: 1 });
  const [weights,    setWeights]    = useState([]);
  const [weightsLoading, setWeightsLoading] = useState(false);
  const [weightsError, setWeightsError] = useState("");
  const [baseWeights, setBaseWeights] = useState("random");
  const [loading,    setLoading]    = useState(false);
  const [result,     setResult]     = useState(null);
  const [processErr, setProcessErr] = useState(null);


  const meta = DATASET_META[dataset] ?? DATASET_META.fashion;

  useEffect(() => {
    const params = new URLSearchParams({ dataset, model: modelType });
    setWeightsLoading(true);
    setWeightsError("");
    fetch(`/api/weights?${params.toString()}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((payload) => {
        const items = payload?.items || [];
        setWeights(items);
        const hasCurrent = items.some((item) => item.key === baseWeights);
        if (!hasCurrent) {
          const hasLatest = items.some((item) => item.key === "latest");
          if (hasLatest) setBaseWeights("latest");
          else if (items.length > 0) setBaseWeights(items[0].key);
          else setBaseWeights("random");
        }
      })
      .catch(() => {
        setWeights([]);
        setWeightsError("Falha ao carregar pesos");
      })
      .finally(() => setWeightsLoading(false));
  }, [dataset, modelType]);

  // ── Compression pipeline ──────────────────────────────────────────────────

  async function handleProcess() {
    setLoading(true);
    setProcessErr(null);
    try {
      const res = await fetch("/api/semantic/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset,
          model_type: modelType,
          bits,
          awgn,
          masking,
          classifier,
          base_weights: baseWeights === "random" ? null : baseWeights,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Erro no servidor");
      setResult(data);
    } catch (err) {
      setProcessErr(err.message);
    } finally {
      setLoading(false);
    }
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="grid gap-6">
      {/* ── Section: Semantic Compression Pipeline ─────────────────────── */}
      <div className="rounded-xl border border-line bg-panel p-6">
        <h2 className="text-xl font-semibold text-neon font-mono mb-1">
          Comunicação Semântica — Pipeline de Compressão
        </h2>
        <p className="text-sm text-slate-400 mb-6 font-mono">
          Converte imagens brutas em vetores latentes quantizados para transmissão
          eficiente — comprova a hipótese central da pesquisa.
        </p>

        {/* Controls */}
        <div className="flex flex-wrap gap-4 mb-6 text-sm font-mono text-slate-300">
          <div className="flex-1 min-w-[140px] max-w-xs">
            <label className="block mb-2 text-xs uppercase tracking-wide text-slate-400">Dataset</label>
            <select
              value={dataset}
              onChange={(e) => { setDataset(e.target.value); setResult(null); }}
              className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
            >
              {Object.entries(DATASET_META).map(([key, d]) => (
                <option key={key} value={key}>{d.label}</option>
              ))}
            </select>
          </div>

          <div className="flex-1 min-w-[140px] max-w-xs">
            <label className="block mb-2 text-xs uppercase tracking-wide text-slate-400">Modelo Encoder</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
            >
              <option value="cnn_vae">VAE Convolucional (Recomendado)</option>
              <option value="cnn_ae">Autoencoder Classico (AE)</option>
            </select>
          </div>

          <div className="flex-1 min-w-[160px] max-w-xs">
            <label className="block mb-2 text-xs uppercase tracking-wide text-slate-400">Pesos Base</label>
            <select
              value={baseWeights}
              onChange={(e) => setBaseWeights(e.target.value)}
              disabled={weightsLoading}
              className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2 disabled:opacity-50"
            >
              <option value="random">Inicializacao aleatoria</option>
              {weights.map((item) => (
                <option key={item.key} value={item.key}>{item.label}</option>
              ))}
            </select>
            {weightsError && (
              <p className="text-[10px] text-[#ff9a9a] mt-1">{weightsError}</p>
            )}
            {!weightsLoading && weights.length === 0 && !weightsError && (
              <p className="text-[10px] text-slate-500 mt-1">Nenhum peso encontrado para este dataset/modelo.</p>
            )}
          </div>

          <div className="flex-1 min-w-[140px] max-w-xs">
            <label className="block mb-2 text-xs uppercase tracking-wide text-slate-400">Quantizacao</label>
            <select
              value={bits}
              onChange={(e) => setBits(Number(e.target.value))}
              className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
            >
              <option value={4}>Int4 (agressiva)</option>
              <option value={8}>Int8 (equilibrada)</option>
              <option value={16}>Int16 (suave)</option>
              <option value={32}>Float32 (sem quantizacao)</option>
            </select>
          </div>

          <div className="flex-1 min-w-[160px] max-w-xs">
            <label className="block mb-2 text-xs uppercase tracking-wide text-slate-400">AWGN</label>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setAwgn((prev) => ({ ...prev, enabled: !prev.enabled }))}
                className={`rounded uppercase text-[10px] px-2 py-1 font-bold transition ${awgn.enabled ? "bg-[#073529] text-neon border border-neon" : "bg-[#1f2937] text-slate-400 border border-transparent"}`}
              >
                {awgn.enabled ? "Ativo" : "Inativo"}
              </button>
              <span className="text-xs text-slate-400">{awgn.enabled ? `${awgn.snr_db} dB` : "-"}</span>
            </div>
            {awgn.enabled && (
              <input
                type="range"
                min="0"
                max="30"
                value={awgn.snr_db}
                onChange={(e) => setAwgn((prev) => ({ ...prev, snr_db: Number(e.target.value) }))}
                className="mt-2 w-full accent-neon"
              />
            )}
          </div>

          <div className="flex-1 min-w-[180px] max-w-xs">
            <label className="block mb-2 text-xs uppercase tracking-wide text-slate-400">Masking</label>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setMasking((prev) => ({ ...prev, enabled: !prev.enabled }))}
                className={`rounded uppercase text-[10px] px-2 py-1 font-bold transition ${masking.enabled ? "bg-[#3d2b05] text-[#ffd166] border border-[#ffd166]" : "bg-[#1f2937] text-slate-400 border border-transparent"}`}
              >
                {masking.enabled ? "Ativo" : "Inativo"}
              </button>
              <span className="text-xs text-slate-400">{masking.enabled ? `${Math.round(masking.drop_rate * 100)}%` : "-"}</span>
            </div>
            {masking.enabled && (
              <input
                type="range"
                min="0"
                max="80"
                value={Math.round(masking.drop_rate * 100)}
                onChange={(e) => setMasking((prev) => ({ ...prev, drop_rate: Number(e.target.value) / 100 }))}
                className="mt-2 w-full accent-[#ffd166]"
              />
            )}
            {masking.enabled && (
              <div className="mt-2 flex items-center gap-2 text-[10px] text-slate-500">
                <span>fill</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={masking.fill_value}
                  onChange={(e) => setMasking((prev) => ({ ...prev, fill_value: Number(e.target.value) }))}
                  className="w-full accent-[#ffd166]"
                />
                <span>{masking.fill_value.toFixed(2)}</span>
              </div>
            )}
          </div>

          <div className="flex-1 min-w-[200px] max-w-xs">
            <label className="block mb-2 text-xs uppercase tracking-wide text-slate-400">Classificador</label>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setClassifier((prev) => ({ ...prev, enabled: !prev.enabled }))}
                className={`rounded uppercase text-[10px] px-2 py-1 font-bold transition ${classifier.enabled ? "bg-[#172554] text-[#7aa2ff] border border-[#7aa2ff]" : "bg-[#1f2937] text-slate-400 border border-transparent"}`}
              >
                {classifier.enabled ? "Ativo" : "Inativo"}
              </button>
              <span className="text-xs text-slate-400">Top-{classifier.top_k}</span>
            </div>
            {classifier.enabled && (
              <div className="mt-2 grid gap-2">
                <div className="flex items-center gap-2 text-[10px] text-slate-500">
                  <span>Conf.</span>
                  <input
                    type="range"
                    min="0.1"
                    max="0.95"
                    step="0.05"
                    value={classifier.min_confidence}
                    onChange={(e) => setClassifier((prev) => ({ ...prev, min_confidence: Number(e.target.value) }))}
                    className="w-full accent-[#7aa2ff]"
                  />
                  <span>{classifier.min_confidence.toFixed(2)}</span>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-slate-500">
                  <span>Top-k</span>
                  <select
                    value={classifier.top_k}
                    onChange={(e) => setClassifier((prev) => ({ ...prev, top_k: Number(e.target.value) }))}
                    className="w-full rounded-md border border-line bg-[#0b1220] px-2 py-1"
                  >
                    {[1, 3, 5].map((k) => (
                      <option key={k} value={k}>Top-{k}</option>
                    ))}
                  </select>
                </div>
              </div>
            )}
          </div>
        </div>

        <button
          id="semantic-process-btn"
          onClick={handleProcess}
          disabled={loading}
          className="rounded-md bg-[#073529] border border-neon text-neon px-4 py-2 font-mono text-sm hover:bg-[#0b2a22] transition disabled:opacity-50"
        >
          {loading ? "Processando..." : "Gerar Vetor Latente & Reconstruir"}
        </button>

        {processErr && (
          <div className="mt-4 rounded border border-red-800 bg-[#1a0f0f] p-3 text-xs text-red-400 font-mono">
            Erro: {processErr}
          </div>
        )}

        {result?.status === "ok" && !result.weights_loaded && (
          <div className="mt-4 rounded border border-[#ff7b7b] bg-[#1a0f0f] p-3 text-xs text-[#ff9a9a] font-mono">
            Aviso: pesos nao carregados. A reconstrucao pode ficar ruim. Treine ou selecione um snapshot.
          </div>
        )}

        {result?.status === "ok" && result.classifier?.enabled && !result.classifier?.loaded && (
          <div className="mt-4 rounded border border-[#7aa2ff] bg-[#0b1220] p-3 text-xs text-[#a5b4fc] font-mono">
            Classificador nao encontrado para este dataset. Rode o treino do classificador para ativar as metricas.
          </div>
        )}

        {result?.status === "ok" && (
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6 items-start">
            {/* Original */}
            <div className="text-center bg-[#0a111b] p-4 rounded-lg border border-line">
              <h3 className="text-sm font-mono text-slate-300 mb-4">Imagem Original</h3>
              <div className="flex justify-center">
                <TensorImage tensorData={result.original} label={result.label} />
              </div>
              <div className="mt-4 pt-3 border-t border-[#1d2a3d]">
                <MetricRow label="Tamanho (float32)" value={`${result.original_bytes ?? meta.rawBytes} B`} />
              </div>
            </div>

            {/* Received */}
            <div className="text-center bg-[#0b2a22] p-4 rounded-lg border border-neon">
              <h3 className="text-sm font-mono text-neon mb-4">Imagem Recebida</h3>
              <div className="flex justify-center">
                <TensorImage tensorData={result.received} />
              </div>
              <div className="mt-4 pt-3 border-t border-[#1d2a3d] grid gap-1">
                <MetricRow label="MSE recebido ↓" value={result.mse_received?.toFixed(5)} />
                <MetricRow label="PSNR recebido ↑" value={`${result.psnr_received?.toFixed(1)} dB`} color="text-[#ffd166]" />
                <MetricRow label="SSIM recebido ↑" value={result.ssim_received?.toFixed(3)} color="text-[#489dff]" />
              </div>
            </div>

            {/* Transmission stats */}
            <div className="text-center font-mono text-sm rounded-lg bg-[#0b2a22] p-4 border border-neon">
              <p className="text-neon mb-1 font-bold">📡 Transmissao Semantica</p>
              <p className="text-slate-400 text-xs mb-4">Canal → Receptor → Reconstrucao</p>

              <div className="text-5xl font-bold text-white my-3">
                {result.compression_ratio != null
                  ? `${result.compression_ratio}×`
                  : `${((result.original_bytes ?? meta.rawBytes) / (result.latent_size_int8 || 1)).toFixed(1)}×`}
              </div>
              <p className="text-xs text-slate-400 mb-4">compressao</p>

              <div className="bg-[#052217] rounded-md p-3 text-left grid gap-1.5">
                <MetricRow
                  label={`Latente Int${bits}`}
                  value={`${result.latent_size_int8} B`}
                  color="text-neon"
                />
                <MetricRow
                  label="Latente Float32"
                  value={`${result.latent_size_float} B`}
                />
                <MetricRow
                  label="Reducao de Banda"
                  value={`${result.bandwidth_reduction_pct ?? "—"}%`}
                  color="text-neon"
                />
              </div>
            </div>

            {/* Reconstructed */}
            <div className="text-center bg-[#0a111b] p-4 rounded-lg border border-line">
              <h3 className="text-sm font-mono text-slate-300 mb-4">Reconstrucao da IA</h3>
              {result.weights_loaded && result.weights_source && (
                <p className="text-[10px] text-slate-500 font-mono mb-2">
                  Pesos: {result.weights_source}
                </p>
              )}
              <div className="flex justify-center">
                <TensorImage tensorData={result.reconstructed} />
              </div>
              <div className="mt-4 pt-3 border-t border-[#1d2a3d] grid gap-1">
                <MetricRow label="MSE reconstrucao ↓" value={result.mse?.toFixed(5)} />
                <MetricRow label="PSNR reconstrucao ↑" value={`${result.psnr?.toFixed(1)} dB`} color="text-[#ffd166]" />
                <MetricRow label="SSIM reconstrucao ↑" value={result.ssim?.toFixed(3)} color="text-[#489dff]" />
              </div>
            </div>
          </div>
        )}

        {result?.status === "ok" && result.classifier?.enabled && result.classifier?.loaded && (
          <div className="mt-6 rounded-lg border border-line bg-[#0a111b] p-4 font-mono text-xs">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-[#7aa2ff]">Classificador Semântico</h3>
              <span className="text-[10px] text-slate-500">
                Top-{result.classifier.top_k} | conf ≥ {result.classifier.min_confidence}
              </span>
            </div>
            <div className="grid gap-2">
              {["original", "received", "reconstructed"].map((key) => {
                const item = result.classifier[key];
                if (!item) return null;
                return (
                  <div key={key} className="grid grid-cols-4 gap-2 border-b border-[#121c2e] py-2 last:border-0">
                    <span className="text-slate-400 uppercase text-[10px]">{key}</span>
                    <span className="text-slate-300">Pred: {item.pred}</span>
                    <span className="text-slate-300">Conf: {item.confidence?.toFixed(3)}</span>
                    <span className={item.recognized ? "text-neon" : "text-[#ff7b7b]"}>
                      {item.recognized ? "Reconhecida" : "Nao reconhecida"}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
