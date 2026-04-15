import { useEffect, useState } from "react";
import { CartesianGrid, Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend } from "recharts";

export default function ResultsPage() {
  const [experiments, setExperiments] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [regenBusy, setRegenBusy] = useState(false);
  const [regenError, setRegenError] = useState("");
  const [figSeed, setFigSeed] = useState(0);

  useEffect(() => {
    fetch("/api/results/experiments")
      .then((res) => res.json())
      .then((payload) => {
        const items = payload.items || [];
        setExperiments(items);
        if (items.length > 0) {
          setSelectedId(items[0].id);
        }
      })
      .catch(() => setExperiments([]));
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setMetrics(null);
      return;
    }

    fetch(`/api/results/experiments/${selectedId}`)
      .then((res) => res.json())
      .then(setMetrics)
      .catch(() => setMetrics(null));
  }, [selectedId]);

  async function handleRegenerate() {
    if (!selectedId || regenBusy) return;
    setRegenBusy(true);
    setRegenError("");
    try {
      const res = await fetch(`/api/results/experiments/${selectedId}/regenerate-figures`, {
        method: "POST",
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload?.detail || "Falha ao regenerar figuras");
      }
      const refreshed = await fetch(`/api/results/experiments/${selectedId}`).then((r) => r.json());
      setMetrics(refreshed);
      setFigSeed((prev) => prev + 1);
    } catch (err) {
      setRegenError(err.message || "Falha ao regenerar figuras");
    } finally {
      setRegenBusy(false);
    }
  }

  const chartData = metrics?.history || [];

  return (
    <section className="grid gap-6">
      <div className="rounded-xl border border-line bg-panel p-6">
        <h2 className="text-xl font-semibold text-neon font-mono mb-2">Relatórios de Pesquisa Científica</h2>
        <p className="text-sm text-slate-400 mb-6">Explore o repositório de execuções finalizadas. Os gráficos ilustram o ponto de estabilidade e o comportamento do FedAvg na topologia.</p>
        
        <div className="flex flex-col md:flex-row md:items-end gap-4 border-b border-[#121c2e] pb-6">
          <div className="flex-1 max-w-sm">
            <label className="text-xs uppercase tracking-wide text-slate-400 mb-2 block font-mono">Registro do Experimento</label>
            <select
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              className="w-full rounded-md border border-neon bg-[#07241e] text-neon px-3 py-2 text-sm font-mono focus:outline-none"
            >
              {experiments.length === 0 ? <option value="">Nenhum registro localizado</option> : null}
              {experiments.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.id} (Dataset: {item.dataset})
                </option>
              ))}
            </select>
          </div>
          <div className="flex gap-2">
            {metrics?.tables?.csv && (
              <a className="rounded-md border border-line bg-[#0d1420] px-4 py-2 text-xs font-mono text-slate-300 hover:text-white transition" href={`/api${metrics.tables.csv}`} target="_blank" rel="noreferrer">
                Exportar matriz CSV
              </a>
            )}
            {metrics?.tables?.tex && (
              <a className="rounded-md border border-line bg-[#0d1420] px-4 py-2 text-xs font-mono text-slate-300 hover:text-white transition" href={`/api${metrics.tables.tex}`} target="_blank" rel="noreferrer">
                Exportar TeX
              </a>
            )}
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-4 mt-6">
          <MetricCard label="Conjunto de Dados" value={metrics?.dataset?.toUpperCase() || "N/A"} highlight />
          <MetricCard label="Modelo IA" value={metrics?.model?.toUpperCase() ?? "N/A"} />
          <MetricCard label="Total de Clientes" value={metrics?.clients ?? "0"} />
          <MetricCard label="Rounds" value={metrics?.rounds ?? "N/A"} />
          <MetricCard label="Épocas por Round" value={metrics?.epochs_per_round ?? "N/A"} />
          <MetricCard label="Modo" value={metrics?.mode === "real_fedavg_containers" ? "FedAvg Real" : "N/A"} />
          <MetricCard label="AWGN" value={formatAwgn(metrics?.awgn)} />
          <MetricCard label="Acurácia de Teste Final" value={metrics?.final_accuracy ? `${(metrics.final_accuracy * 100).toFixed(1)}%` : "N/A"} highlight />
          <MetricCard label="Loss Geral (MSE)" value={metrics?.final_loss ?? "N/A"} />
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <div className="rounded-xl border border-line bg-panel p-6">
          <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-300 font-mono">Estabilidade do Sistema (Loss)</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ffd166" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#ffd166" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#223046" />
                <XAxis dataKey="epoch" stroke="#8a9ab4" style={{fontSize: '11px', fontFamily: 'monospace'}}/>
                <YAxis stroke="#8a9ab4" style={{fontSize: '11px', fontFamily: 'monospace'}}/>
                <Tooltip contentStyle={{ backgroundColor: "#0b1220", border: "1px solid #1d2a3d", fontFamily: 'monospace' }} />
                <Area type="monotone" dataKey="loss" stroke="#ffd166" fillOpacity={1} fill="url(#colorLoss)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-xl border border-line bg-panel p-6">
          <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-300 font-mono">Convergência Qualitativa (Acurácia)</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorAcc" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00f6a2" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#00f6a2" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#223046" />
                <XAxis dataKey="epoch" stroke="#8a9ab4" style={{fontSize: '11px', fontFamily: 'monospace'}}/>
                <YAxis stroke="#8a9ab4" style={{fontSize: '11px', fontFamily: 'monospace'}}/>
                <Tooltip contentStyle={{ backgroundColor: "#0b1220", border: "1px solid #1d2a3d", fontFamily: 'monospace' }} />
                <Area type="monotone" dataKey="accuracy" stroke="#00f6a2" fillOpacity={1} fill="url(#colorAcc)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-line bg-panel p-4 font-mono">
        <h3 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-300 px-2 pt-2">Lente de Aumento: Retenção Semântica</h3>
        <p className="text-xs text-slate-400 mb-6 px-2">Comparativo visual amostrado aleatoriamente após a convergência do treinamento no lado do servidor. Avalia visualmente a degradação e blur natural de modelos Autoencoders perante o gargalo de compressão latente.</p>
        <div className="flex flex-wrap items-center gap-3 px-2 mb-4">
          <button
            type="button"
            onClick={handleRegenerate}
            disabled={!selectedId || regenBusy}
            className={`rounded-md border px-3 py-2 text-xs font-mono transition ${regenBusy ? "border-warn bg-[#3d3313] text-warn" : "border-neon bg-[#073529] text-neon hover:bg-[#0b2a22]"} disabled:opacity-50`}
          >
            {regenBusy ? "Regenerando..." : "Regenerar figuras"}
          </button>
          {regenError && (
            <span className="text-[10px] text-[#ff9a9a]">{regenError}</span>
          )}
        </div>
        
        {metrics?.figures?.reconstruction ? (
          <div className="flex justify-center bg-[#070c14] border border-[#1a2537] rounded-lg p-6">
            <img src={`/api${metrics.figures.reconstruction}?t=${figSeed}`} alt="Reconstruction Diff" className="max-h-[300px] object-contain rounded" style={{ imageRendering: 'pixelated' }} />
          </div>
        ) : (
          <div className="bg-[#0b1220] p-8 text-center text-slate-500 rounded border border-line border-dashed">Amostragem de Imagem não foi gerada nas primitivas deste experimento.</div>
        )}
      </div>

    </section>
  );
}

function MetricCard({ label, value, highlight = false }) {
  return (
    <div className={`rounded-md border ${highlight ? 'border-[#ff7b7b] bg-[#1a0f0f]' : 'border-line bg-[#0a111b]'} p-4`}>
      <p className="text-[10px] sm:text-xs uppercase tracking-wider text-slate-400 font-mono truncate">{label}</p>
      <p className={`mt-1 text-lg font-bold font-mono truncate ${highlight ? 'text-[#ff9a9a]' : 'text-slate-100'}`}>{String(value)}</p>
    </div>
  );
}

function formatAwgn(awgn) {
  if (!awgn || !awgn.enabled) return "Desligado";
  if (awgn.snr_db === null || awgn.snr_db === undefined) return "Ativo";
  return `Ativo (${awgn.snr_db} dB)`;
}

