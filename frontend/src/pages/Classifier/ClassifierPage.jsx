import { useEffect, useMemo, useState } from "react";
import {
  AreaChart,
  Area,
  CartesianGrid,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";
import TerminalLogWindow from "../../components/TerminalLogWindow";

const DATASET_OPTIONS = [
  { value: "mnist", label: "MNIST" },
  { value: "fashion", label: "Fashion-MNIST" },
  { value: "cifar10", label: "CIFAR-10" },
  { value: "cifar100", label: "CIFAR-100" },
];

const ACC_COLORS = {
  original: "#00f6a2",
  received: "#ffd166",
  reconstructed: "#489dff",
};

function parseNumberList(raw, fallback) {
  if (!raw) return fallback;
  const nums = raw
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((value) => Number.isFinite(value));
  return nums.length ? nums : fallback;
}

export default function ClassifierPage() {
  const [dataset, setDataset] = useState("mnist");
  const [epochs, setEpochs] = useState(5);
  const [batch, setBatch] = useState(128);
  const [lr, setLr] = useState(1e-3);
  const [seed, setSeed] = useState(42);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const [topK, setTopK] = useState(1);
  const [minConfidence, setMinConfidence] = useState(0.5);
  const [evalSamples, setEvalSamples] = useState(200);
  const [bits, setBits] = useState(8);
  const [semanticModel, setSemanticModel] = useState("cnn_vae");
  const [snrGrid, setSnrGrid] = useState("5, 10, 15, 20, 25");
  const [maskingGrid, setMaskingGrid] = useState("0.1, 0.25, 0.4, 0.6");

  const [logs, setLogs] = useState([]);
  const [streamEnabled, setStreamEnabled] = useState(false);
  const [connected, setConnected] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [actionPending, setActionPending] = useState(false);

  const [latest, setLatest] = useState(null);
  const [loadError, setLoadError] = useState("");

  const streamUrl = useMemo(() => "/api/classifier/logs/stream", []);

  useEffect(() => {
    if (!streamEnabled) {
      setConnected(false);
      return;
    }

    const source = new EventSource(streamUrl);
    source.onopen = () => {
      setConnected(true);
      setLogs([]);
    };
    source.onerror = () => setConnected(false);
    source.onmessage = (event) => {
      if (!event.data) return;
      if (event.data.startsWith("[heartbeat]")) return;

      setLogs((prev) => [...prev.slice(-180), event.data]);

      if (event.data.includes("[done]") || event.data.includes("[stopped]")) {
        setIsTraining(false);
        fetchLatest();
      }
    };

    return () => source.close();
  }, [streamEnabled, streamUrl]);

  useEffect(() => {
    fetch("/api/classifier/status")
      .then((res) => (res.ok ? res.json() : null))
      .then((status) => {
        if (!status) return;
        setIsTraining(Boolean(status.running));
        if (status.running) setStreamEnabled(true);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const timer = setInterval(() => {
      fetch("/api/classifier/status")
        .then((res) => (res.ok ? res.json() : null))
        .then((status) => {
          if (!status) return;
          setIsTraining(Boolean(status.running));
        })
        .catch(() => {});
    }, 6000);
    return () => clearInterval(timer);
  }, []);

  function fetchLatest() {
    setLoadError("");
    fetch("/api/classifier/results/latest")
      .then((res) => (res.ok ? res.json() : null))
      .then((payload) => setLatest(payload))
      .catch(() => setLoadError("Falha ao carregar resultados"));
  }

  useEffect(() => {
    fetchLatest();
  }, []);

  async function handleStart() {
    setActionPending(true);
    setLoadError("");
    const payload = {
      dataset,
      epochs,
      batch,
      lr,
      seed,
      top_k: topK,
      min_confidence: minConfidence,
      eval_samples: evalSamples,
      bits,
      semantic_model: semanticModel,
      snr_grid: parseNumberList(snrGrid, [5, 10, 15, 20, 25]),
      masking_grid: parseNumberList(maskingGrid, [0.1, 0.25, 0.4, 0.6]),
    };

    const response = await fetch("/api/classifier/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const text = await response.text();
      setLogs((prev) => [...prev, `[error] falha ao iniciar: ${text}`]);
      setActionPending(false);
      return;
    }

    setStreamEnabled(true);
    setIsTraining(true);
    setActionPending(false);
  }

  async function handleStop() {
    setActionPending(true);
    await fetch("/api/classifier/stop", { method: "POST" });
    setLogs((prev) => [...prev, "[controle] parada solicitada"]);
    setActionPending(false);
  }

  async function handleClearLogs() {
    setActionPending(true);
    await fetch("/api/classifier/logs/clear", { method: "POST" });
    setLogs([]);
    setActionPending(false);
  }

  const history = latest?.history || [];
  const evaluation = latest?.evaluation || {};

  const accBaseline = evaluation?.baseline || {};
  const comparisonData = [
    { name: "Original", value: accBaseline.accuracy_original ?? null, color: ACC_COLORS.original },
    { name: "Recebida", value: accBaseline.accuracy_received ?? null, color: ACC_COLORS.received },
    { name: "Reconstruida", value: accBaseline.accuracy_reconstructed ?? null, color: ACC_COLORS.reconstructed },
  ].filter((item) => typeof item.value === "number");

  const snrData = (evaluation?.snr_curve || []).map((row) => ({
    snr: row.snr_db,
    recebida: row.accuracy_received,
    reconstruida: row.accuracy_reconstructed,
  }));

  const maskingData = (evaluation?.masking_curve || []).map((row) => ({
    drop: row.drop_rate,
    recebida: row.accuracy_received,
    reconstruida: row.accuracy_reconstructed,
  }));

  return (
    <section className="grid gap-6 lg:grid-cols-[420px_1fr]">
      <div className="flex flex-col gap-4">
        <div className="rounded-xl border border-line bg-panel p-5">
          <h2 className="text-xl font-semibold text-neon font-mono mb-2">
            Treino do Classificador
          </h2>
          <p className="text-xs text-slate-400 font-mono mb-4">
            Treine o classificador por dataset e valide a preservacao semantica.
          </p>

          <div className="grid gap-4 text-sm font-mono text-slate-300">
            <div>
              <label className="block mb-2 text-xs uppercase tracking-wide text-slate-400">Dataset</label>
              <select
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                disabled={isTraining}
                className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
              >
                {DATASET_OPTIONS.map((item) => (
                  <option key={item.value} value={item.value}>{item.label}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block mb-2 text-xs uppercase tracking-wide text-slate-400">Epochs</label>
              <input
                type="range"
                min="1"
                max="20"
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                disabled={isTraining}
                className="w-full accent-neon"
              />
              <div className="flex justify-between text-[10px] text-slate-500">
                <span>1</span>
                <span>{epochs}</span>
                <span>20</span>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setAdvancedOpen((prev) => !prev)}
              className="text-left text-xs text-slate-400 hover:text-slate-200"
            >
              {advancedOpen ? "- Ocultar avancado" : "+ Opcoes avancadas"}
            </button>

            {advancedOpen && (
              <div className="grid gap-3">
                <div>
                  <label className="block mb-1 text-[10px] uppercase tracking-wide text-slate-400">Batch</label>
                  <input
                    type="number"
                    min="16"
                    max="512"
                    value={batch}
                    onChange={(e) => setBatch(Number(e.target.value))}
                    disabled={isTraining}
                    className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
                  />
                </div>
                <div>
                  <label className="block mb-1 text-[10px] uppercase tracking-wide text-slate-400">Learning rate</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={lr}
                    onChange={(e) => setLr(Number(e.target.value))}
                    disabled={isTraining}
                    className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
                  />
                </div>
                <div>
                  <label className="block mb-1 text-[10px] uppercase tracking-wide text-slate-400">Seed</label>
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value))}
                    disabled={isTraining}
                    className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-line bg-panel p-5">
          <h3 className="text-sm font-semibold text-slate-300 font-mono mb-3">Avaliacao Semantica</h3>
          <div className="grid gap-3 text-xs font-mono text-slate-300">
            <div>
              <label className="block mb-1 text-[10px] uppercase tracking-wide text-slate-400">Top-k</label>
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                disabled={isTraining}
                className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
              >
                {[1, 3, 5].map((k) => (
                  <option key={k} value={k}>Top-{k}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block mb-1 text-[10px] uppercase tracking-wide text-slate-400">Confianca minima</label>
              <input
                type="range"
                min="0.1"
                max="0.95"
                step="0.05"
                value={minConfidence}
                onChange={(e) => setMinConfidence(Number(e.target.value))}
                disabled={isTraining}
                className="w-full accent-[#7aa2ff]"
              />
              <div className="text-[10px] text-slate-500">{minConfidence.toFixed(2)}</div>
            </div>
            <div>
              <label className="block mb-1 text-[10px] uppercase tracking-wide text-slate-400">Amostras de avaliacao</label>
              <input
                type="number"
                min="50"
                max="2000"
                value={evalSamples}
                onChange={(e) => setEvalSamples(Number(e.target.value))}
                disabled={isTraining}
                className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
              />
            </div>
            <div>
              <label className="block mb-1 text-[10px] uppercase tracking-wide text-slate-400">Bits do latente</label>
              <select
                value={bits}
                onChange={(e) => setBits(Number(e.target.value))}
                disabled={isTraining}
                className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
              >
                {[4, 8, 16, 32].map((b) => (
                  <option key={b} value={b}>Int{b}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block mb-1 text-[10px] uppercase tracking-wide text-slate-400">Modelo semantico</label>
              <select
                value={semanticModel}
                onChange={(e) => setSemanticModel(e.target.value)}
                disabled={isTraining}
                className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
              >
                <option value="cnn_vae">CNN-VAE</option>
                <option value="cnn_ae">CNN-AE</option>
              </select>
            </div>
            <div>
              <label className="block mb-1 text-[10px] uppercase tracking-wide text-slate-400">Grade SNR (dB)</label>
              <input
                type="text"
                value={snrGrid}
                onChange={(e) => setSnrGrid(e.target.value)}
                disabled={isTraining}
                className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
              />
            </div>
            <div>
              <label className="block mb-1 text-[10px] uppercase tracking-wide text-slate-400">Grade Masking (drop)</label>
              <input
                type="text"
                value={maskingGrid}
                onChange={(e) => setMaskingGrid(e.target.value)}
                disabled={isTraining}
                className="w-full rounded-md border border-line bg-[#0b1220] px-3 py-2"
              />
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-line bg-panel p-5 grid gap-3">
          <button
            type="button"
            onClick={handleStart}
            disabled={isTraining || actionPending}
            className="rounded-md bg-[#073529] border border-neon text-neon px-4 py-2 font-mono text-sm hover:bg-[#0b2a22] transition disabled:opacity-50"
          >
            {isTraining ? "Treino em andamento" : "Iniciar treino"}
          </button>
          <button
            type="button"
            onClick={handleStop}
            disabled={!isTraining || actionPending}
            className="rounded-md bg-[#2a0b0b] border border-[#ff7b7b] text-[#ff9a9a] px-4 py-2 font-mono text-sm hover:bg-[#3b1111] transition disabled:opacity-50"
          >
            Parar treino
          </button>
          <button
            type="button"
            onClick={handleClearLogs}
            disabled={actionPending}
            className="rounded-md bg-[#0b1220] border border-line text-slate-400 px-4 py-2 font-mono text-xs hover:text-white transition"
          >
            Limpar logs
          </button>
          <div className="text-[10px] text-slate-500 font-mono">
            Status: {connected ? "stream ativo" : "stream desconectado"}
          </div>
        </div>

        <TerminalLogWindow logs={logs} title="classifier.log" streamStarted={streamEnabled} />
      </div>

      <div className="grid gap-6">
        <div className="rounded-xl border border-line bg-panel p-6">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-300 font-mono mb-2">
            Resultados recentes
          </h3>
          <p className="text-xs text-slate-400 font-mono mb-4">
            Curvas de treino e conclusoes sobre preservacao semantica.
          </p>
          {loadError && (
            <div className="text-xs text-[#ff9a9a] mb-4">{loadError}</div>
          )}
          {!latest?.history?.length && (
            <div className="text-xs text-slate-500">Nenhum resultado encontrado.</div>
          )}

          {history.length > 0 && (
            <div className="grid gap-6 md:grid-cols-2">
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={history}>
                    <defs>
                      <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#ffd166" stopOpacity={0.8} />
                        <stop offset="95%" stopColor="#ffd166" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#223046" />
                    <XAxis dataKey="epoch" stroke="#8a9ab4" style={{ fontSize: "10px", fontFamily: "monospace" }} />
                    <YAxis stroke="#8a9ab4" style={{ fontSize: "10px", fontFamily: "monospace" }} />
                    <Tooltip contentStyle={{ backgroundColor: "#0b1220", border: "1px solid #1d2a3d", fontFamily: "monospace" }} />
                    <Area type="monotone" dataKey="loss" stroke="#ffd166" fillOpacity={1} fill="url(#lossGrad)" />
                  </AreaChart>
                </ResponsiveContainer>
                <p className="mt-2 text-[10px] text-slate-500 font-mono">Loss por epoca</p>
              </div>

              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#223046" />
                    <XAxis dataKey="epoch" stroke="#8a9ab4" style={{ fontSize: "10px", fontFamily: "monospace" }} />
                    <YAxis stroke="#8a9ab4" style={{ fontSize: "10px", fontFamily: "monospace" }} />
                    <Tooltip contentStyle={{ backgroundColor: "#0b1220", border: "1px solid #1d2a3d", fontFamily: "monospace" }} />
                    <Legend wrapperStyle={{ fontFamily: "monospace", fontSize: "10px" }} />
                    <Line type="monotone" dataKey="train_accuracy" stroke="#00f6a2" name="Treino" />
                    <Line type="monotone" dataKey="test_accuracy" stroke="#489dff" name="Teste" />
                  </LineChart>
                </ResponsiveContainer>
                <p className="mt-2 text-[10px] text-slate-500 font-mono">Acuracia treino/teste</p>
              </div>
            </div>
          )}
        </div>

        {comparisonData.length > 0 && (
          <div className="rounded-xl border border-line bg-panel p-6">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-300 font-mono mb-4">
              Comparativo Semantico (baseline)
            </h3>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#223046" />
                  <XAxis dataKey="name" stroke="#8a9ab4" style={{ fontSize: "10px", fontFamily: "monospace" }} />
                  <YAxis stroke="#8a9ab4" domain={[0, 1]} style={{ fontSize: "10px", fontFamily: "monospace" }} />
                  <Tooltip contentStyle={{ backgroundColor: "#0b1220", border: "1px solid #1d2a3d", fontFamily: "monospace" }} />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {comparisonData.map((entry, idx) => (
                      <Cell key={idx} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {snrData.length > 0 && (
          <div className="rounded-xl border border-line bg-panel p-6">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-300 font-mono mb-4">
              Robustez vs SNR
            </h3>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={snrData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#223046" />
                  <XAxis dataKey="snr" stroke="#8a9ab4" style={{ fontSize: "10px", fontFamily: "monospace" }} />
                  <YAxis stroke="#8a9ab4" domain={[0, 1]} style={{ fontSize: "10px", fontFamily: "monospace" }} />
                  <Tooltip contentStyle={{ backgroundColor: "#0b1220", border: "1px solid #1d2a3d", fontFamily: "monospace" }} />
                  <Legend wrapperStyle={{ fontFamily: "monospace", fontSize: "10px" }} />
                  <Line type="monotone" dataKey="recebida" stroke="#ffd166" name="Recebida" />
                  <Line type="monotone" dataKey="reconstruida" stroke="#489dff" name="Reconstruida" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {maskingData.length > 0 && (
          <div className="rounded-xl border border-line bg-panel p-6">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-300 font-mono mb-4">
              Robustez vs Masking
            </h3>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={maskingData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#223046" />
                  <XAxis dataKey="drop" stroke="#8a9ab4" style={{ fontSize: "10px", fontFamily: "monospace" }} />
                  <YAxis stroke="#8a9ab4" domain={[0, 1]} style={{ fontSize: "10px", fontFamily: "monospace" }} />
                  <Tooltip contentStyle={{ backgroundColor: "#0b1220", border: "1px solid #1d2a3d", fontFamily: "monospace" }} />
                  <Legend wrapperStyle={{ fontFamily: "monospace", fontSize: "10px" }} />
                  <Line type="monotone" dataKey="recebida" stroke="#ffd166" name="Recebida" />
                  <Line type="monotone" dataKey="reconstruida" stroke="#489dff" name="Reconstruida" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
