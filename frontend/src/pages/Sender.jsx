import { useEffect, useRef, useState } from "react";
import { sgicApi } from "@/lib/api";
import ProtectedShell from "@/components/ProtectedShell";
import { MetricCard, Panel, StatusPill } from "@/components/Primitives";
import {
  HistogramChart,
  MetricVsEpochChart,
  TimingBarChart,
  SteganalysisChart,
  LossCurveChart,
} from "@/components/Charts";

const TABS = ["metrics", "visualizations", "comparison", "robustness", "ablation", "graphs"];

export default function Sender() {
  return (
    <ProtectedShell>
      <SenderContent />
    </ProtectedShell>
  );
}

function SenderContent() {
  const fileRef = useRef(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [message, setMessage] = useState(
    "This is a confidential research payload hidden using SGIC."
  );
  const [secretKey, setSecretKey] = useState("sgic-demo-2026");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [tab, setTab] = useState("metrics");

  // Side data
  const [comparison, setComparison] = useState(null);
  const [graphs, setGraphs] = useState(null);
  const [ablation, setAblation] = useState(null);
  const [ablationLoading, setAblationLoading] = useState(false);
  const [robustness, setRobustness] = useState(null);
  const [robustnessLoading, setRobustnessLoading] = useState(false);

  useEffect(() => {
    sgicApi.comparison().then(setComparison).catch(() => {});
    sgicApi.graphs().then(setGraphs).catch(() => {});
  }, []);

  const onFile = (f) => {
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setRobustness(null);
    setAblation(null);
  };

  const onEmbed = async () => {
    if (!file) {
      setError("Upload a cover image first");
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const data = await sgicApi.embed(file, message, secretKey);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || "Embed failed");
    } finally {
      setLoading(false);
    }
  };

  const onRunAblation = async () => {
    if (!file) return;
    setAblationLoading(true);
    try {
      const data = await sgicApi.ablation(file, message, secretKey);
      setAblation(data.results);
    } catch {
      setAblation([]);
    } finally {
      setAblationLoading(false);
    }
  };

  const onRunRobustness = async () => {
    if (!result) return;
    setRobustnessLoading(true);
    try {
      const data = await sgicApi.robustness(result.stego_image, secretKey, message);
      setRobustness(data.results);
    } catch {
      setRobustness([]);
    } finally {
      setRobustnessLoading(false);
    }
  };

  const downloadStego = () => {
    if (!result) return;
    const a = document.createElement("a");
    a.href = result.stego_image;
    a.download = "stego.png";
    a.click();
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      <PageHeader
        title="SENDER · EMBED PAYLOAD"
        subtitle="// Compress → encrypt → embed via key-seeded LSB diffusion"
      />

      {/* Input panel */}
      <Panel title="/ INPUT_CONTROLS" subtitle="Cover image · secret message · key">
        <div className="grid lg:grid-cols-3 gap-5">
          {/* Image upload */}
          <div>
            <label className="font-mono text-[10px] tracking-[0.25em] text-gray-500">
              [01] COVER IMAGE
            </label>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              hidden
              data-testid="sender-image-upload"
              onChange={(e) => onFile(e.target.files?.[0])}
            />
            <button
              onClick={() => fileRef.current?.click()}
              data-testid="sender-pick-button"
              className="mt-2 w-full aspect-square bg-black border border-dashed border-white/20 hover:border-[#FF3B30]/40 transition-colors flex items-center justify-center overflow-hidden"
            >
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="cover"
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="text-center font-mono text-xs text-gray-500">
                  <div className="text-2xl text-[#FF3B30] mb-1">+</div>
                  CLICK TO UPLOAD
                  <div className="text-[10px] text-gray-700 mt-1">PNG · JPG · resized to 256×256</div>
                </div>
              )}
            </button>
          </div>

          {/* Message + key */}
          <div className="lg:col-span-2 space-y-4">
            <div>
              <label className="font-mono text-[10px] tracking-[0.25em] text-gray-500">
                [02] SECRET MESSAGE
              </label>
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                rows={4}
                data-testid="secret-message-input"
                className="mt-2 w-full bg-black border border-white/10 rounded-sm px-3 py-2 font-mono text-sm text-white placeholder:text-gray-600 focus:border-[#FF3B30] focus:outline-none focus:ring-1 focus:ring-[#FF3B30]"
                placeholder="Enter the message to hide…"
              />
              <div className="mt-1 font-mono text-[10px] text-gray-600">
                {message.length} chars · {new Blob([message]).size} bytes
              </div>
            </div>

            <div>
              <label className="font-mono text-[10px] tracking-[0.25em] text-gray-500">
                [03] SECRET KEY
              </label>
              <input
                type="password"
                value={secretKey}
                onChange={(e) => setSecretKey(e.target.value)}
                data-testid="secret-key-input"
                className="mt-2 w-full bg-black border border-white/10 rounded-sm px-3 py-2 font-mono text-sm text-white placeholder:text-gray-600 focus:border-[#FF3B30] focus:outline-none focus:ring-1 focus:ring-[#FF3B30]"
                placeholder="Shared secret key"
              />
            </div>

            <div className="flex items-center gap-3 pt-2">
              <button
                onClick={onEmbed}
                disabled={loading || !file}
                data-testid="embed-send-button"
                className="bg-[#FF3B30] hover:bg-[#FF453A] disabled:opacity-40 disabled:cursor-not-allowed text-white font-mono text-sm tracking-[0.2em] px-6 py-2.5 transition-colors"
              >
                {loading ? "EMBEDDING…" : "EMBED & SEND →"}
              </button>
              {result && (
                <button
                  onClick={downloadStego}
                  data-testid="download-stego-button"
                  className="border border-white/20 hover:border-white/40 text-white font-mono text-sm tracking-[0.2em] px-6 py-2.5 transition-colors"
                >
                  ↓ DOWNLOAD STEGO
                </button>
              )}
              {error && (
                <span className="font-mono text-xs text-[#FF3B30]" data-testid="sender-error">
                  ! {error}
                </span>
              )}
            </div>
          </div>
        </div>
      </Panel>

      {/* Tabs */}
      <div className="flex flex-wrap gap-1 border-b border-white/5">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            data-testid={`tab-${t}`}
            className={`font-mono text-xs tracking-[0.25em] px-4 py-2.5 border-b-2 transition-colors ${
              tab === t
                ? "border-[#FF3B30] text-[#FF3B30]"
                : "border-transparent text-gray-500 hover:text-white"
            }`}
          >
            {t.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Metrics */}
      {tab === "metrics" && (
        <MetricsTab result={result} />
      )}

      {/* Visualizations */}
      {tab === "visualizations" && (
        <VisualizationsTab result={result} />
      )}

      {/* Comparison */}
      {tab === "comparison" && (
        <ComparisonTab comparison={comparison} />
      )}

      {/* Robustness */}
      {tab === "robustness" && (
        <RobustnessTab
          result={result}
          robustness={robustness}
          loading={robustnessLoading}
          onRun={onRunRobustness}
        />
      )}

      {/* Ablation */}
      {tab === "ablation" && (
        <AblationTab
          file={file}
          ablation={ablation}
          loading={ablationLoading}
          onRun={onRunAblation}
        />
      )}

      {/* Graphs */}
      {tab === "graphs" && <GraphsTab graphs={graphs} />}
    </div>
  );
}

function PageHeader({ title, subtitle }) {
  return (
    <div className="border-b border-white/5 pb-4">
      <div className="font-mono text-[10px] tracking-[0.4em] text-[#FF3B30]">
        {subtitle}
      </div>
      <h1 className="font-mono text-3xl text-white mt-1">{title}</h1>
    </div>
  );
}

function MetricsTab({ result }) {
  if (!result)
    return <EmptyState text="Run an embed to view real-time metrics." />;
  const m = result.metrics;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <MetricCard label="PSNR" value={m.psnr} unit="dB" accent testid="metric-psnr" />
        <MetricCard label="SSIM" value={m.ssim} testid="metric-ssim" />
        <MetricCard label="BPP" value={m.bpp} testid="metric-bpp" />
        <MetricCard label="CHI-SQUARE" value={m.chi_square} testid="metric-chi" />
        <MetricCard label="RS |R-S|" value={m.rs_analysis} testid="metric-rs" />
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="PAYLOAD BITS" value={result.payload_bits} testid="metric-bits" />
        <MetricCard label="RAW BYTES" value={result.raw_message_bytes} testid="metric-raw" />
        <MetricCard label="ENCRYPTED" value={result.encrypted_bytes} unit="B" testid="metric-enc" />
        <MetricCard label="TIME" value={result.elapsed_ms} unit="ms" testid="metric-time" />
      </div>
    </div>
  );
}

function VisualizationsTab({ result }) {
  if (!result)
    return <EmptyState text="Run an embed to view side-by-side visualizations." />;

  return (
    <div className="space-y-4">
      <Panel title="/ SIDE_BY_SIDE" subtitle="Cover · Stego · Difference ×50">
        <div className="grid md:grid-cols-3 gap-4">
          {[
            { src: result.cover_image, label: "01 · COVER" },
            { src: result.stego_image, label: "02 · STEGO (SGIC)", accent: true },
            { src: result.diff_image, label: "03 · DIFF ×50" },
          ].map((v) => (
            <div key={v.label} className="space-y-2">
              <div
                className={`font-mono text-[10px] tracking-[0.25em] ${
                  v.accent ? "text-[#FF3B30]" : "text-gray-500"
                }`}
              >
                {v.label}
              </div>
              <img
                src={v.src}
                alt={v.label}
                className={`w-full aspect-square object-cover border ${
                  v.accent ? "border-[#FF3B30]/40" : "border-white/10"
                }`}
              />
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="/ HISTOGRAMS" subtitle="RGB distribution · cover (solid) vs stego (dashed)">
        <div className="grid md:grid-cols-3 gap-6">
          {[
            { ch: "R", title: "RED" },
            { ch: "G", title: "GREEN" },
            { ch: "B", title: "BLUE" },
          ].map((c) => (
            <div key={c.ch}>
              <div className="font-mono text-[10px] tracking-[0.25em] text-gray-500 mb-2">
                {c.title} · CH
              </div>
              <HistogramChart histograms={result.histograms} channel={c.ch} />
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function ComparisonTab({ comparison }) {
  if (!comparison) return <EmptyState text="Loading comparison…" />;
  return (
    <Panel
      title="/ METHOD_COMPARISON"
      subtitle="Aggregate scores across baselines · PROPOSED highlighted"
    >
      <div className="overflow-x-auto">
        <table className="w-full font-mono text-sm" data-testid="comparison-table">
          <thead>
            <tr className="border-b border-white/20 text-left text-gray-400">
              <th className="py-3 pr-4 text-[10px] tracking-[0.25em]">METHOD</th>
              <th className="py-3 px-4 text-[10px] tracking-[0.25em]">PSNR (dB)</th>
              <th className="py-3 px-4 text-[10px] tracking-[0.25em]">SSIM</th>
              <th className="py-3 px-4 text-[10px] tracking-[0.25em]">CHI-SQ</th>
              <th className="py-3 px-4 text-[10px] tracking-[0.25em]">ENCRYPTION</th>
              <th className="py-3 px-4 text-[10px] tracking-[0.25em]">COMPRESSION</th>
              <th className="py-3 px-4 text-[10px] tracking-[0.25em]">SECURITY</th>
            </tr>
          </thead>
          <tbody>
            {comparison.methods.map((m) => {
              const proposed = m.method === "PROPOSED";
              return (
                <tr
                  key={m.method}
                  data-testid={`row-${m.method}`}
                  className={
                    proposed
                      ? "border-b border-[#FF3B30]/30 bg-[#FF3B30]/5 font-bold text-[#FF3B30]"
                      : "border-b border-white/5 text-gray-300"
                  }
                >
                  <td className="py-3 pr-4">
                    {proposed && <span className="mr-2">▶</span>}
                    {m.method}
                  </td>
                  <td className="py-3 px-4">{m.psnr.toFixed(2)}</td>
                  <td className="py-3 px-4">{m.ssim.toFixed(4)}</td>
                  <td className="py-3 px-4">{m.chi_square.toFixed(2)}</td>
                  <td className="py-3 px-4">{m.encryption}</td>
                  <td className="py-3 px-4">{m.compression}</td>
                  <td className="py-3 px-4">{m.security}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-3 font-mono text-[10px] text-gray-600">
        ▶ PROPOSED row = SGIC (this implementation)
      </div>
    </Panel>
  );
}

function RobustnessTab({ result, robustness, loading, onRun }) {
  if (!result)
    return <EmptyState text="Embed first, then run attacks against the stego image." />;
  return (
    <Panel
      title="/ ROBUSTNESS_TESTING"
      subtitle="Apply attacks → attempt extraction → report success"
      action={
        <button
          onClick={onRun}
          disabled={loading}
          data-testid="run-robustness-button"
          className="bg-[#FF3B30] hover:bg-[#FF453A] disabled:opacity-40 text-white font-mono text-xs tracking-[0.2em] px-4 py-2"
        >
          {loading ? "RUNNING…" : "RUN ATTACKS"}
        </button>
      }
    >
      {!robustness ? (
        <div className="font-mono text-xs text-gray-500">
          Click <span className="text-[#FF3B30]">RUN ATTACKS</span> to test against
          Gaussian noise, JPEG compression, and resize attacks.
        </div>
      ) : (
        <div className="grid md:grid-cols-3 gap-4">
          {robustness.map((r) => (
            <div
              key={r.attack}
              data-testid={`robustness-${r.attack}`}
              className="bg-black border border-white/10 p-4 space-y-3"
            >
              <div className="flex items-center justify-between">
                <div className="font-mono text-xs text-white">{r.attack}</div>
                <StatusPill ok={r.passed}>{r.passed ? "PASS" : "FAIL"}</StatusPill>
              </div>
              <img
                src={r.preview}
                alt={r.attack}
                className="w-full aspect-square object-cover border border-white/5"
              />
              <div className="grid grid-cols-3 font-mono text-[10px] text-gray-500">
                <div>
                  <div className="text-gray-600">PSNR</div>
                  <div className="text-white text-sm">{r.psnr}</div>
                </div>
                <div>
                  <div className="text-gray-600">SSIM</div>
                  <div className="text-white text-sm">{r.ssim}</div>
                </div>
                <div>
                  <div className="text-gray-600">BER</div>
                  <div className="text-white text-sm">
                    {(r.bit_error_rate * 100).toFixed(2)}%
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}

function AblationTab({ file, ablation, loading, onRun }) {
  if (!file)
    return <EmptyState text="Upload an image first to run the ablation study." />;
  return (
    <Panel
      title="/ ABLATION_STUDY"
      subtitle="No-Encryption · No-Diffusion · Full-SGIC"
      action={
        <button
          onClick={onRun}
          disabled={loading}
          data-testid="run-ablation-button"
          className="bg-[#FF3B30] hover:bg-[#FF453A] disabled:opacity-40 text-white font-mono text-xs tracking-[0.2em] px-4 py-2"
        >
          {loading ? "RUNNING…" : "RUN ABLATION"}
        </button>
      }
    >
      {!ablation ? (
        <div className="font-mono text-xs text-gray-500">
          Click <span className="text-[#FF3B30]">RUN ABLATION</span> to compare
          configurations on the uploaded image.
        </div>
      ) : (
        <div className="grid md:grid-cols-3 gap-4">
          {ablation.map((a) => {
            const proposed = a.config.includes("Full");
            return (
              <div
                key={a.config}
                data-testid={`ablation-${a.config}`}
                className={`p-5 border ${
                  proposed
                    ? "border-[#FF3B30]/40 bg-[#FF3B30]/5"
                    : "border-white/10 bg-black"
                }`}
              >
                <div
                  className={`font-mono text-xs tracking-[0.2em] ${
                    proposed ? "text-[#FF3B30]" : "text-gray-400"
                  }`}
                >
                  {a.config.toUpperCase()}
                </div>
                <div className="mt-4 space-y-2 font-mono text-xs">
                  <Row label="PSNR" value={`${a.psnr} dB`} accent={proposed} />
                  <Row label="SSIM" value={a.ssim} accent={proposed} />
                  <Row label="BPP" value={a.bpp} accent={proposed} />
                  <Row label="CHI-SQ" value={a.chi_square} accent={proposed} />
                  <Row label="RS" value={a.rs_analysis} accent={proposed} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </Panel>
  );
}

function GraphsTab({ graphs }) {
  if (!graphs) return <EmptyState text="Loading reference graphs…" />;
  return (
    <div className="space-y-4">
      <div className="grid md:grid-cols-2 gap-4">
        <Panel title="/ PSNR_VS_EPOCH" subtitle="Average PSNR across training epochs">
          <MetricVsEpochChart epochs={graphs.epochs} series={graphs.psnr_vs_epoch} ylabel="PSNR (dB)" />
        </Panel>
        <Panel title="/ SSIM_VS_EPOCH" subtitle="Structural similarity progression">
          <MetricVsEpochChart epochs={graphs.epochs} series={graphs.ssim_vs_epoch} ylabel="SSIM" />
        </Panel>
        <Panel title="/ TIMING" subtitle="Avg embedding time per method (ms)">
          <TimingBarChart timing={graphs.timing_ms} />
        </Panel>
        <Panel title="/ STEGANALYSIS" subtitle="Lower = harder to detect">
          <SteganalysisChart steganalysis={graphs.steganalysis} />
        </Panel>
      </div>
      <Panel title="/ TRAINING_LOSS" subtitle="DDPM noise-prediction MSE">
        <LossCurveChart data={graphs.loss_curves} />
      </Panel>
    </div>
  );
}

function Row({ label, value, accent }) {
  return (
    <div className="flex items-center justify-between border-b border-white/5 py-1.5">
      <span className="text-[10px] tracking-[0.25em] text-gray-500">{label}</span>
      <span className={accent ? "text-[#FF3B30]" : "text-white"}>{value}</span>
    </div>
  );
}

function EmptyState({ text }) {
  return (
    <div className="bg-[#141414] border border-dashed border-white/10 p-12 text-center font-mono text-xs text-gray-500">
      {text}
    </div>
  );
}
