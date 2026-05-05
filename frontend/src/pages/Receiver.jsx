import { useRef, useState } from "react";
import { sgicApi } from "@/lib/api";
import ProtectedShell from "@/components/ProtectedShell";
import { Panel, StatusPill, MetricCard } from "@/components/Primitives";

export default function Receiver() {
  return (
    <ProtectedShell>
      <ReceiverContent />
    </ProtectedShell>
  );
}

function ReceiverContent() {
  const fileRef = useRef(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [secretKey, setSecretKey] = useState("sgic-demo-2026");
  const [originalMessage, setOriginalMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const onFile = (f) => {
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
  };

  const onExtract = async () => {
    if (!file) {
      setError("Upload a stego image first");
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const data = await sgicApi.extract(file, secretKey, originalMessage);
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || "Extraction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      <div className="border-b border-white/5 pb-4">
        <div className="font-mono text-[10px] tracking-[0.4em] text-[#FF3B30]">
          // Decrypt → decompress → verify
        </div>
        <h1 className="font-mono text-3xl text-white mt-1">RECEIVER · EXTRACT PAYLOAD</h1>
      </div>

      <Panel title="/ INPUT_CONTROLS" subtitle="Stego image · shared key · (optional) ground truth">
        <div className="grid lg:grid-cols-3 gap-5">
          <div>
            <label className="font-mono text-[10px] tracking-[0.25em] text-gray-500">
              [01] STEGO IMAGE
            </label>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              hidden
              data-testid="receiver-image-upload"
              onChange={(e) => onFile(e.target.files?.[0])}
            />
            <button
              onClick={() => fileRef.current?.click()}
              data-testid="receiver-pick-button"
              className="mt-2 w-full aspect-square bg-black border border-dashed border-white/20 hover:border-[#FF3B30]/40 transition-colors flex items-center justify-center overflow-hidden"
            >
              {previewUrl ? (
                <img src={previewUrl} alt="stego" className="w-full h-full object-cover" />
              ) : (
                <div className="text-center font-mono text-xs text-gray-500">
                  <div className="text-2xl text-[#FF3B30] mb-1">+</div>
                  CLICK TO UPLOAD STEGO
                </div>
              )}
            </button>
          </div>

          <div className="lg:col-span-2 space-y-4">
            <div>
              <label className="font-mono text-[10px] tracking-[0.25em] text-gray-500">
                [02] SECRET KEY
              </label>
              <input
                type="password"
                value={secretKey}
                onChange={(e) => setSecretKey(e.target.value)}
                data-testid="receiver-secret-key"
                className="mt-2 w-full bg-black border border-white/10 rounded-sm px-3 py-2 font-mono text-sm text-white focus:border-[#FF3B30] focus:outline-none focus:ring-1 focus:ring-[#FF3B30]"
              />
            </div>
            <div>
              <label className="font-mono text-[10px] tracking-[0.25em] text-gray-500">
                [03] EXPECTED MESSAGE (optional · for BER)
              </label>
              <textarea
                rows={3}
                value={originalMessage}
                onChange={(e) => setOriginalMessage(e.target.value)}
                data-testid="receiver-original-message"
                className="mt-2 w-full bg-black border border-white/10 rounded-sm px-3 py-2 font-mono text-sm text-white placeholder:text-gray-600 focus:border-[#FF3B30] focus:outline-none focus:ring-1 focus:ring-[#FF3B30]"
                placeholder="Paste the original message to compute Bit Error Rate"
              />
            </div>
            <div className="flex items-center gap-3 pt-2">
              <button
                onClick={onExtract}
                disabled={loading || !file}
                data-testid="extract-message-button"
                className="bg-[#FF3B30] hover:bg-[#FF453A] disabled:opacity-40 text-white font-mono text-sm tracking-[0.2em] px-6 py-2.5 transition-colors"
              >
                {loading ? "EXTRACTING…" : "EXTRACT MESSAGE →"}
              </button>
              {error && (
                <span className="font-mono text-xs text-[#FF3B30]" data-testid="receiver-error">
                  ! {error}
                </span>
              )}
            </div>
          </div>
        </div>
      </Panel>

      {result && (
        <>
          <Panel
            title="/ VERIFICATION"
            subtitle="Decryption + integrity check"
            action={
              <StatusPill ok={result.success}>
                {result.success ? "MESSAGE RECOVERED" : "RECOVERY FAILED"}
              </StatusPill>
            }
          >
            <div className="grid md:grid-cols-3 gap-4">
              <MetricCard
                label="STATUS"
                value={result.success ? "OK" : "FAIL"}
                accent={result.success}
                testid="receiver-status"
              />
              <MetricCard
                label="EXTRACT TIME"
                value={result.elapsed_ms}
                unit="ms"
                testid="receiver-time"
              />
              <MetricCard
                label="BIT ERROR RATE"
                value={
                  result.bit_error_rate === null
                    ? "—"
                    : `${(result.bit_error_rate * 100).toFixed(2)}%`
                }
                accent={result.bit_error_rate === 0}
                testid="receiver-ber"
              />
            </div>
          </Panel>

          <Panel title="/ RECOVERED_MESSAGE" subtitle="Decrypted plaintext payload">
            {result.success ? (
              <pre
                data-testid="recovered-message-display"
                className="bg-black border border-white/5 p-4 font-mono text-sm text-[#32D74B] whitespace-pre-wrap break-words"
              >
                {result.recovered_message}
              </pre>
            ) : (
              <div className="bg-black border border-[#FF3B30]/30 p-4 font-mono text-sm text-[#FF3B30]">
                {result.error || "Failed to recover message — wrong key or corrupted image."}
              </div>
            )}
          </Panel>

          {originalMessage && (
            <Panel title="/ DIFF · ORIGINAL_VS_RECOVERED" subtitle="Side-by-side comparison">
              <div className="grid md:grid-cols-2 gap-4 font-mono text-xs">
                <div>
                  <div className="text-gray-500 tracking-[0.25em] mb-2">ORIGINAL</div>
                  <pre
                    className="bg-black border border-white/10 p-4 text-gray-300 whitespace-pre-wrap break-words"
                    data-testid="receiver-original-block"
                  >
                    {originalMessage}
                  </pre>
                </div>
                <div>
                  <div
                    className={`tracking-[0.25em] mb-2 ${
                      originalMessage === result.recovered_message
                        ? "text-[#32D74B]"
                        : "text-[#FF3B30]"
                    }`}
                  >
                    RECOVERED · {originalMessage === result.recovered_message ? "MATCH" : "DIFF"}
                  </div>
                  <pre
                    className={`bg-black border p-4 whitespace-pre-wrap break-words ${
                      originalMessage === result.recovered_message
                        ? "border-[#32D74B]/30 text-[#32D74B]"
                        : "border-[#FF3B30]/30 text-[#FF3B30]"
                    }`}
                    data-testid="receiver-recovered-block"
                  >
                    {result.recovered_message || "(empty)"}
                  </pre>
                </div>
              </div>
            </Panel>
          )}
        </>
      )}
    </div>
  );
}
