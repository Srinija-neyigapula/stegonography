// Reusable UI primitives for the SGIC dashboard

export function MetricCard({ label, value, unit, accent = false, testid }) {
  return (
    <div
      data-testid={testid}
      className={`bg-[#141414] border ${
        accent ? "border-[#FF3B30]/40" : "border-white/10"
      } p-5`}
    >
      <div className="font-mono text-[10px] uppercase tracking-[0.25em] text-gray-500">
        {label}
      </div>
      <div className="mt-3 flex items-baseline gap-1">
        <span
          className={`font-mono text-2xl font-bold ${
            accent ? "text-[#FF3B30]" : "text-white"
          }`}
        >
          {value}
        </span>
        {unit && (
          <span className="font-mono text-xs text-gray-500">{unit}</span>
        )}
      </div>
    </div>
  );
}

export function Panel({ title, subtitle, children, action, testid }) {
  return (
    <section
      data-testid={testid}
      className="bg-[#141414] border border-white/10"
    >
      <div className="px-5 py-3 flex items-center justify-between border-b border-white/5">
        <div>
          <div className="font-mono text-[10px] tracking-[0.3em] text-[#FF3B30]">
            {title}
          </div>
          {subtitle && (
            <div className="font-mono text-xs text-gray-500 mt-0.5">{subtitle}</div>
          )}
        </div>
        {action}
      </div>
      <div className="p-5">{children}</div>
    </section>
  );
}

export function StatusPill({ ok, children }) {
  return (
    <span
      className={`inline-flex items-center gap-2 font-mono text-[10px] tracking-[0.2em] px-2.5 py-1 border ${
        ok
          ? "border-[#32D74B]/40 text-[#32D74B] bg-[#32D74B]/5"
          : "border-[#FF3B30]/40 text-[#FF3B30] bg-[#FF3B30]/5"
      }`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${
          ok ? "bg-[#32D74B]" : "bg-[#FF3B30]"
        }`}
      />
      {children}
    </span>
  );
}
