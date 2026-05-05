import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, ResponsiveContainer,
} from "recharts";

const COLORS = {
  LSB: "#2196F3",
  LSB_XOR: "#4CAF50",
  DCT: "#FF9F0A",
  GAN: "#9C27B0",
  PROPOSED: "#FF3B30",
  cover: "#0A84FF",
  stego: "#FF3B30",
  diff: "#A1A1AA",
  R: "#EF4444",
  G: "#22C55E",
  B: "#3B82F6",
};

const tooltipStyle = {
  backgroundColor: "#0A0A0A",
  border: "1px solid rgba(255,255,255,0.1)",
  fontFamily: "IBM Plex Mono, monospace",
  fontSize: "11px",
  color: "#fff",
};

export function HistogramChart({ histograms, channel = "all" }) {
  // histograms = { cover, stego, diff } each is [[R..],[G..],[B..]]
  const bins = histograms.cover[0].length;
  const data = Array.from({ length: bins }, (_, i) => {
    const x = Math.round((i / bins) * 255);
    const row = { bin: x };
    if (channel === "all" || channel === "R") {
      row.cover_R = histograms.cover[0][i];
      row.stego_R = histograms.stego[0][i];
    }
    if (channel === "all" || channel === "G") {
      row.cover_G = histograms.cover[1][i];
      row.stego_G = histograms.stego[1][i];
    }
    if (channel === "all" || channel === "B") {
      row.cover_B = histograms.cover[2][i];
      row.stego_B = histograms.stego[2][i];
    }
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="bin" tick={{ fill: "#71717A", fontSize: 10, fontFamily: "IBM Plex Mono" }} />
        <YAxis tick={{ fill: "#71717A", fontSize: 10, fontFamily: "IBM Plex Mono" }} />
        <Tooltip contentStyle={tooltipStyle} />
        {(channel === "all" || channel === "R") && (
          <>
            <Line type="monotone" dataKey="cover_R" stroke={COLORS.R} strokeOpacity={0.5} dot={false} strokeWidth={1.2} />
            <Line type="monotone" dataKey="stego_R" stroke={COLORS.R} dot={false} strokeWidth={1.5} strokeDasharray="3 3" />
          </>
        )}
        {(channel === "all" || channel === "G") && (
          <>
            <Line type="monotone" dataKey="cover_G" stroke={COLORS.G} strokeOpacity={0.5} dot={false} strokeWidth={1.2} />
            <Line type="monotone" dataKey="stego_G" stroke={COLORS.G} dot={false} strokeWidth={1.5} strokeDasharray="3 3" />
          </>
        )}
        {(channel === "all" || channel === "B") && (
          <>
            <Line type="monotone" dataKey="cover_B" stroke={COLORS.B} strokeOpacity={0.5} dot={false} strokeWidth={1.2} />
            <Line type="monotone" dataKey="stego_B" stroke={COLORS.B} dot={false} strokeWidth={1.5} strokeDasharray="3 3" />
          </>
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function MetricVsEpochChart({ epochs, series, ylabel }) {
  const data = epochs.map((e, i) => {
    const row = { epoch: e };
    Object.keys(series).forEach((m) => {
      row[m] = series[m][i];
    });
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="epoch" tick={{ fill: "#71717A", fontSize: 11, fontFamily: "IBM Plex Mono" }} label={{ value: "Epoch", fill: "#71717A", fontSize: 11, dy: 12, fontFamily: "IBM Plex Mono" }} />
        <YAxis tick={{ fill: "#71717A", fontSize: 11, fontFamily: "IBM Plex Mono" }} label={{ value: ylabel, angle: -90, fill: "#71717A", fontSize: 11, dx: -10, fontFamily: "IBM Plex Mono" }} />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ fontFamily: "IBM Plex Mono", fontSize: 11, color: "#A1A1AA" }} />
        {Object.keys(series).map((m) => (
          <Line
            key={m}
            type="monotone"
            dataKey={m}
            stroke={COLORS[m]}
            strokeWidth={m === "PROPOSED" ? 2.5 : 1.5}
            dot={{ r: m === "PROPOSED" ? 4 : 3, fill: COLORS[m] }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function TimingBarChart({ timing }) {
  const data = Object.keys(timing).map((m) => ({ method: m, ms: timing[m] }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
        <XAxis dataKey="method" tick={{ fill: "#71717A", fontSize: 11, fontFamily: "IBM Plex Mono" }} />
        <YAxis tick={{ fill: "#71717A", fontSize: 11, fontFamily: "IBM Plex Mono" }} label={{ value: "ms", angle: -90, fill: "#71717A", fontSize: 11, dx: -10, fontFamily: "IBM Plex Mono" }} />
        <Tooltip contentStyle={tooltipStyle} />
        <Bar dataKey="ms" fill="#FF3B30" radius={[2, 2, 0, 0]}>
          {data.map((d, i) => (
            <Bar key={i} dataKey="ms" fill={COLORS[d.method]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function SteganalysisChart({ steganalysis }) {
  const data = Object.keys(steganalysis).map((m) => ({
    method: m,
    chi: steganalysis[m].chi_square,
    rs: steganalysis[m].rs_diff,
  }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
        <XAxis dataKey="method" tick={{ fill: "#71717A", fontSize: 11, fontFamily: "IBM Plex Mono" }} />
        <YAxis yAxisId="left" tick={{ fill: "#71717A", fontSize: 11, fontFamily: "IBM Plex Mono" }} />
        <YAxis yAxisId="right" orientation="right" tick={{ fill: "#71717A", fontSize: 11, fontFamily: "IBM Plex Mono" }} />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ fontFamily: "IBM Plex Mono", fontSize: 11, color: "#A1A1AA" }} />
        <Bar yAxisId="left" dataKey="chi" name="Chi-Square" fill="#FF3B30" radius={[2, 2, 0, 0]} />
        <Bar yAxisId="right" dataKey="rs" name="RS |R-S|" fill="#0A84FF" radius={[2, 2, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function LossCurveChart({ data }) {
  const rows = data.epochs.map((e, i) => ({
    epoch: e,
    train: data.train[i],
    val: data.val[i],
  }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={rows} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.05)" />
        <XAxis dataKey="epoch" tick={{ fill: "#71717A", fontSize: 11, fontFamily: "IBM Plex Mono" }} />
        <YAxis tick={{ fill: "#71717A", fontSize: 11, fontFamily: "IBM Plex Mono" }} />
        <Tooltip contentStyle={tooltipStyle} />
        <Legend wrapperStyle={{ fontFamily: "IBM Plex Mono", fontSize: 11, color: "#A1A1AA" }} />
        <Line type="monotone" dataKey="train" name="Train" stroke="#FF3B30" strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="val" name="Validation" stroke="#0A84FF" strokeWidth={2} dot={false} strokeDasharray="3 3" />
      </LineChart>
    </ResponsiveContainer>
  );
}
