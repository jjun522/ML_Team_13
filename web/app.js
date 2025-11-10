const MODE_ORDER = ["hybrid", "cf", "cbf"];
const LABEL = {
  hybrid: "Hybrid",
  cf: "Collaborative Filtering",
  cbf: "Content Based Filtering",
};

const tableBody = document.getElementById("table-body");
const modeSelect = document.getElementById("mode");
const meta = document.getElementById("meta");
const statusBox = document.getElementById("status");
const metricsPanel = document.getElementById("metrics-panel");

async function loadData() {
  try {
    statusBox.textContent = "Loading latest data…";
    const [recsResp, metricsResp] = await Promise.all([
      fetch("../src/result/recommendations.json", { cache: "no-store" }),
      fetch("../src/result/evaluation_metrics.json", { cache: "no-store" }),
    ]);
    if (!recsResp.ok) {
      throw new Error(`Recommendations HTTP ${recsResp.status}`);
    }
    const recsData = await recsResp.json();
    const metricsData = metricsResp.ok ? await metricsResp.json() : null;
    statusBox.textContent = "";
    updateMeta(recsData);
    renderMetrics(metricsData);
    populateTable(recsData, modeSelect.value);
    modeSelect.addEventListener("change", () =>
      populateTable(recsData, modeSelect.value)
    );
  } catch (err) {
    tableBody.innerHTML = `<tr><td colspan="5" class="muted">Failed to load recommendations.</td></tr>`;
    statusBox.textContent = `Error: ${err.message}`;
    metricsPanel.textContent = "Failed to load metrics.";
  }
}

function updateMeta(data) {
  if (!data) return;
  meta.innerHTML = `User: <strong>${data.user ?? "N/A"}</strong> · Updated: ${
    data.generated_at ?? "N/A"
  }`;
}

function renderMetrics(metrics) {
  if (!metrics) {
    metricsPanel.textContent = "Metrics file is missing.";
    return;
  }
  const rows = [
    { label: "RMSE", value: formatNumber(metrics.rmse) },
    {
      label: `Precision@${metrics.k ?? "-"}`,
      value: formatNumber(metrics.precision_at_k),
    },
    {
      label: `Recall@${metrics.k ?? "-"}`,
      value: formatNumber(metrics.recall_at_k),
    },
    {
      label: `NDCG@${metrics.k ?? "-"}`,
      value: formatNumber(metrics.ndcg_at_k),
    },
    {
      label: "Evaluated Users",
      value: metrics.evaluated_users ?? "-",
    },
  ];
  metricsPanel.innerHTML = rows
    .map(
      (row) => `
      <div class="metric-card">
        <h3>${row.label}</h3>
        <p>${row.value}</p>
      </div>
    `
    )
    .join("");
}

function formatNumber(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(4);
}

function populateTable(data, mode) {
  const list = data?.[mode] ?? [];
  if (!list.length) {
    tableBody.innerHTML = `<tr><td colspan="5" class="muted">No ${LABEL[mode]} recommendations.</td></tr>`;
    return;
  }
  tableBody.innerHTML = list
    .map(
      (item) => `
        <tr>
          <td>${item.rank}</td>
          <td>${item.beer_name}</td>
          <td>${item.cf_score?.toFixed?.(2) ?? item.cf_score ?? "-"}</td>
          <td>${item.cbf_score?.toFixed?.(2) ?? item.cbf_score ?? "-"}</td>
          <td>${item.hybrid_score?.toFixed?.(2) ?? item.hybrid_score ?? "-"}</td>
        </tr>
      `
    )
    .join("");
}

modeSelect.value = MODE_ORDER[0];
loadData();
