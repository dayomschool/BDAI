const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const sampleBtn = document.getElementById("sampleBtn");
const uploadCard = document.getElementById("uploadCard");
const loading = document.getElementById("loading");
const loadingText = document.getElementById("loadingText");
const results = document.getElementById("results");
const summaryBar = document.getElementById("summaryBar");
const resultList = document.getElementById("resultList");
const resetBtn = document.getElementById("resetBtn");
const errorMsg = document.getElementById("errorMsg");
const mainTitle = document.getElementById("mainTitle");
const mainSubtitle = document.getElementById("mainSubtitle");

const LOADING_MESSAGES = [
  "모델이 거래를 분석하고 있습니다...",
  "위험 등급을 계산하는 중입니다...",
  "결과를 정리하고 있습니다...",
];

const DEFAULT_TITLE = "FDS 위험 거래 탐지 데모";
const DEFAULT_SUBTITLE = "거래 데이터를 넣으면 LightGBM 모델이 실시간으로 정상/이상 거래를 판별합니다.";
const RESULTS_TITLE = "분석 결과 페이지";

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) handleFile(file);
});

sampleBtn.addEventListener("click", () => {
  runPredict(() =>
    fetch("/api/predict-sample", { method: "POST" })
  );
});

resetBtn.addEventListener("click", () => {
  results.hidden = true;
  uploadCard.hidden = false;
  fileInput.value = "";
  errorMsg.textContent = "";
  setHeader(DEFAULT_TITLE, DEFAULT_SUBTITLE, false);
});

function setHeader(title, subtitle, small) {
  mainTitle.textContent = title;
  mainSubtitle.textContent = subtitle;
  mainSubtitle.classList.toggle("subtitle-small", small);
}

function handleFile(file) {
  if (!file.name.toLowerCase().endsWith(".csv")) {
    errorMsg.textContent = "CSV 파일만 업로드할 수 있어요.";
    return;
  }
  const formData = new FormData();
  formData.append("file", file);

  runPredict(() =>
    fetch("/api/predict", { method: "POST", body: formData })
  );
}

async function runPredict(requestFn) {
  errorMsg.textContent = "";
  uploadCard.hidden = true;
  results.hidden = true;
  loading.hidden = false;

  let msgIndex = 0;
  loadingText.textContent = LOADING_MESSAGES[0];
  const msgTimer = setInterval(() => {
    msgIndex = (msgIndex + 1) % LOADING_MESSAGES.length;
    loadingText.textContent = LOADING_MESSAGES[msgIndex];
  }, 700);

  const minDelay = new Promise((resolve) => setTimeout(resolve, 1400));

  try {
    const [response] = await Promise.all([requestFn(), minDelay]);
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "분석 중 오류가 발생했습니다.");
    }

    renderResults(data);
    loading.hidden = true;
    results.hidden = false;
  } catch (err) {
    loading.hidden = true;
    uploadCard.hidden = false;
    errorMsg.textContent = err.message || "분석 중 오류가 발생했습니다.";
  } finally {
    clearInterval(msgTimer);
  }
}

function renderResults(data) {
  setHeader(
    RESULTS_TITLE,
    `임계값(${data.threshold}%) 이상이면 이상거래로 판정합니다`,
    true
  );

  const fraudCount = data.results.filter((r) => r.prediction === "이상거래").length;
  const normalCount = data.count - fraudCount;

  const withActual = data.results.filter((r) => r.actual !== null);
  const correctCount = withActual.filter((r) => r.actual === r.prediction).length;

  const summaryItems = [
    { value: `${data.count}건`, label: "전체 분석 건수" },
    { value: `${normalCount}건`, label: "정상거래 판정" },
    { value: `${fraudCount}건`, label: "이상거래 판정" },
  ];

  if (withActual.length > 0) {
    const acc = ((correctCount / withActual.length) * 100).toFixed(1);
    summaryItems.push({ value: `${acc}%`, label: "실제 정답과 일치율" });
  }

  summaryBar.innerHTML = summaryItems
    .map(
      (item) => `
      <div class="summary-item">
        <div class="value">${item.value}</div>
        <div class="label">${item.label}</div>
      </div>`
    )
    .join("");

  resultList.innerHTML = data.results
    .map((r) => {
      const predictionClass =
        r.prediction === "이상거래" ? "tag-prediction-fraud" : "tag-prediction-normal";

      let actualTag = "";
      if (r.actual !== null) {
        const match = r.actual === r.prediction;
        actualTag = `<span class="tag ${
          match ? "tag-actual-match" : "tag-actual-mismatch"
        }">실제: ${r.actual}${match ? " (일치)" : " (불일치)"}</span>`;
      }

      return `
        <div class="result-card" style="--tier-color: ${r.tier_color}">
          <div class="result-main">
            <div class="result-top">
              <span class="result-index">#${r.row}</span>
              ${r.trans_date_trans_time ? `<span class="result-time">${r.trans_date_trans_time}</span>` : ""}
            </div>
            <div class="result-meta">
              <b>${r.category ?? "-"}</b> · ${r.amt != null ? "$" + r.amt.toLocaleString() : "-"} ·
              ${r.trans_hour != null ? r.trans_hour + "시" : "-"}
            </div>
            ${r.cluster_type && r.cluster_type !== "해당없음(일반)" ? `<div class="result-cluster">유형: ${r.cluster_type}</div>` : ""}
          </div>
          <div class="result-side">
            <div class="result-prob">${r.probability}%</div>
            <div class="result-tags">
              <span class="tag tag-tier">${r.tier}</span>
              <span class="tag ${predictionClass}">${r.prediction}</span>
              ${actualTag}
            </div>
          </div>
        </div>`;
    })
    .join("");
}
