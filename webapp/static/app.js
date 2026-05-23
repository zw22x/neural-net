const testSelect = document.querySelector("#testSelect");
const epochsInput = document.querySelector("#epochsInput");
const learningRateInput = document.querySelector("#learningRateInput");
const runButton = document.querySelector("#runButton");
const passText = document.querySelector("#passText");
const accuracyText = document.querySelector("#accuracyText");
const lossText = document.querySelector("#lossText");
const testSummary = document.querySelector("#testSummary");
const predictionList = document.querySelector("#predictionList");

const decisionCanvas = document.querySelector("#decisionCanvas");
const networkCanvas = document.querySelector("#networkCanvas");
const lossCanvas = document.querySelector("#lossCanvas");

let tests = [];
let activeResult = null;

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function colorForPrediction(prediction, alpha = 1) {
  const t = clamp(prediction, 0, 1);
  const low = [47, 111, 189];
  const high = [31, 157, 115];
  const mixed = low.map((channel, index) => Math.round(lerp(channel, high[index], t)));
  return `rgba(${mixed[0]}, ${mixed[1]}, ${mixed[2]}, ${alpha})`;
}

function drawDecision(result) {
  const { ctx, width, height } = setupCanvas(decisionCanvas);
  const pad = 34;
  const plotW = width - pad * 2;
  const plotH = height - pad * 2;
  const bounds = result.bounds;
  const steps = Math.round(Math.sqrt(result.decision.length));
  const cellW = plotW / steps + 1;
  const cellH = plotH / steps + 1;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const toX = (x) => pad + ((x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * plotW;
  const toY = (y) => pad + plotH - ((y - bounds.yMin) / (bounds.yMax - bounds.yMin)) * plotH;

  result.decision.forEach((cell) => {
    ctx.fillStyle = colorForPrediction(cell.prediction, 0.22);
    ctx.fillRect(toX(cell.x) - cellW / 2, toY(cell.y) - cellH / 2, cellW, cellH);
  });

  ctx.strokeStyle = "#d8dee8";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, plotW, plotH);

  ctx.strokeStyle = "rgba(29, 36, 51, 0.12)";
  for (let i = 1; i < 5; i += 1) {
    const x = pad + (plotW * i) / 5;
    const y = pad + (plotH * i) / 5;
    ctx.beginPath();
    ctx.moveTo(x, pad);
    ctx.lineTo(x, pad + plotH);
    ctx.moveTo(pad, y);
    ctx.lineTo(pad + plotW, y);
    ctx.stroke();
  }

  result.points.forEach((point) => {
    const x = toX(point.x);
    const y = toY(point.y);
    const radius = result.points.length <= 8 ? 12 : 5;

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = point.label === 1 ? "#1f9d73" : "#2f6fbd";
    ctx.fill();
    ctx.lineWidth = point.correct ? 2 : 4;
    ctx.strokeStyle = point.correct ? "#ffffff" : "#d55c4b";
    ctx.stroke();
  });

  ctx.fillStyle = "#667085";
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.fillText("blue = class 0", pad, height - 10);
  ctx.fillText("green = class 1", width - 128, height - 10);
}

function drawNetwork(result) {
  const { ctx, width, height } = setupCanvas(networkCanvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const layers = [
    Array.from({ length: 2 }, (_, i) => ({ label: `x${i + 1}` })),
    Array.from({ length: 8 }, (_, i) => ({ label: `h${i + 1}` })),
    [{ label: "out" }],
  ];
  const xs = [70, width / 2, width - 70];
  const nodes = layers.map((layer, layerIndex) => {
    const gap = Math.min(48, (height - 92) / Math.max(1, layer.length - 1));
    const total = gap * (layer.length - 1);
    const start = height / 2 - total / 2;
    return layer.map((node, index) => ({
      ...node,
      x: xs[layerIndex],
      y: start + index * gap,
    }));
  });

  const allWeights = [...result.network.W1.flat(), ...result.network.W2.flat()];
  const maxWeight = Math.max(0.001, ...allWeights.map((weight) => Math.abs(weight)));

  function strokeFor(weight) {
    const strength = Math.abs(weight) / maxWeight;
    const alpha = 0.16 + strength * 0.72;
    const widthValue = 1 + strength * 4.5;
    return {
      color: weight >= 0 ? `rgba(31, 157, 115, ${alpha})` : `rgba(213, 92, 75, ${alpha})`,
      width: widthValue,
    };
  }

  result.network.W1.forEach((row, inputIndex) => {
    row.forEach((weight, hiddenIndex) => {
      const from = nodes[0][inputIndex];
      const to = nodes[1][hiddenIndex];
      const style = strokeFor(weight);
      ctx.strokeStyle = style.color;
      ctx.lineWidth = style.width;
      ctx.beginPath();
      ctx.moveTo(from.x + 15, from.y);
      ctx.bezierCurveTo(from.x + 90, from.y, to.x - 90, to.y, to.x - 15, to.y);
      ctx.stroke();
    });
  });

  result.network.W2.forEach((row, hiddenIndex) => {
    row.forEach((weight) => {
      const from = nodes[1][hiddenIndex];
      const to = nodes[2][0];
      const style = strokeFor(weight);
      ctx.strokeStyle = style.color;
      ctx.lineWidth = style.width;
      ctx.beginPath();
      ctx.moveTo(from.x + 15, from.y);
      ctx.bezierCurveTo(from.x + 80, from.y, to.x - 80, to.y, to.x - 15, to.y);
      ctx.stroke();
    });
  });

  nodes.flat().forEach((node) => {
    ctx.beginPath();
    ctx.arc(node.x, node.y, 16, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#1d2433";
    ctx.stroke();
    ctx.fillStyle = "#1d2433";
    ctx.font = "700 11px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(node.label, node.x, node.y);
  });

  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.fillStyle = "#667085";
  ctx.fillText("green weights push up", 18, height - 26);
  ctx.fillStyle = "#d55c4b";
  ctx.fillText("red weights push down", 18, height - 10);
}

function drawLoss(result) {
  const { ctx, width, height } = setupCanvas(lossCanvas);
  const pad = 34;
  const plotW = width - pad * 2;
  const plotH = height - pad * 2;
  const history = result.history;
  const maxLoss = Math.max(...history.map((item) => item.loss), 0.01);
  const maxEpoch = Math.max(...history.map((item) => item.epoch), 1);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d8dee8";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, plotW, plotH);

  const toX = (epoch) => pad + (epoch / maxEpoch) * plotW;
  const toY = (loss) => pad + plotH - (loss / maxLoss) * plotH;

  ctx.beginPath();
  history.forEach((item, index) => {
    const x = toX(item.epoch);
    const y = toY(item.loss);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#2f6fbd";
  ctx.lineWidth = 3;
  ctx.stroke();

  const last = history[history.length - 1];
  ctx.beginPath();
  ctx.arc(toX(last.epoch), toY(last.loss), 5, 0, Math.PI * 2);
  ctx.fillStyle = result.passed ? "#1f9d73" : "#c98a1d";
  ctx.fill();

  ctx.fillStyle = "#667085";
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.fillText(`start ${history[0].loss.toFixed(4)}`, pad, height - 10);
  ctx.fillText(`end ${last.loss.toFixed(4)}`, width - 92, height - 10);
}

function renderPredictions(result) {
  const limit = result.points.length > 24 ? 24 : result.points.length;
  predictionList.innerHTML = "";
  result.points.slice(0, limit).forEach((point) => {
    const row = document.createElement("div");
    row.className = "prediction-row";
    const prediction = point.prediction;
    row.innerHTML = `
      <strong>(${point.x.toFixed(2)}, ${point.y.toFixed(2)})</strong>
      <span>target ${point.label}</span>
      <div class="bar" aria-label="prediction ${prediction.toFixed(3)}">
        <div class="bar-fill" style="width: ${clamp(prediction * 100, 0, 100)}%"></div>
      </div>
    `;
    predictionList.appendChild(row);
  });

  if (result.points.length > limit) {
    const extra = document.createElement("div");
    extra.className = "prediction-row";
    extra.innerHTML = `<strong>${result.points.length - limit} more</strong><span>shown on map</span><div class="bar"></div>`;
    predictionList.appendChild(extra);
  }
}

function renderResult(result) {
  activeResult = result;
  passText.textContent = result.passed ? "Passed" : "Close";
  passText.className = result.passed ? "passed" : "failed";
  accuracyText.textContent = `${Math.round(result.accuracy * 100)}%`;
  lossText.textContent = result.finalLoss.toFixed(4);
  testSummary.textContent = result.summary;
  drawDecision(result);
  drawNetwork(result);
  drawLoss(result);
  renderPredictions(result);
}

async function runTraining() {
  runButton.disabled = true;
  runButton.textContent = "Training";
  passText.textContent = "Training";
  passText.className = "";

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        test: testSelect.value,
        epochs: Number(epochsInput.value),
        learningRate: Number(learningRateInput.value),
      }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Training failed");
    }
    renderResult(result);
  } catch (error) {
    passText.textContent = "Error";
    passText.className = "failed";
    testSummary.textContent = error.message;
  } finally {
    runButton.disabled = false;
    runButton.textContent = "Train";
  }
}

async function loadTests() {
  const response = await fetch("/api/tests");
  const payload = await response.json();
  tests = payload.tests;
  testSelect.innerHTML = tests
    .map((test) => `<option value="${test.id}">${test.title}</option>`)
    .join("");

  const defaultTest = tests.find((test) => test.id === "xor") || tests[0];
  testSelect.value = defaultTest.id;
  applyTestDefaults();
  await runTraining();
}

function applyTestDefaults() {
  const selected = tests.find((test) => test.id === testSelect.value);
  if (!selected) return;
  epochsInput.value = selected.epochs;
  learningRateInput.value = selected.learningRate;
  testSummary.textContent = selected.summary;
}

testSelect.addEventListener("change", () => {
  applyTestDefaults();
  runTraining();
});

runButton.addEventListener("click", runTraining);

window.addEventListener("resize", () => {
  if (activeResult) {
    drawDecision(activeResult);
    drawNetwork(activeResult);
    drawLoss(activeResult);
  }
});

loadTests();
