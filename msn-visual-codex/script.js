const NETWORK_LIBRARY = [
  {
    id: "small",
    label: "Small",
    title: "Small - 6 numbers",
    size: 6,
    comparators: [
      [1, 2], [4, 5], [0, 3],
      [0, 2], [3, 5], [1, 4],
      [0, 1], [3, 4], [2, 5],
      [2, 4], [1, 3],
      [2, 3],
    ],
    input: [6, 1, 5, 2, 4, 3],
  },
  {
    id: "medium",
    label: "Medium",
    title: "Medium - 10 numbers",
    size: 10,
    comparators: [
      [4, 9], [3, 8], [2, 7], [1, 6], [0, 5], [1, 4], [6, 9], [0, 3],
      [5, 8], [0, 2], [3, 6], [7, 9], [0, 1], [2, 4], [5, 7], [8, 9],
      [1, 2], [4, 6], [7, 8], [3, 5], [2, 5], [6, 8], [1, 3], [4, 7],
      [2, 3], [6, 7], [3, 4], [5, 6], [4, 5],
    ],
    input: [10, 2, 8, 1, 9, 4, 7, 3, 6, 5],
  },
  {
    id: "large",
    label: "Large",
    title: "Large - 16 numbers",
    size: 16,
    comparators: [
      [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15],
      [0, 2], [4, 6], [8, 10], [12, 14], [1, 3], [5, 7], [9, 11], [13, 15],
      [0, 4], [8, 12], [1, 5], [9, 13], [2, 6], [10, 14], [3, 7], [11, 15],
      [0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15],
      [5, 10], [6, 9], [3, 12], [13, 14], [7, 11], [1, 2], [4, 8], [1, 4],
      [7, 13], [2, 8], [11, 14], [2, 4], [5, 6], [9, 10], [11, 13], [3, 8],
      [7, 12], [6, 8], [10, 12], [3, 5], [7, 9], [3, 4], [5, 6], [7, 8],
      [9, 10], [11, 12], [6, 7], [8, 9],
    ],
    input: [16, 3, 12, 1, 15, 6, 14, 4, 11, 2, 9, 13, 5, 10, 7, 8],
  },
].map((network) => {
  const layers = packLayers(network.comparators);
  return {
    ...network,
    comparatorCount: network.comparators.length,
    layers,
  };
});

const sizePicker = document.getElementById("sizePicker");
const comparatorCount = document.getElementById("comparatorCount");
const progressText = document.getElementById("progressText");
const networkTitle = document.getElementById("networkTitle");
const backButton = document.getElementById("backButton");
const playButton = document.getElementById("playButton");
const forwardButton = document.getElementById("forwardButton");
const networkViewport = document.getElementById("networkViewport");
const networkCanvas = document.getElementById("networkCanvas");
const networkSvg = document.getElementById("networkSvg");
const tokenLayer = document.getElementById("tokenLayer");
const PLAYBACK_DELAY_MS = 1500;

const state = {
  selectedId: "large",
  currentStep: 0,
  isPlaying: false,
  timer: null,
};

function packLayers(comparators) {
  const layers = [];
  const lastLayerForWire = [];

  comparators.forEach(([a, b]) => {
    const layerIndex = Math.max(lastLayerForWire[a] ?? -1, lastLayerForWire[b] ?? -1) + 1;

    if (!layers[layerIndex]) {
      layers[layerIndex] = [];
    }

    layers[layerIndex].push([a, b]);
    lastLayerForWire[a] = layerIndex;
    lastLayerForWire[b] = layerIndex;
  });

  return layers;
}

function getNetwork() {
  return NETWORK_LIBRARY.find((network) => network.id === state.selectedId);
}

function simulate(network) {
  const start = network.input.map((value, index) => ({ id: `${network.id}-${index}`, value }));
  const arrangements = [start];
  const layerResults = [];
  let current = start.slice();

  network.layers.forEach((layer) => {
    const next = current.slice();
    const comparisons = [];

    layer.forEach(([a, b]) => {
      const swapped = current[a].value > current[b].value;
      comparisons.push({ a, b, swapped });

      if (swapped) {
        next[a] = current[b];
        next[b] = current[a];
      }
    });

    current = next;
    layerResults.push(comparisons);
    arrangements.push(current.slice());
  });

  return { arrangements, layerResults };
}

function buildLayout(network) {
  const depth = network.layers.length;
  const laneCount = network.size;
  const columnWidth = 78;
  const inputX = 76;
  const stateXs = Array.from({ length: depth + 1 }, (_, index) => inputX + index * columnWidth);
  const comparatorXs = Array.from({ length: depth }, (_, index) => stateXs[index] + columnWidth / 2);
  const width = stateXs[stateXs.length - 1] + 80;
  const height = Math.max(420, laneCount * 34 + 140);
  const top = 70;
  const bottom = height - 56;
  const laneGap = laneCount > 1 ? (bottom - top) / (laneCount - 1) : 0;
  const laneYs = Array.from({ length: laneCount }, (_, index) => top + laneGap * index);

  return {
    width,
    height,
    stateXs,
    comparatorXs,
    laneYs,
    top,
    columnWidth,
  };
}

function renderSizePicker() {
  sizePicker.innerHTML = "";

  NETWORK_LIBRARY.forEach((network) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `size-button${network.id === state.selectedId ? " active" : ""}`;
    button.innerHTML = `
      <strong>${network.label}</strong>
      <span>${network.size} numbers - ${network.comparatorCount} comparators</span>
    `;
    button.addEventListener("click", () => {
      if (state.selectedId === network.id) {
        return;
      }

      stopPlayback();
      state.selectedId = network.id;
      state.currentStep = 0;
      render();
    });
    sizePicker.appendChild(button);
  });
}

function renderNetwork() {
  const network = getNetwork();
  const layout = buildLayout(network);
  const simulation = simulate(network);
  const arrangement = simulation.arrangements[state.currentStep];

  networkTitle.textContent = network.title;
  comparatorCount.textContent = String(network.comparatorCount);
  progressText.textContent = `Level ${state.currentStep} / ${network.layers.length}`;
  playButton.textContent = state.isPlaying ? "❚❚" : "▶";

  networkCanvas.style.width = `${layout.width}px`;
  networkCanvas.style.height = `${layout.height}px`;
  networkSvg.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`);
  networkSvg.setAttribute("width", String(layout.width));
  networkSvg.setAttribute("height", String(layout.height));

  renderSvg(network, layout, simulation.layerResults);
  renderTokens(network, arrangement, layout);
  syncButtons(network);
}

function renderSvg(network, layout, layerResults) {
  const parts = [];

  network.layers.forEach((_, index) => {
    const x = layout.stateXs[index] + 14;
    const active = state.currentStep > 0 && index === state.currentStep - 1 ? " active" : "";
    parts.push(
      `<rect class="level-band${active}" x="${x}" y="22" width="${layout.columnWidth - 28}" height="${layout.height - 44}" rx="18"></rect>`
    );
    parts.push(
      `<text class="level-label" x="${layout.comparatorXs[index] - 9}" y="42">${String(index + 1).padStart(2, "0")}</text>`
    );
  });

  layout.laneYs.forEach((y, index) => {
    parts.push(
      `<line class="wire-line" x1="34" y1="${y}" x2="${layout.width - 34}" y2="${y}"></line>`
    );
    parts.push(
      `<text class="lane-label" x="14" y="${y + 4}">${String(index + 1).padStart(2, "0")}</text>`
    );
  });

  network.layers.forEach((layer, layerIndex) => {
    layer.forEach(([a, b], compareIndex) => {
      const x = layout.comparatorXs[layerIndex];
      const y1 = layout.laneYs[a];
      const y2 = layout.laneYs[b];
      const isActive = state.currentStep > 0 && layerIndex === state.currentStep - 1;
      const didSwap = isActive && layerResults[layerIndex]?.[compareIndex]?.swapped;
      const classes = ["compare-group"];

      if (isActive) {
        classes.push("active", didSwap ? "swap" : "steady");
      }

      parts.push(
        `<g class="${classes.join(" ")}">
          <line class="compare-rail" x1="${x}" y1="${y1}" x2="${x}" y2="${y2}"></line>
          <circle class="compare-node" cx="${x}" cy="${y1}" r="4.2"></circle>
          <circle class="compare-node" cx="${x}" cy="${y2}" r="4.2"></circle>
        </g>`
      );
    });
  });

  networkSvg.innerHTML = parts.join("");
}

function renderTokens(network, arrangement, layout) {
  tokenLayer.innerHTML = "";
  const x = layout.stateXs[state.currentStep];

  if (state.currentStep > 0) {
    network.input.forEach((value, laneIndex) => {
      const frozen = document.createElement("div");
      frozen.className = "token token-frozen";
      frozen.textContent = value;
      frozen.style.transform = `translate(${layout.stateXs[0]}px, ${layout.laneYs[laneIndex]}px)`;
      tokenLayer.appendChild(frozen);
    });
  }

  arrangement.forEach((token, laneIndex) => {
    const element = document.createElement("div");
    element.className = "token";
    element.textContent = token.value;
    element.style.transform = `translate(${x}px, ${layout.laneYs[laneIndex]}px)`;
    tokenLayer.appendChild(element);
  });
}

function syncButtons(network) {
  backButton.disabled = state.currentStep === 0;
  forwardButton.disabled = state.currentStep >= network.layers.length;
}

function stopPlayback() {
  state.isPlaying = false;
  if (state.timer) {
    window.clearTimeout(state.timer);
    state.timer = null;
  }
}

function stepForward() {
  const network = getNetwork();
  if (state.currentStep >= network.layers.length) {
    return;
  }

  state.currentStep += 1;
  render();
}

function stepBackward() {
  if (state.currentStep <= 0) {
    return;
  }

  state.currentStep -= 1;
  render();
}

function queuePlayback() {
  const network = getNetwork();

  if (!state.isPlaying) {
    return;
  }

  if (state.currentStep >= network.layers.length) {
    stopPlayback();
    render();
    return;
  }

  state.timer = window.setTimeout(() => {
    stepForward();
    if (state.currentStep >= network.layers.length) {
      stopPlayback();
      render();
      return;
    }

    queuePlayback();
  }, PLAYBACK_DELAY_MS);
}

function togglePlayback() {
  const network = getNetwork();

  if (state.isPlaying) {
    stopPlayback();
    render();
    return;
  }

  if (state.currentStep >= network.layers.length) {
    state.currentStep = 0;
  }

  state.isPlaying = true;
  render();
  queuePlayback();
}

backButton.addEventListener("click", () => {
  stopPlayback();
  stepBackward();
});

playButton.addEventListener("click", togglePlayback);

forwardButton.addEventListener("click", () => {
  stopPlayback();
  stepForward();
});

window.addEventListener("resize", render);

function render() {
  renderSizePicker();
  renderNetwork();
}

render();
