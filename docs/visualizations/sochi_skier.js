(function () {
  "use strict";

  const STYLE_ID = "sochi-skier-style";
  const SCRIPT_URL = document.currentScript && document.currentScript.src
    ? document.currentScript.src
    : window.location.href;
  const ASSET_ROOT = new URL(".", SCRIPT_URL).href;
  const DEFAULT_FPS = 60;
  const DEFAULT_SIGMA = 10;
  const DT = 50;
  const SQRT_DT = Math.sqrt(DT);
  const SUBSTEPS = 5;
  const SKIER_DIRTY_RADIUS = 22;

  function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.sochi-skier {
  --sochi-ink: #0f172a;
  --sochi-muted: #64748b;
  --sochi-border: #cbd5e1;
  --sochi-bg: #f8fafc;
  --sochi-panel: #ffffff;
  --sochi-red: #d62828;
  --sochi-yellow: #ffd23f;
  width: min(100%, calc(100vw - 2rem));
  max-width: 100%;
  margin: 1.2rem 0 1.4rem;
}
.sochi-skier * {
  box-sizing: border-box;
}
.sochi-skier__toolbar {
  display: grid;
  grid-template-columns: minmax(150px, 0.8fr) minmax(190px, 1fr) auto auto auto;
  gap: 0.55rem;
  align-items: end;
  margin-bottom: 0.65rem;
}
.sochi-skier__control {
  min-width: 0;
}
.sochi-skier__label {
  display: flex;
  justify-content: space-between;
  gap: 0.75rem;
  color: var(--sochi-ink);
  font-size: 0.86rem;
  font-weight: 650;
  line-height: 1.2;
  margin-bottom: 0.22rem;
}
.sochi-skier__value {
  color: var(--sochi-muted);
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  white-space: nowrap;
}
.sochi-skier__range {
  display: block;
  width: 100%;
  accent-color: var(--sochi-red);
}
.sochi-skier__button {
  appearance: none;
  border: 1px solid var(--sochi-border);
  border-radius: 6px;
  background: var(--sochi-panel);
  color: var(--sochi-ink);
  cursor: pointer;
  font: inherit;
  font-size: 0.86rem;
  font-weight: 650;
  line-height: 1.2;
  min-height: 2.35rem;
  padding: 0.48rem 0.72rem;
  white-space: nowrap;
}
.sochi-skier__button:hover {
  border-color: #94a3b8;
  background: #f8fafc;
}
.sochi-skier__button:focus-visible,
.sochi-skier__range:focus-visible {
  outline: 2px solid rgba(37, 99, 235, 0.55);
  outline-offset: 2px;
}
.sochi-skier__button--primary {
  background: #0f172a;
  border-color: #0f172a;
  color: #ffffff;
}
.sochi-skier__button--primary:hover {
  background: #1e293b;
  border-color: #1e293b;
}
.sochi-skier__stage {
  position: relative;
  width: 100%;
  aspect-ratio: 53199.43351134687 / 46105.89411468857;
  overflow: hidden;
  border: 1px solid var(--sochi-border);
  border-radius: 8px;
  background-color: #f1f5f9;
  background-repeat: no-repeat;
  cursor: crosshair;
  touch-action: manipulation;
}
.sochi-skier__canvas {
  position: absolute;
  inset: 0;
  display: block;
  width: 100%;
  height: 100%;
  pointer-events: none;
}
.sochi-skier__canvas--trail {
  z-index: 1;
}
.sochi-skier__canvas--skier {
  z-index: 2;
}
.sochi-skier__loading {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: var(--sochi-muted);
  font-size: 0.9rem;
  background: linear-gradient(180deg, rgba(248, 250, 252, 0.92), rgba(241, 245, 249, 0.86));
  z-index: 4;
}
.sochi-skier__loading[hidden] {
  display: none;
}
.sochi-skier__pois {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 3;
}
.sochi-skier__poi {
  position: absolute;
  display: flex;
  align-items: center;
  filter: drop-shadow(0 1px 1.5px rgba(15, 23, 42, 0.28));
}
.sochi-skier__poi-marker {
  flex: 0 0 18px;
  width: 18px;
  height: 18px;
}
.sochi-skier__poi-marker--settlement {
  border: 2px solid #0f172a;
  border-radius: 50%;
  background: var(--sochi-yellow);
}
.sochi-skier__poi-marker--peak {
  width: 0;
  height: 0;
  border-left: 9px solid transparent;
  border-right: 9px solid transparent;
  border-bottom: 16px solid #1e293b;
  filter: drop-shadow(0 0 0 white);
}
.sochi-skier__poi-label {
  max-width: min(12rem, 42vw);
  overflow-wrap: anywhere;
  border: 1px solid rgba(71, 85, 105, 0.55);
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.94);
  color: var(--sochi-ink);
  font-size: 0.78rem;
  font-weight: 700;
  line-height: 1.12;
  padding: 0.18rem 0.38rem;
}
.sochi-skier__poi-label small {
  display: block;
  color: var(--sochi-muted);
  font-size: 0.68rem;
  font-weight: 650;
}
.sochi-skier__poi--right {
  flex-direction: row;
  transform: translate(-9px, -50%);
}
.sochi-skier__poi--right .sochi-skier__poi-label {
  margin-left: 0.35rem;
}
.sochi-skier__poi--left {
  flex-direction: row-reverse;
  transform: translate(calc(-100% + 9px), -50%);
}
.sochi-skier__poi--left .sochi-skier__poi-label {
  margin-right: 0.35rem;
}
.sochi-skier__readout {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem 0.85rem;
  align-items: baseline;
  color: var(--sochi-muted);
  font-size: 0.82rem;
  line-height: 1.35;
  margin-top: 0.55rem;
}
.sochi-skier__readout strong {
  color: var(--sochi-ink);
  font-weight: 700;
}
.sochi-skier-fp {
  width: min(100%, calc(100vw - 2rem));
  margin: 1.5rem 0 1.4rem;
}
.sochi-skier-fp h3 {
  color: var(--sochi-ink);
  font-size: 1rem;
  font-weight: 750;
  line-height: 1.2;
  margin: 1.2rem 0 0.55rem;
}
.sochi-skier-fp__grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 0.9rem;
}
.sochi-skier-fp__item {
  margin: 0;
  min-width: 0;
}
.sochi-skier-fp__video {
  display: block;
  width: 100%;
  aspect-ratio: 1440 / 624;
  border: 1px solid var(--sochi-border);
  border-radius: 8px;
  background: #f1f5f9;
}
.sochi-skier-fp__caption {
  color: var(--sochi-muted);
  font-size: 0.82rem;
  font-weight: 650;
  line-height: 1.25;
  margin-top: 0.32rem;
}
@media (max-width: 980px) {
  .sochi-skier__toolbar {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    align-items: stretch;
  }
  .sochi-skier__button {
    width: 100%;
    min-height: 2.4rem;
    padding-inline: 0.42rem;
  }
}
@media (max-width: 720px) {
  .sochi-skier__toolbar {
    grid-template-columns: minmax(0, 1fr);
  }
}
@media (max-width: 430px) {
  .sochi-skier__poi-marker {
    flex-basis: 14px;
    width: 14px;
    height: 14px;
  }
  .sochi-skier__poi-marker--peak {
    border-left-width: 7px;
    border-right-width: 7px;
    border-bottom-width: 13px;
  }
  .sochi-skier__poi-label {
    font-size: 0.68rem;
    padding: 0.14rem 0.28rem;
  }
  .sochi-skier__poi-label small {
    display: none;
  }
}`;
    document.head.appendChild(style);
  }

  function qs(root, role) {
    return root.querySelector(`[data-role="${role}"]`);
  }

  function clamp(value, lo, hi) {
    return value < lo ? lo : value > hi ? hi : value;
  }

  function clampIndex(value, lo, hi) {
    return value < lo ? lo : value > hi ? hi : value;
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.decoding = "async";
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error(`Cannot load ${src}`));
      image.src = src;
    });
  }

  function getCanvasDpr(width) {
    const raw = window.devicePixelRatio || 1;
    const cores = navigator.hardwareConcurrency || 8;
    const mobileViewport = width < 620;
    const lowPower = cores <= 4;
    const cap = mobileViewport || lowPower ? 1.5 : 2;
    return Math.max(1, Math.min(cap, raw));
  }

  function getMaxStepsPerFrame(width) {
    return width < 480 ? 7 : width < 760 ? 10 : 14;
  }

  let spareNormal = null;

  function randn() {
    if (spareNormal !== null) {
      const value = spareNormal;
      spareNormal = null;
      return value;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    const magnitude = Math.sqrt(-2 * Math.log(u));
    const angle = 2 * Math.PI * v;
    spareNormal = magnitude * Math.sin(angle);
    return magnitude * Math.cos(angle);
  }

  function createInterpolator(dem, meta) {
    const width = meta.width;
    const height = meta.height;
    const xMin = meta.xMin;
    const xMax = meta.xMax;
    const yMin = meta.yMin;
    const yMax = meta.yMax;
    const px = (xMax - xMin) / (width - 1);
    const py = (yMax - yMin) / (height - 1);

    function cubic(p0, p1, p2, p3, t) {
      const t2 = t * t;
      const t3 = t2 * t;
      return (-0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3) * t3
        + (p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3) * t2
        + (-0.5 * p0 + 0.5 * p2) * t
        + p1;
    }

    function rowAt(row, ix, tx) {
      const x0 = clampIndex(ix - 1, 0, width - 1);
      const x1 = clampIndex(ix, 0, width - 1);
      const x2 = clampIndex(ix + 1, 0, width - 1);
      const x3 = clampIndex(ix + 2, 0, width - 1);
      const offset = row * width;
      return cubic(dem[offset + x0], dem[offset + x1], dem[offset + x2], dem[offset + x3], tx);
    }

    function elevation(x, y) {
      const safeX = clamp(x, xMin, xMax);
      const safeY = clamp(y, yMin, yMax);
      const fx = (safeX - xMin) / px;
      const fy = (safeY - yMin) / py;
      const ix = Math.floor(fx);
      const iy = Math.floor(fy);
      const tx = fx - ix;
      const ty = fy - iy;
      const y0 = clampIndex(iy - 1, 0, height - 1);
      const y1 = clampIndex(iy, 0, height - 1);
      const y2 = clampIndex(iy + 1, 0, height - 1);
      const y3 = clampIndex(iy + 2, 0, height - 1);
      return cubic(rowAt(y0, ix, tx), rowAt(y1, ix, tx), rowAt(y2, ix, tx), rowAt(y3, ix, tx), ty);
    }

    function gradientInto(x, y, out) {
      const eps = Math.max(px, py) * 0.5;
      out[0] = (elevation(x + eps, y) - elevation(x - eps, y)) / (2 * eps);
      out[1] = (elevation(x, y + eps) - elevation(x, y - eps)) / (2 * eps);
    }

    return { elevation, gradientInto };
  }

  async function init(root) {
    if (root.dataset.sochiReady) return;
    root.dataset.sochiReady = "loading";

    injectStyles();

    const base = root.dataset.assets || ASSET_ROOT;
    const canvas = qs(root, "canvas");
    const stage = qs(root, "stage");
    const loading = qs(root, "loading");
    const poiLayer = qs(root, "pois");
    const fpsInput = qs(root, "fps");
    const fpsValue = qs(root, "fps-value");
    const sigmaInput = qs(root, "sigma");
    const sigmaValue = qs(root, "sigma-value");
    const restartButton = qs(root, "restart");
    const clearButton = qs(root, "clear");
    const toggleButton = qs(root, "toggle");
    const elevationValue = qs(root, "elevation");
    const positionValue = qs(root, "position");

    const ctx = canvas.getContext("2d", { alpha: true, desynchronized: true });
    const trailCanvas = document.createElement("canvas");
    trailCanvas.className = "sochi-skier__canvas sochi-skier__canvas--trail";
    trailCanvas.setAttribute("aria-hidden", "true");
    canvas.classList.add("sochi-skier__canvas--skier");
    stage.insertBefore(trailCanvas, canvas);

    const trailCtx = trailCanvas.getContext("2d", { alpha: true, desynchronized: true });
    const mapUrl = `${base}sochi_skier_map.jpg`;
    const [meta, demBuffer] = await Promise.all([
      fetch(`${base}sochi_skier_meta.json`).then((response) => response.json()),
      fetch(`${base}sochi_skier_dem_f32.bin`).then((response) => response.arrayBuffer()),
      loadImage(mapUrl),
    ]);
    const dem = new Float32Array(demBuffer);
    const field = createInterpolator(dem, meta);
    const restartPoint = meta.poi.find((poi) => poi.label === "Aibga-2") || meta.start;
    const xSpan = meta.xMax - meta.xMin;
    const ySpan = meta.yMax - meta.yMin;
    const gradient = [0, 0];
    let hasPendingTrail = false;
    const state = {
      paused: false,
      posX: restartPoint.x,
      posY: restartPoint.y,
      trailX: restartPoint.x,
      trailY: restartPoint.y,
      size: { width: 0, height: 0 },
      dpr: 1,
      scaleX: 1,
      scaleY: 1,
      skierScreenX: null,
      skierScreenY: null,
      targetFps: DEFAULT_FPS,
      tickAccumulator: 0,
      lastFrameTime: 0,
      sigma: DEFAULT_SIGMA,
      lastReadout: 0,
    };

    if (fpsInput) {
      state.targetFps = Number(fpsInput.value || DEFAULT_FPS);
      fpsValue.textContent = String(Math.round(state.targetFps));
      fpsInput.addEventListener("input", () => {
        state.targetFps = Number(fpsInput.value);
        fpsValue.textContent = String(Math.round(state.targetFps));
      });
    }

    if (sigmaInput) {
      state.sigma = Number(sigmaInput.value || DEFAULT_SIGMA);
      sigmaInput.addEventListener("input", () => {
        state.sigma = Number(sigmaInput.value);
        sigmaValue.textContent = state.sigma.toFixed(1);
      });
    }

    function isInside(x, y) {
      return x >= meta.xMin && x <= meta.xMax && y >= meta.yMin && y <= meta.yMax;
    }

    function screenX(x) {
      return (x - meta.xMin) * state.scaleX;
    }

    function screenY(y) {
      return (meta.yMax - y) * state.scaleY;
    }

    function configureTrailContext() {
      trailCtx.lineJoin = "round";
      trailCtx.lineCap = "round";
      trailCtx.lineWidth = 1.35;
      trailCtx.strokeStyle = "rgba(214, 40, 40, 0.86)";
    }

    function clearTrailLayer() {
      if (!trailCanvas.width || !trailCanvas.height) return;
      trailCtx.save();
      trailCtx.setTransform(1, 0, 0, 1, 0, 0);
      trailCtx.clearRect(0, 0, trailCanvas.width, trailCanvas.height);
      trailCtx.restore();
    }

    function clearSkierLayer() {
      if (!state.size.width || !state.size.height) return;
      ctx.clearRect(0, 0, state.size.width, state.size.height);
      state.skierScreenX = null;
      state.skierScreenY = null;
    }

    function resizeTrailLayer(width, height, dpr) {
      const pixelWidth = Math.max(1, Math.round(width * dpr));
      const pixelHeight = Math.max(1, Math.round(height * dpr));
      trailCanvas.style.width = `${width}px`;
      trailCanvas.style.height = `${height}px`;
      if (trailCanvas.width === pixelWidth && trailCanvas.height === pixelHeight) {
        trailCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        configureTrailContext();
        return;
      }

      let previous = null;
      if (trailCanvas.width && trailCanvas.height) {
        previous = document.createElement("canvas");
        previous.width = trailCanvas.width;
        previous.height = trailCanvas.height;
        previous.getContext("2d").drawImage(trailCanvas, 0, 0);
      }

      trailCanvas.width = pixelWidth;
      trailCanvas.height = pixelHeight;
      trailCtx.setTransform(1, 0, 0, 1, 0, 0);
      trailCtx.clearRect(0, 0, pixelWidth, pixelHeight);
      if (previous) {
        trailCtx.imageSmoothingEnabled = true;
        trailCtx.drawImage(previous, 0, 0, previous.width, previous.height, 0, 0, pixelWidth, pixelHeight);
      }
      trailCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      configureTrailContext();
    }

    function beginTrailPath() {
      if (!trailCanvas.width || !state.size.width) return;
      hasPendingTrail = false;
      configureTrailContext();
      trailCtx.beginPath();
    }

    function appendTrailSegment(fromX, fromY, toX, toY) {
      if (!trailCanvas.width || !state.size.width) return;
      trailCtx.moveTo(screenX(fromX), screenY(fromY));
      trailCtx.lineTo(screenX(toX), screenY(toY));
      hasPendingTrail = true;
    }

    function strokeTrailPath() {
      if (hasPendingTrail) trailCtx.stroke();
    }

    function resetTrail(x, y) {
      state.posX = clamp(x, meta.xMin, meta.xMax);
      state.posY = clamp(y, meta.yMin, meta.yMax);
      state.trailX = state.posX;
      state.trailY = state.posY;
      state.tickAccumulator = 0;
      clearTrailLayer();
      clearSkierLayer();
      updateReadout(performance.now(), true);
      drawSkier();
    }

    function stepSimulation() {
      let x = state.posX;
      let y = state.posY;
      const noiseScale = state.sigma * SQRT_DT;
      for (let index = 0; index < SUBSTEPS; index += 1) {
        field.gradientInto(x, y, gradient);
        const nx = x + (-gradient[0] * DT + noiseScale * randn());
        const ny = y + (-gradient[1] * DT + noiseScale * randn());
        if (isInside(nx, ny)) {
          x = nx;
          y = ny;
        }
      }
      x = clamp(x, meta.xMin, meta.xMax);
      y = clamp(y, meta.yMin, meta.yMax);
      appendTrailSegment(state.trailX, state.trailY, x, y);
      state.posX = x;
      state.posY = y;
      state.trailX = x;
      state.trailY = y;
    }

    function resize() {
      const rect = stage.getBoundingClientRect();
      const width = Math.max(1, rect.width);
      const height = Math.max(1, rect.height);
      const dpr = getCanvasDpr(width);
      state.size = { width, height };
      state.dpr = dpr;
      state.scaleX = width / xSpan;
      state.scaleY = height / ySpan;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      resizeTrailLayer(width, height, dpr);
      positionMap();
      placePois();
      clearSkierLayer();
      drawSkier();
    }

    function positionMap() {
      if (!state.size.width) return;
      stage.style.backgroundImage = `url("${mapUrl}")`;
      stage.style.backgroundPosition = `${screenX(meta.image.x)}px ${screenY(meta.image.y)}px`;
      stage.style.backgroundSize = `${meta.image.w * state.scaleX}px ${meta.image.h * state.scaleY}px`;
    }

    function drawSkier() {
      if (!state.size.width || !state.size.height) return;
      if (state.skierScreenX === null || state.skierScreenY === null) {
        ctx.clearRect(0, 0, state.size.width, state.size.height);
      } else {
        ctx.clearRect(
          state.skierScreenX - SKIER_DIRTY_RADIUS,
          state.skierScreenY - SKIER_DIRTY_RADIUS,
          SKIER_DIRTY_RADIUS * 2,
          SKIER_DIRTY_RADIUS * 2,
        );
      }
      const x = screenX(state.posX);
      const y = screenY(state.posY);
      ctx.save();
      ctx.shadowBlur = 7;
      ctx.shadowColor = "rgba(15, 23, 42, 0.28)";
      ctx.fillStyle = "#d62828";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.arc(x, y, 6.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
      state.skierScreenX = x;
      state.skierScreenY = y;
    }

    function placePois() {
      if (!poiLayer || !state.size.width) return;
      poiLayer.textContent = "";
      meta.poi.forEach((poi) => {
        const x = screenX(poi.x);
        const y = screenY(poi.y);
        let side = poi.side || "right";
        if (state.size.width < 620 && x > state.size.width * 0.68) side = "left";
        if (state.size.width < 620 && x < state.size.width * 0.24) side = "right";
        const item = document.createElement("div");
        item.className = `sochi-skier__poi sochi-skier__poi--${side}`;
        item.style.left = `${x}px`;
        item.style.top = `${y}px`;

        const marker = document.createElement("span");
        marker.className = `sochi-skier__poi-marker sochi-skier__poi-marker--${poi.type}`;
        marker.setAttribute("aria-hidden", "true");

        const label = document.createElement("span");
        label.className = "sochi-skier__poi-label";
        label.textContent = poi.label;
        const small = document.createElement("small");
        small.textContent = poi.elevation;
        label.appendChild(small);

        item.append(marker, label);
        poiLayer.appendChild(item);
      });
    }

    function updateReadout(now, force) {
      if (!force && now - state.lastReadout < 160) return;
      state.lastReadout = now;
      elevationValue.textContent = `${field.elevation(state.posX, state.posY).toFixed(0)} m`;
      positionValue.textContent = `${(state.posX / 1000).toFixed(2)}, ${(state.posY / 1000).toFixed(2)} km`;
    }

    function frame(now) {
      if (!state.lastFrameTime) state.lastFrameTime = now;
      const elapsedSeconds = Math.min(0.25, Math.max(0, (now - state.lastFrameTime) / 1000));
      state.lastFrameTime = now;
      if (!state.paused) {
        state.tickAccumulator += elapsedSeconds * state.targetFps;
        const maxSteps = getMaxStepsPerFrame(state.size.width);
        const steps = Math.min(maxSteps, Math.floor(state.tickAccumulator));
        state.tickAccumulator -= steps;
        if (steps > 0) {
          beginTrailPath();
          for (let index = 0; index < steps; index += 1) stepSimulation();
          strokeTrailPath();
          drawSkier();
          updateReadout(now, false);
        }
      }
      requestAnimationFrame(frame);
    }

    stage.addEventListener("pointerdown", (event) => {
      if (!state.size.width || !state.size.height) return;
      const rect = stage.getBoundingClientRect();
      const x = meta.xMin + ((event.clientX - rect.left) / state.size.width) * xSpan;
      const y = meta.yMax - ((event.clientY - rect.top) / state.size.height) * ySpan;
      if (isInside(x, y)) resetTrail(x, y);
    });

    restartButton.addEventListener("click", () => resetTrail(restartPoint.x, restartPoint.y));
    clearButton.addEventListener("click", () => resetTrail(state.posX, state.posY));
    toggleButton.addEventListener("click", () => {
      state.paused = !state.paused;
      toggleButton.textContent = state.paused ? "Resume" : "Pause";
      toggleButton.setAttribute("aria-pressed", String(state.paused));
    });

    if ("ResizeObserver" in window) {
      const observer = new ResizeObserver(resize);
      observer.observe(stage);
    } else {
      window.addEventListener("resize", resize);
    }

    loading.hidden = true;
    root.dataset.sochiReady = "ready";
    resize();
    resetTrail(restartPoint.x, restartPoint.y);
    requestAnimationFrame(frame);
  }

  function boot() {
    document.querySelectorAll("[data-sochi-skier]").forEach((root) => {
      init(root).catch((error) => {
        root.dataset.sochiReady = "error";
        const loading = qs(root, "loading");
        if (loading) loading.textContent = "Unable to load Sochi DEM.";
        console.error(error);
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
