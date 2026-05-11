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
  const SUBSTEPS = 5;
  const MAX_TRAIL_LIMIT = 18000;

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
  display: block;
  width: 100%;
  height: 100%;
}
.sochi-skier__loading {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: var(--sochi-muted);
  font-size: 0.9rem;
  background: linear-gradient(180deg, rgba(248, 250, 252, 0.92), rgba(241, 245, 249, 0.86));
}
.sochi-skier__loading[hidden] {
  display: none;
}
.sochi-skier__pois {
  position: absolute;
  inset: 0;
  pointer-events: none;
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

  function getTrailLimit(width) {
    if (width < 480) return 5200;
    if (width < 760) return 8000;
    return MAX_TRAIL_LIMIT;
  }

  function getTrailStride(length, width) {
    const target = width < 480 ? 1200 : width < 760 ? 1900 : 3600;
    return Math.max(1, Math.ceil(length / target));
  }

  function getMaxStepsPerFrame(width) {
    return width < 480 ? 7 : width < 760 ? 10 : 14;
  }

  function randn() {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
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

    function gradient(x, y) {
      const eps = Math.max(px, py) * 0.5;
      return [
        (elevation(x + eps, y) - elevation(x - eps, y)) / (2 * eps),
        (elevation(x, y + eps) - elevation(x, y - eps)) / (2 * eps),
      ];
    }

    return { elevation, gradient };
  }

  function createProjector(meta, size) {
    const xSpan = meta.xMax - meta.xMin;
    const ySpan = meta.yMax - meta.yMin;
    return {
      worldToScreen(x, y) {
        return {
          x: ((x - meta.xMin) / xSpan) * size.width,
          y: ((meta.yMax - y) / ySpan) * size.height,
        };
      },
      screenToWorld(x, y) {
        return {
          x: meta.xMin + (x / size.width) * xSpan,
          y: meta.yMax - (y / size.height) * ySpan,
        };
      },
      worldSizeToScreen(w, h) {
        return {
          width: (w / xSpan) * size.width,
          height: (h / ySpan) * size.height,
        };
      },
    };
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

    const ctx = canvas.getContext("2d");
    const mapUrl = `${base}sochi_skier_map.jpg`;
    const [meta, demBuffer] = await Promise.all([
      fetch(`${base}sochi_skier_meta.json`).then((response) => response.json()),
      fetch(`${base}sochi_skier_dem_f32.bin`).then((response) => response.arrayBuffer()),
      loadImage(mapUrl),
    ]);
    const dem = new Float32Array(demBuffer);
    const field = createInterpolator(dem, meta);
    const state = {
      paused: false,
      pos: [meta.start.x, meta.start.y],
      trail: [],
      size: { width: 0, height: 0 },
      dpr: 1,
      trailLimit: MAX_TRAIL_LIMIT,
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

    function resetTrail(x, y) {
      state.pos = [clamp(x, meta.xMin, meta.xMax), clamp(y, meta.yMin, meta.yMax)];
      state.trail = [{ x: state.pos[0], y: state.pos[1] }];
      updateReadout(performance.now(), true);
      draw();
    }

    function stepSimulation() {
      const sd = Math.sqrt(DT);
      for (let index = 0; index < SUBSTEPS; index += 1) {
        const grad = field.gradient(state.pos[0], state.pos[1]);
        const nx = state.pos[0] + (-grad[0] * DT + state.sigma * sd * randn());
        const ny = state.pos[1] + (-grad[1] * DT + state.sigma * sd * randn());
        if (isInside(nx, ny)) {
          state.pos[0] = nx;
          state.pos[1] = ny;
        }
      }
      state.pos[0] = clamp(state.pos[0], meta.xMin, meta.xMax);
      state.pos[1] = clamp(state.pos[1], meta.yMin, meta.yMax);
      state.trail.push({ x: state.pos[0], y: state.pos[1] });
      if (state.trail.length > state.trailLimit) {
        state.trail.splice(0, state.trail.length - state.trailLimit);
      }
    }

    function resize() {
      const rect = stage.getBoundingClientRect();
      const width = Math.max(1, rect.width);
      const height = Math.max(1, rect.height);
      const dpr = getCanvasDpr(width);
      state.size = { width, height };
      state.dpr = dpr;
      state.trailLimit = getTrailLimit(width);
      if (state.trail.length > state.trailLimit) {
        state.trail.splice(0, state.trail.length - state.trailLimit);
      }
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      positionMap();
      placePois();
      draw();
    }

    function positionMap() {
      if (!state.size.width) return;
      const projector = createProjector(meta, state.size);
      const imageOrigin = projector.worldToScreen(meta.image.x, meta.image.y);
      const imageSize = projector.worldSizeToScreen(meta.image.w, meta.image.h);
      stage.style.backgroundImage = `url("${mapUrl}")`;
      stage.style.backgroundPosition = `${imageOrigin.x}px ${imageOrigin.y}px`;
      stage.style.backgroundSize = `${imageSize.width}px ${imageSize.height}px`;
    }

    function drawTrail(projector) {
      if (state.trail.length < 2) return;
      const stride = getTrailStride(state.trail.length, state.size.width);
      ctx.save();
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.lineWidth = 1.35;
      ctx.strokeStyle = "rgba(214, 40, 40, 0.86)";
      ctx.beginPath();
      for (let index = 0; index < state.trail.length; index += 1) {
        if (index !== 0 && index !== state.trail.length - 1 && index % stride !== 0) continue;
        const point = state.trail[index];
        const screen = projector.worldToScreen(point.x, point.y);
        if (index === 0) ctx.moveTo(screen.x, screen.y);
        else ctx.lineTo(screen.x, screen.y);
      }
      ctx.stroke();
      ctx.restore();
    }

    function drawSkier(projector) {
      const point = projector.worldToScreen(state.pos[0], state.pos[1]);
      ctx.save();
      ctx.shadowBlur = 7;
      ctx.shadowColor = "rgba(15, 23, 42, 0.28)";
      ctx.fillStyle = "#d62828";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.arc(point.x, point.y, 6.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    function draw() {
      const { width, height } = state.size;
      if (!width || !height) return;
      const projector = createProjector(meta, state.size);
      ctx.clearRect(0, 0, width, height);
      drawTrail(projector);
      drawSkier(projector);
    }

    function placePois() {
      if (!poiLayer || !state.size.width) return;
      const projector = createProjector(meta, state.size);
      poiLayer.textContent = "";
      meta.poi.forEach((poi) => {
        const point = projector.worldToScreen(poi.x, poi.y);
        let side = poi.side || "right";
        if (state.size.width < 620 && point.x > state.size.width * 0.68) side = "left";
        if (state.size.width < 620 && point.x < state.size.width * 0.24) side = "right";
        const item = document.createElement("div");
        item.className = `sochi-skier__poi sochi-skier__poi--${side}`;
        item.style.left = `${point.x}px`;
        item.style.top = `${point.y}px`;

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
      elevationValue.textContent = `${field.elevation(state.pos[0], state.pos[1]).toFixed(0)} m`;
      positionValue.textContent = `${(state.pos[0] / 1000).toFixed(2)}, ${(state.pos[1] / 1000).toFixed(2)} km`;
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
          for (let index = 0; index < steps; index += 1) stepSimulation();
          draw();
          updateReadout(now, false);
        }
      }
      requestAnimationFrame(frame);
    }

    stage.addEventListener("pointerdown", (event) => {
      const rect = stage.getBoundingClientRect();
      const projector = createProjector(meta, state.size);
      const point = projector.screenToWorld(event.clientX - rect.left, event.clientY - rect.top);
      if (isInside(point.x, point.y)) resetTrail(point.x, point.y);
    });

    restartButton.addEventListener("click", () => resetTrail(meta.start.x, meta.start.y));
    clearButton.addEventListener("click", () => resetTrail(state.pos[0], state.pos[1]));
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
    resetTrail(meta.start.x, meta.start.y);
    resize();
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
