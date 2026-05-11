---
title: "Brownian skier on the Sochi DEM"
resources:
  - sochi_skier.js
  - sochi_skier_map.jpg
  - sochi_skier_dem_f32.bin
  - sochi_skier_meta.json
---

The skier follows an overdamped Langevin dynamics on a digital elevation
model of the Roza Khutor to Sirius corridor:

$$
dx = -\nabla h(x)\,dt + \sigma\,dW
$$

The drift term moves the point downhill; the Brownian term adds controllable
noise, so larger $\sigma$ makes the trajectory explore the terrain instead of
sliding along one deterministic descent line.

```{=html}
<link rel="preload" href="sochi_skier_map.jpg" as="image">
<link rel="preload" href="sochi_skier_dem_f32.bin" as="fetch" crossorigin>

<div class="sochi-skier" data-sochi-skier>
  <div class="sochi-skier__toolbar" aria-label="Sochi skier controls">
    <label class="sochi-skier__control">
      <span class="sochi-skier__label">
        <span>Simulation fps</span>
        <span class="sochi-skier__value" data-role="fps-value">60</span>
      </span>
      <input class="sochi-skier__range" data-role="fps" type="range" min="10" max="360" step="10" value="60">
    </label>
    <label class="sochi-skier__control">
      <span class="sochi-skier__label">
        <span>Brownian noise, &sigma;</span>
        <span class="sochi-skier__value" data-role="sigma-value">10.0</span>
      </span>
      <input class="sochi-skier__range" data-role="sigma" type="range" min="0" max="120" step="0.5" value="10">
    </label>
    <button class="sochi-skier__button sochi-skier__button--primary" data-role="restart" type="button">Roza Khutor</button>
    <button class="sochi-skier__button" data-role="clear" type="button">Clear trail</button>
    <button class="sochi-skier__button" data-role="toggle" type="button" aria-pressed="false">Pause</button>
  </div>
  <div class="sochi-skier__stage" data-role="stage">
    <canvas class="sochi-skier__canvas" data-role="canvas" aria-label="Brownian skier trajectory on the Sochi DEM"></canvas>
    <div class="sochi-skier__pois" data-role="pois" aria-hidden="true"></div>
    <div class="sochi-skier__loading" data-role="loading">Loading Sochi DEM...</div>
  </div>
  <div class="sochi-skier__readout" aria-live="polite">
    <span>Elevation <strong data-role="elevation">...</strong></span>
    <span>Position <strong data-role="position">...</strong></span>
  </div>
</div>
```

Tap or click the map to set a new starting point.

```{=html}
<script defer src="sochi_skier.js"></script>
```
