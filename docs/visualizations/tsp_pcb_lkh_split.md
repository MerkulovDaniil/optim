---
title: "TSP: LKH-3 находит оптимум на pcb442"
---

Тот же бенчмарк TSPLIB **pcb442** (442 отверстия, оптимум $50\,778$ единиц), но решаемый **LKH-3** (Lin–Kernighan–Helsgaun) — специализированной эвристикой для TSP с **chain edge exchange** переменной глубины.

LKH-3 выходит **точно** на TSPLIB-оптимум $50\,778$ units (gap 0%) за десятки секунд — общие методы оптимизации с 2-opt-эвристикой такое плато (~+5%) пробить не могут.

Доступ через python-биндинг [`elkai`](https://github.com/fikisipi/elkai).

:::{.video}
tsp_pcb_lkh_split.mp4
:::
