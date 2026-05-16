---
title: "TSP: Memetic 1+1 ES на pcb442"
---

Бенчмарк TSPLIB **pcb442** (Groetschel/Juenger/Reinelt, 1991) — реальная задача сверления печатной платы из 442 отверстий. Известный оптимум LKH: $50\,778$ единиц.

Memetic-комбинация: оптимизатор `OnePlusOne` из библиотеки [nevergrad](https://github.com/facebookresearch/nevergrad) генерирует кандидата через random-keys, каждый кандидат полируется полным 2-opt (numba-JIT). При застое — double-bridge perturbation и рестарт.

Слева — маршрут на плате (треугольники — отверстия, красная линия — путь сверла), справа — длина маршрута по числу вызовов оракула на лог-шкале.

Выходит на $54\,154$ единиц (gap +6.65%). 2-opt-плато для этого класса методов пробить нельзя без chain edge exchange — см. LKH-3 ниже.

:::{.video}
tsp_pcb_ga_split.mp4
:::
