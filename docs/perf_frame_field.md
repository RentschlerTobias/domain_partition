# Performance: Non-Iterative Frame Field

Branch `perf_opti` · Commit `6e683f2` · 2026-06-05

## Problem

Partition-Pipeline langsam: **~16.7s pro Versuch** (cProfile, 5 Läufe).
Profiling-Befund:

| Stage | %/total | s/run |
|-------|---------|-------|
| **FrameField** | **47.7%** | ~8.0 |
| StreamlineGenerator | 40.2% | ~6.7 |
| StreamlinePostProc | 10.4% | ~1.7 |
| Rest (MeshGen, QuadMeshGen, Validator, MeshCheck) | <2% | ~0 |

Einzelner größter Posten: `numpy.linalg.solve` = **31.8s / 38% von allem**,
aus `tools/frame_field.py` → `Linearization_Norm_Constraint`.

### Ursache
`Linearization_Norm_Constraint` löste das Cross-Field via
**Ginzburg–Landau-Iteration**: pro Iteration wurde eine **dense** KKT-Matrix
der Größe `(3N)²` neu aus `np.zeros` aufgebaut und mit `np.linalg.solve`
gelöst — bis zu 1000 Iterationen (typisch ~130). `compute_initial_frame_field`
baute zusätzlich die Steifigkeitsmatrix A **dense** `(2N)²` per Python-Doppelloop.

## Lösung

**Globally Optimal Direction Fields** (Knöppel et al. 2013; vgl. Diamanti 2014,
Bommes MIQ): das glatteste Einheits-Richtungsfeld braucht keine Iteration.

Das Cross-Field ist als Einheitsvektor `(cos4θ, sin4θ)` pro Knoten dargestellt.
Für eine **flache, boundary-aligned** Domain (Randkreuze als harte
Dirichlet-BC) ist die **harmonische** Lösung dieses Repräsentationsfeldes plus
**per-Knoten-Normalisierung** das glatteste Feld → **topologisch minimale
Singularitätenzahl** (genau der geforderte Constraint). Downstream
(`singularity_detector`, Streamline-Tracing) nutzt nur den Winkel
`atan2(vy,vx)` → norm-invariant, also kompatibel.

## Änderungen

Datei: `tools/frame_field.py`

### 1. `Linearization_Norm_Constraint` — Iteration entfernt
130× dense KKT-Solve → eine per-Knoten-Normalisierung der bereits berechneten
harmonischen Lösung (`u_init` aus dem sparse boundary-Dirichlet-Solve).

```python
# vorher: for n in range(max_iterations=1000): build dense (3N)^2 KKT; np.linalg.solve(...)
# nachher:
z = u_init.reshape(-1, 2)
norms = np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
u_current = (z / norms).reshape(-1)
self.mesh.frame_field_iteration_number = 0
```
Signatur unverändert (`A, b` weiter übergeben, nicht mehr genutzt).
Rückgabeformat identisch (flach `[x0,y0,x1,y1,...]`) → `generate_cross_field`
und alle Aufrufer unberührt.

### 2. `compute_initial_frame_field` — sparse Assembly
Dense `A = np.zeros((2N, 2N))` + Python-Doppelloop → **COO-Triplets**, dann
`coo_matrix(...).tocsr()` (summiert Duplikate = das alte `+=`). Dirichlet-BC
via `lil`-Zeilen-Edit (Zeile nullen, Diagonale 1). Solve bleibt sparse
`spsolve`. Gibt jetzt `A_sparse` statt dense A zurück.

### 3. Nicht geändert
- `add_cross_at_boundaries`: latenter Index-Mismatch `[i]` vs `[idx]`
  verifiziert als **harmlos** — Mesh nummeriert Randknoten als `0..k-1`,
  daher `i == idx`. Kein Fix nötig.
- Repräsentation, Downstream, Aufrufer.

## Performance-Vergleich

cProfile, 5 volle Pipeline-Läufe, identisches Setup (`profile_partition.py 5`):

| Stage | vorher | nachher | Speedup |
|-------|-------:|--------:|--------:|
| **FrameField** | **39.96s** | **2.50s** | **16.0×** |
| StreamlineGenerator | 33.66s | 2.37s | 14.2× * |
| StreamlinePostProc | 8.71s | 5.58s | 1.6× |
| MeshGen | 0.48s | 0.38s | — |
| QuadMeshGen | 0.87s | 0.41s | — |
| Validator / MeshCheck | ~0 | ~0 | — |
| **TOTAL (5 runs)** | **83.7s** | **11.3s** | **7.4×** |
| **pro Versuch** | **16.7s** | **2.25s** | **7.4×** |

\* StreamlineGenerator-Gewinn ist indirekt: das glattere Feld hat weniger/
sauberere Singularitäten → deutlich weniger Separatrix-Tracing
(`get_best_cross_vector` 180001 → ~Bruchteil Calls).

### Hotspot davor/danach (self-time)
- vorher: `numpy.linalg.solve` 31.8s (651 calls) — weg.
- nachher: neuer Top-Hotspot `streamline_intersection_splitter.check_bounding_boxes`
  (~5.0s, O(n²) bbox + `scipy.optimize.minimize`) → nächstes Ziel.

## Verifikation

| Check | Ergebnis |
|-------|----------|
| `py_compile tools/frame_field.py` | OK |
| Feld-Norm pro Knoten | exakt 1.000 (min=max=1.0) |
| Iterationen | 0 (non-iterativ) |
| `detect_singularities` Euler-Check | "number of singularities is correct" |
| Singularitätenzahl | 0–2 (minimal) |
| End-to-End Erfolgsrate (12 Versuche) | 67% (Baseline ~30%) |

## Reproduktion

```bash
source ~/environments/domain_partition/bin/activate
python profile_partition.py 5     # per-Stage Timing + cProfile Top-N
```
