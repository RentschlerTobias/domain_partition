"""Performance analysis of the partition generation pipeline.

Runs get_mesh() N times under cProfile and reports:
  1. Per-stage wall time (FrameField, Streamlines, PostProcessing, QuadMesh, ...)
  2. Top cumulative functions (cProfile)

Usage: python profile_partition.py [N]
"""
import os
import sys
import cProfile
import pstats
import io
import time

# Silence pipeline + gmsh output at fd level.
_devnull = os.open(os.devnull, os.O_WRONLY)
_saved_out, _saved_err = os.dup(1), os.dup(2)


def _silence():
    sys.stdout.flush(); sys.stderr.flush()
    os.dup2(_devnull, 1); os.dup2(_devnull, 2)


def _unsilence():
    sys.stdout.flush(); sys.stderr.flush()
    os.dup2(_saved_out, 1); os.dup2(_saved_err, 2)


from tools import (MeshGenerator, FrameField, NACA_airfoil,
                   StreamlineGenerator_v2, StreamlinePostProcessor,
                   QuadMeshGenerator, MeshCheck, QuadPartitionValidator)
import numpy as np

STAGE_TIMES = {}


def _timed(name, fn):
    t0 = time.perf_counter()
    out = fn()
    STAGE_TIMES.setdefault(name, []).append(time.perf_counter() - t0)
    return out


def run_once():
    """One full pipeline attempt with per-stage timing."""
    np.random.seed(int(time.time() * 1000) % 2**32 + os.getpid())
    airfoil = NACA_airfoil()
    lc = 0.04 + 0.02 * np.random.rand()
    mesh_gen = _timed("1_MeshGenerator", lambda: MeshGenerator(airfoil, quadMesh=False, lc=lc))
    ff = _timed("2_FrameField", lambda: FrameField(mesh_gen.mesh))
    sl = _timed("3_StreamlineGenerator", lambda: StreamlineGenerator_v2(ff.mesh))
    post = _timed("4_StreamlinePostProc", lambda: StreamlinePostProcessor(sl.mesh))
    blocked = post.block_mesh
    tri = post.mesh
    _timed("5_Validator", lambda: QuadPartitionValidator(blocked, tri, strict=False).is_valid())
    qmg = _timed("6_QuadMeshGen", lambda: QuadMeshGenerator(blocked))
    _timed("7_MeshCheck", lambda: MeshCheck(tri, qmg.transfinite_mesh, tol=1e-3))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    pr = cProfile.Profile()

    ok = 0
    _silence()
    for i in range(n):
        try:
            pr.enable()
            run_once()
            pr.disable()
            ok += 1
        except Exception:
            pr.disable()
    _unsilence()

    # --- Stage report ---
    print(f"\n=== STAGE WALL TIME ({ok}/{n} attempts completed full pipeline) ===")
    rows = []
    for name in sorted(STAGE_TIMES):
        ts = STAGE_TIMES[name]
        rows.append((name, sum(ts), sum(ts) / len(ts), len(ts)))
    total = sum(r[1] for r in rows)
    print(f"{'stage':<24}{'total_s':>10}{'mean_s':>10}{'calls':>7}{'%':>7}")
    for name, tot, mean, c in rows:
        print(f"{name:<24}{tot:>10.3f}{mean:>10.3f}{c:>7}{100*tot/total:>7.1f}")
    print(f"{'TOTAL':<24}{total:>10.3f}")

    # --- cProfile top cumulative ---
    print("\n=== TOP 25 BY CUMULATIVE TIME ===")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(25)
    print(s.getvalue())

    # --- top by total (self) time ---
    print("=== TOP 20 BY SELF TIME ===")
    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(20)
    print(s2.getvalue())


if __name__ == "__main__":
    main()
