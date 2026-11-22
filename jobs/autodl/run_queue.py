"""GPU task queue for the shared AutoDL server (3 cards): runs the tasks of a list on the card with the most free memory.

usage (on the server, in the project copy): nohup setsid python3 jobs/autodl/run_queue.py <list> [--per_card 1] \
            [--headroom_gb 2] > logs/gpu/runner.log 2>&1 &
List lines: "name|dependencies (comma-separated task names, or -)|GPU memory in GB|command" ('#' lines are comments).
A task starts when all its dependencies have logs/gpu/<dep>.done, a card has at least memory + headroom GB free
(nvidia-smi), and fewer than --per_card of our tasks run on that card. Exit status 0 -> logs/gpu/<name>.done, otherwise
logs/gpu/<name>.fail (tasks that depend on it are not started). Other users' processes are never touched; tasks run
with nice 10. The runner ends when no task can start any more.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("list")
ap.add_argument("--per_card", type=int, default=1)
ap.add_argument("--headroom_gb", type=float, default=2.0)
ap.add_argument("--poll", type=int, default=30)
a = ap.parse_args()
LOG = Path("logs/gpu")
LOG.mkdir(parents=True, exist_ok=True)


def tasks():
    out = []
    for line in Path(a.list).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, deps, mem, cmd = line.split("|", 3)
        out.append((name.strip(), [d for d in deps.strip().split(",") if d and d != "-"], float(mem), cmd.strip()))
    return out


def free_mb():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout
    return {int(i): int(t) - int(u) for i, t, u in (l.split(", ") for l in q.strip().splitlines())}


def log(msg):
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


running = {}                                                  # name -> (Popen, card)
log(f"runner start: {a.list}, per card {a.per_card}, headroom {a.headroom_gb} GB")
while True:
    for name, (p, card) in list(running.items()):
        rc = p.poll()
        if rc is not None:
            (LOG / f"{name}.{'done' if rc == 0 else 'fail'}").touch()
            log(f"{name} exit {rc} (card {card})")
            del running[name]
    T = tasks()
    state = {n: ("done" if (LOG / f"{n}.done").exists() else "fail" if (LOG / f"{n}.fail").exists()
                 else "run" if n in running else "wait") for n, *_ in T}
    blocked = lambda deps: any(state.get(d) == "fail" or d not in state for d in deps)  # noqa: E731
    ready = [t for t in T if state[t[0]] == "wait" and all(state.get(d) == "done" for d in t[1])]
    pending = [t for t in T if state[t[0]] == "wait" and not blocked(t[1])]
    if not running and not ready and not pending:
        log("nothing left to start; runner ends")
        break
    if ready:
        fm = free_mb()
        for name, deps, mem, cmd in ready:
            ours = {c: sum(1 for _, (_, cc) in running.items() if cc == c) for c in fm}
            cand = [c for c in fm if fm[c] >= (mem + a.headroom_gb) * 1024 and ours[c] < a.per_card]
            if not cand:
                continue
            card = max(cand, key=lambda c: fm[c])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(card), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1")
            fh = open(LOG / f"{name}.log", "w")
            p = subprocess.Popen(["nice", "-n", "10", "bash", "-c", cmd], stdout=fh, stderr=subprocess.STDOUT, env=env,
                                 start_new_session=True)
            running[name] = (p, card)
            fm[card] -= int(mem * 1024)
            log(f"{name} started on card {card} (pid {p.pid}, {mem} GB)")
    time.sleep(a.poll)
sys.exit(0)
