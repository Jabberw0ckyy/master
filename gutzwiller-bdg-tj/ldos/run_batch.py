import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT = "af_sc.py"
VS = ["0", "0.1", "0.25", "0.5", "0.75", "1", "1.5", "2"]

RUNS = [
#    ["--method", "finite", "--delta", "0.05", "--Vs", *VS, "--max_resid", "2.5e-3", "--show", "stack", "--side", "mag", "map", "--tip", "1", "1"],
#    ["--method", "finite", "--delta", "0.10", "--Vs", *VS, "--max_resid", "2.5e-3", "--show", "stack", "--side", "mag", "map", "--tip", "1", "1"],
#    ["--method", "finite", "--delta", "0.15", "--Vs", *VS, "--max_resid", "2.5e-3", "--show", "stack", "--side", "mag", "map", "--tip", "1", "1"],
#    ["--method", "finite", "--delta", "0.05", "--Vs", *VS, "--max_resid", "2.5e-3", "--show", "stack", "--side", "mag", "map"],
#    ["--method", "finite", "--delta", "0.10", "--Vs", *VS, "--max_resid", "2.5e-3", "--show", "stack", "--side", "mag", "map"],
#    ["--method", "finite", "--delta", "0.15", "--Vs", *VS, "--max_resid", "2.5e-3", "--show", "stack", "--side", "mag", "map"],
     ["--method", "finite", "--delta", "0.05", "--Vs", *VS, "--max_resid", "2.5e-3", "--show", "stack", "--side", "map", "--half", "10", "--sub", "2"],
     ["--method", "finite", "--delta", "0.1", "--Vs", *VS, "--max_resid", "2.5e-3", "--show", "stack", "--side", "map", "--half", "10", "--sub", "2"],
     ["--method", "finite", "--delta", "0.15", "--Vs", *VS, "--max_resid", "2.5e-3", "--show", "stack", "--side", "map", "--half", "10", "--sub", "2"],
]

LOG_FILE = Path(f"batch_{datetime.now():%Y%m%d_%H%M%S}.log")


def fmt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> int:
    results = []
    total_start = time.time()

    with LOG_FILE.open("w", encoding="utf-8") as log:
        for i, args in enumerate(RUNS, 1):
            cmd = [sys.executable, SCRIPT, *args]
            cmd_str = " ".join(cmd)
            header = f"\n===== [{i}/{len(RUNS)}] {datetime.now():%H:%M:%S} =====\n{cmd_str}\n"
            print(header, flush=True)
            log.write(header)
            log.flush()

            start = time.time()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            proc.wait()
            elapsed = time.time() - start

            status = "OK" if proc.returncode == 0 else f"FAILED (code {proc.returncode})"
            summary = f"--> {status}, час: {fmt_time(elapsed)}\n"
            print(summary, flush=True)
            log.write(summary)
            log.flush()
            results.append((i, status, elapsed))

        footer = "\nsummory\n"
        for i, status, elapsed in results:
            footer += f"Run {i}: {status} ({fmt_time(elapsed)})\n"
        footer += f"full time: {fmt_time(time.time() - total_start)}\n"
        print(footer)
        log.write(footer)

    failed = [r for r in results if r[1] != "OK"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())