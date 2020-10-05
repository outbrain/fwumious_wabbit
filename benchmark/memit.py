import subprocess
from subprocess import CalledProcessError
import sys
import psutil
import platform
from timeit import default_timer as timer


def memit(cmd):
    try:
        start = timer()
        cmdp = subprocess.Popen(f"{cmd}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        psp = psutil.Process(cmdp.pid)
        psp.cpu_percent()
        cpu = 0
        mem = 0
        time = 0

        def on_process_termination(p):
            nonlocal time
            time = timer() - start

        while True:
            gone, alive = psutil.wait_procs(procs=[psp], timeout=1, callback=on_process_termination)
            if psp in alive:
                with psp.oneshot():
                    cpu = max(cpu, psp.cpu_percent())
                    if platform.system() == "Darwin":
                        mem = max(mem, psp.memory_info().rss / 1024.)
                    else:
                        mem = max(mem, psp.memory_full_info().uss / 1024.)
            else:
                break
    except CalledProcessError as e:
        output = e.output.decode()
        print(output)
        return None

    return time, mem, cpu


if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:])
    time, mem, cpu = memit(cmd)
    print(f"{time}, {mem}, {cpu}")
