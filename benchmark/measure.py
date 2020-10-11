import subprocess
from subprocess import CalledProcessError
import sys
import psutil
import platform
from timeit import default_timer as timer


def measure(cmd, proc_name):
    try:
        start = timer()
        cmdp = subprocess.Popen(f"{cmd}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        psp = get_model_process(cmdp, proc_name)

        cpu = 0
        mem = 0
        time = 0

        def on_process_termination(p):
            nonlocal time
            time = timer() - start

        while True:
            gone, alive = psutil.wait_procs(procs=[psp], timeout=0.5, callback=on_process_termination)
            if psp in alive:
                with psp.oneshot():
                    try:
                        cpu = max(cpu, psp.cpu_percent())

                        if platform.system() == "Darwin":
                            mem = max(mem, psp.memory_info().rss / 1024.)
                        else:
                            mem = max(mem, psp.memory_full_info().pss / 1024.)
                    except psutil.AccessDenied:
                        pass
                    except psutil.ZombieProcess:
                        pass
            else:
                break
    except CalledProcessError as e:
        output = e.output.decode()
        print(output)
        return None

    return time, mem, cpu


def get_model_process(cmdp, proc_name):
    psp = None
    while True:
        psutil.wait_procs([psutil.Process(cmdp.pid)], timeout=0.05)
        for proc in psutil.process_iter(['pid', 'name', 'username']):
            if proc.name() == proc_name:
                psp = psutil.Process(proc.pid)
        if psp:
            break
    return psp


if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:])
    time, mem, cpu = measure(cmd)
    print(f"{time}, {mem}, {cpu}")
