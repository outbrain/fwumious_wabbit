import subprocess
from subprocess import CalledProcessError
import sys
import psutil
import platform
from timeit import default_timer as timer


def measure(cmd, proc_name):
    try:
        start = timer()
        cmdp = subprocess.Popen(cmd.split(), shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        psp = psutil.Process(cmdp.pid)

        cpu = 0
        mem = 0
        time = 0

        def on_process_termination():
            nonlocal time
            time = timer() - start

        while True:
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
            try:
                psp.wait(timeout=0.1)
                on_process_termination()
            except psutil.TimeoutExpired:        
                continue
            else:
                break
            
    except CalledProcessError as e:
        output = e.output.decode()
        print(output)
        return None

    return time, mem, cpu



if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:])
    time, mem, cpu = measure(cmd)
    print(f"{time}, {mem}, {cpu}")
