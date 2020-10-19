import subprocess
from subprocess import CalledProcessError
import sys
import psutil
import platform
from timeit import default_timer as timer

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def measure(cmd, proc_name):
    try:
        start = timer()
        cmdp = subprocess.Popen(cmd.split(), shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        psp = psutil.Process(cmdp.pid)

        cpu = 0
        mem = 0
        time = 0

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
                psp.wait(timeout=0.5)
                time = timer() - start
            except psutil.TimeoutExpired:        
                continue
            else:
                break
        return_code = cmdp.poll()

#        eprint(f"\nERROR_CODE: {return_code}\n" + str(b"\n".join(cmdp.stdout.readlines())))        
    except CalledProcessError as e:
        output = e.output.decode()
        print(output)
        return None

    
    return time, mem, cpu



if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:])
    time, mem, cpu = measure(cmd)
    print(f"{time}, {mem}, {cpu}")
