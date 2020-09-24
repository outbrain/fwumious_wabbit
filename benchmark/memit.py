import subprocess
from subprocess import check_output, CalledProcessError, STDOUT
import platform
import sys

s = platform.system()
if s == "Linux":
    time_cmd = "/usr/bin/time -v"
elif s == "Darwin":
    time_cmd = "/usr/bin/time -lp"
else:
    print("WARN: memory utilization reporting not supported for Windows!")


def memit(cmd):
    try:
        output = check_output(f"{time_cmd} {cmd}", shell=True, stderr=subprocess.STDOUT).decode()
    except CalledProcessError as e:
        output = e.output.decode()
        success = False

    lines = output.lower().replace("\t", "").split("\n")
    for line in lines:
        if "maximum resident set size" in line:
            split_line = [t.strip() for t in line.strip().split(" ")]
            if s == "Linux":
                res = int(split_line[-1])
            elif s == "Darwin":
                res = int(split_line[0])/1024.0
            break
    return res
   
if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:])
    res = memit(cmd)
    print(res)
