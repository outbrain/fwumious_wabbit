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
        print(output)
        return None

    lines = output.lower().replace("\t", "").split("\n")

    mem = 0
    time = 0
    cpu = 0

    for line in lines:
        if line.startswith("real"):
            time = line.split(" ")[-1]
        elif "percent of cpu this job got" in line:
            cpu = line.split(" ")[-1]
        elif "elapsed (wall clock) time" in line:
            split_line = line.split(" ")[-1].split(":")
            time = 60*float(split_line[0]) + float(split_line[1])
        elif "maximum resident set size" in line:
            split_line = [t.strip() for t in line.strip().split(" ")]
            if s == "Linux":
                mem = int(split_line[-1])
            elif s == "Darwin":
                mem = int(split_line[0])/1024.0
    return time, mem, cpu


if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:])
    time, mem, cpu = memit(cmd)
    print(f"{time}, {mem}, {cpu}")
