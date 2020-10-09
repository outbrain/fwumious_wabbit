import os.path
import sys
import argparse
import timeit
import subprocess
import timeit


def time_bash_cmd(cmd, number=1):
   return timeit.Timer("subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)", f"import subprocess; cmd='{cmd}'").timeit(number=number)

parser = argparse.ArgumentParser(description='Process some integers.')
#parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                    help='an integer for the accumulator')
#parser.add_argument('--sum', dest='accumulate', action='store_const',
#                    const=sum, default=max,
#                    help='sum the integers (default: find the max)')

if __name__ == "__main__":
  args = parser.parse_args()
  vw_train_cmd = "vw --data train.vw.gz -l 0.1 -b 25 -c --adaptive --sgd --loss_function logistic --link logistic --power_t 0.0 --l2 0.0 --hash all --final_regressor vw_model --save_resume --interactions AB" 
  print(time_bash_cmd(vw_train_cmd))

