import os.path
import sys

def rm_quietly(f):
  if os.path.isfile(f):
    os.remove(f)

if __name__ == "__main__":
    rm_quietly("train.vw")
    rm_quietly("train.vw.gz")
    rm_quietly("train.vw.cache")
    rm_quietly("train.vw.fwcache")
    rm_quietly("easy.vw")
    rm_quietly("easy.vw.cache")
    rm_quietly("easy.vw.fwcache")
    rm_quietly("hard.vw")
    rm_quietly("hard.vw.cache")
    rm_quietly("hard.vw.fwcache")

