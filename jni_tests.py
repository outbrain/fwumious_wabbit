#!/usr/bin/env python3

import subprocess
import os
import sys
import re
import time
import shutil
from typing import List, Set, Optional


RELEASE = "release"
DEBUG = "debug"
JNI_TESTS = "jni_tests"


def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            #            print("removing %s" % os.path.join(dir, f))
            os.remove(os.path.join(dir, f))

def find_dir(dir_name: str, start_dir: str) -> str:
    origin_cwd = os.getcwd()
    os.chdir(start_dir)
    dir = os.getcwd()
    last_dir = ''
    while last_dir != dir:
        dir = os.getcwd()
        if dir_name in [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]:
            ret = os.path.join(dir, dir_name)
            os.chdir(origin_cwd)
            return ret
        os.chdir('..')
        last_dir = os.getcwd()
    os.chdir(origin_cwd)
    raise Exception("Can not find %s" % dir_name)


def build_jar(java_dir: str, java_native_dir: str, use_shell: bool) -> str:
    generated_java = [os.path.join(java_native_dir, f) for f in os.listdir(java_native_dir)
                      if os.path.isfile(os.path.join(java_native_dir, f)) and f.endswith(".java")]
    javac_cmd_args = ["javac", "Main.java"]
    javac_cmd_args.extend(generated_java)

    sys.stdout.flush()
    print(javac_cmd_args)
    subprocess.check_call(javac_cmd_args,
                          cwd=java_dir, shell=use_shell)

    jar_dir = str(os.path.join(os.getcwd(), "jni_tests", "src"))
    purge(java_dir, ".*\.jar$")
#    print (jar_dir, "YYYYYYYYY")
#    print("SDD", ["jar", "cfv", "Test.jar", "src", "/home/minmax/obgit/fwumious_wabbit/jni_tests", jar_dir])
    subprocess.check_call(["jar", "cfv", "Test.jar", "com/examples", "-C", "/home/minmax/obgit/fwumious_wabbit/java/src", "com/outbrain"], cwd=jar_dir, shell=use_shell)
    return jar_dir

def find_path_to_cargo_artifacts(path_to_crate: str, cfg: str) -> str:
    target_dir = find_dir("target", path_to_crate)
    if "CARGO_BUILD_TARGET" in os.environ:
        return os.path.join(target_dir, os.environ["CARGO_BUILD_TARGET"], cfg)
    else:
        return os.path.join(target_dir, cfg)
        
        

def run_jar(target_dir: str, jar_dir: str, use_shell: bool, extra_args):
    jvm_args = ["java"]
    jvm_args.extend(extra_args)
    jvm_args.extend(["-ea", "-Djava.library.path=" + target_dir,
                     "-cp", "Test.jar", "com.examples.Main"])
    subprocess.check_call(jvm_args, cwd=jar_dir, shell=use_shell)

        

def run_jni_tests(use_shell: bool, test_cfg: Set[str]):
    print("run_jni_tests begin: cwd %s" % os.getcwd())
    sys.stdout.flush()
    for cfg in test_cfg:
        if cfg == DEBUG:
            subprocess.check_call(["cargo", "build"], shell=False)
        elif cfg == RELEASE:
            subprocess.check_call(["cargo", "build", "--release"], shell=False)
        else:
            raise Exception("Fatal Error: Unknown cfg %s" % cfg)



    # Remove the java bindings classes
    java_native_dir = str(os.path.join(os.getcwd(), "java", "src", "com", "outbrain", "fw"))
    if not os.path.exists(java_native_dir):
        os.makedirs(java_native_dir)
    else:
        purge(java_native_dir, ".*\.class$")

    # Remove test clasess
    java_test_dir = str(os.path.join(os.getcwd(), "jni_tests", "src", "com", "examples"))
    purge(java_test_dir, ".*\.class$")
    jar_dir = build_jar(java_test_dir, java_native_dir, use_shell)



    for cfg in test_cfg:
        target_dir = find_path_to_cargo_artifacts("jni_tests", cfg)
        run_jar(target_dir, jar_dir, use_shell, ["-Xcheck:jni", "-verbose:jni"])
#    if RELEASE in test_cfg:
#        target_dir = find_path_to_cargo_artifacts("jni_tests", RELEASE)
#        run_jar(target_dir, jar_dir, use_shell, ["-Xcomp"])


def main():
    print("Starting build and test: %s" % sys.version)
    sys.stdout.flush()

    test_cfg = set([RELEASE])
    test_set = set([JNI_TESTS])

    has_jdk = "JAVA_HOME" in os.environ
    if (JNI_TESTS in test_set) and (not has_jdk):
        raise Exception("Fatal error JAVA_HOME not defined, so it is impossible to run %s" % JNI_TESTS)

    # becuase of http://bugs.python.org/issue17023
    is_windows = os.name == 'nt'
    use_shell = is_windows

    print("test_set %s" % test_set)
    sys.stdout.flush()

    print("start tests: %s" % test_set)
    if JNI_TESTS in test_set:
        run_jni_tests(use_shell, test_cfg)


if __name__ == "__main__":
    main()
