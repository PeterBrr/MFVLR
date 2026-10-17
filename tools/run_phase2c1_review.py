"""Run an auditable stage; retain all attempts, including failures."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "tokenizer": ["tests/foundation/test_tokenizer.py"],
    "vim": ["tests/foundation/test_vim.py"],
    "foundation": ["tests/foundation"],
    "regression": ["tests/test_mve.py", "tests/decoder"],
    "complete": ["tests"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args()
    out = ROOT/"artifacts/phase2c1"
    (out/"logs").mkdir(parents=True,exist_ok=True)
    (out/"junit").mkdir(exist_ok=True)
    index = 1
    while (out/f"logs/{args.stage}_{index:02}.log").exists():
        index += 1
    name = f"{args.stage}_{index:02}"
    xml = out/f"junit/{name}.xml"
    command = [sys.executable,"-m","pytest",*STAGES[args.stage],"-vv","--durations=10",f"--junitxml={xml}"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT/".phase2c1_deps"),str(ROOT/"src")])
    env["PYTHONIOENCODING"] = "utf-8"
    with (out/f"logs/{name}.log").open("w",encoding="utf-8") as log:
        log.write("COMMAND: "+json.dumps(command)+"\n")
        log.write("PYTHONPATH: "+env["PYTHONPATH"]+"\n")
        log.flush()
        process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   text=True,encoding="utf-8",errors="replace")
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line,end="",flush=True)
        code = process.wait()
        log.write(f"EXIT_CODE: {code}\n")
    (out/f"logs/{name}.json").write_text(json.dumps({"stage":args.stage,"attempt":index,"command":command,
        "exit_code":code,"junit":xml.relative_to(ROOT).as_posix()},indent=2)+"\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
