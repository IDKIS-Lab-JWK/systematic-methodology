"""Provision one model environment without combining incompatible stacks."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
SPECS={"3DINO":("ThreeDINO","3dino"),"FlexiCT":("FlexiCT","flexict"),"COLIPRI":("COLIPRI","colipri"),"Pillar0":("Pillar0","pillar0"),"Merlin":("Merlin","merlin"),"MedicalNet":("_cnn","cnn"),"DenseNet":("_cnn","cnn")}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model",required=True,choices=SPECS)
    a=p.parse_args();module,name=SPECS[a.model]
    if a.model=="3DINO" and (sys.version_info[:2]!=(3,10) or sys.platform!="win32"):
        p.error("The pinned 3DINO stack requires Windows x64 CPython 3.10; invoke this helper with that interpreter.")
    env=ROOT/(".venv_"+name)
    subprocess.run([sys.executable,"-m","venv",str(env)],check=True)
    python=env/("Scripts/python.exe" if sys.platform=="win32" else "bin/python")
    subprocess.run([str(python),"-m","pip","install","-r",str(ROOT/"Code/Models"/module/"requirements.txt")],check=True)
    print("Environment ready:",python)
if __name__=="__main__":main()
