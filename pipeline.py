#!/usr/bin/env python3
import sys, subprocess, re, numpy as np
from pathlib import Path

WANT={"drugs","chase"}
CLIP_LEN=30; N_CLIPS=3; PRE=20; STEP=2; HANDLE=""
OUT=Path.home()/"clipper/outbox"; OUT.mkdir(parents=True,exist_ok=True); (Path.home()/"clipper/done").mkdir(exist_ok=True); FF="/usr/bin/ffmpeg"
PY=str(Path.home()/"faceless/tts/bin/python3"); HOME=Path.home()

def loudness_curve(src):
    out=subprocess.run([FF,"-i",str(src),"-af","aresample=48000,asetnsamples=n=48000,astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level","-f","null","-"],capture_output=True,text=True).stderr
    return np.array([float(x) if x not in("-inf","inf","nan") else -90.0 for x in re.findall(r"RMS_level=(\S+)",out)])

def vlm_curve(src,n):
    cache=Path(str(src)+".scores.txt")
    if cache.exists(): out=cache.read_text()
    else:
        r=subprocess.run([PY,str(HOME/"clipper/vlm_score.py"),str(src)],capture_output=True,text=True)
        if r.returncode or not r.stdout.strip(): print(r.stderr[-3000:]); raise SystemExit("VLM scoring failed")
        out=r.stdout; cache.write_text(out)
    s=np.zeros(n)
    for line in out.splitlines():
        t,v,lab=line.split(); t=int(float(t))
        if t<n and lab in WANT: s[t:t+STEP]=float(v)
    return s

def norm(a): return (a-a.min())/(a.max()-a.min()+1e-9)

def pick(curve):
    m=[]; c=curve.copy()
    while len(m)<N_CLIPS and c.size:
        i=int(np.argmax(c)); m.append(max(0,i-PRE)); c[max(0,i-CLIP_LEN):i+CLIP_LEN]=-999
    return sorted(m)

def main(src):
    src=Path(src); loud=loudness_curve(src); n=len(loud)
    v=vlm_curve(src,n); score=(0.7*norm(v)+0.3*norm(loud))*(v>=5)
    if score.max()<=0: raise SystemExit("no moments matched WANT with score>=5")
    for k,start in enumerate(pick(score),1):
        out=OUT/f"{src.stem}_{k}.mp4"
        subprocess.run([PY,str(HOME/"clipper/polish.py"),"--src",str(src),"--start",str(start),"--len",str(CLIP_LEN),"--out",str(out),"--handle",HANDLE,"--peak",str(PRE),"--no-hook"],check=True)
        print("made",out,"score",round(float(score[min(start+PRE,n-1)]),2))
    src.rename(HOME/"clipper/done"/src.name)

if __name__=="__main__": main(sys.argv[1])
