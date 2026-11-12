#!/usr/bin/env python3
# usage: polish.py --src video.mp4 --start 120 --out final.mp4 [--len 30] [--hook "TEXT"] [--handle "@you"]
import argparse, subprocess, re, tempfile, os, numpy as np, cv2, soundfile as sf
from pathlib import Path
from faster_whisper import WhisperModel

FF="/usr/bin/ffmpeg"; W,H=1080,1920
os.chdir(Path.home()/"clipper")

def run(cmd):
    r=subprocess.run(cmd,capture_output=True,text=True)
    if r.returncode: print(r.stderr[-3000:]); raise SystemExit(f"ffmpeg failed ({r.returncode})")

def loudness(path,hop=0.5):
    n=int(48000*hop)
    out=subprocess.run([FF,"-i",path,"-af",f"aresample=48000,asetnsamples=n={n},astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level","-f","null","-"],capture_output=True,text=True).stderr
    vals=[float(x) if x not in("-inf","inf","nan") else -90.0 for x in re.findall(r"RMS_level=(\S+)",out)]
    return np.array(vals),hop

def steady(cx, cuts, fps, w):
    """Calm camera: locked on short shots, slow follow with a dead zone on long ones, reset at cuts."""
    n = len(cx); out = np.full(n, w / 2.0); idx = np.arange(n)
    edges = [0] + [c for c in cuts if 0 < c < n] + [n]
    dz = 0.12 * w; win = max(3, int(fps) | 1)
    for a, b in zip(edges[:-1], edges[1:]):
        seg = cx[a:b]; good = ~np.isnan(seg)
        if not good.any(): continue
        seg = np.interp(idx[a:b], idx[a:b][good], seg[good])
        pad = np.pad(seg, win // 2, mode="edge")
        tgt = np.median(np.lib.stride_tricks.sliding_window_view(pad, win), axis=1)   # ignore brief hops
        tgt = np.convolve(np.pad(tgt, win // 2, mode="edge"), np.ones(win) / win, mode="valid")
        if b - a < 2.5 * fps or tgt.max() - tgt.min() < 2 * dz:
            out[a:b] = np.median(seg); continue                 # hold the camera still
        cam = np.empty(b - a); c = float(np.median(tgt[:win]))
        for k in range(b - a):
            e = tgt[k] - c
            if abs(e) > dz: c += (e - np.sign(e) * dz) * (2.0 / fps)   # ease toward the subject
            cam[k] = c
        h2 = max(3, (win // 2) | 1)
        out[a:b] = np.convolve(np.pad(cam, h2 // 2, mode="edge"), np.ones(h2) / h2, mode="valid")
    return out

def track(path):
    from ultralytics import YOLO
    model=YOLO("yolov8n.pt")
    cap=cv2.VideoCapture(path); fps=cap.get(5) or 30; n=int(cap.get(7)); w=int(cap.get(3)); h=int(cap.get(4))
    cx=np.full(n,np.nan); prev=None; ph=None; last=None; cuts=[]
    for i in range(n):
        ok,f=cap.read()
        if not ok: break
        if i%3: continue
        g=cv2.cvtColor(cv2.resize(f,(320,180)),cv2.COLOR_BGR2GRAY)
        gh=cv2.calcHist([g],[0],None,[32],[0,256])
        cut=prev is not None and cv2.absdiff(g,prev).mean()>25 and cv2.compareHist(gh,ph,cv2.HISTCMP_CORREL)<0.7
        if cut: cuts.append(i); last=None                       # new shot: forget the old subject
        r=model.predict(f,imgsz=640,conf=0.3,verbose=False)[0]
        if len(r.boxes):
            b=r.boxes.xyxy.cpu().numpy(); a=(b[:,2]-b[:,0])*(b[:,3]-b[:,1]); c=(b[:,0]+b[:,2])/2
            if last is not None: a=a/(1+((c-last)/(0.15*w))**2)  # stay on the same subject
            j=int(a.argmax()); cx[i]=last=c[j]
        elif prev is not None and not cut:
            m=cv2.moments(cv2.absdiff(g,prev))
            if m["m00"]>1e3: cx[i]=m["m10"]/m["m00"]*w/320
        prev=g; ph=gh
    cap.release()
    return steady(cx,cuts,fps,w),fps,w,h

def env(t):  # punch-in envelope around peak
    if t<-0.25 or t>1.4: return 0.0
    if t<0: return (t+0.25)/0.25
    if t<1.0: return 1.0
    return 1-(t-1.0)/0.4

def render(path,cx,fps,w,h,s,e,peak,out):
    cap=cv2.VideoCapture(path); vw=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))
    f0,f1=int(s*fps),int(e*fps); cap.set(cv2.CAP_PROP_POS_FRAMES,f0)
    for i in range(f0,f1):
        ok,f=cap.read()
        if not ok: break
        z=1+0.12*env(i/fps-peak); ch=h/z; cw=ch*9/16
        x=min(max(cx[min(i,len(cx)-1)]-cw/2,0),w-cw); y=(h-ch)/2
        vw.write(cv2.resize(f[int(y):int(y+ch),int(x):int(x+cw)],(W,H)))
    vw.release(); cap.release()

def esc(s): return s.replace("{","(").replace("}",")").strip()

def ass(words,hook,handle,dur,path):
    def t(x): return f"{int(x//3600)}:{int(x%3600//60):02}:{x%60:05.2f}"
    head=f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,78,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,560,1
Style: Hook,DejaVu Sans,84,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,8,60,60,230,1
Style: Handle,DejaVu Sans,38,&H00DDDDDD,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,0,2,60,60,110,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev=[]
    if hook: ev.append(f"Dialogue: 1,{t(0)},{t(min(3,dur))},Hook,,0,0,0,,{{\\fad(0,250)}}{esc(hook)}")
    if handle: ev.append(f"Dialogue: 0,{t(0)},{t(dur)},Handle,,0,0,0,,{esc(handle)}")
    chunk=[]
    def flush():
        if not chunk: return
        txt=" ".join(f"{{\\k{max(1,int(round((b-a)*100)))}}}{esc(wd)}" for wd,a,b in chunk)
        ev.append(f"Dialogue: 0,{t(chunk[0][1])},{t(chunk[-1][2])},Cap,,0,0,0,,{txt}"); chunk.clear()
    for wd,a,b in words:
        if chunk and (len(chunk)>=3 or a-chunk[-1][2]>0.6 or b-chunk[0][1]>1.6): flush()
        chunk.append((wd,a,b))
    flush()
    Path(path).write_text(head+"\n".join(ev)+"\n")

def gen_hook(transcript):
    p=f"GTA clip transcript: '{transcript[:400]}'. Write a 4-6 word all-caps TikTok hook. Output only the hook, no quotes."
    r=subprocess.run(["ollama","run","qwen2.5:7b",p],capture_output=True,text=True).stdout.strip()
    return re.sub(r"[^A-Z0-9 !?'.]","",r.upper())[:40] or "WATCH THIS"

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--src",required=True); ap.add_argument("--start",type=float,required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--len",type=float,default=30)
    ap.add_argument("--lead",type=float,default=8); ap.add_argument("--tail",type=float,default=14)
    ap.add_argument("--peak",type=float,default=None); ap.add_argument("--hook",default=None); ap.add_argument("--no-hook",action="store_true"); ap.add_argument("--handle",default="")
    a=ap.parse_args(); tmp=Path(tempfile.mkdtemp())
    win=str(tmp/"win.mp4")
    run([FF,"-y","-ss",str(a.start),"-t",str(a.len),"-i",a.src,"-c:v","libx264","-preset","fast","-crf","18","-c:a","aac","-ar","48000",win])
    if a.peak is not None: peak=a.peak
    else: curve,hop=loudness(win); peak=float(curve.argmax())*hop+hop/2
    s=max(0.0,peak-a.lead); e=min(a.len,peak+a.tail); dur=e-s
    cx,fps,w,h=track(win)
    render(win,cx,fps,w,h,s,e,peak,str(tmp/"vid.mp4"))
    aud=str(tmp/"aud.wav"); ln="loudnorm=I=-14:TP=-1.5:LRA=11"
    cmd=[FF,"-y","-ss",str(s),"-t",str(dur),"-i",win]
    sfx=Path.home()/"clipper/sfx/impact.wav"
    if sfx.exists():
        d=int((peak-s)*1000)
        cmd+=["-i",str(sfx),"-filter_complex",f"[1:a]adelay={d}|{d},volume=0.8[x];[0:a][x]amix=inputs=2:duration=first:normalize=0,{ln}[a]","-map","[a]"]
    else: cmd+=["-af",ln]
    run(cmd+["-vn","-ar","48000",aud])
    model=WhisperModel("small",device="cuda",compute_type="float16")
    aud16=str(tmp/"aud16.wav"); run([FF,"-y","-i",aud,"-ac","1","-ar","16000",aud16])
    samples,_=sf.read(aud16,dtype="float32")
    segs,_=model.transcribe(samples,word_timestamps=True)
    words=[(x.word.strip(),x.start,x.end) for sg in segs for x in sg.words]
    hook="" if a.no_hook else (a.hook or gen_hook(" ".join(w for w,_,_ in words)))
    ass(words,hook,a.handle,dur,str(tmp/"cap.ass"))
    run([FF,"-y","-i",str(tmp/"vid.mp4"),"-i",aud,"-vf",f"ass={tmp/'cap.ass'},fade=t=in:st=0:d=0.4,fade=t=out:st={dur-0.5:.2f}:d=0.5",
         "-af",f"afade=t=in:st=0:d=0.4,afade=t=out:st={dur-0.5:.2f}:d=0.5","-c:v","libx264","-preset","slow","-crf","17","-pix_fmt","yuv420p","-c:a","aac","-b:a","192k","-shortest",a.out])
    print("polished",a.out,"peak",round(peak,1),"hook",hook)
