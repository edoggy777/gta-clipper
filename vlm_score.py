#!/usr/bin/env python3
# usage: vlm_score.py video.mp4  -> prints "seconds score" lines
import sys, cv2, torch, re
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from PIL import Image

STEP = 2  # seconds between samples
MODEL = "Qwen/Qwen3-VL-8B-Instruct"   # was Qwen/Qwen2.5-VL-3B-Instruct
model = AutoModelForImageTextToText.from_pretrained(MODEL, device_map="cuda",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16))
proc = AutoProcessor.from_pretrained(MODEL)

Q = ("This is a frame from GTA. Classify it with ONE label from: explosion, dancing, guns, drugs, violence, chase, emotional, landscape, airplane, scuba, boat, biker, police, fight, other. "
     "guns=weapons drawn or shooting; drugs=drug use, dealing, labs, pills, bags; explosion=fireball, blast, vehicle or building blowing up, big flames or smoke cloud; emotional=close-up of a sad, crying, hugging or thoughtful character, quiet dramatic moment; landscape=city skyline, streets, buildings, beach, mountains, sunset or other scenery shown as a wide view with no fighting, shooting or chase; violence=blood, injured or dead bodies, WASTED screen (no fight going on); "
     "chase=police pursuit, wanted stars, high-speed driving, crashes; airplane=a plane or jet (not a helicopter) flying, taking off, landing or parked, seen from outside or from its cockpit; scuba=underwater view: a diver in scuba gear, swimming below the surface, sea floor, reef, fish, sharks or a submarine; boat=a boat, speedboat, yacht, jet ski or ship on the water, seen from outside or from on board (not underwater); biker=motorcycles and biker gangs: someone riding a motorcycle or chopper, a group of bikers riding together, bikers in leather vests, parked motorcycles, a biker clubhouse (use biker instead of chase whenever a motorcycle is the main subject); dancing=people dancing in a nightclub, bar, strip club or party: dance floor, crowd dancing, a DJ, club lights and lasers, a character doing a dance move; police=police are the main subject: police cars or police motorcycles with lights flashing, officers on foot, a police helicopter, a roadblock, SWAT, an arrest or a police station (use police instead of chase whenever police are clearly visible in the frame); fight=hand-to-hand fighting: punching, kicking, wrestling, a brawl, or a melee fight with fists, bats or knives (use guns instead when a gun is being aimed or fired); other=calm driving, walking, menus, cutscenes, scenery. "
     "Then rate 0-10 how intense/clip-worthy it is. Reply exactly as: LABEL SCORE")

cap = cv2.VideoCapture(sys.argv[1]); fps = cap.get(cv2.CAP_PROP_FPS)
if not cap.isOpened(): raise SystemExit(f"cannot open {sys.argv[1]}")
t = 0
while True:
    cap.set(cv2.CAP_PROP_POS_MSEC, t*1000); ok, frame = cap.read()
    if not ok: break
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640,360))
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":Q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=8)
    ans = proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"(fight|dancing|airplane|scuba|boat|biker|police|explosion|guns|drugs|violence|chase|emotional|landscape|other)\D*(\d+)", ans.lower())
    label, score = (m.group(1), int(m.group(2))) if m else ("other", 0)
    print(t, score, label, flush=True)
    t += STEP
