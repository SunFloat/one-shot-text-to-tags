from typing import List, Dict, Any
from cog import BasePredictor, Input
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json, re

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # mehrsprachig, performant

ALLOWED = ["Rechnung","Support","Vertrieb","Spam","Kündigung","Zahlung","Termin","Other"]

SYSTEM = (
"You are a multilingual (DE/EN) text-to-tags classifier.\n"
"Return ONLY valid JSON with this schema:\n"
'{"tags":[{"tag":"<one of ALLOWED>","confidence":<float 0..1>}], "reason":"<short explanation in input language>"}\n'
"Rules:\n- Choose 0..5 tags from ALLOWED, no duplicates.\n"
"- Sort by confidence descending.\n"
"- If nothing fits, return an empty list.\n"
f"ALLOWED = {ALLOWED}\n"
)

ONESHOT = (
'### ONE-SHOT EXAMPLE\n'
'Input: "Hallo, anbei die Rechnung für August als PDF."\n'
'Output: {"tags":[{"tag":"Rechnung","confidence":0.93},{"tag":"Zahlung","confidence":0.61}],"reason":"Rechnung erwähnt; Zahlungsbezug naheliegend."}\n'
)

def build_prompt(user_text: str) -> str:
    return f"{SYSTEM}\n{ONESHOT}\n### Now classify:\nInput: {json.dumps(user_text)}\nOutput:"

def extract_json(s: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", s, re.S)
    data = {"tags": [], "reason": ""}
    if not m:
        return data
    try:
        data = json.loads(m.group(0))
    except Exception:
        pass
    # Post-Processing: dedup, limit 5, sanity checks
    seen = set()
    tags = []
    for item in data.get("tags", []):
        tag = item.get("tag")
        conf = float(item.get("confidence", 0))
        if not tag or tag in seen or tag not in ALLOWED:
            continue
        seen.add(tag)
        tags.append({"tag": tag, "confidence": max(0.0, min(1.0, conf))})
        if len(tags) == 5:
            break
    data["tags"] = tags
    if not isinstance(data.get("reason", ""), str):
        data["reason"] = ""
    return data

class Predictor(BasePredictor):
    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
    def predict(
        self,
        text: str = Input(description="DE/EN input text to classify."),
        max_new_tokens: int = Input(default=200),
        temperature: float = Input(default=0.1),
        top_p: float = Input(default=0.9),
    ) -> Dict[str, Any]:
        prompt = build_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # nimm nur den letzten "Output:"-Teil
        raw = raw.split("Output:")[-1].strip()
        return extract_json(raw)
