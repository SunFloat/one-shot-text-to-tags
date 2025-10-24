from typing import List, Dict, Any
from cog import BasePredictor, Input
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json, re

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # mehrsprachig, performant

# Default-Tagliste (kann per custom_allowed_json zur Laufzeit überschrieben werden)
DEFAULT_ALLOWED = ["Rechnung","Support","Vertrieb","Spam","Kündigung","Zahlung","Termin","Other"]

def build_system(allowed: List[str], reason_lang: str) -> str:
    """reason_lang: 'de' | 'en' | 'auto' (auto = Sprache des Inputs)"""
    lang_rule = {
        "de": "Use German for 'reason'.",
        "en": "Use English for 'reason'.",
        "auto": "Use the INPUT language for 'reason'.",
    }.get(reason_lang, "Use the INPUT language for 'reason'.")
    return (
        "You are a multilingual (DE/EN) text-to-tags classifier.\n"
        "Return ONLY valid JSON with this schema:\n"
        '{"tags":[{"tag":"<one of ALLOWED>","confidence":<float 0..1>}], "reason":"<short explanation>"}\n'
        "Rules:\n"
        "- Choose 0..5 tags from ALLOWED, no duplicates.\n"
        "- Sort by confidence descending.\n"
        "- If nothing fits, return an empty list.\n"
        f"- {lang_rule}\n"
        f"ALLOWED = {allowed}\n"
    )

ONESHOT = (
    '### ONE-SHOT EXAMPLE\n'
    'Input: "Hallo, anbei die Rechnung für August als PDF."\n'
    'Output: {"tags":[{"tag":"Rechnung","confidence":0.93},{"tag":"Zahlung","confidence":0.61}],'
    '"reason":"Rechnung erwähnt; Zahlungsbezug naheliegend."}\n'
)

def build_prompt(user_text: str, allowed: List[str], reason_lang: str) -> str:
    system = build_system(allowed, reason_lang)
    return f"{system}\n{ONESHOT}\n### Now classify:\nInput: {json.dumps(user_text)}\nOutput:"

def safe_parse_allowed(custom_allowed_json: str) -> List[str]:
    if not custom_allowed_json.strip():
        return DEFAULT_ALLOWED
    try:
        arr = json.loads(custom_allowed_json)
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr) and len(arr) > 0:
            # dedup + keep order
            seen, out = set(), []
            for t in arr:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
            return out
    except Exception:
        pass
    return DEFAULT_ALLOWED

def extract_json(s: str, allowed: List[str]) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", s, re.S)
    data = {"tags": [], "reason": ""}
    if not m:
        return data
    try:
        data = json.loads(m.group(0))
    except Exception:
        return {"tags": [], "reason": ""}

    # Post-Processing: dedup, only allowed, normalize confidence
    seen = set()
    tags_out = []
    for item in data.get("tags", []):
        tag = item.get("tag")
        conf = item.get("confidence", 0)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0
        if not tag or tag in seen or tag not in allowed:
            continue
        seen.add(tag)
        tags_out.append({"tag": tag, "confidence": max(0.0, min(1.0, conf))})
    data["tags"] = tags_out
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
        temperature: float = Input(default=0.0),
        max_tags: int = Input(default=5),
        min_confidence: float = Input(default=0.5),
        custom_allowed_json: str = Input(default="", description="Optional JSON array to override ALLOWED tags"),
        language: str = Input(choices=["auto","de","en"], default="auto"),
        top_p: float = Input(default=0.9),
    ) -> Dict[str, Any]:
        # Erlaubte Tags bestimmen (zur Laufzeit überschreibbar)
        allowed = safe_parse_allowed(custom_allowed_json)

        # Prompt bauen (Reason-Sprache steuern)
        prompt = build_prompt(text, allowed, language)

        # Inference-Parameter: deterministisch, wenn temperature == 0
        do_sample = temperature > 0.0
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        raw = self.tokenizer.decode(out[0], skip_special_tokens=True)
        raw = raw.split("Output:")[-1].strip()
        data = extract_json(raw, allowed)

        # Post-Processing: Threshold + Limit anwenden
        if min_confidence is not None:
            data["tags"] = [t for t in data["tags"] if t.get("confidence", 0.0) >= float(min_confidence)]
        if max_tags is not None and max_tags > 0:
            data["tags"] = data["tags"][: int(max_tags)]

        return data
