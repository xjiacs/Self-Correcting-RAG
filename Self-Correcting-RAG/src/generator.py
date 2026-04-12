
from __future__ import annotations
from typing import List, Dict, Any, Optional
import os, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from .prompts import SYSTEM_PROMPT, USER_PROMPT

class QwenLocal:
    def __init__(self, model_path: str, device: Optional[str]=None, dtype: str="bfloat16", max_new_tokens: int=256,
                 log_io: bool=False, log_path: str="outputs/qa_io.txt"):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=getattr(torch, dtype), device_map="auto")
        self.max_new_tokens = max_new_tokens
        self.log_io = log_io
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def build_prompt(self, question: str, docs: List[Dict[str, Any]]) -> str:
        ctx_lines = []
        for d in docs:
            ctx_lines.append(f"[doc_id={d.get('doc_id')}] title={d.get('title','')}\n{d.get('text','').strip()}\n")
        ctx_text = "\n---\n".join(ctx_lines)
        user = USER_PROMPT.format(question=question.strip(), context=ctx_text)
        prompt = f"<|system|>\n{SYSTEM_PROMPT}\n</|system|>\n<|user|>\n{user}\n</|user|>\n<|assistant|>"
        return prompt

    @torch.inference_mode()
    def generate(self, question: str, docs: List[Dict[str, Any]], temperature: float=0.0, top_p: float=0.95, logits_processors: LogitsProcessorList | None = None) -> str:
        prompt = self.build_prompt(question, docs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, do_sample=temperature>0, temperature=temperature, top_p=top_p,
                                      max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id,
                                      logits_processor=logits_processors)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<|assistant|>" in text:
            text = text.split("<|assistant|>")[-1].strip()

        if self.log_io:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"question": question, "prompt": prompt, "output": text}, ensure_ascii=False) + "\n")
        return text
