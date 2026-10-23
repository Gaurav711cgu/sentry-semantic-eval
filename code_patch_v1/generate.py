"""One frozen development code-generation attempt; never executes output."""
import hashlib,json,time,resource
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
root=Path(__file__).resolve().parent
p=json.loads((root/'protocol.json').read_text());prompt=(root/'prompt.txt').read_text()
assert hashlib.sha256(prompt.encode()).hexdigest()==p['prompt_sha256']
report={'protocol':p,'inference_completed':False,'candidate_executed':False}
start=time.monotonic()
try:
 torch.set_num_threads(4)
 tokenizer=AutoTokenizer.from_pretrained(p['model'],revision=p['revision'],trust_remote_code=False)
 ids=tokenizer.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,return_tensors='pt')
 report['input_tokens']=ids.shape[-1]
 assert ids.shape[-1]<=p['max_input_tokens']
 model=AutoModelForCausalLM.from_pretrained(p['model'],revision=p['revision'],trust_remote_code=False,torch_dtype=torch.float32)
 model.eval()
 with torch.inference_mode():
  output=model.generate(ids,max_new_tokens=p['max_new_tokens'],do_sample=False,max_time=800,pad_token_id=tokenizer.eos_token_id)
 text=tokenizer.decode(output[0,ids.shape[-1]:],skip_special_tokens=True)
 (root/'candidate.txt').write_text(text)
 report.update(inference_completed=True,generated_tokens=output.shape[-1]-ids.shape[-1],candidate_sha256=hashlib.sha256(text.encode()).hexdigest())
except Exception as error:
 report['error']=type(error).__name__+': '+str(error)
 raise
finally:
 report['elapsed_seconds']=time.monotonic()-start
 report['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
