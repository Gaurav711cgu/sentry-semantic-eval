"""Compute a fresh unadapted base reference on development only, offline."""
import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from worker_v10 import label_ids, peak_rss, score_contents, _contents


def main():
    parser=argparse.ArgumentParser()
    for name in ['model','input','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--stage');parser.add_argument('--adapter-dir')
    args=parser.parse_args();result={'passed':False,'stage':'base-development-reference'}
    try:
        value=json.loads(args.input.read_text())
        if set(value)!={'dev_contents'}:raise ValueError('development-only input required')
        contents=_contents(value['dev_contents'],50)
        if len(contents)!=50:raise ValueError('exact50 development inputs required')
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from lora_v10 import install_lora
        torch.set_num_threads(4);torch.manual_seed(1019);torch.use_deterministic_algorithms(True)
        begun=time.monotonic();tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False);tok.padding_side='right'
        if tok.pad_token_id is None:tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='eager')
        model.config.use_cache=False;install_lora(model)
        metadata={'python':sys.version.split()[0],'torch':torch.__version__}
        for name in ['transformers','safetensors','numpy','tokenizers','huggingface-hub']:metadata[name]=importlib.metadata.version(name)
        result.update(passed=True,metadata=metadata,records=score_contents(model,tok,contents,label_ids(tok),begun),peak_rss_bytes=peak_rss(),training_performed=False,final_inference_performed=False)
    except Exception as error:result.update(passed=False,failure_type=type(error).__name__,failure=str(error)[:2000])
    args.output.write_text(json.dumps(result,allow_nan=False));return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
