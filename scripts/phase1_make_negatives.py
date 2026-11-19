"""Phase 1: LLM-generated subtle negatives for GenAI-Bench prompts (NoisyCLIP-style misalignment categories).

For every prompt an instruction LLM writes minimally edited, *misaligned* versions, one per category:
  count       change a number / quantity            ("three dogs" -> "two dogs")
  color       change a color / attribute            ("a red car" -> "a blue car")
  background  change the scene / setting            ("on a beach" -> "in a forest")
  subject     replace the main subject or swap two objects ("a cat chasing a dog" -> "a dog chasing a cat")
  relation    change a spatial / action relation    ("left of" -> "right of", "riding" -> "pushing")
A category is null when it does not apply (e.g. no number in the prompt).

Output: dataset/genai_bench/negatives.json  {prompt_id: {"prompt": str, "count": str|null, ...}}
Usage: sbatch make_negatives.slurm  (one GPU, ~10-20 min for 1600 prompts with a 7B model)
"""
import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_noisyclip import load_genai_bench  # noqa: E402

CATS = ["count", "color", "background", "subject", "relation"]

SYSTEM = (
    "You create hard negative captions for evaluating text-to-image alignment. Given an image caption, write "
    "minimally edited versions that describe a DIFFERENT image, changing exactly one aspect and keeping all other "
    "words. Categories:\n"
    "- count: change a number or quantity (e.g. 'three dogs' -> 'two dogs'). A single clearly visible object may "
    "become several ('a cat' -> 'three cats'). null only if no object can be counted.\n"
    "- color: change one color or visual attribute (material, size, pattern) that the caption STATES. Do not add "
    "a new attribute (adding 'silver' to 'an oven' is not a contradiction). null if no attribute is stated.\n"
    "- background: change the scene, place or setting. If none is stated, add a clearly different one only if it "
    "makes the caption wrong; otherwise null.\n"
    "- subject: replace the main subject with a different but plausible one, or swap the roles of two objects.\n"
    "- relation: change a spatial relation (left/right, above/below, inside/outside, in front/behind) or an action "
    "between objects. null if there is no relation.\n"
    "Never add a new object or a new sentence: replace words in place. The edited caption must be clearly false for "
    "an image that matches the original. Answer ONLY with a JSON "
    'object with keys "count", "color", "background", "subject", "relation"; each value is a string or null.'
)

EXAMPLE_IN = "Three red apples on a wooden table, the one on the left is bitten."
EXAMPLE_OUT = json.dumps({
    "count": "Five red apples on a wooden table, the one on the left is bitten.",
    "color": "Three green apples on a wooden table, the one on the left is bitten.",
    "background": "Three red apples on a snowy rooftop, the one on the left is bitten.",
    "subject": "Three red tomatoes on a wooden table, the one on the left is bitten.",
    "relation": "Three red apples under a wooden table, the one on the right is bitten.",
})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--genai-bench", default="dataset/genai_bench")
    p.add_argument("--out", default="dataset/genai_bench/negatives.json")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0, help="only the first N prompts (debug)")
    return p.parse_args()


def small_edit(a, b, max_changed=6):
    """Reject rewrites that change much more than a few words (multi-aspect edits, appended sentences)."""
    import difflib
    wa, wb = a.lower().split(), b.lower().split()
    sm = difflib.SequenceMatcher(None, wa, wb)
    changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    return 0 < changed <= max_changed


def parse_json(text, prompt):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for c in CATS:
        v = d.get(c)
        v = v.strip() if isinstance(v, str) else None
        if v and v.lower().strip(" .") in ("null", "none", "n/a", ""):
            v = None
        if v and not small_edit(prompt, v):
            v = None
        # reject empty / unchanged negatives
        out[c] = v if v and v.lower().strip(" .") != prompt.lower().strip(" .") else None
    return out


def main():
    args = parse_args()
    prompts, _ = load_genai_bench(args.genai_bench)
    ids = sorted(prompts)[: args.limit or None]
    done = {}
    if os.path.exists(args.out):
        done = {int(k): v for k, v in json.load(open(args.out)).items()}
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} prompts, {len(todo)} to do", flush=True)

    tok = AutoTokenizer.from_pretrained(args.llm, padding_side="left")
    llm = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    def chat(p):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": EXAMPLE_IN}, {"role": "assistant", "content": EXAMPLE_OUT},
                {"role": "user", "content": p}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    failed = 0
    for b0 in range(0, len(todo), args.batch_size):
        bid = todo[b0:b0 + args.batch_size]
        for attempt in range(3):  # retry unparsable outputs with sampling
            if not bid:
                break
            enc = tok([chat(prompts[i]) for i in bid], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                gen = llm.generate(**enc, max_new_tokens=400, do_sample=attempt > 0, temperature=0.7, top_p=0.9)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            retry = []
            for i, t in zip(bid, texts):
                d = parse_json(t, prompts[i])
                if d is None:
                    retry.append(i)
                else:
                    done[i] = {"prompt": prompts[i], **d}
            bid = retry
        failed += len(bid)
        with open(args.out, "w") as f:
            json.dump({str(k): done[k] for k in sorted(done)}, f, indent=1)
        print(f"[{min(b0 + args.batch_size, len(todo))}/{len(todo)}] failed so far {failed}", flush=True)

    cover = {c: sum(v[c] is not None for v in done.values()) for c in CATS}
    print(f"saved {args.out}: {len(done)} prompts; non-null per category {cover}", flush=True)


if __name__ == "__main__":
    main()
