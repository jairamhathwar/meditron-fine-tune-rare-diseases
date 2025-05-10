import pandas as pd, torch, re
from transformers import AutoTokenizer
from datasets import Dataset
from peft import AutoPeftModelForCausalLM

def prf1(pred, gold):
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not gold:
        return 0.0, 0.0, 0.0
    pred = [p.lower() for p in pred]
    gold = [g.lower() for g in gold]
    tp   = [p for p in pred if p in gold]
    prec = len(tp) / len(pred) if pred else 0.0
    rec  = len(tp) / len(gold)
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1

ANSWER_RE = re.compile(r"answer\s*[:\-]\s*", re.I)

def extract_answer(text: str) -> str:
    """Return the substring after the final 'Answer:' marker; fallback last line."""
    if "answer" in text.lower():
        text = ANSWER_RE.split(text)[-1]
    return text.strip()

lora_paths = {
    "rank4": {
        "adapter": "/scratch/network/jh3258/meditron-fine-tune-rare-diseases/results/rank4/adapter",
        "tok"   :  "/scratch/network/jh3258/meditron-fine-tune-rare-diseases/results/rank4/tokenizer",
    },
    "rank8": {
        "adapter": "/scratch/network/jh3258/meditron-fine-tune-rare-diseases/results/rank8/adapter",
        "tok"   :  "/scratch/network/jh3258/meditron-fine-tune-rare-diseases/results/rank8/tokenizer",
    },
}

device = "cuda" if torch.cuda.is_available() else "cpu"
models, toks = {}, {}
for rk, p in lora_paths.items():
    models[rk] = AutoPeftModelForCausalLM.from_pretrained(
        p["adapter"],
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
    ).to(device).eval()

    tk = AutoTokenizer.from_pretrained(p["tok"])
    tk.pad_token, tk.padding_side = tk.eos_token, "right"
    toks[rk] = tk

test_df = pd.read_csv(
    "/scratch/network/jh3258/meditron-fine-tune-rare-diseases/data/test_out.csv"
)
test_ds = Dataset.from_pandas(test_df).shuffle(seed=42)

def build_prompt_cot(sample):
    cat = sample["category"].upper()
    header, definition = {
        "RAREDISEASE": (
            "rare diseases",
            "Rare diseases are defined as diseases that affect a small number of people compared to the general population.",
        ),
        "DISEASE": (
            "diseases",
            "Diseases are defined as abnormal conditions resulting from various causes, such as infection, inflammation, environmental factors, or genetic defect, and characterized by an identifiable group of signs, symptoms, or both.",
        ),
        "SYMPTOM": (
            "symptoms",
            "Symptoms are defined as physical or mental problems that cannot be measured from tests or observed by a doctor.",
        ),
        "SIGN": (
            "signs",
            "Signs are defined as physical or mental problems that can be measured from tests or observed by a doctor.",
        ),
    }[cat]

    return (
        f"### Task:\nExtract the exact name or names of {header} from the input text "
        f"and output them in a list.\n"
        f"First, reason through the text and explain how you would extract the names. "
        f"Break down the reasoning into steps and then provide your final answer.\n\n"
        f"### Definition:\n{definition}\n\n"
        f"### Input Text:\n{sample['context']}\n\n"
        f"### Output (reasoning):"
    )

GLOBAL = {"rank_name": [], "precision": [], "recall": [], "f1": []}
BY_CAT = []

for rk in lora_paths:
    model, tok = models[rk], toks[rk]
    cat_scores = {c: {"p": [], "r": [], "f": []}
                  for c in ["RAREDISEASE", "DISEASE", "SYMPTOM", "SIGN"]}

    for sample in test_ds:
        prompt = build_prompt_cot(sample)
        with torch.no_grad():
            enc = tok(prompt, return_tensors="pt",
                      truncation=True, max_length=2048).to(device)
            out_ids = model.generate(
                **enc,
                max_new_tokens=256,
                pad_token_id=tok.pad_token_id,
                do_sample=False
            )
        full = tok.decode(out_ids[0], skip_special_tokens=True)
        reason_plus_ans = full[len(prompt):].strip()
        answer_text = extract_answer(reason_plus_ans)

        pred = [s.strip() for s in re.split(r"[,\n]+", answer_text) if s.strip()]
        gold = [s.strip() for s in re.split(r"[,\n]+", sample["response"]) if s.strip()]

        p, r, f1 = prf1(pred, gold)
        c = sample["category"].upper()
        cat_scores[c]["p"].append(p)
        cat_scores[c]["r"].append(r)
        cat_scores[c]["f"].append(f1)

    n_docs = len(test_ds)
    GLOBAL["rank_name"].append(rk)
    GLOBAL["precision"].append(sum(sum(v["p"]) for v in cat_scores.values()) / n_docs)
    GLOBAL["recall"].append   (sum(sum(v["r"]) for v in cat_scores.values()) / n_docs)
    GLOBAL["f1"].append       (sum(sum(v["f"]) for v in cat_scores.values()) / n_docs)

    for c, d in cat_scores.items():
        BY_CAT.append({
            "rank_name": rk,
            "category" : c,
            "precision": sum(d["p"]) / len(d["p"]),
            "recall"   : sum(d["r"]) / len(d["r"]),
            "f1"       : sum(d["f"]) / len(d["f"]),
        })

out_dir = "/scratch/network/jh3258/meditron-fine-tune-rare-diseases/data"
pd.DataFrame(GLOBAL).to_csv(f"{out_dir}/metrics_overall_cot.csv", index=False)
pd.DataFrame(BY_CAT).to_csv(f"{out_dir}/metrics_by_entity_cot.csv", index=False)

print("✓ Wrote metrics_overall_cot.csv and metrics_by_entity_cot.csv")
