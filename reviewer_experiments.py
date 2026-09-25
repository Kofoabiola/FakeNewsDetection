"""
Experiments required by the reviewers (MDPI Analytics, 12-09-2026 decision).

  exp1  Disentangling control: RoBERTa + LoRA vs RoBERTa full FT, multi-seed  (R1.2, R1.3, R2.2, R2.3, R3.1)
  exp2  Unified evaluation: all models on the SAME sample IDs, same filtering, larger n,
        bootstrap CIs, probability-ranked ROC-AUC                            (R3.5a, fixes encoder/decoder non-comparability)
  exp3  LIME-SHAP agreement on 100-200 random instances, IoU across top-k    (R1.4, R2.5, R3.5b)
  exp4  Reuters-dateline intervention on RoBERTa (where the artefact was found) (R2.6)

Run on a GPU box (Colab A100 is fine):
  pip install transformers peft datasets accelerate scikit-learn lime shap pandas
  python reviewer_experiments.py exp1
Everything that depends on your notebook (paths, cleaning, label maps) is in CONFIG / load_split().
"""
import sys, re, json, random
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

CONFIG = {
    "seeds": [13, 42, 2024],            # 3 seeds minimum; 5 is better
    "max_len": 256,                     # use the SAME window for every model in exp2
    "eval_n_per_class": 500,            # exp2: 500+500 per corpus (or None for full split)
    "xai_n": 150,                       # exp3
    "topk": [5, 10, 15, 20],            # exp3
    "welfake_exclude_ids": "welfake_exact_plus_minhash_exclusions.json",  # from your decoder audit
    "out": "results/",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- data
def clean(text):
    """PASTE the exact cleaning function from your training notebook here."""
    text = re.sub(r"https?://\S+|<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()

def load_split(name, split="test"):
    """Return DataFrame[id, text, label] with Fake=0, Real=1. Wire up to your data files."""
    raise NotImplementedError(f"load {name}/{split} using the notebook's label mapping")

def fixed_sample(df, n_per_class, seed=42):
    if n_per_class is None:
        return df
    return (df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(min(n_per_class, len(g)), random_state=seed))
              .reset_index(drop=True))

def eval_sets():
    """One frozen set of IDs per corpus, shared by ALL models. Saved so it is auditable."""
    sets = {}
    excl = set(json.load(open(CONFIG["welfake_exclude_ids"])))
    for name in ["gonzaloa", "pulk17", "liar", "welfake"]:
        df = load_split(name)
        if name == "welfake":
            df = df[~df["id"].isin(excl)]
        s = fixed_sample(df, CONFIG["eval_n_per_class"])
        s[["id", "label"]].to_csv(f"{CONFIG['out']}sample_ids_{name}.csv", index=False)
        sets[name] = s
    return sets


# ----------------------------------------------------------------------------- metrics
def metrics_with_ci(y, p_real, n_boot=2000, seed=0):
    y, p_real = np.asarray(y), np.asarray(p_real)
    yhat = (p_real >= 0.5).astype(int)
    point = dict(acc=accuracy_score(y, yhat), f1=f1_score(y, yhat, average="macro"),
                 auc=roc_auc_score(y, p_real))
    rng = np.random.default_rng(seed)
    idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
    boots = {k: [] for k in point}
    for _ in range(n_boot):                                   # stratified bootstrap
        b = np.concatenate([rng.choice(idx0, len(idx0)), rng.choice(idx1, len(idx1))])
        yb, pb = y[b], p_real[b]
        boots["acc"].append(accuracy_score(yb, pb >= .5))
        boots["f1"].append(f1_score(yb, pb >= .5, average="macro"))
        boots["auc"].append(roc_auc_score(yb, pb))
    return {k: (point[k], *np.percentile(boots[k], [2.5, 97.5])) for k in point}


# ----------------------------------------------------------------------------- scoring
@torch.no_grad()
def encoder_p_real(model, tok, texts, bs=32):
    model.eval().to(DEVICE); out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i+bs], truncation=True, max_length=CONFIG["max_len"],
                  padding=True, return_tensors="pt").to(DEVICE)
        out.append(torch.softmax(model(**enc).logits, -1)[:, 1].cpu())
    return torch.cat(out).numpy()

@torch.no_grad()
def decoder_p_real(model, tok, texts, prompt_fn, bs=8):
    """Label-token logits -> 2-way softmax (same method as manuscript Sec. 3.6)."""
    fake_id = tok(" Fake", add_special_tokens=False).input_ids[0]
    real_id = tok(" Real", add_special_tokens=False).input_ids[0]
    tok.padding_side = "left"; model.eval(); out = []
    for i in range(0, len(texts), bs):
        enc = tok([prompt_fn(t) for t in texts[i:i+bs]], return_tensors="pt",
                  padding=True, truncation=True, max_length=CONFIG["max_len"]).to(model.device)
        last = model(**enc).logits[:, -1, [fake_id, real_id]]
        out.append(torch.softmax(last.float(), -1)[:, 1].cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- exp1
def train_encoder(seed, use_lora, base="roberta-base"):
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
    from datasets import Dataset
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2)
    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["query", "key", "value", "dense"]))   # mirrors decoder: attn + FFN
        model.print_trainable_parameters()                           # report this in the paper
    def ds(split):
        df = load_split("gonzaloa", split)
        d = Dataset.from_pandas(df[["text", "label"]])
        return d.map(lambda b: tok(b["text"], truncation=True, max_length=CONFIG["max_len"]), batched=True)
    args = TrainingArguments(
        output_dir=f"ckpt/{'lora' if use_lora else 'full'}_s{seed}", seed=seed,
        learning_rate=2e-4 if use_lora else 2e-5, num_train_epochs=5,
        per_device_train_batch_size=32, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1", warmup_ratio=0.06,
        lr_scheduler_type="linear", bf16=True, report_to=[])
    def cm(p): return {"f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}
    tr = Trainer(model=model, args=args, train_dataset=ds("train"), eval_dataset=ds("validation"),
                 data_collator=DataCollatorWithPadding(tok), compute_metrics=cm)
    from transformers import EarlyStoppingCallback
    tr.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
    tr.train()
    return tr.model, tok

def exp1():
    sets, rows = eval_sets(), []
    for use_lora in [False, True]:
        for seed in CONFIG["seeds"]:
            model, tok = train_encoder(seed, use_lora)
            for name, s in sets.items():
                m = metrics_with_ci(s["label"], encoder_p_real(model, tok, s["text"].tolist()))
                rows.append(dict(regime="LoRA" if use_lora else "Full", seed=seed, corpus=name,
                                 **{k: v[0] for k, v in m.items()}))
    df = pd.DataFrame(rows); df.to_csv(CONFIG["out"] + "exp1_raw.csv", index=False)
    summary = df.groupby(["regime", "corpus"])[["acc", "f1", "auc"]].agg(["mean", "std"])
    summary.to_csv(CONFIG["out"] + "exp1_summary.csv"); print(summary)
    # paired test across seeds (same seeds, same samples)
    from scipy.stats import ttest_rel
    for c in sets:
        a = df[(df.regime == "Full") & (df.corpus == c)].sort_values("seed").f1
        b = df[(df.regime == "LoRA") & (df.corpus == c)].sort_values("seed").f1
        print(c, "paired t-test Full vs LoRA F1:", ttest_rel(a, b))


# ----------------------------------------------------------------------------- exp2
def exp2(models):
    """models: dict name -> ("enc", model, tok) or ("dec", model, tok, prompt_fn). Load your saved checkpoints."""
    sets, rows = eval_sets(), []
    for mname, spec in models.items():
        for cname, s in sets.items():
            texts = s["text"].tolist()
            p = encoder_p_real(spec[1], spec[2], texts) if spec[0] == "enc" \
                else decoder_p_real(spec[1], spec[2], texts, spec[3])
            pd.DataFrame({"id": s["id"], "label": s["label"], "p_real": p}) \
              .to_csv(f"{CONFIG['out']}probs_{mname}_{cname}.csv", index=False)
            m = metrics_with_ci(s["label"], p)
            rows.append({"model": mname, "corpus": cname,
                         **{f"{k}": f"{v[0]:.3f} ({v[1]:.3f}-{v[2]:.3f})" for k, v in m.items()}})
    pd.DataFrame(rows).to_csv(CONFIG["out"] + "exp2_unified.csv", index=False)


# ----------------------------------------------------------------------------- exp3
def exp3(model, tok):
    from lime.lime_text import LimeTextExplainer
    import shap
    df = load_split("gonzaloa").sample(CONFIG["xai_n"], random_state=42)   # RANDOM, not curated
    predict = lambda xs: np.stack([1 - (p := encoder_p_real(model, tok, list(xs))), p], 1)
    lime_exp = LimeTextExplainer(class_names=["Fake", "Real"], split_expression=r"\W+")
    shap_exp = shap.Explainer(predict, shap.maskers.Text(r"\W+"))
    K = max(CONFIG["topk"]); rows = []
    for _, r in df.iterrows():
        pred = int(predict([r.text])[0].argmax())
        le = lime_exp.explain_instance(r.text, predict, num_features=K, labels=[pred], num_samples=500)
        lime_top = [w.lower() for w, _ in sorted(le.as_list(label=pred), key=lambda t: -abs(t[1]))]
        sv = shap_exp([r.text])
        toks, vals = sv.data[0], sv.values[0][:, pred]
        order = np.argsort(-np.abs(vals))
        shap_top = []
        for j in order:
            t = toks[j].strip().lower()
            if t and t not in shap_top: shap_top.append(t)
            if len(shap_top) >= K: break
        row = {"id": r.id, "label": r.label, "pred": pred}
        for k in CONFIG["topk"]:
            A, B = set(lime_top[:k]), set(shap_top[:k])
            row[f"iou@{k}"] = len(A & B) / max(1, len(A | B))
        rows.append(row)
    out = pd.DataFrame(rows); out.to_csv(CONFIG["out"] + "exp3_iou.csv", index=False)
    print(out[[f"iou@{k}" for k in CONFIG["topk"]]].describe())   # report mean, SD, median, IQR + a histogram


# ----------------------------------------------------------------------------- exp4
REUTERS = re.compile(r"\b[A-Z][A-Za-z .]{0,30}\(reuters\)\s*-\s*|\breuters\s*-\s*", re.I)

def exp4(model, tok, n=200):
    """Same design as the decoder intervention (manuscript Sec. 3.8), but on RoBERTa, with CIs."""
    df = load_split("gonzaloa")
    fake = df[df.label == 0].sample(n, random_state=42)
    real = df[(df.label == 1) & df.text.str.contains(REUTERS)].sample(n, random_state=42)
    res = {}
    for tag, sub, edit in [("inject_into_fake", fake, lambda t: "London (Reuters) - " + t),
                           ("remove_from_real", real, lambda t: REUTERS.sub("", t))]:
        p0 = encoder_p_real(model, tok, sub.text.tolist())
        p1 = encoder_p_real(model, tok, [edit(t) for t in sub.text])
        d, flips = p1 - p0, ((p0 >= .5) != (p1 >= .5))
        rng = np.random.default_rng(0)
        bd = [d[rng.integers(0, n, n)].mean() for _ in range(2000)]
        bf = [flips[rng.integers(0, n, n)].mean() for _ in range(2000)]
        res[tag] = dict(mean_delta=float(d.mean()), delta_ci=np.percentile(bd, [2.5, 97.5]).tolist(),
                        flip_rate=float(flips.mean()), flip_ci=np.percentile(bf, [2.5, 97.5]).tolist())
    json.dump(res, open(CONFIG["out"] + "exp4_roberta_reuters.json", "w"), indent=2); print(res)
    # Optional stronger version (what R2 literally asked for): fix REUTERS in the preprocessing,
    # retrain with train_encoder(seed, use_lora=False) on the cleaned data, and compare
    # in-domain + cross-dataset F1 against the original checkpoint.


if __name__ == "__main__":
    import os; os.makedirs(CONFIG["out"], exist_ok=True)
    {"exp1": exp1}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
