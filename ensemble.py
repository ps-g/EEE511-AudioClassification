"""Deep-ensemble evaluation for the ESC-50 reproduction of
"Rethinking CNN Models for Audio Classification" (Palanisamy et al., 2020).

Run with:

    python ensemble.py --manifest ensemble_manifest.json --out results/ensemble

For every fold it loads each member checkpoint, collects softmax outputs on the
validation fold (deterministic order), and reports:
  - each member's accuracy and the mean single-model accuracy
  - soft-voting ensemble accuracy (average of softmax outputs)
  - hard majority-vote accuracy
  - oracle accuracy (at least one member correct) - upper bound for any combiner
  - mean pairwise disagreement between members (diversity)
  - NLL and ECE for single models vs. the ensemble (calibration)
  - ensemble-size curve: accuracy vs. number of members, averaged over all subsets

Member predictions are cached in <out>/probs/, so adding new members later only
runs inference for the new checkpoints.
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dataloaders.datasetaug as dsa
import models.densenet
import models.resnet

MODELS = {"resnet": models.resnet.ResNet, "densenet": models.densenet.DenseNet}


# --------------------------------------------------------------------------- data
def val_loader(data_dir, fold, dataset_name, batch_size, num_workers):
    # shuffle=False is essential: member predictions must line up sample-for-sample.
    # (The repo's fetch_dataloader uses shuffle=True, which would silently misalign them.)
    tf = dsa.MelSpectrogram(128, "validation", dataset_name)
    ds = dsa.AudioDataset("{}validation128mel{}.pkl".format(data_dir, fold), dataset_name, transforms=tf)
    return DataLoader(ds, shuffle=False, batch_size=batch_size, num_workers=num_workers)


@torch.no_grad()
def member_probs(member, fold, loader, dataset_name, device, cache_dir):
    cache = os.path.join(cache_dir, "{}_fold{}.npz".format(member["name"], fold))
    if os.path.exists(cache):
        d = np.load(cache)
        return d["probs"], d["targets"]

    ckpt_path = member["ckpt"].format(fold=fold)
    model = MODELS[member["model"]](dataset_name, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    probs, targets = [], []
    for x, y in loader:
        probs.append(F.softmax(model(x.to(device)), dim=1).cpu().numpy())
        targets.append(y.squeeze(1).numpy())
    probs, targets = np.concatenate(probs), np.concatenate(targets)
    np.savez(cache, probs=probs, targets=targets)
    print("  cached {} fold {} ({} samples)".format(member["name"], fold, len(targets)))
    return probs, targets


# ------------------------------------------------------------------------ metrics
def accuracy(probs, y):
    return 100.0 * float((probs.argmax(1) == y).mean())


def nll(probs, y):
    return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())


def ece(probs, y, bins=15):
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return 100.0 * float(total)


def majority_vote(preds, num_classes, avg_probs):
    # Ties broken by the averaged softmax probability.
    counts = np.zeros((preds.shape[1], num_classes))
    for p in preds:
        counts[np.arange(preds.shape[1]), p] += 1
    counts += 1e-6 * avg_probs
    return counts.argmax(1)


def ensemble_metrics(P, y):
    """P: (M, N, C) softmax outputs of M members on N samples."""
    M, _, C = P.shape
    preds = P.argmax(2)
    avg = P.mean(0)
    pairs = list(itertools.combinations(range(M), 2))

    size_curve = {}
    for k in range(1, M + 1):
        size_curve[k] = float(np.mean([accuracy(P[list(c)].mean(0), y)
                                       for c in itertools.combinations(range(M), k)]))

    member_acc = [accuracy(p, y) for p in P]
    return {
        "member_acc": member_acc,
        "mean_single_acc": float(np.mean(member_acc)),
        "ensemble_acc": accuracy(avg, y),
        "vote_acc": 100.0 * float((majority_vote(preds, C, avg) == y).mean()),
        "oracle_acc": 100.0 * float((preds == y).any(0).mean()),
        "disagreement": 100.0 * float(np.mean([(preds[i] != preds[j]).mean() for i, j in pairs])) if pairs else 0.0,
        "single_nll": float(np.mean([nll(p, y) for p in P])),
        "ensemble_nll": nll(avg, y),
        "single_ece": float(np.mean([ece(p, y) for p in P])),
        "ensemble_ece": ece(avg, y),
        "size_curve": size_curve,
    }


SUMMARY_KEYS = ["mean_single_acc", "ensemble_acc", "vote_acc", "oracle_acc", "disagreement",
                "single_nll", "ensemble_nll", "single_ece", "ensemble_ece"]


def summarize(per_fold):
    out = {}
    for k in SUMMARY_KEYS:
        v = np.array([r[k] for r in per_fold.values()])
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    sizes = sorted(next(iter(per_fold.values()))["size_curve"])
    out["size_curve"] = {}
    for s in sizes:
        v = np.array([r["size_curve"][s] for r in per_fold.values()])
        out["size_curve"][s] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="ensemble_results")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    with open(args.manifest) as f:
        cfg = json.load(f)
    members = {m["name"]: m for m in cfg["members"]}
    ensembles = cfg.get("ensembles") or {"all": list(members)}
    cache_dir = os.path.join(args.out, "probs")
    os.makedirs(cache_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1) Collect predictions for every member on every fold.
    probs, targets = {}, {}
    for fold in cfg["folds"]:
        print("Fold {}".format(fold))
        loader = val_loader(cfg["data_dir"], fold, cfg["dataset_name"], args.batch_size, args.num_workers)
        for name, m in members.items():
            p, t = member_probs(m, fold, loader, cfg["dataset_name"], device, cache_dir)
            if fold in targets:
                assert np.array_equal(targets[fold], t), \
                    "Target order mismatch for {} on fold {}".format(name, fold)
            targets[fold] = t
            probs[(name, fold)] = p

    # 2) Evaluate each requested ensemble.
    results = {}
    for ens_name, member_names in ensembles.items():
        per_fold = {}
        for fold in cfg["folds"]:
            P = np.stack([probs[(n, fold)] for n in member_names])
            per_fold[fold] = ensemble_metrics(P, targets[fold])
        results[ens_name] = {"members": member_names, "per_fold": per_fold, "summary": summarize(per_fold)}

        s = results[ens_name]["summary"]
        print("\n=== Ensemble '{}' ({} members: {}) ===".format(ens_name, len(member_names), ", ".join(member_names)))
        print("{:>5} | {:>18} | {:>8} | {:>8} | {:>8} | {:>8}".format(
            "Fold", "members acc", "single", "ensemble", "vote", "oracle"))
        for fold, r in per_fold.items():
            print("{:>5} | {:>18} | {:8.2f} | {:8.2f} | {:8.2f} | {:8.2f}".format(
                fold, " ".join("{:.1f}".format(a) for a in r["member_acc"]),
                r["mean_single_acc"], r["ensemble_acc"], r["vote_acc"], r["oracle_acc"]))
        fmt = lambda k: "{:.2f} ± {:.2f}".format(s[k]["mean"], s[k]["std"])
        print("Mean single acc : {}".format(fmt("mean_single_acc")))
        print("Ensemble acc    : {}   (gain {:+.2f} pp)".format(
            fmt("ensemble_acc"), s["ensemble_acc"]["mean"] - s["mean_single_acc"]["mean"]))
        print("Majority vote   : {}".format(fmt("vote_acc")))
        print("Oracle acc      : {}".format(fmt("oracle_acc")))
        print("Disagreement %  : {}".format(fmt("disagreement")))
        print("NLL single/ens  : {:.3f} / {:.3f}".format(s["single_nll"]["mean"], s["ensemble_nll"]["mean"]))
        print("ECE single/ens  : {:.2f} / {:.2f}".format(s["single_ece"]["mean"], s["ensemble_ece"]["mean"]))

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 3) Ensemble-size curve plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for ens_name, r in results.items():
            curve = r["summary"]["size_curve"]
            ks = sorted(curve)
            mu = np.array([curve[k]["mean"] for k in ks])
            sd = np.array([curve[k]["std"] for k in ks])
            ax.plot(ks, mu, marker="o", label=ens_name)
            ax.fill_between(ks, mu - sd, mu + sd, alpha=0.2)
        ax.set_xlabel("Number of ensemble members")
        ax.set_ylabel("Validation accuracy (%)")
        ax.set_title("Ensemble size vs. accuracy (mean ± SD over folds)")
        ax.set_xticks(range(1, max(len(r["members"]) for r in results.values()) + 1))
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "ensemble_size_curve.png"), dpi=200)
        print("\nSaved {}".format(os.path.join(args.out, "ensemble_size_curve.png")))
    except ImportError:
        print("matplotlib not installed; skipping plot")


if __name__ == "__main__":
    main()
