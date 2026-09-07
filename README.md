<div align="center">

# 📱 Mobile Spyware Detection

**Does looking at DNS traffic actually help you catch mobile spyware, or is flow-level analysis enough?**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Data: real + disclosed synthetic](https://img.shields.io/badge/data-real%20%2B%20disclosed%20synthetic-orange.svg)](#a-note-on-data-integrity)

</div>

---

This project builds a two-tier detection pipeline for mobile spyware — one layer using only network flow metadata (packet sizes, timing, TCP flags), a second layer adding DNS query analysis on top — and tests both against three architecturally different threat patterns:

| Pattern | C2 behavior | Expected DNS-layer value |
|---|---|---|
| `pegasus_like` | Dynamic DNS infrastructure, high-entropy subdomains | High |
| `graphite_like` | Direct-to-IP C2, no DNS queries at all | None by construction |
| `stalkerware_like` | Static, low-effort commercial infrastructure | Moderate |

These three patterns aren't arbitrary — they mirror real, publicly documented differences in how NSO Group's Pegasus and Paragon's Graphite spyware talk to their command-and-control infrastructure, and how commercial stalkerware (mSpy, FlexiSpy, Hoverwatch) behaves compared to both. Citizen Lab has documented that Graphite deliberately avoids DNS specifically to evade the kind of monitoring this project implements — which makes it a genuinely interesting edge case for a DNS-based detector to fail gracefully on.

## 📌 What's actually in here

- **Two detection pipelines**: L1 (flow + TLS metadata only) and L2 (L1 + DNS features — lexical entropy, beaconing patterns, domain age, CA reputation), compared via both feature fusion and confidence-weighted score fusion.
- **A non-ML baseline**: a rule-based IoC/signature engine (domain blocklists + Suricata-style heuristics) to compare against, because "we beat a baseline" only means something if the baseline is real.
- **Real threat intelligence**, not made-up indicators: Pegasus domains pulled from [Amnesty International's public investigations repo](https://github.com/AmnestyTech/investigations), and commercial stalkerware C2 domains/IPs from [ECHAP's stalkerware-indicators](https://github.com/AssoEchap/stalkerware-indicators) (CC-BY).
- **Real-world validation** on a subset of [CIC-AndMal2017](https://www.unb.ca/cic/datasets/andmal2017.html) (104 real Adware samples across 10 families + 202 benign, CICFlowMeter-processed), evaluated with nested cross-validation, Leave-One-Family-Out generalization testing, and feature-importance analysis.
- **A synthetic traffic simulator** for everything that can't ethically or practically be captured for real (no physical testbed, no live spyware execution): it crafts genuine TCP packets with scapy and produces Unbound/Zeek-style DNS and TLS logs, so the rest of the pipeline runs on realistic-shaped data without pretending it's a real capture.

## 📂 A note on data integrity

Every result in this repo is labeled by what it actually is: computed from real captured traffic, or from a disclosed simulation. Nothing here is presented as real when it isn't — that distinction is maintained on purpose, and checked, not just claimed.

> **Case in point:** an early run on the real dataset produced a suspiciously perfect result — one classifier at 100% accuracy on *every single fold*. Instead of reporting it, the pipeline was audited: a Leave-One-Family-Out evaluation ruled out family-level leakage, and feature-importance inspection turned up a single feature (\`min_packet_length\`) that separated the two classes with **zero overlap** (benign ≤ 300, malicious ≥ 315) — a strong signature of a capture-environment artifact rather than real malicious behavior. That feature is now excluded by default (\`KNOWN_ARTIFACT_FEATURE_PREFIXES\` in \`src/pipeline/real_dataset.py\`), and the corrected numbers below — still strong, no longer suspicious — are what's reported.

## 🎯 Results (real data: CIC-AndMal2017, Adware + Benign)

Nested 5-fold cross-validation, hyperparameter search confined to each fold's training data, artifact feature excluded:

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| `Random Forest` | 0.990 ± 0.013 | 0.990 ± 0.019 | 0.980 ± 0.024 | 0.985 ± 0.019 | 0.999 ± 0.001 |
| `XGBoost` | 0.990 ± 0.013 | 0.990 ± 0.020 | 0.980 ± 0.024 | 0.985 ± 0.020 | 0.993 ± 0.015 |
| `Logistic Regression` | 0.993 ± 0.008 | 1.000 ± 0.000 | 0.981 ± 0.023 | 0.990 ± 0.012 | 0.981 ± 0.023 |

**Generalization to unseen malware families** (Leave-One-Family-Out: train on 9 families, test on the 10th, never seen in training): mean recall 0.961 (Random Forest) to 0.982 (XGBoost).

**Detection latency** (recall as a function of observation window, Random Forest, held-out test set):

| Window | 5 min | 10 min | 15 min | 20 min | 30 min |
|---|---|---|---|---|---|
| Recall | 0.13 | 0.39 | 0.55 | 0.58 | 0.94 |

Reaches 80% recall at 30 minutes.

> This dataset has no DNS data, so it only validates the L1 (flow-only) pipeline. The L1/L2 comparison — the actual point of this project — runs on the synthetic simulator and is disclosed as such; see \`scripts/run_full_pipeline.py\` and the caveat above.

## 📊 How it works

```
config/         Reference config for a real capture node (hostapd, Unbound, iptables, SpyGuard)
data/
  real_iocs/      Real threat-intel indicators (Amnesty, ECHAP) — see attribution below
src/
  simulate/       Synthetic traffic generator (scapy pcaps + Unbound/Zeek-style DNS/TLS logs)
  features/       Flow features (CICFlowMeter-equivalent, incl. bulk-rate/exfiltration detection),
                  DNS features (entropy, beaconing, domain age, reputation), TLS/cert features
  baseline/       Rule-based IoC/signature detection engine (non-ML baseline)
  pipeline/       Dataset assembly, train/val/test split, L1 pipeline, L2 pipeline (feature + score fusion),
                  real CIC-AndMal2017 loader
  models/         Random Forest / XGBoost / Logistic Regression + score fusion
  evaluation/     Metrics, detection latency, per-family breakdown, nested CV, Leave-One-Family-Out,
                  feature importance
scripts/        Entry points (see below)
```

## 🚀 Quickstart

Requires Python 3.10+.

```bash
pip install -r requirements.txt

# End-to-end on synthetic data: baseline, L1, L2 (feature + score fusion), detection latency
python scripts/run_full_pipeline.py --n-benign 40 --n-per-family 15
```

To reproduce the real-data results, download [CIC-AndMal2017 via Kaggle](https://www.kaggle.com/datasets/tinlmnguyn/cicandmal2017) (~266 MB; requires a free Kaggle API token):

```bash
pip install kaggle
# generate a token at kaggle.com/settings > API > Create New Token,
# save the downloaded file as ~/.kaggle/kaggle.json
kaggle datasets download -d tinlmnguyn/cicandmal2017 -p data/cic_andmal2017 --unzip

python scripts/run_real_dataset_pipeline.py            # main metrics (nested 5-fold CV)
python scripts/run_real_dataset_diagnostics.py          # Leave-One-Family-Out + feature importance
python scripts/run_real_dataset_detection_latency.py    # recall vs. observation window
```

Each script writes a JSON report to `results/report/`.

## 🧠 Real threat-intel sources

| Source | Repository | Used for | License |
|---|---|---|---|
| Amnesty International Security Lab, Pegasus Project | [AmnestyTech/investigations](https://github.com/AmnestyTech/investigations) | Root domains for `pegasus_like` synthetic sessions | Published for defensive detection use |
| ECHAP / Coalition Against Stalkerware | [AssoEchap/stalkerware-indicators](https://github.com/AssoEchap/stalkerware-indicators) | Real mSpy/FlexiSpy/Hoverwatch domains & IPs, used both as C2 targets in synthetic sessions and as the baseline engine's known-indicator list | CC-BY 4.0 |
| Canadian Institute for Cybersecurity (UNB), CIC-AndMal2017 | [Kaggle mirror](https://www.kaggle.com/datasets/tinlmnguyn/cicandmal2017) | Real-world Adware/Benign validation set | Academic dataset, UNB/CIC |

**Design note:** Pegasus domains are deliberately **not** added to the rule-based baseline's known-indicator list, even though they're real. NSO's C2 infrastructure rotates constantly — a static list of historical indicators wouldn't be available at the time of a new, undocumented infection. That's the exact limitation the baseline is meant to expose.

## ❗ Known limitations

- The real dataset only covers Adware (10 families) vs. Benign — not the full spyware/stalkerware spectrum, and not the Ransomware/Scareware/SMS-malware categories that exist in the full CIC-AndMal2017 release but aren't in this mirror.
- It also has no DNS data, so the L1/L2 (RQ2-style) comparison is necessarily synthetic.
- The synthetic simulator is disclosed, structurally realistic traffic — not a substitute for real spyware capture. It's useful for exercising the architecture end-to-end, not as evidence about real-world detectability.
- No physical capture testbed (Raspberry Pi + SpyGuard + Unbound) was available; `config/` contains reference configuration only.

## 📋 License

MIT — see [LICENSE](LICENSE).

## 👥 Contributor

This project was created by 👩 `Agresta Arianna`. [Click here](https://github.com/Arianna6400) to see my other projects!

Feel free to contribute by submitting pull requests or reporting issues! 🚀
