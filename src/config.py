"""Shared project constants (paths, threat families, parameters)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SESSIONS_DIR = RESULTS_DIR / "sessions"
DATASET_DIR = RESULTS_DIR / "datasets"
REPORT_DIR = RESULTS_DIR / "report"

# Simulated threat families, modeled on documented differences between
# real spyware C2 architectures:
#   - pegasus_like: dynamic DNS infrastructure, high-entropy subdomains,
#     self-signed/free-CA wildcard TLS certs (pre-2021 Pegasus pattern).
#   - graphite_like: direct-to-IP C2, no DNS queries at all (Paragon/
#     Graphite pattern — deliberate evasion of DNS-based monitoring).
#   - stalkerware_like: low-effort commercial spyware, static and
#     unsophisticated DNS infrastructure (analogous to CIC-AndMal2017).
MALICIOUS_FAMILIES = ["pegasus_like", "graphite_like", "stalkerware_like"]
ALL_LABELS = ["benign"] + MALICIOUS_FAMILIES

SESSION_DURATION_S = 1800  # 30 minutes, a common reference window for capture sessions
DETECTION_WINDOWS_MIN = [5, 10, 15, 20, 30]
RECALL_THRESHOLD_DETECTION = 0.80  # rho* used for the detection-latency metric

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

RANDOM_SEED = 42
