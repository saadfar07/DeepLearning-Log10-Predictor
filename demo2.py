# # import math
# # import pandas as pd
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim

# # # =============================
# # # Load Excel Log Table
# # # =============================
# # excel_path = r"C:\Users\Saad Farooqui\Downloads\logs.xlsx"
# # df = pd.read_excel(excel_path)

# # # assume first column = x , second column = log(x)
# # x_vals = df.iloc[:, 0].values
# # true_logs = df.iloc[:, 1].values


# # # =============================
# # # Taylor Series for ln(x)
# # # ln(x) = 2 * [ y + y^3/3 + y^5/5 + ... ]
# # # where y = (x-1)/(x+1)
# # # =============================
# # def taylor_ln(x, terms=7):
# #     y = (x - 1) / (x + 1)
# #     result = 0.0
# #     for n in range(terms):
# #         power = 2 * n + 1
# #         result += (y ** power) / power
# #     return 2 * result


# # # =============================
# # # Simple Neural Network
# # # =============================
# # class LogNN(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(1, 16),
# #             nn.ReLU(),
# #             nn.Linear(16, 16),
# #             nn.ReLU(),
# #             nn.Linear(16, 1)
# #         )

# #     def forward(self, x):
# #         return self.net(x)


# # # =============================
# # # Train NN (FIXED)
# # # =============================
# # def train_nn(x_vals, y_vals, epochs=4000, lr=0.01):
# #     model = LogNN()
# #     opt = torch.optim.Adam(model.parameters(), lr=lr)
# #     loss_fn = nn.MSELoss()

# #     # NORMALIZE INPUT
# #     x_train = torch.tensor(x_vals - 1.0, dtype=torch.float32).view(-1, 1)

# #     # TARGET = log10(x) (EXACTLY EXCEL)
# #     y_train = torch.tensor(y_vals, dtype=torch.float32).view(-1, 1)

# #     for _ in range(epochs):
# #         opt.zero_grad()
# #         out = model(x_train)
# #         loss = loss_fn(out, y_train)
# #         loss.backward()
# #         opt.step()

# #     return model

# #     print("❌ NN precision FAILED -> stop")
# #     return None


# # # =============================
# # # Run Training
# # # =============================
# # print("\n--- Searching minimum NN ---")
# # best_model = train_nn(x_vals, true_logs)

# # print("\n--- Searching minimum Taylor terms ---")
# # min_terms = 7
# # print("Minimum Taylor terms:", min_terms)

# # # =============================
# # # Comparison
# # # =============================
# # print("\n--- Comparison ---")
# # print("x     | NN        | Taylor     | True      | NN_err   | Tay_err")
# # print("-" * 65)

# # for x, true_val in zip(x_vals[:10], true_logs[:10]):
# #     # Taylor
# #     tay_val = taylor_ln(x, min_terms)
# #     tay_err = abs(tay_val - true_val)

# #     # NN (safe check)
# #     if best_model is not None:
# #         inp = torch.tensor([[x]], dtype=torch.float32)
# #         nn_val = best_model(inp).item()
# #         nn_err = abs(nn_val - true_val)
# #     else:
# #         nn_val = "N/A"
# #         nn_err = "N/A"

# #     print(f"{x:.3f} | {nn_val} | {tay_val:.6f} | {true_val:.6f} | {nn_err} | {tay_err:.6e}")

# import math
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim


# # ============================================================
# # USER CONFIG
# # ============================================================
# EXCEL_PATH = r"C:\Users\Saad Farooqui\Downloads\logs.xlsx"

# # Best practical default precision (you can raise to 6 if you want)
# DECIMALS = 4

# # NN search list: big -> smaller (gradual)
# CANDIDATE_ARCHS = [
#     (3, 64), (3, 32), (3, 16),
#     (2, 64), (2, 32), (2, 16), (2, 8),
#     (1, 64), (1, 32), (1, 16), (1, 8),
# ]

# # Taylor series terms (start high, reduce)
# MAX_TAYLOR_TERMS = 400

# # Training stability
# MAX_EPOCHS = 50000
# BATCH_SIZE = 256
# LR = 5e-4
# PATIENCE = 60

# DEVICE = "cpu"
# DTYPE = torch.float64

# # Reproducibility
# torch.manual_seed(42)
# np.random.seed(42)


# # ============================================================
# # 1) Load log table from Excel
# # ============================================================
# def load_excel_table(path: str):
#     df = pd.read_excel(path)

#     if df.shape[1] < 2:
#         raise ValueError("Excel must have at least 2 columns.")

#     pairs = []

#     # Read every row, take values, extract (x, y) as sequential pairs across ALL columns
#     for _, row in df.iterrows():
#         vals = []
#         for v in row.values:
#             try:
#                 if pd.isna(v):
#                     continue
#                 vals.append(float(v))
#             except Exception:
#                 continue

#         # Take sequential pairs (x,y)
#         for i in range(0, len(vals) - 1, 2):
#             x = vals[i]
#             y = vals[i + 1]
#             if x > 0 and np.isfinite(x) and np.isfinite(y):
#                 pairs.append((x, y))

#     if len(pairs) == 0:
#         raise ValueError("No numeric (x, log) pairs found in Excel.")

#     # De-duplicate by x (keep last), then sort
#     d = {}
#     for x, y in pairs:
#         d[float(x)] = float(y)

#     xs = np.array(sorted(d.keys()), dtype=float)
#     ys = np.array([d[x] for x in xs], dtype=float)

#     # Keep only [1,10) for training
#     mask = (xs >= 1.0) & (xs < 10.0)
#     xs, ys = xs[mask], ys[mask]

#     if len(xs) < 50:
#         raise ValueError(
#             f"Not enough training points in [1,10). Found {len(xs)} only. "
#             "Your Excel might not contain full 1..10 table."
#         )

#     # IMPORTANT: print range so you can confirm it’s correct
#     print(f"Training points: {len(xs)} (range: {xs.min():.3f} to {xs.max():.3f})")

#     return xs, ys


# # ============================================================
# # 2) Range reduction without log:
# # x = m * 10^k, m in [1,10)
# # log10(x) = log10(m) + k
# # ============================================================
# def normalize_to_1_10_scalar(x: float):
#     if x <= 0:
#         raise ValueError("x must be > 0")
#     m = float(x)
#     k = 0
#     # bring down
#     while m >= 10.0:
#         m /= 10.0
#         k += 1
#         if k > 4000:  # safety
#             break
#     # bring up
#     while m < 1.0:
#         m *= 10.0
#         k -= 1
#         if k < -4000:  # safety
#             break
#     return m, k

# def m_to_unit(m: float):
#     # map [1,10) -> [0,1]
#     return (m - 1.0) / 9.0


# # ============================================================
# # 3) Neural Network (MLP)
# # Learns f(u)=log10(m) where u=(m-1)/9 in [0,1]
# # ============================================================
# class LogNN(nn.Module):
#     def __init__(self, layers: int, hidden: int):
#         super().__init__()
#         modules = []
#         in_dim = 1
#         for _ in range(layers):
#             modules.append(nn.Linear(in_dim, hidden))
#             modules.append(nn.Tanh())
#             in_dim = hidden
#         modules.append(nn.Linear(in_dim, 1))
#         self.net = nn.Sequential(*modules)

#     def forward(self, u):
#         return self.net(u)

# def count_params(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def train_model(x_train, y_train, layers, hidden):
#     model = LogNN(layers=layers, hidden=hidden).to(DEVICE).to(DTYPE)

#     # u in [0,1]
#     u = torch.tensor(((x_train - 1.0) / 9.0), dtype=DTYPE).view(-1, 1).to(DEVICE)
#     y = torch.tensor(y_train, dtype=DTYPE).view(-1, 1).to(DEVICE)

#     mse = nn.MSELoss()

#     # -------------------------
#     # Stage 1: Adam (coarse fit)
#     # -------------------------
#     opt = optim.Adam(model.parameters(), lr=1e-3)
#     for ep in range(1, 15001):  # 15k epochs
#         opt.zero_grad()
#         pred = model(u)
#         loss = mse(pred, y)
#         loss.backward()
#         opt.step()

#         if ep == 1 or ep % 3000 == 0:
#             print(f"  [Adam]  epoch {ep:5d} | loss={loss.item():.12e}")

#     # --------------------------------
#     # Stage 2: LBFGS (precision refine)
#     # --------------------------------
#     opt2 = optim.LBFGS(
#         model.parameters(),
#         lr=1.0,
#         max_iter=800,          # strong refinement
#         history_size=50,
#         line_search_fn="strong_wolfe"
#     )

#     def closure():
#         opt2.zero_grad()
#         pred = model(u)
#         loss = mse(pred, y)
#         loss.backward()
#         return loss

#     loss2 = opt2.step(closure)
#     final_loss = mse(model(u), y).item()
#     print(f"  [LBFGS] final loss={final_loss:.12e}")

#     return model


# @torch.no_grad()
# def nn_predict_log10(model, x: float) -> float:
#     m, k = normalize_to_1_10_scalar(x)
#     u = m_to_unit(m)
#     inp = torch.tensor([[u]], dtype=DTYPE).to(DEVICE)
#     return float(model(inp).item() + k)


# # ============================================================
# # 4) Taylor Series for log10(x)
# # Using:
# # ln(m) = 2 * [ y + y^3/3 + y^5/5 + ... ], y=(m-1)/(m+1)
# # log10(x) = ln(m)/ln(10) + k
# # ============================================================
# def taylor_log10(x: float, terms: int) -> float:
#     m, k = normalize_to_1_10_scalar(x)
#     y = (m - 1.0) / (m + 1.0)
#     y2 = y * y

#     s = y
#     cur = y
#     for n in range(1, terms):
#         cur = cur * y2
#         s = s + (cur / (2 * n + 1))

#     ln_m = 2.0 * s
#     return (ln_m / math.log(10.0)) + k


# # ============================================================
# # 5) Precision match check
# # ============================================================
# def match_precision(a: float, b: float, decimals: int) -> bool:
#     return round(a, decimals) == round(b, decimals)


# # ============================================================
# # 6) Evaluation set (hard points + wide range)
# # ============================================================
# def build_eval_set():
#     xs = []

#     # Hard mantissa edges near 1 and near 10 across magnitudes
#     hard_m = [1.00001, 1.0001, 1.001, 1.01, 1.1, 3.27, 9.9, 9.99, 9.9999]

#     for p in range(-8, 9):  # 1e-8 to 1e8
#         for m in hard_m:
#             xs.append(m * (10 ** p))

#     # Random log-uniform
#     rng = np.random.default_rng(123)
#     for _ in range(400):
#         exp = rng.uniform(-8, 8)
#         mant = rng.uniform(1.0, 10.0)
#         xs.append(mant * (10 ** exp))

#     # Include some typical test values
#     xs += [0.0327, 0.1, 0.5, 1.23, 7.89, 9.99, 10.0, 327.0, 5830.0, 1e-6, 1e6]

#     # Ensure strictly >0
#     xs = [x for x in xs if x > 0]
#     return xs


# # ============================================================
# # 7) Find minimum NN architecture (big -> small)
# # Reference = math.log10 (allowed)
# # ============================================================
# def find_min_nn(x_train, y_train, eval_xs):
#     print("\n=== NN ARCH SEARCH (big -> small) ===")
#     last_good = None

#     for layers, hidden in CANDIDATE_ARCHS:
#         print(f"\nTraining arch layers={layers}, hidden={hidden}")
#         model = train_model(x_train, y_train, layers, hidden)
#         params = count_params(model)

#         ok = True
#         worst_err = 0.0
#         worst_x = None

#         for x in eval_xs:
#             pred = nn_predict_log10(model, x)
#             ref = math.log10(x)  # ground truth check
#             if not match_precision(pred, ref, DECIMALS):
#                 ok = False
#                 worst_err = abs(pred - ref)
#                 worst_x = x
#                 break

#         print(f"Arch layers={layers}, hidden={hidden} | params={params} | PASS={ok}")

#         if ok:
#             last_good = (layers, hidden, model, params)
#         else:
#             if last_good is not None:
#                 print(f"-> Divergence at x={worst_x} | abs_err={worst_err:.3e}")
#                 print("-> Reverting to last PASS configuration.")
#                 return last_good

#     return last_good


# # ============================================================
# # 8) Find minimum Taylor terms (high -> low)
# # Reference = math.log10
# # ============================================================
# def find_min_taylor(eval_xs):
#     print("\n=== TAYLOR TERM SEARCH (high -> low) ===")
#     last_good = None

#     for terms in range(MAX_TAYLOR_TERMS, 0, -1):
#         ok = True
#         for x in eval_xs:
#             tv = taylor_log10(x, terms=terms)
#             ref = math.log10(x)
#             if not match_precision(tv, ref, DECIMALS):
#                 ok = False
#                 break

#         if terms % 25 == 0 or terms == MAX_TAYLOR_TERMS or terms == 1:
#             print(f"Terms={terms:3d} | PASS={ok}")

#         if ok:
#             last_good = terms
#         else:
#             if last_good is not None:
#                 print("-> Divergence detected. Reverting to last PASS terms.")
#                 return last_good

#     return last_good


# # ============================================================
# # 9) Operation counts (multiplications + additions) per inference
# # Notes: ignores activation cost & divisions (as requested)
# # ============================================================
# def nn_ops_count(layers, hidden):
#     dims = [1] + [hidden] * layers + [1]
#     mults = 0
#     adds = 0
#     for i in range(len(dims) - 1):
#         in_d = dims[i]
#         out_d = dims[i + 1]
#         mults += in_d * out_d
#         adds += out_d * (in_d - 1)  # sums of products
#         adds += out_d              # bias adds
#     return mults, adds

# def taylor_ops_count(terms):
#     # y2 = y*y : 1 mult
#     mults = 1
#     # m-1, m+1 : 2 adds
#     adds = 2

#     loops = max(0, terms - 1)
#     # per loop: cur *= y2 (1 mult), s += term (1 add)
#     mults += loops * 1
#     adds += loops * 1

#     # ln_m = 2*s : 1 mult
#     mults += 1
#     return mults, adds


# # ============================================================
# # MAIN
# # ============================================================
# def main():
#     print("Loading Excel training table...")
#     x_train, y_train = load_excel_table(EXCEL_PATH)
#     print(f"Training points: {len(x_train)} (range: {x_train.min():.3f} to {x_train.max():.3f})")

#     eval_xs = build_eval_set()
#     print(f"Evaluation points: {len(eval_xs)}")
#     print(f"Precision target: {DECIMALS} decimals\n")

#     # 1) Minimum NN
#     best = find_min_nn(x_train, y_train, eval_xs)
#     if best is None:
#         print("\n❌ No NN configuration met the required precision.")
#         print("Try: (a) increase MAX_EPOCHS, (b) increase candidate arch sizes, or (c) reduce DECIMALS by 1.")
#         return

#     layers, hidden, model, params = best
#     print(f"\n✅ FINAL NN: layers={layers}, hidden={hidden}, params={params}")

#     # 2) Minimum Taylor terms
#     best_terms = find_min_taylor(eval_xs)
#     if best_terms is None:
#         print("\n❌ No Taylor term count met the required precision within MAX_TAYLOR_TERMS.")
#         print("Increase MAX_TAYLOR_TERMS (e.g., 400) or reduce DECIMALS.")
#         return

#     print(f"\n✅ FINAL TAYLOR: terms={best_terms}")

#     # 3) Operation counts
#     nn_m, nn_a = nn_ops_count(layers, hidden)
#     t_m, t_a = taylor_ops_count(best_terms)

#     print("\n=== OPERATION COUNT (per inference; mult/add only) ===")
#     print(f"NN:     multiplications={nn_m}, additions={nn_a}, total={nn_m+nn_a}")
#     print(f"Taylor: multiplications={t_m}, additions={t_a}, total={t_m+t_a}")

#     # 4) Sample check
#     print("\n=== SAMPLE CHECKS ===")
#     samples = [0.0327, 0.1, 0.5, 1.0, 1.23, 3.27, 9.99, 10.0, 327.0, 5830.0, 1e-6, 1e6]
#     for x in samples:
#         nnv = nn_predict_log10(model, x)
#         tv = taylor_log10(x, best_terms)
#         ref = math.log10(x)
#         ok_nn = match_precision(nnv, ref, DECIMALS)
#         ok_t = match_precision(tv, ref, DECIMALS)
#         print(f"x={x:<10g} NN={nnv:+.{DECIMALS+2}f}  Taylor={tv:+.{DECIMALS+2}f}  true={ref:+.{DECIMALS+2}f}  "
#               f"NN_OK={ok_nn}  Tay_OK={ok_t}")


# if __name__ == "__main__":
#     main()


# final_log10_project.py
# Requirements: pip install torch pandas openpyxl

import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =========================
# USER SETTINGS
# =========================
EXCEL_PATH = r"F:\teamx\website backup\log10 predictor\logs.xlsx"  # <-- change if needed
DEVICE = "cpu"

# Precision goal: code will try highest decimals first, then relax if impossible
TRY_DECIMALS = [3,2]

# NN search space (big -> small)
CANDIDATES = [
    (3, 128),
    (3, 64),
    (3, 32),
    (2, 64),
    (2, 32),
    (2, 16),
    (1, 64),
    (1, 32),
    (1, 16),
    (1, 8),
    (1, 4),
]

MAX_EPOCHS_ADAM = 12000
LR = 1e-3
USE_LBFGS = True

# Taylor search
MAX_TAYLOR_TERMS = 30  # we'll reduce terms to minimum that passes


# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
# Load Excel (x, log10(x))
# =========================
def load_table(excel_path):
    df = pd.read_excel(excel_path)
    x = df.iloc[:, 0].astype(float).to_numpy()
    y = df.iloc[:, 1].astype(float).to_numpy()
    # Filter only positive
    mask = x > 0
    x, y = x[mask], y[mask]
    return x, y

def load_table(excel_path):
    df = pd.read_excel(excel_path)

    pairs = []
    for _, row in df.iterrows():
        vals = []
        for v in row.values:
            try:
                if pd.isna(v):
                    continue
                vals.append(float(v))
            except Exception:
                continue

        # sequential (x,y) pairs
        for i in range(0, len(vals) - 1, 2):
            x = vals[i]
            y = vals[i + 1]
            if x > 0 and np.isfinite(x) and np.isfinite(y):
                pairs.append((x, y))

    if not pairs:
        raise ValueError("No numeric (x,log) pairs found in Excel.")

    # deduplicate
    d = {}
    for x, y in pairs:
        d[float(x)] = float(y)

    xs = np.array(sorted(d.keys()), dtype=float)
    ys = np.array([d[x] for x in xs], dtype=float)

    # keep [1,10)
    mask = (xs >= 1.0) & (xs < 10.0)
    xs, ys = xs[mask], ys[mask]

    if len(xs) < 200:
        raise ValueError(f"Not enough training points in [1,10). Found {len(xs)} only.")

    return xs, ys

# =========================
# Range reduction: x = m * 10^k , m in [1,10)
# log10(x) = k + log10(m)
# =========================
def split_mantissa_exponent_base10(x: float):
    if x <= 0:
        raise ValueError("x must be > 0")
    k = math.floor(math.log10(x))
    m = x / (10 ** k)
    # fix floating rounding drift
    if m < 1.0:
        k -= 1
        m *= 10.0
    elif m >= 10.0:
        k += 1
        m /= 10.0
    return m, k


def vector_split_mk(xs):
    ms = np.zeros_like(xs, dtype=np.float64)
    ks = np.zeros_like(xs, dtype=np.int64)
    for i, v in enumerate(xs):
        m, k = split_mantissa_exponent_base10(float(v))
        ms[i] = m
        ks[i] = k
    return ms, ks


# =========================
# Taylor series for ln(m) using:
# ln(m) = 2 * [ y + y^3/3 + y^5/5 + ... ]
# y = (m-1)/(m+1)
# Then log10(m) = ln(m)/ln(10)
# =========================
LN10 = math.log(10.0)

def taylor_log10_m(m: float, terms: int):
    # m should be in [1,10)
    y = (m - 1.0) / (m + 1.0)
    y2 = y * y
    # accumulate y^(2n+1) iteratively to avoid pow()
    p = y  # y^(1)
    s = 0.0
    for n in range(terms):
        denom = 2 * n + 1
        s += p / denom
        p *= y2  # next odd power
    ln_m = 2.0 * s
    return ln_m / LN10


def taylor_log10_x(x: float, terms: int):
    m, k = split_mantissa_exponent_base10(x)
    return k + taylor_log10_m(m, terms)


# =========================
# Neural Network (tiny MLP)
# We train on mantissa m in [1,10)
# Input normalization: scale to [-1,1]
# z = (m - 5.5) / 4.5
# =========================
def norm_m(m):
    return (m - 5.5) / 4.5

def denorm_m(z):
    return z * 4.5 + 5.5


class LogNN(nn.Module):
    def __init__(self, layers: int, hidden: int):
        super().__init__()
        blocks = []
        in_dim = 1
        for _ in range(layers):
            blocks.append(nn.Linear(in_dim, hidden))
            blocks.append(nn.Tanh())  # smoother than ReLU for function approx
            in_dim = hidden
        blocks.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Operation count for MLP:
# For Linear(in,out):
# multiplications = in*out
# additions ~ out*(in-1) + out(bias) = out*in
def count_ops_mlp(layers: int, hidden: int):
    mult = 0
    add = 0
    in_dim = 1
    for _ in range(layers):
        out_dim = hidden
        mult += in_dim * out_dim
        add += in_dim * out_dim
        in_dim = out_dim
    # last layer to 1
    out_dim = 1
    mult += in_dim * out_dim
    add += in_dim * out_dim
    return mult, add


# Operation count for Taylor (per x):
# y = (m-1)/(m+1): ~2 adds, 1 div
# We'll count only + and * as asked, ignore division cost (you can add if needed)
# y2 = y*y: 1 mul
# loop each term: s += p/denom (add), p *= y2 (mul)
def count_ops_taylor(terms: int):
    mult = 1  # y2
    add = 2   # (m-1),(m+1)
    # loop
    mult += terms * 1
    add += terms * 1
    # scalar multiplies 2*s and /ln10 ignored as mul count? include:
    mult += 1  # 2*s
    return mult, add


# =========================
# Training + Precision Check
# =========================
@torch.no_grad()
def nn_predict_log10_x(model, x: float):
    m, k = split_mantissa_exponent_base10(x)
    z = norm_m(m)
    inp = torch.tensor([[z]], dtype=torch.float32, device=DEVICE)
    pred_m = model(inp).item()
    return k + pred_m


def make_eval_grid():
    # Evaluate many points outside training too
    eval_x = []
    # inside [1,10)
    for v in np.linspace(1.001, 9.999, 800):
        eval_x.append(float(v))
    # outside: very small and large
    for v in [1e-6, 1e-3, 1e-2, 1e-1, 0.5, 10, 25, 100, 1e3, 1e6]:
        eval_x.append(float(v))
    # random across many magnitudes
    rng = np.random.default_rng(0)
    for _ in range(600):
        exp = rng.integers(-6, 7)
        mant = rng.uniform(1.0, 10.0)
        eval_x.append(float(mant * (10.0 ** exp)))
    return eval_x


def pass_precision(model, decimals: int, eval_x):
    # compare with math.log10 (ground truth)
    tol = 0.5 * (10 ** (-decimals))  # rounding equivalence
    worst = 0.0
    worst_x = None
    for x in eval_x:
        truev = math.log10(x)
        predv = nn_predict_log10_x(model, x)
        err = abs(predv - truev)
        if err > worst:
            worst = err
            worst_x = x
        if err > tol:
            return False, worst, worst_x
    return True, worst, worst_x


def train_one_arch(ms_train, y_m_train, layers, hidden, decimals, eval_x):
    set_seed(42)
    model = LogNN(layers, hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    x_train = torch.tensor(norm_m(ms_train), dtype=torch.float32, device=DEVICE).view(-1, 1)
    y_train = torch.tensor(y_m_train, dtype=torch.float32, device=DEVICE).view(-1, 1)

    for epoch in range(1, MAX_EPOCHS_ADAM + 1):
        model.train()
        opt.zero_grad()
        out = model(x_train)
        loss = loss_fn(out, y_train)
        loss.backward()
        opt.step()

        # early check every 1000
        if epoch % 1000 == 0 or epoch == 1:
            model.eval()
            ok, worst, wx = pass_precision(model, decimals, eval_x)
            if ok:
                break

    if USE_LBFGS:
        # fine-tune with LBFGS for better numeric fit
        model.train()
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=400, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            out = model(x_train)
            loss = loss_fn(out, y_train)
            loss.backward()
            return loss

        lbfgs.step(closure)

    model.eval()
    ok, worst, wx = pass_precision(model, decimals, eval_x)
    return model, ok, worst, wx


def search_best_nn(ms_train, y_m_train, eval_x):
    print("\n=== NN ARCH SEARCH (big -> small) ===")
    best = None  # (decimals, layers, hidden, params, model, worst_err, worst_x)

    for decimals in TRY_DECIMALS:
        print(f"\n--- Trying precision target: {decimals} decimals ---")
        for layers, hidden in CANDIDATES:
            print(f"\nTraining arch layers={layers}, hidden={hidden}")
            model, ok, worst, wx = train_one_arch(ms_train, y_m_train, layers, hidden, decimals, eval_x)
            params = count_params(model)
            print(f"Arch layers={layers}, hidden={hidden} | params={params} | PASS={ok} | worst_err={worst:.3e} @ x={wx}")
            if ok:
                # since we go big->small, first PASS is minimal for this decimals
                best = (decimals, layers, hidden, params, model, worst, wx)
                return best

    return best


# =========================
# Taylor terms minimization
# =========================
def taylor_pass_terms(decimals: int, terms: int, eval_x):
    tol = 0.5 * (10 ** (-decimals))
    worst = 0.0
    worst_x = None
    for x in eval_x:
        truev = math.log10(x)
        predv = taylor_log10_x(x, terms)
        err = abs(predv - truev)
        if err > worst:
            worst = err
            worst_x = x
        if err > tol:
            return False, worst, worst_x
    return True, worst, worst_x


def find_min_taylor_terms(decimals: int, eval_x):
    # start from MAX and reduce
    print("\n=== TAYLOR TERMS SEARCH (smallest that still passes) ===")
    # first ensure MAX works
    ok, worst, wx = taylor_pass_terms(decimals, MAX_TAYLOR_TERMS, eval_x)
    if not ok:
        print(f"Even {MAX_TAYLOR_TERMS} terms did NOT pass {decimals} decimals. Increase MAX_TAYLOR_TERMS.")
        return None

    best_terms = MAX_TAYLOR_TERMS
    for terms in range(MAX_TAYLOR_TERMS, 0, -1):
        ok, worst, wx = taylor_pass_terms(decimals, terms, eval_x)
        if ok:
            best_terms = terms
        else:
            break

    # because loop breaks AFTER failing, last passing is best_terms
    ok, worst, wx = taylor_pass_terms(decimals, best_terms, eval_x)
    print(f"Minimum Taylor terms for {decimals} decimals: {best_terms} | worst_err={worst:.3e} @ x={wx}")
    return best_terms


# =========================
# MAIN
# =========================
def main():
    print("Loading Excel training table...")
    x_vals, true_logs = load_table(EXCEL_PATH)

    # Convert training into mantissa-space targets:
    # true log10(x) = k + log10(m) => log10(m) = true - k
    ms, ks = vector_split_mk(x_vals)
    y_m = true_logs - ks.astype(np.float64)

    print(f"Training points: {len(x_vals)} (range: {x_vals.min():.3f} to {x_vals.max():.3f})")
    eval_x = make_eval_grid()
    print(f"Evaluation points: {len(eval_x)}")

    # Search NN (highest decimals first, minimal arch that passes)
    best = search_best_nn(ms, y_m, eval_x)
    if best is None:
        print("\n❌ No NN configuration met the required precision even at 3 decimals.")
        return

    decimals, layers, hidden, params, model, worst_err, worst_x = best
    print("\n✅ BEST NN FOUND")
    print(f"Precision: {decimals} decimals")
    print(f"Architecture: layers={layers}, hidden={hidden}")
    print(f"Trainable params: {params}")
    nn_mult, nn_add = count_ops_mlp(layers, hidden)
    print(f"NN ops per inference (approx): mult={nn_mult}, add={nn_add}")

    # Taylor: minimal terms for same decimals
    terms = find_min_taylor_terms(decimals, eval_x)
    if terms is None:
        return
    tay_mult, tay_add = count_ops_taylor(terms)
    print(f"Taylor ops per inference (approx): mult={tay_mult}, add={tay_add}")

    # Compare efficiency
    print("\n=== Efficiency Comparison (lower is better) ===")
    print(f"NN     : mult={nn_mult}, add={nn_add}, total={nn_mult + nn_add}")
    print(f"Taylor : mult={tay_mult}, add={tay_add}, total={tay_mult + tay_add}")

    # Demo predictions
    tests = [0.1, 0.5, 1.0, 2.0, 9.99, 10.0, 25.0, 100.0, 1e-6, 1e6]
    print("\n=== Sample Predictions ===")
    print("x\t\tNN\t\tTaylor\t\tTrue\t\t|NN-True|\t|Tay-True|")
    for x in tests:
        nnv = nn_predict_log10_x(model, x)
        tayv = taylor_log10_x(x, terms)
        tr = math.log10(x)
        print(f"{x:<10g}\t{nnv:+.8f}\t{tayv:+.8f}\t{tr:+.8f}\t{abs(nnv-tr):.2e}\t{abs(tayv-tr):.2e}")

    # Save model for "pretrained load"
    save_path = "best_log10_nn.pth"
    torch.save(
        {
            "decimals": decimals,
            "layers": layers,
            "hidden": hidden,
            "state_dict": model.state_dict(),
        },
        save_path,
    )
    print(f"\nSaved pretrained model to: {save_path}")
    print("To load later, use the load snippet at bottom of this file.")


if __name__ == "__main__":
    main()

"""
# =========================
# LOAD PRETRAINED (later)
# =========================
ckpt = torch.load("best_log10_nn.pth", map_location="cpu")
model = LogNN(ckpt["layers"], ckpt["hidden"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(nn_predict_log10_x(model, 100.0))  # should be ~2.0
"""
