# Fitness Chain: One Brand, Three Systems
# Synthetic class-session dataset generator (Team 4)
# Runs as-is in Google Colab or any Python 3.9+ environment (pandas, numpy, openpyxl).
#
# Grain: one row = one scheduled group-fitness class session.
# Period: 2026-01-01 to 2026-06-30, 900 sessions per location, 2700 rows, 16 columns.
#
# LOC01 = scheduling app, LOC02 = shared spreadsheet, LOC03 = manual front-desk log
# (modeling assumption: each location runs its own separate tool).

import numpy as np
import pandas as pd
from datetime import date

SEED = 2026
rng = np.random.default_rng(SEED)

START, END = date(2026, 1, 1), date(2026, 6, 30)
DAYS = pd.date_range(START, END)
LOCS = ["LOC01", "LOC02", "LOC03"]
SESSIONS_PER_LOC = 900

# ---------- Reference tables (rules, see Data Sources and Rules slide) ----------
CLASS_CAP = {"Spin": 20, "HIIT": 24, "Strength": 24, "Pilates": 28, "Yoga": 36, "Zumba": 36}
CLASS_POP = {"Spin": 1.15, "HIIT": 1.10, "Strength": 0.95, "Pilates": 1.05, "Yoga": 0.95, "Zumba": 0.82}
CLASS_QUAL = {"Spin": 0.05, "HIIT": 0.00, "Strength": -0.03, "Pilates": 0.05, "Yoga": 0.03, "Zumba": 0.02}
VIGOROUS = {"HIIT", "Spin", "Zumba"}

LOC_DEMAND = {"LOC01": 1.00, "LOC02": 0.90, "LOC03": 0.84}
LOC_NOSHOW = {"LOC01": 0.10, "LOC02": 0.16, "LOC03": 0.22}      # 10-30% benchmark, reminders lower it
LOC_ERROR = {"LOC01": 0.03, "LOC02": 0.09, "LOC03": 0.14}       # 2-5% base human error, per record
LOC_XLOC = {"LOC01": 1.25, "LOC02": 1.00, "LOC03": 0.85}        # cross-location interest
ERROR_MIX = {
    "LOC01": (["Attendance Mismatch", "Duplicate Booking"], [0.60, 0.40]),
    "LOC02": (["Duplicate Booking", "Overwritten Entry", "Attendance Mismatch"], [0.35, 0.35, 0.30]),
    "LOC03": (["Missing Sign-In", "Attendance Mismatch", "Wrong Class Listed"], [0.45, 0.30, 0.25]),
}
MONTH_DEMAND = {1: 1.30, 2: 1.12, 3: 1.02, 4: 0.95, 5: 0.90, 6: 0.86}   # January surge then fade
DOW_DEMAND = {0: 1.06, 1: 1.05, 2: 1.02, 3: 1.00, 4: 0.90, 5: 0.95, 6: 0.80}
DOW_SCHEDULE = np.array([0.165, 0.165, 0.155, 0.155, 0.115, 0.140, 0.105])

DUES_PER_MONTH = 65.0            # HFA average monthly dues
VISITS_PER_MONTH = 7e9 / 81e6 / 12   # ~7.2 visits per member per month (HFA)
REV_PER_VISIT = round(DUES_PER_MONTH / VISITS_PER_MONTH, 2)   # ~9.03
VAR_COST_PER_ATTENDEE = 1.10
OVERHEAD_BASE = 79.0             # allocated facility and front desk cost per class slot
OVERHEAD_PEAK_EXTRA = 8.0        # extra floor staffing in peak windows

# ---------- Instructors: 5 per location, one home location each ----------
INSTR = {}
spec_sets = [
    ["Spin", "HIIT"],
    ["HIIT", "Strength", "Zumba"],
    ["Yoga", "Pilates"],
    ["Strength", "Pilates", "Yoga"],
    ["Zumba", "Spin", "Yoga"],
]
load_w = [0.32, 0.26, 0.20, 0.14, 0.08]     # uneven instructor workload
for li, loc in enumerate(LOCS):
    for k in range(5):
        iid = f"INS{li * 5 + k + 1:02d}"
        INSTR[iid] = {
            "loc": loc,
            "w": load_w[k],
            "spec": spec_sets[(k + li) % 5],
            "rate": float(np.clip(rng.normal(24.5, 3.8), 18.0, 35.0).round(2)),
            "quality": float(rng.normal(0, 0.12)),
        }

def loc_class_mix(loc):
    base = np.array([1.0, 1.0, 0.8, 0.9, 1.0, 0.7])
    shift = {"LOC01": [1.2, 1.2, 1, 1, 0.9, 0.9], "LOC02": [1, 1, 1.1, 1.2, 1, 0.9], "LOC03": [0.8, 0.9, 0.9, 1, 1.3, 1.4]}[loc]
    return dict(zip(["Spin", "HIIT", "Strength", "Pilates", "Yoga", "Zumba"], base * np.array(shift)))

WEEKDAY_SLOTS = (["EM", "MID", "AFT", "EVE"], [0.24, 0.22, 0.10, 0.44])
WEEKEND_SLOTS = (["WKM", "WKA"], [0.70, 0.30])
PEAK_SLOTS = {"EM", "EVE"}

# ---------- Build the schedule ----------
rows = []
for loc in LOCS:
    ids = [i for i, v in INSTR.items() if v["loc"] == loc]
    mix = loc_class_mix(loc)
    day_w = np.array([DOW_SCHEDULE[d.dayofweek] for d in DAYS], dtype=float)
    day_w /= day_w.sum()
    used_slot = {}      # (date, slot) -> count of sessions in the location
    instr_slot = set()  # (date, instructor, slot)
    instr_day = {}      # (date, instructor) -> [total, vigorous]
    placed = 0
    while placed < SESSIONS_PER_LOC:
        d = DAYS[rng.choice(len(DAYS), p=day_w)]
        slots, sp = WEEKEND_SLOTS if d.dayofweek >= 5 else WEEKDAY_SLOTS
        ok = False
        for _ in range(25):
            slot = rng.choice(slots, p=sp)
            if used_slot.get((d, slot), 0) >= 3:
                continue
            w = np.array([INSTR[i]["w"] for i in ids]); w = w / w.sum()
            iid = rng.choice(ids, p=w)
            if (d, iid, slot) in instr_slot:
                continue
            spec = INSTR[iid]["spec"]
            cw = np.array([mix[c] for c in spec]); cw = cw / cw.sum()
            ctype = rng.choice(spec, p=cw)
            tot, vig = instr_day.get((d, iid), [0, 0])
            if tot >= 3 or (ctype in VIGOROUS and vig >= 2):
                continue
            ok = True
            break
        if not ok:
            continue
        used_slot[(d, slot)] = used_slot.get((d, slot), 0) + 1
        instr_slot.add((d, iid, slot))
        instr_day[(d, iid)] = [tot + 1, vig + (1 if ctype in VIGOROUS else 0)]
        rows.append({"date": d, "loc": loc, "iid": str(iid), "ctype": str(ctype), "slot": str(slot)})
        placed += 1

s = pd.DataFrame(rows).sort_values(["date", "loc", "slot"]).reset_index(drop=True)
n = len(s)

s["peak"] = s["slot"].isin(PEAK_SLOTS).astype(int)
s["dow"] = s["date"].dt.dayofweek
s["month"] = s["date"].dt.month
s["cap"] = s["ctype"].map(CLASS_CAP)
s["day_load"] = s.groupby(["date", "iid"])["iid"].transform("count")

# ---------- Demand and bookings ----------
slot_factor = np.where(s["peak"] == 1, 1.15, np.where(s["dow"] >= 5, 0.85, 0.52))
f = (slot_factor * s["ctype"].map(CLASS_POP) * s["loc"].map(LOC_DEMAND)
     * s["month"].map(MONTH_DEMAND) * s["dow"].map(DOW_DEMAND))
noise = rng.lognormal(mean=0.0, sigma=0.18, size=n)
bookings = np.minimum(s["cap"].values, np.maximum(2, np.round(s["cap"].values * f.values * noise))).astype(int)
s["bookings"] = bookings

# ---------- Data errors (depend on the location's tool) ----------
p_err = s["loc"].map(LOC_ERROR).values * np.where(s["peak"] == 1, 1.3, 1.0) * np.where(s["bookings"] >= 0.9 * s["cap"], 1.15, 1.0)
is_err = rng.random(n) < p_err
err_type = np.array(["No Error"] * n, dtype=object)
for loc in LOCS:
    idx = np.where(is_err & (s["loc"].values == loc))[0]
    kinds, pk = ERROR_MIX[loc]
    err_type[idx] = rng.choice(kinds, size=len(idx), p=pk)
s["error"] = err_type
s["has_err"] = (s["error"] != "No Error").astype(int)

# ---------- No-shows and attendance ----------
p_ns = s["loc"].map(LOC_NOSHOW).values * np.where(s["peak"] == 1, 0.90, 1.10)
att = rng.binomial(s["bookings"].values, 1 - p_ns)
att = np.maximum(att, np.ceil(0.55 * s["bookings"].values).astype(int))   # show rate floor 55%
att = np.minimum(att, s["bookings"].values)
att = np.maximum(att, 1)
s["attendance"] = att.astype(int)
util = s["attendance"] / s["cap"]

# ---------- Cross-location requests denied (memberships are location locked) ----------
lam = (1.4 * s["ctype"].map(CLASS_POP).values
       * np.where(s["peak"] == 1, 1.5, np.where(s["dow"] >= 5, 0.8, 0.9))
       * s["loc"].map(LOC_XLOC).values * (0.8 + 0.2 * s["month"].map(MONTH_DEMAND).values))
s["denied"] = np.minimum(rng.poisson(lam), 8).astype(int)

# ---------- Money ----------
s["revenue"] = (s["attendance"] * REV_PER_VISIT).round(2)
rate = s["iid"].map({k: v["rate"] for k, v in INSTR.items()})
overhead = OVERHEAD_BASE + np.where(s["peak"] == 1, OVERHEAD_PEAK_EXTRA, 0.0)
s["cost"] = (rate + VAR_COST_PER_ATTENDEE * s["attendance"] + overhead).round(2)
s["profit"] = (s["revenue"] - s["cost"]).round(2)

# ---------- Member rating (data errors, crowding and instructor load pull it down) ----------
r = (4.55
     + s["iid"].map({k: v["quality"] for k, v in INSTR.items()}).values
     + s["ctype"].map(CLASS_QUAL).values
     - 0.55 * s["has_err"].values
     - np.where(util >= 0.95, 0.22, 0.0)
     - np.where(util < 0.35, 0.10, 0.0)
     - np.where(s["day_load"] >= 3, 0.10, 0.0)
     + rng.normal(0, 0.18, size=n))
s["rating"] = np.clip(r, 2.5, 5.0).round(2)

# ---------- Churn-risk attendees (infrequent attenders, rises after the January surge) ----------
p_cr = (0.11 + 0.012 * (s["month"].values - 1)
        + 0.09 * s["has_err"].values
        + 0.06 * np.clip(4.4 - s["rating"].values, 0, 1.5)
        + np.where(s["peak"].values == 0, 0.02, 0.0))
p_cr = np.clip(p_cr, 0.03, 0.6)
s["churn"] = rng.binomial(s["attendance"].values, p_cr).astype(int)

# ---------- Final 16 columns ----------
out = pd.DataFrame({
    "Session_ID": [f"SES{i + 1:05d}" for i in range(n)],
    "Session_Date": s["date"].dt.strftime("%Y-%m-%d"),
    "Location_ID": s["loc"],
    "Instructor_ID": s["iid"],
    "Class_Type": s["ctype"],
    "Class_Capacity": s["cap"].astype(int),
    "Peak_Hour_Flag": s["peak"].astype(int),
    "Bookings": s["bookings"].astype(int),
    "Attendance": s["attendance"].astype(int),
    "Cross_Location_Denied": s["denied"].astype(int),
    "Churn_Risk_Attendees": s["churn"].astype(int),
    "Data_Error_Type": s["error"],
    "Session_Revenue_USD": s["revenue"],
    "Session_Cost_USD": s["cost"],
    "Session_Profit_USD": s["profit"],
    "Member_Rating": s["rating"],
})

if __name__ == "__main__":
    out.to_csv("fitness_chain_sessions.csv", index=False)
    print(out.shape)
    print(out.head())
