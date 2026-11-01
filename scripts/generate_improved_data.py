# OneFit Connect: PROJECTED "after" dataset (Team 4)
# Takes the baseline 2,700-session dataset (seed 2026) and applies five documented,
# rule-based effects of the OneFit Connect prototype. This is a simulation of expected
# impact, not measured data. Every assumption is a named constant below.
import json
import numpy as np
import pandas as pd
import generate_fitness_chain_data as base   # builds the baseline `out` frame (no file output on import)

rng = np.random.default_rng(2027)
b = base.out.copy()
s = base.s                      # working frame with peak/dow/month/day_load etc.

# ------------------------- ASSUMPTIONS (edit here) -------------------------
A_ERR_PREVENT = 0.80      # validation + one-tap check-in prevent 80% of record errors
A_XLOC_CONVERT = 0.60     # 60% of turned-away cross-location requests become bookings (seat permitting)
A_REMINDER_RECOVER = 0.25 # SMS/push reminders recover 25% of would-be no-shows
A_OUTREACH_CUT = 0.20     # early win-back outreach cuts churn-risk attendees by 20%
RATING_ERR_PENALTY = 0.55 # same penalty the baseline model applied to a session with a data error
PLATFORM_COST_TOTAL = 14400.0   # $2,400 / month x 6 months for hosting, messaging, licences
# ----------------------------------------------------------------------------

n = len(b)
a = b.copy()

# 1) Data errors: shared validated system
had_err = (b["Data_Error_Type"] != "No Error").values
prevented = had_err & (rng.random(n) < A_ERR_PREVENT)
a.loc[prevented, "Data_Error_Type"] = "No Error"
has_err_after = (a["Data_Error_Type"] != "No Error").astype(int).values

# 2) Cross-location access: location-locked memberships become one shared membership
denied = b["Cross_Location_Denied"].values
want = rng.binomial(denied, A_XLOC_CONVERT)
free = (b["Class_Capacity"] - b["Bookings"]).values
placed = np.minimum(want, free)
a["Bookings"] = b["Bookings"].values + placed
a["Cross_Location_Booked"] = placed
a["Cross_Location_Denied"] = denied - placed

# 3) Reminders: fewer no-shows; plus attendance from the newly placed members
p_ns = s["loc"].map(base.LOC_NOSHOW).values * np.where(s["peak"] == 1, 0.90, 1.10)
noshows = b["Bookings"].values - b["Attendance"].values
recovered = rng.binomial(noshows, A_REMINDER_RECOVER)
extra_att = rng.binomial(placed, 1 - p_ns * (1 - A_REMINDER_RECOVER))
att = np.minimum(b["Attendance"].values + recovered + extra_att, a["Class_Capacity"].values)
a["Attendance"] = att.astype(int)

# 4) Rating: sessions whose error was prevented regain the error penalty
rating = b["Member_Rating"].values + np.where(prevented, RATING_ERR_PENALTY, 0.0)
a["Member_Rating"] = np.clip(rating, 2.5, 5.0).round(2)

# 5) Churn-risk attendees: same risk model, new inputs, then outreach reduces it
p_cr = (0.11 + 0.012 * (s["month"].values - 1)
        + 0.09 * has_err_after
        + 0.06 * np.clip(4.4 - a["Member_Rating"].values, 0, 1.5)
        + np.where(s["peak"].values == 0, 0.02, 0.0))
p_cr = np.clip(p_cr, 0.03, 0.6) * (1 - A_OUTREACH_CUT)
a["Churn_Risk_Attendees"] = rng.binomial(a["Attendance"].values, p_cr).astype(int)

# Money (platform cost is spread evenly over sessions so the margin is honest)
rate = s["iid"].map({k: v["rate"] for k, v in base.INSTR.items()}).values
overhead = base.OVERHEAD_BASE + np.where(s["peak"] == 1, base.OVERHEAD_PEAK_EXTRA, 0.0)
a["Session_Revenue_USD"] = (a["Attendance"] * base.REV_PER_VISIT).round(2)
a["Session_Cost_USD"] = (rate + base.VAR_COST_PER_ATTENDEE * a["Attendance"] + overhead
                         + PLATFORM_COST_TOTAL / n).round(2)
a["Session_Profit_USD"] = (a["Session_Revenue_USD"] - a["Session_Cost_USD"]).round(2)

# ------------------------- summaries -------------------------
def kpis(d):
    e = (d["Data_Error_Type"] != "No Error")
    return {
        "util": round(d.Attendance.sum() / d.Class_Capacity.sum() * 100, 1),
        "margin": round(d.Session_Profit_USD.sum() / d.Session_Revenue_USD.sum() * 100, 1),
        "denied": int(d.Cross_Location_Denied.sum()),
        "err": round(e.mean() * 100, 1),
        "churn": round(d.Churn_Risk_Attendees.sum() / d.Attendance.sum() * 100, 1),
        "rating": round(d.Member_Rating.mean(), 2),
        "attendance": int(d.Attendance.sum()),
        "bookings": int(d.Bookings.sum()),
        "revenue": round(d.Session_Revenue_USD.sum()),
        "profit": round(d.Session_Profit_USD.sum()),
        "errors": int(e.sum()),
        "churners": int(d.Churn_Risk_Attendees.sum()),
    }

def group(d, col, order=None):
    out = {}
    keys = order or sorted(d[col].unique())
    for k in keys:
        g = d[d[col] == k]
        out[str(k)] = kpis(g)
    return out

months = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun"}
for d in (b, a):
    d["Month"] = pd.to_datetime(d["Session_Date"]).dt.month

summary = {}
for name, d in (("before", b), ("after", a)):
    summary[name] = {
        "overall": kpis(d),
        "byLoc": group(d, "Location_ID", ["LOC01", "LOC02", "LOC03"]),
        "byMonth": group(d, "Month", [1, 2, 3, 4, 5, 6]),
        "byClass": group(d, "Class_Type", ["Spin", "HIIT", "Strength", "Pilates", "Yoga", "Zumba"]),
    }
summary["after"]["crossBooked"] = int(placed.sum())
summary["after"]["errorsPrevented"] = int(prevented.sum())
summary["after"]["noshowsRecovered"] = int(recovered.sum())
summary["assumptions"] = {
    "A_ERR_PREVENT": A_ERR_PREVENT, "A_XLOC_CONVERT": A_XLOC_CONVERT,
    "A_REMINDER_RECOVER": A_REMINDER_RECOVER, "A_OUTREACH_CUT": A_OUTREACH_CUT,
    "RATING_ERR_PENALTY": RATING_ERR_PENALTY, "PLATFORM_COST_TOTAL": PLATFORM_COST_TOTAL,
}
summary["months"] = [months[i] for i in range(1, 7)]

if __name__ == "__main__":
    b.drop(columns=["Month"]).to_csv("fitness_chain_sessions_before.csv", index=False)
    a.drop(columns=["Month"]).to_csv("fitness_chain_sessions_after.csv", index=False)
    json.dump(summary, open("summary.json", "w"), indent=1)
    print(json.dumps({"before": summary["before"]["overall"], "after": summary["after"]["overall"]}, indent=1))
    print("byLoc err before", {k: v["err"] for k, v in summary["before"]["byLoc"].items()})
    print("byLoc err after", {k: v["err"] for k, v in summary["after"]["byLoc"].items()})
    print("crossBooked", summary["after"]["crossBooked"], "prevented", summary["after"]["errorsPrevented"])
    print("nulls", a.isna().sum().sum(), "dups", a.Session_ID.duplicated().sum(),
          "att>cap", int((a.Attendance > a.Class_Capacity).sum()), "bk>cap", int((a.Bookings > a.Class_Capacity).sum()),
          "att>bk", int((a.Attendance > a.Bookings).sum()))
