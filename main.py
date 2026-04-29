import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score


"""
Side project to draft a predictive model using best r2 of Linear Regression (probably best), Random Forest or Gradient Boosting
Replace Gradient Boosting with XGBoost maybe, might be better than Linear regression but I am unsure. 

Sample 4/28/26:
Using 110 lb tube (Sample 2 x 3 x .180 ordered today):
Prediction - 1.19/lb - $131 total
Real - (108 lb tube) - ~1.17/lb - $126.11 total
Roughly 4% difference!
"""

def parse_weight(value):
    match = re.search(r"[\d.]+", str(value))
    return float(match.group()) if match else np.nan


def extract_shape(item):
    item = str(item).upper()
    if "REBAR" in item:
        return "REBAR"
    if "GRATING" in item:
        return "GRATING"
    if "BEARING PILE" in item or "HP BEAM" in item:
        return "BEAM"
    # TUBE before SQUARE/ROUND so "TUBING SQUARE" -> TUBE, not SQUARE BAR
    if any(k in item for k in ("TUBE", "TUBING", "R/T", "S/T")):
        return "TUBE"
    if "BEAM" in item:
        return "BEAM"
    if "CHANNEL" in item:
        return "CHANNEL"
    if "PIPE" in item:
        return "PIPE"
    if any(k in item for k in ("PLATE", "SHEET")):
        return "PLATE/SHEET"
    if any(k in item for k in ("FLAT BAR", "FLAT", "STRIP")):
        return "FLAT BAR"
    if "ANGLE" in item:
        return "ANGLE"
    if any(k in item for k in ("ROUND BAR", "HR ROUND", "ROUND CD", "ROUND HR")):
        return "ROUND BAR"
    if any(k in item for k in ("SQUARE BAR", "HR SQUARE", "SQUARE HR")):
        return "SQUARE BAR"
    return "OTHER"


def fraction_to_float(s):
    s = str(s).strip()
    m = re.match(r"^(\d+)-(\d+)/(\d+)$", s)
    if m:
        return int(m.group(1)) + int(m.group(2)) / int(m.group(3))
    m = re.match(r"^(\d+)/(\d+)$", s)
    if m:
        return int(m.group(1)) / int(m.group(2))
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_dimensions(item):
    """Extract up to 3 cross-section dimensions from an item description string."""
    text = str(item).upper()
    # Remove grade designators so their numbers aren't captured as dimensions
    text = re.sub(r"\bA-?\d{2,4}(?:[/-]\d+(?:-\d+)?)?\b", "", text)
    text = re.sub(r"\b(?:SCH|GR|GA|PE)\s*\d+\b", "", text)

    num_pat = r"\d+-\d+/\d+|\d+/\d+|\.\d+|\d+(?:\.\d+)?"
    dims = []
    for token in re.findall(num_pat, text):
        val = fraction_to_float(token)
        if not np.isnan(val) and val > 0:
            dims.append(val)
        if len(dims) == 3:
            break

    while len(dims) < 3:
        dims.append(np.nan)
    return dims[0], dims[1], dims[2]


# Minor cleaning — should be moved to SO-Processing proper later.
def prepare_data(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    df["Weight_num"] = df["Weight/Length"].apply(parse_weight)
    df["Shape"] = df["Item"].apply(extract_shape)
    df["Is_Domestic"] = df["Item"].str.contains(r"domestic", case=False, na=False).astype(int)

    dims = df["Item"].apply(lambda x: pd.Series(parse_dimensions(x), index=["dim1", "dim2", "dim3"]))
    df = pd.concat([df, dims], axis=1)

    # Keep only lb-based rows; derive consistent $/lb target from Total / weight
    df = df[df["Weight/Length"].str.contains("LB", case=False, na=False)].copy()
    df["price_per_lb"] = df["Total"] / df["Weight_num"]
    df = df[df["price_per_lb"] > 0].dropna(subset=["price_per_lb", "Weight_num", "Year", "Month"])

    vendor_encoder = LabelEncoder()
    df["Vendor_enc"] = vendor_encoder.fit_transform(df["Vendor"].astype(str))

    return df, vendor_encoder


FEATURES = ["Vendor_enc", "Quantity", "Weight_num", "Year", "Month", "Is_Domestic", "dim1", "dim2", "dim3"]


def best_model_for(subset):
    """Train and select the best regressor on a shape subset.
    Returns (model, r2, dim_medians). dim_medians used to impute missing dims at predict time."""
    #  missing dimension values with the shape median
    dim_medians = {}
    for col in ("dim1", "dim2", "dim3"):
        median = subset[col].median()
        dim_medians[col] = median if not np.isnan(median) else 0.0
        subset = subset.copy()
        subset[col] = subset[col].fillna(dim_medians[col])

    x = subset[FEATURES]
    y = subset["price_per_lb"]

    if len(subset) < 20:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(x, y)
        return model, float("nan"), dim_medians

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
    }

    best, best_r2, best_name = None, -np.inf, ""
    for name, m in candidates.items():
        m.fit(x_train, y_train)
        r2 = r2_score(y_test, m.predict(x_test))
        if r2 > best_r2:
            best_r2, best, best_name = r2, m, name

    best.fit(x, y)
    return best, best_r2, dim_medians


# Train one model per shape category so each learns only its own pricing history.
def train_shape_models(df):
    shape_models = {}
    name_map = {
        "LinearRegression": "Linear",
        "RandomForestRegressor": "Random Forest",
        "GradientBoostingRegressor": "Gradient Boosting",
    }

    print("\nTraining per-shape models:")
    print(f"  {'Shape':<15} {'Rows':>5}  {'Best model':<20} {'R²':>7}")
    print(f"  {'-'*15} {'-'*5}  {'-'*20} {'-'*7}")

    for shape, group in df.groupby("Shape"):
        model, r2, dim_medians = best_model_for(group.copy())
        shape_models[shape] = {"model": model, "dim_medians": dim_medians}
        r2_str = f"{r2:.4f}" if not np.isnan(r2) else "  n/a"
        model_name = name_map.get(type(model).__name__, type(model).__name__)
        print(f"  {shape:<15} {len(group):>5}  {model_name:<20} {r2_str:>7}")

    return shape_models

"""

Loops until quit after models trained. Will need to export this to an outside function eventually. 

"""
def interactive_predict(shape_models, vendor_encoder):
    now = datetime.now()
    known_shapes = sorted(shape_models.keys())
    all_vendor_encs = list(range(len(vendor_encoder.classes_)))

    print("\n" + "=" * 52)
    print("  STEEL PRICE PREDICTOR")
    print(f"  Pricing as of: {now.strftime('%B %Y')}")
    print(f"  Price unit: $/lb  |  Type 'quit' to exit")
    print("=" * 52)

    while True:
        print()
        print(f"Shape categories: {', '.join(known_shapes)}")
        item_desc = input("Item description (or 'quit'): ").strip()
        if item_desc.lower() == "quit":
            break

        shape = extract_shape(item_desc)
        if shape not in shape_models:
            print(f"  [!] No model for shape '{shape}' — defaulting to 'OTHER'.")
            shape = "OTHER"

        is_domestic = 1 if "domestic" in item_desc.lower() else 0

        # Parse and impute dimensions from description
        dim_medians = shape_models[shape]["dim_medians"]
        d1, d2, d3 = parse_dimensions(item_desc)
        d1 = d1 if not np.isnan(d1) else dim_medians["dim1"]
        d2 = d2 if not np.isnan(d2) else dim_medians["dim2"]
        d3 = d3 if not np.isnan(d3) else dim_medians["dim3"]

        dim_str = f"{d1} x {d2} x {d3}" if not np.isnan(d3) else f"{d1} x {d2}"
        print(f"  -> Shape: {shape}  |  Dims: {dim_str}  |  Domestic: {'Yes' if is_domestic else 'No'}")

        try:
            quantity = int(input("Quantity (pieces): ").strip())
        except ValueError:
            print("  [!] Invalid quantity — skipping.")
            continue

        try:
            total_weight = float(input("Total weight (lbs): ").strip())
        except ValueError:
            print("  [!] Invalid weight — skipping.")
            continue

        model = shape_models[shape]["model"]

        # Average prediction across all vendors for a market-wide estimate
        rows = pd.DataFrame([{
            "Vendor_enc": v,
            "Quantity": quantity,
            "Weight_num": total_weight,
            "Year": now.year,
            "Month": now.month,
            "Is_Domestic": is_domestic,
            "dim1": d1,
            "dim2": d2,
            "dim3": d3,
        } for v in all_vendor_encs])[FEATURES]

        price_per_lb = model.predict(rows).mean()
        line_total = price_per_lb * total_weight

        print(f"\n  Predicted price:  ${price_per_lb:.4f} / lb")
        print(f"  Total weight:     {total_weight:.2f} lbs")
        print(f"  Estimated total:  ${line_total:.2f}  ({quantity} pc)")


def main():
    df = pd.read_excel("Steel_Inventory_Final.xlsx")
    df, vendor_encoder = prepare_data(df)
    shape_models = train_shape_models(df)
    interactive_predict(shape_models, vendor_encoder)


if __name__ == "__main__":
    main()
