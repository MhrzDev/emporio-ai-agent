import unicodedata

import pandas as pd

from .config import DATA_DIR


def _normalize(value: object) -> str:
    text = str(value or "").casefold()
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))

def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)

def search_products(query: str = "", category: str = "", max_price: float | None = None,
                    in_stock_only: bool = True, out_of_stock_only: bool = False) -> list[dict]:
    products, categories = _load("products.csv"), _load("categories.csv")
    df = products.merge(categories[["category_id", "name"]], on="category_id", how="left",
                        suffixes=("_product", "_category"))
    df["_search"] = (df["name_product"] + " " + df["description"] + " " +
                     df["name_category"].fillna("")).map(_normalize)
    if query:
        needle = _normalize(query).strip()
        df = df[df["_search"].str.contains(needle, regex=False, na=False)]
    if category:
        needle = _normalize(category).strip()
        df = df[df["name_category"].map(_normalize).str.contains(needle, regex=False, na=False)]
    if max_price is not None:
        df = df[df["price_brl"] <= max_price]
    if in_stock_only and out_of_stock_only:
        raise ValueError("Choose either in_stock_only or out_of_stock_only, not both.")
    if in_stock_only:
        df = df[(df["status"].str.lower() == "active") & (df["stock_quantity"] > 0)]
    elif out_of_stock_only:
        df = df[(df["status"].str.lower() == "active") & (df["stock_quantity"] <= 0)]
    return df.rename(columns={"name_product": "product_name", "name_category": "category"})[
        ["product_id", "product_name", "category", "price_brl", "stock_quantity", "status", "description"]
    ].to_dict("records")

def get_product_details(product_name: str) -> dict | None:
    results = search_products(query=product_name, in_stock_only=False)
    exact = [p for p in results if p["product_name"].casefold() == product_name.casefold()]
    return exact[0] if exact else (results[0] if len(results) == 1 else None)

def get_active_promotions() -> list[dict]:
    promos = _load("promotions.csv")
    products = _load("products.csv")[["product_id", "name", "price_brl", "stock_quantity", "status"]]
    df = promos[promos["is_active"].astype(int) == 1].merge(products, on="product_id", how="inner")
    df = df[(df["stock_quantity"] > 0) & (df["status"].str.lower() == "active")]
    return df.to_dict("records")

def get_order_status(order_id: int, customer_email: str) -> dict | None:
    """Require both order ID and matching customer email before disclosing order data."""
    orders, customers = _load("orders.csv"), _load("customers.csv")
    match = orders[orders["order_id"] == int(order_id)].merge(customers, on="customer_id")
    if match.empty or str(match.iloc[0]["email"]).strip().lower() != customer_email.strip().lower():
        return None
    row = match.iloc[0]
    safe = ["order_id", "order_date", "status", "total_brl", "payment_method",
            "tracking_code", "estimated_delivery", "name"]
    return row[safe].where(pd.notna(row[safe]), None).to_dict()


def list_available_categories() -> list[dict]:
    """Return only categories that have active products with positive stock."""
    products, categories = _load("products.csv"), _load("categories.csv")
    active = products[
        (products["status"].str.lower() == "active") &
        (products["stock_quantity"] > 0)
    ].merge(
        categories[["category_id", "name"]],
        on="category_id",
        how="left",
        suffixes=("_product", "_category"),
    )
    if active.empty:
        return []
    summary = []
    for category_name, group in active.groupby("name_category", sort=True):
        examples = group.sort_values("name_product")["name_product"].head(3).tolist()
        summary.append({
            "category": str(category_name),
            "available_count": len(group),
            "examples": examples,
        })
    return summary
