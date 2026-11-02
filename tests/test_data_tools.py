from src.data_tools import search_products, get_order_status

def test_product_search_respects_price_and_stock():
    products = search_products(max_price=1000, in_stock_only=True)
    assert all(p["price_brl"] <= 1000 and p["stock_quantity"] > 0 for p in products)

def test_order_data_requires_matching_email():
    assert get_order_status(1, "wrong@example.com") is None

def test_order_data_returns_for_matching_email():
    import pandas as pd
    from src.config import DATA_DIR
    customer_id = int(pd.read_csv(DATA_DIR / "orders.csv").query("order_id == 1").iloc[0]["customer_id"])
    email = pd.read_csv(DATA_DIR / "customers.csv").query("customer_id == @customer_id").iloc[0]["email"]
    assert get_order_status(1, email) is not None



def test_category_overview_only_includes_in_stock_active_products():
    from src.data_tools import list_available_categories
    categories = list_available_categories()
    assert categories
    assert all(item["available_count"] > 0 for item in categories)
    assert all(item["examples"] for item in categories)
