import pytest

from src.ollama_agent import _dispatch


def test_catalogue_tool_returns_structured_results():
    result = _dispatch("search_catalogue", {"category": "Violões", "max_price": 1000})
    assert "products" in result
    assert result["currency"] == "BRL"
    assert all(item["price_brl"] <= 1000 for item in result["products"])


def test_unknown_tool_is_rejected():
    with pytest.raises(ValueError):
        _dispatch("read_customer_database", {})


def test_unexpected_arguments_are_rejected():
    with pytest.raises(ValueError):
        _dispatch("list_active_promotions", {"customer_email": "x@example.com"})


def test_policy_tool_returns_page_aware_excerpts():
    result = _dispatch("search_store_policies", {"query": "devolução compra online"})
    assert "excerpts" in result
    assert all("page" in item for item in result["excerpts"])



def test_catalogue_category_tool_lists_categories():
    result = _dispatch("list_catalogue_categories", {})
    assert result["categories"]
    assert all(item["available_count"] > 0 for item in result["categories"])
    assert result["note"]


def test_catalogue_tool_can_search_only_out_of_stock_products():
    result = _dispatch("search_catalogue", {
        "category": "Violões", "include_out_of_stock": True
    })
    assert result["stock_filter"] == "out_of_stock"
    assert all(item["stock_quantity"] <= 0 for item in result["products"])


def test_catalogue_tool_returns_complete_category_result():
    result = _dispatch("search_catalogue", {"category": "Violões"})
    assert result["count"] == len(result["products"])
    assert result["count"] > 12


def test_named_model_lookup_is_prefetched_and_answered_directly(monkeypatch):
    import src.ollama_agent as module

    captured = {}
    class FakeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"message": {"role": "assistant", "content": "Yamaha C40 Nylon Natural custa R$ 599,90 e há 12 unidades em estoque."}}

    def fake_post(url, json, timeout):
        captured["messages"] = json["messages"]
        return FakeResponse()

    monkeypatch.setattr(module.requests, "post", fake_post)
    answer = module.answer_with_ollama("Qual é o preço e a disponibilidade do Yamaha C40?")
    assert "599,90" in answer
    assert "12 unidade(s)" in answer
    assert "Yamaha C40 Nylon Natural" in answer
    # Exact catalogue facts are returned directly, without relying on model tool selection.
    assert "messages" not in captured


def test_llm_output_repairs_escaped_newlines_and_currency(monkeypatch):
    import src.ollama_agent as module

    class FakeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"message": {"role": "assistant", "content": r"Preço: R` 599,90\nEstoque: 12"}}

    monkeypatch.setattr(module.requests, "post", lambda *a, **k: FakeResponse())
    answer = module.answer_with_ollama("teste")
    assert "R$ 599,90" in answer
    assert "\\n" not in answer
