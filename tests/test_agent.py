from src.agent import answer_without_llm

def test_violao_category_returns_catalog_items():
    response = answer_without_llm("Quais violões vocês possuem?")
    assert "Yamaha C40" in response
    assert "R$ 599,90" in response
    assert "\\\\n" not in response

def test_returns_policy_details_not_only_page_numbers():
    response = answer_without_llm("Qual é a política de devolução?")
    assert "7 dias corridos" in response
    assert "10 dias úteis" in response

def test_accessory_scope_is_clear():
    response = answer_without_llm("Vocês vendem palhetas?")
    assert "não comercializa acessórios" in response

def test_price_limit_filters_catalog():
    response = answer_without_llm("Quais violões vocês possuem por menos de R$ 1.000?")
    assert "Yamaha C40" in response
    assert "Crafter HT-100" not in response
    assert "R$ 1.000,00" in response

def test_return_question_about_guitar_routes_to_policy_not_catalog():
    response = answer_without_llm("Comprei um violão pela internet há 5 dias. Posso devolver?")
    assert "arrependimento em compra online" in response.lower()
    assert "7 dias corridos" in response
    assert "Yamaha C40" not in response

def test_guarantee_policy_is_grounded():
    response = answer_without_llm("Qual é a garantia legal dos instrumentos?")
    assert "90 dias" in response
    assert "página 7" in response


def test_defect_after_40_days_routes_to_warranty_guidance():
    response = answer_without_llm("Comprei um instrumento há 40 dias e ele apresentou defeito. O que faço?")
    assert "30 dias corridos" in response
    assert "fabricante" in response.lower()
    assert "catálogo" not in response.lower()

def test_problem_phrase_routes_to_defect_policy():
    response = answer_without_llm("Meu instrumento parou de funcionar. Como procedo?")
    assert "30 dias corridos" in response
    assert "fabricante" in response.lower()



def test_generic_catalogue_question_lists_categories():
    response = answer_without_llm("Quais instrumentos musicais estão disponíveis na loja?")
    assert "Guitarras" in response
    assert "Violões" in response
    assert "Ukuleles" in response
    assert "produto(s) disponível(is)" in response
    assert "\\\\n" not in response


def test_category_query_returns_all_in_stock_guitars_not_truncated():
    from src.data_tools import search_products
    response = answer_without_llm("Quais violões estão disponíveis e quanto custam?")
    expected = search_products(category="Violões")
    assert len(expected) > 12
    assert all(item["product_name"] in response for item in expected)


def test_out_of_stock_question_does_not_return_in_stock_items():
    response = answer_without_llm("Existe algum violão sem estoque?")
    assert "sem estoque" in response.lower()
    assert "Yamaha C40" not in response
    assert "disponíveis no catálogo" not in response.lower()


def test_named_product_price_and_stock_works_without_llm():
    response = answer_without_llm("Qual é o preço e a disponibilidade do Yamaha C40?")
    assert "Yamaha C40 Nylon Natural" in response
    assert "R$ 599,90" in response
    assert "12 unidade(s)" in response
    assert "Qual instrumento ou modelo" not in response
