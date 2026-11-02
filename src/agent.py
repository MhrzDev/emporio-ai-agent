
import re

from requests.exceptions import RequestException

from .config import USE_LLM
from .data_tools import (
    get_active_promotions,
    get_order_status,
    list_available_categories,
    search_products,
)


def _format_brl(value: float) -> str:
    return (
        f"R$ {value:,.2f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def _extract_max_price(message: str) -> float | None:
    text = message.casefold().replace(".", "").replace(",", ".")

    patterns = [
        r"(?:menos de|abaixo de|até|ate|no máximo|maximo|máximo de)\s*r?\$\s*([0-9]+(?:\.[0-9]{1,2})?)",
        r"r?\$\s*([0-9]+(?:\.[0-9]{1,2})?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))

    return None


def answer_without_llm(message: str) -> str:
    q = message.casefold()

    # Normalize the message for product searches.
    cleaned = re.sub(r"[^a-z0-9áéíóúâêôãõç ]", " ", q)

    stop_words = {
        "qual", "é", "e", "o", "a", "do", "da", "dos", "das",
        "quanto", "custa", "preço", "preco", "disponibilidade",
        "disponível", "disponivel", "está", "esta", "tem",
        "vocês", "voces", "têm", "me", "informe", "diga",
        "por", "favor", "valor", "de",
    }


    product_query = " ".join(
        word for word in cleaned.split()
        if word not in stop_words
    )

    selected = []

    # Detect category queries before searching for individual products.
    category_terms = [
        "viol", "guitarr", "baixo", "bateri",
        "percuss", "teclad", "piano", "ukulele",
        "violino", "sopro",
    ]

    mentions_category = any(term in q for term in category_terms)

    # Search for a specific product or model only when
    # the customer is not asking about an entire category.
    if (
        product_query
        and len(product_query.split()) <= 7
        and not mentions_category
    ):

        matches = search_products(
            query=product_query,
            in_stock_only=False,
        )

        exact = [
            item for item in matches
            if (
                product_query in item["product_name"].casefold()
                or item["product_name"].casefold() in product_query
            )
        ]

        selected = exact or (
            matches if len(matches) == 1 else []
        )

        if selected:
            lines = [
                f"- **{item['product_name']}** — "
                f"{_format_brl(item['price_brl'])} "
                f"(estoque informado: "
                f"{int(item['stock_quantity'])} unidade(s))"
                for item in selected[:5]
            ]

            return (
                "Encontrei este produto no catálogo:\n\n"
                + "\n".join(lines)
                + "\n\nValores e quantidades conforme os arquivos "
                "fornecidos para este desafio."
            )

    # Avoid treating category searches as failed model searches.

    mentions_category = any(term in q for term in category_terms)

    product_intent = any(
        term in q
        for term in [
            "preço", "preco", "quanto custa",
            "qual valor", "disponibilidade",
            "disponível", "disponivel",
        ]
    )

    if (
        product_intent
        and not mentions_category
        and 2 <= len(product_query.split()) <= 7
        and not selected
    ):
        return (
            f"Não encontrei uma correspondência exata para "
            f"'{product_query}' no catálogo fornecido. "
            "Posso ajudar a procurar outro modelo ou instrumento."
        )

    # Resolve policy questions.
    policy_terms = [
        "devolução", "devolucao", "devolver",
        "arrependimento", "troca", "garantia",
        "defeito", "problema", "falha",
        "parou de funcionar", "não funciona",
        "nao funciona",
    ]

    if any(term in q for term in policy_terms):

        if "garantia" in q:
            return (
                "O manual informa garantia legal de 90 dias contra "
                "defeitos de fabricação, contados a partir do recebimento. "
                "A garantia do fabricante pode variar de 6 meses a 2 anos, "
                "conforme o certificado do produto. Há exclusões como "
                "desgaste natural, mau uso, quedas, condições climáticas "
                "extremas e reparos não autorizados (página 7)."
            )

        if any(
            term in q
            for term in [
                "defeito", "problema", "falha",
                "parou de funcionar", "não funciona",
                "nao funciona",
            ]
        ):
            return (
                "Para defeitos de fabricação, o manual prevê troca em "
                "até 30 dias corridos após a compra. Depois desse prazo, "
                "o cliente deve acionar a garantia do fabricante; a loja "
                "pode intermediar. Danos por mau uso, quedas, umidade "
                "excessiva ou modificações não autorizadas não são "
                "cobertos (página 4)."
            )

        received_match = re.search(
            r"\b(?:há|ha)\s+(\d+)\s+dias?\b",
            q,
        )

        received_days = (
            int(received_match.group(1))
            if received_match
            else None
        )

        policy_text = (
            "O manual prevê estas condições (página 4):\n\n"
            "- **Arrependimento em compra online:** solicitação em "
            "até 7 dias corridos após o recebimento, sem necessidade "
            "de justificativa. O produto deve estar na embalagem "
            "original, sem sinais de uso e com todos os acessórios "
            "e manuais. O reembolso ocorre em até 10 dias úteis, "
            "na forma original de pagamento; o frete de devolução "
            "é pago pela loja.\n"
            "- **Defeito de fabricação:** troca em até 30 dias "
            "corridos após a compra; depois disso, acione a garantia "
            "do fabricante.\n"
            "- **Troca por preferência:** até 7 dias, com o produto "
            "em perfeito estado, embalagem original e disponibilidade "
            "do item desejado."
        )

        if received_days is not None:
            if received_days <= 7:
                conclusion = (
                    f"\n\nComo você recebeu o produto há "
                    f"{received_days} dias, a solicitação está dentro "
                    "do prazo de 7 dias corridos. A devolução ainda "
                    "depende do cumprimento das condições de conservação "
                    "e dos acessórios descritos acima."
                )
            else:
                conclusion = (
                    f"\n\nComo você recebeu o produto há "
                    f"{received_days} dias, o prazo de 7 dias corridos "
                    "para arrependimento previsto no manual já foi "
                    "ultrapassado. Entre em contato com a loja para "
                    "verificar se existe alguma alternativa."
                )
        else:
            conclusion = (
                "\n\nPara verificar se você ainda está dentro do prazo "
                "de arrependimento, preciso saber há quantos dias "
                "recebeu o produto."
            )

        return policy_text + conclusion

    # Order status.
    if any(
        term in q
        for term in [
            "pedido", "rastreamento",
            "rastrear", "status do pedido",
        ]
    ):
        return (
            "Posso consultar seu pedido com segurança. Informe "
            "o número do pedido e o e-mail cadastrado na compra."
        )

    # Promotions.
    if any(term in q for term in ["promo", "desconto", "oferta"]):
        promotions = get_active_promotions()

        if not promotions:
            return (
                "Não encontrei promoções ativas nos dados fornecidos. "
                "Posso consultar o preço regular de um instrumento."
            )

        lines = [
            f"- **{item['name']}** — "
            f"{item['discount_percent']}% de desconto"
            for item in promotions
        ]

        return (
            "Estas são as promoções ativas identificadas:\n\n"
            + "\n".join(lines)
        )

    # Store address and contact information.
    if any(
        term in q
        for term in [
            "endereço", "endereco", "telefone",
            "whatsapp", "horário", "horario",
            "funcionamento",
        ]
    ):
        return (
            "A loja fica na Rua 14 de Maio, 3200 — Centro, "
            "Campo Grande/MS, CEP 79202-333. "
            "Telefone/WhatsApp: (67) 3341-4444. "
            "Atendimento presencial: segunda a sexta, das 9h às 18h; "
            "sábado, das 9h às 13h. Fecha aos domingos e feriados "
            "(manual, página 2)."
        )

    # Accessory questions.
    accessory_terms = [
        "palheta", "cabo", "case",
        "pedal", "amplificador", "jogo de cordas",
    ]

    if any(term in q for term in accessory_terms):
        return (
            "A Empório da Música trabalha exclusivamente com "
            "instrumentos musicais e não comercializa acessórios "
            "como esse (manual, página 2). Posso ajudar a encontrar "
            "um instrumento!"
        )

    # Product categories.
    categories = {
        "viol": "Violões",
        "guitarr": "Guitarras",
        "baixo": "Baixos",
        "bateri": "Baterias",
        "percuss": "Baterias",
        "teclad": "Teclados",
        "piano": "Teclados",
        "ukulele": "Ukuleles",
        "violino": "Cordas Orquestrais",
        "sopro": "Instrumentos de Sopro",
    }

    selected_category = next(
        (
            value
            for key, value in categories.items()
            if key in q
        ),
        None,
    )

    if selected_category:
        max_price = _extract_max_price(message)

        asks_out_of_stock = any(
            term in q
            for term in [
                "sem estoque",
                "não tem estoque",
                "nao tem estoque",
                "indisponível",
                "indisponivel",
                "esgotado",
                "esgotados",
                "sem disponibilidade",
            ]
        )

        if asks_out_of_stock:
            products = search_products(
                category=selected_category,
                in_stock_only=False,
                out_of_stock_only=True,
            )
        else:
            products = search_products(
    category=selected_category,
    max_price=max_price,
    in_stock_only=True,
)

        if not products:
            qualifier = (
                f" com preço até {_format_brl(max_price)}"
                if max_price is not None
                else ""
            )

            if asks_out_of_stock:
                return (
                    f"Não encontrei {selected_category.lower()} "
                    "ativos sem estoque no catálogo fornecido. "
                    "Consulte a loja para confirmar possíveis atualizações."
                )

            return (
                f"Não encontrei {selected_category.lower()} "
                f"disponíveis{qualifier} no catálogo."
            )

        lines = [
            f"- **{item['product_name']}** — "
            f"{_format_brl(item['price_brl'])} "
            f"(estoque informado: "
            f"{int(item['stock_quantity'])} unidade(s))"
            for item in products
        ]

        qualifier = (
            f" com preço até {_format_brl(max_price)}"
            if max_price is not None
            else ""
        )

        heading = (
            f"Estes {selected_category.lower()} estão sem estoque"
            if asks_out_of_stock
            else (
                f"Temos estas opções de "
                f"{selected_category.lower()}{qualifier} disponíveis"
            )
        )

        return (
            heading
            + " no catálogo:\n\n"
            + "\n".join(lines)
            + "\n\nValores e quantidades conforme os arquivos "
            "fornecidos para este desafio."
        )

    # List all available categories.
    if any(
        term in q
        for term in [
            "quais instrumentos",
            "que instrumentos",
            "o que a loja vende",
            "o que vocês vendem",
            "o que voces vendem",
            "o que tem na loja",
            "quais produtos",
            "catálogo",
            "catalogo",
            "disponíveis na loja",
            "disponiveis na loja",
        ]
    ):
        categories_list = list_available_categories()

        if not categories_list:
            return (
                "Não encontrei produtos ativos com estoque "
                "no catálogo fornecido."
            )

        lines = [
            f"- **{item['category']}** — "
            f"{item['available_count']} produto(s) disponível(is). "
            f"Exemplos: {', '.join(item['examples'])}."
            for item in categories_list
        ]

        return (
            "Temos estas categorias com produtos ativos em estoque "
            "no catálogo:\n\n"
            + "\n".join(lines)
            + "\n\nPosso mostrar os modelos e preços de qualquer "
            "categoria. As informações refletem os arquivos "
            "fornecidos para este desafio."
        )

    # Generic product questions.
    if any(
        term in q
        for term in [
            "preço", "preco", "quanto custa",
            "produto", "instrumento",
        ]
    ):
        return (
            "Posso pesquisar nosso catálogo. Qual instrumento ou "
            "modelo você procura? Também posso filtrar por categoria "
            "e faixa de preço."
        )

    return (
        "Claro! Posso ajudar com instrumentos, preços, disponibilidade, "
        "promoções, pedidos e políticas da loja. O que você gostaria "
        "de saber?"
    )


def answer(
    message: str,
    order_id: int | None = None,
    customer_email: str | None = None,
) -> str:

    if order_id is not None and customer_email:
        result = get_order_status(order_id, customer_email)

        if result:
            return f"Encontrei seu pedido: {result}"

        return (
            "Não consegui validar os dados do pedido. Confira o número "
            "e o e-mail cadastrado ou fale com nossa equipe."
        )

    if USE_LLM:
        try:
            from .ollama_agent import answer_with_ollama
            return answer_with_ollama(message)
        except (RequestException, ValueError, TypeError, KeyError):
            # Keep the deterministic baseline available if Ollama is offline.
            return answer_without_llm(message)

    return answer_without_llm(message)
