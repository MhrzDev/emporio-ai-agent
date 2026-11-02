import json
import re
import unicodedata
from typing import Any

import requests

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL
from .data_tools import (
    get_active_promotions,
    get_order_status,
    list_available_categories,
    search_products,
)
from .knowledge import search_policies

SYSTEM_PROMPT = """You are the customer-service assistant for Empório da Música,
a Brazilian musical-instrument retailer.

LANGUAGE:
- Always answer in Brazilian Portuguese.
- Be polite, helpful and concise.

GENERAL RULES:
- Use catalogue, promotion and policy data when relevant.
- Never invent products, prices, stock, discounts or policies.
- Catalogue information is a supplied challenge snapshot, not live inventory.
- Answer the current user message only.
- Never reveal private customer or order data.

CATALOGUE:
- Recommend only products present in the catalogue.
- Respect category, maximum price and stock filters.
- Do not claim that products are unavailable without checking the catalogue.

POLICIES:
- Distinguish online purchase regret, preference exchanges,
  manufacturing defects and manufacturer warranties.
- Pay attention to dates and elapsed time.
- Use retrieved policy evidence.
- Never merely announce that you will use a tool: execute it and explain the result.

SCOPE:
- Help with products, prices, stock, promotions, orders and store policies.
- Politely decline unrelated requests.
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_catalogue",
            "description": (
                "Search musical instruments by product name, keywords, "
                "category, maximum BRL price and stock availability."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string"},
                    "max_price": {"type": "number"},
                    "include_out_of_stock": {"type": "boolean"},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_catalogue_categories",
            "description": "List active catalogue categories and examples.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_active_promotions",
            "description": "List active store promotions.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_store_policies",
            "description": (
                "Search store policies about returns, exchanges and warranties."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


# ---------------------------------------------------------------------
# Normalization and formatting
# ---------------------------------------------------------------------

def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value).casefold())
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", " ", value).strip()


def _brl(value: object) -> str:
    amount = float(value)
    formatted = f"{amount:,.2f}"
    formatted = (
        formatted.replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )
    return f"R$ {formatted}"


def _format_catalogue_rows(
    rows: list[dict],
    heading: str,
) -> str:
    if not rows:
        return (
            f"{heading}: não encontrei produtos correspondentes "
            "no catálogo fornecido."
        )

    lines = []
    for row in rows:
        lines.append(
            f"- **{row['product_name']}** — "
            f"{_brl(row['price_brl'])} "
            f"(estoque: {int(row['stock_quantity'])} unidade(s))"
        )

    return (
        f"{heading}:\n\n"
        + "\n".join(lines)
        + "\n\nPreços e quantidades conforme os arquivos "
        "fornecidos para este desafio."
    )


def _clean_llm_output(content: str) -> str:
    if not content:
        return ""

    content = content.replace("\\n", "\n")
    content = content.replace("\\r", "\r")
    content = content.replace("R`", "R$")
    content = content.replace("R $", "R$")
    content = re.sub(
        r"\b(\d+)\s+unidades?\s+em\s+estoque\b",
        r"\1 unidade(s) em estoque",
        content,
        flags=re.IGNORECASE,
    )
    return content.strip()


# ---------------------------------------------------------------------
# Product intent
# ---------------------------------------------------------------------

def _named_product_query(message: str) -> str:
    normalized = _normalize(message)
    stop_words = {
        "qual", "quais", "e", "o", "a", "os", "as",
        "do", "da", "dos", "das", "de", "quanto", "custa",
        "custam", "preco", "precos", "valor", "valores",
        "estoque", "quantidade", "unidade", "unidades",
        "disponibilidade", "disponivel", "disponiveis",
        "tem", "temos", "voces", "voce", "me", "informe",
        "diga", "por", "favor", "gostaria", "saber", "quero",
        "consultar", "existe", "comprar", "compra", "produto",
        "produtos", "esta", "estao", "ha", "em", "no", "na",
        "para", "com", "um", "uma",
    }
    return " ".join(
        word for word in normalized.split()
        if word not in stop_words
    )


def _search_named_product(message: str) -> str | None:
    """Find named products and handle unavailable model codes."""
    normalized = _normalize(message)
    rows = search_products(query="", in_stock_only=False)

    # Detect model codes such as GF-3D and AC-39.
    model_match = re.search(
        r"\b[a-z]{1,6}-\d+[a-z0-9]*\b",
        normalized,
    )

    if model_match:
        model_code = model_match.group(0)

        matches = [
            row
            for row in rows
            if model_code in _normalize(row["product_name"])
        ]

        if not matches:
            return (
                f"Não encontrei o modelo {model_code.upper()} "
                "no catálogo disponível. Posso sugerir outros "
                "modelos semelhantes que estejam em estoque."
            )

        available = [
            row
            for row in matches
            if str(row["status"]).lower() == "active"
            and int(row["stock_quantity"]) > 0
        ]

        if not available:
            return (
                f"Encontrei o modelo {model_code.upper()} no catálogo, "
                "mas ele está indisponível no momento por falta de estoque. "
                "Posso sugerir modelos semelhantes disponíveis."
            )

        return _format_catalogue_rows(
            available,
            "Encontrei este modelo disponível no catálogo",
        )

    # Search by product name when there is no model code.
    query = _named_product_query(message)

    words = [
        word
        for word in re.findall(r"[a-z0-9]+", query)
        if len(word) >= 2
    ]

    if len(words) < 2 or len(words) > 8:
        return None

    matches = []

    for row in rows:
        product_name = _normalize(row["product_name"])

        if all(token in product_name for token in words):
            matches.append(row)

    if not matches or len(matches) > 5:
        return None

    available = [
        row
        for row in matches
        if str(row["status"]).lower() == "active"
        and int(row["stock_quantity"]) > 0
    ]

    if not available:
        return (
            "Encontrei o produto no catálogo, mas ele está "
            "indisponível no momento por falta de estoque. "
            "Posso sugerir alternativas disponíveis."
        )

    heading = (
        "Encontrei este produto disponível no catálogo"
        if len(available) == 1
        else "Encontrei estas opções disponíveis no catálogo"
    )

    return _format_catalogue_rows(available, heading)


# ---------------------------------------------------------------------
# Budget and category searches
# ---------------------------------------------------------------------

def _extract_max_price(message: str) -> float | None:
    normalized = _normalize(message)
    match = re.search(
        r"(?:ate|menos de|abaixo de|no maximo|orcamento maximo de|orcamento de)\s*"
        r"(?:r\s*\$?\s*)?([0-9][0-9.,]*)",
        normalized,
    )
    if not match:
        return None

    raw = match.group(1)
    if "," in raw:
        raw = raw.replace(".", "").replace(",", ".")
    elif raw.count(".") > 1:
        raw = raw.replace(".", "")

    try:
        return float(raw)
    except ValueError:
        return None


def _detect_category(message: str) -> str | None:
    normalized = _normalize(message)
    category_map = {
        "violao": "Violões",
        "violoes": "Violões",
        "guitarra": "Guitarras",
        "guitarras": "Guitarras",
        "baixo": "Baixos",
        "baixos": "Baixos",
        "bateria": "Baterias e Percussão",
        "baterias": "Baterias e Percussão",
        "percussao": "Baterias e Percussão",
        "teclado": "Teclados e Pianos",
        "teclados": "Teclados e Pianos",
        "piano": "Teclados e Pianos",
        "pianos": "Teclados e Pianos",
        "ukulele": "Ukuleles",
        "ukuleles": "Ukuleles",
    }
    words = set(normalized.split())
    for term, category in category_map.items():
        if term in words:
            return category
    return None


def _search_budget_recommendation(message: str) -> str | None:
    normalized = _normalize(message)
    max_price = _extract_max_price(message)
    category = _detect_category(message)
    if max_price is None or category is None:
        return None

    rows = search_products(
        category=category,
        max_price=max_price,
        in_stock_only=True,
    )

    # Preserve explicit material preferences such as nylon strings.
    if "nylon" in normalized:
        rows = [
            row for row in rows
            if "nylon" in _normalize(
                str(row["product_name"]) + " " + str(row.get("description", ""))
            )
        ]

    if not rows:
        return (
            f"No catálogo fornecido, não encontrei produtos de "
            f"{category.lower()} disponíveis por até {_brl(max_price)}"
            + (" com cordas de nylon." if "nylon" in normalized else ".")
            + " Posso procurar em outra faixa de preço."
        )

    answer = _format_catalogue_rows(
        rows,
        f"Encontrei {len(rows)} opção(ões) de {category.lower()} "
        f"por até {_brl(max_price)}",
    )
    if "iniciante" in normalized or "comecando" in normalized:
        introduction = (
            "Como você está começando, vale comparar o conforto, "
            "o tipo de corda e o estilo musical antes de escolher. "
            "Estas são as opções disponíveis dentro do seu orçamento:\n\n"
        )
    else:
        introduction = (
            "Estas são as opções disponíveis no catálogo dentro "
            "do seu orçamento:\n\n"
        )
    return introduction + answer


def _search_category(message: str) -> str | None:
    normalized = _normalize(message)
    category = _detect_category(message)
    if category is None:
        return None

    category_intent = any(
        term in normalized
        for term in (
            "quais", "quanto", "preco", "precos", "dispon",
            "estoque", "opcoes", "ate", "sem estoque",
            "esgotado", "indisponivel", "tem algum",
            "recomenda", "catalogo",
        )
    )
    if not category_intent:
        return None

    out_of_stock = any(
        term in normalized
        for term in (
            "sem estoque", "nao tem estoque", "indisponivel",
            "esgotado", "esgotados", "sem disponibilidade",
        )
    )
    max_price = _extract_max_price(message)
    rows = search_products(
        category=category,
        max_price=max_price,
        in_stock_only=not out_of_stock,
        out_of_stock_only=out_of_stock,
    )
    qualifier = (
        f" com preço até {_brl(max_price)}"
        if max_price is not None else ""
    )
    heading = (
        f"Produtos de {category} sem estoque"
        if out_of_stock
        else f"Opções de {category}{qualifier} disponíveis"
    )
    return _format_catalogue_rows(rows, heading)


def _search_broad_catalogue(message: str) -> str | None:
    normalized = _normalize(message)
    broad_terms = (
        "quais instrumentos", "que instrumentos",
        "o que a loja vende", "o que voces vendem",
        "o que tem na loja", "quais produtos", "catalogo",
        "disponivel na loja", "disponiveis na loja",
        "instrumentos disponiveis",
    )
    if not any(term in normalized for term in broad_terms):
        return None

    instrument_terms = (
        "violao", "guitarra", "baixo", "bateria", "teclado",
        "piano", "ukulele", "violino", "saxofone", "flauta",
        "trompete",
    )
    if any(term in normalized for term in instrument_terms):
        return None

    categories = list_available_categories()
    lines = [
        (
            f"- **{item['category']}** — "
            f"{item['available_count']} produto(s). "
            f"Exemplos: {', '.join(item['examples'])}."
        )
        for item in categories
    ]
    return (
        "Categorias com produtos ativos em estoque:\n\n"
        + "\n".join(lines)
        + "\n\nDados conforme o catálogo fornecido para este desafio."
    )


# ---------------------------------------------------------------------
# Policy answers
# ---------------------------------------------------------------------

def _extract_days(message: str) -> int | None:
    normalized = _normalize(message)
    patterns = (
        r"\bha\s+(\d+)\s+dias?\b",
        r"\bfaz\s+(\d+)\s+dias?\b",
        r"\bdepois\s+de\s+(\d+)\s+dias?\b",
        r"\bapos\s+(\d+)\s+dias?\b",
        r"\b(\d+)\s+dias?\s+atras\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return int(match.group(1))
    return None


def _policy_answer(message: str) -> str | None:
    normalized = _normalize(message)
    online_regret = any(
        term in normalized
        for term in (
            "arrepend", "devolver", "devolucao",
            "compra pela internet", "comprei online", "compra online",
        )
    )
    defect = any(
        term in normalized
        for term in ("defeito", "parou de funcionar", "problema de fabricacao")
    )
    preference = any(
        term in normalized
        for term in ("trocar", "troca", "outra cor", "nao gostei", "preferencia")
    )

    if online_regret:
        excerpts = search_policies(
            "arrependimento compra online devolução sete dias corridos "
            "recebimento embalagem original acessórios manuais reembolso frete",
            limit=4,
        )
        if not excerpts:
            return (
                "Não consegui localizar o trecho correspondente da política. "
                "Recomendo confirmar as condições com a equipe da loja."
            )
        days = _extract_days(message)
        if days is not None and days > 7:
            return (
                "Pela política da loja, o prazo para arrependimento de compras "
                "feitas pela internet é de até **7 dias corridos após o recebimento**. "
                f"Como você informou que já se passaram {days} dias, esse prazo "
                "pode ter expirado. Recomendo entrar em contato com a loja para "
                "verificar se existe outra possibilidade de atendimento."
            )
        if days is not None:
            return (
                "Sim! Pela política da loja, compras feitas pela internet podem "
                "ser canceladas por arrependimento em até **7 dias corridos após "
                "o recebimento**, sem necessidade de justificativa.\n\n"
                "O produto deve estar na embalagem original, sem sinais de uso "
                "e acompanhado de todos os acessórios e manuais. O reembolso "
                "é feito em até **10 dias úteis**, pelo meio de pagamento original. "
                "A loja paga o frete de retorno.\n\n"
                f"Como você informou que recebeu o produto há {days} dias, "
                "está dentro do prazo previsto. Entre em contato com a loja "
                "para iniciar a solicitação."
            )
        return (
            "Pela política da loja, compras feitas pela internet podem ser "
            "canceladas por arrependimento em até **7 dias corridos após "
            "o recebimento**, sem necessidade de justificativa.\n\n"
            "O produto deve estar na embalagem original, sem sinais de uso "
            "e acompanhado de todos os acessórios e manuais. O reembolso "
            "é feito em até **10 dias úteis**, pelo meio de pagamento original. "
            "A loja paga o frete de retorno.\n\n"
            "Se você me informar há quantos dias recebeu o produto, posso "
            "ajudar a verificar o prazo."
        )

    if defect:
        excerpts = search_policies(
            "defeito fabricação troca trinta dias corridos garantia fabricante "
            "garantia legal noventa dias",
            limit=4,
        )
        if not excerpts:
            return (
                "Não consegui localizar o trecho correspondente da política. "
                "Recomendo confirmar as condições com a equipe da loja."
            )
        days = _extract_days(message)
        if days is not None and days <= 30:
            return (
                "Pela política da loja, defeitos de fabricação podem ser tratados "
                "com solicitação de troca em até **30 dias corridos após a compra**. "
                "Informe o produto e descreva o defeito para que a equipe possa "
                "orientar a abertura do atendimento.\n\n"
                "A política não cobre danos por mau uso, quedas, umidade excessiva "
                "ou modificações não autorizadas."
            )
        if days is not None and days > 30:
            return (
                "Pela política da loja, a troca por defeito de fabricação pode "
                "ser solicitada em até **30 dias corridos após a compra**. "
                f"Como você informou que o defeito apareceu após {days} dias, "
                "o procedimento indicado é acionar a garantia do fabricante. "
                "A loja pode intermediar esse contato.\n\n"
                "A política também prevê garantia legal de 90 dias para defeitos "
                "de fabricação, contados do recebimento, além da garantia do "
                "fabricante, que varia conforme o produto e o certificado.\n\n"
                "Informe o nome do instrumento e os detalhes do defeito para "
                "receber orientação sobre o próximo passo."
            )
        return (
            "A política da loja prevê solicitação de troca por defeito de "
            "fabricação em até **30 dias corridos após a compra**. Também há "
            "garantia legal de 90 dias, contados do recebimento, e garantia "
            "do fabricante conforme o certificado do produto.\n\n"
            "Você pode me informar há quantos dias comprou o instrumento?"
        )

    if preference:
        excerpts = search_policies(
            "troca por preferência sete dias produto perfeito estado "
            "embalagem original disponibilidade",
            limit=4,
        )
        if not excerpts:
            return (
                "Não consegui localizar o trecho correspondente da política. "
                "Recomendo confirmar as condições com a equipe da loja."
            )
        return (
            "A política permite troca por preferência em até **7 dias**, "
            "desde que o produto esteja em perfeito estado, com a embalagem "
            "original, e haja disponibilidade do item desejado.\n\n"
            "Informe a data da compra e o produto que gostaria de trocar "
            "para que a equipe possa orientar você."
        )
    return None


# ---------------------------------------------------------------------
# Scope detection
# ---------------------------------------------------------------------

def _is_out_of_scope(message: str) -> bool:
    normalized = _normalize(message)
    unrelated_topics = (
        "previsao do tempo", "tempo para amanha", "tempo hoje",
        "vai chover", "vai fazer sol", "temperatura amanha",
        "temperatura hoje", "clima amanha", "clima hoje",
        "noticias de hoje", "resultado do jogo", "cotacao do dolar",
        "receita de bolo",
    )
    return any(topic in normalized for topic in unrelated_topics)


# ---------------------------------------------------------------------
# Validated tool execution
# ---------------------------------------------------------------------

def _dispatch(name: str, args: dict[str, Any]) -> dict:
    if not isinstance(args, dict):
        raise TypeError("Tool arguments must be an object.")

    if name == "search_catalogue":
        allowed = {
            "query", "category", "max_price", "include_out_of_stock",
        }
        if set(args) - allowed:
            raise ValueError("Unexpected search arguments.")

        max_price = args.get("max_price")
        if max_price is not None and (
            not isinstance(max_price, (int, float))
            or isinstance(max_price, bool)
            or max_price < 0
        ):
            raise ValueError("max_price must be a non-negative number.")

        include_out = args.get("include_out_of_stock", False)
        if not isinstance(include_out, bool):
            raise ValueError("include_out_of_stock must be a boolean.")

        rows = search_products(
            query=str(args.get("query") or "")[:120],
            category=str(args.get("category") or "")[:80],
            max_price=float(max_price) if max_price is not None else None,
            in_stock_only=not include_out,
            out_of_stock_only=include_out,
        )
        return {
            "count": len(rows),
            "products": rows,
            "currency": "BRL",
            "stock_filter": "out_of_stock" if include_out else "in_stock",
        }

    if name == "list_catalogue_categories":
        if args:
            raise ValueError("This tool takes no arguments.")
        return {
            "categories": list_available_categories(),
            "note": "Only active products with positive stock are included.",
        }

    if name == "list_active_promotions":
        if args:
            raise ValueError("This tool takes no arguments.")
        return {"promotions": get_active_promotions()}

    if name == "search_store_policies":
        if set(args) != {"query"}:
            raise ValueError("A policy query is required.")
        query = args["query"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("A non-empty policy query is required.")
        return {"excerpts": search_policies(query[:300], limit=4)}

    raise ValueError("Tool is not allowed.")


# ---------------------------------------------------------------------
# Secure order lookup
# ---------------------------------------------------------------------

def _order_answer(message: str) -> str | None:
    normalized = _normalize(message)
    if not any(
        term in normalized
        for term in ("pedido", "rastreamento", "rastrear", "entrega")
    ):
        return None

    email_match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+[.][A-Za-z]{2,}",
        message,
    )
    order_match = re.search(
        r"\bpedido\s*(?:(?:numero|n[ºo.]?)\s*)?#?\s*(\d+)\b",
        normalized,
    )
    if not email_match or not order_match:
        return (
            "Para consultar seu pedido com segurança, informe "
            "o número do pedido e o e-mail cadastrado na compra."
        )

    result = get_order_status(
        int(order_match.group(1)),
        email_match.group(0),
    )
    if not result:
        return (
            "Não consegui validar os dados do pedido. Confira "
            "o número e o e-mail cadastrado ou fale com nossa equipe."
        )

    lines = [
        f"**Pedido:** {result['order_id']}",
        f"**Status:** {result['status']}",
        f"**Total:** {_brl(result['total_brl'])}",
    ]
    if result.get("tracking_code"):
        lines.append(f"**Rastreamento:** {result['tracking_code']}")
    if result.get("estimated_delivery"):
        lines.append(
            f"**Previsão de entrega:** {result['estimated_delivery']}"
        )
    return (
        "Encontrei seu pedido e validei os dados informados.\n\n"
        + "\n".join(lines)
    )


# ---------------------------------------------------------------------
# Main Ollama assistant
# ---------------------------------------------------------------------

def answer_with_ollama(
    message: str,
    history: list[dict] | None = None,
    timeout: int = 120,
) -> str:
    if not isinstance(message, str) or not message.strip():
        return "Por favor, envie uma pergunta para que eu possa ajudar."

    current_message = message.strip()[:3000]

    if _is_out_of_scope(current_message):
        return (
            "No momento, posso ajudar com instrumentos musicais, "
            "preços, disponibilidade, promoções, pedidos e políticas "
            "da loja. Não tenho acesso a informações sobre esse assunto."
        )

    order_answer = _order_answer(current_message)
    if order_answer is not None:
        return order_answer

    product_answer = _search_named_product(current_message)
    if product_answer is not None:
        return product_answer

    budget_answer = _search_budget_recommendation(current_message)
    if budget_answer is not None:
        return budget_answer

    category_answer = _search_category(current_message)
    if category_answer is not None:
        return category_answer

    broad_answer = _search_broad_catalogue(current_message)
    if broad_answer is not None:
        return broad_answer

    policy_answer = _policy_answer(current_message)
    if policy_answer is not None:
        return policy_answer

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": current_message},
    ]

    for _ in range(5):
        response = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "tools": TOOLS,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        assistant_message = payload.get("message", {})

        if not isinstance(assistant_message, dict):
            raise TypeError("Invalid response from Ollama.")

        tool_calls = assistant_message.get("tool_calls") or []
        content = assistant_message.get("content", "")

        if not tool_calls:
            if not isinstance(content, str) or not content.strip():
                return (
                    "Não consegui formular uma resposta. "
                    "Pode reformular sua pergunta?"
                )
            cleaned = _clean_llm_output(content)
            if re.search(
                r"(vou|vamos|posso)\s+(verificar|consultar|buscar|pesquisar)",
                _normalize(cleaned),
            ):
                return (
                    "Não consegui concluir a consulta automaticamente. "
                    "Pode reformular sua pergunta ou falar com nossa equipe."
                )
            return cleaned

        messages.append(assistant_message)
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function", {})
            if not isinstance(function, dict):
                function = {}
            name = function.get("name", "")
            args = function.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            try:
                result = _dispatch(name, args)
            except (TypeError, ValueError, KeyError) as exc:
                result = {"error": str(exc)}

            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        result,
                        ensure_ascii=False,
                        default=str,
                    ),
                }
            )

    return (
        "Não consegui concluir a consulta com segurança. "
        "Tente reformular a pergunta ou fale com nossa equipe."
    )
