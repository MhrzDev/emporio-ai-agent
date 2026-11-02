
import re
import unicodedata

from pypdf import PdfReader

from .config import DATA_DIR


def _normalize(text: str) -> str:
    """Normalize accents, case and punctuation."""
    text = unicodedata.normalize("NFKD", str(text).casefold())
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def load_policy_pages() -> list[dict]:
    """Load policy text page-by-page to preserve page references."""
    path = DATA_DIR / "store_policies.pdf"
    reader = PdfReader(str(path))

    return [
        {
            "page": number,
            "text": page.extract_text() or "",
        }
        for number, page in enumerate(reader.pages, start=1)
    ]


def _expand_terms(query: str) -> set[str]:
    """Expand customer wording with common policy synonyms."""
    normalized = _normalize(query)

    terms = {
        word
        for word in normalized.split()
        if len(word) > 2
    }

    synonym_groups = (
        {
            "arrependimento", "arrependi", "arrependido",
            "desisti", "desistir", "cancelamento", "cancelar",
        },
        {
            "devolucao", "devolver", "devolvido", "retorno",
        },
        {
            "online", "internet", "site", "compra",
        },
        {
            "defeito", "falha", "problema", "fabricacao",
        },
        {
            "garantia", "fabricante", "legal",
        },
        {
            "troca", "trocar", "substituicao", "substituir",
        },
        {
            "preferencia", "cor", "modelo", "gosto",
        },
        {
            "recebimento", "recebi", "entrega", "entregue",
        },
        {
            "embalagem", "acessorios", "manuais", "uso",
        },
        {
            "frete", "envio", "reembolso", "estorno",
        },
    )

    for group in synonym_groups:
        if terms.intersection(group):
            terms.update(group)

    # Preserve numeric terms such as 7, 30 and 90 days.
    terms.update(re.findall(r"\b\d+\b", normalized))

    return terms


def search_policies(
    query: str,
    limit: int = 3,
) -> list[dict]:
    """
    Transparent lexical retrieval with synonym expansion.

    Returns policy excerpts with page references and relevance scores.
    This is keyword retrieval, not vector or semantic search.
    """
    if not isinstance(query, str) or not query.strip():
        return []

    terms = _expand_terms(query)
    scored = []

    for item in load_policy_pages():
        normalized_text = _normalize(item["text"])
        words = set(normalized_text.split())

        score = sum(
            1
            for term in terms
            if term in words or term in normalized_text
        )

        if score:
            scored.append((score, item))

    scored.sort(
        key=lambda pair: pair[0],
        reverse=True,
    )

    safe_limit = max(1, min(int(limit), 10))

    return [
        {
            "page": item["page"],
            "text": item["text"][:2200],
            "score": score,
        }
        for score, item in scored[:safe_limit]
    ]
