from __future__ import annotations

import re
import unicodedata


ALIASES = {
    "padaria": "food",
    "bakery": "food",
    "comida": "food",
    "alimentacao": "food",
    "restaurante": "food",
    "lanchonete": "food",
    "pizzaria": "food",
    "sorveteria": "food",
    "cafe": "coffee",
    "cafeteria": "coffee",
    "supermercado": "grocery",
    "mercado": "grocery",
    "supermarket": "grocery",
    "grocery": "grocery",
    "sabonete": "personal_care",
    "higiene": "personal_care",
    "hygiene": "personal_care",
    "personal_care": "personal_care",
    "cosmeticos": "beauty",
    "cosmetics": "beauty",
    "beleza": "beauty",
    "salao": "beauty",
    "cabeleireiro": "beauty",
    "barbearia": "beauty",
    "saude": "health",
    "farmacia": "health",
    "clinica": "health",
    "academia": "fitness",
    "petshop": "pets",
    "pet_shop": "pets",
    "animais": "pets",
    "familia": "family",
    "infantil": "family",
    "criancas": "children",
    "moda": "fashion",
    "roupas": "fashion",
    "eletronicos": "technology",
    "tecnologia": "technology",
    "automotivo": "cars",
    "carros": "cars",
    "imoveis": "home",
    "moveis": "home",
    "casa": "home",
    "limpeza": "home",
    "educacao": "education",
    "escola": "education",
    "curso": "education",
    "financeiro": "finance",
    "banco": "finance",
    "turismo": "travel",
    "viagem": "travel",
    "entretenimento": "entertainment",
    "cinema": "entertainment",
    "servicos": "services",
    "delivery": "services",
    "entrega": "services",
    "manha": "morning",
    "cafe_da_manha": "breakfast",
    "almoco": "lunch",
    "noite": "evening",
    "jovem": "young",
    "jovens": "young",
    "mulheres": "women",
    "homens": "men",
    "compras": "shopping",
}


def normalize_token(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.strip().lower())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def canonical_tag(value: str) -> str:
    token = normalize_token(value)
    return ALIASES.get(token, token)


def canonical_tags(values: list[str] | set[str] | tuple[str, ...]) -> set[str]:
    return {canonical_tag(v) for v in values if normalize_token(v)}
