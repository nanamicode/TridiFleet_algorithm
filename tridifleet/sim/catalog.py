from __future__ import annotations

import random

from ..models import Ad


CREATIVE_LIBRARY = [
    ("Pão Quentinho da Praça", "padaria", ["food", "breakfast", "family", "local"]),
    ("Café Aurora", "cafeteria", ["coffee", "food", "morning", "work"]),
    ("Mercado BomVizinho", "supermercado", ["food", "family", "home", "local"]),
    ("Corte & Barba 21", "barbearia", ["beauty", "services", "fashion", "men"]),
    ("Studio Bela", "salao", ["beauty", "services", "fashion", "women"]),
    ("Farmácia Vida", "farmacia", ["health", "family", "services"]),
    ("Academia Pulso", "academia", ["fitness", "health", "young"]),
    ("Pizzaria Forno Alto", "pizzaria", ["food", "family", "evening"]),
    ("Restaurante Sabor Urbano", "restaurante", ["food", "family", "lunch"]),
    ("PetMais", "petshop", ["pets", "family", "services"]),
    ("Odonto Prime", "dentista", ["health", "family", "services"]),
    ("Veste Rua", "moda", ["fashion", "young", "shopping"]),
    ("TechBox", "eletronicos", ["technology", "shopping", "young"]),
    ("Colégio Horizonte", "educacao", ["education", "family", "children"]),
    ("Auto Center Norte", "automotivo", ["cars", "services", "local"]),
    ("Imobiliária Morar", "imoveis", ["home", "finance", "family"]),
    ("Entrega Já", "delivery", ["food", "services", "technology"]),
    ("Cosméticos Lumi", "cosmeticos", ["beauty", "fashion", "women"]),
    ("Lavanderia Nuvem", "lavanderia", ["home", "services", "family"]),
    ("Casa Clara Móveis", "moveis", ["home", "shopping", "family"]),
    ("Gelato Centro", "sorveteria", ["food", "family", "young"]),
    ("Ótica Vista", "otica", ["health", "fashion", "services"]),
    ("Viaje Leve", "turismo", ["travel", "finance", "family"]),
    ("Banco Ponto", "financeiro", ["finance", "services", "work"]),
    ("Curso Código Local", "educacao", ["education", "technology", "young"]),
    ("Clínica Equilíbrio", "saude", ["health", "services", "family"]),
    ("Loja Mundo Kids", "infantil", ["family", "children", "shopping"]),
    ("Hambúrguer 77", "lanchonete", ["food", "young", "evening"]),
    ("Super Sabão", "limpeza", ["home", "family", "supermarket", "hygiene"]),
    ("Cinema Bairro", "entretenimento", ["entertainment", "young", "family", "evening"]),
]


def default_creatives(seed: int) -> list[Ad]:
    rng = random.Random(seed + 303)
    ads: list[Ad] = []
    for i, (name, category, tags) in enumerate(CREATIVE_LIBRARY, start=1):
        ads.append(
            Ad(
                ad_id=f"AD{i:03d}",
                name=name,
                category=category,
                tags=tags,
                duration_seconds=rng.choice([8.0, 10.0, 12.0, 15.0]),
                daily_budget=round(rng.uniform(90, 650), 2),
                cost_per_play=round(rng.uniform(0.035, 0.095), 3),
            )
        )
    return ads
