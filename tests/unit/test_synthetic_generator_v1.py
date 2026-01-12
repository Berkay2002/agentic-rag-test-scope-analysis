import random

from agrag.data.generators import synthetic as synthetic_module
from agrag.data.generators.synthetic import TelecomDataGenerator


class _StubEmbeddingService:
    def embed_query(self, text: str):
        return [0.0] * 768


def test_generator_emits_v1_entities_and_edges(monkeypatch) -> None:
    monkeypatch.setattr(
        synthetic_module,
        "get_embedding_service",
        lambda: _StubEmbeddingService(),
    )
    random.seed(0)

    gen = TelecomDataGenerator()
    dataset = gen.generate_full_dataset(requirement_count=3, testcase_count=5)

    ids = [entity["id"] for entity in dataset["entities"]]
    assert any(i.startswith("CR_") for i in ids)
    assert any(i.startswith("FILE_") for i in ids)
    assert any(i.startswith("COMP_") for i in ids)

    rel_types = {rel["relationship_type"] for rel in dataset["relationships"]}
    assert "TOUCHES" in rel_types
    assert "PART_OF" in rel_types
    assert "DEFINED_IN" in rel_types
