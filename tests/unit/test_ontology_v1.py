from agrag.kg.ontology import NodeLabel, RelationshipType


def test_v1_labels_and_relationships() -> None:
    assert NodeLabel.CHANGE_REQUEST.value == "ChangeRequest"
    assert NodeLabel.FILE.value == "File"
    assert NodeLabel.COMPONENT.value == "Component"
    assert RelationshipType.TOUCHES.value == "TOUCHES"
    assert RelationshipType.PART_OF.value == "PART_OF"
