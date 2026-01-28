from agrag.evaluation.utils.entity_extractor import extract_entity_ids, extract_entity_ids_detailed


def test_extracts_v1_entity_ids() -> None:
    text = "CR_HANDOVER_001 touches FILE_src_network_handover_py in COMP_NETWORK"
    ids = extract_entity_ids(text, prioritize_test_cases=False)
    assert "CR_HANDOVER_001" in ids
    assert "FILE_src_network_handover_py" in ids
    assert "COMP_NETWORK" in ids

    detailed = extract_entity_ids_detailed(text)
    assert detailed.by_type["ChangeRequest"] == ["CR_HANDOVER_001"]
    assert detailed.by_type["File"] == ["FILE_src_network_handover_py"]
    assert detailed.by_type["Component"] == ["COMP_NETWORK"]
