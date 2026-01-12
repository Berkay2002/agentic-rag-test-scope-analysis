from agrag.data.ingestion import DataIngestion


def test_infer_entity_type_v1_prefixes() -> None:
    assert DataIngestion._infer_entity_type("CR_HANDOVER_001") == "ChangeRequest"
    assert DataIngestion._infer_entity_type("FILE_src_network_handover_py") == "File"
    assert DataIngestion._infer_entity_type("COMP_NETWORK") == "Component"
