from agrag.cli.app import main as cli_main


def test_parse_result_ids_includes_v1_entities() -> None:
    sample = (
        "1. ID: CR_HANDOVER_001 (Score: 0.9)\n"
        "2. ID: FILE_src_network_handover_py (Score: 0.8)\n"
        "3. ID: COMP_NETWORK (Score: 0.7)"
    )
    ids = cli_main._parse_result_ids(sample)
    assert "CR_HANDOVER_001" in ids
    assert "FILE_src_network_handover_py" in ids
    assert "COMP_NETWORK" in ids
