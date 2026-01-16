"""Inspect entity_id coverage and distribution in document_chunks.

Run:
  poetry run python scripts/inspect_entity_ids.py
"""

from __future__ import annotations

from typing import Iterable

from agrag.storage import PostgresClient


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _iter_ids() -> Iterable[str]:
    return ["TC_MOBILITY_001", "TC_DATA_002"]


def main() -> None:
    client = PostgresClient()
    try:
        client.connect()
        if client.conn is None:
            print("Postgres: not connected")
            return

        _print_section("Missing entity_id count")
        count = None
        try:
            with client.conn.cursor() as cur:
                try:
                    cur.execute("SET enable_indexscan = off")
                    cur.execute("SET enable_bitmapscan = off")
                except Exception:
                    pass
                cur.execute(
                    "SELECT COUNT(*) AS count FROM document_chunks WHERE metadata->>'entity_id' IS NULL"
                )
                row = cur.fetchone() or {}
                count = row.get("count")
        except Exception:
            client.conn.rollback()
            try:
                with client.conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COUNT(*) AS count
                        FROM document_chunks
                        WHERE metadata IS NULL OR metadata::text NOT LIKE '%"entity_id"%'
                        """
                    )
                    row = cur.fetchone() or {}
                    count = row.get("count")
            except Exception as exc:
                client.conn.rollback()
                print(f"Missing entity_id count failed: {exc}")

        print(f"Missing entity_id: {count if count is not None else 'unknown'}")

        _print_section("Entity type distribution (top 20)")
        try:
            with client.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COALESCE(metadata->>'entity_type', '(missing)') AS entity_type,
                        COUNT(*) AS count
                    FROM document_chunks
                    GROUP BY 1
                    ORDER BY count DESC
                    LIMIT 20
                    """
                )
                for row in cur.fetchall() or []:
                    print(f"- {row.get('entity_type')}: {row.get('count')}")
        except Exception as exc:
            client.conn.rollback()
            print(f"Entity type distribution failed: {exc}")

        _print_section("Sample rows for key entity IDs")
        for entity_id in _iter_ids():
            try:
                with client.conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT chunk_id, content, metadata
                        FROM document_chunks
                        WHERE metadata->>'entity_id' = %s
                        LIMIT 5
                        """,
                        [entity_id],
                    )
                    rows = cur.fetchall() or []
                print(f"\n== {entity_id} rows: {len(rows)}")
                for row in rows:
                    content = (row.get("content") or "")[:120]
                    print(f"- {row.get('chunk_id')}: {content}")
            except Exception as exc:
                client.conn.rollback()
                print(f"Sample rows failed for {entity_id}: {exc}")

    except Exception as exc:
        print(f"Entity ID inspection failed: {exc}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
