"""Inspect current contents of Postgres + Neo4j.

This script is intended for manual validation/debugging.
It prints high-level counts and small samples without exposing secrets.

Run:
  poetry run python scripts/inspect_dbs.py
"""

from __future__ import annotations

from typing import Any

from agrag.storage import Neo4jClient, PostgresClient


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _safe_str(value: Any, max_len: int = 160) -> str:
    text = str(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def inspect_postgres() -> None:
    _print_section("PostgreSQL: document_chunks")

    client = PostgresClient()
    try:
        client.connect()
        if client.conn is None:
            print("Postgres: not connected (conn is None)")
            return

        with client.conn.cursor() as cur:
            cur.execute(
                """
                SELECT format_type(a.atttypid, a.atttypmod) AS column_type
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = 'public'
                  AND c.relname = 'document_chunks'
                  AND a.attname = 'embedding'
                """
            )
            type_row = cur.fetchone()
            if type_row and type_row.get("column_type"):
                print(f"Embedding column type: {type_row['column_type']}")

            try:
                cur.execute(
                    """
                    SELECT vector_dims(embedding) AS dims
                    FROM document_chunks
                    WHERE embedding IS NOT NULL
                    LIMIT 5
                    """
                )
                dims_rows = cur.fetchall() or []
                if dims_rows:
                    dims_list = [row["dims"] for row in dims_rows if row.get("dims") is not None]
                    print(f"Sample embedding dims: {dims_list}")
            except Exception as exc:
                print(f"Embedding dimension check failed: {_safe_str(exc)}")

            cur.execute("SELECT COUNT(*) AS count FROM document_chunks")
            total = cur.fetchone()
            print(f"Total rows: {total['count'] if total else 'unknown'}")

            # Breakdown by entity_type (if present)
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
            rows = cur.fetchall() or []
            print("\nBy metadata.entity_type (top 20):")
            for row in rows:
                print(f"- {row['entity_type']}: {row['count']}")

            # Small sample of chunk IDs
            cur.execute(
                """
                SELECT chunk_id, LEFT(content, 120) AS content_preview
                FROM document_chunks
                ORDER BY updated_at DESC
                LIMIT 10
                """
            )
            sample = cur.fetchall() or []
            print("\nMost recently updated chunks (up to 10):")
            for row in sample:
                print(f"- {row['chunk_id']}: {_safe_str(row['content_preview'])}")

    except Exception as exc:
        print(f"Postgres inspection failed: {_safe_str(exc)}")
    finally:
        client.close()


def inspect_neo4j() -> None:
    _print_section("Neo4j: nodes + relationships")

    client = Neo4jClient()
    try:
        if not client.verify_connectivity():
            print("Neo4j: connectivity check failed")
            return

        with client.driver.session(database=client.database) as session:
            result = session.run(
                """
                MATCH (n)
                RETURN labels(n)[0] AS label, count(*) AS count
                ORDER BY count DESC
                """
            )
            rows = list(result)
            print("Node counts by label:")
            for record in rows:
                print(f"- {record['label']}: {record['count']}")

            result = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(*) AS count
                ORDER BY count DESC
                """
            )
            rows = list(result)
            print("\nRelationship counts by type:")
            for record in rows:
                print(f"- {record['type']}: {record['count']}")

            result = session.run(
                """
                MATCH (n)
                RETURN labels(n)[0] AS label, n.id AS id
                ORDER BY label, id
                LIMIT 20
                """
            )
            rows = list(result)
            print("\nSample node IDs (up to 20):")
            for record in rows:
                print(f"- {record['label']}: {record.get('id')}")

    except Exception as exc:
        print(f"Neo4j inspection failed: {_safe_str(exc)}")
    finally:
        client.close()


def main() -> None:
    inspect_postgres()
    inspect_neo4j()


if __name__ == "__main__":
    main()
