from yolop_postgres_runtime import PostgresRuntimeStore


def test_public_store_import() -> None:
    assert PostgresRuntimeStore.__name__ == "PostgresRuntimeStore"


def test_store_construction_does_not_open_a_connection_pool() -> None:
    store = PostgresRuntimeStore("postgresql://unused")

    assert store.pool.closed is True
