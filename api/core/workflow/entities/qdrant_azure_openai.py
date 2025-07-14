import re
from functools import cached_property
from typing import Union

import sqlglot
from fastembed.sparse.bm25 import Bm25
from openai import AzureOpenAI, OpenAI
from qdrant_client import models
from sqlglot.expressions import CTE, Schema, Subquery, Table
from vanna.qdrant import Qdrant_VectorStore
from vanna.utils import deterministic_uuid


class QdrantAzureOpenAI(Qdrant_VectorStore):
    """
    Class for Qdrant with Text Embedding using Azure Open AI
    """

    def __init__(
        self,
        config = {},
        client: AzureOpenAI = None,
    ):
        self._client_azure_openai = client
        self.model = config.get('model_embedding', None)
        super().__init__(config=config)

    def generate_embedding(self, data: str, **kwargs) -> list[float]:
        embeddings_response = self._client_azure_openai.embeddings.create(model=self.model, input=data)

        # Extract the embedding vector from the response
        return embeddings_response.data[0].embedding


class QdrantHybridOpenAI(Qdrant_VectorStore):
    """
    Class for Qdrant with hybrid search.
    Using OpenAI or Azure OpenAI Embedding for dense vector; and fastembed for sparse vector)
    """

    def __init__(
        self,
        config = {},
        client: Union[OpenAI, AzureOpenAI] = None,
    ):
        self._client_openai = client
        self.model = config.get('model_embedding', None)
        self.sparse_model = config.get('model_sparse_embedding', Bm25('Qdrant/bm25'))
        super().__init__(config=config)

    def _embed_dense(self, data: str):
        embeddings_response = self._client_openai.embeddings.create(model=self.model, input=data)
        return embeddings_response.data[0].embedding

    def _embed_sparse(self, data: str):
        return list(self.sparse_model.passage_embed(data))[0].as_object()

    def generate_embedding(self, data: str, **kwargs) -> models.VectorStruct:
        # generate dense and sparse vector
        return {'dense': self._embed_dense(data), 'sparse': self._embed_sparse(data.lower())}

    def _hybrid_search(
        self, collection_name, query_vectors, n_dense_pre=10, n_sparse_pre=10, with_payload=True, **kwargs
    ) -> list[models.ScoredPoint]:
        enabled_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key='enabled',
                    match=models.MatchValue(value=True)
                ),
                models.IsEmptyCondition(
                    is_empty=models.PayloadField(key='enabled')
                ),
            ]
        )
        prefetch = [
            models.Prefetch(
                query=query_vectors['dense'],
                using='dense',
                filter=enabled_filter,
                limit=n_dense_pre,
            ),
            models.Prefetch(
                query=models.SparseVector(**query_vectors['sparse']),
                using='sparse',
                filter=enabled_filter,
                limit=n_sparse_pre,
            ),
        ]

        results = self._client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=with_payload,
            limit=self.n_results,
            **kwargs,
        )

        return results.points

    def _partial_search(self, collection_name, query, using='dense', with_payload=True, **kwargs):
        # search using one of the vectors only (dense or sparse)
        results = self._client.query_points(
            collection_name=collection_name,
            query=query,
            with_payload=with_payload,
            using=using,
            limit=self.n_results,
            **kwargs,
        )

        return results.points

    @cached_property
    def embeddings_dimension(self):
        return len(self._embed_dense('ABCDEF'))

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        results = self._hybrid_search(self.sql_collection_name, self.generate_embedding(question), **kwargs)

        return [dict(result.payload) for result in results]

    def get_related_ddl(self, question: str, **kwargs) -> list:
        results = self._hybrid_search(self.ddl_collection_name, self.generate_embedding(question), **kwargs)

        return [result.payload['ddl'] for result in results]

    def get_related_documentation(self, question: str, **kwargs) -> list:
        results = self._hybrid_search(self.documentation_collection_name, self.generate_embedding(question), **kwargs)

        return [result.payload['documentation'] for result in results]

    def _setup_collections(self):
        # setup for multiple vector per point
        if not self._client.collection_exists(self.sql_collection_name):
            self._client.create_collection(
                collection_name=self.sql_collection_name,
                vectors_config={
                    'dense': models.VectorParams(
                        size=self.embeddings_dimension,
                        distance=self.distance_metric,
                    )
                },
                sparse_vectors_config={'sparse': models.SparseVectorParams(modifier=models.Modifier.IDF)},
                **self.collection_params,
            )

        if not self._client.collection_exists(self.ddl_collection_name):
            self._client.create_collection(
                collection_name=self.ddl_collection_name,
                vectors_config={
                    'dense': models.VectorParams(
                        size=self.embeddings_dimension,
                        distance=self.distance_metric,
                    )
                },
                sparse_vectors_config={'sparse': models.SparseVectorParams(modifier=models.Modifier.IDF)},
                **self.collection_params,
            )
        if not self._client.collection_exists(self.documentation_collection_name):
            self._client.create_collection(
                collection_name=self.documentation_collection_name,
                vectors_config={
                    'dense': models.VectorParams(
                        size=self.embeddings_dimension,
                        distance=self.distance_metric,
                    )
                },
                sparse_vectors_config={'sparse': models.SparseVectorParams(modifier=models.Modifier.IDF)},
                **self.collection_params,
            )


class RCAQdrant(QdrantHybridOpenAI):
    qdrant_table_name_field = "table_name"

    def __init__(self, config = None, client = None):
        super().__init__(config=config or {}, client=client)
        self.table_name_pattern = re.compile(
            r'\bTABLE NAME: (?:([\'"`])?(?P<schema>\w+?)(?:\1|)\.)?([\'"`])?(?P<table_name>\w+)(?:\4|\b)',
            flags=re.IGNORECASE
        )

    def _setup_collections(self):
        super()._setup_collections()

        for collection_name in [self.sql_collection_name, self.ddl_collection_name, self.documentation_collection_name]:
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name=self.qdrant_table_name_field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def _extract_table_name_from_documentation(self, search_str) -> dict[str, str]:
        # in `schema.table_name` pattern, take the table_name and omit the schema
        m = self.table_name_pattern.search(search_str)

        return {
            self.qdrant_table_name_field: m.groupdict()['table_name'].lower(),
        } if m else {}

    @property
    def dialect_for_sqlglot(self) -> str:
        match self.dialect:
            case "Oracle SQL":
                return "oracle"
            case "PostgreSQL":
                return "postgres"
            case "SQL":
                return "mysql"
            case _:
                raise NotImplementedError(f"Unknown dialect {self.dialect}")

    def _extract_table_name_from_ddl(self, query: str) -> dict[str, str]:
        try:
            ast: sqlglot.Expression = sqlglot.parse_one(query, dialect=self.dialect_for_sqlglot)
            table_found = [_from.name.lower() for _from in ast.find_all(Table) if _from.name]
            schema_found = [_from.this.db for _from in ast.find_all(Schema) if _from.this]

        except Exception as e:
            raise ValueError(f"Invalid DDL query. {e}")

        result = {self.qdrant_table_name_field: table_found[0]} if table_found else {}
        if schema_found:
            result.update({'schema': schema_found[0]})
        return result

    def _extract_table_name_from_sql(self, query: str) -> dict[str, list[str]]:
        try:
            ast: sqlglot.Expression = sqlglot.parse_one(query, dialect=self.dialect_for_sqlglot)

            cte = (_cte.alias_or_name.lower() for _cte in ast.find_all(CTE) if _cte.alias_or_name)
            subquery = (_sub.alias.lower() for _sub in ast.find_all(Subquery) if _sub.alias)
            all_tables = (_from.name.lower() for _from in ast.find_all(Table) if _from.name)
            valid_table = list(all_tables - cte - subquery)

        except Exception as e:
            raise ValueError(f"Invalid SQL query. {e}")

        return {
            self.qdrant_table_name_field: valid_table
        } if valid_table else {}

    def add_question_sql(self, question: str, sql: str, id: str | None = None, **kwargs) -> str:
        question_answer = "Question: {}\n\nSQL: {}".format(question, sql)
        id = id or deterministic_uuid(question_answer)
        self._client.upsert(
            self.sql_collection_name,
            points=[
                models.PointStruct(
                    id=id,
                    vector=self.generate_embedding(question_answer),
                    payload={
                        "question": question,
                        "sql": sql,
                        "enabled": True,
                        **self._extract_table_name_from_sql(sql)
                    },
                )
            ],
        )

        return self._format_point_id(id, self.sql_collection_name)

    def add_ddl(self, ddl: str, id: str | None = None, **kwargs) -> str:
        id = id or deterministic_uuid(ddl)
        self._client.upsert(
            self.ddl_collection_name,
            points=[
                models.PointStruct(
                    id=id,
                    vector=self.generate_embedding(ddl),
                    payload={
                        "ddl": ddl,
                        "enabled": True,
                        **self._extract_table_name_from_ddl(ddl)
                    },
                )
            ],
        )

        return self._format_point_id(id, self.ddl_collection_name)

    def add_documentation(self, documentation: str, id: str | None = None, **kwargs) -> str:
        id = id or deterministic_uuid(documentation)
        self._client.upsert(
            self.documentation_collection_name,
            points=[
                models.PointStruct(
                    id=id,
                    vector=self.generate_embedding(documentation),
                    payload={
                        "documentation": documentation,
                        "enabled": True,
                        **self._extract_table_name_from_documentation(documentation)
                    },
                )
            ],
        )

        return self._format_point_id(id, self.documentation_collection_name)
