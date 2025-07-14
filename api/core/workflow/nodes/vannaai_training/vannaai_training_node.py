import logging
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional, Union

from sqlalchemy import URL, Engine, create_engine, inspect
from sqlalchemy.exc import CompileError, NoSuchTableError
from sqlalchemy.schema import CreateTable, MetaData, Table

from configs import dify_config
from core.app.entities.app_invoke_entities import ModelConfigWithCredentialsEntity
from core.model_manager import ModelInstance
from core.model_runtime.entities.llm_entities import LLMUsage
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel
from core.model_runtime.utils.encoders import jsonable_encoder
from core.workflow.entities.node_entities import NodeRunResult, WorkflowNodeExecutionMetadataKey
from core.workflow.entities.workflow_node_execution import WorkflowNodeExecutionStatus
from core.workflow.nodes.base import BaseNode
from core.workflow.nodes.enums import NodeType
from core.workflow.nodes.llm.entities import ModelConfig
from core.workflow.nodes.vannaai_training.ddl_constructor import DDLConstructor
from core.workflow.nodes.vannaai_training.entities import VannaaiTrainingNodeData, VannaaiTrainingSqlData
from core.workflow.utils.setup_vannaai import (
    GalileiVannaEmbeddingHybrid,
    RCAVanna,
    SetupVannaai,
    VannaaiQdrant,
    VannaaiQdrantGoogle,
)

logger = logging.getLogger(__name__)


class VannaaiTrainingNode(BaseNode[VannaaiTrainingNodeData]):
    _node_data_cls = VannaaiTrainingNodeData
    _node_type = NodeType.VANNAAI_TRAINING

    _model_instance: Optional[ModelInstance] = None
    _model_config: Optional[ModelConfigWithCredentialsEntity] = None

    @classmethod
    def version(cls) -> str:
        return "1"

    def _run(self) -> NodeRunResult:
        # get connector data
        connector = self.node_data.connector
        if connector:
            if not isinstance(connector, dict):
                raise ValueError('Invalid connector!')

        # get the LLM model
        model_instance, model_config = self._fetch_model_config(self.node_data.model)
        if not isinstance(model_instance.model_type_instance, LargeLanguageModel):
            raise ValueError('Model is not a Large Language Model')

        llm_model = model_instance.model_type_instance
        model_schema = llm_model.get_model_schema(model_config.model, model_config.credentials)
        if not model_schema:
            raise ValueError('Model schema not found')

        # setup vanna.ai
        vn_setup = SetupVannaai(app_id=self.app_id,
                                connector=connector,
                                model_instance=model_instance,
                                use_system_embedding=dify_config.VANNAAI_USE_SYSTEM_EMBEDDING,
                                max_tokens=self.node_data.max_tokens,
                                is_rca=False)
        vn = vn_setup.get_vn()

        ddl_filter = self.node_data.ddl_filter
        training_sql = self.node_data.training_sql
        documentation = self.node_data.documentation
        is_clear_data = self.node_data.is_clear_data

        node_inputs = {}
        node_inputs['#ddl_filter#'] = ddl_filter
        node_inputs['#training_sql#'] = [{'question': t.question, 'sql_query': t.sql_query} for t in training_sql]
        node_inputs['#documentation#'] = documentation
        node_inputs['#is_clear_data#'] = is_clear_data

        if is_clear_data:
            self._clear_data(vn, vn_setup.config)

        self._train(vn, connector, ddl_filter, training_sql, documentation)

        # calculate token usage
        usage = LLMUsage.empty_usage()

        output = {'success': True}
        outputs = {
            'result': output,
            'usage': jsonable_encoder(usage),
        }

        return NodeRunResult(
            status=WorkflowNodeExecutionStatus.SUCCEEDED,
            inputs=node_inputs,
            outputs=outputs,
            metadata={
                WorkflowNodeExecutionMetadataKey.TOTAL_TOKENS: usage.total_tokens,
                WorkflowNodeExecutionMetadataKey.TOTAL_PRICE: usage.total_price,
                WorkflowNodeExecutionMetadataKey.CURRENCY: usage.currency
            }
        )
    
    def _clear_data(self,
                    vn: VannaaiQdrant | GalileiVannaEmbeddingHybrid | VannaaiQdrantGoogle,
                    config: dict):
        """
        Clear ddl, sql, documentation collection
        """
        ddl_status = vn.remove_collection(config['ddl_collection_name'])
        sql_status = vn.remove_collection(config['sql_collection_name'])
        doc_status = vn.remove_collection(config['documentation_collection_name'])
        if not ddl_status:
            logger.warning('Remove ddl collection failed')
        if not sql_status:
            logger.warning('Remove sql collection failed')
        if not doc_status:
            logger.warning('Remove documentation collection failed')
    
    def _train(self,
               vn: VannaaiQdrant | GalileiVannaEmbeddingHybrid | VannaaiQdrantGoogle,
               connector: dict,
               ddl_filter: list[str],
               training_sql: list[VannaaiTrainingSqlData],
               documentation: list[str]):
        """
        Vanna.ai training DDL, SQL, Documentation
        """
        self._train_ddl(vn, connector, ddl_filter)
        self._train_sql(vn, training_sql)
        self._train_documentation(vn, documentation)

    @contextmanager
    def _setup_engine(self, connector: dict, **kwargs) -> Iterator[Engine]:
        """
        Setup SQL Alchemy engine
        """
        db_type = connector['database']['database_type']
        match db_type:
            case 'postgresql':
                drivername = 'postgresql+psycopg2'
            case 'mysql':
                drivername = 'mysql+pymysql'
            case 'oracle':
                drivername = 'oracle+oracledb'
            case _:
                raise NotImplementedError(f"Not supported {db_type}")

        engine_url = URL.create(
            drivername=drivername,
            username=connector['database']['username'],
            password=connector['database']['password'],
            host=connector['database']['host'],
            port=connector['database']['port'],
            database=connector['database']['database_name'],
            **kwargs
        )

        # In oracle database, database name is specified as service name
        if db_type == 'oracle':
            engine_url = engine_url._replace(
                database=None, query={'service_name': connector['database']['database_name']}
            )

        engine = None
        try:
            engine = create_engine(engine_url)
            yield engine
        # DO NOT CATCH ANY ERROR HERE, we need to raise the error
        finally:
            if engine:
                engine.dispose()

    @staticmethod
    def parse_ddl_filter(table_names: list[str]) -> dict[Union[str, None], set[str]]:
        result = defaultdict(set)
        for table_name in table_names:
            arr = table_name.lower().split('.')
            if len(arr) == 1:
                result[None].add(arr[-1])
            else:
                result[arr[-2] or None].add(arr[-1])
        return result

    def _train_ddl(self,
                   vn: VannaaiQdrant | GalileiVannaEmbeddingHybrid | VannaaiQdrantGoogle | RCAVanna,
                   connector: dict,
                   ddl_filter: list[str],
                   quote_schema: bool | None = None):
        with self._setup_engine(connector) as engine:  # type: Engine
            db_type = connector['database']['database_type']
            if db_type in ['oracle', 'postgresql']:
                schemas: list = connector['database']['schemas']
                if not schemas:
                    # for database oracle, default schema is the username
                    # do not supply None, to avoid unintended result
                    schemas.append(connector['database']['username'] if db_type == 'oracle' else 'public')
            else:
                schemas = [None]

            # filter table names
            inspector = inspect(engine)

            all_table_names = {}
            try:
                for s in schemas:
                    # schema is case-sensitive, 
                    # the returned all_table_names is in lower case regardless in the real database
                    all_table_names[s] = inspector.get_table_names(schema=s)
            finally:
                inspector.bind.dispose()

            filtered_table_names = defaultdict(set)
            if ddl_filter:
                parsed_ddl_filter = self.parse_ddl_filter(ddl_filter)

                # if no schema in the filter table, apply to the all selected schemas
                schemaless_filter = parsed_ddl_filter.pop(None, set())
                if schemaless_filter:
                    for schema, tbl_names in all_table_names.items():
                        items = schemaless_filter & set(tbl_names)
                        if items:
                            filtered_table_names[schema].update()

                # filter table with schema
                for schema, schema_based_filter in parsed_ddl_filter.items():
                    # if user provides a ddl filter with schema
                    entries = all_table_names.get(schema)
                    if not entries:
                        raise KeyError(f'No schema "{schema}" in the database, check your database and your connector')
                    entries = set(entries)
                    # check the filter with schema if it not exists
                    not_found_table = schema_based_filter - entries
                    if len(not_found_table) > 0:
                        raise ValueError(f'There is no such table "{list(not_found_table)}" in the schema "{schema}"')
                    # filter tables
                    items = schema_based_filter & entries
                    if items:
                        filtered_table_names[schema].update(items)

            else:
                filtered_table_names = all_table_names

            if not filtered_table_names:
                raise ValueError('No matching table to trained in the database')

            with engine.connect() as connection:
                for schema, table_names_ in filtered_table_names.items():
                    meta = MetaData(schema=schema, quote_schema=quote_schema)
                    for table_name in table_names_:
                        if not table_names_:
                            raise ValueError(f'There is no table to be trained in the schema "{schema}"')
                        try:
                            # table_name is case sensitive and should lower case, 
                            # though in real database it's upper case
                            tbl_meta = Table(table_name, meta, autoload_with=engine)
                            # remove TABLESPACE statement on oracle
                            if db_type == 'oracle':
                                tbl_meta.dialect_options['oracle'].pop('tablespace', None)
                            table_ddl = str(CreateTable(tbl_meta).compile(engine)).strip()
                        except (NoSuchTableError, CompileError) as e:
                            logger.exception("Create DDL using SQLAlchemy is failed, trying manually...")
                            try:
                                tbl_meta, table_ddl = DDLConstructor.create_ddl_manually(
                                    db_type,
                                    connection,
                                    schema,
                                    table_name
                                )
                            except NoSuchTableError as e:
                                logger.exception("NoSuchTableError")
                                continue

                        # add additional info to ddl
                        table_info = DDLConstructor.generate_table_info(engine, tbl_meta)
                        vn.train(ddl=table_ddl + table_info)

    def _train_sql(self,
                   vn: VannaaiQdrant | GalileiVannaEmbeddingHybrid | VannaaiQdrantGoogle,
                   list_training_sql: list[VannaaiTrainingSqlData]):
        if not list_training_sql:
            return
        
        for training_sql in list_training_sql:
            if training_sql.is_question_sql_query:
                vn.train(question=training_sql.question, sql=training_sql.sql_query)
            else:
                vn.train(sql=training_sql.sql_query)

    def _train_documentation(self,
                             vn: VannaaiQdrant | GalileiVannaEmbeddingHybrid | VannaaiQdrantGoogle,
                             documentation: list[str]):
        if not documentation:
            return
        
        for doc in documentation:
            vn.train(documentation=doc)

    def _fetch_model_config(
        self, node_data_model: ModelConfig
    ) -> tuple[ModelInstance, ModelConfigWithCredentialsEntity]:
        """
        Fetch model config.
        """
        if not self._model_instance or not self._model_config:
            self._model_instance, self._model_config = super()._fetch_model_config(node_data_model)

        return self._model_instance, self._model_config

    @classmethod
    def get_default_config(cls, filters: Optional[dict] = None) -> dict:
        """
        Get default config of node.
        :param filters: filter by node config parameters.
        :return:
        """
        return {
            "connector": {
                "database": {
                    "host": "",
                    "port": 5432,
                    "username": "postgres",
                    "password": "",
                    "database_name": "postgres",
                    "database_type": "postgresql",
                    "ssl_cert": "",
                    "schemas": [],
                },
                "qdrant": {"url": "http://localhost:6333", "api_key": "", "grpc_port": 6334},
            },
            "training_sql": [],
            "documentation": [],
        }
