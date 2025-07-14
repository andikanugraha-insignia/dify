from typing import Optional, cast

from configs import dify_config
from core.app.entities.app_invoke_entities import ModelConfigWithCredentialsEntity
from core.model_manager import ModelInstance

from core.workflow.entities.node_entities import NodeRunResult
from core.workflow.entities.variable_pool import VariablePool
from core.workflow.entities.workflow_node_execution import WorkflowNodeExecutionStatus
from core.workflow.nodes.base import BaseNode
from core.workflow.nodes.enums import NodeType
from core.workflow.nodes.vannaai_connector.entities import VannaaiConnectorNodeData
from extensions.ext_database import db
from services.qdrant_service import QdrantService
from services.database_connector_service import DatabaseConnectorService


class VannaaiConnectorNode(BaseNode[VannaaiConnectorNodeData]):
    _node_data_cls = VannaaiConnectorNodeData
    _node_type = NodeType.VANNAAI_CONNECTOR

    _model_instance: Optional[ModelInstance] = None
    _model_config: Optional[ModelConfigWithCredentialsEntity] = None

    @classmethod
    def version(cls) -> str:
        return "1"

    def _run(self) -> NodeRunResult:
        # extract node data from workflow
        node_data = cast(VannaaiConnectorNodeData, self.node_data)

        node_inputs = {}

        # check database connection
        args_db = {
            "host": node_data.database.host,
            "port": node_data.database.port,
            "username": node_data.database.username,
            "password": node_data.database.password,
            "database_name": node_data.database.database_name,
            "database_type": node_data.database.database_type,
            "ssl_cert": node_data.database.ssl_cert,
            "schemas": node_data.database.schemas,
        }
        result_db = DatabaseConnectorService.check_database_connection(args_db)
        if result_db[0]["result"] != "success":
            raise ValueError("Invalid Database connection!")

        # check qdrant connection, use env qdrant
        args_qdrant = {
            "url": dify_config.QDRANT_URL,
            "api_key": dify_config.QDRANT_API_KEY,
            "grpc_enabled": dify_config.QDRANT_GRPC_ENABLED,
            "grpc_port": dify_config.QDRANT_GRPC_PORT,
        }
        result_qdrant = QdrantService.check_connection(args_qdrant)
        if not result_qdrant:
            raise ValueError("Invalid Qdrant connection!")
        
        # create connector dict
        connector = {
            "database": args_db,
            "qdrant": args_qdrant,
        }
        
        outputs = {
            "connector": connector
        }

        return NodeRunResult(
            status=WorkflowNodeExecutionStatus.SUCCEEDED,
            inputs=node_inputs,
            outputs=outputs,
        )
    
    @classmethod
    def get_default_config(cls, filters: Optional[dict] = None) -> dict:
        """
        Get default config of node.
        :param filters: filter by node config parameters.
        :return:
        """
        return {
            "database": {
                "host": "",
                "port": 5432,
                "username": "postgres",
                "password": "",
                "database_name": "postgres",
                "database_type": "postgresql",
                "ssl_cert": "",
            },
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "grpc_port": 6334
            },
        }

