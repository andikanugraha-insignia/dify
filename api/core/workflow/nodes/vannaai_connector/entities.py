from pydantic import BaseModel

from core.workflow.nodes.base.entities import BaseNodeData


class DatabaseConnectionData(BaseModel):
    """
    Database connection
    """
    host: str
    port: int
    username: str
    password: str
    database_name: str
    database_type: str
    ssl_cert: str
    schemas: list[str]

class QdrantConnectionData(BaseModel):
    """
    Qdrant connection
    """
    url: str
    api_key: str
    grpc_enabled: bool
    grpc_port: int

class VannaaiConnectorNodeData(BaseNodeData):
    """
    Vanna.AI Connector Node Data.
    """
    database: DatabaseConnectionData
    qdrant: QdrantConnectionData
