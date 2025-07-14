from typing import Any, Optional

from pydantic import BaseModel

from core.workflow.nodes.base.entities import BaseNodeData


class ModelConfig(BaseModel):
    """
    Model Config.
    """
    provider: str
    name: str
    mode: str
    completion_params: dict[str, Any] = {}

class VannaaiTrainingSqlData(BaseModel):
    question: str
    sql_query: str
    is_question_sql_query: bool

class VannaaiTrainingNodeData(BaseNodeData):
    """
    Vanna.AI Training Node Data.
    """
    connector: Optional[list[str]] = None
    model: ModelConfig
    ddl_filter: list[str]
    training_sql: list[VannaaiTrainingSqlData]
    documentation: list[str]
    is_clear_data: bool
    max_tokens: int
