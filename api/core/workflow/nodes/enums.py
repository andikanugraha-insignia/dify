from enum import StrEnum


class NodeType(StrEnum):
    START = "start"
    END = "end"
    ANSWER = "answer"
    LLM = "llm"
    KNOWLEDGE_RETRIEVAL = "knowledge-retrieval"
    IF_ELSE = "if-else"
    CODE = "code"
    TEMPLATE_TRANSFORM = "template-transform"
    QUESTION_CLASSIFIER = "question-classifier"
    HTTP_REQUEST = "http-request"
    TOOL = "tool"
    VARIABLE_AGGREGATOR = "variable-aggregator"
    LEGACY_VARIABLE_AGGREGATOR = "variable-assigner"  # TODO: Merge this into VARIABLE_AGGREGATOR in the database.
    LOOP = "loop"
    LOOP_START = "loop-start"
    LOOP_END = "loop-end"
    ITERATION = "iteration"
    ITERATION_START = "iteration-start"  # Fake start node for iteration.
    PARAMETER_EXTRACTOR = "parameter-extractor"
    VARIABLE_ASSIGNER = "assigner"
    DOCUMENT_EXTRACTOR = "document-extractor"
    LIST_OPERATOR = "list-operator"
    AGENT = "agent"

    # custom blocks
    SUMMARIZER = "summarizer"
    EXPLAINER = "explainer"
    FILTERED_KNOWLEDGE_RETRIEVAL = "filtered-knowledge-retrieval"
    CACHE_RETRIEVE = "cache-retrieve"
    CACHE_STORE = "cache-store"

    # vanna.ai
    VANNAAI_CONNECTOR = "vannaai-connector"
    VANNAAI_TRAINING = "vannaai-training"
    VANNAAI_QUESTION = "vannaai-question"
    SQL_OUTPUT_TABLE = "sql-output-table"
    SQL_OUTPUT_SUMMARY = "sql-output-summary"
    SQL_OUTPUT_CHART = "sql-output-chart"
    
    # RCA
    RCA_TRAINING = "rca-training"
    RCA_QUESTION = "rca-question"


class ErrorStrategy(StrEnum):
    FAIL_BRANCH = "fail-branch"
    DEFAULT_VALUE = "default-value"


class FailBranchSourceHandle(StrEnum):
    FAILED = "fail-branch"
    SUCCESS = "success-branch"


CONTINUE_ON_ERROR_NODE_TYPE = [NodeType.LLM, NodeType.CODE, NodeType.TOOL, NodeType.HTTP_REQUEST]
RETRY_ON_ERROR_NODE_TYPE = CONTINUE_ON_ERROR_NODE_TYPE
