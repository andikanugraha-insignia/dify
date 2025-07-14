import base64
import json

import pandas as pd
from google.cloud import aiplatform
from google.oauth2 import service_account
from loguru import logger
from openai import AzureOpenAI, OpenAI
from qdrant_client import QdrantClient
from vanna.google import GoogleGeminiChat
from vanna.openai import OpenAI_Chat
from vanna.qdrant import Qdrant_VectorStore

from core.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    PromptMessage,
    SystemPromptMessage,
    UserPromptMessage,
)
from core.model_runtime.entities.model_entities import ModelType
from core.workflow.entities.qdrant_azure_openai import (
    QdrantAzureOpenAI,
    QdrantHybridOpenAI,
    RCAQdrant,
    models,
)
from extensions.ext_database import db
from models.model import App
from services.model_provider_service import DefaultModelResponse, ModelProviderService


class OpenAIChatExt(OpenAI_Chat):

    def log(self, message: str, title: str = "Info"):
        logger.debug(f"{title}: {message}")

    def generate_sql(
            self, question: str, allow_llm_to_see_data: bool = False, **kwargs
    ) -> tuple[str, list[PromptMessage]]:
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None
        ddl_list = self.get_related_ddl(question, **kwargs)
        # ddl is mandatory
        if not ddl_list:
            raise NotImplementedError(f"No DDL found for question: {question}")

        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)
        prompt: list[dict] = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **kwargs,
        )
        self.log(title="SQL Prompt", message=json.dumps(prompt, default=str, indent=2))
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(title="LLM Response", message=llm_response)

        if 'intermediate_sql' in llm_response:
            if not allow_llm_to_see_data:
                raise AssertionError(
                    "The LLM is not allowed to see the data in your database. "
                    "Your question requires database introspection to generate the necessary SQL. "
                    "Please set allow_llm_to_see_data=True to enable this."
                )

            if allow_llm_to_see_data:
                intermediate_sql = self.extract_sql(llm_response)

                try:
                    self.log(title="Running Intermediate SQL", message=intermediate_sql)
                    df = self.run_sql(intermediate_sql)

                    prompt: list[dict]  = self.get_sql_prompt(
                        initial_prompt=initial_prompt,
                        question=question,
                        question_sql_list=question_sql_list,
                        ddl_list=ddl_list,
                        doc_list=doc_list
                            + [f"The following is a pandas DataFrame with the results of the intermediate SQL query "
                            f"{intermediate_sql}: \n{df.to_markdown()}"],
                        **kwargs,
                    )
                    self.log(title="Final SQL Prompt", message='\n'.join(prompt))
                    llm_response = self.submit_prompt(prompt, **kwargs)
                    self.log(title="LLM Response", message=llm_response)
                except Exception as e:
                    raise ValueError(f"Error running intermediate SQL: {intermediate_sql}") from e
        
        prompt_messages = self._get_prompt_messages(prompt, llm_response)
        return self.extract_sql(llm_response), prompt_messages

    def test_sql(self, sql: str) -> list[dict]:
        sql = sql.strip(' \n\t;')
        # limit the result to lower the execution time
        match self.dialect:
            case 'Oracle SQL':
                sql = f"""
                    SELECT *
                      FROM ({sql})
                     WHERE ROWNUM <= 3;
                """
            case _:
                sql = f"""
                    SELECT *
                      FROM ({sql}) sql
                     LIMIT 3;
                """
        return self.run_sql(sql).to_dict('records')

    def generate_summary(self, question: str, df: pd.DataFrame, **kwargs) -> tuple[str, list[PromptMessage]]:
        message_log = [
            self.system_message(
                f"You are a helpful data assistant. The user asked the question: '{question}'\n\n"
                "The following is a pandas DataFrame with the results of the query: \n{df.to_markdown()}\n\n"
            ),
            self.user_message(
                "Briefly summarize the data based on the question that was asked."
                " Do not respond with any additional explanation beyond the summary." +
                self._response_language()
            ),
        ]

        summary = self.submit_prompt(message_log, **kwargs)

        prompt_messages = self._get_prompt_messages(message_log, summary)
        return summary, prompt_messages
    
    def _get_prompt_messages(self, input_messages: list, output: str) -> list[PromptMessage]:
        prompt_messages = []
        for p in input_messages:
            if p['role'] == 'system':
                prompt_messages.append(SystemPromptMessage(content=p['content'])) 
            elif p['role'] == 'user':
                prompt_messages.append(UserPromptMessage(content=p['content']))
            elif p['role'] == 'assistant':
                prompt_messages.append(AssistantPromptMessage(content=p['content']))
            else:
                raise ValueError(f'Prompt role: {p["role"]} is not supported.')
        prompt_messages.append(AssistantPromptMessage(content=output))
        return prompt_messages


class GoogleGeminiChatExt(GoogleGeminiChat):

    def generate_sql(
            self, question: str, allow_llm_to_see_data: bool = False, **kwargs
    ) -> tuple[str, list[PromptMessage]]:
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None
        ddl_list = self.get_related_ddl(question, **kwargs)
        # ddl is mandatory
        if not ddl_list:
            raise NotImplementedError(f"No DDL found for question: {question}")

        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)
        prompt: list[dict]  = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **kwargs,
        )
        self.log(title="SQL Prompt", message=json.dumps(prompt, default=str, indent=2))
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(title="LLM Response", message=llm_response)

        if 'intermediate_sql' in llm_response:
            if not allow_llm_to_see_data:
                raise AssertionError(
                    "The LLM is not allowed to see the data in your database. "
                    "Your question requires database introspection to generate the necessary SQL. "
                    "Please set allow_llm_to_see_data=True to enable this."
                )

            if allow_llm_to_see_data:
                intermediate_sql = self.extract_sql(llm_response)

                try:
                    self.log(title="Running Intermediate SQL", message=intermediate_sql)
                    df = self.run_sql(intermediate_sql)

                    prompt: list[dict]  = self.get_sql_prompt(
                        initial_prompt=initial_prompt,
                        question=question,
                        question_sql_list=question_sql_list,
                        ddl_list=ddl_list,
                        doc_list=doc_list
                            + [f"The following is a pandas DataFrame with the results of the intermediate SQL query "
                            f"{intermediate_sql}: \n{df.to_markdown()}"],
                        **kwargs,
                    )
                    self.log(title="Final SQL Prompt", message='\n'.join(prompt))
                    llm_response = self.submit_prompt(prompt, **kwargs)
                    self.log(title="LLM Response", message=llm_response)
                except Exception as e:
                    raise ValueError(f"Error running intermediate SQL: {intermediate_sql}") from e
        
        prompt_messages = self._get_prompt_messages(prompt, llm_response)
        return self.extract_sql(llm_response), prompt_messages
    
    def generate_summary(self, question: str, df: pd.DataFrame, **kwargs) -> tuple[str, list[PromptMessage]]:
        message_log = [
            self.system_message(
                f"You are a helpful data assistant. The user asked the question: '{question}'\n\n"
                "The following is a pandas DataFrame with the results of the query: \n{df.to_markdown()}\n\n"
            ),
            self.user_message(
                "Briefly summarize the data based on the question that was asked."
                " Do not respond with any additional explanation beyond the summary." +
                self._response_language()
            ),
        ]

        summary = self.submit_prompt(message_log, **kwargs)

        prompt_messages = self._get_prompt_messages(message_log, summary)
        return summary, prompt_messages

    def _get_prompt_messages(self, input_messages: list, output: str) -> list[PromptMessage]:
        prompt_messages = [UserPromptMessage(content=p) for p in input_messages]
        prompt_messages.append(AssistantPromptMessage(content=output))
        return prompt_messages


class VannaaiQdrant(Qdrant_VectorStore, OpenAIChatExt):
    def __init__(self, config=None, llm_client=None):
        Qdrant_VectorStore.__init__(self, config=config)
        OpenAIChatExt.__init__(self, client=llm_client, config=config)


class VannaaiQdrantAzureOpenAIEmbedding(QdrantAzureOpenAI, OpenAIChatExt):
    def __init__(self, config=None, llm_client=None, embedding_client=None):
        QdrantAzureOpenAI.__init__(self, config=config, client=embedding_client)
        OpenAIChatExt.__init__(self, client=llm_client, config=config)


class VannaaiQdrantGoogle(Qdrant_VectorStore, GoogleGeminiChatExt):
    def __init__(self, config=None):
        Qdrant_VectorStore.__init__(self, config=config)
        GoogleGeminiChatExt.__init__(self, config=config)


class GalileiVannaEmbeddingHybrid(QdrantHybridOpenAI, OpenAIChatExt):
    def __init__(self, config=None, llm_client=None, embedding_client=None):
        QdrantHybridOpenAI.__init__(self, config=config, client=embedding_client)
        OpenAIChatExt.__init__(self, client=llm_client, config=config)


class RCAVanna(RCAQdrant, OpenAIChatExt):
    def __init__(self, config=None, llm_client=None, embedding_client=None):
        RCAQdrant.__init__(self, config=config, client=embedding_client)
        OpenAIChatExt.__init__(self, client=llm_client, config=config)

    def generate_sql(
            self, question: str, allow_llm_to_see_data: bool = False,
            filter_tables: dict[str, list[str]] | None = None, additional_info: str | None = None
    ) -> tuple[str, list[PromptMessage]]:

        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = (f"You are a {self.dialect} expert. "
                              f"Please help to generate a SQL query to answer the question. "
                              f"Your response should ONLY be based on the given context and "
                              f"follow the response guidelines and format instructions.")

        if additional_info:
            initial_prompt += additional_info

        if not filter_tables:
            logger.warning('Filter table is empty. Executing CWD with all available tables')
            query_filter = None
        else:
            query_filter = self.create_filter(filter_tables)

        ddl_list = self.get_related_ddl(question, query_filter=query_filter)
        # ddl is mandatory
        if not ddl_list:
            raise NotImplementedError(f"No DDL found for question: `{question}`, using filters: `{query_filter}`")

        question_sql_list = self.get_similar_question_sql(question, query_filter=query_filter)
        doc_list = self.get_related_documentation(question, query_filter=query_filter)

        prompt: list[dict]  = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list = doc_list,
        )
        self.log(title="SQL Prompt", message=json.dumps(prompt, default=str, indent=2))
        llm_response = self.submit_prompt(prompt)
        self.log(title="LLM Response", message=llm_response)

        if 'intermediate_sql' in llm_response:
            if not allow_llm_to_see_data:
                raise AssertionError(
                    "The LLM is not allowed to see the data in your database. "
                    "Your question requires database introspection to generate the necessary SQL. "
                    "Please set allow_llm_to_see_data=True to enable this."
                )

            if allow_llm_to_see_data:
                intermediate_sql = self.extract_sql(llm_response)

                try:
                    self.log(title="Running Intermediate SQL", message=intermediate_sql)
                    df = self.run_sql(intermediate_sql)

                    prompt: list[dict]  = self.get_sql_prompt(
                        initial_prompt=initial_prompt,
                        question=question,
                        question_sql_list=question_sql_list,
                        ddl_list=ddl_list,
                        doc_list=doc_list
                            + [f"The following is a pandas DataFrame with the results of the intermediate SQL query "
                            f"{intermediate_sql}: \n{df.to_markdown(floatfmt='.2f')}"]
                    )
                    self.log(title="Final SQL Prompt", message='\n'.join(prompt))
                    llm_response = self.submit_prompt(prompt)
                    self.log(title="LLM Response", message=llm_response)
                except Exception as e:
                    raise ValueError(f"Error running intermediate SQL: {intermediate_sql}") from e

        prompt_messages = self._get_prompt_messages(prompt, llm_response)
        return self.extract_sql(llm_response), prompt_messages

    def create_filter(self, filters: dict[str, list[str]]) -> models.Filter:
        # TODO: legacy, it will be removed in Q4 2025
        filters = filters.copy()
        filter_tables = filters.pop('filter_tables', None)
        if filter_tables:
            return models.Filter(must=[
                models.FieldCondition(
                    key=self.qdrant_table_name_field,
                    match=models.MatchAny(any=list(lambda x: x.lower(), filter_tables)))
            ])

        return models.Filter(must=[
            models.FieldCondition(
                key=key,
                match=models.MatchAny(
                    # lower the value because oracle DDL is in lower case, though it's upper case in the real DB
                    any=list(lambda x: x.lower(), value) if key == self.qdrant_table_name_field else value
                ),
            ) for key, value in filters.items()
        ])


def create_qdrant_config(app_id: str, connector: dict) -> dict:
    return {
        'client': QdrantClient(
            url=connector['url'],
            api_key=connector['api_key'],
            grpc_port=connector['grpc_port'],
        ),
        'ddl_collection_name': f'{app_id}_ddl',
        'sql_collection_name': f'{app_id}_sql',
        'documentation_collection_name': f'{app_id}_documentation'
    }


class SetupVannaai:
    """
    Base class to implement Vanna.ai block used by
    Vanna.ai training, Vanna.ai question.
    SQL Output Table, SQL Output Summary, SQL Output Chart
    """

    def __init__(self,
                 app_id,
                 connector, 
                 model_instance, 
                 use_system_embedding: bool = False, 
                 max_tokens: int = 128000,
                 app_id_target: str | None = None,
                 is_rca: bool = False
                 ) -> None:
        self.app_id = app_id

        embedding_client = None
        if use_system_embedding:
            result = self._setup_embedding_from_system(self.app_id)
            self._embedding_model, self._system_model_credentials, embedding_client = result

        if not app_id_target:
            app_id_target = app_id

        # setup and initiate vanna
        config = {
            'api_key': model_instance.credentials.get('openai_api_key'),
            'model': model_instance.model,
            'max_tokens': max_tokens,
            **create_qdrant_config(app_id_target, connector['qdrant'])
        }
        db_config = {
            'host': connector['database']['host'],
            'port': connector['database']['port'],
            'dbname': connector['database']['database_name'],
            'user': connector['database']['username'],
            'password': connector['database']['password'],
        }

        if model_instance.provider == 'azure_openai':
            openai_client = AzureOpenAI(
                api_key=model_instance.credentials.get('openai_api_key'),
                api_version=model_instance.credentials.get('openai_api_version'),
                azure_endpoint=model_instance.credentials.get('openai_api_base'),
            )
            if not use_system_embedding:
                self._vn = VannaaiQdrant(config=config, llm_client=openai_client)
            else:
                config['openai_api_base'] = self._system_model_credentials.get('openai_api_base')
                config['api_key'] = self._system_model_credentials.get('openai_api_key')
                config['model_embedding'] = self._embedding_model.model

                if is_rca:
                    self._vn = RCAVanna(config=config, llm_client=openai_client, embedding_client=embedding_client)
                else:
                    self._vn = GalileiVannaEmbeddingHybrid(
                        config=config, llm_client=openai_client, embedding_client=embedding_client
                    )

        elif model_instance.provider == 'google':
            if is_rca:
                raise NotImplementedError(f"RCA Vanna isn't supported by provider=`{model_instance.provider}`. "
                                          f"Only OpenAI and Azure OpenAI that can be used")
            # Google Gemini
            config['api_key'] = model_instance.credentials.get('google_api_key')
            self._vn = VannaaiQdrantGoogle(config=config)

        elif model_instance.provider == 'vertex_ai':
            if is_rca:
                raise NotImplementedError(f"RCA Vanna isn't supported by provider=`{model_instance.provider}`. "
                                          f"Only OpenAI and Azure OpenAI that can be used")
            # Vertex AI
            credentials = model_instance.credentials
            service_account_info = json.loads(base64.b64decode(credentials['vertex_service_account_key']))
            project_id = credentials['vertex_project_id']
            location = credentials['vertex_location']
            if service_account_info:
                service_accountSA = service_account.Credentials.from_service_account_info(service_account_info)
                aiplatform.init(credentials=service_accountSA, project=project_id, location=location)
            else:
                aiplatform.init(project=project_id, location=location)

            config['model_name'] = model_instance.model
            self._vn = VannaaiQdrantGoogle(config=config)

        elif model_instance.provider == 'openai':
            openai_client = OpenAI(
                api_key=model_instance.credentials.get('openai_api_key'),
                base_url=model_instance.credentials.get('openai_api_base'),
            )
            if not use_system_embedding:
                self._vn = VannaaiQdrant(config=config, llm_client=openai_client)
            else:
                config['openai_api_base'] = self._system_model_credentials.get('openai_api_base')
                config['api_key'] = self._system_model_credentials.get('openai_api_key')
                config['model_embedding'] = self._embedding_model.model

                if is_rca:
                    self._vn = RCAVanna(config=config, llm_client=openai_client, embedding_client=embedding_client)
                else:
                    self._vn = GalileiVannaEmbeddingHybrid(
                        config=config, llm_client=openai_client, embedding_client=embedding_client
                    )

        else:
            raise NotImplementedError(f"Model provider `{model_instance.provider}` not implemented yet.")

        # Connect to database
        if connector['database']['database_type'] == 'postgresql':
            self._vn.connect_to_postgres(**db_config)
        elif connector['database']['database_type'] == 'mysql':
            self._vn.connect_to_mysql(**db_config)
        elif connector['database']['database_type'] == 'oracle':
            self._vn.connect_to_oracle(
                user=db_config['user'],
                password=db_config['password'],
                dsn=f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}",
            )
            self._vn.dialect = 'Oracle SQL'
        else:
            logger.warning('Database type not supported: ', connector['database']['database_type'])
        self.config = config

    @classmethod
    def _setup_embedding_from_system(cls, app_id: str) -> tuple[DefaultModelResponse, dict, OpenAI]:
        # get app
        app = db.session.query(App).filter(
            App.id == app_id,
        ).first()
        tenant_id = app.tenant_id

        model_provider_service = ModelProviderService()
        default_model_entity = model_provider_service.get_default_model_of_model_type(
            tenant_id=tenant_id, model_type=ModelType.TEXT_EMBEDDING.value
        )

        credentials = cls._get_model_credentials(
            tenant_id=tenant_id,
            provider=default_model_entity.provider.provider,
            model_type=ModelType.TEXT_EMBEDDING.value,
            model=default_model_entity.model,
        )

        if default_model_entity.provider.provider == 'azure_openai':
            embedding_client = AzureOpenAI(
                api_key=credentials.get('openai_api_key'),
                api_version=credentials.get('openai_api_version'),
                azure_endpoint=credentials.get('openai_api_base'),
            )
        elif default_model_entity.provider.provider == 'openai':
            embedding_client = OpenAI(
                api_key=credentials.get('openai_api_key'),
                base_url=credentials.get('openai_api_base'),
            )
        else:
            raise NotImplementedError(
                f"System embedding model provider=`{default_model_entity.provider.provider}`"
                " has not been implemented yet."
            )

        return default_model_entity, credentials, embedding_client

    @staticmethod
    def _get_model_credentials(tenant_id: str, provider: str, model_type: str, model: str) -> dict:
        """
        get model credentials.

        :param tenant_id: workspace id
        :param provider: provider name
        :param model_type: model type
        :param model: model name
        :return:
        """
        model_provider_service = ModelProviderService()

        # Get all provider configurations of the current workspace
        provider_configurations = model_provider_service.provider_manager.get_configurations(tenant_id)

        # Get provider configuration
        provider_configuration = provider_configurations.get(provider)
        if not provider_configuration:
            raise ValueError(f'Provider {provider} does not exist.')

        return provider_configuration.get_current_credentials(ModelType.value_of(model_type), model)

    def get_vn(self):
        return self._vn
