import logging
import re
from typing import Literal

import numpy as np
import pandas as pd
import sqlalchemy.types as sqltype
from sqlalchemy import Column, Connection, Engine, func, select, text
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.orm import Session
from sqlalchemy.schema import MetaData, Table

logger = logging.getLogger(__name__)


class DDLConstructor:
    # compile regex pattern for timestamp column prefix/suffix/keyword matching
    ts_col_suffix = [
        r'_date\b',
        r'_dt\b',
        r'_day\b',
        r'_time\b',
        r'_datetime\b',
        r'_timestamp\b',
        r'_ts\b',
        r'_at\b',
        r'_on\b',
        r'_when\b',
    ]

    ts_col_prefix = [
        r'\bdate_',
        r'\btime_',
        r'\bdt_',
        r'\bdt_',
    ]

    ts_col_keyword = [
        r'(\b|_)created(\b|_)',
        r'(\b|_)updated(\b|_)',
        r'(\b|_)modified(\b|_)',
        r'(\b|_)deleted(\b|_)',
        r'(\b|_)registered(\b|_)',
        r'(\b|_)last_login(\b|_)',
        r'(\b|_)start(\b|_)',
        r'(\b|_)end(\b|_)',
        r'(\b|_)birth(\b|_)',
        r'(\b|_)expire(\b|_)',
    ]

    ts_col_pattern = re.compile(r'|'.join(ts_col_suffix + ts_col_prefix + ts_col_keyword), re.MULTILINE | re.IGNORECASE)
    id_col_pattern = re.compile(r"^id_|_id$|^[iI]d$|^ID$|^uuid$|^UUID$")

    @classmethod
    def create_ddl_manually(
            cls, dialect: str, connection: Connection, schema: str, table_name: str
    ) -> tuple[Table, str]:
        # Get all columns inside the table and its constraints
        if dialect == 'oracle':
            if schema is None:
                schemas = pd.read_sql(text(f"""
                      SELECT DISTINCT owner
                        FROM all_tab_cols
                       WHERE table_name = '{table_name.upper()}'
                """), connection).to_dict(orient='list')['owner']
                if len(schemas) == 1:
                    schema = schemas[0]
                elif not schemas:
                    raise NoSuchTableError(table_name)
                else:
                    raise ValueError(f"Found table `{table_name}` across multiple schemas: {schemas}")
            df_column_definition = pd.read_sql(text(f"""
                  SELECT column_name,
                         data_type,
                         data_length,
                         nullable
                    FROM all_tab_cols
                   WHERE table_name = '{table_name.upper()}' AND owner = '{schema.upper()}'
            """), connection)
            df_column_constraint = pd.read_sql(text(f"""
                  SELECT col.constraint_name,
                         c.constraint_type,
                         c.search_condition_vc AS search_condition,
                         COUNT(col.column_name) AS n_columns,
                         LISTAGG(col.column_name, ', ') AS column_name
                    FROM all_cons_columns col LEFT JOIN all_constraints c
                          ON col.owner = c.owner
                         AND col.table_name = c.table_name
                         AND col.constraint_name = c.constraint_name
                   WHERE col.table_name = '{table_name.upper()}' AND col.owner = '{schema.upper()}'
                GROUP BY col.constraint_name, c.constraint_type, c.search_condition_vc
            """), connection)

        elif dialect == 'postgresql':
            if schema is None:
                schemas = pd.read_sql(text(f"""
                      SELECT DISTINCT table_schema
                        FROM information_schema.columns
                       WHERE table_name = '{table_name.upper()}'
                """), connection).to_dict(orient='list')['owner']
                if len(schemas) == 1:
                    schema = schemas[0]
                elif not schemas:
                    raise NoSuchTableError(table_name)
                else:
                    raise ValueError(f"Found table `{table_name}` across multiple schemas: {schemas}")
            df_column_definition = pd.read_sql(text(f"""
                  SELECT column_name,
                         udt_name,
                         data_type,
                         character_maximum_length AS data_length,
                         is_nullable AS nullable
                    FROM information_schema.columns
                   WHERE table_name = '{table_name}' AND table_schema = '{schema}'
                ORDER BY ordinal_position;
            """), connection)
            df_column_constraint = pd.read_sql(text(f"""
                  SELECT tc.constraint_name,
                         tc.constraint_type,
                         cc.check_clause AS search_condition,
                         COUNT(kcu.column_name) AS n_columns,
                         STRING_AGG(kcu.column_name, ', ') AS column_name
                    FROM information_schema.table_constraints tc
                         LEFT JOIN information_schema.key_column_usage kcu
                                ON tc.constraint_name = kcu.constraint_name
                               AND tc.table_schema = kcu.table_schema
                               AND tc.table_name = kcu.table_name
                         LEFT JOIN information_schema.check_constraints cc
                                ON tc.constraint_name = cc.constraint_name
                               AND tc.constraint_schema = cc.constraint_schema
                   WHERE tc.table_name = '{table_name}' AND tc.table_schema = '{schema}'
                GROUP BY tc.constraint_schema,
                         tc.constraint_name,
                         tc.constraint_type,
                         cc.check_clause;
            """), connection)

        else:
            raise NotImplementedError(f"Not supported {dialect}")

        return cls._construct_ddl(dialect, schema, table_name, df_column_definition, df_column_constraint)

    @classmethod
    def _construct_ddl(
            cls, dialect: str, schema: str, table_name: str, df_column_definition: pd.DataFrame,
            df_column_constraint: pd.DataFrame
    ) -> tuple[Table, str]:
        match dialect:
            case 'oracle':
                ddl = f'CREATE TABLE "{schema}"."{table_name}" (\n'
            case 'postgresql':
                # lower the column name
                df_column_definition.rename(lambda x: x.lower(), axis='columns', inplace=True)
                df_column_constraint.rename(lambda x: x.lower(), axis='columns', inplace=True)
                # include the schema
                ddl = f'CREATE TABLE "{schema}"."{table_name}" (\n'
            case _:
                raise NotImplementedError(f"Not supported {dialect}")

        columns = []
        columns_str_list = []
        for _, row in df_column_definition.iterrows():
            match dialect:
                case 'oracle':
                    data_type = cls._resolve_oracle_data_type(row)
                    column_str = f'    "{row["column_name"]}" {data_type}'
                    column_func = cls._row2column
                case 'postgresql':
                    data_type = cls._resolve_postgresql_data_type(row)
                    column_str = f'    "{row["column_name"]}" {data_type}'
                    column_func = cls._row2column
                case _:
                    raise NotImplementedError(f"Not supported {dialect}")

            # add column-level constraint, if multi columns will be thrown to table-level constraint
            filters = df_column_constraint['column_name'] == row['column_name']
            constraints = df_column_constraint[filters].to_dict(orient='records')
            if len(constraints) > 0:
                column_str += cls._resolve_constraint(
                    'column',
                    not_null=row['nullable'].startswith('N'),  # oracle: Y, N; postgresql: YES, NO
                    **constraints[0])
                df_column_constraint.drop(df_column_constraint.index[filters].tolist()[0], inplace=True)

            # add the column definition string
            columns_str_list.append(column_str)

            # create the column object for table object
            columns.append(column_func(dialect, row, constraints))

        if not columns_str_list:
            raise NoSuchTableError(table_name)

        # add table-level constraints
        if not df_column_constraint.empty:
            columns_str_list.extend([
                f"   {cls._resolve_constraint('table', **constraint)}"
                for constraint in df_column_constraint.to_dict(orient='records')
            ])

        # create the table object for next process using new MetaData to avoid collision with existing
        table = Table(table_name, MetaData(), *columns, schema=schema)
        # compile into single DDL string
        ddl += ",\n".join(columns_str_list) + "\n);"
        return table, ddl

    @staticmethod
    def _resolve_oracle_data_type(row: pd.Series) -> str:
        data_type, data_length = row['data_type'], row['data_length']
        match data_type:
            case 'VARCHAR2':
                data_length = f"({data_length:.0f} CHAR)"
            case 'CHAR' | 'FLOAT':
                data_length = f"({data_length:.0f})"
            case _:
                data_length = ""
        not_null = " NOT NULL" if row['nullable'] == 'N' else ""

        return f"{data_type}{data_length}{not_null}"

    @staticmethod
    def _resolve_postgresql_data_type(row: pd.Series) -> str:
        data_type, data_length = row['udt_name'], row['data_length']

        data_length = f"({data_length:.0f})" if data_length and not np.isnan(data_length) else ""
        not_null = " NOT NULL" if row['nullable'] == 'NO' else ""
        # e.g: auto_increment
        extra = f" {row['extra'].upper()}" if 'extra' in row else ""

        return f"{data_type}{data_length}{not_null}{extra}"

    @staticmethod
    def _row2column(dialect: str, row: pd.Series, constraints: dict):
        match dialect:
            case 'oracle':
                from sqlalchemy.dialects.oracle.base import ischema_names
            case 'postgresql':
                from sqlalchemy.dialects.postgresql.base import ischema_names
                # TODO: for now, treat custom type as binary
                if row['data_type'] == 'USER-DEFINED':
                    row['data_type'] = 'bytea'
            case _:
                raise NotImplementedError(f"Not supported {dialect}")

        ColumnTypeClass = ischema_names[row['data_type']]
        if issubclass(ColumnTypeClass, sqltype.String):
            column_type_instance = ColumnTypeClass(length=row['data_length'])
        else:  # Numeric, Integer, DateTime, _Binary, etc
            column_type_instance = ColumnTypeClass()

        return Column(
            row['column_name'],
            column_type_instance,
            primary_key=any(c['constraint_type'] in ['P', 'PRIMARY KEY'] for c in constraints)
        )

    @staticmethod
    def _resolve_constraint(
            level: Literal['table', 'column'], constraint_name: str, constraint_type: str, column_name: str,
            search_condition: str, not_null: bool = False, **kwargs
    ) -> str:
        # Oracle uses single letter constraint_type
        match constraint_type:
            case 'C' | 'CHECK':
                # Oracle:
                # - column-level constraint:
                #   - SALARY DECIMAL(9,2) CONSTRAINT SAL_CK CHECK (SALARY >= 10000)
                # - table-level constraint:
                #   - CONSTRAINT BONUS_CK CHECK (BONUS > TAX)
                #   - CONSTRAINT SYS_C0026335 CHECK ("NAME_BARU" IS NOT NULL)
                # whereas postgresql, the CHECK doesn't have column_name, need more database to test
                # exclude "NOT NULL" if it is already covered in data type
                if not(
                        not_null and
                        re.search(fr'^"?{column_name}"? IS NOT NULL$', search_condition, flags=re.IGNORECASE)
                ):
                    constraint_type = f"CHECK ({search_condition})"
                else:
                    return ""
            case 'P' | 'PRIMARY KEY':
                constraint_type = 'PRIMARY KEY'
                if level == 'table':
                    constraint_type += f" ({column_name})"
            case 'U' | 'UNIQUE':
                constraint_type = 'UNIQUE'
                if level == 'table':
                    constraint_type += f" ({column_name})"
            case _:
                logger.warning(f"Unhandled constraint type: `{constraint_type}`")
                return ""

        # quote the constraint name, there is a case the constraint name become unparsed, e.g: 2200_26892_1_not_null
        return f' CONSTRAINT "{constraint_name}" {constraint_type}'

    @classmethod
    def generate_table_info(cls, engine: Engine, tbl_meta: Table) -> str:
        info_categorical = []
        info_numerical = []
        with Session(engine) as sess:
            for c in tbl_meta.columns:
                # get range value of numerical, datetime, and timestamp-like column
                if isinstance(c.type, 
                              (sqltype.Integer, sqltype.Numeric, sqltype.DateTime, sqltype.Date, sqltype.Time)) \
                        or cls.ts_col_pattern.search(c.name):
                    # get min and max value
                    min_value, max_value = sess.query(func.min(c), func.max(c)).one()
                    info_numerical.append((c, min_value, max_value))

                # get unique value of the string (Varchar) column
                elif isinstance(c.type, sqltype.String) and not cls.id_col_pattern.search(c.name):
                    # count distinct values
                    col_distinct_count = sess.query(func.count(c.distinct())).scalar()
                    if col_distinct_count > 30:
                        # if too many distinct found, random select them up to 10
                        subq = sess.query(c.distinct()).subquery()
                        unique_val = sess.execute(
                            select(subq.c[0]).order_by(cls.get_engine_rand(engine)).limit(10)
                        ).scalars().all()
                    else:
                        unique_val = sess.execute(select(c).distinct()).scalars().all()
                    info_categorical.append((c, unique_val))

        table_info_str = ''

        # write the info as comments
        if info_categorical:
            table_info_str += \
                f'\n/* Unique values for categorical columns on table {tbl_meta.schema}.{tbl_meta.name}:'
            for c, v in info_categorical:
                vals = ', '.join([f"'{val}'" for val in v if val])
                if len(vals) > 0:
                    table_info_str += f'\n- Column "{c.name}": {vals}'
            table_info_str += ' */'

        if info_numerical:
            table_info_str += \
            f'\n/* Statistical value for date time and numerical columns on table {tbl_meta.schema}.{tbl_meta.name}:'
            for c, min_v, max_v in info_numerical:
                table_info_str += f'\n- Column "{c.name}" ranged from {min_v} to {max_v}'
            table_info_str += ' */'

        return table_info_str

    @classmethod
    def get_engine_rand(cls, engine: Engine):
        match engine.dialect.name:
            case 'oracle':
                random_func = text('dbms_random.value')
            case 'postgresql':
                random_func = func.random()  # might be not efficient for large table
            case 'mysql':
                random_func = func.rand()
            case _:
                raise NotImplementedError(
                    f"Random function is not yet implemented for `{engine.dialect.name}` dialect.")

        return random_func


