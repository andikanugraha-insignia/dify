import mysql.connector
import oracledb
import psycopg2


class DatabaseConnectorService:

    @classmethod
    def check_database_connection(cls, args: dict) -> dict[str, str] | tuple[dict[str, str], int] | tuple[
            dict[str, str], int]:
        try:
            if args['database_type'] == 'postgresql':
                conn = psycopg2.connect(
                    host=args['host'],
                    port=args['port'],
                    user=args['username'],
                    password=args['password'],
                    dbname=args['database_name'],
                    sslmode='require' if args.get('ssl_cert') else 'prefer',
                    sslcert=args.get('ssl_cert')
                )
                conn.close()
            elif args['database_type'] == 'mysql':
                conn = mysql.connector.connect(
                    host=args['host'],
                    port=args['port'],
                    user=args['username'],
                    password=args['password'],
                    database=args['database_name'],
                    ssl_ca=args.get('ssl_cert')
                )
                conn.close()
            elif args['database_type'] == 'oracle':
                conn = oracledb.connect(
                    user=args['username'],
                    password=args['password'],
                    dsn=f"{args['host']}:{args['port']}/{args['database_name']}",
                )
                conn.close()
            else:
                return {'result': 'fail', 'message': 'Unsupported database type'}, 400
        except Exception as e:
            return {'result': 'fail', 'message': str(e)}, 400

        return {'result': 'success', 'message': 'Connection successful'}, 200


