from qdrant_client import QdrantClient


class QdrantService:

    @classmethod
    def check_connection(cls, args: dict) -> bool:
        """
        Check Qdrant connection.
        """
        try:
            client = QdrantClient(
                url=args['url'],
                # host=args['host'], 
                # port=args['port'],
                api_key=args.get('api_key', ''),
                grpc_port=args.get('grpc_port', 6334))
            # call one API to check the connection
            client.get_collections()
            client.close()
        except Exception as e:
            return False

        return True
