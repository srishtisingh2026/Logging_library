import os
from datetime import datetime

class Telemetry:
    def __init__(self, cosmos_conn=None, db_name=None, container_name=None):
        
        # Cosmos configuration
        self.cosmos_conn = cosmos_conn or os.getenv("COSMOS_CONN_WRITE")
        self.db_name = db_name or os.getenv("COSMOS_DB", "llmops-data")
        self.container_name = container_name or os.getenv("COSMOS_CONTAINER", "raw_traces")

        self.client = None
        self.container = None

        if self.cosmos_conn:
            try:
                from azure.cosmos import CosmosClient
                self.client = CosmosClient.from_connection_string(self.cosmos_conn)
                db = self.client.get_database_client(self.db_name)
                self.container = db.get_container_client(self.container_name)
                print("Cosmos telemetry enabled")
            except Exception as e:
                print(f"Cosmos initialization failed: {e}")

    def log_trace(self, trace: dict):
        try:
            # Ensure ID exists
            if "trace_id" in trace:
                trace["id"] = trace["trace_id"]

            # Ensure partition key
            trace["partitionKey"] = trace.get("partitionKey", trace.get("id"))

            # Ensure timestamp
            trace.setdefault("logged_at", datetime.utcnow().isoformat())

            # Cosmos DB
            if self.container:
                try:
                    self.container.upsert_item(body=trace)
                except Exception as e:
                    print(f"Cosmos logging failed: {e}")

        except Exception as e:
            print(f"Telemetry error: {e}")

