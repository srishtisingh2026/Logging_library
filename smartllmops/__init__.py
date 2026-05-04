import os
from .sdk import SDKTracer
from .transport import Telemetry

def init(cosmos_conn=None, db_name=None, container_name=None, application_name=None, environment="prod", model=None, provider=None, tags=None):
    """Initializes and returns a tracer instance with optional auto-patching."""
    
    # Auto-load from environment if not provided
    cosmos_conn = cosmos_conn or os.getenv("COSMOS_CONN_WRITE")
    db_name = db_name or os.getenv("COSMOS_DB")
    container_name = container_name or os.getenv("COSMOS_CONTAINER")
    
    if not cosmos_conn:
        print("⚠️ smartllmops: COSMOS_CONN_WRITE not found. Telemetry disabled.")
        return None

    telemetry = Telemetry(
        cosmos_conn=cosmos_conn,
        db_name=db_name,
        container_name=container_name
    )
    
    tracer = SDKTracer(
        telemetry,
        application_name=application_name,
        environment=environment,
        model=model,
        provider=provider,
        tags=tags
    )
    
    # LangSmith-style: Auto-patch OpenAI if requested via env var
    if os.getenv("SMART_LLMOPS_AUTO_INSTRUMENT", "false").lower() == "true":
        tracer.patch_openai()
        
    return tracer

__all__ = ["SDKTracer", "Telemetry", "init"]
