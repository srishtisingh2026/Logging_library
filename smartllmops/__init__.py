from .sdk import SDKTracer
from .transport import Telemetry

def init(cosmos_conn=None, db_name=None, container_name=None, application_name=None, environment="prod", model=None, provider=None, tags=None):
    """Initializes and returns a tracer instance."""
    telemetry = Telemetry(
        cosmos_conn=cosmos_conn,
        db_name=db_name,
        container_name=container_name
    )
    return SDKTracer(
        telemetry,
        application_name=application_name,
        environment=environment,
        model=model,
        provider=provider,
        tags=tags
    )

__all__ = ["SDKTracer", "Telemetry", "init"]
