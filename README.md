# SmartLLMOps SDK

A lightweight, declarative SDK for high-fidelity tracing and observability in RAG and Agentic LLM applications.

## 🚀 Installation

### 1. Local Development
To install the SDK in your application's environment during development:
```bash
pip install -e /path/to/smartllmops-sdk
```

### 2. From Source (Production)
```bash
pip install git+https://github.com/srishtisingh2026/Logging_library.git
```

## ⚙️ Configuration

The SDK uses environment variables for destination logging (e.g., Cosmos DB).

| Variable | Description |
|----------|-------------|
| `COSMOS_CONN_WRITE` | Primary connection string for Azure Cosmos DB |
| `COSMOS_DB` | Database name (default: `llmops-data`) |
| `COSMOS_CONTAINER` | Container name (default: `raw_traces`) |
| `SMART_LLMOPS_APP` | Application name for grouping traces (optional) |

## 🛠️ Quick Start

### 1. Initialize the Tracer
Initialize the tracer once in your application (e.g., `app.py`).

```python
import smartllmops

# Initialize with application identity and environment
# (model and provider are optional; they are auto-detected at runtime for supported LLMs)
tracer = smartllmops.init(
    application_name="my-portfolio-analyst",
    environment="production",
    tags={"version": "1.0.2", "team": "fin-tech"}
)
```

### 2. Decorate your Functions
Use the `@trace` decorator to capture execution flow, inputs/outputs, and metadata automatically.

```python
@tracer.trace(span_type="llm", name="my_llm_span")
def get_llm_response(prompt):
    # Your LLM call logic
    return content, prompt, usage_metadata
```

#### Supported Span Types
The SDK performs "Smart Parsing" based on the `span_type`:
- `intent-classification`: Captures detected labels and token usage.
- `chain`: Captures rewritten queries or intermediate reasoning.
- `retrieval`: Captures document snippets and similarity scores.
- `llm`: Captures detailed token usage (prompt, completion, total) and context token counts.
- `agent`: Captures agent internal plans (from raw responses) and structured final outputs.

### 3. Trace Life Cycle
Wrap your main entry point (pipeline) with `start_trace` and `export_trace`.

```python
def run_pipeline(user_query):
    # A. Start a new trace context
    tracer.start_trace()
    
    try:
        # B. Run your logic
        result = my_rag_engine.run(user_query)
        
        # C. Export the completed trace
        tracer.export_trace(
            result, 
            query=user_query, 
            session_id="session-123", 
            user_id="user-456"
        )
        return result
    except Exception as e:
        # Trace is automatically updated with Error status if using decorators
        raise e
```

## 📊 Telemetry
Traces are automatically upserted to your configured Azure Cosmos DB container for dashboarding.

