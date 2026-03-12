import uuid
import time
import functools
import inspect
from typing import Dict, Any
from contextvars import ContextVar


# Context variables (async-safe)
_spans_var: ContextVar[list] = ContextVar("spans", default=[])
_stack_var: ContextVar[list] = ContextVar("active_span_stack", default=[])
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default=None)


class SDKTracer:

    def __init__(self, telemetry, environment="prod", model="unknown", provider="unknown"):
        self.telemetry = telemetry
        self.environment = environment
        self.model = model
        self.provider = provider

    # ---------------------------------------------------------
    # TRACE INITIALIZATION
    # ---------------------------------------------------------

    def start_trace(self):
        _spans_var.set([])
        _stack_var.set([])
        trace_id = f"trace-{uuid.uuid4().hex[:8]}"
        _trace_id_var.set(trace_id)

    # ---------------------------------------------------------
    # USAGE NORMALIZATION
    # ---------------------------------------------------------

    def _normalize_usage(self, usage: Dict[str, Any]) -> Dict[str, Any]:

        if not usage:
            return {}

        if "prompt_tokens" in usage or "completion_tokens" in usage:
            prompt = int(usage.get("prompt_tokens", 0) or 0)
            completion = int(usage.get("completion_tokens", 0) or 0)

            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": int(usage.get("total_tokens", prompt + completion))
            }

        if "usage_metadata" in usage:
            meta = usage["usage_metadata"]

            prompt = int(meta.get("prompt_token_count", 0))
            completion = int(meta.get("candidates_token_count", 0))

            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": int(meta.get("total_token_count", prompt + completion))
            }

        if "input_tokens" in usage or "output_tokens" in usage:
            prompt = int(usage.get("input_tokens", 0))
            completion = int(usage.get("output_tokens", 0))

            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion
            }

        return {}

    # ---------------------------------------------------------
    # SAFE SERIALIZATION
    # ---------------------------------------------------------

    def _safe_serialize(self, obj, max_length=150):

        def _serialize(o, depth=0):

            if depth > 2:
                return "..."

            if o is None:
                return "None"

            if isinstance(o, (int, float, bool)):
                return str(o)

            if hasattr(o, "page_content"):
                return f"Document(len={len(o.page_content)})"

            if isinstance(o, (list, tuple)):
                items = [_serialize(i, depth + 1) for i in o[:3]]
                if len(o) > 3:
                    items.append("...")
                return "[" + ", ".join(items) + "]"

            if isinstance(o, dict):
                items = []
                for k, v in list(o.items())[:2]:
                    items.append(f"{k}: {_serialize(v, depth + 1)}")
                if len(o) > 2:
                    items.append("...")
                return "{" + ", ".join(items) + "}"

            return str(o).replace("\n", " ")

        s = _serialize(obj)

        if len(s) > max_length:
            return s[:max_length] + "... [TRUNCATED]"

        return s

    # ---------------------------------------------------------
    # SPAN EXECUTION CORE
    # ---------------------------------------------------------

    def _before_span(self, func, name, parent_span_id):

        if not _trace_id_var.get():
            self.start_trace()

        span_name = name or func.__name__
        span_id = f"span-{uuid.uuid4().hex[:8]}"
        start_time = int(time.time() * 1000)

        stack = _stack_var.get()
        spans = _spans_var.get()

        effective_parent = parent_span_id or (stack[-1] if stack else None)

        stack = stack + [span_id]
        _stack_var.set(stack)

        return span_id, span_name, start_time, effective_parent

    def _after_span(
        self,
        span_id,
        span_name,
        start_time,
        effective_parent,
        status,
        output,
        metadata,
        usage,
        include_io,
        result_parser,
        args,
        kwargs,
        error_metadata,
        span_type,
    ):

        stack = _stack_var.get()
        spans = _spans_var.get()

        if stack:
            stack = stack[:-1]
            _stack_var.set(stack)

        end_time = int(time.time() * 1000)

        trace_id = _trace_id_var.get()

        final_metadata = (metadata or {}).copy()
        final_usage = (usage or {}).copy()

        final_metadata.update(error_metadata or {})

        if result_parser and status == "success":

            try:
                parsed = result_parser(output, args, kwargs)

                if isinstance(parsed, dict):

                    final_metadata.update(parsed.get("metadata", {}))

                    raw_usage = parsed.get("usage")

                    if raw_usage:
                        normalized = self._normalize_usage(raw_usage)
                        final_usage.update(normalized)
                        final_metadata["_provider_raw_usage"] = raw_usage

            except Exception as e:
                final_metadata["_parser_error"] = str(e)

        if include_io and not final_metadata:
            final_metadata = {
                "input": self._safe_serialize(args),
                "output": self._safe_serialize(output),
            }

        span = {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": effective_parent,
            "type": span_type or "generic",
            "name": span_name,
            "start_time": start_time,
            "end_time": end_time,
            "latency_ms": end_time - start_time,
            "status": status,
            "metadata": final_metadata,
            "usage": final_usage,
        }

        spans = spans + [span]
        _spans_var.set(spans)

    # ---------------------------------------------------------
    # SPAN EXECUTION
    # ---------------------------------------------------------

    def _execute_span(
        self,
        func,
        args,
        kwargs,
        name,
        span_type,
        metadata,
        usage,
        include_io,
        result_parser,
        parent_span_id,
        is_async=False,
    ):

        span_id, span_name, start_time, parent = self._before_span(func, name, parent_span_id)

        def finalize(status, output, error_meta):
            self._after_span(
                span_id,
                span_name,
                start_time,
                parent,
                status,
                output,
                metadata,
                usage,
                include_io,
                result_parser,
                args,
                kwargs,
                error_meta,
                span_type,
            )

        if is_async:

            async def wrapper():
                status = "success"
                output = None
                error_meta = {}

                try:
                    output = await func(*args, **kwargs)
                    return output

                except Exception as e:
                    status = "error"
                    output = str(e)
                    error_meta = {"error": str(e), "error_type": type(e).__name__}
                    raise

                finally:
                    finalize(status, output, error_meta)

            return wrapper()

        else:

            status = "success"
            output = None
            error_meta = {}

            try:
                output = func(*args, **kwargs)
                return output

            except Exception as e:
                status = "error"
                output = str(e)
                error_meta = {"error": str(e), "error_type": type(e).__name__}
                raise

            finally:
                finalize(status, output, error_meta)

    # ---------------------------------------------------------
    # DECORATOR
    # ---------------------------------------------------------

    def trace(
        self,
        name=None,
        span_type=None,
        metadata=None,
        usage=None,
        include_io=True,
        result_parser=None,
        parent_span_id=None,
    ):

        def decorator(func):

            is_async = inspect.iscoroutinefunction(func)

            if is_async:

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await self._execute_span(
                        func, args, kwargs, name, span_type, metadata,
                        usage, include_io, result_parser, parent_span_id, True
                    )

                return async_wrapper

            else:

                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    return self._execute_span(
                        func, args, kwargs, name, span_type, metadata,
                        usage, include_io, result_parser, parent_span_id, False
                    )

                return sync_wrapper

        return decorator

    # ---------------------------------------------------------
    # TRACE EXPORT
    # ---------------------------------------------------------

    def export_trace(
        self,
        output,
        query=None,
        session_id=None,
        user_id=None,
        timestamp=None,
        rag_docs=None,
    ):

        spans = _spans_var.get()

        trace_id = _trace_id_var.get() or f"trace-{uuid.uuid4().hex[:8]}"

        provider_raw = None

        for span in spans:
            if span["type"] == "llm":
                provider_raw = (
                    span["metadata"].get("_provider_raw_usage")
                    or span["usage"]
                )
                break

        latency = None

        if spans:
            start = min(s["start_time"] for s in spans)
            end = max(s["end_time"] for s in spans)
            latency = end - start

        answer = output.get("output") if isinstance(output, dict) else output

        trace = {
            "id": trace_id,
            "trace_id": trace_id,
            "trace_name": output.get("trace_name", "ai-trace") if isinstance(output, dict) else "ai-trace",
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": timestamp or int(time.time() * 1000),
            "environment": self.environment,
            "provider": self.provider,
            "model": self.model,
            "input": {"query": query},
            "output": {"answer": answer},
            "latency_ms": latency,
            "rag_docs": self._safe_serialize(rag_docs, 1000) if rag_docs else None,
            "spans": spans,
            "provider_raw": provider_raw,
            "status": "success",
        }

        _spans_var.set([])
        _stack_var.set([])
        _trace_id_var.set(None)

        self.telemetry.log_trace(trace)