import functools
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import sys
from typing import Concatenate, Literal, ParamSpec, TypeAlias, TypeVar

from latch_config.config import DatadogConfig, LoggingMode, read_config
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.util.types import AttributeValue
from opentelemetry.sdk.resources import Attributes, LabelValue, Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Tracer
from opentelemetry.trace.span import Span


class NoopSpanExporter(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None: ...

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def format_console_span(x: ReadableSpan) -> str:
    ctx = x.context

    parent_ctx = x.parent
    parent_id = "      root      "
    if parent_ctx is not None:
        parent_id = f"{parent_ctx.span_id:x}"

    s_id = "?" * 16
    if ctx is not None:
        s_id = f"{ctx.span_id:x}"
    return f"<{parent_id}> |- <{s_id}> {x.name}\n"


@dataclass(frozen=True)
class Config:
    datadog: DatadogConfig
    logging_mode: LoggingMode = LoggingMode.console_json


config = read_config(Config)

app_tracer: Tracer


def setup(
    *,
    span_exporter: Literal["otlp", "console", "noop"] = "otlp",
    resource_attrs: Attributes | None = None,
) -> None:
    if resource_attrs is None:
        resource_attrs = {}

    service_data: Attributes = {
        "service.name": config.datadog.service_name,
        "service.version": config.datadog.service_version,
        "deployment.environment": config.datadog.deployment_environment,
        **resource_attrs,
    }

    tracer_provider = TracerProvider(resource=Resource(service_data))

    if span_exporter == "otlp":
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    elif span_exporter == "console":
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(ConsoleSpanExporter(formatter=format_console_span))
        )
    elif span_exporter == "noop":
        tracer_provider.add_span_processor(SimpleSpanProcessor(NoopSpanExporter()))

    trace.set_tracer_provider(tracer_provider)

    global app_tracer
    # todo(maximsmol): setup trace sampling based on datadog settings
    # todo(maximsmol): port over stuff from https://github.com/open-telemetry/opentelemetry-python-contrib/blob/934af7ea4f9b1e0294ced6a014d6eefdda156b2b/exporter/opentelemetry-exporter-datadog/src/opentelemetry/exporter/datadog/exporter.py

    frame = sys._getframe(1)
    try:
        app_tracer = trace.get_tracer(frame.f_globals["__name__"])
    finally:
        del frame


T = TypeVar("T")
P = ParamSpec("P")


def _trace_function_with_span_async(
    tracer: Tracer,
) -> Callable[
    [Callable[Concatenate[Span, P], Awaitable[T]]], Callable[P, Awaitable[T]]
]:
    def decorator(
        f: Callable[Concatenate[Span, P], Awaitable[T]],
    ) -> Callable[P, Awaitable[T]]:
        @functools.wraps(f)
        async def inner(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_as_current_span(
                f.__qualname__,
                attributes={
                    "code.function": f.__name__,
                    "code.namespace": f.__module__,
                },
            ) as s:
                return await f(s, *args, **kwargs)

        return inner

    return decorator


def trace_function_with_span(
    tracer: Tracer,
) -> Callable[[Callable[Concatenate[Span, P], T]], Callable[P, T]]:
    def decorator(f: Callable[Concatenate[Span, P], T]) -> Callable[P, T]:
        @functools.wraps(f)
        def inner(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_as_current_span(
                f.__qualname__,
                attributes={
                    "code.function": f.__name__,
                    "code.namespace": f.__module__,
                },
            ) as s:
                return f(s, *args, **kwargs)

        if inspect.iscoroutinefunction(f):
            return _trace_function_with_span_async(tracer)(f)

        return inner

    return decorator


def trace_app_function_with_span(
    f: Callable[Concatenate[Span, P], T],
) -> Callable[P, T]:
    return trace_function_with_span(app_tracer)(f)


def _trace_function_async(
    tracer: Tracer,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    def decorator(f: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @_trace_function_with_span_async(tracer)
        @functools.wraps(f)
        async def inner(span: Span, *args: P.args, **kwargs: P.kwargs) -> T:
            return await f(*args, **kwargs)

        return inner

    return decorator


def trace_function(tracer: Tracer) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(f: Callable[P, T]) -> Callable[P, T]:
        @trace_function_with_span(tracer)
        @functools.wraps(f)
        def inner(span: Span, *args: P.args, **kwargs: P.kwargs) -> T:
            return f(*args, **kwargs)

        if inspect.iscoroutinefunction(f):
            return _trace_function_async(tracer)(f)

        return inner

    return decorator


def trace_app_function(f: Callable[P, T]) -> Callable[P, T]:
    return trace_function(app_tracer)(f)


AttributesDict: TypeAlias = "dict[str, LabelValue | None | AttributesDict]"


def dict_to_attrs(x: AttributesDict, prefix: str) -> dict[str, AttributeValue]:
    res: Attributes = {}

    def inner(x: LabelValue | AttributesDict | None, prefix: str) -> None:
        if isinstance(x, list):
            for i, y in enumerate(x):
                inner(y, f"{prefix}.{i}")
            return

        if isinstance(x, dict):
            for k, v in x.items():
                inner(v, f"{prefix}.{k}")
            return

        if x is None:
            x = repr(None)

        res[prefix] = x

    inner(x, f"{prefix}")

    return res
