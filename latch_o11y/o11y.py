import functools
import inspect
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Concatenate, ParamSpec, TypeAlias, TypeVar

import orjson
import structlog as slog
from latch_config.config import DatadogConfig, LoggingMode, read_config
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Attributes, LabelValue, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer
from opentelemetry.trace.span import INVALID_SPAN, INVALID_SPAN_CONTEXT, Span
from structlog.types import EventDict, WrappedLogger


@dataclass(frozen=True)
class Config:
    datadog: DatadogConfig
    logging_mode: LoggingMode = LoggingMode.console_json


config = read_config(Config)


def add_timestamp(logger: WrappedLogger, name: str, x: EventDict) -> EventDict:
    # the default slog.processors.TimeStamper is not great with timezones
    utc = datetime.now(timezone.utc)
    x["timestamp"] = utc.astimezone().isoformat()
    return x


def add_dd_otel(logger: WrappedLogger, name: str, x: EventDict) -> EventDict:
    s = trace.get_current_span()
    if s == INVALID_SPAN:
        return x

    ctx = s.get_span_context()
    if ctx == INVALID_SPAN_CONTEXT:
        return x

    x["trace_id"] = format(ctx.trace_id, "032x")
    x["span_id"] = format(ctx.span_id, "016x")

    x["dd.trace_id"] = str(ctx.trace_id & 0xFFFF_FFFF_FFFF_FFFF)
    x["dd.span_id"] = str(ctx.span_id)

    x["dd.service"] = config.datadog.service_name
    x["dd.version"] = config.datadog.service_version
    x["dd.env"] = config.datadog.deployment_environment

    return x


app_tracer: Tracer
# fixme(maximsmol): add support for .exception
log: slog.stdlib.BoundLogger


def setup():
    if config.logging_mode == LoggingMode.file_json:
        log_file = Path(__file__).parent.parent / "o11y_debug" / "log.json"
    else:
        log_file = Path("/dev/stdout")

    service_data: Attributes = {
        "service.name": config.datadog.service_name,
        "service.version": config.datadog.service_version,
        "deployment.environment": config.datadog.deployment_environment,
    }

    tracer_provider = TracerProvider(resource=Resource(service_data))
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    global app_tracer
    # todo(maximsmol): setup trace sampling based on datadog settings
    # todo(maximsmol): port over stuff from https://github.com/open-telemetry/opentelemetry-python-contrib/blob/934af7ea4f9b1e0294ced6a014d6eefdda156b2b/exporter/opentelemetry-exporter-datadog/src/opentelemetry/exporter/datadog/exporter.py
    app_tracer = trace.get_tracer(__name__)

    slog.configure_once(
        processors=[
            # todo(maximsmol): allow named loggers (needs a custom logging base)
            # slog.stdlib.add_logger_name,
            slog.stdlib.add_log_level,
            add_timestamp,
            add_dd_otel,
        ]
        + (
            [slog.dev.ConsoleRenderer()]
            if config.logging_mode == LoggingMode.console
            else [
                slog.processors.ExceptionRenderer(
                    slog.tracebacks.ExceptionDictTransformer()
                ),
                slog.processors.JSONRenderer(orjson.dumps),
            ]
        ),
        logger_factory=slog.BytesLoggerFactory(file=log_file.open("wb"))
        if config.logging_mode != LoggingMode.console
        else None,
        wrapper_class=slog.stdlib.AsyncBoundLogger,
        cache_logger_on_first_use=True,
    )

    global log
    log = slog.stdlib.get_logger()


T = TypeVar("T")
P = ParamSpec("P")


def _trace_function_with_span_async(
    tracer: Tracer,
) -> Callable[
    [Callable[Concatenate[Span, P], Awaitable[T]]], Callable[P, Awaitable[T]]
]:
    def decorator(
        f: Callable[Concatenate[Span, P], Awaitable[T]]
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
    f: Callable[Concatenate[Span, P], T]
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


AttributesDict: TypeAlias = dict[str, LabelValue | "AttributesDict"]


def dict_to_attrs(x: AttributesDict, prefix: str) -> Attributes:
    res: Attributes = {}

    def inner(x: LabelValue | AttributesDict, prefix: str):
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
