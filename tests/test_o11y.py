from unittest.mock import patch

with patch.dict(
    "os.environ", {"DD_VERSION": "test", "DD_ENV": "test", "DD_SERVICE": "test"}
):
    from latch_o11y.o11y import dict_to_attrs


def test_aws_request_params() -> None:
    assert dict_to_attrs({"SecretId": "test-secret"}, "rpc.aws-api.params") == {
        "rpc.aws-api.params.SecretId": "test-secret"
    }


def test_nested_values() -> None:
    assert dict_to_attrs(
        {"nested": {"items": ["value", {"enabled": True}, None]}}, "params"
    ) == {
        "params.nested.items.0": "value",
        "params.nested.items.1.enabled": True,
        "params.nested.items.2": "None",
    }


def test_scalar_values() -> None:
    assert dict_to_attrs(
        {"count": 2, "ratio": 0.5, "enabled": False, "missing": None}, "params"
    ) == {
        "params.count": 2,
        "params.ratio": 0.5,
        "params.enabled": False,
        "params.missing": "None",
    }


def test_empty_containers() -> None:
    assert dict_to_attrs({}, "params") == {}
    assert dict_to_attrs({"dict": {}, "list": []}, "params") == {}
