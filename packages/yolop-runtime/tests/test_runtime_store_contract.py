from pydantic_ai import AgentSpec
from pytest import raises
from yolop_runtime import (
    ExecutionPin,
    InvalidNamespaceError,
    RuntimeSessionSnapshot,
    SessionPinMismatchError,
    agent_spec_digest,
    ensure_session_pin,
    validate_namespace,
)


def test_agent_spec_digest_is_canonical() -> None:
    first = AgentSpec(
        name="analyst",
        instructions="Answer from the available data.",
        model_settings={"temperature": 0, "max_tokens": 100},
    )
    second = AgentSpec(
        name="analyst",
        instructions="Answer from the available data.",
        model_settings={"max_tokens": 100, "temperature": 0},
    )

    assert agent_spec_digest(first) == agent_spec_digest(second)
    assert agent_spec_digest(first) != agent_spec_digest(AgentSpec(name="other"))


def test_execution_pin_uses_the_canonical_agent_spec_digest() -> None:
    spec = AgentSpec(name="analyst", model="openai:gpt-5.6-luna")

    pin = ExecutionPin.from_spec(spec, model_id="openai:gpt-5.6-luna")

    assert pin.agent_spec_id == agent_spec_digest(spec)
    assert pin.model_id == "openai:gpt-5.6-luna"


def test_namespace_must_be_a_nonempty_host_value() -> None:
    assert validate_namespace("tenant/acme") == "tenant/acme"

    for invalid in ("", "   ", "x" * 256):
        with raises(InvalidNamespaceError):
            validate_namespace(invalid)


def test_session_pin_mismatch_has_a_stable_error() -> None:
    session = RuntimeSessionSnapshot(
        id="00000000-0000-4000-8000-000000000001",
        namespace="tenant/acme",
        pin=ExecutionPin(agent_spec_id="a" * 64, model_id="openai:model-a"),
        messages=[],
        revision="revision",
    )
    expected = ExecutionPin(agent_spec_id="b" * 64, model_id="openai:model-b")

    with raises(SessionPinMismatchError) as raised:
        ensure_session_pin(session, expected)

    assert raised.value.code == "session_pin_mismatch"
