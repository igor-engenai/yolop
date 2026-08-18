from ipaddress import IPv4Address

from yolop_http import EgressPolicy, validate_destination


def test_egress_policy_rejects_private_and_metadata_addresses() -> None:
    policy = EgressPolicy(allowed_schemes=frozenset({"https"}), allowed_ports=frozenset({443}))

    for address in (
        IPv4Address("127.0.0.1"),
        IPv4Address("10.0.0.1"),
        IPv4Address("169.254.169.254"),
    ):
        assert validate_destination(address, policy) is False
