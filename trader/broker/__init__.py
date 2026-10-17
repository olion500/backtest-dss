"""Broker adapters behind one interface (:class:`trader.broker.base.Broker`).

To add a broker: write ``trader/broker/<name>.py`` with a class that implements
``Broker`` (and lists its ``required_credentials``), then register it in ``BROKERS``.
"""

from __future__ import annotations

from typing import Callable, Mapping

from trader.broker.base import Broker, BrokerError, Credentials
from trader.broker.meritz import MeritzBroker

BrokerFactory = Callable[[Credentials, Mapping[str, str]], Broker]

BROKERS: dict[str, BrokerFactory] = {
    "meritz": MeritzBroker,
}


def make_broker(name: str, credentials: Credentials, listings: Mapping[str, str]) -> Broker:
    factory = BROKERS.get(name)
    if factory is None:
        raise BrokerError(f"지원하지 않는 증권사입니다: {name} (가능: {', '.join(sorted(BROKERS))})", code="UNKNOWN_BROKER")
    return factory(credentials, listings)
