# Copyright (c) Microsoft. All rights reserved.

import asyncio
import contextlib
import logging
from asyncio import Event
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from semantic_kernel.agents.runtime.core import CoreAgentId
from semantic_kernel.agents.runtime.core.agent_type import CoreAgentType
from semantic_kernel.agents.runtime.core.base_agent import BaseAgent
from semantic_kernel.agents.runtime.core.message_context import MessageContext
from semantic_kernel.agents.runtime.core.routed_agent import RoutedAgent, event, message_handler
from semantic_kernel.agents.runtime.core.serialization import try_get_known_serializers_for_type
from semantic_kernel.agents.runtime.core.topic import TopicId
from semantic_kernel.agents.runtime.in_process.agent_instantiation_context import AgentInstantiationContext
from semantic_kernel.agents.runtime.in_process.default_subscription import default_subscription, type_subscription
from semantic_kernel.agents.runtime.in_process.default_topic import DefaultTopicId
from semantic_kernel.agents.runtime.in_process.in_process_runtime import InProcessRuntime
from semantic_kernel.agents.runtime.in_process.type_subscription import TypeSubscription


@dataclass
class MessageType: ...


@dataclass
class CascadingMessageType:
    round: int


@dataclass
class ContentMessage:
    content: str


class LoopbackAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("A loop back agent.")
        self.num_calls = 0
        self.received_messages: list[Any] = []
        self.event = Event()

    @message_handler
    async def on_new_message(
        self, message: MessageType | ContentMessage, ctx: MessageContext
    ) -> MessageType | ContentMessage:
        self.num_calls += 1
        self.received_messages.append(message)
        self.event.set()
        return message


@default_subscription
class LoopbackAgentWithDefaultSubscription(LoopbackAgent): ...


@default_subscription
class CascadingAgent(RoutedAgent):
    def __init__(self, max_rounds: int) -> None:
        super().__init__("A cascading agent.")
        self.num_calls = 0
        self.max_rounds = max_rounds

    @message_handler
    async def on_new_message(self, message: CascadingMessageType, ctx: MessageContext) -> None:
        self.num_calls += 1
        if message.round == self.max_rounds:
            return
        await self.publish_message(CascadingMessageType(round=message.round + 1), topic_id=DefaultTopicId())


class NoopAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("A no op agent")

    async def on_message_impl(self, message: Any, ctx: MessageContext) -> Any:
        raise NotImplementedError


class MyTestExporter(SpanExporter):
    def __init__(self) -> None:
        self.exported_spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.exported_spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def clear(self) -> None:
        """Clears the list of exported spans."""
        self.exported_spans.clear()

    def get_exported_spans(self) -> list[ReadableSpan]:
        """Returns the list of exported spans."""
        return self.exported_spans


def get_test_tracer_provider(exporter: MyTestExporter) -> TracerProvider:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return tracer_provider


test_exporter = MyTestExporter()


class FakeAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("A fake agent")

    async def on_message_impl(self, message: Any, ctx: MessageContext) -> Any:
        raise NotImplementedError


@pytest.fixture
def tracer_provider() -> TracerProvider:
    test_exporter.clear()
    return get_test_tracer_provider(test_exporter)


@pytest.mark.asyncio
async def test_agent_type_register_factory() -> None:
    runtime = InProcessRuntime()

    def agent_factory() -> NoopAgent:
        id = AgentInstantiationContext.current_agent_id()
        assert id == CoreAgentId("name1", "default")
        agent = NoopAgent()
        assert agent.id == id
        return agent

    await runtime.register_factory(type=CoreAgentType("name1"), agent_factory=agent_factory, expected_class=NoopAgent)

    with pytest.raises(ValueError):
        # This should fail because the expected class does not match the actual class.
        await runtime.register_factory(
            type=CoreAgentType("name1"),
            agent_factory=agent_factory,  # type: ignore
            expected_class=FakeAgent,
        )

    # Without expected_class, no error.
    await runtime.register_factory(type=CoreAgentType("name2"), agent_factory=agent_factory)


@pytest.mark.asyncio
async def test_agent_type_must_be_unique() -> None:
    runtime = InProcessRuntime()

    def agent_factory() -> NoopAgent:
        id = AgentInstantiationContext.current_agent_id()
        assert id == CoreAgentId("name1", "default")
        agent = NoopAgent()
        assert agent.id == id
        return agent

    await NoopAgent.register(runtime, "name1", agent_factory)

    with pytest.raises(ValueError):
        await runtime.register_factory(
            type=CoreAgentType("name1"), agent_factory=agent_factory, expected_class=NoopAgent
        )

    await runtime.register_factory(type=CoreAgentType("name2"), agent_factory=agent_factory, expected_class=NoopAgent)


@pytest.mark.asyncio
async def test_register_receives_publish(tracer_provider: TracerProvider) -> None:
    runtime = InProcessRuntime(tracer_provider=tracer_provider)

    runtime.add_message_serializer(try_get_known_serializers_for_type(MessageType))
    await runtime.register_factory(
        type=CoreAgentType("name"), agent_factory=lambda: LoopbackAgent(), expected_class=LoopbackAgent
    )
    await runtime.add_subscription(TypeSubscription("default", "name"))

    runtime.start()
    await runtime.publish_message(MessageType(), topic_id=TopicId("default", "default"))
    await runtime.stop_when_idle()

    # Agent in default namespace should have received the message
    long_running_agent = await runtime.try_get_underlying_agent_instance(
        CoreAgentId("name", "default"),
        type=LoopbackAgent,
    )
    assert long_running_agent.num_calls == 1

    # Agent in other namespace should not have received the message
    other_long_running_agent: LoopbackAgent = await runtime.try_get_underlying_agent_instance(
        CoreAgentId("name", key="other"), type=LoopbackAgent
    )
    assert other_long_running_agent.num_calls == 0

    await runtime.close()


@pytest.mark.asyncio
async def test_register_receives_publish_with_construction(caplog: pytest.LogCaptureFixture) -> None:
    runtime = InProcessRuntime()

    runtime.add_message_serializer(try_get_known_serializers_for_type(MessageType))

    async def agent_factory() -> LoopbackAgent:
        raise ValueError("test")

    await runtime.register_factory(
        type=CoreAgentType("name"), agent_factory=agent_factory, expected_class=LoopbackAgent
    )
    await runtime.add_subscription(TypeSubscription("default", "name"))

    with caplog.at_level(logging.ERROR):
        runtime.start()
        await runtime.publish_message(MessageType(), topic_id=TopicId("default", "default"))
        await runtime.stop_when_idle()

    # Check if logger has the exception.
    assert any("Error constructing agent" in e.message for e in caplog.records)

    await runtime.close()


@pytest.mark.asyncio
async def test_register_receives_publish_cascade() -> None:
    num_agents = 5
    num_initial_messages = 5
    max_rounds = 5
    total_num_calls_expected = 0
    for i in range(0, max_rounds):
        total_num_calls_expected += num_initial_messages * ((num_agents - 1) ** i)

    runtime = InProcessRuntime()

    # Register agents
    for i in range(num_agents):
        await CascadingAgent.register(runtime, f"name{i}", lambda: CascadingAgent(max_rounds))

    runtime.start()

    # Publish messages
    for _ in range(num_initial_messages):
        await runtime.publish_message(CascadingMessageType(round=1), DefaultTopicId())

    # Process until idle.
    await runtime.stop_when_idle()

    # Check that each agent received the correct number of messages.
    for i in range(num_agents):
        agent = await runtime.try_get_underlying_agent_instance(CoreAgentId(f"name{i}", "default"), CascadingAgent)
        assert agent.num_calls == total_num_calls_expected

    await runtime.close()


@pytest.mark.asyncio
async def test_register_factory_explicit_name() -> None:
    runtime = InProcessRuntime()

    await LoopbackAgent.register(runtime, "name", LoopbackAgent)
    await runtime.add_subscription(TypeSubscription("default", "name"))

    runtime.start()
    agent_id = CoreAgentId("name", key="default")
    topic_id = TopicId("default", "default")
    await runtime.publish_message(MessageType(), topic_id=topic_id)

    await runtime.stop_when_idle()

    # Agent in default namespace should have received the message
    long_running_agent = await runtime.try_get_underlying_agent_instance(agent_id, type=LoopbackAgent)
    assert long_running_agent.num_calls == 1

    # Agent in other namespace should not have received the message
    other_long_running_agent: LoopbackAgent = await runtime.try_get_underlying_agent_instance(
        CoreAgentId("name", key="other"), type=LoopbackAgent
    )
    assert other_long_running_agent.num_calls == 0

    await runtime.close()


@pytest.mark.asyncio
async def test_default_subscription() -> None:
    runtime = InProcessRuntime()
    runtime.start()

    await LoopbackAgentWithDefaultSubscription.register(runtime, "name", LoopbackAgentWithDefaultSubscription)

    agent_id = CoreAgentId("name", key="default")
    await runtime.publish_message(MessageType(), topic_id=DefaultTopicId())

    await runtime.stop_when_idle()

    long_running_agent = await runtime.try_get_underlying_agent_instance(
        agent_id, type=LoopbackAgentWithDefaultSubscription
    )
    assert long_running_agent.num_calls == 1

    other_long_running_agent = await runtime.try_get_underlying_agent_instance(
        CoreAgentId("name", key="other"), type=LoopbackAgentWithDefaultSubscription
    )
    assert other_long_running_agent.num_calls == 0

    await runtime.close()


@pytest.mark.asyncio
async def test_type_subscription() -> None:
    runtime = InProcessRuntime()
    runtime.start()

    @type_subscription(topic_type="Other")
    class LoopbackAgentWithSubscription(LoopbackAgent): ...

    await LoopbackAgentWithSubscription.register(runtime, "name", LoopbackAgentWithSubscription)

    agent_id = CoreAgentId("name", key="default")
    await runtime.publish_message(MessageType(), topic_id=TopicId("Other", "default"))
    await runtime.stop_when_idle()

    long_running_agent = await runtime.try_get_underlying_agent_instance(agent_id, type=LoopbackAgentWithSubscription)
    assert long_running_agent.num_calls == 1

    other_long_running_agent = await runtime.try_get_underlying_agent_instance(
        CoreAgentId("name", key="other"), type=LoopbackAgentWithSubscription
    )
    assert other_long_running_agent.num_calls == 0

    await runtime.close()


@pytest.mark.asyncio
async def test_default_subscription_publish_to_other_source() -> None:
    runtime = InProcessRuntime()
    runtime.start()

    await LoopbackAgentWithDefaultSubscription.register(runtime, "name", LoopbackAgentWithDefaultSubscription)

    agent_id = CoreAgentId("name", key="default")
    await runtime.publish_message(MessageType(), topic_id=DefaultTopicId(source="other"))
    await runtime.stop_when_idle()

    long_running_agent = await runtime.try_get_underlying_agent_instance(
        agent_id, type=LoopbackAgentWithDefaultSubscription
    )
    assert long_running_agent.num_calls == 0

    other_long_running_agent = await runtime.try_get_underlying_agent_instance(
        CoreAgentId("name", key="other"), type=LoopbackAgentWithDefaultSubscription
    )
    assert other_long_running_agent.num_calls == 1

    await runtime.close()


@default_subscription
class FailingAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("A failing agent.")

    @event
    async def on_new_message_event(self, message: MessageType, ctx: MessageContext) -> None:
        raise ValueError("Test exception")


@pytest.mark.asyncio
async def test_event_handler_exception_propagates() -> None:
    runtime = InProcessRuntime(ignore_unhandled_exceptions=False)
    await FailingAgent.register(runtime, "name", FailingAgent)

    with pytest.raises(ValueError, match="Test exception"):
        runtime.start()
        await runtime.publish_message(MessageType(), topic_id=DefaultTopicId())
        await runtime.stop_when_idle()

    await runtime.close()


@pytest.mark.asyncio
async def test_event_handler_exception_multi_message() -> None:
    runtime = InProcessRuntime(ignore_unhandled_exceptions=False)
    await FailingAgent.register(runtime, "name", FailingAgent)

    with pytest.raises(ValueError, match="Test exception"):
        runtime.start()
        await runtime.publish_message(MessageType(), topic_id=DefaultTopicId())
        await runtime.publish_message(MessageType(), topic_id=DefaultTopicId())
        await runtime.publish_message(MessageType(), topic_id=DefaultTopicId())
        await runtime.stop_when_idle()

    await runtime.close()


@pytest.fixture
async def loop_errors() -> list[dict[str, Any]]:
    """Record every event-loop exception-handler invocation, e.g. never-retrieved task errors."""
    records: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()

    def _handler(lp: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        records.append(dict(context))
        if previous is not None:
            previous(lp, context)

    loop.set_exception_handler(_handler)
    yield records
    loop.set_exception_handler(previous)


def _never_retrieved(records: list[dict[str, Any]]) -> list[BaseException]:
    return [
        ctx["exception"]
        for ctx in records
        if ctx.get("message") == "Task exception was never retrieved" and "exception" in ctx
    ]


@pytest.mark.asyncio
async def test_send_message_unknown_recipient_propagates_lookup_error(loop_errors: list[dict[str, Any]]) -> None:
    """An unregistered recipient raises a typed LookupError to the caller with no unretrieved background error."""
    runtime = InProcessRuntime()
    runtime.start()

    raised: BaseException | None = None
    try:
        await runtime.send_message(ContentMessage(content="x"), recipient=CoreAgentId("does-not-exist", "default"))
    except BaseException as e:  # noqa: BLE001 - surface the typed error for assertion
        raised = e

    for _ in range(4):
        await asyncio.sleep(0)

    await runtime.stop()

    assert isinstance(raised, LookupError)
    assert "does-not-exist" in str(raised)
    assert runtime.unprocessed_messages_count == 0
    assert not _never_retrieved(loop_errors)


@pytest.mark.asyncio
async def test_send_message_unknown_recipient_does_not_block_stop_when_idle() -> None:
    """A failed unknown-recipient send must not leave the queue unfinished so stop_when_idle() can drain."""
    runtime = InProcessRuntime()
    runtime.start()

    try:
        with contextlib.suppress(BaseException):
            # Error type is covered by the typed-propagation test; we only need a failed send here.
            await runtime.send_message(ContentMessage(content="x"), recipient=CoreAgentId("ghost", "default"))

        # Guard only: if queue.join() hangs, wait_for raises TimeoutError here.
        await asyncio.wait_for(runtime.stop_when_idle(), timeout=2.0)
    finally:
        if runtime._run_context is not None:
            await runtime.stop()
