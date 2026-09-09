# Copyright (c) Microsoft. All rights reserved.

"""AG-UI handoff workflow demo 2 backend.

Durable SQLite authorities + a multi-worker-friendly run shape:

1. AG-UI Thread Snapshot — UI transcript
2. SqliteHistoryProvider — LLM history keyed by AgentSession.session_id
3. SqliteCheckpointStorage — resume across processes

Each request builds a **fresh** workflow from ``workflow_id``. When the body
includes ``resume``, the host resolves ``checkpoint_id`` by looking up the
interrupt id in ``pending_request_info_events`` and injects it into
``forwardedProps`` so the AG-UI / MAF stack can restore. Requests without
``resume`` run as a new workflow turn (no checkpoint).

Also keeps ``_with_latest_user_turn`` so snapshot reconstruction can be full
while ``workflow.run(message=...)`` only sees the latest user turn.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import random
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import uvicorn
from agent_framework import (
    Agent,
    Message,
    Workflow,
    WorkflowCheckpoint,
    tool,
)
from agent_framework.ag_ui import AgentFrameworkWorkflow, add_agent_framework_fastapi_endpoint
from agent_framework.orchestrations import HandoffBuilder
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sqlite_stores import (
    DemoSqliteStore,
    SqliteAGUIThreadSnapshotStore,
    SqliteCheckpointStorage,
    SqliteHistoryProvider,
)

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_BASE_URL = "https://eloquent-reseal-viewless.ngrok-free.dev/v1"
DEFAULT_OPENAI_MODEL = "cyankiwi/Qwen3.8-27B-AWQ-INT4"
DEFAULT_SNAPSHOT_SCOPE = "local-demo"
DEFAULT_WORKFLOW_ID = "handoff_support"
WORKFLOW_NAME = "ag_ui_handoff2_workflow_demo"
_CHECKPOINT_REQUEST_OWNER_KEY = "ag_ui_workflow_request_owner"
_SNAPSHOT_SCOPE_INPUT_KEY = "__ag_ui_snapshot_scope"
# Handoff HITL request payloads are pickled into checkpoints and must be allowlisted to reload.
_ALLOWED_CHECKPOINT_TYPES = [
    "agent_framework_orchestrations._handoff:HandoffAgentUserRequest",
    "types:GenericAlias",
]


def _message_role(message: Message | Any) -> str | None:
    role = getattr(message, "role", None)
    if role is None and isinstance(message, Mapping):
        role = message.get("role")
    if role is None:
        return None
    return str(getattr(role, "value", role))


def _latest_user_turn_messages(messages: Sequence[Message]) -> list[Message]:
    last_user = -1
    for index, message in enumerate(messages):
        if _message_role(message) == "user":
            last_user = index
    if last_user < 0:
        return list(messages)
    return list(messages[last_user:])


def _with_latest_user_turn(workflow: Workflow) -> Workflow:
    """Patch ``workflow.run`` so AG-UI still receives a real ``Workflow`` instance.

    Full snapshot history can be reconstructed for the UI, while the handoff graph
    only sees the latest user turn (plus any resume/checkpoint path unchanged).
    """

    original_run = workflow.run

    def run(message: Any | None = None, **kwargs: Any) -> Any:
        if (
            message is not None
            and kwargs.get("responses") is None
            and kwargs.get("checkpoint_id") is None
            and isinstance(message, list)
            and message
            and all(isinstance(item, Message) for item in message)
        ):
            message = _latest_user_turn_messages(cast(Sequence[Message], message))
        return original_run(message, **kwargs)

    workflow.run = run  # type: ignore[method-assign]
    return workflow


@tool(approval_mode="always_require")
def submit_refund(refund_description: str, amount: str, order_id: str) -> str:
    """Capture a refund request for manual review before processing."""
    return f"refund recorded for order {order_id} (amount: {amount}) with details: {refund_description}"


@tool(approval_mode="always_require")
def submit_replacement(order_id: str, shipping_preference: str, replacement_note: str) -> str:
    """Capture a replacement request for manual review before processing."""
    return (
        f"replacement recorded for order {order_id} (shipping: {shipping_preference}) with details: {replacement_note}"
    )


@tool(approval_mode="never_require")
def lookup_order_details(order_id: str) -> dict[str, str]:
    """Return synthetic order details for a given order ID."""
    normalized_order_id = "".join(ch for ch in order_id if ch.isdigit()) or order_id
    rng = random.Random(normalized_order_id)
    catalog = [
        "Wireless Headphones",
        "Mechanical Keyboard",
        "Gaming Mouse",
        "27-inch Monitor",
        "USB-C Dock",
        "Bluetooth Speaker",
        "Laptop Stand",
    ]
    item_name = catalog[rng.randrange(len(catalog))]
    amount = f"${rng.randint(39, 349)}.{rng.randint(0, 99):02d}"
    purchase_date = f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    return {
        "order_id": normalized_order_id,
        "item_name": item_name,
        "amount": amount,
        "currency": "USD",
        "purchase_date": purchase_date,
        "status": "delivered",
    }


def create_client() -> Any:
    """Create the chat client selected by ``CHAT_PROVIDER``."""

    provider = os.getenv("CHAT_PROVIDER", "openai").strip().casefold()
    if provider in {"foundry", "azure"}:
        from agent_framework.foundry import FoundryChatClient
        from azure.identity import AzureCliCredential

        return FoundryChatClient(
            project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
            model=os.environ["FOUNDRY_MODEL"],
            credential=AzureCliCredential(),
        )
    if provider not in {"openai", "openai-compatible", "lmstudio"}:
        raise ValueError("CHAT_PROVIDER must be 'openai' or 'foundry'.")

    from agent_framework.openai import OpenAIChatCompletionClient

    return OpenAIChatCompletionClient(
        model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
        base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
    )


def is_case_complete_text(text: str) -> bool:
    """Return True when a message ends with the explicit demo completion marker."""

    return text.strip().lower().endswith("case complete.")


def _termination_condition(conversation: list[Message]) -> bool:
    """Stop when any assistant emits an explicit completion marker."""

    for message in reversed(conversation):
        if message.role != "assistant":
            continue
        if is_case_complete_text(message.text or ""):
            return True
    return False


def create_agents(*, db: DemoSqliteStore) -> tuple[Agent, Agent, Agent]:
    """Create triage/refund/order agents with a shared session-id HistoryProvider."""

    client = create_client()
    history = SqliteHistoryProvider(db)

    triage = Agent(
        id="triage_agent",
        name="triage_agent",
        instructions=(
            "You are the customer support triage agent.\n"
            "Routing policy:\n"
            "1. Route refund-related requests to refund_agent.\n"
            "2. Route replacement/shipping requests to order_agent.\n"
            "3. Do not force replacement if the user asked for refund only.\n"
            "4. If the issue is fully resolved, send a concise wrap-up that ends with exactly: Case complete."
        ),
        client=client,
        context_providers=[history],
        require_per_service_call_history_persistence=True,
    )

    refund = Agent(
        id="refund_agent",
        name="refund_agent",
        instructions=(
            "You are the refund specialist.\n"
            "Workflow policy:\n"
            "1. If order_id is missing, ask only for order_id.\n"
            "2. Once order_id is available, call lookup_order_details(order_id) to retrieve item and amount.\n"
            "3. Do not ask the customer how much they paid unless lookup_order_details fails.\n"
            "4. If user intent is ambiguous, ask one clear choice question and wait for the answer:\n"
            "   refund only, replacement only, or both.\n"
            "   Do not call submit_refund until this choice is known.\n"
            "5. Gather a short refund reason from user context if needed.\n"
            "6. If the user wants a refund (refund-only or both),\n"
            "   call submit_refund with order_id, amount (from lookup), and refund_description.\n"
            "7. After approval and successful refund submission:\n"
            "   - If the user explicitly requested replacement/exchange, handoff to order_agent.\n"
            "   - If the user asked for refund only, do not hand off for replacement.\n"
            "     Finalize in this agent and end with exactly: Case complete.\n"
            "8. If the user wants replacement only and no refund, handoff to order_agent directly."
        ),
        client=client,
        tools=[lookup_order_details, submit_refund],
        context_providers=[history],
        require_per_service_call_history_persistence=True,
    )

    order = Agent(
        id="order_agent",
        name="order_agent",
        instructions=(
            "You are the order specialist.\n"
            "Only handle replacement/exchange/shipping tasks.\n"
            "1. If replacement intent is confirmed but shipping preference is missing,\n"
            "   ask for shipping preference (standard or expedited).\n"
            "2. If order_id is missing, ask for order_id.\n"
            "3. Once order_id and shipping preference are known,\n"
            "   call submit_replacement(order_id, shipping_preference, replacement_note).\n"
            "4. While the replacement tool call is pending approval, do not claim completion.\n"
            "5. If you receive a submit_replacement function result,\n"
            "   approval has already occurred and submission succeeded.\n"
            "6. Immediately send a final customer-facing confirmation and end with exactly: Case complete.\n"
            "If the user wants refund only and no replacement, do not ask shipping questions.\n"
            "Acknowledge and hand off back to triage_agent for final closure.\n"
            "Do not fabricate tool outputs."
        ),
        client=client,
        tools=[lookup_order_details, submit_replacement],
        context_providers=[history],
        require_per_service_call_history_persistence=True,
    )

    return triage, refund, order


def build_handoff_support_workflow(*, db: DemoSqliteStore) -> Workflow:
    """Concrete builder for workflow_id ``handoff_support``."""

    triage, refund, order = create_agents(db=db)
    builder = HandoffBuilder(
        name=WORKFLOW_NAME,
        participants=[triage, refund, order],
        termination_condition=_termination_condition,
    )
    (
        builder.add_handoff(
            triage,
            [refund],
            description="Route when the user requests refunds, damaged-item claims, or refund status updates.",
        )
        .add_handoff(
            triage,
            [order],
            description="Route when the user requests replacement, exchange, shipping preference, or shipment changes.",
        )
        .add_handoff(
            refund,
            [order],
            description="Route after refund work only if replacement/exchange logistics are explicitly needed.",
        )
        .add_handoff(
            refund,
            [triage],
            description="Route back for final case closure when refund-only work is complete.",
        )
        .add_handoff(
            order,
            [triage],
            description="Route back after replacement/shipping tasks are complete for final closure.",
        )
        .add_handoff(
            order,
            [refund],
            description="Route to refund specialist if the user pivots from replacement to refund processing.",
        )
    )
    return _with_latest_user_turn(builder.with_start_agent(triage).build())


WORKFLOW_BUILDERS: dict[str, Callable[..., Workflow]] = {
    DEFAULT_WORKFLOW_ID: build_handoff_support_workflow,
}


def build_workflow_by_id(*, workflow_id: str, db: DemoSqliteStore) -> Workflow:
    """Resolve ``workflow_id`` → builder and construct a fresh Workflow instance."""

    builder = WORKFLOW_BUILDERS.get(workflow_id)
    if builder is None:
        known = ", ".join(sorted(WORKFLOW_BUILDERS))
        raise KeyError(f"Unknown workflow_id={workflow_id!r}. Known: {known}")
    logger.info("Building workflow_id=%s", workflow_id)
    return builder(db=db)


def _workflow_id_from_input(input_data: Mapping[str, Any]) -> str:
    for key in ("workflow_id", "workflowId"):
        value = input_data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    forwarded = input_data.get("forwarded_props") or input_data.get("forwardedProps") or {}
    if isinstance(forwarded, dict):
        for key in ("workflow_id", "workflowId"):
            value = forwarded.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return DEFAULT_WORKFLOW_ID


def _checkpoint_id_from_input(input_data: Mapping[str, Any]) -> str | None:
    forwarded = input_data.get("forwarded_props") or input_data.get("forwardedProps") or {}
    if not isinstance(forwarded, dict):
        return None
    value = forwarded.get("checkpoint_id") or forwarded.get("checkpointId")
    if value is None:
        return None
    return str(value)


def _inject_checkpoint_id(input_data: dict[str, Any], checkpoint_id: str) -> None:
    forwarded = input_data.get("forwarded_props") or input_data.get("forwardedProps")
    if not isinstance(forwarded, dict):
        forwarded = {}
    forwarded = dict(forwarded)
    forwarded["checkpoint_id"] = checkpoint_id
    input_data["forwardedProps"] = forwarded
    input_data["forwarded_props"] = forwarded


def _resume_interrupt_ids(input_data: Mapping[str, Any]) -> set[str]:
    """Collect interrupt ids from an AG-UI ``resume`` payload."""

    resume = input_data.get("resume")
    if resume is None:
        return set()

    entries: list[Any]
    if isinstance(resume, list):
        entries = list(resume)
    elif isinstance(resume, Mapping):
        raw = resume.get("interrupts") or resume.get("interrupt")
        if isinstance(raw, list):
            entries = list(raw)
        elif raw is not None:
            entries = [raw]
        elif any(key in resume for key in ("id", "interruptId", "interrupt_id")):
            entries = [resume]
        else:
            entries = []
    else:
        entries = []

    ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        interrupt_id = entry.get("id") or entry.get("interruptId") or entry.get("interrupt_id")
        if interrupt_id is not None:
            ids.add(str(interrupt_id))
    return ids


def _checkpoint_owner(checkpoint: WorkflowCheckpoint) -> tuple[str | None, str | None] | None:
    raw_owner = checkpoint.metadata.get(_CHECKPOINT_REQUEST_OWNER_KEY)
    if not isinstance(raw_owner, dict):
        return None
    scope = raw_owner.get("snapshot_scope")
    thread_id = raw_owner.get("thread_id")
    if scope is not None and not isinstance(scope, str):
        return None
    if thread_id is not None and not isinstance(thread_id, str):
        return None
    return scope, thread_id


def _pending_ids(checkpoint: WorkflowCheckpoint) -> set[str]:
    return {str(request_id) for request_id in checkpoint.pending_request_info_events}


async def _checkpoint_id_for_interrupts(
    storage: SqliteCheckpointStorage,
    *,
    interrupt_ids: set[str],
    snapshot_scope: str | None,
    thread_id: str,
) -> str | None:
    """Find the checkpoint whose pending request_info map contains the resume interrupt id(s)."""

    if not interrupt_ids:
        return None

    matches: list[WorkflowCheckpoint] = []
    for checkpoint in await storage.list_checkpoints(workflow_name=WORKFLOW_NAME):
        if not interrupt_ids.intersection(_pending_ids(checkpoint)):
            continue
        owner = _checkpoint_owner(checkpoint)
        if owner is not None and owner != (snapshot_scope, thread_id):
            continue
        matches.append(checkpoint)

    if not matches:
        return None
    latest = max(matches, key=lambda item: datetime.fromisoformat(item.timestamp))
    return latest.checkpoint_id


class DemoHandoffWorkflow(AgentFrameworkWorkflow):
    """Per-request workflow factory + interrupt→checkpoint resume for multi-worker demos."""

    def __init__(
        self,
        *,
        db: DemoSqliteStore,
        snapshot_store: SqliteAGUIThreadSnapshotStore,
        checkpoint_storage: SqliteCheckpointStorage,
    ) -> None:
        self._db = db
        self._checkpoint_storage = checkpoint_storage
        self._thread_workflow_id: dict[str, str] = {}

        def workflow_factory(thread_id: str) -> Workflow:
            workflow_id = self._thread_workflow_id.get(thread_id)
            if workflow_id is None:
                raise RuntimeError(
                    f"No workflow_id bound for thread_id={thread_id!r}. "
                    "DemoHandoffWorkflow.run must bind it before super().run()."
                )
            return build_workflow_by_id(workflow_id=workflow_id, db=db)

        super().__init__(
            workflow_factory=workflow_factory,
            name=WORKFLOW_NAME,
            description="Handoff demo 2: fresh workflow per request; resume via interrupt→checkpoint lookup.",
            snapshot_store=snapshot_store,
            checkpoint_storage=checkpoint_storage,
        )

    async def run(self, input_data: dict[str, Any]) -> AsyncGenerator[Any]:
        """Bind workflow_id, resolve checkpoint on resume, then delegate to MAF AG-UI."""

        thread_id = self._thread_id_from_input(input_data)
        workflow_id = _workflow_id_from_input(input_data)
        input_data["thread_id"] = thread_id
        input_data["workflow_id"] = workflow_id

        if workflow_id not in WORKFLOW_BUILDERS:
            known = ", ".join(sorted(WORKFLOW_BUILDERS))
            raise KeyError(f"Unknown workflow_id={workflow_id!r}. Known: {known}")

        snapshot_scope = input_data.get(_SNAPSHOT_SCOPE_INPUT_KEY)
        if not isinstance(snapshot_scope, str):
            snapshot_scope = DEFAULT_SNAPSHOT_SCOPE

        # Always build a fresh in-memory instance (multi-worker friendly).
        self.clear_thread_workflow(thread_id, snapshot_scope)
        self._thread_workflow_id[thread_id] = workflow_id

        if _checkpoint_id_from_input(input_data) is None:
            interrupt_ids = _resume_interrupt_ids(input_data)
            if interrupt_ids:
                checkpoint_id = await _checkpoint_id_for_interrupts(
                    self._checkpoint_storage,
                    interrupt_ids=interrupt_ids,
                    snapshot_scope=snapshot_scope,
                    thread_id=thread_id,
                )
                if checkpoint_id is None:
                    raise LookupError(
                        f"No pending checkpoint found for interrupt id(s) {sorted(interrupt_ids)} "
                        f"on thread={thread_id!r}."
                    )
                logger.info(
                    "Resolved checkpoint_id=%s from interrupt id(s) %s thread=%s",
                    checkpoint_id,
                    sorted(interrupt_ids),
                    thread_id,
                )
                _inject_checkpoint_id(input_data, checkpoint_id)

        async for event in super().run(input_data):
            yield event

        # Do not keep process-local instances across requests.
        self.clear_thread_workflow(thread_id, snapshot_scope)
        self._thread_workflow_id.pop(thread_id, None)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(title="AG-UI Handoff Workflow Demo 2")

    cors_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://127.0.0.1:5175").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    data_dir = Path(os.getenv("AF_DATA_DIR", str(BACKEND_DIR / "data")))
    db_path = Path(os.getenv("AF_SQLITE_DB", str(data_dir / "handoff2.sqlite")))
    db = DemoSqliteStore(db_path)
    snapshot_store = SqliteAGUIThreadSnapshotStore(db)
    checkpoint_storage = SqliteCheckpointStorage(db, allowed_checkpoint_types=_ALLOWED_CHECKPOINT_TYPES)

    demo_workflow = DemoHandoffWorkflow(
        db=db,
        snapshot_store=snapshot_store,
        checkpoint_storage=checkpoint_storage,
    )

    def resolve_snapshot_scope(_request: Any = None) -> str:
        return os.getenv("SNAPSHOT_SCOPE", DEFAULT_SNAPSHOT_SCOPE)

    add_agent_framework_fastapi_endpoint(
        app=app,
        agent=demo_workflow,
        path="/handoff2_demo",
        snapshot_store=snapshot_store,
        snapshot_scope_resolver=resolve_snapshot_scope,
        checkpoint_storage=checkpoint_storage,
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:  # pyright: ignore[reportUnusedFunction]
        return {
            "status": "ok",
            "sqlite_db": str(db_path),
            "endpoint": "/handoff2_demo",
            "default_workflow_id": DEFAULT_WORKFLOW_ID,
        }

    return app


app = create_app()


def main() -> None:
    """Run the AG-UI handoff2 demo backend."""

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ag_ui_handoff2_demo.log")
    try:
        file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        print(f"Logging to file: {log_file}")
    except Exception as exc:
        print(f"Warning: Failed to set up file logging: {exc}")

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8892"))

    print(f"AG-UI handoff2 demo backend running at http://{host}:{port}")
    print("AG-UI endpoint: POST /handoff2_demo")
    print("Each request: new workflow from workflow_id; resume resolves checkpoint via interrupt id.")
    print("Known workflows:", ", ".join(sorted(WORKFLOW_BUILDERS)))
    print("SQLite: snapshot (UI) + history (LLM) + checkpoints")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
