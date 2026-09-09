# AG-UI Handoff Workflow Demo 2

Copy of [`ag_ui_workflow_handoff`](../ag_ui_workflow_handoff) with durable SQLite stores and a
**multi-worker-friendly** run shape.

## Model

| Concern | Storage |
|---|---|
| UI transcript / hydrate / HITL cards | SQLite `thread_snapshots` |
| LLM history | SQLite `history_messages` keyed by `AgentSession.session_id` |
| HITL / cold resume | SQLite `checkpoints` |

Each AG-UI request:

1. Reads `workflow_id` from the body (`workflow_id` / `forwardedProps.workflow_id`)
2. Builds a **fresh** workflow instance (no process-local reuse)
3. If the body has `resume`, looks up `checkpoint_id` by interrupt id in
   `pending_request_info_events`, injects `forwardedProps.checkpoint_id`, then runs
4. Otherwise runs as a new turn (MAF handles restore only when `checkpoint_id` is present)

Frontend keeps `thread_id` in `localStorage` + `?thread_id=` and on load POSTs empty
`messages` so the backend can **hydrate** from the Thread Snapshot (chat + pending
interrupts). **Start New Case** allocates a new thread id.

Also keeps **`_with_latest_user_turn`**: full snapshot for UI; only the latest user turn
into `workflow.run(message=...)`.

## Ports

- Backend: `http://127.0.0.1:8892` — `POST /handoff2_demo`
- Frontend: `http://127.0.0.1:5175` (`VITE_WORKFLOW_ID`, default `handoff_support`)

## Run

```bash
# backend
cd python
uv sync
cp samples/05-end-to-end/ag_ui_workflow_handoff2/backend/.env.example \
   samples/05-end-to-end/ag_ui_workflow_handoff2/backend/.env
uv run python samples/05-end-to-end/ag_ui_workflow_handoff2/backend/server.py

# frontend
cd samples/05-end-to-end/ag_ui_workflow_handoff2/frontend
npm install
npm run dev
```

Default chat client is OpenAI-compatible (`CHAT_PROVIDER=openai`). Set `CHAT_PROVIDER=foundry` for Foundry.

SQLite file default: `backend/data/handoff2.sqlite` (`AF_SQLITE_DB` to override).

## Demo flow

Same as handoff1 (refund / replacement / approvals). After `Case complete.`, the next
message is a new workflow build (new AgentSessions). HITL resume does not need the
frontend to send `checkpoint_id`; the backend resolves it from the interrupt id.

## Folder layout

- `backend/server.py` — FastAPI + `AgentFrameworkWorkflow` + builders
- `backend/sqlite_stores.py` — snapshot / history / checkpoint stores
- `frontend/` — Vite + React AG-UI client
