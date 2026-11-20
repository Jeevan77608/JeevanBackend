# Jeevan AI Chatbot — FastAPI Backend

FastAPI backend implementing the Model Context Protocol (MCP) for AI-powered Jira issue search, generation, review, and creation. It also retains Azure DevOps MCP tools for compatibility with existing integrations.

## Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.11 | Runtime |
| FastAPI | >= 0.109.0 | REST API + WebSocket server |
| Uvicorn | >= 0.27.0 | ASGI application server |
| Pydantic | >= 2.5.0 | Data validation and settings |
| Redis | >= 5.0.0 | Session storage (production) |
| Supabase Auth, Postgres, Storage | hosted | Sign-in, saved conversations, private file context |
| Azure Cosmos DB | >= 4.5.0 | Legacy conversation persistence (optional) |
| MSAL | >= 1.26.0 | Microsoft OAuth 2.0 |
| httpx | >= 0.26.0 | Async HTTP client |
| cryptography | >= 42.0.0 | AES-256-GCM encryption |
| slowapi | >= 0.1.9 | Rate limiting |
| tenacity | >= 8.2.0 | Retry with exponential backoff |
| python-docx | >= 1.1.0 | Word document processing |
| pypdf | >= 3.0.0 | PDF processing |

## Project Structure

```
src/
├── main.py                          # FastAPI app entry point, middleware, lifespan
├── api/
│   ├── routes.py                    # REST API endpoints
│   └── websocket.py                 # WebSocket MCP protocol handler
├── auth/
│   ├── oauth.py                     # Azure AD OAuth 2.0 with PKCE
│   └── dependencies.py              # FastAPI auth dependency injection
├── clients/
│   ├── devops.py                    # Azure DevOps REST API v7.0 client
│   ├── openai.py                    # Azure OpenAI client (GPT-4 generation)
│   └── cosmos.py                    # Cosmos DB client
├── config/
│   └── settings.py                  # Pydantic BaseSettings configuration
├── models/
│   ├── user.py                      # UserProfile, TokenInfo, SessionContext
│   ├── conversation.py              # ConversationDocument (Cosmos DB)
│   ├── devops.py                    # Organization, Project, WorkItem, Board
│   └── work_items.py                # DraftWorkItem, DraftCollection
├── mcp_tools/
│   ├── base.py                      # BaseTool interface and ToolContext
│   ├── core.py                      # User profile, organizations, projects
│   ├── jira.py                      # Jira project and issue operations
│   ├── generation.py                # AI generation (stories, tasks, epics, bugs)
│   ├── push.py                      # Push drafts to Azure DevOps
│   ├── conversations.py             # Conversation CRUD (Cosmos DB)
│   ├── board_wiki.py                # Board and wiki query tools
│   ├── edit_review.py               # Draft editing and validation
│   └── context.py                   # Context management tools
├── session/
│   ├── manager.py                   # Session create/read/update/destroy
│   ├── encryption.py                # AES-256-GCM encryption
│   ├── models.py                    # SessionState, SessionFingerprint
│   └── memory_store.py              # In-memory session store (dev fallback)
├── middleware/
│   └── rate_limit.py                # slowapi rate limiting configuration
├── services/
│   ├── conversation_repository.py   # Supabase/Cosmos conversation operations
│   └── file_processor.py            # Document/PDF text extraction
└── templates/                       # Reserved for future use
```

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

**Minimum configuration for local development:**

```env
AUTH_ENABLED=false
REDIS_ENABLED=false

AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4-turbo
AZURE_OPENAI_API_VERSION=2024-02-01

AZURE_DEVOPS_DEFAULT_ORG_URL=https://dev.azure.com/YourOrg
AZURE_DEVOPS_DEFAULT_PROJECT=YourProject
AZURE_DEVOPS_PAT=your-personal-access-token
```

### 3. Run

```bash
uvicorn src.main:app --reload --port 8000
```

- **Swagger UI:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/api/v1/health

### Supabase Auth, Postgres, Storage, and Jira Cloud

Set `AUTH_ENABLED=true` and `AUTH_PROVIDER=supabase`. Configure Google as an
OAuth provider in Supabase Auth; the Google client secret belongs in the
Supabase dashboard. The frontend callback must be included in Supabase Auth's
allowed redirect URLs, and Google Cloud must use Supabase's provider callback
(`https://<project-ref>.supabase.co/auth/v1/callback`).

Run `supabase/migrations/20261120123400_jira_chat_storage.sql` in the Supabase
SQL editor. It creates the RLS-protected conversations table, a backend-only
file metadata table, and a private bucket for extracted upload text. Enable
`CHAT_SAVE_ENABLED=true` and `SUPABASE_STORAGE_ENABLED=true`. Set `SUPABASE_URL`,
`SUPABASE_ANON_KEY`, and `SUPABASE_SERVICE_ROLE_KEY` on the backend host. The
service-role key must never be exposed to the browser or committed.

Redis remains the backend's short-lived session store for encrypted Jira OAuth
tokens, drafts, and request context. Use a hosted Redis service with TLS in
production; Supabase Auth does not replace this application data. Saved chats
are stored in Supabase Postgres and file text in private Supabase Storage.
Expired file objects are cleaned up by the backend, and logout removes that
session's files.

For Jira Cloud, create an Atlassian OAuth 2.0 (3LO) app and configure
`JIRA_CLIENT_ID`, `JIRA_CLIENT_SECRET`, `JIRA_REDIRECT_URI`, and
`JIRA_SITE_URL`. Add the exact callback URI to the Atlassian app and enable
`read:jira-work`, `write:jira-work`, `read:jira-user`, and `offline_access`.
Each user connects their own Atlassian account through the frontend; the
backend encrypts Jira access and refresh tokens in the user's app session.
Set `JIRA_SITE_URL` to your Atlassian site root (for example,
`https://your-site.atlassian.net`, without `/jira`). The Jira assistant flow
lists projects, searches issues within the selected project, and can create
supported reviewed drafts using issue types available in that project.
Never commit OAuth secrets.

### Render deployment

Deploy this directory as a Python web service with:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
- **Health-check path:** `/api/v1/health`

Set `CORS_ORIGINS` to the exact HTTPS Vercel origin. Configure the frontend and
backend with the same `INTERNAL_API_KEY`; keep all service-role, Jira, OpenAI,
Redis, encryption, and application secrets in Render's environment settings.
For production, also set `REDIS_SSL_ENABLED=true` and provide the Azure OpenAI
endpoint, API key, deployment name, and API version.

## API Endpoints

### Health and Configuration

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Root endpoint with API info |
| `GET` | `/api/v1/health` | Health check (Redis, Cosmos status) |
| `GET` | `/api/v1/config` | Public config (feature flags, chat settings) |

### Authentication

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/auth/start` | Initiate OAuth flow, returns authorization URL |
| `POST` | `/api/v1/auth/callback` | Exchange authorization code for session |
| `GET` | `/api/v1/session` | Get current session info |
| `DELETE` | `/api/v1/session` | Logout and destroy session |

### Tools

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/tools` | List all available MCP tools |
| `POST` | `/api/v1/tools/execute` | Execute a tool: `{ "tool_name": "...", "parameters": {...} }` |

### Drafts

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/drafts` | List all draft work items |
| `DELETE` | `/api/v1/drafts` | Clear all drafts |

### Files

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/files/upload` | Upload document (PDF, DOCX, TXT, MD — max 10 MB) |

### WebSocket

| Path | Description |
|---|---|
| `WS /ws/mcp` | Real-time MCP protocol |

**WebSocket message types:** `authenticate`, `execute_tool`, `list_tools`, `ping`
**WebSocket response types:** `authenticated`, `tool_result`, `tools_list`, `error`, `pong`

## MCP Tools

### Core Tools
- **`get_user_profile`** — Returns authenticated user info (display name, email, tenant)
- **`list_organizations`** — Lists all accessible Azure DevOps organizations
- **`list_projects`** — Lists projects in the selected organization
- **`select_project`** — Sets the active project for the session

### AI Generation Tools (GPT-4)
- **`generate_user_stories`** — Generates user stories with acceptance criteria, story points, and priority
- **`generate_tasks`** — Breaks work items into actionable development tasks
- **`generate_epic`** — Generates epic-level work items from high-level requirements
- **`generate_bug_report`** — Creates structured bug reports (repro steps, severity, priority)

### Review and Edit Tools
- **`list_drafts`** — Returns all draft work items in the session
- **`validate_drafts`** — Validates drafts against the Azure DevOps project schema
- **`edit_draft`** — Modifies a specific draft (title, description, criteria, points, priority)
- **`delete_draft`** — Removes a draft work item

### Push Tools
- **`push_drafts`** — Creates work items in Azure DevOps from validated drafts
- **`push_with_hierarchy`** — Pushes with parent-child relationships (Epic > Story > Task)

### Query Tools
- **`get_work_item`** — Retrieves an existing work item by ID
- **`search_work_items`** — Searches by type, state, area, iteration, or title keyword
- **`get_board_structure`** — Fetches board types, hierarchy, and process template info
- **`list_board_items`** — Lists items on a specific board

### Conversation Tools (requires Cosmos DB)
- **`save_conversation`** — Saves chat with messages, context, and draft snapshots
- **`list_conversations`** — Lists saved conversations for the user
- **`load_conversation`** — Restores a conversation (messages, context, drafts)
- **`delete_conversation`** — Removes a saved conversation

## Authentication

### Production: OAuth 2.0 with PKCE

When `AUTH_ENABLED=true`:

1. Frontend calls `POST /api/v1/auth/start` with `redirect_uri`
2. Backend generates PKCE challenge pair and stores state in Redis (10 min TTL)
3. Backend returns `authorization_url` for Azure AD
4. User authenticates at Azure AD, redirected back with authorization code
5. Frontend calls `POST /api/v1/auth/callback` with `code` and `state`
6. Backend exchanges code for tokens using PKCE verifier via MSAL
7. Session created with encrypted tokens, returned as httpOnly cookie

**Scopes requested:** `openid`, `profile`, `offline_access`, `499b84ac-1321-427f-aa17-267ca6975798/.default` (Azure DevOps)

**Token refresh:** Automatic when access token expires within 5-minute buffer. Uses refresh_token grant with MSAL silent acquisition fallback.

### Development: PAT Mode

When `AUTH_ENABLED=false`:

- Uses `AZURE_DEVOPS_PAT` for Basic auth against Azure DevOps
- No OAuth flow, no Azure AD configuration needed
- Returns mock session data

**Safety:** Application refuses to start if both `AUTH_ENABLED=true` and `AZURE_DEVOPS_PAT` are set.

## Session Management

### Storage

| Mode | Backend | Use Case |
|---|---|---|
| Redis | `redis>=5.0` | Production — distributed, supports multiple workers |
| In-Memory | Built-in | Development — single worker only |

Controlled by `REDIS_ENABLED` environment variable. Falls back to in-memory automatically.

### Session Lifecycle

1. **Created** at login with 256-bit cryptographically random ID
2. **Encrypted** with AES-256-GCM before storage
3. **Fingerprinted** with user-agent and IP for device binding
4. **Extended** on each request (sliding window expiry, default 24h)
5. **Auto-refreshes** OAuth access token when expiring (within 5 min buffer)
6. **Destroyed** on logout or timeout

### Session Data

```python
SessionContext:
    session_id: str
    user: UserProfile          # Azure AD user info
    token_info: TokenInfo      # Encrypted access/refresh/id tokens
    selected_org: str | None   # Current Azure DevOps organization
    selected_project: str | None  # Current project
    created_at: datetime
    last_accessed: datetime
    metadata: dict
```

## AI Generation (Azure OpenAI)

The backend uses GPT-4 with JSON mode to generate structured work items.

### Context-Aware Prompts

Before generation, the system fetches the project's board context:
- Available work item types (varies by process template)
- Hierarchy (Epic > Feature > User Story > Task)
- Process template detection (Agile, Scrum, CMMI, Basic)
- Field mapping (`story_points` vs `effort` based on template)
- Default requirement type (User Story vs Issue vs PBI)

This prevents the AI from generating unsupported work item types for the project.

### Generated Item Structure

```json
{
    "work_items": [
        {
            "work_item_type": "User Story",
            "title": "Implement OAuth login",
            "description": "As a user, I want to log in with my Azure AD account...",
            "acceptance_criteria": "Given user is on login page\nWhen user clicks 'Login'...",
            "story_points": 8,
            "priority": 2
        }
    ]
}
```

## Rate Limiting

| Endpoint Category | Limit | Purpose |
|---|---|---|
| Standard endpoints | 100 req/min | General protection |
| Tool execution | 30 req/min | AI generation is expensive |
| Auth endpoints | 10 req/min | Brute force prevention |
| Health checks | 200 req/min | Monitoring tolerance |

Rate limiting uses the session ID when available, falls back to client IP.

## Conversation Persistence (Cosmos DB)

When `CHAT_SAVE_ENABLED=true`, conversations are stored in Azure Cosmos DB:

- **Partition key:** `/user_id` (secure per-user isolation)
- **TTL:** Configurable 1-365 days (auto-deletion)
- **Snapshots:** Each save captures messages, context state (org/project), and draft work items
- **Restore:** Loading a conversation restores the full UI state

Supports both serverless and provisioned throughput Cosmos DB accounts.

## File Processing

The `FileProcessor` service extracts text from uploaded documents:

| Format | Library | Max Size |
|---|---|---|
| `.txt`, `.md` | Built-in | 10 MB |
| `.docx` | python-docx | 10 MB |
| `.pdf` | pypdf | 10 MB |

Extracted text is stored in the session and used as additional context for AI generation.

## Error Handling

### Custom Exception Hierarchy

**Auth:** `AuthenticationError`, `AuthorizationError`, `TokenError`, `UserInfoError`

**Azure DevOps:** `AzureDevOpsError`, `RateLimitError`, `AuthenticationError`, `NotFoundError`, `ValidationError`

**Session:** `SessionError`, `SessionNotFoundError`, `SessionExpiredError`, `SessionInvalidError`, `SessionFingerprintMismatchError`

All exceptions include context information and map to appropriate HTTP status codes.

## Production Deployment

### Azure App Service

The included `startup.sh` handles worker scaling:

```bash
# With Redis (distributed sessions): 4 workers
# Without Redis (in-memory sessions): 1 worker
```

### Production Checklist

- [ ] `AUTH_ENABLED=true` with Azure AD credentials
- [ ] `REDIS_ENABLED=true` with Redis instance (TLS recommended)
- [ ] `AZURE_DEVOPS_PAT` removed or commented out
- [ ] `ENCRYPTION_KEY` set (generate: `python -c "import secrets, base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"`)
- [ ] `APP_SECRET_KEY` set to 32+ character random string
- [ ] `CORS_ORIGINS` set to frontend URL(s)
- [ ] `APP_ENV=production`, `APP_DEBUG=false`
- [ ] `LOG_LEVEL=WARNING` or `ERROR`

## Documentation

- [`docs/API_RESPONSE_REFERENCE.md`](docs/API_RESPONSE_REFERENCE.md) — Complete response examples for all API endpoints
- [`docs/AZURE_AD_FLOW_SETUP.md`](docs/AZURE_AD_FLOW_SETUP.md) — Azure AD setup, verification, and troubleshooting
- [`docs/FRONTEND_INTEGRATION_GUIDE.md`](docs/FRONTEND_INTEGRATION_GUIDE.md) — Integration reference for the frontend
