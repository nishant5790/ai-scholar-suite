# REST API Reference

Base URL: `http://localhost:8000/api/v1`

All request and response bodies are JSON. Errors return an appropriate HTTP status code with a JSON body.

---

## Sessions

### Create Session

```
POST /sessions
```

Creates a new paper-writing session with an empty `PaperState`.

**Response** `201 Created`

```json
{
  "session_id": "a1b2c3d4-..."
}
```

---

### Chat (Agent)

```
POST /sessions/{session_id}/chat
```

Send a free-form message to the LangChain agent. The agent decides which tool to invoke based on intent.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | str | Yes | User message |

**Response** `200 OK`

```json
{
  "response": "Agent reply text..."
}
```

> **Note:** Full agent integration requires an LLM to be configured for the session. Without one, the endpoint returns a placeholder acknowledgement.

---

## Outline

### Generate Outline

```
POST /sessions/{session_id}/outline
```

Generates a structured paper outline and stores it in the session's `PaperState`.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `topic` | str | Yes | Research topic |
| `instructions` | str | No | Additional constraints |

**Response** `200 OK`

Returns the full `PaperOutline` JSON with `topic` and `sections` array.

**Errors**

| Code | Condition |
|------|-----------|
| 400 | Empty topic |
| 404 | Session not found |
| 500 | No agent configured / internal error |

---

## Sections

### Generate Section

```
POST /sessions/{session_id}/sections/{section_name}
```

Generates content for one section and stores it in `PaperState.sections`.

**Path Parameters**

| Parameter | Description |
|-----------|-------------|
| `section_name` | One of: `abstract`, `introduction`, `literature_review`, `methodology`, `results`, `discussion`, `conclusion` |

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `feedback` | str | No | Revision instructions |

**Response** `200 OK`

Returns the `SectionContent` JSON with `section_type`, `title`, `content`, and `citations`.

**Errors**

| Code | Condition |
|------|-----------|
| 400 | Invalid section name (returns list of valid types) |
| 404 | Session not found |
| 500 | No agent configured / internal error |

---

## References

### Ingest Reference Folder

```
POST /sessions/{session_id}/references/ingest
```

Reads files from a local directory and indexes them in ChromaDB.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `folder_path` | str | Yes | Absolute or relative path to a directory |

**Response** `200 OK`

```json
{
  "files_processed": 3,
  "files_skipped": 1,
  "skipped_files": ["image.png"],
  "total_chunks": 42
}
```

**Errors**

| Code | Condition |
|------|-----------|
| 400 | Invalid or inaccessible path |
| 404 | Session not found |

---

### Get Bibliography

```
GET /sessions/{session_id}/bibliography?style=apa
```

Returns the formatted bibliography for all citations in the session.

**Query Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `style` | str | No | `apa` | One of: `apa`, `ieee`, `mla` |

**Response** `200 OK`

```json
{
  "bibliography": "Formatted reference list...",
  "style": "apa"
}
```

**Errors**

| Code | Condition |
|------|-----------|
| 400 | Invalid citation style |
| 404 | Session not found |

---

## Export

### Export to PDF

```
POST /sessions/{session_id}/export/pdf
```

Compiles all sections into a formatted PDF file.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `output_path` | str | Yes | File path for the generated PDF |

**Response** `200 OK`

```json
{
  "output_path": "./output/my_paper.pdf"
}
```

**Errors**

| Code | Condition |
|------|-----------|
| 400 | Missing required sections |
| 404 | Session not found |

---

## State Persistence

### Save State

```
POST /sessions/{session_id}/save
```

Serialises the session's `PaperState` to a JSON file.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file_path` | str | Yes | Output file path |

**Response** `200 OK`

```json
{
  "message": "State saved to ./output/state.json"
}
```

---

### Load State

```
POST /sessions/{session_id}/load
```

Restores `PaperState` from a previously saved JSON file.

**Request Body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file_path` | str | Yes | Path to a saved state file |

**Response** `200 OK`

```json
{
  "message": "State loaded from ./output/state.json"
}
```

**Errors**

| Code | Condition |
|------|-----------|
| 400 | File not found or invalid JSON |
| 404 | Session not found |

---

## Example: Full Workflow with curl

```bash
# 1. Create session
SESSION=$(curl -s -X POST http://localhost:8000/api/v1/sessions | jq -r .session_id)

# 2. Generate outline
curl -s -X POST "http://localhost:8000/api/v1/sessions/$SESSION/outline" \
  -H "Content-Type: application/json" \
  -d '{"topic": "AI fine tuning", "instructions": "Cover LoRA and QLoRA"}' | jq .

# 3. Generate each section
for SECTION in abstract introduction literature_review methodology results discussion conclusion; do
  curl -s -X POST "http://localhost:8000/api/v1/sessions/$SESSION/sections/$SECTION" \
    -H "Content-Type: application/json" \
    -d '{"feedback": ""}' | jq .title
done

# 4. Export PDF
curl -s -X POST "http://localhost:8000/api/v1/sessions/$SESSION/export/pdf" \
  -H "Content-Type: application/json" \
  -d '{"output_path": "./output/paper.pdf"}' | jq .

# 5. Save state
curl -s -X POST "http://localhost:8000/api/v1/sessions/$SESSION/save" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./output/state.json"}' | jq .
```
