# Hotel Booking Assistant

A Streamlit chat application powered by the Groq API (LLaMA 4 Scout) that simulates a hotel booking assistant. The model uses registered tools to check room availability and pricing, collects booking details through natural conversation, validates them with Pydantic, and persists confirmed bookings to disk. Guest preferences are remembered across sessions via SQLite.

---

## Features

- **Conversational booking flow** — the assistant gathers room type, dates, and guest name through dialogue before confirming
- **Tool calling** — Groq's function-calling API drives two tools the model can invoke:
  - `check_room_availability` — randomly simulates which rooms are free for the requested dates
  - `check_prices` — returns dynamically noised nightly rates (base price × random multiplier in `[0.5, 3.0]`)
- **Booking validation** — confirmed bookings are extracted via a silent secondary LLM call, validated with Pydantic, and appended to `bookings.json`
  - Guest name must include first and last name
  - Room type must be one of: Single, Double, Suite, Penthouse
  - Dates must be ISO format, upcoming, and check-out after check-in
  - Extraction retries up to 5 times, feeding Pydantic errors back to the model for self-correction
- **Long-term memory** — a sidebar toggle enables/disables preference persistence across sessions
  - Preferences (bed type, accessibility needs, view, floor, etc.) are extracted at session end and stored in SQLite
  - On session start, stored preferences are injected into the system prompt so the model tailors its suggestions
  - Toggling memory mid-session immediately rebuilds the system message

---

## Project Structure

```
.
├── chat_app.py       # Streamlit UI — session state, sidebar, chat loop
├── agent.py          # Agentic loop — get_response(), booking extractor, preference extractor
├── tools.py          # Tool schemas (registered with Groq), implementations, dispatcher
├── validation.py     # Pydantic BookingRecord schema and save_booking()
├── messages.py       # Typed message constructors and role-aware accessors
├── preferences.py    # SQLite-backed guest preference persistence
├── bookings.json     # Persisted confirmed bookings (created on first booking)
├── preferences.db    # SQLite database for long-term guest preferences (created on first save)
└── .env              # API keys and model config (not committed)
```

---

## Setup

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com)

### Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
```

### Run

```bash
streamlit run chat_app.py
```

---

## How It Works

### Conversation flow

1. Guest asks about rooms or prices → model calls `check_room_availability` or `check_prices`
2. Guest decides to book → model asks for first and last name, then confirms all details
3. On each assistant reply, a silent extraction call checks whether a booking has been confirmed
4. If confirmed, the booking is validated with Pydantic and written to `bookings.json`
5. On **End Session**, a silent extraction call scans for preferences and saves them to `preferences.db`

### Agentic tool-call loop

```
User message
    └─▶ Groq API (with TOOLS)
            ├─ finish_reason == "tool_calls"
            │       └─▶ execute tool → append tool result → loop
            └─ finish_reason == "stop"
                    └─▶ return text reply to UI
```

If Groq rejects a tool call (e.g. missing or invalid parameter), the error is caught, injected as a user message, and the model is asked to request the missing information from the guest before retrying.

### Message roles

All messages in the payload are constructed through typed helpers in `messages.py` to ensure role separation:

| Role | When used |
|---|---|
| `system` | Session initialisation only (index 0) |
| `user` | Guest input and error-recovery injections |
| `assistant` | Model replies and tool-call stubs |
| `tool` | Tool execution results |

---

## Data Files

| File | Format | Contents |
|---|---|---|
| `bookings.json` | JSON array | One object per confirmed booking: `guest_name`, `room_type`, `check_in_date`, `check_out_date` |
| `preferences.db` | SQLite | Single-row table with a JSON blob of accumulated guest preferences and a last-updated timestamp |
