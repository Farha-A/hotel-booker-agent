"""
Typed message constructors and role-aware accessors for the Groq API payload.

Three conversational roles are strictly separated:
  - 'user'      — text sent by the human guest.
  - 'assistant' — text (and optional tool_calls) produced by the model.
  - 'tool'      — the result of executing a tool the model requested.

'system' messages are constructed via system_message() and always occupy
index 0 of the message list. They are intentionally excluded from the
conversation accessors used by secondary (extraction) LLM calls.

Using the named constructors instead of raw dicts ensures every message
in the payload has the correct shape and role value.
"""

from typing import TypedDict, NotRequired


# ---------------------------------------------------------------------------
# TypedDicts — document the exact shape the Groq API expects per role
# ---------------------------------------------------------------------------

class UserMessage(TypedDict):
    role: str        # always "user"
    content: str


class FunctionCall(TypedDict):
    name: str
    arguments: str   # JSON-encoded string of the call's arguments


class ToolCall(TypedDict):
    id: str
    type: str        # always "function"
    function: FunctionCall


class AssistantMessage(TypedDict):
    role: str        # always "assistant"
    content: str
    tool_calls: NotRequired[list[ToolCall]]  # present only when the model is invoking tools


class ToolMessage(TypedDict):
    role: str        # always "tool"
    tool_call_id: str  # must match the id from the corresponding ToolCall
    content: str


class SystemMessage(TypedDict):
    role: str        # always "system"
    content: str


# ---------------------------------------------------------------------------
# Constructors — the only way a message should enter the payload list
# ---------------------------------------------------------------------------

def user_message(content: str) -> UserMessage:
    """Create a user-role message."""
    return {"role": "user", "content": content}


def assistant_message(
    content: str,
    tool_calls: list[ToolCall] | None = None,
) -> AssistantMessage:
    """Create an assistant-role message, optionally carrying tool call requests."""
    msg: AssistantMessage = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def tool_message(tool_call_id: str, content: str) -> ToolMessage:
    """Create a tool-role message carrying the result of a tool execution."""
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def system_message(content: str) -> SystemMessage:
    """Create a system-role message (used exclusively as messages[0])."""
    return {"role": "system", "content": content}


# ---------------------------------------------------------------------------
# Accessors — role-filtered views of the message list
# ---------------------------------------------------------------------------

def display_messages(messages: list) -> list[UserMessage | AssistantMessage]:
    """Return only user and assistant messages that have visible text content.

    Used by the Streamlit UI to render the chat history — tool results and
    assistant turns that contain only a tool_calls stub are excluded.
    """
    return [
        m for m in messages
        if m["role"] in ("user", "assistant") and m.get("content")
    ]


def conversation_messages(messages: list) -> list[UserMessage | AssistantMessage]:
    """Return user and assistant text turns only, stripping tool results.

    Used when building the context for secondary (extraction) LLM calls that
    should reason over the human conversation, not raw tool payloads.
    """
    return [
        m for m in messages
        if m["role"] in ("user", "assistant") and m.get("content")
    ]
