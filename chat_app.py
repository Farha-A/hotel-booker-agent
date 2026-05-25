"""
Streamlit UI for the hotel booking assistant.

Responsibilities:
  - Render the sidebar (long-term memory toggle, saved preferences display,
    End Session button).
  - Initialise and maintain st.session_state.messages, rebuilding the system
    message whenever the memory toggle changes.
  - Drive the chat loop: append user input, call get_response(), display the
    reply, then trigger silent background extractions as needed.
"""

import json
import os
from datetime import date

import streamlit as st

from agent import get_response, try_extract_and_save_booking, extract_and_save_preferences
from preferences import load_preferences
from messages import user_message, assistant_message, system_message, display_messages

SESSION_STATE_FILE = os.path.join(os.path.dirname(__file__), "session_state.json")


def save_session_state() -> None:
    """Dump the current session state to a JSON file for debugging and auditing.
    The file is overwritten on each call — it reflects the latest snapshot only."""
    snapshot = {k: v for k, v in st.session_state.items()}
    with open(SESSION_STATE_FILE, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

st.set_page_config(page_title="Hotel Booking", page_icon="🏨", layout="centered")
st.title("🏨 Hotel Booking Assistant")


# ---------------------------------------------------------------------------
# Sidebar — settings, saved preferences, session controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    memory_enabled = st.toggle("Long-term memory", value=True)
    st.caption("When enabled, guest preferences are saved between sessions.")

    st.divider()

    if st.button("End Session", use_container_width=True):
        if memory_enabled:
            with st.spinner("Saving preferences..."):
                extract_and_save_preferences(st.session_state.get("messages", []))
            st.success("Preferences saved.")
        # Save a final snapshot before wiping the conversation.
        save_session_state()
        # Clear the conversation so the next interaction starts fresh.
        st.session_state.messages = []
        st.session_state.booking_saved = False
        st.rerun()


# ---------------------------------------------------------------------------
# System message builder
# ---------------------------------------------------------------------------

def _build_system_message(with_memory: bool) -> dict:
    """Build the system message, optionally injecting saved guest preferences.

    Preferences are only loaded and included when with_memory is True.
    If no preferences are stored yet, the context string is left empty.
    """
    preference_context = ""
    if with_memory:
        saved_prefs = load_preferences()
        if saved_prefs:
            prefs_formatted = ", ".join(
                f"{k.replace('_', ' ')}: {v}" for k, v in saved_prefs.items()
            )
            preference_context = (
                f"The guest has the following saved preferences from previous sessions: {prefs_formatted}. "
                "Take these into account when making suggestions, but always confirm with the guest. "
            )

    return system_message(
        f"Today's date is {date.today().isoformat()}. "
        "You are a helpful hotel booking assistant. "
        + preference_context +
        "Use check_room_availability to show available rooms — no name needed for this step. "
        "When the guest provides any date, compare it against today's date. "
        "If the check-in or check-out date is in the past, do not proceed — "
        "inform the guest the date has already passed and ask them to provide a new upcoming date. "
        "When the guest wants to confirm a booking, ask for their first and last name, "
        "then confirm all details (guest name, room type, check-in date, check-out date) "
        "with the guest before finalising. "
        "Room types are: Single, Double, Suite, Penthouse."
    )


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "booking_saved" not in st.session_state:
    st.session_state.booking_saved = False

# Detect whether the memory toggle changed since the last rerun so the system
# message can be rebuilt with or without preferences immediately.
memory_changed = st.session_state.get("memory_enabled") != memory_enabled
st.session_state.memory_enabled = memory_enabled

if "messages" not in st.session_state or not st.session_state.messages:
    # First run or after End Session — initialise with just the system message.
    st.session_state.messages = [_build_system_message(memory_enabled)]
elif memory_changed:
    # Toggle flipped mid-session — replace only the system message (index 0)
    # while leaving the rest of the conversation history intact.
    st.session_state.messages[0] = _build_system_message(memory_enabled)


# ---------------------------------------------------------------------------
# Sidebar — remembered preferences display (rendered after session init so
# load_preferences() reflects the current toggle state)
# ---------------------------------------------------------------------------

with st.sidebar:
    if memory_enabled:
        prefs = load_preferences()
        if prefs:
            st.subheader("Remembered preferences")
            for k, v in prefs.items():
                st.markdown(f"- **{k.replace('_', ' ').title()}**: {v}")
        else:
            st.caption("No preferences saved yet.")


# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------

# Render existing conversation (user and assistant turns only — tool messages
# are internal and never shown to the guest).
for msg in display_messages(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask for a booking!"):
    st.session_state.messages.append(user_message(prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = get_response(st.session_state.messages)
        st.markdown(reply)

    st.session_state.messages.append(assistant_message(reply))

    save_session_state()

    # Only attempt extraction while no booking has been saved this session.
    if not st.session_state.booking_saved:
        try_extract_and_save_booking(st.session_state.messages)
