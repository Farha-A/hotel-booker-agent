"""
Agentic loop orchestration for the hotel booking assistant.

Three responsibilities live here:

  get_response()
      The main chat loop. Sends the message history to Groq, handles tool
      call execution, and continues looping until the model returns a plain
      text reply. Tool call errors are caught and fed back to the model so
      it can ask the guest for missing information rather than crashing.

  try_extract_and_save_booking()
      A silent secondary LLM call made after every assistant reply. Asks the
      model to extract a confirmed booking from the conversation as JSON,
      validates it with Pydantic, and persists it to disk. Retries up to
      max_retries times, feeding validation errors back to the model each
      time so it can self-correct.

  extract_and_save_preferences()
      A silent secondary LLM call triggered at session end. Scans the
      conversation for any guest preferences (bed type, accessibility needs,
      etc.) and persists them to SQLite for use in future sessions.
"""

import json

import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

from tools import TOOLS, run_tool
from validation import BookingRecord, save_booking
from preferences import save_preferences
from messages import (
    user_message,
    assistant_message,
    tool_message,
    system_message,
    conversation_messages,
)

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")


def get_response(messages: list) -> str:
    """Run the agentic tool-call loop and return the model's final text reply.

    On each iteration the full message history is sent to Groq. If the model
    requests one or more tool calls, each tool is executed and its result
    appended as a 'tool' message before the next iteration. The loop exits
    when the model produces a non-tool response.

    If Groq rejects a tool call (e.g. a parameter fails schema validation),
    the error is injected as a 'user' message so the model can ask the guest
    to clarify before retrying.
    """
    while True:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            # Surface the API error to the model so it can recover gracefully
            # by asking the guest for the missing or corrected parameter.
            messages.append(
                user_message(
                    f"The tool call failed because of invalid or missing parameters: {e}. "
                    "Please ask the guest for the missing or correct information, then try again."
                )
            )
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            raw_msg = choice.message

            # Serialise the tool calls into the typed dict format the API expects
            # on subsequent turns.
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_msg.tool_calls
            ]
            messages.append(assistant_message(raw_msg.content or "", tool_calls))

            # Execute each requested tool and append its result before the next loop.
            for tc in raw_msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = run_tool(tc.function.name, args)
                messages.append(tool_message(tc.id, result))
        else:
            return choice.message.content


def try_extract_and_save_booking(messages: list, max_retries: int = 5) -> None:
    """Silently extract a confirmed booking from the conversation and persist it.

    Builds a separate message thread (not visible to the guest) and asks the
    model to output the booking as JSON. If Pydantic validation fails, the
    error is fed back to the model and another attempt is made, up to
    max_retries times. Sets st.session_state.booking_saved on success so
    subsequent messages skip this call.
    """
    extraction_messages = [
        system_message(
            "You extract confirmed hotel booking details from a conversation. "
            "If the assistant has explicitly confirmed a booking with the guest, "
            "output a JSON object with exactly these keys: "
            "guest_name, room_type, check_in_date, check_out_date. "
            "room_type must be one of: Single, Double, Suite, Penthouse. "
            "Dates must be in YYYY-MM-DD format. "
            "guest_name must include both first and last name. "
            "check_in_date must be today or a future date. "
            "check_out_date must be after check_in_date. "
            'If no booking has been confirmed yet, output: {"booking": null}'
        ),
        *conversation_messages(messages),
        user_message(
            "Extract the confirmed booking as JSON. "
            'If none confirmed, return {"booking": null}.'
        ),
    ]

    print(f"\n[BOOKING EXTRACTOR] Starting extraction (max {max_retries} attempts)")

    for attempt in range(max_retries):
        print(f"[BOOKING EXTRACTOR] Attempt {attempt + 1}/{max_retries} — calling LLM...")
        response = client.chat.completions.create(
            model=MODEL,
            messages=extraction_messages,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        print(f"[BOOKING EXTRACTOR] Raw response: {raw}")

        # Append the model's attempt so it has context if a retry is needed.
        extraction_messages.append(assistant_message(raw))

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"[BOOKING EXTRACTOR] JSON parse error: {e}")
            extraction_messages.append(
                user_message(
                    f"Your response was not valid JSON: {e}. "
                    "Please output a valid JSON object."
                )
            )
            continue

        # Model signalled no confirmed booking in the conversation yet.
        if data.get("booking") is None and "guest_name" not in data:
            print("[BOOKING EXTRACTOR] No confirmed booking found in conversation — skipping save.")
            return

        # Support both top-level keys and a nested "booking" key.
        booking_data = data if "guest_name" in data else data.get("booking", {})
        if not booking_data:
            print("[BOOKING EXTRACTOR] Booking key present but empty — skipping save.")
            return

        try:
            record = BookingRecord(**booking_data)
            save_booking(record)
            st.session_state.booking_saved = True
            print(f"[BOOKING EXTRACTOR] Booking saved successfully: {record.model_dump()}")
            return
        except Exception as e:
            # Feed the exact Pydantic error back so the model knows what to fix.
            print(f"[BOOKING EXTRACTOR] Pydantic validation error: {e}")
            extraction_messages.append(
                user_message(
                    f"The booking data failed validation with this error: {e}. "
                    "Please correct the JSON and output the fixed booking object."
                )
            )

    print(f"[BOOKING EXTRACTOR] All {max_retries} attempts exhausted — booking not saved.")


def extract_and_save_preferences(messages: list) -> None:
    """Silently extract guest preferences from the conversation and persist to SQLite.

    Called once at session end. Makes a single LLM call asking for a flat
    JSON object of preference key/value pairs. No retry loop — preferences
    are best-effort; a failed extraction is not critical.
    """
    extraction_messages = [
        system_message(
            "You extract guest preferences from a hotel booking conversation. "
            "Look for any preferences the guest mentioned, such as: "
            "bed type (single, double, queen, king), accessibility needs (wheelchair access, elevator), "
            "floor preference (high floor, low floor, ground floor), "
            "view preference (sea view, garden view, city view), "
            "dietary requirements, smoking/non-smoking, pet-friendly, quiet room, etc. "
            "Output a flat JSON object where each key is a preference name in snake_case "
            "and the value is what the guest specified (string). "
            "Only include preferences explicitly mentioned by the guest. "
            'If no preferences were mentioned, output: {"preferences": null}'
        ),
        *conversation_messages(messages),
        user_message(
            "Extract guest preferences as a flat JSON object. "
            'If none, return {"preferences": null}.'
        ),
    ]

    print("\n[PREFERENCES EXTRACTOR] Starting extraction...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=extraction_messages,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    print(f"[PREFERENCES EXTRACTOR] Raw response: {raw}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[PREFERENCES EXTRACTOR] JSON parse error: {e} — skipping save.")
        return

    # Model returned the null sentinel — no preferences to save.
    if data.get("preferences") is None and not any(k != "preferences" for k in data):
        print("[PREFERENCES EXTRACTOR] No preferences found — skipping save.")
        return

    # Handle both top-level keys and a nested "preferences" key.
    prefs = {k: v for k, v in data.items() if k != "preferences"} or data.get("preferences") or {}
    if not prefs:
        print("[PREFERENCES EXTRACTOR] Empty preferences — skipping save.")
        return

    save_preferences(prefs)
