"""
Pydantic schema and persistence for confirmed hotel bookings.

BookingRecord enforces all four field-level constraints before anything is
written to disk. save_booking appends valid records to a local JSON file so
bookings survive across Streamlit reruns.
"""

import json
import os
from datetime import date
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

# Bookings are appended to this file as a JSON array.
BOOKINGS_FILE = os.path.join(os.path.dirname(__file__), "bookings.json")


class BookingRecord(BaseModel):
    """A validated, confirmed hotel booking.

    Constraints enforced at instantiation:
      - guest_name  : must contain at least a first and last name.
      - room_type   : must be one of the four recognised room categories.
      - check_in_date / check_out_date : ISO format (YYYY-MM-DD), upcoming dates,
                                         check-out strictly after check-in.
    """

    guest_name: str
    room_type: Literal["Single", "Double", "Suite", "Penthouse"]
    check_in_date: str
    check_out_date: str

    @field_validator("guest_name")
    @classmethod
    def must_have_first_and_last(cls, v: str) -> str:
        """Reject names that don't contain at least two whitespace-separated parts."""
        if len(v.strip().split()) < 2:
            raise ValueError("guest_name must contain both a first and last name")
        return v.strip()

    @field_validator("check_in_date", "check_out_date")
    @classmethod
    def must_be_valid_iso(cls, v: str) -> str:
        """Ensure the value can be parsed as a YYYY-MM-DD date."""
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError(f"'{v}' is not a valid ISO date (YYYY-MM-DD)")
        return v

    @model_validator(mode="after")
    def dates_must_be_valid_range(self) -> "BookingRecord":
        """Cross-field check: check-in must be in the future; check-out after check-in."""
        ci = date.fromisoformat(self.check_in_date)
        co = date.fromisoformat(self.check_out_date)
        if ci < date.today():
            raise ValueError("check_in_date must be an upcoming date")
        if co <= ci:
            raise ValueError("check_out_date must be chronologically after check_in_date")
        return self


def save_booking(record: BookingRecord) -> None:
    """Append a validated BookingRecord to the local JSON bookings file.

    If the file doesn't exist yet it is created. Existing entries are
    preserved — this function never overwrites previous bookings.
    """
    existing: list = []
    if os.path.exists(BOOKINGS_FILE):
        with open(BOOKINGS_FILE, "r") as f:
            existing = json.load(f)
    existing.append(record.model_dump())
    with open(BOOKINGS_FILE, "w") as f:
        json.dump(existing, f, indent=2)
