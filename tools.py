"""
Tool schemas, implementations, and dispatcher for the hotel booking assistant.

Each tool is defined in two places:
  - TOOLS: the JSON schema list passed to the Groq API so the model knows what it can call.
  - A Python function that executes when the model requests that tool.

TOOL_DISPATCH maps tool names to their implementations so run_tool() can resolve
any tool call the model returns without a chain of if/elif checks.
"""

import json
import random

# Room numbers available per room type.
ROOM_INVENTORY: dict[str, list[int]] = {
    "Single": [101, 102, 103],
    "Double": [201, 202],
    "Suite": [301],
    "Penthouse": [401],
}

# Fixed anchor prices (USD per night) before noise is applied.
BASE_PRICES: dict[str, float] = {
    "Single": 89.0,
    "Double": 149.0,
    "Suite": 299.0,
    "Penthouse": 599.0,
}

# JSON schemas registered with Groq — the model uses these to decide which tool to call
# and what arguments to pass.
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "check_prices",
            "description": (
                "Get the current nightly price for one or all room types. "
                "Prices fluctuate dynamically — always call this tool for up-to-date rates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "room_type": {
                        "type": "string",
                        "enum": ["Single", "Double", "Suite", "Penthouse", "all"],
                        "description": "The room type to price, or 'all' to return rates for every type.",
                    },
                },
                "required": ["room_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_room_availability",
            "description": (
                "Check whether a room type is available at the hotel for a given "
                "check-in and check-out date range. Returns available room numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "room_type": {
                        "type": "string",
                        "enum": ["Single", "Double", "Suite", "Penthouse"],
                        "description": "The category of room to check.",
                    },
                    "check_in": {
                        "type": "string",
                        "description": "Check-in date in YYYY-MM-DD format.",
                    },
                    "check_out": {
                        "type": "string",
                        "description": "Check-out date in YYYY-MM-DD format.",
                    },
                },
                "required": ["room_type", "check_in", "check_out"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _noisy_price(base: float) -> float:
    """Apply a random multiplier in [0.5, 3.0] to simulate dynamic pricing."""
    return round(base * random.uniform(0.5, 3.0), 2)


def check_prices(room_type: str) -> dict:
    """Return the current dynamic nightly price for one room type or all types."""
    if room_type == "all":
        return {
            "prices_per_night_usd": {rt: _noisy_price(base) for rt, base in BASE_PRICES.items()}
        }
    base = BASE_PRICES.get(room_type)
    if base is None:
        return {"error": f"Unknown room type: {room_type}"}
    return {"room_type": room_type, "price_per_night_usd": _noisy_price(base)}


def check_room_availability(room_type: str, check_in: str, check_out: str) -> dict:
    """Randomly simulate which rooms of a given type are available for the requested dates."""
    rooms = ROOM_INVENTORY.get(room_type, [])
    # Each room independently has a 50/50 chance of being available.
    available = [room for room in rooms if random.choice([True, False])]
    return {
        "room_type": room_type,
        "check_in": check_in,
        "check_out": check_out,
        "available_rooms": available,
        "available": len(available) > 0,
    }


# Maps tool names (as the model calls them) to their Python implementations.
TOOL_DISPATCH: dict[str, callable] = {
    "check_prices": lambda args: check_prices(**args),
    "check_room_availability": lambda args: check_room_availability(**args),
}


def run_tool(name: str, args: dict) -> str:
    """Look up and execute a tool by name, returning its result as a JSON string."""
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    return json.dumps(fn(args))
