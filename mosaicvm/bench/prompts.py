"""Golden prompts shared by quality sweeps. Do not fork this list."""

GOLDEN_PROMPTS = [
    "def fibonacci(n):",
    "The history of the silk road begins",
    "Prove that sqrt(2) is irrational:",
    "Call the weather API for Tokyo tomorrow with units=celsius.",
    "In a long-context retrieval setting, the key idea is",
]

# Phase 2.10B route-capture suite. GOLDEN_PROMPTS stays untouched; this list
# adds the categories the roadmap requires (JSON, long generation) as tags.
ROUTE_CAPTURE_PROMPTS = [
    ("coding", "def fibonacci(n):"),
    ("agent_tool_use", "Call the weather API for Tokyo tomorrow with units=celsius."),
    ("math_reasoning", "Prove that sqrt(2) is irrational:"),
    ("general_conversation", "The history of the silk road begins"),
    ("long_context_retrieval", "In a long-context retrieval setting, the key idea is"),
    ("structured_json", 'Return a JSON object with fields "name", "age", "email" for a user profile schema, validating that'),
    ("long_generation", "Write a detailed technical explanation of how residual networks improved image classification, covering"),
]
