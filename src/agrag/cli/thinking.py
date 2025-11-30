"""Thinking budget configuration for Gemini models."""

from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agrag.config import settings

THINKING_PRESETS = {
    "low": 256,
    "medium": 1024,
    "high": 4096,
    "dynamic": -1,
}


def format_thinking_budget(value: Optional[int]) -> str:
    """Return a friendly label for the current thinking budget.

    Args:
        value: The thinking budget value (tokens or preset).

    Returns:
        Human-readable description of the budget.
    """
    if value is None:
        return "Default (model decides)"
    preset = next((name for name, budget in THINKING_PRESETS.items() if budget == value), None)
    if preset:
        label = preset.capitalize()
    else:
        label = str(value)
    if value == -1:
        label += " (dynamic)"
    elif value == 0:
        label += " (disabled)"
    return label


def print_thinking_help(console: Console) -> None:
    """Display thinking configuration help.

    Args:
        console: Rich console for output.
    """
    preset_details = "\n".join(
        f"- `{name}` = {budget if budget != -1 else '-1 (dynamic)'} tokens"
        for name, budget in THINKING_PRESETS.items()
    )
    help_text = f"""
**Thinking Settings**
- `/thinking` - Show current setting
- `/thinking <preset>` - Apply preset (low, medium, high, dynamic) or an integer token budget

{preset_details}

Use `dynamic` (-1) to let the model decide, or provide a numeric token budget (e.g., `/thinking 512`).
"""
    console.print(
        Panel(Markdown(help_text), title="Thinking Configuration", border_style="magenta")
    )


def handle_thinking_command(
    console: Console,
    raw_command: str,
    current_budget: Optional[int],
) -> Optional[int]:
    """Handle the /thinking command.

    Args:
        console: Rich console for output.
        raw_command: The full command string.
        current_budget: Current thinking budget value.

    Returns:
        New thinking budget if changed, None otherwise.
    """
    parts = raw_command.split()

    if len(parts) == 1:
        console.print(
            f"[cyan]Current thinking budget:[/cyan] {format_thinking_budget(current_budget)}"
        )
        print_thinking_help(console)
        return None

    target = parts[1].lower()
    if target in THINKING_PRESETS:
        value = THINKING_PRESETS[target]
    else:
        try:
            value = int(target)
        except ValueError:
            console.print(
                "[red]Invalid thinking value. Use a preset (low/medium/high/dynamic) or integer tokens.[/red]"
            )
            return None

    settings.google_thinking_budget = value
    console.print(f"[green]✓ Thinking budget set to {format_thinking_budget(value)}[/green]")
    return value
