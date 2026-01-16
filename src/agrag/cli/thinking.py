"""Thinking configuration for Gemini models."""

from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agrag.config import settings

THINKING_LEVELS = {"low", "medium", "high", "minimal"}

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


def format_thinking_setting(
    thinking_level: Optional[str], thinking_budget: Optional[int]
) -> str:
    """Return a friendly label for the current thinking setting."""
    if thinking_level:
        return f"Level: {thinking_level}"
    return f"Budget: {format_thinking_budget(thinking_budget)}"


def print_thinking_help(console: Console) -> None:
    """Display thinking configuration help.

    Args:
        console: Rich console for output.
    """
    preset_details = "\n".join(
        f"- `{name}` = {budget if budget != -1 else '-1 (dynamic)'} tokens"
        for name, budget in THINKING_PRESETS.items()
    )
    level_details = "\n".join(f"- `{level}`" for level in sorted(THINKING_LEVELS))
    help_text = f"""
**Thinking Settings**
- `/thinking` - Show current setting
- `/thinking <level>` - Set Gemini 3 thinking level ({', '.join(sorted(THINKING_LEVELS))})
- `/thinking <preset>` - Apply legacy budget preset (low, medium, high, dynamic)
- `/thinking <int>` - Set legacy thinking budget tokens

**Levels**
{level_details}

**Budget Presets**
{preset_details}

Use `dynamic` (-1) to let the model decide, or provide a numeric token budget (e.g., `/thinking 512`).
"""
    console.print(
        Panel(Markdown(help_text), title="Thinking Configuration", border_style="magenta")
    )


def handle_thinking_command(
    console: Console,
    raw_command: str,
    current_level: Optional[str],
    current_budget: Optional[int],
) -> Optional[tuple[Optional[str], Optional[int]]]:
    """Handle the /thinking command.

    Args:
        console: Rich console for output.
        raw_command: The full command string.
        current_level: Current thinking level value.
        current_budget: Current thinking budget value.

    Returns:
        Tuple of (thinking_level, thinking_budget) if changed, None otherwise.
    """
    parts = raw_command.split()

    if len(parts) == 1:
        console.print(
            f"[cyan]Current thinking setting:[/cyan] {format_thinking_setting(current_level, current_budget)}"
        )
        print_thinking_help(console)
        return None

    target = parts[1].lower()
    if target in THINKING_LEVELS:
        settings.google_thinking_level = target
        settings.google_thinking_budget = None
        console.print(f"[green]✓ Thinking level set to {target}[/green]")
        return (target, None)

    if target in THINKING_PRESETS:
        value = THINKING_PRESETS[target]
    else:
        try:
            value = int(target)
        except ValueError:
            console.print(
                "[red]Invalid thinking value. Use a level (low/medium/high/minimal), "
                "a preset (low/medium/high/dynamic), or integer tokens.[/red]"
            )
            return None

    settings.google_thinking_level = None
    settings.google_thinking_budget = value
    console.print(f"[green]✓ Thinking budget set to {format_thinking_budget(value)}[/green]")
    return (None, value)
