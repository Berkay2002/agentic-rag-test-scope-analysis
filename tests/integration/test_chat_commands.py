"""Integration tests for interactive chat commands."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from rich.console import Console
from io import StringIO

from agrag.cli.commands import CommandHandler


class MockSession:
    """Mock chat session for testing."""
    
    def __init__(self):
        self.thread_id = "test-thread-123"
        self.message_count = 5
        self.tool_calls_total = 10
        self.model_calls_total = 8
        self.start_time = datetime(2024, 1, 1, 12, 0, 0)
        self.thinking_budget = 1024
        self.thinking_level = None
        self.enable_hitl = True
        self.verbose = False
        self.checkpointer = Mock()
    
    def _persistence_label(self):
        return "PostgreSQL (durable)"
    
    def export_conversation(self, filename=None, include_tool_details=False):
        return "test_conversation.md"
    
    def reset_conversation(self):
        pass
    
    def set_verbose(self, enabled):
        self.verbose = enabled


def test_help_command():
    """Test /help command displays help."""
    console = Console(file=StringIO())
    session = MockSession()
    handler = CommandHandler(console, session)
    
    result = handler.handle("/help")
    
    assert result is True  # Should continue


def test_stats_command():
    """Test /stats command displays statistics."""
    console = Console(file=StringIO())
    session = MockSession()
    handler = CommandHandler(console, session)
    
    result = handler.handle("/stats")
    
    assert result is True


def test_exit_command():
    """Test /exit command returns False to exit."""
    console = Console(file=StringIO())
    session = MockSession()
    handler = CommandHandler(console, session)
    
    result = handler.handle("/exit")
    
    assert result is False


def test_quit_command():
    """Test /quit command returns False to exit."""
    console = Console(file=StringIO())
    session = MockSession()
    handler = CommandHandler(console, session)
    
    result = handler.handle("/quit")
    
    assert result is False


def test_verbose_toggle():
    """Test /verbose command toggles verbose mode."""
    console = Console(file=StringIO())
    session = MockSession()
    handler = CommandHandler(console, session)
    
    assert session.verbose is False
    
    handler.handle("/verbose")
    assert session.verbose is True
    
    handler.handle("/verbose")
    assert session.verbose is False


def test_verbose_on():
    """Test /verbose on command enables verbose mode."""
    console = Console(file=StringIO())
    session = MockSession()
    handler = CommandHandler(console, session)
    
    handler.handle("/verbose off")
    assert session.verbose is False
    
    handler.handle("/verbose on")
    assert session.verbose is True


def test_reset_command():
    """Test /reset command resets session."""
    console = Console(file=StringIO())
    session = MockSession()
    handler = CommandHandler(console, session)
    
    original_thread_id = session.thread_id
    
    handler.handle("/reset")
    
    # Thread ID should change
    assert session.thread_id != original_thread_id
    assert session.thread_id.startswith("chat-")
    
    # Counters should reset
    assert session.message_count == 0
    assert session.tool_calls_total == 0
    assert session.model_calls_total == 0


def test_export_command():
    """Test /export command exports conversation."""
    console = Console(file=StringIO())
    session = MockSession()
    handler = CommandHandler(console, session)
    
    result = handler.handle("/export")
    
    assert result is True


def test_export_with_filename():
    """Test /export command with custom filename."""
    console = Console(file=StringIO())
    session = MockSession()
    handler = CommandHandler(console, session)
    
    result = handler.handle("/export my_conversation.md")
    
    assert result is True


def test_export_with_verbose_flag():
    """Test /export command with --verbose flag."""
    console = Console(file=StringIO())
    session = MockSession()
    handler = CommandHandler(console, session)
    
    result = handler.handle("/export --verbose")
    
    assert result is True


def test_history_command_with_checkpointer():
    """Test /history command with checkpointer."""
    console = Console(file=StringIO())
    session = MockSession()
    
    # Mock checkpointer list method
    mock_checkpoint = Mock()
    mock_checkpoint.checkpoint = {
        "id": "checkpoint-1",
        "channel_values": {"messages": [1, 2, 3]},
    }
    mock_checkpoint.metadata = {"checkpoint_ns": "default"}
    session.checkpointer.list = Mock(return_value=[mock_checkpoint])
    
    handler = CommandHandler(console, session)
    result = handler.handle("/history")
    
    assert result is True


def test_history_command_without_checkpointer():
    """Test /history command without checkpointer."""
    console = Console(file=StringIO())
    session = MockSession()
    session.checkpointer = None
    
    handler = CommandHandler(console, session)
    result = handler.handle("/history")
    
    assert result is True  # Should show warning


def test_branches_command():
    """Test /branches command."""
    console = Console(file=StringIO())
    session = MockSession()
    handler = CommandHandler(console, session)
    
    result = handler.handle("/branches")
    
    assert result is True


def test_fork_command():
    """Test /fork command creates branch."""
    console = Console(file=StringIO())
    session = MockSession()
    handler = CommandHandler(console, session)
    
    result = handler.handle("/fork checkpoint-123 my-branch")
    
    assert result is True


def test_fork_command_auto_name():
    """Test /fork command with auto-generated branch name."""
    console = Console(file=StringIO())
    session = MockSession()
    handler = CommandHandler(console, session)
    
    result = handler.handle("/fork checkpoint-123")
    
    assert result is True


def test_fork_command_without_checkpoint():
    """Test /fork command without checkpoint shows usage."""
    console = Console(file=StringIO())
    session = MockSession()
    handler = CommandHandler(console, session)
    
    result = handler.handle("/fork")
    
    assert result is True  # Should show usage


def test_unknown_command():
    """Test unknown command shows error."""
    console = Console(file=StringIO())
    session = MockSession()
    handler = CommandHandler(console, session)
    
    result = handler.handle("/unknown")
    
    assert result is True  # Should continue


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
