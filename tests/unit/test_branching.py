"""Unit tests for conversation branching functionality."""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime
from agrag.cli.branching import BranchManager, BranchInfo


def test_branch_manager_initialization():
    """Test BranchManager initialization."""
    checkpointer = Mock()
    thread_id = "test-thread-123"
    
    manager = BranchManager(checkpointer, thread_id)
    
    assert manager.checkpointer == checkpointer
    assert manager.thread_id == thread_id
    assert manager.current_branch == "main"


def test_list_checkpoints_no_checkpointer():
    """Test listing checkpoints when checkpointer is None."""
    manager = BranchManager(None, "test-thread")
    
    checkpoints = manager.list_checkpoints()
    
    assert checkpoints == []


def test_list_checkpoints_with_data():
    """Test listing checkpoints with mock data."""
    checkpointer = Mock()
    
    # Create mock checkpoint tuples
    mock_checkpoint1 = Mock()
    mock_checkpoint1.checkpoint = {
        "id": "checkpoint-1",
        "channel_values": {"messages": [1, 2, 3]},
    }
    mock_checkpoint1.metadata = {"checkpoint_ns": "namespace-1"}
    
    mock_checkpoint2 = Mock()
    mock_checkpoint2.checkpoint = {
        "id": "checkpoint-2",
        "channel_values": {"messages": [1, 2]},
    }
    mock_checkpoint2.metadata = {"checkpoint_ns": "namespace-2"}
    
    checkpointer.list = Mock(return_value=[mock_checkpoint1, mock_checkpoint2])
    
    manager = BranchManager(checkpointer, "test-thread")
    checkpoints = manager.list_checkpoints()
    
    assert len(checkpoints) == 2
    assert checkpoints[0]["checkpoint_id"] == "checkpoint-1"
    assert checkpoints[0]["messages"] == 3
    assert checkpoints[1]["checkpoint_id"] == "checkpoint-2"
    assert checkpoints[1]["messages"] == 2


def test_create_branch_auto_name():
    """Test creating a branch with auto-generated name."""
    checkpointer = Mock()
    manager = BranchManager(checkpointer, "test-thread")
    
    new_thread_id = manager.create_branch("checkpoint-123")
    
    assert new_thread_id.startswith("test-thread_fork_")
    assert "checkpoint-123" not in new_thread_id  # Uses timestamp, not checkpoint ID


def test_create_branch_custom_name():
    """Test creating a branch with custom name."""
    checkpointer = Mock()
    manager = BranchManager(checkpointer, "test-thread")
    
    new_thread_id = manager.create_branch("checkpoint-123", "my-branch")
    
    assert new_thread_id == "test-thread_my-branch"


def test_list_branches():
    """Test listing branches."""
    checkpointer = Mock()
    manager = BranchManager(checkpointer, "test-thread")
    
    branches = manager.list_branches()
    
    # Currently returns main branch only
    assert len(branches) == 1
    assert branches[0].branch_id == "main"
    assert isinstance(branches[0], BranchInfo)


def test_branch_info_dataclass():
    """Test BranchInfo dataclass."""
    branch = BranchInfo(
        branch_id="test-branch",
        checkpoint_id="checkpoint-456",
        parent_checkpoint="checkpoint-123",
        created_at=datetime(2024, 1, 1, 12, 0, 0),
        message_count=5,
        description="Test branch",
    )
    
    assert branch.branch_id == "test-branch"
    assert branch.checkpoint_id == "checkpoint-456"
    assert branch.parent_checkpoint == "checkpoint-123"
    assert branch.message_count == 5
    assert branch.description == "Test branch"


def test_list_checkpoints_error_handling():
    """Test checkpoint listing with exception."""
    checkpointer = Mock()
    checkpointer.list = Mock(side_effect=Exception("Database error"))
    
    manager = BranchManager(checkpointer, "test-thread")
    checkpoints = manager.list_checkpoints()
    
    # Should return empty list on error
    assert checkpoints == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
