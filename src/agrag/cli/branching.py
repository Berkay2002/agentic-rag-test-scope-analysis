"""Branch management for conversation checkpoints.

This module implements git-like branching for conversation threads,
allowing users to fork from checkpoints and explore different conversation paths.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text
from langgraph.checkpoint.base import BaseCheckpointSaver

from agrag.cli.display import COLORS

logger = logging.getLogger(__name__)


@dataclass
class BranchInfo:
    """Information about a conversation branch."""
    
    branch_id: str
    checkpoint_id: str
    parent_checkpoint: Optional[str]
    created_at: datetime
    message_count: int
    description: str


class BranchManager:
    """Manages conversation branching and checkpoints."""
    
    def __init__(self, checkpointer: BaseCheckpointSaver, thread_id: str):
        """Initialize branch manager.
        
        Args:
            checkpointer: Checkpoint saver instance.
            thread_id: Current thread ID.
        """
        self.checkpointer = checkpointer
        self.thread_id = thread_id
        self.current_branch = "main"
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints for the current thread.
        
        Returns:
            List of checkpoint metadata dicts.
        """
        if not self.checkpointer:
            return []
        
        try:
            checkpoints = []
            config = {"configurable": {"thread_id": self.thread_id}}
            
            # Get checkpoint history from checkpointer
            for checkpoint_tuple in self.checkpointer.list(config):
                checkpoint, metadata = checkpoint_tuple.checkpoint, checkpoint_tuple.metadata
                checkpoints.append({
                    "checkpoint_id": checkpoint.get("id", "unknown"),
                    "checkpoint_ns": metadata.get("checkpoint_ns", ""),
                    "messages": len(checkpoint.get("channel_values", {}).get("messages", [])),
                    "metadata": metadata,
                })
            
            return checkpoints
        except Exception as e:
            logger.warning(f"Failed to list checkpoints: {e}")
            return []
    
    def create_branch(self, checkpoint_id: str, branch_name: Optional[str] = None) -> str:
        """Create a new branch from a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to branch from.
            branch_name: Optional branch name (auto-generated if not provided).
            
        Returns:
            New branch ID.
        """
        if not branch_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            branch_name = f"fork_{timestamp}"
        
        # In LangGraph, branching is done by creating a new thread with parent metadata
        new_thread_id = f"{self.thread_id}_{branch_name}"
        
        logger.info(f"Created branch {branch_name} at checkpoint {checkpoint_id}")
        return new_thread_id
    
    def list_branches(self) -> List[BranchInfo]:
        """List all branches for this conversation.
        
        Returns:
            List of BranchInfo objects.
        """
        # For now, return a simple list showing the main branch
        # In a full implementation, this would query the checkpointer for all related threads
        return [
            BranchInfo(
                branch_id="main",
                checkpoint_id="latest",
                parent_checkpoint=None,
                created_at=datetime.now(),
                message_count=0,
                description="Main conversation branch",
            )
        ]


def print_checkpoints(console: Console, checkpoints: List[Dict[str, Any]]) -> None:
    """Display checkpoints in a formatted table.
    
    Args:
        console: Rich console for output.
        checkpoints: List of checkpoint metadata dicts.
    """
    if not checkpoints:
        console.print("[yellow]No checkpoints found for this thread.[/yellow]")
        return
    
    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("#", style=COLORS["neutral"], width=4)
    table.add_column("Checkpoint ID", style=COLORS["accent"], width=20)
    table.add_column("Messages", style=COLORS["primary"], width=10)
    table.add_column("Namespace", style="dim", width=15)
    
    for idx, checkpoint in enumerate(checkpoints[:20], 1):  # Limit to 20 most recent
        checkpoint_id = checkpoint.get("checkpoint_id", "unknown")
        messages = checkpoint.get("messages", 0)
        namespace = checkpoint.get("checkpoint_ns", "")
        
        table.add_row(
            str(idx),
            checkpoint_id[:20] + "..." if len(checkpoint_id) > 20 else checkpoint_id,
            str(messages),
            namespace,
        )
    
    console.print()
    console.print(table)
    console.print()
    
    if len(checkpoints) > 20:
        console.print(f"[dim]Showing 20 of {len(checkpoints)} checkpoints[/dim]")


def print_branches(console: Console, branches: List[BranchInfo]) -> None:
    """Display branches in a formatted table.
    
    Args:
        console: Rich console for output.
        branches: List of BranchInfo objects.
    """
    if not branches:
        console.print("[yellow]No branches found.[/yellow]")
        return
    
    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Branch", style=COLORS["accent"], width=20)
    table.add_column("Checkpoint", style=COLORS["neutral"], width=20)
    table.add_column("Messages", style=COLORS["primary"], width=10)
    table.add_column("Created", style="dim", width=20)
    
    for branch in branches:
        created_str = branch.created_at.strftime("%Y-%m-%d %H:%M:%S")
        table.add_row(
            branch.branch_id,
            branch.checkpoint_id[:20] if len(branch.checkpoint_id) > 20 else branch.checkpoint_id,
            str(branch.message_count),
            created_str,
        )
    
    console.print()
    console.print(table)
    console.print()
