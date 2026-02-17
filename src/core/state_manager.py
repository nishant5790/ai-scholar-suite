"""State manager for serializing and deserializing paper state."""

import json
from pathlib import Path

from src.models.schemas import PaperState


class StateManager:
    """Handles saving and loading PaperState to/from JSON files."""

    def save_state(self, paper_state: PaperState, file_path: str) -> None:
        """Serialize a PaperState to a JSON file.

        Args:
            paper_state: The paper state to save.
            file_path: Path to the output JSON file.

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(paper_state.model_dump_json(indent=2), encoding="utf-8")

    def load_state(self, file_path: str) -> PaperState:
        """Deserialize a JSON file to a PaperState.

        Args:
            file_path: Path to the JSON file to load.

        Returns:
            The deserialized PaperState.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file contains invalid JSON or schema.
        """
        path = Path(file_path)
        json_str = path.read_text(encoding="utf-8")
        return PaperState.model_validate_json(json_str)
