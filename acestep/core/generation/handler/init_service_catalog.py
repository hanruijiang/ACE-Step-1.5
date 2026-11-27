"""Catalog and capability helpers for initialization flow."""

import os
from typing import List


class InitServiceCatalogMixin:
    """Checkpoint discovery and backend capability helpers."""

    def _get_project_root(self) -> str:
        """Get project root directory path.

        Returns the directory set by the ``ACESTEP_PROJECT_ROOT`` environment
        variable when present, otherwise the current working directory.  Using
        the working directory (rather than ``__file__``) keeps generated cache
        files and the checkpoints folder next to where the user launched the
        process, regardless of whether the package was installed via
        ``pip install .`` or run from source.
        """
        from acestep.model_downloader import get_project_root
        root = getattr(self, "project_root", None)
        if root:
            return os.path.abspath(os.path.expanduser(root))
        return str(get_project_root())

    def _device_type(self) -> str:
        """Normalize the host device value to a backend type string."""
        if isinstance(self.device, str):
            return self.device.split(":", 1)[0]
        return self.device.type

    def _resolve_checkpoint_dir(self) -> str:
        """Return the checkpoints directory, respecting ACESTEP_CHECKPOINTS_DIR."""
        from acestep.model_downloader import get_checkpoints_dir
        return str(get_checkpoints_dir(getattr(self, "checkpoint_dir", None), project_root=self._get_project_root()))

    def get_available_acestep_v15_models(self) -> List[str]:
        """Scan and return all model directory names starting with ``acestep-v15-``."""
        checkpoint_dir = self._resolve_checkpoint_dir()

        models = []
        if os.path.exists(checkpoint_dir):
            for item in os.listdir(checkpoint_dir):
                item_path = os.path.join(checkpoint_dir, item)
                if os.path.isdir(item_path) and item.startswith("acestep-v15-"):
                    models.append(item)

        models.sort()
        return models

    def is_turbo_model(self) -> bool:
        """Check whether the currently loaded model is a turbo variant."""
        if self.config is None:
            return False
        return getattr(self.config, "is_turbo", False)
