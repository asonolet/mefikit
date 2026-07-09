"""Tests for mesh cell splitting functionality."""

import numpy as np
import mefikit as mf


class TestSplitCells:
    """Test cases for cell splitting operations."""

    def test_split_simple_1d(self):
        """Test splitting a simple 1D mesh to confirm python binding works."""
        # Create a 1D mesh with 2 nodes and 1 SEG2 element
        coords = np.array([[0.0, 1.0], [0.0, 1.0]])
        mesh = mf.UMesh(coords)
        mesh.add_regular_block("SEG2", np.array([[0, 1]], dtype=np.uint))

        # Split each cell into 2 sub-cells
        splitted_mesh = mesh.split()
        assert len(splitted_mesh.coords()) == 3
        assert splitted_mesh.num_elements() == 2
