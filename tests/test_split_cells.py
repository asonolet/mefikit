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

    def test_split_quad4(self):
        """Test splitting a simple 2D mesh to confirm the NON merging of nodes after split
         6    7    8
          .___.___.
          |   |   |
        3 .___.___. 5
          |   |   |
          .___.___.
         0    1    2
        """

        coords = np.array(
            [
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )
        mesh = mf.UMesh(coords)
        mesh.add_regular_block(
            "QUAD4",
            np.array(
                [[0, 1, 4, 3], [1, 2, 5, 4], [4, 5, 8, 7], [3, 4, 7, 6]], dtype=np.uint
            ),
        )

        # Split each cell into 4 sub-cells and check duplicated nodes are not merged
        splitted_mesh = mesh.split()
        assert (
            len(splitted_mesh.coords()) == 29
        )  # 9 originals + 12 edges + 4 centers + 4 duplicated edges
        assert splitted_mesh.num_elements() == 16

        unique_nodes = set(splitted_mesh.blocks()["QUAD4"].flatten())
        assert len(unique_nodes) == 25  # 4 edges have duplicated nodes
