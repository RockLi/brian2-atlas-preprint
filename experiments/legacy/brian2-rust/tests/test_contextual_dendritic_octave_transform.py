"""No-simulation checks for the pinned MATLAB-to-Octave source adaptation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from contextual_dendritic_octave_job import transform  # noqa: E402


class S5ATransformTest(unittest.TestCase):
    def test_normal_draw_is_masked_and_normalized_as_one_expression(self) -> None:
        source = """clear all
repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
W_CtoI_0=normrnd(8,4,N_I,N_C).*W_CtoI_mask0./sum(W_CtoI_mask0,2);
mean(x,1,'omitmissing');
%% plot results
"""
        staged, edits, _ = transform(source, "s5a", 0, None, 10)
        self.assertIn(
            "W_CtoI_0=(8 + 4*randn(N_I,N_C)).*W_CtoI_mask0./sum(W_CtoI_mask0,2);",
            staged,
        )
        self.assertNotIn(
            "W_CtoI_0=8 + 4*randn(N_I,N_C).*W_CtoI_mask0",
            staged,
        )
        self.assertTrue(
            any("parentheses preserve the following mask" in edit["reason"] for edit in edits)
        )


if __name__ == "__main__":
    unittest.main()
