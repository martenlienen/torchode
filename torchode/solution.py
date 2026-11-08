from typing import Any

from .typing import *


class Solution:
    def __init__(
        self,
        ts: EvaluationTimesTensor,
        ys: SolutionDataTensor,
        stats: dict[str, Any],
        status: StatusTensor,
    ):
        self.ts = ts
        self.ys = ys
        self.stats = stats
        self.status = status

    def __repr__(self):
        return (
            f"Solution(ts={self.ts}, ys={self.ys}, "
            f"stats={self.stats}, status={self.status})"
        )
