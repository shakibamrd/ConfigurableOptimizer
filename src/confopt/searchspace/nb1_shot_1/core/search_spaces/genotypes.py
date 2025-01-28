from __future__ import annotations

PRIMITIVES = ["maxpool3x3", "conv3x3-bn-relu", "conv1x1-bn-relu"]


class NASBench1Shot1ConfoptGenotype:
    def __init__(self, matrix: list[list[float]], ops: list[str]) -> None:
        self.matrix = matrix
        self.ops = ops

    def tostr(self) -> str:
        return "(" + repr(self.matrix) + "," + str(self.ops) + ")"


if __name__ == "__main__":
    genotype = NASBench1Shot1ConfoptGenotype(
        [[1, 2, 3], [2, 1, 2]], ["input", "maxpool3x3", "output"]
    )
    matrix, ops = eval(genotype.tostr())
    print("matrix", matrix)
    print("ops", ops)
