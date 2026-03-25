from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class DagNode:
    name: str
    inputs: Tuple[str, ...] = ()
    output: str | None = None


class Dag:
    """Minimal DAG API with dependency validation and parallel batches."""

    def __init__(self) -> None:
        self._nodes: Dict[str, DagNode] = {}
        self._parents: Dict[str, set[str]] = defaultdict(set)
        self._children: Dict[str, set[str]] = defaultdict(set)
        self._data_producers: Dict[str, str] = {}

    def task(self, name: str, *, needs: Sequence[str] = (), produces: str | None = None) -> Dag:
        if name in self._nodes:
            raise ValueError(f"Task '{name}' already exists")

        if produces and produces in self._data_producers:
            producer = self._data_producers[produces]
            raise ValueError(f"Data '{produces}' is already produced by '{producer}'")

        node = DagNode(name=name, inputs=tuple(needs), output=produces)
        self._nodes[name] = node
        if produces:
            self._data_producers[produces] = name
        return self

    def wire(self) -> Dag:
        """Resolve data dependencies into task edges."""
        for node in self._nodes.values():
            for data_id in node.inputs:
                parent = self._data_producers.get(data_id)
                if parent is None:
                    raise ValueError(
                        f"Task '{node.name}' requires '{data_id}', but no upstream task produces it"
                    )
                self._parents[node.name].add(parent)
                self._children[parent].add(node.name)

        self._assert_acyclic()
        return self

    def execution_batches(self) -> List[List[str]]:
        """
        Return topological levels.
        Tasks in the same inner list can run in parallel.
        """
        indegree = {name: len(self._parents[name]) for name in self._nodes}
        frontier = deque(sorted([n for n, d in indegree.items() if d == 0]))
        batches: List[List[str]] = []

        while frontier:
            level: List[str] = list(frontier)
            frontier.clear()
            batches.append(level)

            for task_name in level:
                for child in sorted(self._children[task_name]):
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        frontier.append(child)

        total = sum(len(batch) for batch in batches)
        if total != len(self._nodes):
            raise ValueError("Graph contains a cycle")
        return batches

    def edges(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for parent, children in sorted(self._children.items()):
            for child in sorted(children):
                out.append((parent, child))
        return out

    def to_mermaid_mmd(self, direction: str = "LR") -> str:
        """
        Export graph as Mermaid mmd text.
        Call this after `wire()` so task dependencies are resolved.
        """
        lines: List[str] = [f"flowchart {direction}"]

        for node_name, node in sorted(self._nodes.items()):
            node_lines = [node.name]
            if node.inputs:
                node_lines.append(f"needs: {', '.join(node.inputs)}")
            if node.output:
                node_lines.append(f"produces: {node.output}")
            label = "<br/>".join(node_lines)
            lines.append(f'    {node_name}["{label}"]')

        for parent, child in self.edges():
            lines.append(f"    {parent} --> {child}")

        return "\n".join(lines)

    def _assert_acyclic(self) -> None:
        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(node_name: str) -> None:
            visited.add(node_name)
            in_stack.add(node_name)
            for child_name in self._children[node_name]:
                if child_name not in visited:
                    dfs(child_name)
                elif child_name in in_stack:
                    raise ValueError(f"Cycle detected at '{child_name}'")
            in_stack.remove(node_name)

        for name in self._nodes:
            if name not in visited:
                dfs(name)


def dag_example() -> Dag:
    """
    Typical syntax:
    - split: one output consumed by several branches
    - join: one task requiring multiple inputs
    - final output: last task produces the sink artifact
    """
    dag = (
        Dag()
        .task("load_users", produces="users")
        .task("load_orders", produces="orders")
        .task("clean_users", needs=("users",), produces="users_clean")
        .task("clean_orders", needs=("orders",), produces="orders_clean")
        .task("extract_features_a", needs=("users_clean","orders_clean"), produces="features_a")
        .task("extract_features_b", needs=("users_clean",), produces="features_b")
        .task(
            "join_user_order_features",
            needs=("features_a", "features_b", "orders_clean"),
            produces="joined_features",
        )
        .task("train_model", needs=("joined_features",), produces="model")
        .task("evaluate_model", needs=("model",), produces="report")
        .wire()
    )
    return dag


if __name__ == "__main__":
    dag = dag_example()
    print("Edges:", dag.edges())
    print("Parallel batches:", dag.execution_batches())
    print("\nMermaid mmd:\n")
    print(dag.to_mermaid_mmd())


