from typing import List, Dict
import re
import json


def tasks_to_mermaid(tasks: List[Dict], *, title: str = "Data Flow", direction: str = "LR") -> str:
    """
    Convert a list of task dicts into a Mermaid flowchart.
    
    Example input:
        [
          {"name": "corenlp", "input": ["TEXT"], "output": ["0_TE_JSON"]},
          {"name": "spotlight", "input": ["TEXT"], "output": ["1_TE_JSON"]},
        ]
    """
    all_tasks = set()
    all_data = set()
    edges_in, edges_out = [], []

    for t in tasks:
        name = t["name"]
        ins = t.get("input", []) or []
        outs = t.get("output", []) or []
        all_tasks.add(name)
        for i in ins:
            all_data.add(i)
            edges_in.append((i, name))
        for o in outs:
            all_data.add(o)
            edges_out.append((name, o))

    # Sanitize ids for Mermaid
    def task_id(name: str) -> str:
        return f"t_{re.sub(r'[^A-Za-z0-9_]', '_', name)}"
    def data_id(name: str) -> str:
        return f"d_{re.sub(r'[^A-Za-z0-9_]', '_', name)}"

    lines = [f"flowchart {direction}", f"%% {title}"]

    # Declare nodes
    for d in sorted(all_data):
        lines.append(f'    {data_id(d)}(({d}))')   # rounded nodes for data
    for t in sorted(all_tasks):
        lines.append(f'    {task_id(t)}[{t}]')     # rectangles for tasks

    # Edges
    for src, dst in edges_in:
        lines.append(f'    {data_id(src)} --> {task_id(dst)}')
    for src, dst in edges_out:
        lines.append(f'    {task_id(src)} --> {data_id(dst)}')

    return "\n".join(lines)


if __name__ == "__main__":
    with open("all_tasks_dict.json", "r") as f:
        all_tasks_dict = json.load(f)

    # for task_dict in all_tasks_dict:
    #     print(task_dict["name"])
    #     print(task_dict["input"])
    #     print(task_dict["output"])
    #     print("-" * 100)

    for idx, task_dict in enumerate(all_tasks_dict):
        with open(f"task_{idx}.mmd", "w") as f:
            f.write(tasks_to_mermaid(task_dict))