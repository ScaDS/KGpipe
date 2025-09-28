import json, os, argparse
from collections import defaultdict

def type_name(x):
    if x is None: return "null"
    if isinstance(x, bool): return "bool"
    if isinstance(x, (int, float)): return "number"
    if isinstance(x, str): return "string"
    if isinstance(x, list): return "array"
    if isinstance(x, dict): return "object"
    return "unknown"

def enumerate_paths(value, prefix="", include_containers=False, wildcard_arrays=True,
                    leaf_only=True, include_type_suffix=False):
    """
    Returns a set of JSONPaths for 'value'.
    - Arrays are generalized to [*] when wildcard_arrays=True.
    - If leaf_only=False, includes container nodes too.
    - If include_type_suffix=True, appends :type at leaves (e.g., 'a.b:string').
    """
    paths = set()

    def add_path(p, v, is_leaf):
        if include_type_suffix and is_leaf:
            p = f"{p}:{type_name(v)}"
        paths.add(p)

    def walk(v, p):
        if isinstance(v, dict):
            if include_containers and not leaf_only:
                add_path(p or "$", v, is_leaf=False)
            for k, vv in v.items():
                kp = f"{p}.{k}" if p else k
                walk(vv, kp)
        elif isinstance(v, list):
            if include_containers and not leaf_only:
                add_path(p or "$", v, is_leaf=False)
            if not v:  # empty array → still record the path to the array
                ap = f"{p}[*]" if (p and wildcard_arrays) else (f"{p}[0]" if p else "[0]")
                add_path(ap, None, is_leaf=True)  # no inner leaf; mark presence
            else:
                # recurse into elements using wildcard
                ap = f"{p}[*]" if p else "[*]"
                for elem in v:
                    walk(elem, ap if wildcard_arrays else ap.replace("[*]", "[0]"))
        else:
            # primitive/null → leaf
            add_path(p or "$", v, is_leaf=True)

    walk(value, prefix)
    return paths

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def greedy_set_cover(universe, sets_by_doc):
    covered = set()
    chosen = []
    remaining = set(sets_by_doc.keys())
    while covered != universe:
        best_doc, best_gain = None, -1
        for d in remaining:
            gain = len(sets_by_doc[d] - covered)
            if gain > best_gain:
                best_gain = gain
                best_doc = d
        if best_doc is None or best_gain == 0:
            # No progress (shouldn't happen if universe was built from these sets)
            break
        chosen.append(best_doc)
        covered |= sets_by_doc[best_doc]
        remaining.remove(best_doc)
    return chosen, covered

def exact_set_cover(universe, sets_by_doc) -> tuple[list[str], set[str]]:
    """
    Optional exact solver using ILP via pulp (install: pip install pulp).
    Minimizes the number of selected documents subject to covering all paths.
    """
    try:
        import pulp
    except ImportError:
        raise RuntimeError("Exact mode requires 'pulp' (pip install pulp)")

    docs = list(sets_by_doc.keys())
    x = pulp.LpVariable.dicts("use", docs, lowBound=0, upBound=1, cat="Binary")
    prob = pulp.LpProblem("json_path_cover", pulp.LpMinimize)
    prob += pulp.lpSum([x[d] for d in docs])  # objective

    # For each path, require at least one selected doc that contains it
    for p in universe:
        prob += pulp.lpSum([x[d] for d in docs if p in sets_by_doc[d]]) >= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    picked = [d for d in docs if x[d].value() == 1]
    return picked, None

def main():
    parser = argparse.ArgumentParser(description="Pick minimal sample of JSON docs that covers all JSONPaths.")
    parser.add_argument("folder", help="Folder containing .json files")
    parser.add_argument("--include-containers", action="store_true",
                        help="Also include object/array container paths (not just leaves)")
    parser.add_argument("--no-wildcards", action="store_true",
                        help="Do NOT wildcard array indices (use [0] instead of [*])")
    parser.add_argument("--include-type", action="store_true",
                        help="Append :type to leaf paths (e.g., .price:number)")
    parser.add_argument("--exact", action="store_true",
                        help="Use an exact ILP solver (requires 'pulp').")
    args = parser.parse_args()

    folder = args.folder
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".json")]
    if not files:
        print("No .json files found.")
        return

    sets_by_doc = {}
    for fp in files:
        try:
            data = load_json(fp)
        except Exception as e:
            print(f"Skipping {fp}: {e}")
            continue
        paths = enumerate_paths(
            data,
            include_containers=args.include_containers,
            wildcard_arrays=not args.no_wildcards,
            leaf_only=not args.include_containers,
            include_type_suffix=args.include_type,
        )
        sets_by_doc[fp] = paths

    universe = set().union(*sets_by_doc.values()) if sets_by_doc else set()
    if not universe:
        print("No paths found in the provided files.")
        return

    if args.exact:
        picked, _ = exact_set_cover(universe, sets_by_doc)
    else:
        picked, covered = greedy_set_cover(universe, sets_by_doc)

    print("\n=== Summary ===")
    print(f"Total unique paths: {len(universe)}")
    print(f"Total documents scanned: {len(sets_by_doc)}")
    print(f"Selected documents: {len(picked)}\n")
    for i, p in enumerate(picked, 1):
        print(f"{i:2d}. {p}")

    # (optional) show which paths each picked doc contributes
    print("\n=== Coverage by selected documents ===")
    covered_so_far = set()
    for p in picked:
        contrib = sets_by_doc[p] - covered_so_far
        print(f"\n{p}  → contributes {len(contrib)} new paths")
        for path in sorted(contrib):
            print(f"  - {path}")
        covered_so_far |= sets_by_doc[p]

if __name__ == "__main__":
    main()
