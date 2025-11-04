
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Literal, Tuple
from enum import Enum
from dataclasses import dataclass
import csv
import json
import itertools

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

# -----------------------------
# Enums & small helpers
# -----------------------------

class SourceType(str, Enum):
    rdf = "rdf"
    json = "json"
    text = "text"

    def __str__(self):
        return self.value

class MatchType(str, Enum):
    exact = "exact"
    variant = "variant"
    broader = "broader"
    narrower = "narrower"
    related = "related"

EXPECTED_HEADERS = {
    "verified_matches.csv": ["left_dataset","right_dataset","left_id","right_id","entity_type"],
    "verified_entities.csv": ["dataset","entity_id","entity_label","entity_type"],
    "verified_links.csv": ["doc_id","entity_id","entity_type","dataset"],
}

def read_csv_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, [])
    return header

def ensure_headers(path: Path, expected: List[str]) -> None:
    got = read_csv_header(path)
    if got != expected:
        raise ValueError(f"Bad header for {path}: expected {expected}, got {got}")

def list_parts(dirpath: Path, exts: Tuple[str, ...]) -> List[Path]:
    return sorted([p for p in dirpath.glob("*") if p.suffix in exts and p.is_file()])

# -----------------------------
# Metadata models
# -----------------------------

class EntitiesRow(BaseModel):
    entity_id: str
    entity_label: str
    entity_type: str
    dataset: str

class LinksRow(BaseModel):
    doc_id: str
    entity_id: str
    entity_type: str
    dataset: str

class MatchesRow(BaseModel):
    left_dataset: str
    right_dataset: str
    left_id: str
    right_id: str
    entity_type: str

def read_matches_csv(path: Path) -> List[MatchesRow]:
    return [MatchesRow(left_dataset=row["left_dataset"], right_dataset=row["right_dataset"], left_id=row["left_id"], right_id=row["right_id"], entity_type=row["entity_type"]) for row in csv.DictReader(path.open("r"), delimiter="\t")]

class VerifiedMatches(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    file: Path

    @model_validator(mode="after")
    def _check(self):
        if not self.file.exists():
            raise ValueError(f"verified_matches file not found: {self.file}")
        ensure_headers(self.file, EXPECTED_HEADERS["verified_matches.csv"])
        return self

    def read_csv(self) -> List[MatchesRow]:
        return read_matches_csv(self.file)

def read_entities_csv(path: Path) -> List[EntitiesRow]:
    return [EntitiesRow(entity_id=row["entity_id"], entity_label=row["entity_label"], entity_type=row["entity_type"], dataset=row["dataset"]) for row in csv.DictReader(path.open("r"), delimiter="\t")]

class VerifiedEntities(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    file: Path

    @model_validator(mode="after")
    def _check(self):
        if not self.file.exists():
            raise ValueError(f"verified_entities file not found: {self.file}")
        ensure_headers(self.file, EXPECTED_HEADERS["verified_entities.csv"])
        return self

    def read_csv(self) -> List[EntitiesRow]:
        return read_entities_csv(self.file)

def read_links_csv(path: Path) -> List[LinksRow]:
    return [LinksRow(doc_id=row["doc_id"], entity_id=row["entity_id"], entity_type=row["entity_type"], dataset=row["dataset"]) for row in csv.DictReader(path.open("r"), delimiter="\t")]

class VerifiedLinks(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    file: Path

    @model_validator(mode="after")
    def _check(self):
        if not self.file.exists():
            raise ValueError(f"verified_links file not found: {self.file}")
        ensure_headers(self.file, EXPECTED_HEADERS["verified_links.csv"])
        return self

# -----------------------------
# Source bundles (rdf/json/text)
# -----------------------------

class SourceMeta(BaseModel):
    root: Path
    entities: Optional[VerifiedEntities] = None
    links: Optional[VerifiedLinks] = None
    matches: Optional[VerifiedMatches] = None

    def set_entities(self, entities: List[EntitiesRow]):
        path = self.root / "verified_entities.csv"
        with path.open("w") as f:
            f.write("dataset\tentity_id\tentity_label\tentity_type\n")
            for entity in entities:
                f.write(f"{entity.dataset}\t{entity.entity_id}\t{entity.entity_label}\t{entity.entity_type}\n")
        self.entities = VerifiedEntities(file=path)

    def set_links(self, links: List[LinksRow]):
        path = self.root / "verified_links.csv"
        with path.open("w") as f:
            f.write("doc_id\tentity_id\tentity_type\tdataset\n")
            for link in links:
                f.write(f"{link.doc_id}\t{link.entity_id}\t{link.entity_type}\t{link.dataset}\n")
        self.links = VerifiedLinks(file=path)

    def set_matches(self, matches: List[MatchesRow]):
        path = self.root / "verified_matches.csv"
        with path.open("w") as f:
            f.write("left_dataset\tright_dataset\tleft_id\tright_id\tentity_type\n")
            for match in matches:
                f.write(f"{match.left_dataset}\t{match.right_dataset}\t{match.left_id}\t{match.right_id}\t{match.entity_type}\n")
        self.matches = VerifiedMatches(file=path)

    # def set_provenance(self, provenance: Provenance):
    #     path = self.root / "provenance.json"
    #     with path.open("w") as f:
    #         json.dump(provenance.model_dump(), f, indent=4)
    #     self.provenance = Provenance(file=path)

class SourceData(BaseModel):
    dir: Path
    parts: List[Path] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self):
        if not self.dir.exists():
            raise ValueError(f"missing data dir: {self.dir}")
        return self

class SourceBundle(BaseModel):
    type: SourceType
    root: Path
    data: SourceData
    meta: SourceMeta

    @model_validator(mode="after")
    def _check(self):
        # sanity: no "verfied_" typos
        for csvpath in self.root.rglob("*.csv"):
            if "verfied_" in csvpath.name:
                raise ValueError(f"Filename typo detected (should be 'verified_'): {csvpath}")
        return self

# -----------------------------
# KG bundles (reference / seed)
# -----------------------------

class KGBundle(BaseModel):
    kind: Literal["reference","seed"]
    root: Path
    data: SourceData
    meta: SourceMeta = Field(default_factory=lambda: SourceMeta(root=Path("meta")))

# -----------------------------
# Split
# -----------------------------

class SplitIndex(BaseModel):
    dir: Path
    entities_csv: Path

    # def read_csv(self) -> List[str]:

    # @model_validator(mode="after")
    # def _check(self):
    #     if not self.entities_csv.exists():
    #         raise ValueError(f"index/entities.csv missing for split at {self.dir}")
    #     # minimal header check: must contain "entity_id" column
    #     header = read_csv_header(self.entities_csv)
    #     if "entity_id" not in header:
    #         raise ValueError(f"{self.entities_csv} must contain an 'entity_id' column; got {header}")
    #     return self

class Split(BaseModel):
    split_id: str
    root: Path
    index: SplitIndex
    kg_reference: Optional[KGBundle] = None
    kg_seed: Optional[KGBundle] = None
    sources: Dict[str, SourceBundle]

    def set_index(self, entities: List[EntitiesRow]):
        self.index.dir.mkdir(parents=True, exist_ok=True)
        self.index.entities_csv.touch()
        with self.index.entities_csv.open("w") as f: 
            f.write("entity_id\n")
            for entity in entities:
                f.write(f"{entity.entity_id}\n")

    def set_empty_reference(self):
        ref_root = self.root / "kg" / "reference"
        ref_data_dir = ref_root / "data"
        ref_meta_dir = ref_root / "meta"
        ref_data_dir.mkdir(parents=True, exist_ok=True)
        ref_meta_dir.mkdir(parents=True, exist_ok=True)
        self.kg_reference = KGBundle(
            kind="reference",
            root=ref_root,
            data=SourceData(dir=ref_data_dir, parts=[]),
            meta=SourceMeta(root=ref_meta_dir)
        )
    
    def set_empty_seed(self):
        seed_root = self.root / "kg" / "seed"
        seed_data_dir = seed_root / "data"
        seed_meta_dir = seed_root / "meta"
        seed_data_dir.mkdir(parents=True, exist_ok=True)
        seed_meta_dir.mkdir(parents=True, exist_ok=True)
        self.kg_seed = KGBundle(
            kind="seed",
            root=seed_root,
            data=SourceData(dir=seed_data_dir, parts=[]),
            meta=SourceMeta(root=seed_meta_dir)
        )

    def set_sources(self, sources: List[SourceType]):
        source_dir = self.root / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        for source_type in sources:
            path = source_dir / f"{source_type}"
            path.mkdir(parents=True, exist_ok=True)
            data_dir = path / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            meta_dir = path / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
        
            bundle = SourceBundle(
                type=source_type,
                root=path,
                data=SourceData(dir=data_dir, parts=[]),
                meta=SourceMeta(root=meta_dir)
            )
            self.sources[source_type] = bundle

# -----------------------------
# Provenance (Overlap Plan subset)
# -----------------------------

class PairwiseOverride(BaseModel):
    left: str
    right: str
    targetPct: float

class RealizedPair(BaseModel):
    left: str
    right: str
    pctOfLeft: float
    pctOfRight: float
    count: int

class OverlapPlan(BaseModel):
    overlapGlobalPct: float
    tolerancePct: float = 0.001
    randomSeed: Optional[int] = None
    splitIds: List[str]
    pairwiseOverrides: List[PairwiseOverride] = Field(default_factory=list)

class OverlapRealized(BaseModel):
    pairs: List[RealizedPair] = Field(default_factory=list)

class Provenance(BaseModel):
    plan: Optional[OverlapPlan] = None
    realized: Optional[OverlapRealized] = None
    raw: Dict = Field(default_factory=dict)

# -----------------------------
# Dataset root
# -----------------------------

class Dataset(BaseModel):
    root: Path
    ontology: Optional[Path] = None
    entities_master: Optional[Path] = None
    splits: Dict[str, Split] = Field(default_factory=dict)
    provenance: Optional[Provenance] = None

    def set_entities_master(self, entities_list: List[str]):
        self.entities_master = self.root / "entities" / "master_entities.csv"
        self.entities_master.parent.mkdir(parents=True, exist_ok=True)
        with self.entities_master.open("w") as f:
            f.write("entity_id\n")
            for entity in entities_list:
                f.write(f"{entity}\n")

    def set_splits(self, range_start: int, range_end: int):
        for split_id in range(range_start, range_end):
            split = Split(
                split_id=f"split_{split_id}",
                root=self.root / f"split_{split_id}",
                index=SplitIndex(dir=self.root / f"split_{split_id}" / "index", entities_csv=self.root / f"split_{split_id}" / "index/entities.csv"),
                sources={}
            )
            split.root.mkdir(parents=True, exist_ok=True)
            split.index.dir.mkdir(parents=True, exist_ok=True)
            split.index.entities_csv.touch()
            self.splits[split.split_id] = split



    # -------- utilities --------

    def overlap_matrix(self) -> Dict[Tuple[str,str], Dict[str, int | float]]:
        """Compute pairwise overlap counts based on index/entities.csv for each split."""
        index_sets: Dict[str, set] = {}
        for sid, split in self.splits.items():
            ids = set()
            with split.index.entities_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ids.add(row["entity_id"])
            index_sets[sid] = ids
        pairs: Dict[Tuple[str,str], Dict[str, int | float]] = {}
        for a, b in itertools.combinations(sorted(index_sets.keys()), 2):
            pairs[(a,b)] = {
                "cnt": len(index_sets[a] & index_sets[b]),
                "len_left": len(index_sets[a]),
                "len_right": len(index_sets[b]),
                "pct_left": len(index_sets[a] & index_sets[b]) / len(index_sets[a]),
                "pct_right": len(index_sets[a] & index_sets[b]) / len(index_sets[b]),
            }
        return pairs

    def validate_against_plan(self) -> None:
        if not self.provenance or not self.provenance.plan:
            raise ValueError("No overlap plan found in provenance")
        plan = self.provenance.plan
        pairs = self.overlap_matrix()

        # derive per-pair target
        overrides = {(min(p.left,p.right), max(p.left,p.right)): p.targetPct for p in plan.pairwiseOverrides}
        # get sizes
        sizes = {}
        for sid, split in self.splits.items():
            n = 0
            with split.index.entities_csv.open("r", encoding="utf-8") as f:
                n = sum(1 for _ in csv.DictReader(f))
            sizes[sid] = n

        for (a,b), count in pairs.items():
            na, nb = sizes[a], sizes[b]
            target = overrides.get((a,b), plan.overlapGlobalPct)
            pct_a = count / na if na else 0.0
            pct_b = count / nb if nb else 0.0
            if abs(pct_a - target) > plan.tolerancePct or abs(pct_b - target) > plan.tolerancePct:
                raise ValueError(f"Overlap for ({a},{b}) out of tolerance: "
                                 f"pctOfLeft={pct_a:.4f}, pctOfRight={pct_b:.4f}, target={target:.4f}, tol={plan.tolerancePct:.4f}")

# -----------------------------
# Loader
# -----------------------------

# def _load_overlap_from_prov(prov_path: Path) -> Provenance:
#     try:
#         data = json.loads(prov_path.read_text(encoding="utf-8"))
#     except Exception:
#         return Provenance()
#     # Try to locate plan/realized in a few common shapes
#     plan = None
#     realized = None

#     # Common patterns from the earlier suggestion:
#     # { "@id":"plan:overlap-constraints", ... }
#     def _find_plan(obj):
#         if isinstance(obj, dict) and obj.get("@type") and ("OverlapPlan" in str(obj.get("@type")) or "x:OverlapPlan" in str(obj.get("@type"))):
#             return obj
#         return None

#     node = None
#     if isinstance(data, dict):
#         # maybe a single doc with top-level keys
#         node = data
#         # or an @graph list
#         graph = data.get("@graph")
#         if isinstance(graph, list):
#             for item in graph:
#                 hit = _find_plan(item)
#                 if hit:
#                     node = hit
#                     break
#     elif isinstance(data, list):
#         for item in data:
#             hit = _find_plan(item)
#             if hit:
#                 node = hit
#                 break

#     if isinstance(node, dict):
#         # Accept either "x:overlapGlobalPct" or "overlapGlobalPct"
#         def g(k):
#             return node.get(k) or node.get(f"x:{k}")
#         try:
#             plan = OverlapPlan(
#                 overlapGlobalPct=float(g("overlapGlobalPct")),
#                 tolerancePct=float(g("tolerancePct") or 0.001),
#                 randomSeed=(g("randomSeed") if g("randomSeed") is None else int(g("randomSeed"))),
#                 splitIds=list(g("splitIds") or []),
#                 pairwiseOverrides=[
#                     PairwiseOverride(left=o.get("left"), right=o.get("right"), targetPct=float(o.get("targetPct")))
#                     for o in (g("pairwiseOverrides") or [])
#                 ]
#             )
#         except Exception:
#             plan = None

#     # realized parsing (optional)
#     # look for array like x:realizedOverlap or realizedOverlap with entries
#     realized_pairs = []
#     def collect_realized(obj):
#         nonlocal realized_pairs
#         if isinstance(obj, dict):
#             arr = obj.get("x:realizedOverlap") or obj.get("realizedOverlap")
#             if isinstance(arr, list):
#                 for r in arr:
#                     try:
#                         realized_pairs.append(RealizedPair(
#                             left=r["left"], right=r["right"],
#                             pctOfLeft=float(r["pctOfLeft"]), pctOfRight=float(r["pctOfRight"]),
#                             count=int(r["count"])
#                         ))
#                     except Exception:
#                         pass

#     if isinstance(data, dict):
#         collect_realized(data)
#         for item in data.get("@graph", []):
#             collect_realized(item)
#     elif isinstance(data, list):
#         for item in data:
#             collect_realized(item)

#     if realized_pairs:
#         realized = OverlapRealized(pairs=realized_pairs)

#     return Provenance(plan=plan, realized=realized, raw=data if isinstance(data, dict) else {})

def load_dataset(root: Path) -> Dataset:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    # ontology & master entities (optional but recommended)
    ontology = None
    entities_master = None
    if (root / "ontology" / "ontology.ttl").exists():
        ontology = (root / "ontology" / "ontology.ttl")
    elif (root / "ontology.ttl").exists():
        ontology = (root / "ontology.ttl")
    if (root / "entities" / "master_entities.csv").exists():
        entities_master = (root / "entities" / "master_entities.csv")

    # provenance
    # prov = None
    # p1 = root / "provenance" / "prov.jsonld"
    # p2 = root / "provenance.json"
    # p3 = root / "provenance.jsonld"
    # if p1.exists():
    #     prov = _load_overlap_from_prov(p1)
    # elif p3.exists():
    #     prov = _load_overlap_from_prov(p3)
    # elif p2.exists():
    #     prov = _load_overlap_from_prov(p2)

    # splits
    splits_dir = root / "splits"
    if not splits_dir.exists():
        # fallback to flat split_* in root
        split_dirs = [p for p in root.glob("split_*") if p.is_dir()]
    else:
        split_dirs = [p for p in splits_dir.glob("split_*") if p.is_dir()]

    splits: Dict[str, Split] = {}

    for sdir in sorted(split_dirs):
        sid = sdir.name
        # index
        index_dir = sdir / "index"
        entities_csv = index_dir / "entities.csv"
        index = SplitIndex(dir=index_dir, entities_csv=entities_csv)

        # kg/reference & kg/seed (optional)
        kg_ref = None
        kg_seed = None
        kg_dir = sdir / "kg"
        if kg_dir.exists():
            ref_dir = kg_dir / "reference"
            if ref_dir.exists():
                ref_data_dir = ref_dir / "data"
                ref_meta_dir = ref_dir / "meta"
                ref_parts = list_parts(ref_data_dir, (".nt", ".ttl", ".nq"))
                ref_meta = SourceMeta(root=ref_meta_dir)
                vm = ref_meta_dir / "verified_matches.csv"
                if vm.exists():
                    ref_meta.matches = VerifiedMatches(file=vm)
                ve = ref_meta_dir / "verified_entities.csv"
                if ve.exists():
                    ref_meta.entities = VerifiedEntities(file=ve)
                kg_ref = KGBundle(
                    kind="reference",
                    root=ref_dir,
                    data=SourceData(dir=ref_data_dir, parts=ref_parts),
                    meta=ref_meta
                )
            seed_dir = kg_dir / "seed"
            if seed_dir.exists():
                seed_data_dir = seed_dir / "data"
                seed_meta_dir = seed_dir / "meta"
                seed_parts = list_parts(seed_data_dir, (".nt", ".ttl", ".nq"))
                kg_seed = KGBundle(
                    kind="seed",
                    root=seed_dir,
                    data=SourceData(dir=seed_data_dir, parts=seed_parts),
                    meta=SourceMeta(root=seed_meta_dir)
                )

        # sources
        sources_dir = sdir / "sources"
        sources: Dict[SourceType, SourceBundle] = {}
        for stype in [SourceType.rdf, SourceType.json, SourceType.text]:
            sroot = sources_dir / stype.value
            if not sroot.exists():
                continue
            data_dir = sroot / "data"
            meta_dir = sroot / "meta"
            if stype == SourceType.rdf:
                parts = list_parts(data_dir, (".nt", ".ttl", ".nq"))
            elif stype == SourceType.json:
                parts = list_parts(data_dir, (".jsonl", ".json"))
            else:
                parts = list_parts(data_dir, (".txt",))
            meta = SourceMeta(root=meta_dir)
            ve = meta_dir / "verified_entities.csv"
            if ve.exists():
                meta.entities = VerifiedEntities(file=ve)
            vl = meta_dir / "verified_links.csv"
            if vl.exists():
                meta.links = VerifiedLinks(file=vl)
            vm = meta_dir / "verified_matches.csv"
            if vm.exists():
                meta.matches = VerifiedMatches(file=vm)

            sources[stype] = SourceBundle(
                type=stype,
                root=sroot,
                data=SourceData(dir=data_dir, parts=parts),
                meta=meta
            )

        splits[sid] = Split(
            split_id=sid,
            root=sdir,
            index=index,
            kg_reference=kg_ref,
            kg_seed=kg_seed,
            sources={str(stype): sources[stype] for stype in sources}
        )

    return Dataset(
        root=root,
        ontology=ontology,
        entities_master=entities_master,
        splits=splits,
        provenance=None
    )

# -----------------------------
# CLI (optional helper)
# -----------------------------

def test_multipart_multisource():

    # temp dir
    import tempfile
    import shutil
    tmpdir = tempfile.mkdtemp()

    ds = Dataset(
        root=Path(tmpdir),
    )
    ds.set_entities_master(["a", "b", "c"])

    range_start = 0
    range_end = 4
    ds.set_splits(range_start, range_end)

    for split_id in range(range_start, range_end):
        split = ds.splits[f"split_{split_id}"]

        # Prepare reusable entity/link/match rows
        entities_1 = [
            EntitiesRow(entity_id="a", entity_label="a", entity_type="a", dataset="a"),
            EntitiesRow(entity_id="b", entity_label="b", entity_type="b", dataset="b"),
            EntitiesRow(entity_id="c", entity_label="c", entity_type="c", dataset="c"),
            EntitiesRow(entity_id="d", entity_label="d", entity_type="d", dataset="d"),
            EntitiesRow(entity_id="e", entity_label="e", entity_type="e", dataset="e"),
        ]
        entities_2 = [
            EntitiesRow(entity_id="a", entity_label="a", entity_type="a", dataset="a"),
            EntitiesRow(entity_id="b", entity_label="b", entity_type="b", dataset="b"),
            EntitiesRow(entity_id="c", entity_label="c", entity_type="c", dataset="c"),
        ]
        links = [
            LinksRow(doc_id="a", entity_id="a", entity_type="a", dataset="a"),
            LinksRow(doc_id="b", entity_id="b", entity_type="b", dataset="b"),
            LinksRow(doc_id="c", entity_id="c", entity_type="c", dataset="c"),
        ]
        matches = [
            MatchesRow(
                left_dataset="a",
                right_dataset="b",
                left_id="a",
                right_id="b",
                match_type="exact"
            )
        ]

        if split_id == 0:
            split.set_index(entities_1)
        else:
            split.set_index(entities_2)

        sources = [
            SourceType.rdf,
            SourceType.json,
            SourceType.text
        ]
        split.set_sources(sources)

        for source_type in sources:
            source = split.sources[str(source_type)]
            meta = source.meta
            meta.set_entities(entities_1)
            if source_type == SourceType.rdf:
                meta.set_matches(matches)
            elif source_type in (SourceType.text, SourceType.json):
                meta.set_links(links)

    json1 = json.dumps(ds.model_dump(), indent=4, default=str, sort_keys=True)

    ds2 = load_dataset(ds.root)
    json2 = json.dumps(ds2.model_dump(), indent=4, default=str, sort_keys=True)

    assert json.loads(json1) == json.loads(json2)


    flat = { str(k): v for k, v in ds.overlap_matrix().items() }
    print(json.dumps(flat, indent=4, default=str, sort_keys=True))

    assert flat["('split_0', 'split_1')"]["cnt"] == 3

    # cleanup
    shutil.rmtree(tmpdir)