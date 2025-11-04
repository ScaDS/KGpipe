import os
from pathlib import Path
from typing import Dict

from kgpipe.common.registry import Registry
from kgpipe.common.models import Data, DataFormat, KgPipePlan, KgStageReport
from kgpipe.generation.loaders import build_from_conf
from kgpipe.datasets.multipart_multisource import Dataset

from moviekg.datasets.pipe_out import PipeOut, StageOut
from moviekg.config import dataset, catalog


def get_format_split_path(format: str, split_id: int, dataset: Dataset) -> Path:
    if format == "seed":
        kg_seed = dataset.splits[f"split_{split_id}"].kg_seed
        if kg_seed is None:
            raise ValueError(f"KG seed is not found for split {split_id}")
        return kg_seed.root / "data.nt"
    elif format == "reference":
        kg_reference = dataset.splits[f"split_{split_id}"].kg_reference
        if kg_reference is None:
            raise ValueError(f"KG reference is not found for split {split_id}")
        return kg_reference.root / "data.nt"
    else:
        source = dataset.splits[f"split_{split_id}"].sources[format]
        if source is None:
            raise ValueError(f"Source {format} is not found for split {split_id}")
        if format == "text":
            return source.data.dir
        elif format == "json":
            return source.data.dir
        elif format == "rdf":
            return source.root / "data.nt"
        else:
            raise ValueError(f"Invalid source format: {format}")

def override_env_with_config(config: Dict[str, str]):
    if config.get("LLM_MODEL") is not None:
        os.environ["DEFAULT_LLM_MODEL_NAME"] = config["LLM_MODEL"]

    if config.get("ENTITY_MATCHING_THRESHOLD") is not None:
        os.environ["ENTITY_MATCHING_THRESHOLD"] = config["ENTITY_MATCHING_THRESHOLD"]

    if config.get("RELATION_MATCHING_THRESHOLD") is not None:
        os.environ["RELATION_MATCHING_THRESHOLD"] = config["RELATION_MATCHING_THRESHOLD"]

def run_helper(
    pipeline_name: str,
    source_format: str,
    source_id: int,
    stage_id: int,  
    output_dir: Path
) -> PipeOut:

    stage_dir = output_dir / f"stage_{stage_id}"

    pipeline_conf = catalog.root[pipeline_name]
    if pipeline_conf.config is not None:
        override_env_with_config(pipeline_conf.config)

    if stage_id == 1:
        target_data = Data(
            format=DataFormat.RDF_NTRIPLES, 
            path=get_format_split_path("seed", 0, dataset))
    else:
        target_data = Data(
            format=DataFormat.RDF_NTRIPLES, 
            path=output_dir / f"stage_{stage_id - 1}" / "result.nt")
    
    tmp_dir = stage_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_from_conf(pipeline_conf, target_data, tmp_dir.as_posix())

    stage_dir.mkdir(parents=True, exist_ok=True)

    source_format_type = None
    if source_format == "rdf":
        source_format_type = DataFormat.RDF_NTRIPLES
    elif source_format == "json":
        source_format_type = DataFormat.JSON
    elif source_format == "text":
        source_format_type = DataFormat.TEXT
    else:
        raise ValueError(f"Invalid source format: {source_format}")


    source_data = Data(
        format=source_format_type, 
        path=get_format_split_path(source_format, source_id, dataset))

    result_data = Data(
        format=DataFormat.RDF_NTRIPLES, 
        path=stage_dir / "result.nt")

    pipeline.build(source=source_data, result=result_data, stable_files=True)

    plan = KgPipePlan.model_validate(pipeline.plan)

    with open(stage_dir / "exec-plan.json", "w") as f:
        f.write(plan.model_dump_json(indent=4))

    reports = pipeline.run(stable_files_override=False)
    start_ts = reports[0].start_ts

    if os.path.exists(stage_dir / "exec-report.json"):
        print("Loading old report from", stage_dir / "exec-report.json")
        with open(stage_dir / "exec-report.json", "r") as f:
            old_report = KgStageReport.model_validate_json(f.read())
            # replace skipped task_report in new reports with old report
            old_tasks_report_dict = {report.task_name: report for report in old_report.task_reports}
            for idx, task_report in enumerate(reports):
                if task_report.status == "skipped":
                    reports[idx] = old_tasks_report_dict[task_report.task_name]

    duration = sum([report.duration for report in reports])
    status = "success" if all([report.status == "success" for report in reports]) else "failed"
    error = None if all([report.status == "success" for report in reports]) else "Some tasks failed"

    new_report = KgStageReport(
        stage_name=f"stage_{stage_id}",
        task_reports=reports,
        start_ts=start_ts,
        duration=duration,
        status=status,
        error=error
    )

    with open(stage_dir / "exec-report.json", "w") as f:
        f.write(new_report.model_dump_json(indent=4))

    return PipeOut(root=output_dir, pipeline_name=pipeline_name, stages=[
        StageOut(
            root=stage_dir,
            stage_name=f"stage_{stage_id}",
            tasks=[],
            resultKG=stage_dir / "result.nt",
            plan=KgPipePlan.model_validate(pipeline.plan),
            report=new_report
        )
    ])


def run_helper_to_json(
    pipeline_name: str,
    input_format: str,
    output_format: str
):
    pipeline_conf = catalog.root[pipeline_name]
    tasks = [Registry.get_task(task_name) for task_name in pipeline_conf.tasks]
    # each task as {"name": task_name, "input": input, "output": output}
    tasks_dict = []

    data_catalog = ["input."+input_format.replace("rdf", "nt").replace("text", "txt"), "target.nt"]
    catalog_counter = 0


    number_of_tasks = len(tasks)

    for idx, task in enumerate(tasks):
        
        # get matching data from data_catalog
        current_input = []
        for in_name, in_format in task.input_spec.items():
            for data in data_catalog:
                if data.endswith(str(in_format)) and data not in current_input:
                    current_input.append(data)
                    break

        if len(current_input) != len(task.input_spec):
            raise ValueError(f"Input is not found in data catalog"+
            f"\nDATA CATALOG: {data_catalog}"
            +f"\nCURRENT INPUT: {current_input} \nTASK INPUT: {task.input_spec}")

        # create output name
        current_output = []
        for out_name, out_format in task.output_spec.items():
            
            output = f"{catalog_counter}{out_format}"
            print(idx, number_of_tasks, output)
            if idx == number_of_tasks-1:
                output = f"result.{output_format}"

            current_output.append(output)
            data_catalog = [output] + data_catalog
            catalog_counter += 1
        

        tasks_dict.append({
            "name": task.name,
            "input": current_input,
            "output": current_output
        })

    return tasks_dict
