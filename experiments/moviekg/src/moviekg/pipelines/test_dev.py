# from kgpipe.generation.loaders import load_pipeline_catalog, build_from_conf
# from pathlib import Path

# from pyodibel.datasets.mp_mf.multipart_multisource import load_dataset, Dataset
# from kgpipe.common import Data, DataFormat

# catalog = load_pipeline_catalog(Path("pipeline.conf"))
# dataset_small = load_dataset(Path("/home/marvin/project/data/old/acquisiton/film100_bundle"))

# def test_pipeline():

#     output_dir = Path("./out_dev/")

#     target_data = Data(
#         format=DataFormat.RDF_NTRIPLES, 
#         path=output_dir / f"stage_{0}" / "result.nt")
    
#     tmp_dir = output_dir / "tmp"
#     tmp_dir.mkdir(parents=True, exist_ok=True)

#     print(catalog.root["test_pipeline"].config)
#     # pipeline = build_from_conf(catalog.root["test_pipeline"], target_data, str(tmp_dir))
#     # pipeline.run()

# if __name__ == "__main__":
#     test_pipeline()
