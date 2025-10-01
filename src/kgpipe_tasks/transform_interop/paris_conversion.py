# """
# Paris Format Conversion

# This module provides conversion from Paris CSV format to RDF format.
# """

# import csv
# import os
# from pathlib import Path
# from typing import Dict, Any

# from kgpipe.common.models import KgTask, Data, DataFormat
# from kgpipe.common.registry import Registry


# def paris_to_rdf_conversion(inputs: Dict[str, Data], outputs: Dict[str, Data]):
#     """Convert Paris CSV output to RDF format."""
#     matches_data = inputs["matches"]
#     source_kg_data = inputs["source_kg"]
#     target_kg_data = inputs["target_kg"]
#     result_data = outputs["result"]
    
#     # Read Paris CSV matches
#     matches = []
#     with open(matches_data.path, 'r', newline='', encoding='utf-8') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             matches.append({
#                 'source_entity': row.get('source_entity', ''),
#                 'target_entity': row.get('target_entity', ''),
#                 'confidence': float(row.get('confidence', 0.0))
#             })
    
#     # Create RDF output
#     rdf_content = []
#     for match in matches:
#         source_entity = match['source_entity']
#         target_entity = match['target_entity']
#         confidence = match['confidence']
        
#         # Create owl:sameAs triple
#         rdf_triple = f'<{source_entity}> <http://www.w3.org/2002/07/owl#sameAs> <{target_entity}> .'
#         rdf_content.append(rdf_triple)
    
#     # Write RDF output
#     with open(result_data.path, 'w', encoding='utf-8') as f:
#         f.write('\n'.join(rdf_content))



# # Registry.task(
# #     input_spec={
# #         "matches": DataFormat.PARIS_CSV,
# #         "source_kg": DataFormat.RDF,
# #         "target_kg": DataFormat.RDF
# #     },
# #     output_spec={
# #         "result": DataFormat.RDF
# #     },
# #     description="Convert Paris CSV output to RDF format"
# # )
# # def paris2(i,o):
# #     paris_to_rdf_conversion(i,o)