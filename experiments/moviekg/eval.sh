kgpipe eval -c metric_config.yaml \
	-m ReferenceTripleAlignmentMetricSoftEV \
	-m entity_count \
	-m incorrect_relation_direction \
	-m incorrect_relation_cardinality \
	-m incorrect_relation_range \
	-m incorrect_relation_domain \
	-m incorrect_datatype \
	-m incorrect_datatype_format \
	data/out/small/rdf_a/stage_3/result.nt
