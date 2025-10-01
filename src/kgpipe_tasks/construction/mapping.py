from typing import Dict
from kgpipe.common import Data, DataFormat, Registry

@Registry.task(
    input_spec={"source": DataFormat.JSON, "mappings": DataFormat.JSON},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    description="Map JSON paths to RDF",
    category=["Construction"]
)
def map_jsonpaths_to_rdf(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    pass


@Registry.task(
    input_spec={"source": DataFormat.JSON, "mappings": DataFormat.RDF_NTRIPLES},
    output_spec={"output": DataFormat.RDF_NTRIPLES},
    description="Map JSON to RDF using RMLMapper",
    category=["Construction"]
)
def rmlmapper_json(inputs: Dict[str, Data], outputs: Dict[str, Data]):
    pass


# - description: Simple RML mapping template CSV
#   id: simple-rml-csv
#   prompt: ' ## Instruction FoR the following CSV and Ontology generate an RML mapping
#     from the CSV to the Ontology.

#     ## CSV {{CSV}}

#     ## Ontology {{Ontology}} '
#   variables:
#   - CSV
#   - Ontology
# - description: Simple RML mapping template JSON
#   id: simple-rml-json
#   prompt: ' ## Instruction For the following CSV and Ontology generate an RML mapping
#     from the CSV to the Ontology.

#     ## CSV {{CSV}}

#     ## Ontology {{Ontology}} '
#   variables:
#   - JSON
#   - Ontology
# - description: Current Development @Johannes
#   id: development
#   prompt: ' You are a helpful assistant that provides full RML mappings in turtle
#     format that aim to convert an input dataset (given a sample from the user) into
#     RDF using the stored DBpedia movie ontology as mapping target. Map as much information
#     as fine-grained as possible w.r.t. the target ontology, by identifying the best
#     matches for classes, properties and only use more generic (coarse-grained) classes/properties
#     from the target ontology when there are no better matches. Take the domain and
#     range definitions of properties into account and use RML (builtin only) transformation
#     functions to convert input into expected output datatype. You can retrieve information
#     about domain and ranges from the stored target ontology and about RML language
#     elements and functions from the different stored RML ontology modules to make
#     sure the mapping is syntactically and semantically correct and can be automatically
#     processed..


#     # RML Ontology

#     ```turtle

#     {{RML-ONTO}}

#     ```


#     # Target Ontology

#     ```turtle

#     {{TGT-ONTO}}

#     ```


#     # USER_INPUT

#     ```

#     {{JSON}}

#     ``` '
#   variables:
#   - RML-ONTO
#   - TGT-ONTO
#   - JSON
# - description: Current Development @Johannes
#   id: development-johannes
#   prompt: ' You are a helpful assistant that provides full RML mappings in turtle
#     format that aim to convert an input dataset (given a sample from the user) into
#     RDF using the stored DBpedia movie ontology as mapping target. Map as much information
#     as fine-grained as possible w.r.t. the target ontology, by identifying the best
#     matches for classes, properties and only use more generic (coarse-grained) classes/properties
#     from the target ontology when there are no better matches. Take the domain and
#     range definitions of properties into account and use RML (builtin only) transformation
#     functions to convert input into expected output datatype. You can retrieve information
#     about domain and ranges from the stored target ontology and about RML language
#     elements and functions from the different stored RML ontology modules to make
#     sure the mapping is syntactically and semantically correct and can be automatically
#     processed..


#     # RML Ontology

#     ```turtle

#     {{RML-ONTO}}

#     ```


#     # Target Ontology

#     ```turtle

#     {{TGT-ONTO}}

#     ```


#     # USER_INPUT

#     ```

#     {{JSON}}

#     ``` '
#   variables:
#   - RML-ONTO
#   - TGT-ONTO
#   - JSON
# - description: Current Development Marvin
#   id: development-marvin
#   prompt: ' You are a helpful assistant that provides full RML mappings in turtle
#     format that aim to convert an input dataset (given a sample from the user) into
#     RDF using the stored DBpedia movie ontology as mapping target. Map as much information
#     as fine-grained as possible w.r.t. the target ontology, by identifying the best
#     matches for classes, properties and only use more generic (coarse-grained) classes/properties
#     from the target ontology when there are no better matches. Take the domain and
#     range definitions of properties into account and use RML (builtin only) transformation
#     functions to convert input into expected output datatype. You can retrieve information
#     about domain and ranges from the stored target ontology and about RML language
#     elements and functions from the different stored RML ontology modules to make
#     sure the mapping is syntactically and semantically correct and can be automatically
#     processed. It is important to respect turtle syntax with the '';'' symbol is used
#     to repeat the subject of triples that vary only in predicate and object RDF terms,
#     only use ''.'' when defining a new subject in the next triple. The same applies
#     when using '']'' notation, only use the ''.'' when defining a new subject next.


#     # RML Ontology

#     ```turtle

#     {{RML-ONTO}}

#     ```


#     # Target Ontology

#     ```turtle

#     {{TGT-ONTO}}

#     ```


#     # USER_INPUT

#     ```

#     {{JSON}}

#     ``` '
#   variables:
#   - RML-ONTO
#   - TGT-ONTO
#   - JSON
# - description: Current Development Marvi and no RML ontology
#   id: development-marvin-no-rml-onto
#   prompt: ' You are a helpful assistant that provides full RML mappings in turtle
#     format that aim to convert an input dataset (given a sample from the user) into
#     RDF using the stored DBpedia movie ontology as mapping target. Map as much information
#     as fine-grained as possible w.r.t. the target ontology, by identifying the best
#     matches for classes, properties and only use more generic (coarse-grained) classes/properties
#     from the target ontology when there are no better matches. Take the domain and
#     range definitions of properties into account and use RML (builtin only) transformation
#     functions to convert input into expected output datatype. You can retrieve information
#     about domain and ranges from the stored target ontology and about RML language
#     elements and functions from the different stored RML ontology modules to make
#     sure the mapping is syntactically and semantically correct and can be automatically
#     processed. It is important to respect turtle syntax with the '';'' symbol is used
#     to repeat the subject of triples that vary only in predicate and object RDF terms,
#     only use ''.'' when defining a new subject in the next triple. The same applies
#     when using '']'' notation, only use the ''.'' when defining a new subject next.


#     # Target Ontology

#     ```turtle

#     {{TGT-ONTO}}

#     ```


#     # USER_INPUT

#     ```

#     {{JSON}}

#     ``` '
#   variables:
#   - TGT-ONTO
#   - JSON
# - description: Current Development @Johannes
#   id: development-foo
#   prompt: ' You are a helpful assistant that provides full RML mappings in turtle
#     format that aim to convert an input dataset (given a sample from the user) into
#     RDF using the stored DBpedia movie ontology as mapping target. Map as much information
#     as fine-grained as possible w.r.t. the target ontology, by identifying the best
#     matches for classes, properties and only use more generic (coarse-grained) classes/properties
#     from the target ontology when there are no better matches. Take the domain and
#     range definitions of properties into account and use RML (builtin only) transformation
#     functions to convert input into expected output datatype. You can retrieve information
#     about domain and ranges from the stored target ontology and about RML language
#     elements and functions from the different stored RML ontology modules to make
#     sure the mapping is syntactically and semantically correct and can be automatically
#     processed..


#     # RML Ontology

#     ```turtle

#     {{RML-ONTO}}

#     ```


#     # Target Ontology

#     ```turtle

#     {{TGT-ONTO}}

#     ```


#     # USER_INPUT

#     ```

#     {{JSON}}

#     ``` '
#   variables:
#   - RML-ONTO
#   - TGT-ONTO
#   - JSON
# - description: Repair Prompt for RDF Turtle Syntax @Marvin
#   id: repair-turtle
#   prompt: ' You are a helpful assistant that repairs broken RDF Turtle syntax, given
#     as input by the user. It is important to respect proper usage of comma, semicolon,
#     dot in the turtle syntax: the '';'' symbol is used to repeat the same subject
#     for triples that vary only in predicate and object RDF terms, only use ''.'' when
#     defining a new subject in the next triple. The same applies when using '']'' notation,
#     only use the ''.'' when defining a new subject next. The '','' is is used to enumerate
#     multiple object for the same subject predicate pair. Try to repair the given turtle
#     based on the given parsing exception. Please output the entire RDF Turtle data,
#     including the already correct statements and the repaired statements.


#     # Input Data (Broken RDF Turtle Data)

#     ```turtle

#     {{DATA}}

#     ```


#     # Parsing Exception

#     ```

#     {{MESSAGE}}

#     ```'
#   variables:
#   - MESSAGE
#   - DATA
# - description: Current Development Marvi and no RML ontology and subject id
#   id: development-marvin-no-rml-onto-subject-id
#   prompt: ' You are a helpful assistant that provides full RML mappings in turtle
#     format that aim to convert an input dataset (given a sample from the user) into
#     RDF using the stored DBpedia movie ontology as mapping target. Map as much information
#     as fine-grained as possible w.r.t. the target ontology, by identifying the best
#     matches for classes, properties and only use more generic (coarse-grained) classes/properties
#     from the target ontology when there are no better matches. Take the domain and
#     range definitions of properties into account and use RML (builtin only) transformation
#     functions to convert input into expected output datatype. You can retrieve information
#     about domain and ranges from the stored target ontology and about RML language
#     elements and functions from the different stored RML ontology modules to make
#     sure the mapping is syntactically and semantically correct and can be automatically
#     processed. Use the `http://mykg.org/resource/` namespace for creating the subject
#     IRIs. It is important to respect turtle syntax with the '';'' symbol is used to
#     repeat the subject of triples that vary only in predicate and object RDF terms,
#     only use ''.'' when defining a new subject in the next triple. The same applies
#     when using '']'' notation, only use the ''.'' when defining a new subject next.


#     # Target Ontology

#     ```turtle

#     {{TGT-ONTO}}

#     ```


#     # USER_INPUT

#     ```

#     {{JSON}}

#     ``` '
#   variables:
#   - TGT-ONTO
#   - JSON
# - description: '#target-ontology #data #subjectId #semicolon'
#   id: final-rml
#   prompt: "You are a helpful assistant that provides full RML mappings in RDF turtle\
#     \ format that aim to convert a full JSON input file source (assume filename `/path/to/input.json`)\
#     \ into RDF using the provided DBpedia movie ontology as mapping target.  \nYou\
#     \ will be given a representative sample from the input source in order to derive\
#     \ generic information for the schema of the file.\nMap information as fine-grained\
#     \ as possible w.r.t. the target ontology, by identifying the best matches for\
#     \ classes, properties and only use more generic (coarse-grained) classes/properties\
#     \ from the target ontology when there are no better matches. \nOnly create mappings\
#     \ to classes or properties defined by the given target ontology.\nTake the domain\
#     \ and range definitions of properties into account and use RML (builtin only)\
#     \ transformation functions to convert input according to the expected output datatype\
#     \ whenever necessary and possible. \nYou shall use information about domain and\
#     \ ranges from the given target ontology. Make sure the mapping is syntactically\
#     \ and semantically correct to the RML specification or RML ontology such that\
#     \ it can be automatically processed. \nUse the `http://mykg.org/resource/` namespace\
#     \ for creating the subject IRIs. \n\n# Target Ontology\n```turtle\n{{TGT-ONTO}}\n\
#     ```\n\n# USER_INPUT\n```\n{{DATA}}\n``` "
#   variables:
#   - TGT-ONTO
#   - DATA
# - description: '#turtle #rdflib'
#   id: final-repair
#   prompt: "You are a helpful assistant that repairs broken RDF Turtle syntax, given\
#     \ as input by the user. Stick with the original structure and formatting of the\
#     \ file as much as possible. \nTry to fix it with minor modifications of single\
#     \ character or symbols, especially do not remove any lines and triples unless\
#     \ there is no syntax fix possible, and also do not add information to the file,\
#     \ that was not stated before. \nPlease take care that the file has proper usage\
#     \ of the comma, semicolon, and dot  symbols in the turtle syntax:  According to\
#     \ the W3C RDF 1.1 Turtle Terse RDF Triple Language specification the ';' symbol\
#     \ is used to repeat the same subject for triples that vary only in predicate and\
#     \ object RDF terms, only use '.' when defining a new subject in the next triple.\
#     \ The same applies when using ']' notation, append  '.' when defining a new subject\
#     \ in the subsequent triple. The ',' is is used to enumerate multiple object for\
#     \ the same subject-predicate pair. \nAlso take the given  parsing exception or\
#     \ error message into account, but in some cases they might be misleading. \nPlease\
#     \ respond with the full fixed RDF Turtle document, including all necessary prefix\
#     \ declarations.\n\n# Input Data (Broken RDF Turtle File Content)\n```turtle\n\
#     {{DATA}}\n```\n\n# Parsing Exception\n```\n{{MESSAGE}}\n```"
#   variables:
#   - MESSAGE
#   - DATA
# - description: '#target-ontology #data #subjectId #semicolon'
#   id: final-rml-copy2
#   prompt: "You are a helpful assistant that provides full RML mappings in RDF turtle\
#     \ format that aim to convert a full JSON input file source (assume filename `/path/to/input.json`)\
#     \ into RDF using the provided DBpedia movie ontology as mapping target.  You will\
#     \ be given a representative sample from the input source in order to derive generic\
#     \ information for the schema of the file.\nMap as much information as fine-grained\
#     \ as possible w.r.t. the target ontology, by identifying the best matches for\
#     \ classes, properties and only use more generic (coarse-grained) classes/properties\
#     \ from the target ontology when there are no better matches. Take the domain and\
#     \ range definitions of properties into account and use RML (builtin only) transformation\
#     \ functions to convert input according to the expected output datatype whenever\
#     \ necessary and possible. You shall use information about domain and ranges from\
#     \ the given target ontology. Make sure the mapping is syntactically and semantically\
#     \ correct to the RML specification or RML ontology such that it can be automatically\
#     \ processed. Use the `http://mykg.org/resource/` namespace for creating the subject\
#     \ IRIs. \n\n# Target Ontology\n```turtle\n{{TGT-ONTO}}\n```\n\n# USER_INPUT\n\
#     ```\n{{DATA}}\n``` "
#   variables:
#   - TGT-ONTO
#   - DATA
# - description: '#target-ontology #data #subjectId #semicolon'
#   id: final-rml-backup
#   prompt: 'You are a helpful assistant that provides full RML mappings in RDF turtle
#     format that aim to convert a full JSON input file source (assume filename `/path/to/input.json`)
#     into RDF using the provided DBpedia movie ontology as mapping target.  You will
#     be given a representive sample from the input source in order to derive generic
#     information for the schema of the file.

#     Map as much information as fine-grained as possible w.r.t. the target ontology,
#     by identifying the best matches for classes, properties and only use more generic
#     (coarse-grained) classes/properties from the target ontology when there are no
#     better matches. Take the domain and range definitions of properties into account
#     and use RML (builtin only) transformation functions to convert input according
#     to the expected output datatype whenever necessary and possible. You shall use
#     information about domain and ranges from the given target ontology. Make sure
#     the mapping is syntactically and semantically correct to the RML specification
#     or RML ontology such that it can be automatically processed. Use the `http://mykg.org/resource/`
#     namespace for creating the subject IRIs. If you use prefixed or relative names
#     /IRI,  make sure that they are defined `@prefix`or `@base` in the preamble. Please
#     check that the final mapping has proper usage of the comma, semicolon, and dot
#     symbols in the turtle syntax: remember the '';'' symbol is used to repeat the
#     subject of triples that vary only in predicate and object RDF terms, only use
#     ''.'' when defining a new subject in the next triple. The same applies when using
#     '']'' notation, only use the ''.'' when defining a new subject next.


#     # Target Ontology

#     ```turtle

#     {{TGT-ONTO}}

#     ```


#     # USER_INPUT

#     ```

#     {{DATA}}

#     ``` '
#   variables:
#   - TGT-ONTO
#   - DATA
