from kgpipe_tasks.common import Ontology, OntologyUtil, OwlPropertyType


def generate_ontology_snippet(ontology: Ontology):
    """
    Generate a snippet of the ontology.
    like:
    Movie           A motion picture work.
    Person          A human being.
    director        ObjectProperty(Person->Movie)   The director of the movie.
    actor           ObjectProperty(Person->Movie)   The actor starring in the movie.
    writer          ObjectProperty(Person->Movie)   The writer of the movie.
    producer        ObjectProperty(Person->Movie)   The producer of the movie.
    genre           ObjectProperty(Movie->Genre)    The genre of the movie.
    """
    snippet = ""

    for class_ in ontology.classes:
        snippet += f"{class_.label.replace(' ', '_')}\t{class_.description}\n"

    snippet += "\n"

    for property_ in ontology.properties:
        property_type = "ObjectProperty" if property_.type == OwlPropertyType.ObjectProperty else "DatatypeProperty"

        range_label = property_.range.label.replace(' ', '_') if property_.range else "Literal"
        domain_label = property_.domain.label.replace(' ', '_') if property_.domain else "UNDEFINED"

        if property_type == "ObjectProperty":
            snippet += f"{property_.label.replace(' ', '_')}\t{property_type}({domain_label}->{range_label})\t{property_.description}\n"
        else:
            snippet += f"{property_.label.replace(' ', '_')}\t{property_type}({domain_label}->{range_label})\t{property_.description}\n"

    return snippet

def generate_ontology_snippet_v2(ontology: Ontology):
    """
    Generate a snippet of the ontology.
    Movie {
        id: string
        title: string
        genres: list[string]
        actors: list[Person]
        directors: list[Person]
        runtime: int
        releaseYear: int
    }
    
    Person {
        id: string
        name: string
        birthYear: int
    }
    """

    snippet = ""
    for class_ in ontology.classes:
        snippet += f"{class_.uri}\t\"{class_.label}\"\t{class_.description}\n"
        for property_ in ontology.properties:
            if property_.domain == class_:
                snippet += f"{property_.uri}\t\"{property_.label}\"\t{property_.type}\t{property_.description}\n"
    return snippet


def generate_ontology_snippet_v3(ontology: Ontology):
    """
    Generate a snippet of the ontology.
    like:
    Movie           A motion picture work.
    Person          A human being.
    director        ObjectProperty(Person->Movie)   The director of the movie.
    actor           ObjectProperty(Person->Movie)   The actor starring in the movie.
    writer          ObjectProperty(Person->Movie)   The writer of the movie.
    producer        ObjectProperty(Person->Movie)   The producer of the movie.
    genre           ObjectProperty(Movie->Genre)    The genre of the movie.
    """
    snippet = ""

    for class_ in ontology.classes:
        snippet += f"{class_.uri}\t\"{class_.label}\"\t{class_.description}\n"

    snippet += "\n\n"

    for property_ in ontology.properties:
        property_type = "ObjectProperty" if property_.type == OwlPropertyType.ObjectProperty else "DatatypeProperty"

        range_label = property_.range.label.replace(' ', '_') if property_.range else "Literal"
        domain_label = property_.domain.label.replace(' ', '_') if property_.domain else "UNDEFINED"

        if property_type == "ObjectProperty":
            snippet += f"{property_.uri}\t{property_type}({domain_label}->{range_label})\t{property_.description}\n"
        else:
            snippet += f"{property_.uri}\t{property_type}({domain_label}->{range_label})\t{property_.description}\n"

    return snippet

if __name__ == "__main__":
    ontology = OntologyUtil.load_ontology_from_file("/home/marvin/project/data/ontology.ttl")
    print(generate_ontology_snippet(ontology))

