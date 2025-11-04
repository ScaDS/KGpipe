from kgcore.model.ontology import Ontology

def test_ontology():
    onto = Ontology()
    Person = "http://ex/Person"
    Employee = "http://ex/Employee"
    Org = "http://ex/Organization"
    worksFor = "http://ex/worksFor"

    onto.add_class(Person, label="Person")
    onto.add_class(Employee, label="Employee", parents=[Person])
    onto.add_class(Org, label="Organization")
    onto.add_predicate(worksFor, label="worksFor", domain=[Person], range=[Org])

    ok, reason = onto.validate_triple(Employee, worksFor, Org)
    assert ok, reason
