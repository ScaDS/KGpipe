from kgcore.api import KG
from kgcore.decorators.event import event

def test_meta():
    kg = KG(backend='memory', name='test')
    kg.create_entity(["Task"], props={"name": "test", "description": "test"})

    
    @event("Task", "create")
    def task_created(e):
        print(f"Task created: {e.id}")

    @event("Task", "update")
    def task_updated(e):
        print(f"Task updated: {e.id}")

    @event("Task", "delete")
    def task_deleted(e):
        print(f"Task deleted: {e.id}")

    task_created("e")

    es = kg.find_entities()
    for e in es:
        print(e)