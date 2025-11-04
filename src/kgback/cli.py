import click, random
from kgback.config import load_config
from kgback.back import get_backend

@click.group()
def back():
    pass


@click.command()
def serv():
    print("Initialized")

@click.command()
def list():
    print("Listing")

@click.group()
def op():
    pass

@click.command()
def config():
    config = load_config()
    print(config)

@click.command()
def clear():
    # print random integer that needs to be typed in to clear the database
    token = random.randint(1, 1000000)
    print(f"Enter the token to clear the database: {token}")
    input_token = input()
    if input_token == str(token):
        config = load_config()
        backend = get_backend(config)
        backend.clear_database()
    else:
        print("Invalid token")
        exit(1)

@click.command()
@click.option("--subject", "-s", type=str)
def get(subject):
    import kgback.postgres_back
    pb = kgback.postgres_back.PostgresECB()
    if subject:
        triples = pb.get_triples_by_subject(subject)
    else:
        triples = pb.get_all_triples()
    for triple in triples:
        print(triple)

@click.command()
def show():
    config = load_config()
    backend = get_backend(config)
    for triple in backend.get_all_triples():
        print(triple)

# def parse_triples(triples: list[str]):
#     from kgcore.model.graph import Triple, Node
#     triples = []
#     for triple in triples:
#         split = triple.split(" ")
#         triples.append(Triple(Node(split[0]), Node(split[1]), Node(split[2])))
#     return triples

@click.command()
@click.argument("triples", nargs=-1)
@click.option("--file", "-f", type=str)
def add(triples, file):
    from kgcore.model.graph import Triple, Node, Literal
    config = load_config()
    backend = get_backend(config)
    parsed_triples = []
    if file:
        with open(file, "r") as f:
            triples = f.readlines()
    for triple in triples:
        split = triple.split(" ")
        parsed_triples.append(Triple(subject=Node(uri=split[0]), predicate=Node(uri=split[1]), object=Node(uri=split[2])))
    backend.add_triples(parsed_triples)


op.add_command(show)
op.add_command(add)
op.add_command(clear)
op.add_command(get)
back.add_command(op)

back.add_command(serv)
back.add_command(list)
back.add_command(config)

if __name__ == "__main__":
    back()