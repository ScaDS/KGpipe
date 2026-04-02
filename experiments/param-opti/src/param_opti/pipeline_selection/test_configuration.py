from random import random, seed, sample
from typing import List



def entity_matching_a() -> List[str]:
    seed(42)
    # select 5 positive values and 5 negative values
    positive_values=["+A", "+B", "+C", "+D", "+E", "+F", "+G", "+H", "+I", "+J", "+K", "+L", "+M", "+N", "+O", "+P", "+Q", "+R", "+S", "+T", "+U", "+V", "+W", "+X", "+Y", "+Z"]
    negative_values=["-A", "-B", "-C", "-D", "-E", "-F", "-G", "-H", "-I", "-J", "-K", "-L", "-M", "-N", "-O", "-P", "-Q", "-R", "-S", "-T", "-U", "-V", "-W", "-X", "-Y", "-Z"]
    positive_values = sample(positive_values, 5)
    negative_values = sample(negative_values, 5)
    return positive_values + negative_values

def schmea_matching_a(): pass

def test_selecting_pipelines(): pass


def test_run():

    values = entity_matching_a()
    print(values)
    values2 = entity_matching_a()
    print(values2)
    # print(values == values2)