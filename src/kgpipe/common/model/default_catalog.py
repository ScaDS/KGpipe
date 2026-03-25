

# TODO impl later for typed api
class TaskCategory():pass

class EntityResolution(TaskCategory): pass
class EntityMatching(EntityResolution): pass
class Fusion(EntityResolution): pass
class InformationExtraction(TaskCategory): pass
class EntityLinking(InformationExtraction): pass
class RelationExtraction(InformationExtraction): pass
class RelationLinking(InformationExtraction): pass
class DataMapping(TaskCategory): pass