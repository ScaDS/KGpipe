import re

from rdflib import XSD

def validate_datatype(value: str, datatype: str):
    if datatype == str(XSD.integer):
        return value.isdigit()
    elif datatype == str(XSD.double):
        return re.match(r"^-?\d*\.?\d+$", value) is not None
    # elif datatype == "xsd:string":
    #     return True
    # elif datatype == "xsd:langString":
    #     return True
    elif datatype == str(XSD.date):
        return re.match(r"^\d{4}-\d{2}-\d{2}$", value) is not None
    elif datatype == str(XSD.gYear):
        return re.match(r"^\d{4}$", value) is not None
    elif datatype == str(XSD.gMonth):
        return re.match(r"^\d{2}$", value) is not None
    elif datatype == str(XSD.gDay):
        return re.match(r"^\d{2}$", value) is not None
    elif datatype == str(XSD.gYearMonth):
        return re.match(r"^\d{4}-\d{2}$", value) is not None
    elif datatype == str(XSD.gMonthDay):
        return re.match(r"^\d{2}-\d{2}$", value) is not None
    elif datatype == str(XSD.gDay):
        return re.match(r"^\d{2}$", value) is not None
    elif datatype == str(XSD.gMonth):
        return re.match(r"^\d{2}$", value) is not None
    else:
        return True