from docling.document_converter import DocumentConverter

source = "/home/marvin/Documents/rw/Suchanek2011Paris.pdf"
converter = DocumentConverter()
result = converter.convert(source)
doc = result.document

# Export to markdown
print(doc.export_to_markdown())