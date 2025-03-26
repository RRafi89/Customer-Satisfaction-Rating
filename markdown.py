import nbformat

# Load the Jupyter Notebook file
notebook_path = "simple.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    notebook_data = nbformat.read(f, as_version=4)

# Extract code cells
code_cells = [cell["source"] for cell in notebook_data.cells if cell.cell_type == "code"]

# Combine extracted code
extracted_code = "\n\n".join(code_cells)
print(extracted_code)
with open('text.txt', 'wt+', encoding="utf-8") as f:
    f.write(extracted_code)
