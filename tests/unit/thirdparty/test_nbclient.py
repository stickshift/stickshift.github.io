from nbclient import NotebookClient
from nbformat import NotebookNode
from nbformat import v4 as nbf


def test_cell_events(kernel_name: str):
    #
    # Givens
    #

    # I created an empty notebook
    nb = nbf.new_notebook()

    # I add a markdown cell
    cell = nbf.new_markdown_cell("Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
    nb.cells.append(cell)

    # I add a code cell
    cell = nbf.new_code_cell("print('Hello, World!')")
    nb.cells.append(cell)

    # I collect cell events
    started, executed = set(), set()

    def on_cell_start(cell: NotebookNode, cell_index: int, **_):
        started.add(cell_index)

    def on_cell_execute(cell: NotebookNode, cell_index: int, **_):
        executed.add(cell_index)

    # I created a notebook client from nb
    client = NotebookClient(nb, kernel_name=kernel_name, on_cell_start=on_cell_start, on_cell_execute=on_cell_execute)

    #
    # Whens
    #

    # I execute cell
    client.execute()

    #
    # Thens
    #

    # Both cells should have fired started event
    assert 0 in started
    assert 1 in started

    # Only cell 1 should have fired executed event
    assert 0 not in executed
    assert 1 in executed
