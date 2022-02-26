from docarray import Document, DocumentArray
from jina import Executor

encoder = Executor.load_config(
    'TransformerTorchEncoder/config.yml', uses_with={'device': 'cuda'})
indexer = Executor.load_config(
    'SimpleIndexer/config.yml', uses_metas={'workspace': './workspace'})

da = DocumentArray([
    Document(text="abs() Returns the absolute value of a number"),
    Document(text="chr() Returns a character from the specified Unicode code."),
    Document(text="set() Returns a new set object"),
    Document(text="sum() Sums the items of an iterator"),
    Document(text="tuple() Returns a tuple")])

encoder.encode(docs=da)
indexer.index(docs=da)

q_da = DocumentArray([Document(text='convert number to modulus')])
encoder.encode(docs=q_da)
indexer.search(docs=q_da)

for rank, m in enumerate(q_da[0].matches):
    print(f'rank: {rank}, text: {m.text}')
