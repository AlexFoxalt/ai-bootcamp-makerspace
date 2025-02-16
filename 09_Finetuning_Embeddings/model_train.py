import json

import wandb
from sentence_transformers import InputExample
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from torch.utils.data import DataLoader


with open("cities_test_dataset.jsonl") as f:
    test_dataset = json.load(f)

with open("cities_train_dataset.jsonl") as f:
    train_dataset = json.load(f)

with open("cities_val_dataset.jsonl") as f:
    val_dataset = json.load(f)

embedding_model_id = "Snowflake/snowflake-arctic-embed-l"
embedding_model = SentenceTransformer(embedding_model_id)

BATCH_SIZE = 64

corpus = train_dataset['corpus']
queries = train_dataset['questions']
relevant_docs = train_dataset['relevant_contexts']

examples = []
for query_id, query in queries.items():
    doc_id = relevant_docs[query_id][0]
    text = corpus[doc_id]
    example = InputExample(texts=[query, text])
    examples.append(example)

loader = DataLoader(
    examples, batch_size=BATCH_SIZE
)

matryoshka_dimensions = [768, 512, 256, 128, 64]
inner_train_loss = MultipleNegativesRankingLoss(embedding_model)
train_loss = MatryoshkaLoss(
    embedding_model, inner_train_loss, matryoshka_dims=matryoshka_dimensions
)

corpus = val_dataset['corpus']
queries = val_dataset['questions']
relevant_docs = val_dataset['relevant_contexts']

evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs)

EPOCHS = 10

wandb.init(mode="disabled")

warmup_steps = int(len(loader) * EPOCHS * 0.1)

embedding_model.fit(
    train_objectives=[(loader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path='cities_optimized_embedding_model',
    show_progress_bar=True,
    evaluator=evaluator,
    evaluation_steps=50
)
