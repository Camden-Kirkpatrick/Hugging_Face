"""
Extractive question answering with a Hugging Face pipeline.

The model reads a context document and answers questions by highlighting the
span of text that contains the answer. It doesn't generate new words — the
answer is always copied directly from the document.

Model: distilbert-base-cased-distilled-squad (trained on the SQuAD QA dataset).
"""

from transformers import pipeline

# The document the model will search for answers
document_text = """
Welcome to Northwind Technologies. This handbook summarizes the benefits and
policies available to all full-time employees.

Employees accrue 20 days of paid vacation per year, in addition to 10 public
holidays. New hires are eligible for benefits after a 30-day probation period.

Northwind offers 5 volunteer days annually, allowing staff to support causes
they care about during regular working hours. The company also matches
charitable donations up to $1,000 per employee each year.

The office operates on a hybrid schedule: employees are expected on-site three
days per week, with Mondays and Fridays available as remote days. Core working
hours are 10 a.m. to 4 p.m.

Health insurance covers medical, dental, and vision, and the company contributes
6% of salary to each employee's retirement plan.
"""

# Load the question-answering pipeline
qa_pipeline = pipeline(
    task="question-answering",
    model="distilbert-base-cased-distilled-squad",
    device=0,
)

question = "How many volunteer days are offered annually?"

# Get the answer from the QA pipeline
result = qa_pipeline(question=question, context=document_text)
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['score']:.2%}")