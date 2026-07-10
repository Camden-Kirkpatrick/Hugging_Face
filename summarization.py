"""
Summarize a passage with facebook/bart-large-cnn (an encoder-decoder / seq2seq model).

Uses the Hugging Face `summarization` pipeline to compress a block of text into a
short summary. Beam search is used to produce multiple candidate summaries and
return the top-ranked ones.

Requires transformers < v5  (the `summarization` pipeline task was removed in v5.x).
"""

from transformers import pipeline

pipe = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0,
)

text  = """
The mechanical clock, invented in medieval Europe, did more than
help people tell time — it reshaped how they thought about the world.
Before clocks, time followed natural rhythms: the sun, the seasons,
the ringing of monastery bells. An "hour" in summer was literally
longer than one in winter, and time was experienced rather than
measured. When towns installed clock towers in the fourteenth century,
time became uniform and public. Markets could open at set times,
workers could be scheduled, and wages could be tied to hours rather
than tasks.
Some historians argue that the clock, not the steam engine, was the
true key machine of the industrial age, because it taught people to
treat time as a resource to be divided, saved, and spent. The cultural
effects ran deep: punctuality became a virtue and lateness a failing,
and philosophers began to imagine the universe itself as a kind of
clockwork. But something was lost too — the older, looser sense of
time allowed a closer bond with the natural world. Today, surrounded
by clocks on every screen, many people feel ruled by time rather than
served by it.
"""

# Run beam search with 4 beams, then return the 2 highest-scoring summaries.
# Rule: num_return_sequences must be <= num_beams. The results are ranked
# best-first and tend to be similar.
summaries = pipe(
    text,
    num_beams=4,
    num_return_sequences=2,
)

for s in summaries:
    print(s["summary_text"], end="\n\n")