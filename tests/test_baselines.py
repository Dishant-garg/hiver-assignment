import pandas as pd

from support_agent.baselines.escalation import AlwaysEscalate, RulesOnlyEscalation
from support_agent.baselines.intent import MajorityIntent, TfidfLogRegIntent
from support_agent.baselines.reply import CannedReply, NearestNeighbourReply
from support_agent.retrieve import Retriever


def test_majority_predicts_most_common():
    m = MajorityIntent().fit(["a", "b", "b"])
    assert m.predict(["x", "y"]) == ["b", "b"]


def test_tfidf_logreg_learns_separable_labels():
    texts = ["songs skipping", "music skipping again", "charged twice", "double charge on card"] * 5
    labels = ["play", "play", "bill", "bill"] * 5
    m = TfidfLogRegIntent().fit(texts, labels)
    assert m.predict(["skipping songs", "charged again"]) == ["play", "bill"]


def test_reply_baselines():
    corpus = pd.DataFrame({"pair_id": [0], "customer_text": ["charged twice"], "brand_reply": ["Let's check that charge."]})
    assert isinstance(CannedReply().reply("anything"), str)
    assert NearestNeighbourReply(Retriever(corpus)).reply("I was charged twice") == "Let's check that charge."


def test_escalation_baselines():
    assert AlwaysEscalate().decide("hi")[0] is True
    assert RulesOnlyEscalation().decide("my playlist vanished")[0] is False
    assert RulesOnlyEscalation().decide("I will sue you")[0] is True
