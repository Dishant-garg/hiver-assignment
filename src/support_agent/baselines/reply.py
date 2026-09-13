"""Reply baselines. Trivial: one canned reply. Simple: the historical brand reply of the most
similar past message, returned verbatim (no LLM)."""
from support_agent.retrieve import Retriever

CANNED = "Sorry to hear that! Can you send us a DM with more details so we can take a closer look?"


class CannedReply:
    def reply(self, message: str) -> str:
        return CANNED


class NearestNeighbourReply:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def reply(self, message: str) -> str:
        ev = self.retriever.top_k(message, 1)
        return ev[0].brand_reply if ev else CANNED
