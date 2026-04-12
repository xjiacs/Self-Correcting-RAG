
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import math, random
from transformers import LogitsProcessorList
from .generator import QwenLocal
from .reasoning.evidence import extract_top_evidence
from .reasoning.verifier import NLIVerifier

class MCTSGuidedGenerator:

    def __init__(self,
                 base_model_path: str,
                 nli_model_name: str = "roberta-large-mnli",
                 simulations: int = 24,
                 branching: int = 3,
                 max_depth: int = 3,
                 entail_w: float = 1.0,
                 contra_w: float = -2.0,
                 neutral_w: float = -0.2,
                 temperature: float = 0.7,
                 top_p: float = 0.9):
        self.gen = QwenLocal(model_path=base_model_path, max_new_tokens=256)
        self.nli = NLIVerifier(model_name=nli_model_name)
        self.simulations = simulations
        self.branching = branching
        self.max_depth = max_depth
        self.entail_w = entail_w
        self.contra_w = contra_w
        self.neutral_w = neutral_w
        self.temperature = temperature
        self.top_p = top_p

    def _evidence_reward(self, question: str, answer: str, docs: List[Dict[str, Any]], emb_fn) -> float:
        # Build evidence sentence pool
        evs = extract_top_evidence(question, docs, emb_fn, per_doc=3, total=8)
        if not evs:
            return 0.0
        # Aggregate entailment scores over top evidence vs answer sentences
        # We compare each evidence sentence to (answer) as hypothesis for simplicity.
        score = 0.0
        for e in evs:
            sc = self.nli.score(e["sent"], answer)
            score += (self.entail_w * sc["entail"] + self.neutral_w * sc["neutral"] + self.contra_w * sc["contradict"])
        return score / max(1, len(evs))

    def _generate_once(self, q: str, docs: List[Dict[str, Any]], logits_processors: Optional[LogitsProcessorList]=None) -> str:
        return self.gen.generate(q, docs, temperature=self.temperature, top_p=self.top_p, logits_processors=logits_processors)

    def generate_mcts(self, question: str, docs: List[Dict[str, Any]], retriever, emb_fn) -> str:

        class Node:
            __slots__ = ("parent","children","n","q","docs","answer","is_answer","ucb_c","prior")
            def __init__(self, parent, docs, is_answer: bool=False, prior: float=0.0, ucb_c: float=1.4):
                self.parent = parent
                self.children: List[Node] = []
                self.n = 0
                self.q = 0.0
                self.docs = docs
                self.answer: Optional[str] = None
                self.is_answer = is_answer
                self.ucb_c = ucb_c
                self.prior = prior

            def ucb(self, total_n: int) -> float:
                if self.n == 0:
                    return float("inf")
                exploit = self.q / self.n
                explore = self.ucb_c * math.sqrt(math.log(max(1,total_n)) / self.n)
                return exploit + explore + 0.05 * self.prior

        root = Node(parent=None, docs=docs, is_answer=False)

        def select(node: Node) -> Node:
            cur = node
            while cur.children:
                total_n = sum(ch.n for ch in cur.children) + 1
                cur = max(cur.children, key=lambda ch: ch.ucb(total_n))
            return cur

        def expand(node: Node):
            for _ in range(self.branching):
                ch = Node(parent=node, docs=node.docs, is_answer=True, prior=0.5)
                node.children.append(ch)
            extra = retriever.search(question, k=10)
            have = {str(d.get("doc_id")) for d in node.docs}
            new_docs = [d for d in extra if str(d.get("doc_id")) not in have][:3]
            if new_docs:
                ch = Node(parent=node, docs=node.docs + new_docs, is_answer=False, prior=0.5)
                node.children.append(ch)

        def rollout(node: Node) -> float:
            if node.is_answer:
                if node.answer is None:
                    node.answer = self._generate_once(question, node.docs, logits_processors=None)
                reward = self._evidence_reward(question, node.answer, node.docs, emb_fn=emb_fn)
                return reward
            if not node.children:
                expand(node)
            child = random.choice(node.children)
            if child.is_answer and child.answer is None:
                child.answer = self._generate_once(question, child.docs, logits_processors=None)
            reward = self._evidence_reward(question, child.answer or "", child.docs, emb_fn=emb_fn) if child.is_answer else 0.0
            return reward

        def backprop(node: Node, reward: float):
            cur = node
            while cur is not None:
                cur.n += 1
                cur.q += reward
                cur = cur.parent

        for _ in range(self.simulations):
            leaf = select(root)
            if not leaf.children:
                expand(leaf)
            for ch in leaf.children:
                r = rollout(ch)
                backprop(ch, r)

        best_ans = ""
        best_val = -1e18
        def collect(node: Node):
            nonlocal best_ans, best_val
            if node.is_answer and node.answer is not None:
                val = node.q / max(1, node.n)
                if val > best_val:
                    best_val = val
                    best_ans = node.answer
            for ch in node.children:
                collect(ch)
        collect(root)
        if not best_ans:
            best_ans = self._generate_once(question, docs, logits_processors=None)
        return best_ans
