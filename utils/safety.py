"""
Basic safety checker utilities (prototype).
In production, replace with calibrated classifiers, policy engine, and human fallback.
"""
from typing import List
import re

class SafetyChecker:
    def __init__(self, banned_patterns: List[str] = None):
        # simple regex patterns — replace with ML classifiers in prod
        self.banned_patterns = banned_patterns or [
            r"(bomb|explode|detonate)",
            r"(password|ssn|social security|credit card)",
            r"(how to make a weapon|instructions to build a weapon)"
        ]

    def check_prompt(self, prompt: str) -> bool:
        txt = prompt.lower()
        for p in self.banned_patterns:
            if re.search(p, txt):
                return False
        return True

    def check_output(self, output: str) -> bool:
        return self.check_prompt(output)
