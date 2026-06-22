"""Generate the injected-defect benchmark: 20 buggy PRs spanning many defect
categories + 8 clean PRs (behaviour-preserving) for false-alarm measurement.

Each entry reverses a "fix" into a bug-introducing PR with exact ground-truth
lines. Roughly half the bugs are pattern-detectable (a static analyser should
catch them); the other half are purely semantic (only a reasoning model should).

To use REAL bugs instead, replace the inline pairs with BugsInPy / Defects4J
file pairs and call make_buggy_pr(...) the same way.

    python -m pullpilot.benchmark.make_example_dataset
"""
from __future__ import annotations

import os

from .build_dataset import make_buggy_pr, make_clean_pr, save_dataset

# (id, file, category, title, fixed, buggy)
BUGGY = [
    ("b01", "indexing.py", "off_by_one", "Adjust last-element index",
     "def get_last(items):\n    return items[len(items) - 1]\n",
     "def get_last(items):\n    return items[len(items)]\n"),

    ("b02", "bucket.py", "mutable_default", "Simplify bucket arg",
     "def add_item(item, bucket=None):\n    if bucket is None:\n        bucket = []\n    bucket.append(item)\n    return bucket\n",
     "def add_item(item, bucket=[]):\n    bucket.append(item)\n    return bucket\n"),

    ("b03", "checks.py", "eq_none", "Tidy None check",
     "def is_empty(x):\n    return x is None\n",
     "def is_empty(x):\n    return x == None\n"),

    ("b04", "parsing.py", "bare_except", "Broaden error handling",
     "def safe_int(s):\n    try:\n        return int(s)\n    except ValueError:\n        return 0\n",
     "def safe_int(s):\n    try:\n        return int(s)\n    except:\n        return 0\n"),

    ("b05", "money.py", "wrong_operator", "Compute net price",
     "def net(price, discount):\n    return price - discount\n",
     "def net(price, discount):\n    return price + discount\n"),

    ("b06", "age.py", "inverted_condition", "Refine adult check",
     "def is_adult(age):\n    return age >= 18\n",
     "def is_adult(age):\n    return age <= 18\n"),

    ("b07", "catalog.py", "none_subscript", "Simplify price lookup",
     "def price_of(catalog, item):\n    entry = catalog.get(item)\n    if entry is None:\n        return 0\n    return entry[\"price\"]\n",
     "def price_of(catalog, item):\n    entry = catalog.get(item)\n    return entry[\"price\"]\n"),

    ("b08", "equality.py", "is_literal", "Compare to one",
     "def is_one(n):\n    return n == 1\n",
     "def is_one(n):\n    return n is 1\n"),

    ("b09", "arith.py", "swapped_args", "Reorder division",
     "def divide(a, b):\n    return a / b\n",
     "def divide(a, b):\n    return b / a\n"),

    ("b10", "io_utils.py", "resource_leak", "Read file contents",
     "def read_file(path):\n    with open(path) as f:\n        return f.read()\n",
     "def read_file(path):\n    f = open(path)\n    return f.read()\n"),

    ("b11", "voting.py", "wrong_boundary", "Voting eligibility",
     "def can_vote(age):\n    return age > 17\n",
     "def can_vote(age):\n    return age > 18\n"),

    ("b12", "numbers.py", "missing_return", "Double a number",
     "def double(x):\n    return x * 2\n",
     "def double(x):\n    x * 2\n"),

    ("b13", "greeting.py", "undefined_name", "Build greeting",
     "def greet(name):\n    msg = \"Hi \" + name\n    return msg\n",
     "def greet(name):\n    msg = \"Hi \" + name\n    return message\n"),

    ("b14", "validate.py", "and_or_mixup", "Range validation",
     "def valid(x):\n    return x > 0 and x < 100\n",
     "def valid(x):\n    return x > 0 or x < 100\n"),

    ("b15", "slicing.py", "off_by_one_slice", "Take first n",
     "def first_n(items, n):\n    return items[:n]\n",
     "def first_n(items, n):\n    return items[:n + 1]\n"),

    ("b16", "presence.py", "neq_none", "Presence check",
     "def has_value(d, k):\n    return d.get(k) is not None\n",
     "def has_value(d, k):\n    return d.get(k) != None\n"),

    ("b17", "access.py", "wrong_index", "Get second item",
     "def second(items):\n    return items[1]\n",
     "def second(items):\n    return items[2]\n"),

    ("b18", "store.py", "mutable_default_dict", "Simplify store arg",
     "def record(key, store=None):\n    if store is None:\n        store = {}\n    store[key] = True\n    return store\n",
     "def record(key, store={}):\n    store[key] = True\n    return store\n"),

    ("b19", "distance.py", "sign_error", "Compute difference",
     "def abs_diff(a, b):\n    return abs(a - b)\n",
     "def abs_diff(a, b):\n    return a - b\n"),

    ("b20", "loader.py", "bare_except_2", "Catch parse errors",
     "def load(parse, text):\n    try:\n        return parse(text)\n    except (ValueError, KeyError):\n        return None\n",
     "def load(parse, text):\n    try:\n        return parse(text)\n    except:\n        return None\n"),
]

# (id, file, title, before, after)
CLEAN = [
    ("c01", "greet2.py", "Add docstring",
     "def greet(name):\n    return \"Hi \" + name\n",
     "def greet(name):\n    \"\"\"Return a greeting.\"\"\"\n    return \"Hi \" + name\n"),
    ("c02", "totals.py", "Rename accumulator",
     "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
     "def total(xs):\n    acc = 0\n    for x in xs:\n        acc += x\n    return acc\n"),
    ("c03", "adder.py", "Add type hints",
     "def add(a, b):\n    return a + b\n",
     "def add(a: int, b: int) -> int:\n    return a + b\n"),
    ("c04", "labels.py", "Use f-string",
     "def label(n):\n    return \"item \" + str(n)\n",
     "def label(n):\n    return f\"item {n}\"\n"),
    ("c05", "geometry.py", "Extract pi constant",
     "def circle_area(r):\n    return 3.14159 * r * r\n",
     "def circle_area(r):\n    pi = 3.14159\n    return pi * r * r\n"),
    ("c06", "sign.py", "Document predicate",
     "def is_pos(x):\n    return x > 0\n",
     "def is_pos(x):\n    \"\"\"True if x is positive.\"\"\"\n    return x > 0\n"),
    ("c07", "combine.py", "Name intermediate result",
     "def combine(a, b, c):\n    return a + b + c\n",
     "def combine(a, b, c):\n    result = a + b + c\n    return result\n"),
    ("c08", "square.py", "Add comment",
     "def square(x):\n    return x * x\n",
     "def square(x):\n    # square the input\n    return x * x\n"),
]


def build():
    prs = []
    for pid, file, category, title, fixed, buggy in BUGGY:
        prs.append(make_buggy_pr(pid, file, buggy, fixed, category=category, title=title))
    for pid, file, title, before, after in CLEAN:
        prs.append(make_clean_pr(pid, file, before, after, title=title))

    out = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "examples", "dataset.json")
    )
    save_dataset(prs, out)
    n_buggy = sum(1 for p in prs if p.label == "buggy")
    print(f"wrote {len(prs)} PRs ({n_buggy} buggy, {len(prs) - n_buggy} clean) to {out}")


if __name__ == "__main__":
    build()
