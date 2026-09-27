"""Optimal Huffman code lengths, unrestricted and length-limited.

Used by the decision features to measure how far each block's transmitted
Huffman table is from the optimal one for the symbols the block actually
contains (proposal section III.A, "how far each block's Huffman table is from
the optimal one").

Two constructions are provided:

* :func:`huffman_lengths` -- classic Huffman (heap), optimal without a length
  limit;
* :func:`limited_lengths` -- the package-merge algorithm (Larmore and
  Hirschberg), optimal under a maximum code length such as DEFLATE's 15 bits.
  It falls back to the plain Huffman result whenever that already fits.
"""

from __future__ import annotations

import heapq


def huffman_lengths(freqs: list[int]) -> list[int]:
    """Optimal (unrestricted) code lengths for the non-zero frequencies."""
    lengths = [0] * len(freqs)
    used = [(f, i) for i, f in enumerate(freqs) if f > 0]
    if not used:
        return lengths
    if len(used) == 1:
        lengths[used[0][1]] = 1
        return lengths
    # heap items: (weight, tie-breaker, members)
    heap = [(f, i, [i]) for f, i in used]
    heapq.heapify(heap)
    counter = len(freqs)
    while len(heap) > 1:
        w1, _, m1 = heapq.heappop(heap)
        w2, _, m2 = heapq.heappop(heap)
        for s in m1:
            lengths[s] += 1
        for s in m2:
            lengths[s] += 1
        heapq.heappush(heap, (w1 + w2, counter, m1 + m2))
        counter += 1
    return lengths


def limited_lengths(freqs: list[int], max_bits: int = 15) -> list[int]:
    """Optimal code lengths subject to ``max_bits`` (package-merge)."""
    lengths = huffman_lengths(freqs)
    if not lengths or max(lengths) <= max_bits:
        return lengths

    leaves = sorted((f, i) for i, f in enumerate(freqs) if f > 0)
    n = len(leaves)
    if n > (1 << max_bits):
        raise ValueError("too many symbols for the length limit")
    # each list item: (weight, symbol or -1, (list_index, first_child) or None)
    leaf_items = [(f, i, None) for f, i in leaves]
    lists: list[list[tuple]] = []
    current = leaf_items
    for _ in range(max_bits - 1):
        lists.append(current)
        li = len(lists) - 1
        packages = [
            (current[k][0] + current[k + 1][0], -1, (li, k))
            for k in range(0, len(current) - 1, 2)
        ]
        # merge by weight; leaves win ties so shorter codes go to leaves
        merged: list[tuple] = []
        a = b = 0
        while a < len(leaf_items) or b < len(packages):
            if b >= len(packages) or (
                a < len(leaf_items) and leaf_items[a][0] <= packages[b][0]
            ):
                merged.append(leaf_items[a])
                a += 1
            else:
                merged.append(packages[b])
                b += 1
        current = merged
    lists.append(current)

    counts = [0] * len(freqs)
    top = len(lists) - 1
    stack = [(top, j) for j in range(2 * n - 2)]
    while stack:
        li, j = stack.pop()
        _, symbol, child = lists[li][j]
        if child is None:
            counts[symbol] += 1
        else:
            cli, k = child
            stack.append((cli, k))
            stack.append((cli, k + 1))
    return counts


def code_cost(freqs: list[int], lengths: list[int]) -> int:
    """Bits needed to code ``freqs`` with ``lengths`` (ignoring extra bits)."""
    return sum(f * l for f, l in zip(freqs, lengths))


def kraft_ok(lengths: list[int], max_bits: int = 15) -> bool:
    """True when the lengths form a prefix code (Kraft sum <= 1)."""
    limit = 1 << max_bits
    return sum(1 << (max_bits - l) for l in lengths if l) <= limit
