"""Sphinx configuration — fleet standard via py-canon."""

from py_canon.sphinx import configure

configure(globals())
globals()["exclude_patterns"].append("evidence-results.md")
# Render $...$ and $$...$$ math in Markdown pages with MathJax.
globals()["myst_enable_extensions"] += ["dollarmath", "amsmath"]
