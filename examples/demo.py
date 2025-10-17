#!/usr/bin/env python
"""Quick demo of RLM functionality."""

from rlm.repl import REPLExecutor
from rlm.parser import extract_final, is_final
import re

print("=" * 60)
print("RLM Library Demo")
print("=" * 60)
print()

# Demo 1: REPL Execution
print("1. REPL Executor Demo")
print("-" * 60)

repl = REPLExecutor()

# Execute some Python code
context = """
Machine Learning Report 2024

Q1 Revenue: $1.2M
Q2 Revenue: $1.5M
Q3 Revenue: $1.8M
Q4 Revenue: $2.1M

Total: $6.6M
"""

env = {'context': context, 're': re}

# Example 1: Extract all revenue numbers
code1 = """
revenues = re.findall(r'\\$([\\d.]+)M', context)
print(f"Found revenues: {revenues}")
"""

print("Code:")
print(code1)
result = repl.execute(code1, env)
print("Output:", result)
print()

# Example 2: Calculate sum
code2 = """
revenue_values = [float(r) for r in revenues]
total = sum(revenue_values)
print(f"Total revenue: ${total}M")
"""

print("Code:")
print(code2)
result = repl.execute(code2, env)
print("Output:", result)
print()

# Demo 2: Parser
print("2. Response Parser Demo")
print("-" * 60)

# Example LLM response with FINAL
response = """
Let me analyze this...

revenues = re.findall(r'\\$([\\d.]+)M', context)
total = sum([float(r) for r in revenues])

FINAL(f"The total revenue is ${total}M")
"""

print("LLM Response:")
print(response)
print()

if is_final(response):
    answer = extract_final(response)
    print(f"Detected FINAL statement!")
    print(f"Extracted answer: {answer}")
else:
    print("No FINAL statement detected")

print()

# Demo 3: Show how context is used
print("3. Context as Variable Demo")
print("-" * 60)

print("Instead of passing context in the prompt like this:")
print("  prompt = f'Context: {huge_document}\\n\\nQuestion: {query}'")
print()
print("RLM stores context as a Python variable:")
print("  env = {'context': huge_document, 'query': query}")
print()
print("The LLM can then interact with it programmatically:")
print("  - context[:100]  # Peek at start")
print("  - re.findall(pattern, context)  # Search")
print("  - recursive_llm(query, context[1000:2000])  # Recurse")
print()

print("=" * 60)
print("Demo Complete!")
print("=" * 60)
print()
print("To use RLM with a real model:")
print()
print("  from rlm import RLM")
print("  rlm = RLM(model='gpt-5-mini')")
print("  result = rlm.completion(query, long_document)")
print()
