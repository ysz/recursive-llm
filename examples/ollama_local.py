"""Example using Ollama for local LLM."""

from rlm import RLM

# Sample document
document = """
Product Inventory Report - Q4 2024

Electronics Department:
- Laptops: 45 units in stock
- Smartphones: 120 units in stock
- Tablets: 30 units in stock
- Headphones: 200 units in stock

Home & Garden:
- Furniture: 15 units in stock
- Tools: 80 units in stock
- Plants: 150 units in stock

Pricing:
- Laptops: $899 each
- Smartphones: $599 each
- Tablets: $399 each
- Headphones: $149 each
- Furniture: $499 average
- Tools: $79 average
- Plants: $25 average

Total inventory value: $247,000
Last updated: December 15, 2024
"""


def main():
    """Run RLM with Ollama."""
    # Initialize RLM with Ollama
    # Make sure Ollama is running: ollama serve
    # And you have a model installed: ollama pull llama3.2
    rlm = RLM(
        model="ollama/llama3.2",
        max_iterations=10,
        temperature=0.5
    )

    # Ask questions
    queries = [
        "How many smartphones are in stock?",
        "What is the total value of electronics inventory?",
        "List all products with less than 50 units in stock",
    ]

    print("Using Ollama (local LLM)\n")

    for query in queries:
        print(f"Query: {query}")

        try:
            result = rlm.completion(query, document)
            print(f"Answer: {result}")
            print(f"Stats: {rlm.stats['llm_calls']} LLM calls, "
                  f"{rlm.stats['iterations']} iterations\n")

        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    # Make sure Ollama is running:
    # 1. Install Ollama: https://ollama.ai
    # 2. Start server: ollama serve
    # 3. Pull model: ollama pull llama3.2
    main()
