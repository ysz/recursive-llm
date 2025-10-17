"""Example of extracting structured data from unstructured text."""

from rlm import RLM

# Unstructured document with embedded data
document = """
Meeting Notes - Product Planning Session
Date: January 15, 2025
Attendees: Sarah Chen, Mike Johnson, Lisa Park, David Kim

Discussion Topics:

Sarah mentioned that we need to increase our Q1 budget to $250,000 to accommodate
the new marketing campaign. She also noted that our customer satisfaction score
improved from 7.5 to 8.9 in the last quarter.

Mike presented the engineering roadmap. The team plans to ship Feature A by February 15,
Feature B by March 30, and Feature C by April 20. He mentioned they need 3 additional
engineers to meet these deadlines.

Lisa reported that website traffic increased 45% last month, with 125,000 unique visitors.
The conversion rate improved from 2.1% to 3.4%. Email campaign open rates are at 28%.

David shared customer feedback. Key requests include:
- Mobile app improvements (mentioned by 89 customers)
- Better search functionality (67 customers)
- Dark mode support (134 customers)
- Faster load times (45 customers)

Action Items:
- Sarah: Approve budget increase by Jan 20
- Mike: Post job listings for 3 engineers
- Lisa: Launch new email campaign by Feb 1
- David: Prioritize dark mode feature

Next meeting: February 15, 2025 at 2:00 PM
"""


def main():
    """Extract structured data using RLM."""
    rlm = RLM(
        model="gpt-5-mini",
        max_iterations=15,
        temperature=0.3  # Lower temp for more precise extraction
    )

    # Different extraction tasks
    tasks = [
        "Extract all dates mentioned in the document",
        "Extract all numerical metrics (percentages, counts, etc.)",
        "List all action items with assigned owners",
        "Extract feature names and their deadlines",
        "What are the top 3 customer feature requests by number of requests?",
    ]

    print("Data Extraction Examples\n")
    print("=" * 80)

    for task in tasks:
        print(f"\nTask: {task}")
        print("-" * 80)

        try:
            result = rlm.completion(task, document)
            print(f"Result:\n{result}")

        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    main()
