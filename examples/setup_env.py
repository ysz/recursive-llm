#!/usr/bin/env python
"""Helper script to set up .env file with API keys."""

import os
from pathlib import Path


def setup_env():
    """Interactive setup for .env file."""
    env_file = Path(".env")

    print("=" * 60)
    print("RLM Environment Setup")
    print("=" * 60)
    print()

    if env_file.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
        print()

    print("Enter your API keys (press Enter to skip optional keys):")
    print()

    # OpenAI
    openai_key = input("OpenAI API Key (required): ").strip()
    if not openai_key:
        print("❌ OpenAI API key is required!")
        return

    # Optional keys
    anthropic_key = input("Anthropic API Key (optional): ").strip()
    azure_key = input("Azure OpenAI API Key (optional): ").strip()
    azure_base = ""
    if azure_key:
        azure_base = input("Azure API Base URL: ").strip()

    # Write .env file
    with open(env_file, 'w') as f:
        f.write("# OpenAI API Key\n")
        f.write(f"OPENAI_API_KEY={openai_key}\n")
        f.write("\n")

        if anthropic_key:
            f.write("# Anthropic API Key\n")
            f.write(f"ANTHROPIC_API_KEY={anthropic_key}\n")
            f.write("\n")
        else:
            f.write("# Anthropic API Key (optional)\n")
            f.write("# ANTHROPIC_API_KEY=your-anthropic-api-key-here\n")
            f.write("\n")

        if azure_key:
            f.write("# Azure OpenAI\n")
            f.write(f"AZURE_API_KEY={azure_key}\n")
            if azure_base:
                f.write(f"AZURE_API_BASE={azure_base}\n")
            f.write("\n")
        else:
            f.write("# Azure OpenAI (optional)\n")
            f.write("# AZURE_API_KEY=your-azure-api-key-here\n")
            f.write("# AZURE_API_BASE=https://your-resource.openai.azure.com\n")
            f.write("\n")

        f.write("# Other providers (optional)\n")
        f.write("# GEMINI_API_KEY=your-gemini-api-key-here\n")
        f.write("# COHERE_API_KEY=your-cohere-api-key-here\n")

    print()
    print("✅ .env file created successfully!")
    print()
    print("You can now run the examples:")
    print("  python examples/basic_usage.py")
    print("  python examples/two_models.py")
    print()


def test_env():
    """Test if API keys are loaded."""
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 60)
    print("Testing Environment Variables")
    print("=" * 60)
    print()

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if openai_key:
        masked = openai_key[:7] + "..." + openai_key[-4:] if len(openai_key) > 11 else "***"
        print(f"✅ OPENAI_API_KEY: {masked}")
    else:
        print("❌ OPENAI_API_KEY: Not set")

    if anthropic_key:
        masked = anthropic_key[:7] + "..." + anthropic_key[-4:] if len(anthropic_key) > 11 else "***"
        print(f"✅ ANTHROPIC_API_KEY: {masked}")
    else:
        print("⚠️  ANTHROPIC_API_KEY: Not set (optional)")

    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        try:
            test_env()
        except ImportError:
            print("⚠️  python-dotenv not installed. Install it with:")
            print("   pip install python-dotenv")
    else:
        setup_env()
