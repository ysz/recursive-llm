"""Example processing multiple documents with shared context."""

from rlm import RLM

# Simulate multiple related documents
documents = {
    "user_manual.txt": """
User Manual - CloudSync Pro

Installation:
1. Download CloudSync Pro from our website
2. Run the installer (CloudSync-Setup.exe)
3. Follow the setup wizard
4. Enter your license key when prompted

Getting Started:
- Create an account at cloudsync.com
- Install the desktop application
- Configure your sync folders
- CloudSync will automatically backup your files

Features:
- Real-time file synchronization
- End-to-end encryption
- Version history (up to 30 days)
- Cross-platform support (Windows, Mac, Linux)
- 2GB free storage, upgrade to 1TB for $9.99/month
""",

    "troubleshooting.txt": """
Troubleshooting Guide - CloudSync Pro

Common Issues:

Issue: Sync not working
Solution: Check your internet connection. Restart the CloudSync application.
If problem persists, check firewall settings (allow port 443).

Issue: Files not appearing
Solution: Sync can take up to 5 minutes. Check sync status in the app.
Verify you're logged into the same account on all devices.

Issue: Storage full
Solution: Free plan includes 2GB. Delete old files or upgrade to Premium ($9.99/month)
for 1TB storage. Check storage usage in Settings -> Account.

Issue: Login failed
Solution: Reset password at cloudsync.com/reset. Check if Caps Lock is on.
Contact support@cloudsync.com if you can't access your account.
""",

    "pricing.txt": """
CloudSync Pro - Pricing Plans

Free Plan:
- 2GB storage
- 3 devices max
- 30-day version history
- Email support
- Price: $0/month

Premium Plan:
- 1TB storage
- Unlimited devices
- 90-day version history
- Priority email + chat support
- Advanced sharing controls
- Price: $9.99/month

Business Plan:
- 5TB storage
- Unlimited devices
- 1-year version history
- 24/7 phone support
- Team management tools
- Admin console
- Price: $49.99/month per user (min 5 users)

Education Discount:
- 50% off Premium for students/teachers
- Verify with .edu email address
"""
}


def main():
    """Process multiple documents."""
    # Combine all documents
    combined = "\n\n--- FILE: " + "\n\n--- FILE: ".join(
        f"{name} ---\n{content}"
        for name, content in documents.items()
    )

    print(f"Processing {len(documents)} documents")
    print(f"Total size: {len(combined):,} characters\n")

    rlm = RLM(
        model="gpt-5-mini",
        max_iterations=15,
        temperature=0.5
    )

    # Questions that require information from multiple documents
    queries = [
        "What should I do if my sync is not working?",
        "How much does it cost to get 1TB of storage?",
        "What are the steps to install CloudSync Pro?",
        "What's included in the Business plan that's not in Premium?",
        "If I'm a student, how much would Premium cost?",
    ]

    for query in queries:
        print(f"Query: {query}")

        try:
            result = rlm.completion(query, combined)
            print(f"Answer: {result}")
            print(f"Stats: {rlm.stats['llm_calls']} calls\n")

        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
