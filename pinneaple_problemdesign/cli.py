"""CLI for interactive Pinneaple Design agent."""
from __future__ import annotations

from .protocol import GeminiProvider
from .agent import DesignAgent


def main():
    llm = GeminiProvider()
    agent = DesignAgent(llm=llm)
    state = agent.start()

    print("Pinneaple Design CLI (type 'exit' to quit)\n")

    while True:
        user = input("You: ").strip()
        if user.lower() in ("exit", "quit"):
            break

        agent.ingest_user_message(state, user)
        out = agent.step(state)

        if out["type"] == "questions":
            print("\nAssistant (questions):")
            print(out["questions_text"])
            if out.get("warnings"):
                print("\nWarnings:")
                for w in out["warnings"]:
                    print(" -", w)
            print("")
        else:
            print("\n=== FINAL REPORT ===\n")
            print(out["markdown"])
            break


if __name__ == "__main__":
    main()
