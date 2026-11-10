"""Railway Passenger Help Desk."""


def ask(question, agent=None, llm=None, **kwargs):
    """Public entry point returning {"answer", "sources", "sql", "dont_know"}."""
    from .agent import ask as _ask

    return _ask(question, agent=agent, llm=llm, **kwargs)


__all__ = ["ask"]
