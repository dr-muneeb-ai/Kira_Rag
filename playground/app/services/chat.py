from pydantic import BaseModel, Field

from app.rag import ask_question


class Source(BaseModel):
    source: str
    page: int | None = None
    excerpt: str | None = None


class Answer(BaseModel):
    question: str
    answer: str
    sources: list[Source] = Field(default_factory=list)


class RagService:
    """Small service wrapper around the application's RAG pipeline."""

    def ask(self, question: str) -> Answer:
        result = ask_question(question)
        return Answer(**result)
if __name__ == "__main__":
    service = RagService()

    print("RAG Chatbot Started!")

    while True:
        question = input("\nYou: ")

        if question.lower() in ["exit", "quit"]:
            break

        response = service.ask(question)

        print("\nAssistant:", response.answer)
