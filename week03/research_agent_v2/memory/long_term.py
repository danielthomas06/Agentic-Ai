import json
from pathlib import Path


class LongTermMemory:

    def __init__(
        self,
        path: str = "week03/research_agent_v2/memory/store.json",
    ):
        self.path = Path(path)

        if self.path.exists():
            self.memories = json.loads(
                self.path.read_text(encoding="utf-8")
            )
        else:
            self.memories = []

    def add(self, memory: str):

        if memory not in self.memories:

            self.memories.append(memory)

            self._save()

    def get_all(self) -> list[str]:
        return self.memories

    def _save(self):

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.path.write_text(
            json.dumps(
                self.memories,
                indent=2,
            ),
            encoding="utf-8",
        )