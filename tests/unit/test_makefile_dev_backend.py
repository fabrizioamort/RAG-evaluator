from pathlib import Path
import unittest


def _target_recipe(makefile_text: str, target: str) -> list[str]:
    lines = makefile_text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == f"{target}:":
            recipe: list[str] = []
            for candidate in lines[index + 1 :]:
                if not candidate.startswith("\t"):
                    break
                recipe.append(candidate.strip())
            return recipe
    raise AssertionError(f"Target {target!r} not found in Makefile")


class DevBackendMakefileTests(unittest.TestCase):
    def test_dev_backend_uses_backend_launcher_script(self) -> None:
        makefile_text = Path("Makefile").read_text(encoding="utf-8")

        recipe = _target_recipe(makefile_text, "dev-backend")

        self.assertEqual(recipe, ["cd platform/backend && uv run python dev_server.py"])

    def test_kill_backend_uses_backend_launcher_kill_mode(self) -> None:
        makefile_text = Path("Makefile").read_text(encoding="utf-8")

        recipe = _target_recipe(makefile_text, "kill-backend")

        self.assertEqual(recipe, ["cd platform/backend && uv run python dev_server.py --kill-port 8000"])


if __name__ == "__main__":
    unittest.main()
