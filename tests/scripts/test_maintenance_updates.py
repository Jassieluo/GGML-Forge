import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
UPDATES = ROOT / "scripts" / "maintenance" / "updates"


class MaintenanceEntryPointTest(unittest.TestCase):
    def test_all_discovers_library_updaters(self):
        spec = importlib.util.spec_from_file_location("forge_update_all", UPDATES / "all.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertEqual(
            [path.name for path in module.update_scripts()],
            ["llama_cpp.py", "stable_diffusion_cpp.py", "whisper_cpp.py"],
        )

    def test_entry_points_do_not_depend_on_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for script in ("all.py", "llama_cpp.py", "stable_diffusion_cpp.py", "whisper_cpp.py"):
                result = subprocess.run(
                    [sys.executable, str(UPDATES / script), "--help"],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_copied_conversion_tools_are_locked_to_submodule(self):
        dependency = ROOT / "scripts" / "maintenance" / "dependencies" / "llama_cpp"
        provider = ROOT / "scripts" / "conversion" / "categories" / "llm" / "providers" / "llama_cpp"
        lock = json.loads((dependency / "lock.json").read_text(encoding="utf-8"))
        source = json.loads((provider / "source.json").read_text(encoding="utf-8"))
        self.assertEqual(source["llama_cpp_commit"], lock["llama_cpp_commit"])
        self.assertEqual(source["tree_sha256"], lock["conversion_tools_tree_sha256"])
        self.assertEqual(source["scripts"], lock["conversion_scripts"])

        whisper_dependency = ROOT / "scripts" / "maintenance" / "dependencies" / "whisper_cpp"
        whisper_provider = ROOT / "scripts" / "conversion" / "categories" / "asr" / "providers" / "whisper_cpp"
        whisper_lock = json.loads((whisper_dependency / "lock.json").read_text(encoding="utf-8"))
        whisper_source = json.loads((whisper_provider / "source.json").read_text(encoding="utf-8"))
        self.assertEqual(whisper_source["whisper_cpp_commit"], whisper_lock["whisper_cpp_commit"])
        self.assertEqual(whisper_source["tree_sha256"], whisper_lock["conversion_tools_tree_sha256"])
        self.assertEqual(whisper_source["scripts"], whisper_lock["conversion_scripts"])

        visual_dependency = ROOT / "scripts" / "maintenance" / "dependencies" / "stable_diffusion_cpp"
        visual_provider = (
            ROOT / "scripts" / "conversion" / "categories" / "visual_generation"
            / "providers" / "stable_diffusion_cpp"
        )
        visual_lock = json.loads((visual_dependency / "lock.json").read_text(encoding="utf-8"))
        visual_source = json.loads((visual_provider / "source.json").read_text(encoding="utf-8"))
        self.assertEqual(
            visual_source["stable_diffusion_cpp_commit"],
            visual_lock["stable_diffusion_cpp_commit"],
        )
        self.assertEqual(visual_source["tree_sha256"], visual_lock["conversion_tools_tree_sha256"])
        self.assertEqual(visual_source["scripts"], visual_lock["conversion_scripts"])


if __name__ == "__main__":
    unittest.main()
