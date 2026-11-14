"""ALFWorld environment: planner-library reuse (bounded /tmp usage), no-op detection and real-game checks.

The real-game tests need the ALFWorld/TextWorld packages and ALFWORLD_DATA_ROOT pointing at a directory with games/
(unzipped games.zip); they are skipped otherwise.
"""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

from recipes.alfworld.env import alfworld_wrapper
from recipes.alfworld.env.alfworld_wrapper import AlfworldTextworldEnv

HAS_VERL = importlib.util.find_spec("verl") is not None
HAS_TEXTWORLD = all(importlib.util.find_spec(m) is not None for m in ("textworld", "alfworld"))
DATA_ROOT = os.environ.get("ALFWORLD_DATA_ROOT", "")
HAS_GAMES = bool(DATA_ROOT) and (Path(DATA_ROOT) / "games").is_dir()


class _FakePddlEnv:
    """Bottom of a TextWorld wrapper chain; owns one copy of the planner library."""

    def __init__(self, lib):
        self.downward_lib = lib


class _FakeWrapper:
    def __init__(self, inner):
        self._wrapped_env = inner


class _FakeGymEnv:
    def __init__(self, envs):
        self.batch_env = types.SimpleNamespace(envs=envs)
        self.closed = False

    def close(self):
        self.closed = True


class DownwardPoolTest(unittest.TestCase):
    """A closed environment returns its library copy, and the next environment takes it instead of loading a new one."""

    def setUp(self):
        self.loaded = []

        def load_lib():
            lib = object()
            self.loaded.append(lib)
            return lib

        self.fake = types.ModuleType("fast_downward")
        self.fake.load_lib = load_lib
        self.saved_module = sys.modules.get("fast_downward")
        sys.modules["fast_downward"] = self.fake
        alfworld_wrapper._free_downward_libs.clear()
        alfworld_wrapper._downward_pool_installed = False

    def tearDown(self):
        if self.saved_module is None:
            sys.modules.pop("fast_downward", None)
        else:
            sys.modules["fast_downward"] = self.saved_module
        alfworld_wrapper._free_downward_libs.clear()
        alfworld_wrapper._downward_pool_installed = False

    def env_holding(self, lib):
        env = AlfworldTextworldEnv(data_root="unused")
        pddl_env = _FakePddlEnv(lib)
        env._env = _FakeGymEnv([_FakeWrapper(_FakeWrapper(pddl_env))])
        return env, pddl_env

    def test_closed_env_hands_its_library_to_the_next_one(self):
        alfworld_wrapper._install_downward_pool()
        first = self.fake.load_lib()
        env, pddl_env = self.env_holding(first)
        gym_env = env._env
        env.close()
        self.assertTrue(gym_env.closed)
        self.assertIsNone(pddl_env.downward_lib)
        self.assertIs(self.fake.load_lib(), first)  # reused: no new copy
        self.assertEqual(len(self.loaded), 1)
        self.assertIsNot(self.fake.load_lib(), first)  # pool empty again: a new copy
        self.assertEqual(len(self.loaded), 2)

    def test_install_is_idempotent(self):
        alfworld_wrapper._install_downward_pool()
        patched = self.fake.load_lib
        alfworld_wrapper._install_downward_pool()
        self.assertIs(self.fake.load_lib, patched)

    def test_close_without_an_env(self):
        AlfworldTextworldEnv(data_root="unused").close()
        self.assertEqual(alfworld_wrapper._free_downward_libs, [])


@unittest.skipUnless(HAS_VERL, "needs verl: the no-op rule lives in the agent flow")
class NoopDetectionTest(unittest.TestCase):
    def test_noop_rule(self):
        from recipes.alfworld.alfworld_agent_flow import _is_noop

        self.assertTrue(_is_noop("You are in the kitchen.", "You are in the kitchen."))
        self.assertTrue(_is_noop("You are in the kitchen.", "Nothing happens."))
        self.assertFalse(_is_noop("You are in the kitchen.", "You arrive at countertop 1."))


def _mapped_library_copies() -> int:
    """Count distinct deleted-but-mapped libdownward copies of this process (each one holds its disk blocks)."""
    inodes = set()
    with open("/proc/self/maps", encoding="utf-8") as f:
        for line in f:
            if "libdownward" in line and "(deleted)" in line:
                inodes.add(line.split()[4])
    return len(inodes)


@unittest.skipUnless(HAS_TEXTWORLD and HAS_GAMES, "needs alfworld/textworld and ALFWORLD_DATA_ROOT with games/")
class RealGameTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        games_root = Path(DATA_ROOT) / "games"
        games = sorted((games_root / "valid_seen").rglob("game.tw-pddl"))[:2]
        cls.games = [str(path.relative_to(games_root)) for path in games]

    def play(self, game: str, steps: int = 4) -> list:
        """Play a deterministic command sequence and return everything the agent would observe."""
        env = AlfworldTextworldEnv(data_root=DATA_ROOT)
        try:
            obs, info = env.reset_with_info(game)
            trace = [(obs, tuple(info["admissible_commands"]))]
            for _ in range(steps):
                command = sorted(c for c in info["admissible_commands"] if c not in ("look", "inventory"))[0]
                obs, reward, done, info = env.step(command)
                trace.append((command, obs, reward, done, tuple(info.get("admissible_commands") or ())))
                if done:
                    break
            return trace
        finally:
            env.close()

    def test_reset_and_noop_commands(self):
        env = AlfworldTextworldEnv(data_root=DATA_ROOT)
        try:
            obs, info = env.reset_with_info(self.games[0])
            self.assertIn("Your task is to", obs)
            self.assertTrue(info["admissible_commands"])
            first_look = env.step("look")[0]
            self.assertEqual(env.step("look")[0], first_look)  # repeating `look` changes nothing
            self.assertTrue(env.step("jump around")[0].strip().startswith("Nothing happens"))
        finally:
            env.close()

    def test_reused_library_plays_identically(self):
        alfworld_wrapper._free_downward_libs.clear()  # the first play below loads a fresh copy
        fresh = self.play(self.games[0])
        reused = self.play(self.games[0])  # takes the copy the first play returned
        self.assertEqual(fresh, reused)
        self.assertEqual(self.play(self.games[1]), self.play(self.games[1]))

    @unittest.skipUnless(Path("/proc/self/maps").exists(), "needs /proc")
    def test_library_copies_stay_bounded(self):
        self.play(self.games[0], steps=1)
        before = _mapped_library_copies()
        self.assertGreaterEqual(before, 1)  # the counter sees the planner library at all
        for _ in range(6):
            self.play(self.games[1], steps=1)
        self.assertEqual(_mapped_library_copies(), before)
        # Control: forgetting the pooled copy forces a new load, and the counter catches the leaked copy
        alfworld_wrapper._free_downward_libs.clear()
        self.play(self.games[1], steps=1)
        self.assertEqual(_mapped_library_copies(), before + 1)


if __name__ == "__main__":
    unittest.main()
