#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str


def _extract_front_matter(text: str) -> str | None:
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    return parts[1].strip("\n")


def _parse_skill_spec(front_matter: str) -> SkillSpec | None:
    name: str | None = None
    description: str | None = None

    for raw_line in front_matter.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value.startswith(("|", ">")):
            return None

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        if key == "name":
            name = value
        elif key == "description":
            description = value

    if not name or not description:
        return None

    return SkillSpec(name=name, description=description)


def validate_skills(skills_root: Path) -> list[str]:
    errors: list[str] = []

    if not skills_root.exists():
        return [f"Missing skills root: {skills_root}"]

    skill_dirs = [
        p for p in sorted(skills_root.iterdir()) if p.is_dir() and not p.name.startswith(".")
    ]
    if not skill_dirs:
        errors.append(f"No skill directories found under {skills_root}")
        return errors

    for skill_dir in skill_dirs:
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            errors.append(f"Missing SKILL.md: {skill_md}")
            continue

        nested_skill_mds = sorted(p for p in skill_dir.rglob("SKILL.md") if p != skill_md)
        if nested_skill_mds:
            nested_list = ", ".join(str(p) for p in nested_skill_mds)
            errors.append(f"Nested SKILL.md not allowed under {skill_dir}: {nested_list}")

        front_matter = _extract_front_matter(skill_md.read_text(encoding="utf-8"))
        if front_matter is None:
            errors.append(f"Missing or malformed YAML front matter: {skill_md}")
            continue

        spec = _parse_skill_spec(front_matter)
        if spec is None:
            errors.append(f"Missing required `name`/`description` or invalid YAML: {skill_md}")
            continue

        if "\n" in spec.name or not spec.name.strip():
            errors.append(f"Invalid `name` (must be single-line, non-empty): {skill_md}")
        if len(spec.name) > 100:
            errors.append(f"Invalid `name` (max 100 chars): {skill_md}")

        if "\n" in spec.description or not spec.description.strip():
            errors.append(f"Invalid `description` (must be single-line, non-empty): {skill_md}")
        if len(spec.description) > 500:
            errors.append(f"Invalid `description` (max 500 chars): {skill_md}")

        if skill_dir.name != spec.name:
            errors.append(
                "Skill directory name must match front matter `name`: "
                f"{skill_dir.name!r} != {spec.name!r} ({skill_md})"
            )

    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    skills_root = repo_root / ".codex" / "skills"

    errors = validate_skills(skills_root)
    if errors:
        print("Skill validation failed:")
        for e in errors:
            print(f"- {e}")
        return 1

    print(f"Skill validation OK ({skills_root})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
