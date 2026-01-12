from __future__ import annotations
from typing import Optional, Set

def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

def pause() -> None:
    input("\nPress Enter to continue...")

def ask_int(prompt: str, default: Optional[int] = None, valid: Optional[Set[int]] = None) -> int:
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            v = int(raw)
            if valid is not None and v not in valid:
                print(f"Please choose one of: {sorted(valid)}")
                continue
            return v
        except ValueError:
            print("Please enter an integer.")

def ask_str(prompt: str, default: Optional[str] = None, valid: Optional[Set[str]] = None) -> str:
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            raw = default
        if valid is not None and raw not in valid:
            print(f"Please choose one of: {sorted(valid)}")
            continue
        return raw
