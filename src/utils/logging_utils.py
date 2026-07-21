# src/utils/logging_utils.py

def log_section(message: str) -> None:
    print("=" * 80)
    print(message)
    print("=" * 80)


def log_step(message: str) -> None:
    print(f"[STEP] {message}")


def log_success(message: str) -> None:
    print(f"[SUCCESS] {message}")