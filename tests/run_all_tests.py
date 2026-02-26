# ./run_tests.py
import sys
import pytest

def main() -> int:
    # -q: quiet, feel free to remove
    return pytest.main(["-q"])

if __name__ == "__main__":
    raise SystemExit(main())