"""Build-time helper: write the cusadi2 kernel source (CMake calls this)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cusadi2.kernel_template import write_kernel_source  # noqa: E402

if __name__ == "__main__":
    write_kernel_source(sys.argv[1], dtype=sys.argv[2] if len(sys.argv) > 2 else "double")
