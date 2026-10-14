"""AMD kernels compiled by the system's clang, so that the AMD path needs no ROCm.

tinygrad renders C for AMD GPUs and compiles it with ROCm's comgr. clang compiles that C itself,
given the AMDGPU target that most distributions' clang has: to assembly, which it then assembles to
an object file that tinygrad's ELF loader links, as clang's own HIP compiles would need a linker,
lld. The few device-library functions tinygrad's C calls become clang's builtins.
"""

import hashlib
import pathlib
import re
import shutil
import subprocess
import tempfile

from tinygrad.device import CompileError, Compiler
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.uop.ops import Ops

FLAGS: tuple[str, ...] = ()  # clang's flags beside its own: the emulator's tests set some

# what ROCm's device libraries would give tinygrad's C, as clang's builtins: exp2 and log2 as the
# GPU's instructions, without the libraries' care for denormal values, under 2^-126
PRELUDE = """
#define __ockl_get_local_id(d) ((d) == 0 ? __builtin_amdgcn_workitem_id_x() : \\
    (d) == 1 ? __builtin_amdgcn_workitem_id_y() : __builtin_amdgcn_workitem_id_z())
#define __ockl_get_group_id(d) ((d) == 0 ? __builtin_amdgcn_workgroup_id_x() : \\
    (d) == 1 ? __builtin_amdgcn_workgroup_id_y() : __builtin_amdgcn_workgroup_id_z())
#define __ocml_exp2_f32 __builtin_amdgcn_exp2f
#define __ocml_log2_f32 __builtin_amdgcn_logf
#define __ocml_sqrt_f32 __builtin_sqrtf
#define __ocml_trunc_f32 __builtin_truncf
"""

# the device libraries' declarations, which PRELUDE's macros stand in for
_DECLARATIONS = re.compile(
    r'extern "C" __attribute__\(\(device[^)]*\)\) [^;]*__o(?:ckl|cml)_[^;]*;\n'
)


class ClangCompiler(Compiler):
    def __init__(self, arch: str, flags: tuple[str, ...] = ()):
        if shutil.which("clang") is None:
            raise RuntimeError("LeatEngine's AMD kernels need clang, with its AMDGPU target")
        self.arch, self.flags = arch, flags
        # tinygrad's cache of compiled kernels keys them by this and their C, without PRELUDE
        digest = hashlib.sha256(" ".join((PRELUDE, *flags)).encode()).hexdigest()[:16]
        super().__init__(f"compile_clang_{arch}_{digest}")

    def compile(self, src: str) -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            source, asm, out = (pathlib.Path(tmp) / f"kernel.{x}" for x in ("hip", "s", "o"))
            source.write_text(PRELUDE + _DECLARATIONS.sub("", src))
            _clang("-x", "hip", "--cuda-device-only", "-S", "-nogpulib", "-nogpuinc",
                   f"--offload-arch={self.arch}", "-O3", "-std=c++17", "-mcumode", *self.flags,
                   "-o", str(asm), str(source))  # fmt: skip
            _clang("-target", "amdgcn-amd-amdhsa", f"-mcpu={self.arch}", "-c", str(asm),
                   "-o", str(out))  # fmt: skip
            return out.read_bytes()


def _clang(*args: str) -> None:
    done = subprocess.run(["clang", *args], capture_output=True, text=True)
    if done.returncode:
        raise CompileError(done.stderr)


class ClangRenderer(HIPRenderer):
    """tinygrad's C for AMD GPUs, compiled by ClangCompiler. No device library has sines here:
    tinygrad works them out of other ops."""

    code_for_op = {k: v for k, v in HIPRenderer.code_for_op.items() if k is not Ops.SIN}

    def __init__(self, target: Target, use_hipcc: bool = False):
        # hipcc's compiler, unlike comgr's, checks for nothing when made, and is replaced
        super().__init__(target, use_hipcc=True)
        self.compiler = ClangCompiler(target.arch, FLAGS)


def install() -> None:
    """Makes tinygrad's AMD devices render C for clang rather than for ROCm."""
    import tinygrad.runtime.ops_amd as amd

    amd.HIPRenderer = ClangRenderer  # type: ignore[misc]
