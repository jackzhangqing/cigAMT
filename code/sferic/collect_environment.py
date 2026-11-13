"""Save the actual server hardware/software environment without changing it."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def main():
    report = dict(status="ready",python=sys.version,executable=sys.executable,
                  platform=platform.platform(),packages={})
    for name in ("numpy","scipy","scikit-learn","torch","h5py","matplotlib"):
        try:report["packages"][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:report["packages"][name]=None
    try:
        import torch
        report.update(torch_version=torch.__version__,cuda_available=torch.cuda.is_available(),
            torch_cuda_runtime=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
            gpu_devices=[dict(index=i,name=torch.cuda.get_device_name(i),
                total_memory_bytes=torch.cuda.get_device_properties(i).total_memory,
                compute_capability=list(torch.cuda.get_device_capability(i)))
                for i in range(torch.cuda.device_count())])
    except (ImportError,OSError) as error:
        report.update(status="torch_unavailable",torch_error=str(error))
    try:
        result=subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],
            capture_output=True,text=True,timeout=5,check=False)
        report["nvidia_smi"]=result.stdout.strip() if result.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):report["nvidia_smi"]=None
    path=Path(__file__).resolve().parent/"results/environment_report.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))
    return 0 if report["status"]=="ready" else 1


if __name__=="__main__":
    raise SystemExit(main())
