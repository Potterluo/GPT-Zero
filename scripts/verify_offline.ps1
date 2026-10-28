<#
.SYNOPSIS
    证明镜像「纯离线可用」：在 --network none（无网卡、无 DNS、无外网）下跑完整 smoke。

.DESCRIPTION
    这是交付验收的硬证据：容器被彻底断网后仍能起 mock、跑通 chat/流式/指标/统计，
    说明运行期没有任何外网依赖（pip / HF Hub / CDN 全都不参与）。

    顺便校验：镜像内确实没有 curl/wget（探活靠标准库）、HF 离线环境变量已生效。

.PARAMETER Tag
    镜像 tag，默认 gptzero:v3。

.PARAMETER Port
    容器内端口，默认 8090。

.EXAMPLE
    .\scripts\verify_offline.ps1
    .\scripts\verify_offline.ps1 -Tag gptzero:v3-http
#>
[CmdletBinding()]
param(
    [string]$Tag = "gptzero:v3",
    [int]$Port = 8090
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "    [PASS] $msg" -ForegroundColor Green }
function Write-Bad($msg) { Write-Host "    [FAIL] $msg" -ForegroundColor Red }

$failures = @()

Write-Step "1/3 镜像存在性"
docker image inspect $Tag --format '{{.Id}}' | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Bad "本地没有镜像 $Tag（先跑 scripts/offline_image.ps1）"
    exit 1
}
$size = [double](docker image inspect $Tag --format '{{.Size}}').Trim()
Write-Ok ("{0}（{1:N1} MB）" -f $Tag, ($size / 1MB))

Write-Step "2/3 断网自检：docker run --rm --network none $Tag smoke"
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$env:GPTZERO_PORT = "$Port"
& docker run --rm --network none -e "GPTZERO_PORT=$Port" -e "GPTZERO_HOST=127.0.0.1" $Tag smoke
$rc = $LASTEXITCODE
$sw.Stop()
if ($rc -eq 0) {
    Write-Ok ("smoke 通过（--network none），耗时 {0:N1}s" -f $sw.Elapsed.TotalSeconds)
}
else {
    Write-Bad "smoke 失败，退出码 $rc"
    $failures += "smoke(rc=$rc)"
}

Write-Step "3/3 离线环境校验（无 curl/wget、HF 离线开关、依赖可导入）"
$probe = @'
import importlib.util, os, shutil, sys
missing = [m for m in ("fastapi", "uvicorn", "pydantic", "httpx", "matplotlib", "numpy", "pytest")
           if importlib.util.find_spec(m) is None]
problems = []
if missing:
    problems.append("缺依赖: " + ",".join(missing))
for tool in ("curl", "wget"):
    if shutil.which(tool):
        problems.append(f"镜像里存在 {tool}（应只用标准库探活）")
if os.environ.get("HF_HUB_OFFLINE") != "1":
    problems.append("HF_HUB_OFFLINE 未设置")
sys.path.insert(0, "/app")
import gptzero  # noqa: E402
print("gptzero version:", gptzero.__version__)
print("tokenizer 默认:", os.environ.get("GPTZERO_TOKENIZER", "auto"))
print("PASS" if not problems else "FAIL: " + "; ".join(problems))
sys.exit(0 if not problems else 1)
'@
$probe | docker run --rm --network none -i --entrypoint python3 -e "GPTZERO_PORT=$Port" $Tag -
if ($LASTEXITCODE -ne 0) {
    Write-Bad "离线环境校验失败"
    $failures += "env-probe"
}
else {
    Write-Ok "离线环境校验通过（标准库探活、HF 离线、依赖齐全）"
}

Write-Host "`n══════════════════════════════════════════════════════════════"
if ($failures.Count -gt 0) {
    Write-Host " 结论: 离线验证失败 → $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host " 结论: 离线验证通过 —— 镜像在 --network none 下可独立完成 mock + 全部 smoke 检查" -ForegroundColor Green
Write-Host "══════════════════════════════════════════════════════════════"
exit 0
