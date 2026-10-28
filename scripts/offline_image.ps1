<#
.SYNOPSIS
    GPT-Zero v3 离线镜像构建 / 导出（Windows 主用脚本）。

.DESCRIPTION
    一次做完：docker build → docker save 成 tar → 打印校验值与 docker load 用法。
    构建期只访问清华 PyPI 镜像；运行期纯离线。

    产物体积：默认（WITH_TRANSFORMERS=0）约 700MB tar；带 transformers 约 +160MB。

.PARAMETER Tag
    镜像 tag，默认 gptzero:v3。

.PARAMETER Out
    tar 输出路径，默认 dist/gptzero-v3.tar（dist/ 已在 .gitignore 里）。

.PARAMETER NoBuild
    跳过 docker build，只打包本地已有镜像（用于「镜像已在别的机器构建好」的场景）。

.PARAMETER NoCache
    docker build --no-cache，强制重装依赖。

.PARAMETER WithTransformers
    构建时安装 transformers（精确 token 计数，镜像更大）。

.PARAMETER Platform
    目标平台，如 linux/arm64（需 docker buildx + QEMU）。

.EXAMPLE
    .\scripts\offline_image.ps1
    .\scripts\offline_image.ps1 -NoBuild -Tag gptzero:v3
    .\scripts\offline_image.ps1 -WithTransformers -Out dist/gptzero-v3-hf.tar
#>
[CmdletBinding()]
param(
    [string]$Tag = "gptzero:v3",
    [string]$Out = "",
    [switch]$NoBuild,
    [switch]$NoCache,
    [switch]$WithTransformers,
    [string]$Platform = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Out) { $Out = Join-Path $root "dist/gptzero-v3.tar" }
if (-not [System.IO.Path]::IsPathRooted($Out)) { $Out = Join-Path $root $Out }

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "    $msg" -ForegroundColor Yellow }

function Format-Size([double]$bytes) {
    if ($bytes -ge 1GB) { return ("{0:N2} GB" -f ($bytes / 1GB)) }
    return ("{0:N1} MB" -f ($bytes / 1MB))
}

Write-Step "检查 Docker 环境"
docker version --format '{{.Server.Version}}' | Out-Null
if ($LASTEXITCODE -ne 0) { throw "docker 不可用：请先启动 Docker Desktop" }
$serverVer = (docker version --format '{{.Server.Version}}').Trim()
Write-Ok "Docker Server $serverVer"

$withTf = 0
if ($WithTransformers) { $withTf = 1 }

Push-Location $root
try {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    if (-not $NoBuild) {
        Write-Step "构建镜像 $Tag（WITH_TRANSFORMERS=$withTf）"
        $buildArgs = @("build", "-t", $Tag, "--build-arg", "WITH_TRANSFORMERS=$withTf", ".")
        if ($NoCache) { $buildArgs += "--no-cache" }
        if ($Platform) {
            $buildArgs = @("buildx", "build", "--load", "--platform", $Platform,
                "-t", $Tag, "--build-arg", "WITH_TRANSFORMERS=$withTf") + ($(if ($NoCache) { @("--no-cache") } else { @() })) + @(".")
        }
        Write-Host ("    docker " + ($buildArgs -join " ")) -ForegroundColor DarkGray
        & docker @buildArgs
        if ($LASTEXITCODE -ne 0) { throw "docker build 失败（退出码 $LASTEXITCODE）" }
        Write-Ok ("构建完成，用时 {0:N0}s" -f $sw.Elapsed.TotalSeconds)
    }
    else {
        Write-Step "跳过构建（-NoBuild），直接打包本地镜像"
    }

    Write-Step "确认镜像信息"
    $sizeRaw = docker image inspect $Tag --format '{{.Size}}'
    if ($LASTEXITCODE -ne 0) { throw "本地没有镜像 $Tag；去掉 -NoBuild 先构建" }
    $size = [double]$sizeRaw.Trim()
    $imgId = (docker image inspect $Tag --format '{{.Id}}').Trim()
    $created = (docker image inspect $Tag --format '{{.Created}}').Trim()
    Write-Ok ("镜像        : $Tag")
    Write-Ok ("镜像 ID     : {0}" -f $imgId.Substring(0, [Math]::Min(19, $imgId.Length)))
    Write-Ok ("镜像大小    : {0}" -f (Format-Size $size))
    Write-Ok ("构建时间    : $created")

    Write-Step "docker save → $Out"
    $outDir = Split-Path -Parent $Out
    if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }
    if (Test-Path $Out) { Remove-Item $Out -Force }
    & docker save -o $Out $Tag
    if ($LASTEXITCODE -ne 0) { throw "docker save 失败（退出码 $LASTEXITCODE）" }
    $tarSize = (Get-Item $Out).Length
    Write-Ok ("tar 路径    : $Out")
    Write-Ok ("tar 大小    : {0}" -f (Format-Size $tarSize))

    Write-Step "计算校验值"
    $sha = (Get-FileHash -Path $Out -Algorithm SHA256).Hash.ToLower()
    $shaPath = "$Out.sha256"
    "$sha  $([System.IO.Path]::GetFileName($Out))" | Set-Content -Path $shaPath -Encoding ASCII
    Write-Ok "sha256      : $sha"
    Write-Ok "校验文件    : $shaPath"

    $sw.Stop()
    Write-Step ("完成（总用时 {0:N0}s）" -f $sw.Elapsed.TotalSeconds)
}
finally {
    Pop-Location
}

Write-Host @"

────────────────────────────────────────────────────────────────────
离线交付：把 $Out 拷到目标机，然后
  docker load -i "$([System.IO.Path]::GetFileName($Out))"
  docker images | Select-String gptzero

离线自检（不起任何网络）：
  docker run --rm --network none $Tag smoke
  或  .\scripts\verify_offline.ps1 -Tag $Tag

起服务：
  docker run --rm -p 8090:8090 $Tag
  # 或  docker compose up -d mock
────────────────────────────────────────────────────────────────────
"@ -ForegroundColor DarkGray
