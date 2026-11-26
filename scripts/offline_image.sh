#!/usr/bin/env bash
# =============================================================================
# GPT-Zero v4 离线镜像构建 / 导出（Linux/macOS 主用；Windows 用 offline_image.ps1）
# =============================================================================
#   bash scripts/offline_image.sh                       # build + save
#   bash scripts/offline_image.sh -n                     # 只 save 已有镜像（不 build）
#   bash scripts/offline_image.sh -t gptzero:v4 -o dist/gptzero-v4.tar
#   bash scripts/offline_image.sh -f                     # 带 transformers（精确 token 计数）
#   bash scripts/offline_image.sh -p linux/arm64         # buildx 跨架构
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="gptzero:v4"
OUT="dist/gptzero-v4.tar"
WITH_TF=0
NO_BUILD=0
NO_CACHE=0
PLATFORM=""

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while getopts ":t:o:p:nfch" opt; do
  case "$opt" in
    t) TAG="$OPTARG" ;;
    o) OUT="$OPTARG" ;;
    p) PLATFORM="$OPTARG" ;;
    n) NO_BUILD=1 ;;
    f) WITH_TF=1 ;;
    c) NO_CACHE=1 ;;
    h) usage 0 ;;
    *) usage 2 ;;
  esac
done

step() { printf '\n==> %s\n' "$*"; }
ok() { printf '    %s\n' "$*"; }
fmt_size() {
  awk -v b="$1" 'BEGIN {
    if (b >= 1073741824) printf "%.2f GB", b/1073741824;
    else printf "%.1f MB", b/1048576;
  }'
}

cd "$ROOT"

step "检查 Docker 环境"
if ! docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
  echo "错误：docker 不可用（守护进程未启动？）" >&2
  exit 1
fi
ok "Docker Server $(docker version --format '{{.Server.Version}}')"

START=$(date +%s)

if [[ "$NO_BUILD" == "0" ]]; then
  step "构建镜像 ${TAG}（WITH_TRANSFORMERS=${WITH_TF}）"
  if [[ -n "$PLATFORM" ]]; then
    build_cmd=(docker buildx build --load --platform "$PLATFORM" -t "$TAG"
               --build-arg "WITH_TRANSFORMERS=${WITH_TF}")
  else
    build_cmd=(docker build -t "$TAG" --build-arg "WITH_TRANSFORMERS=${WITH_TF}")
  fi
  [[ "$NO_CACHE" == "1" ]] && build_cmd+=(--no-cache)
  build_cmd+=(.)
  printf '    %s\n' "${build_cmd[*]}"
  "${build_cmd[@]}"
  ok "构建完成，用时 $(( $(date +%s) - START ))s"
else
  step "跳过构建（-n），直接打包本地镜像"
fi

step "确认镜像信息"
if ! docker image inspect "$TAG" >/dev/null 2>&1; then
  echo "错误：本地没有镜像 $TAG（去掉 -n 先构建）" >&2
  exit 1
fi
SIZE=$(docker image inspect "$TAG" --format '{{.Size}}')
IMG_ID=$(docker image inspect "$TAG" --format '{{.Id}}')
CREATED=$(docker image inspect "$TAG" --format '{{.Created}}')
ok "镜像      : $TAG"
ok "镜像 ID   : ${IMG_ID:0:19}"
ok "镜像大小  : $(fmt_size "$SIZE")"
ok "构建时间  : $CREATED"

step "docker save → $OUT"
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
docker save -o "$OUT" "$TAG"
TAR_SIZE=$(wc -c <"$OUT" | tr -d ' ')
ok "tar 路径  : $OUT"
ok "tar 大小  : $(fmt_size "$TAR_SIZE")"

step "计算校验值"
if command -v sha256sum >/dev/null 2>&1; then
  SHA=$(sha256sum "$OUT" | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
  SHA=$(shasum -a 256 "$OUT" | awk '{print $1}')
else
  SHA="(无 sha256 工具)"
fi
printf '%s  %s\n' "$SHA" "$(basename "$OUT")" >"${OUT}.sha256"
ok "sha256    : $SHA"
ok "校验文件  : ${OUT}.sha256"

step "完成（总用时 $(( $(date +%s) - START ))s）"
cat <<EOF

────────────────────────────────────────────────────────────────────
离线交付：把 $(basename "$OUT") 拷到目标机，然后
  docker load -i $(basename "$OUT")
  docker images | grep gptzero

离线自检（零网络）：
  docker run --rm --network none ${TAG} smoke

起服务：
  docker run --rm -p 8090:8090 ${TAG}
  # 或  docker compose up -d mock
────────────────────────────────────────────────────────────────────
EOF
