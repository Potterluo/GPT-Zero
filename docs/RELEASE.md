# 发布流程（RELEASE）

> 一次 tag push 同时产出三样东西：**PyPI 包**（`openai-mock`）、**GHCR 多架构镜像**（amd64 + arm64）、
> 以及 **GitHub Pages 文档站**（pages.yml 独立触发）。所有发布动作由
> [.github/workflows/release.yml](../.github/workflows/release.yml) 完成。

## 1. 链路总览

```
push tag v4.0.0
      │
      ▼
┌─ release.yml ─────────────────────────────────────────────────┐
│ verify        tag ↔ gptzero.__version__ 一致 + pytest 全绿    │
│   ├─ python-package   python -m build → twine check → wheel   │
│   │                     内容自检 → upload-artifact             │
│   │     └─ pypi-publish   OIDC Trusted Publishing → PyPI       │
│   ├─ docker-amd64       原生构建 → push-by-digest             │
│   ├─ docker-arm64       (ubuntu-24.04-arm) → push-by-digest   │
│   │     └─ docker-manifest  合并两架构 → :4.0.0 + :latest      │
└──────────────────────────────────────────────────────────────┘

push main
      └─ pages.yml → mkdocs build --strict → GitHub Pages
```

| 产物 | 地址 | 触发 |
|---|---|---|
| PyPI 包 | <https://pypi.org/project/openai-mock/> | tag `v*` |
| GHCR 镜像 | `ghcr.io/potterluo/gpt-zero:{版本,latest}` | tag `v*` |
| 文档站 | <https://potterluo.github.io/GPT-Zero/> | push main / 手动 |

## 2. 日常发布步骤

```bash
# 1) 确认版本号已更新（gptzero/__init__.py 的 __version__，tag 必须与之完全一致）
grep __version__ gptzero/__init__.py     # 例如 4.0.0

# 2) 打 tag 并推送（tag 即发布动作，没有 dry-run）
git tag v4.0.0
git push origin v4.0.0

# 3) 去 Actions 看 release workflow：verify → package/pypi → docker → manifest 全绿
```

发布后验证：

```bash
pip install openai-mock==4.0.0 && gptzero --port 8090      # PyPI 包可装可跑
docker pull ghcr.io/potterluo/gpt-zero:4.0.0             # 多架构 manifest（docker inspect 可见 amd64+arm64）
```

## 3. 一次性配置（首次发布前）

### 3.1 GitHub Pages

仓库 **Settings → Pages → Build and deployment → Source** 选 **GitHub Actions**。
不配置时 pages.yml 会在 deploy 步骤报 "Pages not enabled"。配置一次后自动部署，URL：
<https://potterluo.github.io/GPT-Zero/>

### 3.2 PyPI Trusted Publishing（免 token）

1. 在 <https://pypi.org> 上传者账号里打开 **Publishing → Add a new pending publisher**；
2. 按下表填写（必须与 workflow 完全一致）：

| 字段 | 值 |
|---|---|
| PyPI project name | `openai-mock` |
| Owner | `Potterluo` |
| Repository | `GPT-Zero` |
| Workflow filename | `release.yml` |
| Environment name | `pypi` |

3. 该 environment 建议在仓库 Settings → Environments → pypi 里加 **Required reviewers**，
   让每次 PyPI 发布需要人工点一次确认（双保险）。

未配置 Trusted Publishing 时 `pypi-publish` job 会失败，但**不影响** GHCR 镜像发布；
也可以改用经典方式：在 job 里给 `pypa/gh-action-pypi-publish` 加
`password: ${{ secrets.PYPI_API_TOKEN }}`（先在 PyPI 生成 API token 存入 secret）。

### 3.3 GHCR

无需额外配置：`GITHUB_TOKEN` 自带 `packages: write` 权限（workflow 已声明）。
首次发布后到仓库 **Packages** 页面把 `gpt-zero` 关联到本仓库（Settings → Package settings）。

## 4. 产物细节

- **包名**：PyPI 分发名 `openai-mock`（import 名仍是 `gptzero`）。历史：曾用 `gpt-zero`，因与已被占用的
  `gptzero` 过于相似被 PyPI 拒绝，改用 `openai-mock`；Trusted Publishing 的 project name 必须与之一致。
- **版本自检**：release.yml 的 verify job 要求 tag（去 `v` 前缀）等于 `gptzero.__version__`，
  不一致直接 fail，杜绝「tag 是 v4.0.1、包还是 4.0.0」的错位发布。
- **wheel 自检**：dashboard.html / protocols.py / entry_points 必须在包里，bench 包完整且不混入 agentx。
- **多架构镜像**：amd64（ubuntu-latest）与 arm64（`ubuntu-24.04-arm` 原生 runner）分别构建并按 digest
  推送，`docker buildx imagetools create` 合并成 manifest list——不用 QEMU 模拟，arm 构建速度快且稳。
- **镜像内容**：与 `Dockerfile` 一致（默认 heuristic 分词；`WITH_TRANSFORMERS=1` 精确计数版可本地构建，
  release 镜像走默认配置以保持体积小）。

## 5. 常见问题

| 症状 | 原因 | 处理 |
|---|---|---|
| pages deploy 报 " Pages not enabled " | 仓库没开 Pages | §3.1 配置 Source: GitHub Actions |
| pypi-publish 报 "Trusted publisher could not be validated" | §3.2 的五项配置有一项不一致 | 核对 owner/repo/workflow/environment 拼写 |
| pypi 报 "File already exists" | 同版本号重复发布 | PyPI 不可覆盖：改 `__version__`（如 4.0.1）重新打 tag |
| arm64 job 一直排队 | 仓库不可用 arm runner | 临时把 arm64 job 的 `runs-on` 改回 `ubuntu-latest` 并给 buildx 加 `platforms: linux/arm64`（QEMU 慢但可用） |
| GHCR 401 | 未登录或包未关联仓库 | `echo $GITHUB_TOKEN | docker login ghcr.io -u <user> --password-stdin` |

> PyPI 版本一经发布**不可覆盖**（只能 yank）。所以改完代码想发版时，永远先 bump `__version__`，再打 tag。
