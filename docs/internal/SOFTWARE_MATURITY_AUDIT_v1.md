# 소프트웨어 성숙도 점검 (v1)

## 0. 문서 목적과 범위

**목적.** "현재 리포지토리가 소프트웨어로서 성숙한가"라는 질문에, 코드를 읽는 것뿐 아니라 실제로 **빌드하고 테스트를 돌려서** 확인한 사실을 근거로 답한다. 리팩터링/모듈화 관점은 이미 [`CODE_STRUCTURE_MODULARITY_ANALYSIS_v1.md`](CODE_STRUCTURE_MODULARITY_ANALYSIS_v1.md)가, SoftwareX 제출 관점의 포지셔닝·체크리스트는 [`SoftwareX_SSAT_positioning_and_submission_checklist.md`](SoftwareX_SSAT_positioning_and_submission_checklist.md)가 다루고 있으므로, 이 문서는 **그 두 문서가 다루지 않은 부분**(패키징/릴리스 위생, 정적 분석 도구 부재, 문서 잔재물, 실측 검증)에 집중한다. §5에서 세 문서의 관계를 정리한다.

**방법.** `region-sensitivity-workspace` 컨테이너 안에서 전체 테스트 스위트와 README의 quickstart 절차를 실제로 실행했고, 저장소 메타데이터(git 태그, `.gitignore`, 버전 문자열)를 직접 조회했다. 모든 findings는 실제 명령 실행 결과 또는 `file:line` 근거를 가진다. 코드 변경은 포함하지 않는다.

---

## 1. 총평

**핵심 결론: 이 코드베이스는 "프로토타입"이 아니라 "잘 관리되는 alpha 소프트웨어"에 가깝다.** 실제로 실행해 확인한 결과, 1093개 테스트가 전부 통과했고(§2), README에 적힌 quickstart 명령이 문서 그대로 동작했으며, TODO/FIXME/bare-except 같은 미완성의 흔적이 소스 전체에서 발견되지 않았다. 체크포인트 로딩은 `weights_only=True`와 pickle global 허용목록으로 방어되어 있는 등(`ssat/core/adapter/checkpoint.py`, `mmaction_checkpoint.py`), 연구용 도구에서 종종 생략되는 보안 관례까지 이미 지켜지고 있다.

**그럼에도 남아 있는 성숙도 공백은 "핵심 로직"이 아니라 "패키징·릴리스·정적 검증" 층위에 몰려 있다.** 아래 §3에서 다루는 항목들은 공통적으로 다음 성격을 갖는다: 각각은 개별적으로는 작은 수정이지만, 방치하면 **다른 연구자가 이 소프트웨어를 설치·인용·재현하려는 순간**(SoftwareX 심사자, 논문 재현을 시도하는 독자) 가장 먼저 부딪히는 지점이라는 점에서 우선순위가 낮지 않다.

---

## 2. 실측으로 검증된 현재 상태

아래는 "코드를 읽고 추정"한 것이 아니라 **실제로 명령을 실행해 확인**한 사실이다.

| 항목 | 결과 | 근거 |
|---|---|---|
| 전체 테스트 스위트 | **1093 passed, 0 failed**, 83.8초 | `docker compose exec region-sensitivity-workspace python -m pytest -q` (본 세션에서 직접 실행) |
| README quickstart의 `ssat estimate` | 문서에 적힌 그대로 동작(의도된 partial-check FAIL 포함) | 본 세션에서 `configs/examples/quickstart.yaml`로 직접 실행 |
| `TODO`/`FIXME`/`XXX`/`HACK` 마커 | `ssat/` 전체에서 **0건** | `grep -rnE "TODO|FIXME|XXX|HACK" ssat/` |
| bare `except:` | **0건** | `grep -rnE "except\s*:" ssat/` |
| 체크포인트 로딩 보안 관례 | `torch.load(..., weights_only=True)` + pickle global 허용목록 적용 | [`checkpoint.py:25-29`](../../ssat/core/adapter/checkpoint.py#L25-L29), [`mmaction_checkpoint.py:94-120`](../../ssat/core/adapter/mmaction_checkpoint.py#L94-L120) |
| 린트/타입체크 도구 | **미구성** — `ruff`/`mypy`/`black`/`flake8` 중 어느 것도 `pyproject.toml`, CI, `CONTRIBUTING.md`에 없음 | §3.3 |
| git 태그 | `v0.1.0` 하나, HEAD로부터 22 커밋 뒤처짐 | `git log v0.1.0..HEAD --oneline \| wc -l` |
| CI 구성 | 필수 job 2개(`test`, `clean-install`) + advisory job 2개(재현성 데모, 벤치마크) — 구성 자체는 성숙함 | [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) |

이 표 자체가 이미 하나의 결론이다: **"작동하는가"라는 질문에는 이미 명확히 "그렇다"로 답할 수 있는 상태**이고, 남은 작업은 "그 상태를 외부인이 신뢰하고 재사용할 수 있는 형태로 포장하는 것"에 집중되어야 한다.

---

## 3. 구현자가 조치할 수 있는 findings

우선순위 순으로 정렬했다. 각 항목은 문제→왜 중요한지→구체적 조치안 순서로 서술한다.

### 3.1 [High] 버전 문자열이 3곳에서 독립적으로 관리됨 — 단일 소스 없음

**문제.** 패키지 버전 `0.1.0`이 서로 다른 메커니즘으로 동기화 없이 3곳에 하드코딩되어 있다:

- [`pyproject.toml:7`](../../pyproject.toml#L7) — `version = "0.1.0"` (setuptools가 실제 빌드에 사용하는 canonical 값)
- [`ssat/__init__.py:5`](../../ssat/__init__.py#L5) — `__version__ = "0.1.0"` (런타임에 `ssat --version`, `AuditApplication.CODE_VERSION`, 리포트의 `tool_version` 필드가 참조하는 값 — [`cli.py:80`](../../ssat/cli.py#L80), [`application.py:101`](../../ssat/application/application.py#L101), [`assembler.py:781`](../../ssat/report/assembler.py#L781))
- [`compose.deploy.yaml:5`](../../compose.deploy.yaml#L5) — `image: local/ssat:0.1.0` (배포용 Docker 이미지 태그)

**왜 중요한가.** 다음 릴리스에서 `pyproject.toml`만 갱신하고 `__init__.py`를 잊으면, `pip show ssat`이 보고하는 버전과 리포트 산출물에 실제로 기록되는 `tool_version`(§2 표의 재현성 항목이기도 하다 — provenance에 기록되는 값)이 서로 어긋난다. 이는 재현성 문서(`docs/REPRODUCIBILITY_DEMO_v1.md`)가 약속하는 "code version 보존"의 신뢰도를 직접 훼손하는 종류의 드리프트다.

**조치안.** `ssat/__init__.py`가 `pyproject.toml`의 값을 단일 진실 공급원으로 삼도록 `importlib.metadata.version("ssat")`로 런타임에 조회하거나, `setuptools`의 dynamic version(`[tool.setuptools.dynamic]`)을 사용해 `pyproject.toml`이 유일한 값이 되게 한다. `compose.deploy.yaml`의 이미지 태그는 릴리스 스크립트나 CI에서 `pyproject.toml` 값을 읽어 채우거나, 최소한 릴리스 체크리스트에 "3곳 동기화"를 명시한다.

### 3.2 [High] 런타임 의존성 대부분이 버전 상한/범위 없이 선언됨

**문제.** [`pyproject.toml:21-37`](../../pyproject.toml#L21-L37)의 `dependencies` 목록 중 `torch`/`torchvision`(`>=` 하한만 있음)과 `pydantic`(`>=2,<3`)을 제외한 나머지 9개 — `numpy`, `pandas`, `pyarrow`, `typer`, `pyyaml`, `timm`, `jinja2`, `opencv-python-headless`, `pillow`, `decord`, `tqdm`, `matplotlib` — 는 **어떤 버전 제약도 없다.**

**왜 중요한가.** 이 프로젝트의 핵심 가치 제안 중 하나가 "재현 가능한 raw dump/report"(§2 표, `SoftwareX_SSAT_positioning_and_submission_checklist.md` §5.3)인데, `pip install ssat`을 미래 시점에 실행하면 지금 테스트를 통과시킨 조합과 다른 `numpy`/`pandas` 메이저 버전이 설치될 수 있다. 특히 `numpy`는 2.x 전환에서 여러 하위 라이브러리의 ABI가 깨진 선례가 있고, `pyarrow`의 Parquet 스키마 직렬화 방식도 메이저 버전 간 달라질 수 있어 — 이 도구의 핵심 산출물인 raw dump의 장기 재현성에 직접 영향을 준다. Alpha 상태에서 모든 의존성을 엄격히 pin할 필요는 없지만, **현재는 "어떤 조합에서 검증했는지"를 선언하는 장치가 전혀 없다.**

**조치안.** 전면적인 상한 고정보다는, CI가 실제로 통과시킨 조합을 어딘가에 기록하는 것이 우선이다: (1) `requirements.txt`에 CI에서 검증된 정확한 버전을 `pip freeze` 스냅샷으로 별도 lock 파일(예: `requirements-lock.txt`)로 남기거나, (2) 최소한 알려진 breaking-change 이력이 있는 `numpy`/`pandas`/`pyarrow`에는 `<메이저+1` 형태의 상한을 추가한다. 두 방법 중 하나를 선택하되, README나 `INSTALLATION.md`에 "이 조합에서 검증됨"이라는 문장을 근거와 함께 남기는 것이 SoftwareX/JOSS 심사에서 특히 잘 보이는 부분이다.

### 3.3 [Medium] 정적 분석 도구(린트/타입체크)가 전혀 구성되어 있지 않음

**문제.** `ruff`, `mypy`, `black`, `flake8` 중 어느 것도 `pyproject.toml`에 설정 섹션이 없고, CI(`ci.yml`)의 `test`/`clean-install` job 어디에도 포함되지 않으며, `CONTRIBUTING.md`의 "Code style" 절도 docstring/comment 컨벤션만 규정할 뿐 자동 검사 도구를 전혀 언급하지 않는다.

**왜 중요한가.** 코드 자체는 타입 힌트가 광범위하게 쓰이고 있고(`from __future__ import annotations`가 일관되게 쓰임, 104/142 파일에 반환 타입 힌트 존재) 스타일도 상당히 일관돼 있어 — **이미 타입체커를 통과할 준비가 상당 부분 되어 있는 코드베이스**다. 그런데 이를 강제하는 자동화가 없으면, 기여자가 늘어나는 시점(SoftwareX 게재 이후 외부 PR이 들어올 가능성이 있는 시점)부터 일관성이 조용히 무너지기 시작한다. 지금은 저자 한 명이 스스로 규율을 지키고 있어 드러나지 않을 뿐이다.

**조치안.** 최소 구성으로 `ruff`(lint + format, 별도 `black` 불필요)를 `pyproject.toml`의 `[tool.ruff]`에 추가하고, CI `test` job에 `ruff check .` 한 줄을 추가한다. `mypy`는 이 코드베이스 규모(142개 파일)에서 처음부터 strict 모드로 붙이면 기존 코드에 대한 대량의 조정이 필요해질 수 있으므로, 우선 `ruff`만 CI 게이트로 넣고 `mypy`는 `dev` extras에 추가해 로컬에서 선택적으로 돌리는 정도로 시작하는 편을 권한다.

### 3.4 [Medium] 태그된 릴리스가 현재 상태를 대표하지 못함

**문제.** `git tag`에는 `v0.1.0`(2026-08-20 15:05:24 KST) 하나뿐이며, 이는 HEAD로부터 22개 커밋 뒤처져 있다. 그 22개 커밋에는 real-dataset case study 전체 6-run 매트릭스, reproducibility demo, benchmark 결과, 이번 세션에서 다룬 Phase 6 문서화 작업 등 — README와 `SoftwareX_SSAT_positioning_and_submission_checklist.md`가 프로젝트의 핵심 증거로 제시하는 내용 상당수가 포함된다. `CITATION.cff`의 `version: 0.1.0` / `date-released: "2026-08-20"`도 같은 시점을 가리킨다.

**왜 중요한가.** 논문이나 GitHub README가 "이 버전을 인용하라"고 안내할 시점에 `v0.1.0` 태그를 체크아웃하면, 논문이 실제로 근거로 삼은 real-dataset 결과나 reproducibility demo 스크립트가 존재하지 않는 훨씬 이전 상태가 나온다. 이는 재현성 주장 자체를 무력화할 수 있는 문제다.

**조치안.** SoftwareX 제출 직전(Phase 6 마무리 시점)에 새 태그(예: `v0.2.0` 또는 제출용으로 `v1.0.0`)를 찍고, `CITATION.cff`의 `version`/`date-released`를 그 시점에 맞춰 갱신한다. Zenodo 등 DOI 아카이브를 사용할 계획이라면 이 태그가 그 아카이브 스냅샷의 기준점이 되므로, 릴리스 노트에 "이 태그부터 real-dataset case study와 reproducibility demo가 포함됨"을 명시하면 심사자가 확인하기 쉬워진다.

### 3.5 [Low] `docs/internal/report_layout_improve/`가 의사결정 결과 없이 남아 있음

**문제.** [`docs/internal/report_layout_improve/`](report_layout_improve/)에는 세 가지 리포트 레이아웃 시안(`report_layout_A_interpretation_first.html`, `_B_question_driven.html`, `_C_behavioral_fingerprint.html`)과 검토 의견(`AGENTS_OPINION_1.md`)이 2026-08-20에 커밋된 채로 남아 있다. README([README.md](../../README.md))의 덤프 레이아웃 설명은 `report/report.html`과 `report/report_question_driven.html` 두 파일을 실제 산출물로 명시하므로, 이름으로 미루어 시안 B("question_driven")가 채택되어 실제 구현(`ssat/report/html_renderer.py`의 `_REPORT_TEMPLATE_B`)으로 이어진 것으로 보인다.

**왜 중요한가.** 그러나 이 디렉터리 안에는 "B가 채택되었고 A/C는 기각되었다"는 결론을 담은 파일이 없다 — 세 시안이 여전히 동등하게 검토 중인 것처럼 보인다. 나중에 이 문서를 처음 보는 사람(구현자 본인이라도 몇 달 뒤)은 A/C가 왜 코드에 반영되지 않았는지, 혹은 아직 반영 대기 중인지 판단할 근거가 없다.

**조치안.** 이 디렉터리에 결론을 담은 짧은 `DECISION.md`(또는 `README.md`) 하나를 추가해 "B가 `report_question_driven.html`로 채택됨, A/C는 참고용으로 보존" 정도의 한두 문단만 남기면 해결된다. 코드 변경이 필요 없는 가장 저렴한 조치다.

### 3.6 [Low] `CHANGELOG.md` 부재

**문제.** 저장소 어디에도 `CHANGELOG.md`(또는 동등한 릴리스 노트 파일)가 없다. `git log`의 커밋 메시지 품질은 높지만(Conventional Commits 스타일이 일관되게 지켜짐), 이는 "이번 릴리스에서 무엇이 바뀌었는지"를 사용자 관점에서 요약해 주지 않는다.

**왜 중요한가.** §3.4의 재태깅과 맞물려, 다음 릴리스부터라도 `CHANGELOG.md`를 시작하면 "v0.1.0 이후 무엇이 추가/변경/수정되었는가"를 심사자와 사용자에게 한눈에 보여줄 수 있다. 지금 당장 과거분을 소급 작성할 필요는 없다.

**조치안.** `Keep a Changelog` 형식으로 파일을 새로 만들고, §3.4의 다음 태그부터 `Added`/`Changed`/`Fixed` 섹션을 채워 나가기 시작한다.

---

## 4. 우선순위 요약

| # | 항목 | 심각도 | 예상 작업량 | 위치 |
|---|---|---|---|---|
| 3.1 | 버전 문자열 3중 관리 | High | 작음(반나절 미만) | `pyproject.toml`, `ssat/__init__.py`, `compose.deploy.yaml` |
| 3.2 | 의존성 버전 미고정 | High | 중간(검증 필요) | `pyproject.toml`, `requirements.txt` |
| 3.3 | 정적 분석 도구 부재 | Medium | 중간(ruff 도입 + 기존 코드 조정) | `pyproject.toml`, `.github/workflows/ci.yml`, `CONTRIBUTING.md` |
| 3.4 | 릴리스 태그가 최신 상태를 대표 못함 | Medium | 작음(제출 직전 1회 작업) | git tag, `CITATION.cff` |
| 3.5 | report_layout_improve 결정 미문서화 | Low | 매우 작음 | `docs/internal/report_layout_improve/` |
| 3.6 | CHANGELOG 부재 | Low | 작음 | 신규 `CHANGELOG.md` |

3.1과 3.2는 코드 로직을 건드리지 않으면서도 "재현 가능한 연구 소프트웨어"라는 이 프로젝트의 핵심 주장(§1, `SoftwareX_SSAT_positioning_and_submission_checklist.md` §16)을 가장 직접적으로 뒷받침하는 항목이라 우선순위를 High로 매겼다.

---

## 5. 기존 내부 문서와의 관계

이 저장소의 "성숙도/품질" 관련 내부 문서는 이제 세 개이며, 서로 다른 축을 본다:

- [`SoftwareX_SSAT_positioning_and_submission_checklist.md`](SoftwareX_SSAT_positioning_and_submission_checklist.md) — **연구/포지셔닝 축**. 왜 이 소프트웨어가 필요한가, 기존 도구(Captum, RobustBench 등) 대비 무엇을 주장할 것인가, 논문에서 피해야 할 표현. 코드 세부사항은 다루지 않는다.
- [`CODE_STRUCTURE_MODULARITY_ANALYSIS_v1.md`](CODE_STRUCTURE_MODULARITY_ANALYSIS_v1.md) — **내부 설계/리팩터링 축**. 책임 분리, 확장성, 코드 중복. `ssat/` 패키지 내부 구조에 집중하며 `experiments/`는 범위 밖으로 명시적으로 제외한다.
- 본 문서(`SOFTWARE_MATURITY_AUDIT_v1.md`) — **패키징/릴리스/외부 신뢰성 축**. 실제로 빌드·테스트를 돌려 확인한 사실과, 다른 연구자가 이 소프트웨어를 설치·인용·재현하려 할 때 처음 마주치는 지점(버전, 의존성, 태그, 정적 분석)에 집중한다.

세 문서 모두 "제출 직전 브랜치에 넣기에는 회귀 위험이 큰 구조 변경"과 "문서/설정만으로 정직하게 처리 가능한 항목"을 구분해 다루고 있으므로, 실제 구현자는 이번 Phase에서는 본 문서의 §3.1/§3.4/§3.5(회귀 위험 없음)부터 먼저 처리하고, §3.2/§3.3은 검증 시간이 필요한 항목으로 별도 일정에 배치하는 것을 권한다.

---

*이 문서는 2026-08-24 기준 `phase0-softwarex-prep` 브랜치에서, 컨테이너 내부 실제 테스트 실행(1093 passed) 및 quickstart 재현 결과를 근거로 작성되었으며, 코드 변경은 포함하지 않는다.*

---

## 6. 처리 현황 (2026-08-24 추가)

작성 직후 같은 세션에서 §4의 High 항목 두 개를 처리했다. Medium/Low 항목(§3.3~§3.6)은 여전히 미착수 상태로 남아 있다.

- **처리됨 — §3.1 (버전 문자열 3중 관리)**: [`ssat/__init__.py`](../../ssat/__init__.py)가 더 이상 `__version__`을 리터럴로 하드코딩하지 않고, `importlib.metadata.version("ssat")`로 설치된 패키지 메타데이터(즉 `pyproject.toml`의 `[project].version`)를 조회하도록 바꿨다. 패키지가 설치되지 않은 상태로 소스에서 직접 실행되는 드문 경우를 위해 `PackageNotFoundError` 폴백(`"0.0.0+unknown"`)을 남겼다. `compose.deploy.yaml`의 이미지 태그는 `pyproject.toml`을 자동으로 읽어올 방법이 없어 완전히 없앨 수는 없었지만, `local/ssat:${SSAT_IMAGE_TAG:-0.1.0}`로 바꿔 릴리스 시 이 파일을 편집하지 않고도 태그를 오버라이드할 수 있게 했고, 주석으로 `pyproject.toml`이 canonical source임을 명시했다. `pip install --no-deps -e .` 재설치 후 `ssat.__version__ == "0.1.0"`을 직접 확인했고, `docker compose -f compose.deploy.yaml config --quiet`가 여전히 통과함을 확인했다.
- **처리됨 — §3.2 (의존성 버전 미고정)**: `torch`/`torchvision`(의도적으로 하한만 유지 — CUDA wheel 선택 유연성을 위한 것으로 `requirements.txt`의 기존 주석이 설명함)과 `pydantic`(이미 범위 지정됨)을 제외한 9개 의존성(`numpy`, `pandas`, `pyarrow`, `typer`, `pyyaml`, `timm`, `jinja2`, `opencv-python-headless`, `pillow`, `decord`, `tqdm`, `matplotlib`) 모두에 `~=<검증된 major.minor>` (PEP 440 compatible-release) 범위를 추가했다 — `pyproject.toml`과 `requirements.txt` 양쪽에 동일하게 반영. 값은 추측이 아니라, 1093개 테스트를 통과시킨 컨테이너에서 `pip show`로 직접 조회한 실제 설치 버전(예: `numpy==2.3.2` → `numpy~=2.3`)에서 가져왔다. 변경 후 (1) 설치된 각 버전이 새 specifier를 만족하는지 `packaging.requirements`로 개별 검증했고, (2) 전체 테스트 스위트를 재실행해 1093 passed를 다시 확인했다.
- **미착수 — §3.3~§3.6**: 정적 분석 도구 도입(ruff/mypy), 릴리스 재태깅, `report_layout_improve/` 의사결정 문서화, `CHANGELOG.md` 신설은 각각 별도 검토·작업 시간이 필요하다고 판단해 이번 라운드에서는 다루지 않았다. §4의 우선순위 표는 여전히 유효하다.

