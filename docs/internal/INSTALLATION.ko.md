# 설치와 배포

## 개발 워크스페이스

```bash
docker compose up -d --build region-sensitivity-workspace
docker compose exec region-sensitivity-workspace pip install --no-deps -e .
docker compose exec region-sensitivity-workspace pytest -q
docker compose exec region-sensitivity-workspace ssat --help
```

Dev Container도 같은 Dockerfile과 `/workspace` bind mount를 사용합니다.

## 로컬 Python 3.11+

시스템/CUDA 환경에 맞는 PyTorch가 설치될 수 있는지 먼저 확인합니다.

```bash
bash scripts/install_deps.sh
pip install -r requirements.txt
pip install --no-deps -e .
ssat --version
```

의존성의 기준은 `requirements.txt`이며 `pyproject.toml`은 패키지와 console script만
정의합니다.

## 배포 Compose

기본 배포 설정은 config와 data를 읽기 전용으로, dump를 named volume으로 연결합니다.

```bash
docker compose -f compose.deploy.yaml build ssat
docker compose -f compose.deploy.yaml run --rm ssat \
  run /config/deploy/quickstart.yaml --output /dumps/quickstart
docker compose -f compose.deploy.yaml run --rm ssat \
  inspect /dumps/quickstart
```

실제 데이터 위치는 `SSAT_DATA_DIR`, 설정 디렉터리는 `SSAT_CONFIG_DIR`로 바꿀 수
있습니다. 기본 Compose는 GPU를 요청합니다. CPU 전용 호스트에서는 GPU 설정을 제거한
override 파일을 사용하고 adapter의 `device`를 `cpu`로 지정하세요.

pretrained selector는 framework cache 또는 네트워크를 사용할 수 있습니다. 완전한
오프라인 실행에는 `weights: null`, `pretrained: false` 또는 로컬 checkpoint를
사용하세요.
