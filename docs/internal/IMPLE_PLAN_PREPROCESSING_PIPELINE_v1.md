# 구현 계획서 (v1 선언적 전처리 파이프라인 레지스트리)
## Spatial Sensitivity Audit Toolkit — Preprocessing TRANSFORMS Registry

> 본 계획서는 세션 중 논의된 방향을 구현 계획으로 구체화한 것이다. MMAction2/MMCV처럼
> `{"type": "Resize", ...}` 형태의 설정 목록을 넘기면 그에 맞는 전처리 callable을 조립해 주는
> registry 방식을 SSAT에 도입한다. 다만 이번 계획은 세션에서 이미 확정한 다음 범위 제한을 그대로
> 따른다.
>
> 1. **preprocessing 계층에만 국소 도입한다.** Source의 decode/frame-sampling
>    ([ssat/core/source/video_folder.py](../ssat/core/source/video_folder.py)의 고정
>    `uniform_frame_indices`)과 Perturb의 region corruption
>    ([ssat/core/perturb/dispatch.py](../ssat/core/perturb/dispatch.py))은 이번 계획에서 건드리지
>    않는다. 즉 이전 세션에서 예시로 든 `DecordInit`/`SampleFrames`(디코드 전
>    프레임 선택)/`DecordDecode`/`ApplyCorruptionAfterDecode`는 **이번 계획의 산출물이 아니다** —
>    이 계획이 실제로 구현하는 것은 이미 `(T,H,W,C)` uint8로 디코드된 배치 위에서 동작하는
>    post-decode 전처리 단계뿐이다.
> 2. **기존 `ssat/core/adapter/preprocessing.py`/`DeclarativePreprocessor`는 그대로 유지한다.** 새
>    registry는 그 자리를 대체하지 않고, 같은 `Preprocessor` 계약을 만족하는 새 구현체
>    (`PipelinePreprocessor`)를 어댑터가 선택적으로 쓸 수 있게 추가하는 것뿐이다. 기존 op(`op:
>    resize` 등)와 새 op(`type: Resize` 등)는 서로 다른 이름공간으로 공존한다.
> 3. **내장 op은 결정적 변형만 다룬다.** 최소 `SampleFrames`(랜덤 없이 중앙 clip만 선택),
>    `Resize`, `CenterCrop`, `TenCrop`, `ToFloat`, `Normalize`, `FormatShape`를 제공하고, 증강용
>    무작위 변형(RandomCrop, RandomFlip, ColorJitter 등)은 이 도구의 목적(감사 재현성)과 맞지 않아
>    포함하지 않는다.
>
> 전제: 직전 세션에서 확인한 아키텍처 사실 — `SourceProviderRegistry`/`AdapterProviderRegistry`
> ([ssat/core/source/provider.py](../ssat/core/source/provider.py),
> [ssat/core/adapter/provider.py](../ssat/core/adapter/provider.py))가 이미 이 저장소 관용구의
> registry 패턴(인스턴스 로컬, `name`/`config_model` + `parse()`/`build()` + `default_*_registry()`
> 팩토리)을 확립해 두었고, 새 TRANSFORMS registry는 이를 그대로 미러링한다. `Preprocessor` ABC
> ([ssat/core/adapter/preprocessor.py](../ssat/core/adapter/preprocessor.py))와
> `PreprocessingSpec`/`fingerprint_payload`/`validate_mask`는 세 번째 구현체를 추가해도 계약이
> 그대로 유지되도록 이미 열려 있다.

---

## 0. 결론 요약

| 우선순위 | 무엇을 | 왜 | 이번 계획의 범위 |
|---|---|---|---|
| 1 | `ssat/core/adapter/transform_registry.py`(신규) — `BaseTransform`/`TransformRegistry`/`Pipeline`/`build_pipeline` | `preprocessing.py`의 `parse_preprocessing_ops`가 고정 if/elif라 확장하려면 그 파일을 계속 고쳐야 함(직전 세션 진단). `SourceProviderRegistry`와 같은 모양의 열린 registry로 이 지점만 연다 | 순수 신규 파일. 아직 아무 곳에서도 import되지 않음 |
| 2 | `ssat/core/adapter/transforms.py`(신규) — 내장 7종 op + `default_transform_registry()` + `PipelinePreprocessor` | `SampleFrames`(중앙 clip)/`Resize`/`CenterCrop`/`TenCrop`/`ToFloat`/`Normalize`/`FormatShape` 최소 세트. `Resize`/`ToFloat`/`Normalize`는 기존 `preprocessing.py` 엔진에 위임(수치 로직 중복 금지), `SampleFrames`/`TenCrop`/`FormatShape`만 새 로직 | `Preprocessor` 계약을 만족하는 `PipelinePreprocessor`까지 포함, 아직 어댑터에는 배선하지 않음 |
| 3 | `TorchvisionProviderConfig`/`TorchvisionVideoProviderConfig`에 `pipeline_config` 필드 + `TorchvisionAdapter`/`TorchvisionVideoAdapter` 배선 | 사용자가 YAML `adapter.pipeline_config: [...]`로 실제로 이 registry를 쓸 수 있게 하는 연결점 | 기존 `preprocessing`(flat op list) 필드는 그대로 유지하고 상호 배타로만 추가. Timm은 이번 계획에서 제외(§8) |
| 4 | 문서화(`CONFIG_REFERENCE.md`) | "커스텀 transform 등록" 절을 기존 "커스텀 source provider 등록" 절과 대칭으로 추가 | 문서만, 신규 동작 없음 |

우선순위 1·2는 서로 독립적이지 않고 순차적이다(2가 1을 소비). 3은 1·2가 끝나야 시작할 수 있다.
4는 3과 병행 가능.

---

## 1. 현재 구현 상태 대비 격차

| # | 항목 | 상태 | 근거 |
|---|---|---|---|
| 1 | preprocessing 쪽에 "이름 문자열 → 클래스" registry가 없다 | **격차 — 우선순위 1** | [`ssat/core/adapter/preprocessing.py:156-198`](../ssat/core/adapter/preprocessing.py)의 `parse_preprocessing_ops`는 `known_types` 튜플 + if/elif 체인이다. 새 op을 추가하려면 `PreprocessingOp` TypeAlias, `parse_preprocessing_ops`, `apply_preprocessing`, `transform_mask_geometry`, `_ops_payload` 다섯 곳을 매번 고쳐야 한다. |
| 2 | `SampleFrames`/`TenCrop`처럼 시간축·다중 crop을 다루는 op이 아예 없다 | **격차 — 우선순위 2** | 기존 엔진은 `Resize`/`CenterCrop`/`ToFloat`/`Normalize`/`ChannelsFirst`/`SqueezeTime` 6종뿐이며 전부 기하·광도 변형이다(시간축 서브샘플링, 다중 crop 없음). |
| 3 | `TorchvisionVideoProviderConfig`에는 preprocessing override 필드 자체가 없다 | 격차 — 우선순위 3에서 함께 해소 | [`ssat/core/adapter/provider.py:80-105`](../ssat/core/adapter/provider.py)의 `TorchvisionVideoProviderConfig`는 `resize_size`/`crop_size`/`mean`/`std` 4개 스칼라뿐이고, `TorchvisionAdapter`의 `preprocessing_ops`([provider.py:50](../ssat/core/adapter/provider.py))에 대응하는 override가 없다. |
| 4 | `TorchvisionVideoAdapter.predict()`가 전처리 결과 레이아웃을 하드코딩으로 가정한다 | 격차 아님 — 지켜야 할 제약(§7 결정 사항) | [`torchvision_video_adapter.py:173-183`](../ssat/core/adapter/torchvision_video_adapter.py)의 `predict()`는 `self._preprocessor.transform_batch(batch)`가 항상 `(B,T,C,H,W)`를 반환한다고 가정하고 `.permute(0,2,1,3,4)`로 `(B,C,T,H,W)`를 직접 만든다. 새 파이프라인이 이 자리를 대체할 때 이 가정을 깨면 안 된다. |
| 5 | Timm 어댑터에는 전처리 override 지점이 원래 없다 | 격차 아님 — 범위 밖(§8) | [`timm_adapter.py`](../ssat/core/adapter/timm_adapter.py)는 `create_transform(**data_config, is_training=False)`로 만든 `TimmPreprocessor`만 쓰고 override 인자 자체가 없다. 이번 계획에서 새로 열지 않는다. |

---

## 2. 기술 스택과 의존성 방침

**신규 하드 의존성 없음.** `numpy`/`opencv-python-headless`(cv2)는 이미
[`requirements.txt`](../requirements.txt)의 최상위 의존성이고, 새 op은 전부 이 둘과 표준 라이브러리로
충분하다.

**위치 원칙.** `ssat/core/adapter/preprocessor.py`(제네릭 `Preprocessor` ABC)와
`preprocessing.py`(구체적인 op + `DeclarativePreprocessor`)가 이미 "계약/구현 분리"를 이 디렉터리
안에서 확립해 두었다. 새 코드도 같은 분리를 그대로 반복한다.

- `ssat/core/adapter/transform_registry.py`(신규) — 데이터셋·프레임워크에 무관한 제네릭 registry
  엔진(`preprocessor.py`에 대응).
- `ssat/core/adapter/transforms.py`(신규) — 구체적인 내장 op 7종 + `default_transform_registry()` +
  `PipelinePreprocessor`(`preprocessing.py`에 대응).

새 하위 패키지(`ssat/core/adapter/transforms/` 디렉터리)를 만들지 않는다 — 지금 분량이라면 기존
2-파일 관례를 그대로 따르는 편이 탐색하기 쉽고, `ssat/core/runtime/pipeline.py`(런타임 오케스트레이션,
전혀 다른 모듈)와 이름이 겹치는 `.../pipeline/` 디렉터리를 만들지 않아도 된다.

---

## 3. 설계 상세

### 3.1 `ssat/core/adapter/transform_registry.py`(신규): 제네릭 registry 엔진

`SourceProviderRegistry`([provider.py:155-227](../ssat/core/source/provider.py))와 동일한 오류 처리
관례를 따르되, discriminator 필드명은 이 세션에서 합의한 대로 MMCV 관례를 그대로 살려 `type`을 쓴다
(다른 provider들의 `kind`/`provider`와 이름공간이 겹치지 않으므로 문제 없음).

```python
class TransformError(ValueError):
    """Indicate invalid transform registration, configuration, or execution."""


class BaseTransform(ABC):
    """Apply one deterministic pixel/mask operation inside a registry-built pipeline.

    Attributes:
        type_name: Registry key this transform is registered under by default.
        mask_supported: Whether apply_mask() preserves the (H,W)/(T,H,W) mask
            contract 1:1. False for ops that break that geometry (e.g. TenCrop's
            batch expansion) — the owning Pipeline reports
            PreprocessingSpec.mask_transform_available=False when any step sets
            this to False.
    """

    type_name: ClassVar[str]
    mask_supported: ClassVar[bool] = True

    @abstractmethod
    def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
        """Transform one (B, T, H, W, C) batch."""

    def apply_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Apply matching geometry to a (H, W) or (T, H, W) mask.

        Default is identity — correct for photometric-only ops. Geometry ops
        must override this; ops with mask_supported=False never have this
        called (the Pipeline short-circuits to None instead).
        """

        return mask


class TransformRegistry:
    """Instance-local name registry with explicit transform-class registration."""

    def __init__(self) -> None:
        self._transforms: dict[str, type[BaseTransform]] = {}

    @property
    def names(self) -> tuple[str, ...]: ...

    def register(self, transform_cls: type[BaseTransform], *, name: str | None = None) -> None:
        """Register one BaseTransform subclass under name or its own type_name."""
        ...  # duplicate/empty-name/not-a-BaseTransform-subclass -> TransformError, mirrors
            # SourceProviderRegistry.register()

    def register_module(
        self, name: str | None = None
    ) -> Callable[[type[BaseTransform]], type[BaseTransform]]:
        """Decorator sugar for register() — this is the "register_module" the user asked for.

        Usage:
            TRANSFORMS = default_transform_registry()

            @TRANSFORMS.register_module()
            class MyOp(BaseTransform):
                type_name = "MyOp"
                ...
        """

        def _decorator(transform_cls: type[BaseTransform]) -> type[BaseTransform]:
            self.register(transform_cls, name=name)
            return transform_cls

        return _decorator

    def build(self, cfg: Mapping[str, Any]) -> BaseTransform:
        """Build one transform from a {"type": ..., **kwargs} mapping.

        kwargs are passed straight into the registered class's constructor —
        each built-in is a frozen dataclass whose __post_init__ validates its
        own fields (same convention as preprocessing.py's Resize/CenterCrop),
        so no separate pydantic config_model is required per op. A TypeError
        from a bad kwarg is wrapped into TransformError with the type name.
        """
        ...


@dataclass(frozen=True, slots=True)
class Pipeline:
    """Compose registry-built transforms into one ordered callable (mmcv Compose)."""

    config: tuple[dict[str, Any], ...]
    transforms: tuple[BaseTransform, ...]

    @property
    def mask_supported(self) -> bool:
        return all(t.mask_supported for t in self.transforms)

    def __call__(self, batch: NDArray[Any]) -> NDArray[Any]:
        result = batch
        for transform in self.transforms:
            result = transform.apply_batch(result)
        return np.ascontiguousarray(result)

    def apply_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        if not self.mask_supported:
            raise TransformError("pipeline contains a transform without mask support")
        result = mask
        for transform in self.transforms:
            result = transform.apply_mask(result)
        return result

    def describe(self) -> str:
        return " -> ".join(str(step.get("type")) for step in self.config) or "identity"


def build_pipeline(
    config: Sequence[Mapping[str, Any]], registry: TransformRegistry
) -> Pipeline:
    """Build one Pipeline from a MMAction2-style step-list config.

    Raises:
        TransformError: If config is empty, a step is malformed, or a step's
            "type" is not registered.
    """
    ...
```

**테스트(`tests/unit/test_transform_registry.py`, 신규).**
- `TransformRegistry.register()`가 중복 이름/빈 이름/`BaseTransform` 아닌 값을 거부(§`SourceProviderRegistry`와 동일한 3케이스).
- `register_module()` 데코레이터가 `register()`와 동일하게 동작(데코레이트된 클래스가 그대로 반환되는지 포함).
- `build()`가 `type` 생략/미등록 `type`(등록된 이름 목록을 에러 메시지에 포함)/생성자 `TypeError`를 각각 `TransformError`로 감싸는지.
- `build_pipeline([], registry)`가 빈 설정을 거부.
- 커스텀 `BaseTransform` 서브클래스를 등록해 `build_pipeline`으로 실제 실행되는 end-to-end 케이스(§3.6에서 문서화할 확장 경로의 유일한 실증 근거).

**성공 조건.** 위 테스트 통과. 신규 파일이라 기존 전체 스위트에 영향 없음.

### 3.2 `ssat/core/adapter/transforms.py`(신규): 내장 op 7종 + `PipelinePreprocessor`

기존 `preprocessing.py`의 수치 엔진(`Resize`/`CenterCrop`/`ToFloat`/`Normalize`/`apply_preprocessing`/
`transform_mask_geometry`)을 **재구현하지 않고 위임**한다 — 새 이름공간이 필요한 4종(`Resize`,
`ToFloat`, `Normalize`, `CenterCrop`)은 얇은 wrapper로만 존재하고, 실제 픽셀 연산은 여전히
`preprocessing.py` 한 곳에서만 일어난다.

```python
from ssat.core.adapter import preprocessing as _decl
from ssat.core.adapter.transform_registry import BaseTransform, TransformError, TransformRegistry
```

**`SampleFrames`(신규 로직) — 결정적 중앙 clip만 지원.**

```python
@dataclass(frozen=True, slots=True)
class SampleFrames(BaseTransform):
    """Select clip_len frames centered on the batch's existing time axis.

    v1 supports only the deterministic center-clip case (num_clips=1,
    frame_interval=1) — no random shift, no multi-clip TSN-style sampling.
    Frame decode/uniform sampling already happened at the Source stage
    (VideoFolderSource.load()); this only re-slices the already-decoded
    (B, T, H, W, C) batch's T axis.
    """

    type_name: ClassVar[str] = "SampleFrames"
    clip_len: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.clip_len, bool) or not isinstance(self.clip_len, int) or self.clip_len <= 0:
            raise ValueError("clip_len must be a positive int")

    def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
        return _center_slice(batch, self.clip_len, axis=1)  # (B,T,H,W,C) -> T at axis 1

    def apply_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        if mask.ndim != 3:  # (H,W) broadcasts across every frame -- no T axis to slice
            return mask
        return _center_slice(mask, self.clip_len, axis=0)  # (T,H,W) -> T at axis 0


def _center_slice(array: NDArray[Any], clip_len: int, *, axis: int) -> NDArray[Any]:
    total = array.shape[axis]
    if clip_len > total:
        raise TransformError(f"clip_len={clip_len} exceeds available frames={total}")
    start = (total - clip_len) // 2
    index = [slice(None)] * array.ndim
    index[axis] = slice(start, start + clip_len)
    return array[tuple(index)]
```

`num_clips`/`frame_interval`을 v1에서 아예 받지 않는다(생성자에 없으므로 config에 넘기면
`build()`가 `TypeError`→`TransformError`로 즉시 거부) — "일부만 지원하되 조용히 무시"가 아니라
"지원하지 않는 파라미터는 구성 시점에 실패"하는 쪽을 택한다.

**`Resize`(위임 + 관례 변환) — mmaction 스타일 `scale`을 기존 엔진의 `size`로 변환.**

```python
@dataclass(frozen=True, slots=True)
class Resize(BaseTransform):
    """MMAction2-style scale=[w, h] (one axis may be -1) resize.

    ⚠️ mmcv/mmaction2 convention is (width, height); the underlying
    preprocessing.Resize.size convention is (height, width). _coerce_scale
    converts explicitly -- this axis-order mismatch is the single highest-risk
    correctness bug in this file (see §7 risk table) and must have a
    non-square regression test.
    """

    type_name: ClassVar[str] = "Resize"
    scale: int | tuple[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_op", _decl.Resize(_coerce_scale(self.scale)))

    def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
        return _decl.apply_preprocessing(batch, (self._op,))

    def apply_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        return _decl.transform_mask_geometry(mask, (self._op,))


def _coerce_scale(scale: int | Sequence[int]) -> _decl.Size:
    """Convert an mmaction (w, h)-with-optional--1 pair, or a bare short edge int."""

    if isinstance(scale, int) and not isinstance(scale, bool):
        return scale
    w, h = scale
    if (w == -1) == (h == -1):
        raise ValueError("Resize.scale must set exactly one of (w, h) to -1, or pass a bare int")
    return h if w == -1 else w if h == -1 else (h, w)  # short-edge int form when one axis is -1
```

**`CenterCrop`/`ToFloat`/`Normalize`** — 같은 패턴의 얇은 위임(각각 `_decl.CenterCrop`/`_decl.ToFloat`/
`_decl.Normalize` 한 개짜리 op 튜플을 `apply_preprocessing`/`transform_mask_geometry`에 넘김).
`CenterCrop`은 정사각 `crop_size: int`만 v1에서 받는다(직전 세션 예시가 정사각이었고, 비정사각까지
받으면 `Resize`와 동일한 축 순서 위험이 또 생기므로 필요해질 때 별도로 연다 — §7).

**`TenCrop`(신규 로직) — 5-crop × 수평 flip.**

```python
@dataclass(frozen=True, slots=True)
class TenCrop(BaseTransform):
    """Corners + center crop, each mirrored horizontally -- 10 deterministic views.

    Not augmentation (no randomness) but does expand the batch axis 10x:
    (B,T,H,W,C) -> (B*10,T,H,W,C), crop index varying fastest within each
    original sample's block (sample0's 10 crops, then sample1's 10 crops, ...).
    Averaging the resulting 10x predictions back down to one score per sample
    is NOT handled here -- that is the adapter/caller's responsibility and is
    out of scope for this plan (§7).
    """

    type_name: ClassVar[str] = "TenCrop"
    mask_supported: ClassVar[bool] = False  # batch expansion breaks the 1:1 mask contract
    crop_size: int

    def __post_init__(self) -> None:
        if isinstance(self.crop_size, bool) or not isinstance(self.crop_size, int) or self.crop_size <= 0:
            raise ValueError("crop_size must be a positive int")

    def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
        # pad like preprocessing._center_crop_batch when undersized, then take
        # 5 fixed (top,left) windows on the padded canvas + horizontal flip
        # each; stack along a new axis and merge it into B.
        ...

    def apply_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        raise TransformError("TenCrop does not support mask geometry (mask_supported=False)")
```

**`FormatShape`(신규 로직, 하지만 기존 op 재조합) — v1은 두 값만 지원.**

```python
@dataclass(frozen=True, slots=True)
class FormatShape(BaseTransform):
    """Final layout conversion -- delegates entirely to existing ChannelsFirst/SqueezeTime.

    Only two input_format values are supported in v1, chosen to exactly match
    what the two existing adapters already consume -- not the full mmaction2
    set:
      - "NCHW":  squeeze T (requires T==1, e.g. after SampleFrames(clip_len=1))
                 then channels-first -> (B,C,H,W). Matches TorchvisionAdapter/
                 DeclarativePreprocessor's existing image path.
      - "NTCHW": channels-first, T kept -> (B,T,C,H,W). Matches what
                 TorchvisionVideoAdapter.predict() already expects before its
                 own hardcoded .permute(0,2,1,3,4) to (B,C,T,H,W) (§7 risk).

    A literal mmaction "NCTHW" (channel-before-time, no adapter-side permute
    needed) is deliberately NOT offered in v1 -- see §7/§8.
    """

    type_name: ClassVar[str] = "FormatShape"
    input_format: Literal["NCHW", "NTCHW"]

    def apply_batch(self, batch: NDArray[Any]) -> NDArray[Any]:
        ops: tuple[Any, ...] = (_decl.ChannelsFirst(),)
        if self.input_format == "NCHW":
            ops += (_decl.SqueezeTime(),)  # raises ValueError if T != 1 -- existing, reused behavior
        return _decl.apply_preprocessing(batch, ops)

    # apply_mask: default identity is correct -- transform_mask_geometry already
    # ignores ChannelsFirst/SqueezeTime (channel/time reshuffling doesn't move
    # pixels), so FormatShape doesn't override apply_mask.
```

**`default_transform_registry()` + `PipelinePreprocessor`.**

```python
def default_transform_registry() -> TransformRegistry:
    """Return a fresh registry containing only the v1 built-in 7 transforms."""

    registry = TransformRegistry()
    for transform_cls in (SampleFrames, Resize, CenterCrop, TenCrop, ToFloat, Normalize, FormatShape):
        registry.register(transform_cls)
    return registry


class PipelinePreprocessor(Preprocessor):
    """Bridge a registry-built Pipeline into the existing Preprocessor ABC.

    Sits alongside DeclarativePreprocessor/TorchvisionPreprocessor/
    TimmPreprocessor as a fourth Preprocessor implementation -- no change to
    the ABC or its three existing implementations.
    """

    def __init__(
        self,
        config: Sequence[Mapping[str, Any]],
        *,
        registry: TransformRegistry | None = None,
    ) -> None:
        self._pipeline = build_pipeline(config, registry or default_transform_registry())
        self._spec = PreprocessingSpec(
            kind="pipeline",
            deterministic=True,
            description=self._pipeline.describe(),
            fingerprint=fingerprint_payload(list(self._pipeline.config)),
            mask_transform_available=self._pipeline.mask_supported,
        )

    def describe(self) -> PreprocessingSpec:
        return self._spec

    def transform_batch(self, batch: NDArray[np.uint8]) -> NDArray[Any]:
        return self._pipeline(batch)

    def transform_mask(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_] | None:
        validate_mask(mask)
        if not self._pipeline.mask_supported:
            return None
        return self._pipeline.apply_mask(mask)
```

`registry` 인자는 Python API로 직접 어댑터를 만드는 호출자가 자신의 커스텀 transform이 등록된
registry를 주입할 수 있게 하는 유일한 통로다(§3.6).

**테스트(`tests/unit/test_transforms.py`, 신규).**
- `SampleFrames`: 짝수/홀수 T에서의 중앙 슬라이스 좌표 손계산 대조, `clip_len > T` 에러, `(T,H,W)`
  마스크가 배치와 동일한 인덱스로 슬라이스되는지, `(H,W)` 마스크는 변경되지 않는지.
- `Resize`: **비정사각** 입력(예: H=100,W=200)에 `scale=[-1, 50]`을 줘서 축이 뒤바뀌지 않는지(§7
  최우선 리스크의 회귀 테스트), 정수 short-edge 폼과의 등가성.
- `CenterCrop`/`ToFloat`/`Normalize`: 기존 `preprocessing.py` 단위 테스트와 동일한 입력으로 돌려
  `_decl` 직접 호출 결과와 일치하는지(순수 위임 확인 — 새 수치 로직이 없음을 증명).
- `TenCrop`: crop 10개의 위치·순서·flip 여부, `apply_mask` 호출 시 `TransformError`, 패딩이 필요한
  undersized 입력.
- `FormatShape`: `"NCHW"`가 `T=1`에서만 성공하고 `T>1`이면 기존 `SqueezeTime`의 `ValueError`를 그대로
  전파하는지, `"NTCHW"`의 출력 shape.
- `default_transform_registry()`가 정확히 7개 이름을 등록.
- `PipelinePreprocessor`: `describe()`/`transform_batch()`/`transform_mask()` 왕복,
  `TenCrop`을 포함한 파이프라인에서 `mask_transform_available=False`이고 `transform_mask()`가
  `None`을 반환(§`IdentityPreprocessor`와 동일한 계약), fingerprint가 파라미터 변경에 민감함
  (`test_declarative_preprocessing_fingerprint_is_canonical_and_sensitive`와 동일한 방식).

**성공 조건.** 위 테스트 통과. 이 파일도 아직 아무 어댑터에서 import되지 않으므로 회귀 없음.

### 3.3 `ssat/core/adapter/provider.py` 배선 — `pipeline_config` 필드

**`TorchvisionProviderConfig`**([provider.py:36-77](../ssat/core/adapter/provider.py))에 필드 추가:

```python
pipeline_config: tuple[dict[str, Any], ...] | None = None

@model_validator(mode="after")
def validate_pipeline_config(self) -> TorchvisionProviderConfig:
    if self.pipeline_config is not None:
        if self.preprocessing is not None:
            raise ValueError("preprocessing and pipeline_config are mutually exclusive")
        if not self.pipeline_config:
            raise ValueError("pipeline_config must not be empty when provided")
        from ssat.core.adapter.transform_registry import build_pipeline
        from ssat.core.adapter.transforms import default_transform_registry

        build_pipeline(self.pipeline_config, default_transform_registry())
    return self
```

기존 `validate_preprocessing`과 정확히 같은 이유(§`TorchvisionProviderConfig.validate_preprocessing`
docstring)로 config 로드 시점에 실패시킨다 — 실행 중간에야 알 수 없는 op을 발견하지 않도록.

**`TorchvisionVideoProviderConfig`**([provider.py:80-105](../ssat/core/adapter/provider.py))에도 동일한
`pipeline_config` 필드 + validator 추가(이 config는 지금 override 필드가 아예 없었으므로 이번이 첫
확장 지점).

**`TorchvisionProvider.build()`/`TorchvisionVideoProvider.build()`** — 각각
`pipeline_config=config.pipeline_config`를 어댑터 생성자 호출에 추가.

**`TorchvisionAdapter.__init__`**([torchvision_adapter.py:110-178](../ssat/core/adapter/torchvision_adapter.py))
에 `pipeline_config: Sequence[Mapping[str, Any]] | None = None`,
`transform_registry: TransformRegistry | None = None`(Python API 전용, §3.6) 인자 추가, 선택 사다리를
한 단 더 확장:

```python
self._preprocessor: Preprocessor = (
    PipelinePreprocessor(pipeline_config, registry=transform_registry)
    if pipeline_config is not None
    else DeclarativePreprocessor(preprocessing_ops)
    if preprocessing_ops is not None
    else TorchvisionPreprocessor(preprocessing)
)
```

**`TorchvisionVideoAdapter.__init__`**([torchvision_video_adapter.py:43-109](../ssat/core/adapter/torchvision_video_adapter.py))
도 동일하게 `pipeline_config`/`transform_registry` 인자를 받아, 주어지면 지금 하드코딩된
`DeclarativePreprocessor([Resize(...), CenterCrop(...), ToFloat(), Normalize(...), ChannelsFirst()])`
대신 `PipelinePreprocessor(pipeline_config, registry=transform_registry)`를 쓴다.

> ⚠️ **반드시 문서와 테스트로 못박아야 할 제약**(§1 항목4, §7): 이 어댑터의 `predict()`는
> `self._preprocessor.transform_batch(batch)`가 `(B,T,C,H,W)`를 반환한다고 하드코딩으로 가정하고
> `.permute(0,2,1,3,4)`를 직접 한다. 따라서 이 어댑터에 쓰는 `pipeline_config`는 **반드시**
> `FormatShape(input_format="NTCHW")`로 끝나야 한다 — `"NCHW"`로 끝내면 `predict()`의 permute가
> 잘못된 shape에 걸려 명확한 에러로 즉시 실패한다(조용히 틀린 추론이 아니라 즉시 실패이므로 안전은
> 하지만, 왜 실패하는지는 문서화가 필요하다).

**테스트.**
- `tests/unit/test_adapter_provider.py`(기존 파일 확장): `pipeline_config`/`preprocessing` 동시 지정
  거부, 빈 `pipeline_config` 거부, 알 수 없는 `type` 거부(모두 config 로드 시점 에러), 유효한
  `pipeline_config`로 `TorchvisionProviderConfig`/`TorchvisionVideoProviderConfig`가 정상 파싱.
- `tests/unit/test_model_adapter.py`(기존 파일 확장): `TorchvisionAdapter(pipeline_config=...)`로
  합성 이미지 배치 end-to-end(§3.2의 `test_declarative_adapter_applies_pixels_but_only_geometry_to_mask`와
  대응하는 케이스), `TorchvisionVideoAdapter(pipeline_config=[..., FormatShape(NTCHW)])`로 합성 클립
  end-to-end, 그리고 **`FormatShape(NCHW)`로 끝나는 `pipeline_config`를 video 어댑터에 주면 `predict()`가
  명확한 에러로 실패**함을 확인하는 회귀 테스트(§ 위 경고의 실증).

**성공 조건.** 신규 테스트 통과 + 기존 전체 스위트 회귀 0건(기존 `preprocessing`/`preprocessing_ops`
경로는 완전히 그대로 남아 있으므로 회귀 위험은 낮다).

### 3.4 문서화(`docs/CONFIG_REFERENCE.md`)

**"Adapter" 절**에 `pipeline_config` 예시 추가:

```yaml
adapter:
  provider: torchvision
  model_name: resnet50
  pipeline_config:
    - type: Resize
      scale: [-1, 256]
    - type: CenterCrop
      crop_size: 224
    - type: ToFloat
    - type: Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    - type: FormatShape
      input_format: NCHW
```

```yaml
adapter:
  provider: torchvision_video
  model_name: r3d_18
  pipeline_config:
    - type: SampleFrames
      clip_len: 8
    - type: Resize
      scale: [-1, 256]
    - type: CenterCrop
      crop_size: 224
    - type: ToFloat
    - type: Normalize
      mean: [0.43216, 0.394666, 0.37645]
      std: [0.22803, 0.22145, 0.216989]
    - type: FormatShape
      input_format: NTCHW   # NCHW는 안 됨 -- predict()가 (B,T,C,H,W)를 기대함(§3.3 경고)
```

**신규 "커스텀 transform 등록" 절**을 기존 "커스텀 source provider 등록" 절
([CONFIG_REFERENCE.md:175-234](../docs/CONFIG_REFERENCE.md))과 대칭으로 추가 — `TransformRegistry`/
`BaseTransform`/`default_transform_registry()`를 Python API에서 직접 써서 커스텀 op을 등록하고
`TorchvisionAdapter(pipeline_config=..., transform_registry=...)`로 주입하는 예시. "이 확장 지점은
Python API 전용이며 CLI는 항상 기본 registry만 쓴다"는 동일한 비대칭을 명시한다(§3.6).

### 3.5 `ssat/core/adapter/__init__.py` 노출

기존 [`__init__.py`](../ssat/core/adapter/__init__.py)의 지연 `__getattr__` 관례를 그대로 따라
`TransformError`/`BaseTransform`/`TransformRegistry`/`Pipeline`/`build_pipeline`/
`default_transform_registry`/`PipelinePreprocessor`/`SampleFrames`/`Resize`/`CenterCrop`/`TenCrop`/
`ToFloat`/`Normalize`/`FormatShape`를 `__all__` + `__getattr__`에 추가(기존 `preprocessing.py`의
7개 이름과 동일한 지연 로딩 패턴).

### 3.6 커스텀 transform 등록 경로 (Python API 전용)

Application 계층(`AuditApplication`)에는 이번 계획에서 `transform_registry` 생성자 인자를 추가하지
**않는다** — 그렇게 하면 `ssat/application/*` 배선까지 건드리게 되어 "preprocessing에만 국소 도입"
범위를 벗어난다. 대신 `AdapterProvider`/`AdapterProviderRegistry`를 거치지 않고 어댑터 클래스를 직접
생성하는 기존 Python API 경로(`TorchvisionAdapter(...)`를 코드에서 직접 호출)에 새 `transform_registry`
인자를 노출하는 것으로 충분하다:

```python
from ssat.core.adapter import (
    BaseTransform,
    TorchvisionAdapter,
    default_transform_registry,
)


class MySharpen(BaseTransform):
    type_name = "MySharpen"
    mask_supported = False  # 광도 변형이라 마스크 지오메트리엔 영향 없음 -> True로 둬도 되지만
                             # 명시적으로 선언해 둔다

    def apply_batch(self, batch):
        ...  # 커스텀 numpy 연산


registry = default_transform_registry()
registry.register(MySharpen)

adapter = TorchvisionAdapter(
    "resnet50",
    pipeline_config=[{"type": "Resize", "scale": [-1, 256]}, {"type": "MySharpen"}],
    transform_registry=registry,
)
```

YAML/CLI(`ssat run`/`ssat estimate`)는 항상 `default_transform_registry()`만 쓴다 — source/adapter
provider 확장과 동일한 비대칭(§`CONFIG_REFERENCE.md`의 "Python API 전용" 각주들)이며, 이번 계획에서
새로 발명하는 예외가 아니다.

---

## 4. 구현 순서

각 단계는 **구현 → 테스트 작성 → 성공 조건 충족** 순으로 진행한다.

### 단계 0. `transform_registry.py` 골격 (§3.1)
**작업.** `BaseTransform`/`TransformRegistry`/`Pipeline`/`build_pipeline` 신규 작성.
**테스트.** §3.1의 5개 케이스.
**성공 조건.** 신규 테스트 통과, 아직 어디서도 import되지 않아 회귀 영향 없음.

### 단계 1. `transforms.py` 내장 op + `PipelinePreprocessor` (§3.2)
**작업.** `SampleFrames`/`Resize`/`CenterCrop`/`TenCrop`/`ToFloat`/`Normalize`/`FormatShape`/
`default_transform_registry()`/`PipelinePreprocessor`.
**테스트.** §3.2의 케이스 전부 — 특히 `Resize`의 비정사각 축-순서 회귀 테스트는 이 단계에서 반드시
포함(§7 최우선 리스크).
**성공 조건.** 신규 테스트 통과, 회귀 없음(단계0과 동일한 이유).

### 단계 2. Provider config 필드 + validator (§3.3 앞부분)
**작업.** `TorchvisionProviderConfig`/`TorchvisionVideoProviderConfig`에 `pipeline_config` +
`validate_pipeline_config`, `TorchvisionProvider.build()`/`TorchvisionVideoProvider.build()` 배선.
**테스트.** `tests/unit/test_adapter_provider.py` 확장분.
**성공 조건.** 신규 테스트 통과 + 기존 전체 스위트 회귀 0건.

### 단계 3. 어댑터 생성자 배선 (§3.3 뒷부분)
**작업.** `TorchvisionAdapter`/`TorchvisionVideoAdapter`에 `pipeline_config`/`transform_registry` 인자.
**테스트.** `tests/unit/test_model_adapter.py` 확장분 — video 어댑터의 `FormatShape(NCHW)` 오사용
회귀 테스트 포함.
**성공 조건.** 신규 테스트 통과 + 기존 전체 스위트 회귀 0건. 이 단계가 이번 계획의 병목이다(§1 항목4
제약을 실제로 지키는지가 여기서 처음 검증됨).

### 단계 4. `__init__.py` 노출 (§3.5)
**작업/테스트/성공 조건.** 지연 로딩 이름 추가, 기존 `from ssat.core.adapter import ...` 스모크
테스트가 있다면 그대로 통과하는지 확인.

### 단계 5. 문서화 (§3.4, §3.6)
**작업.** `CONFIG_REFERENCE.md`의 "Adapter" 절 예시 + 신규 "커스텀 transform 등록" 절.
**성공 조건.** 문서 갱신 완료, 기존 링크 깨짐 없음, 문서에 실린 YAML 예시가 실제로 단계3 테스트에서
검증된 것과 동일해야 함(문서 따로/코드 따로가 되지 않도록).

---

## 5. 단계 간 의존

```
0 ──> 1 ──> 2 ──> 3 ──> 4
                    └──> 5
```

전부 선형 의존이다(직전 세션의 우선순위1/2·3 같은 병렬 트랙이 이번엔 없다) — 각 단계가 다음 단계가
가정하는 API 표면을 만들기 때문.

---

## 6. Application/CLI 배선 변화 요약

| 항목 | 이전 | 이후 |
|---|---|---|
| `TorchvisionProviderConfig` | `preprocessing: tuple[dict, ...] \| None` | 동일 + `pipeline_config: tuple[dict, ...] \| None`(상호 배타) |
| `TorchvisionVideoProviderConfig` | override 필드 없음 | `pipeline_config: tuple[dict, ...] \| None` 신규 |
| `TorchvisionAdapter.__init__` | `(..., preprocessing_ops=None)` | `(..., preprocessing_ops=None, pipeline_config=None, transform_registry=None)` |
| `TorchvisionVideoAdapter.__init__` | override 인자 없음 | `(..., pipeline_config=None, transform_registry=None)` |
| `AuditApplication`/`load_application_config` | 변경 없음 | 변경 없음 — 이번 계획은 Application 계층을 건드리지 않는다(§3.6) |
| CLI(`ssat run`/`estimate`) | 변경 없음 | 변경 없음 — YAML의 `adapter.pipeline_config`는 쓸 수 있지만(config 필드이므로), 커스텀 transform 등록은 여전히 Python API 전용 |

---

## 7. 리스크와 완화

| 리스크 | 완화 |
|---|---|
| `Resize.scale`의 mmaction `(w,h)` 관례와 기존 `preprocessing.Resize.size`의 `(h,w)` 관례가 뒤바뀌어 비정사각 입력에서 조용히 잘못된 크기로 리사이즈됨 | §3.2에서 `_coerce_scale`을 명시적으로 분리하고, 단계1 테스트에 **비정사각** 입력을 반드시 포함(§4 단계1) — 정사각 입력만으로는 축이 바뀌어도 테스트가 통과해 버림 |
| `TorchvisionVideoAdapter.predict()`의 하드코딩된 `.permute(0,2,1,3,4)`와 `pipeline_config`의 `FormatShape`가 어긋나면 사용자가 이해하기 어려운 에러를 만남 | v1은 `FormatShape`에 `"NCTHW"`를 아예 제공하지 않아 "NTCHW로 끝내야 한다"는 선택지를 하나로 좁히고, §4 단계3에 `FormatShape(NCHW)` 오사용이 즉시 명확히 실패하는 회귀 테스트를 필수 포함. 방어적 사전 검증(예: `__init__`에서 파이프라인의 마지막 op을 검사)은 결합도를 높이므로 v1에서는 추가하지 않고 §8로 이월 |
| `TenCrop`이 배치를 10배로 늘리는데 그 다음 추론/평균화 계층은 이를 모른 채 그대로 모델에 넣힘 | `mask_supported=False`로 최소한 마스크 오정합(가장 위험한 조용한 오류)은 막는다. 로짓 10개를 평균해 샘플당 1개로 되돌리는 것은 `ModelAdapter.infer`/`predict()` 영역이라 이번 계획 범위 밖임을 §3.2/§8에 명시 — `TenCrop`을 쓰려면 호출자가 직접 후처리해야 한다 |
| `transforms.py`의 `Resize`/`CenterCrop`/`ToFloat`/`Normalize`/`FormatShape`가 `preprocessing.py`의 동명 클래스와 다른 모듈에 있어 두 시스템을 혼동할 위험 | 각 클래스 docstring에 "위임 대상"을 명시하고(§3.2 스니펫에 이미 반영), `__init__.py`의 `__getattr__`가 두 이름공간을 분리해서 노출(§3.5) — `from ssat.core.adapter import Resize`(구 flat-op)와 `pipeline_config`의 `{"type": "Resize", ...}`(신규)는 이름은 같아도 서로 다른 곳에서 온다는 점을 CONFIG_REFERENCE.md에도 한 줄로 명시 |
| `SampleFrames`가 "디코드 전 프레임 선택"이 아니라 "이미 디코드된 배치의 시간축 재슬라이스"라는 점을 사용자가 mmaction2와 동일하다고 오해할 위험 | §0 서두와 `SampleFrames` docstring에 명시. Source 단계의 `num_frames`(디코드 시 균등 샘플링 개수)보다 `SampleFrames.clip_len`이 커지면 `clip_len > T` 에러로 즉시 드러나므로 조용한 오해로 이어지진 않음 |

---

## 8. 잔여 결정 사항의 처리 시점

| 항목 | 결정 시점 | 결정 내용 |
|---|---|---|
| Source 단계(`DecordInit`/디코드 전 `SampleFrames`/`DecordDecode`) 통합 | 이번 계획 범위 밖으로 확정 | §0에서 이미 확정 — decode/frame-sampling은 여전히 `VideoFolderSource.load()`의 고정 로직. 필요해지면 별도 계획서에서 source 단계까지 registry화할지(직전 세션 방향 문서의 옵션 B) 다시 논의 |
| `ApplyCorruptionAfterDecode` 훅 | 이번 계획 범위 밖으로 확정 | Perturb는 여전히 `ssat/core/perturb/dispatch.py`의 별도 스테이지. region corruption을 파이프라인 스텝으로 노출하려면 `results`에 mask/RNG/params를 실어 나르는 새 계약이 필요한데, 이는 이 계획의 "preprocessing 전용" 전제와 정면으로 배치됨 |
| `TenCrop`의 10배 배치를 평균화해 샘플당 1개 예측으로 되돌리는 로직 | 수요 기반, 이번 계획 범위 밖 | `ModelAdapter.infer`/각 어댑터의 `predict()`를 건드려야 하는 별도 설계 질문(§7) — 실제로 TenCrop 평가가 필요해지면 그때 |
| Timm 어댑터에 `pipeline_config` 확장 | 수요 기반, 이번 계획 범위 밖 | `TimmAdapter`는 원래 override 지점이 없었고(§1 항목5), timm 자체 프리셋(`create_transform`)과의 관계를 어떻게 정리할지가 새 설계 질문이라 별도로 다룬다 |
| `FormatShape`에 진짜 `"NCTHW"`(어댑터 쪽 permute 없이 바로 3D-conv 레이아웃) 추가 | 필요 시, 이번 계획 범위 밖 | 지금은 두 기존 어댑터가 요구하는 두 값(`NCHW`/`NTCHW`)만으로 충분. 어댑터 쪽에서 자체 permute를 하지 않는 새 프레임워크 어댑터가 생기면 그때 값 하나를 추가하는 정도로 충분히 확장 가능(등록형 `FormatShape`이므로 마이그레이션 부담 없음) |
| `AuditApplication(transform_registry=...)`로 Application 계층까지 커스텀 transform 주입을 넓힐지 | 수요 기반, 이번 계획 범위 밖 | §3.6에서 이번 계획은 어댑터 직접 생성(Python API)만 지원하기로 확정. CLI/YAML에서도 커스텀 transform이 필요해지면 source/adapter provider 확장과 함께 다루는 편이 일관적 |

---

## 9. 참고: 검토한 주요 파일 목록

`ssat/core/adapter/preprocessor.py`(ABC/`validate_mask`/`fingerprint_payload` 선례),
`ssat/core/adapter/preprocessing.py`(수치 엔진, 위임 대상),
`ssat/core/adapter/provider.py`(`TorchvisionProviderConfig`/`TorchvisionVideoProviderConfig`/
`AdapterProviderRegistry` 선례), `ssat/core/adapter/torchvision_adapter.py`,
`ssat/core/adapter/torchvision_video_adapter.py`(§1 항목4 `predict()` 하드코딩 permute),
`ssat/core/adapter/timm_adapter.py`(§8 범위 밖 확인), `ssat/core/adapter/__init__.py`(지연 로딩 관례),
`ssat/core/adapter/types.py`(`PreprocessingSpec`/`AdapterSpec`), `ssat/core/source/provider.py`(registry
패턴 원형), `ssat/core/source/video_folder.py`(§0 범위 제외 확인),
`ssat/core/perturb/dispatch.py`(§0/§8 범위 제외 확인), `docs/CONFIG_REFERENCE.md`,
`docs/IMPLE_PLAN_DATASET_INGESTION_v1.md`(이 계획서의 형식·관례 원형), `requirements.txt`.
