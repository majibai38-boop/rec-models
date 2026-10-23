"""Typed contracts: unknown specs stay None; empirical fallback ratios are explicit assumptions."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


METRIC_UNITS = {
    **{f"matrix_{p}": "TFLOPS" for p in ("fp16", "bf16", "fp32", "tf32", "hf32")},
    **{f"vector_{p}": "TFLOPS" for p in ("fp16", "bf16", "fp32")},
    "memory_bandwidth": "GB/s", "memory_capacity": "GB",
    "l2_bandwidth": "GB/s", "launch_latency": "us",
    "cube_count": "count", "vector_count": "count",
}


class Metric(StrictModel):
    value: float | None = Field(default=None, gt=0)
    unit: str
    evidence: Literal["official", "measured", "estimated", "user", "unknown"] = "unknown"
    source: str = Field(default="", max_length=2000)
    note: str = Field(default="", max_length=2000)
    dense: bool | None = None

    @model_validator(mode="after")
    def valid_evidence(self):
        if self.value is not None and (self.evidence == "unknown" or not self.source.strip()):
            raise ValueError("数值必须标注来源与证据类型；未知规格请留空")
        return self


class Hardware(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    vendor: Literal["NVIDIA", "Ascend", "Other"]
    metrics: dict[str, Metric] = Field(default_factory=dict)
    note: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def units(self):
        for key, value in self.metrics.items():
            if key not in METRIC_UNITS or value.unit != METRIC_UNITS[key]:
                raise ValueError(f"规格 {key} 的单位必须是 {METRIC_UNITS.get(key, '受支持的字段')}")
            if key.endswith("_count") and value.value is not None and not value.value.is_integer():
                raise ValueError("核数必须为整数")
        return self


class Platform(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    hardware_id: str
    hardware_revision: int = Field(ge=1)
    card_count: int = Field(default=1, ge=1, le=1024)
    cpu_model: str = Field(default="", max_length=200)
    cpu_count: int | None = Field(default=None, ge=1)
    cores_per_cpu: int | None = Field(default=None, ge=1)
    cpu_mhz: float | None = Field(default=None, gt=0)
    dimm_count: int | None = Field(default=None, ge=1)
    dimm_gb: float | None = Field(default=None, gt=0)
    memory_rated_mts: float | None = Field(default=None, gt=0)
    memory_configured_mts: float | None = Field(default=None, gt=0)
    topology: str = Field(default="未提供", max_length=2000)
    software: str = Field(default="未提供", max_length=2000)
    note: str = Field(default="", max_length=4000)


class Scope(StrictModel):
    side: Literal["gpu", "npu"] = "gpu"
    pid: str = ""
    step: str = ""
    all_steps: bool = False
    device: str = ""

    @model_validator(mode="after")
    def exclusive(self):
        if self.all_steps and self.step:
            raise ValueError("单个 Step 和全采集范围不能同时选择")
        return self


class Assumptions(StrictModel):
    source_compute_efficiency: float = Field(default=.6, gt=0, le=1)
    target_compute_efficiency: float = Field(default=.6, gt=0, le=1)
    source_memory_efficiency: float = Field(default=.7, gt=0, le=1)
    target_memory_efficiency: float = Field(default=.7, gt=0, le=1)
    source_launch_us: float = Field(default=3, ge=0, le=10000)
    target_launch_us: float = Field(default=3, ge=0, le=10000)
    source_fp32_mode: Literal["unknown", "fp32", "tf32", "hf32"] = "unknown"
    target_fp32_mode: Literal["unknown", "fp32", "tf32", "hf32"] = "unknown"
    precision_equivalence_confirmed: bool = False
    allow_estimated_specs: bool = False
    fixed_gap_serial_step: bool = False


class Calibration(StrictModel):
    signature: str = Field(min_length=1, max_length=300)
    source_us: float = Field(gt=0)
    target_us: float = Field(gt=0)
    evidence: str = Field(min_length=8, max_length=2000)


class ScenarioRatios(StrictModel):
    conservative: float = Field(default=.7, gt=0, le=1000)
    baseline: float = Field(default=1., gt=0, le=1000)
    optimistic: float = Field(default=1.3, gt=0, le=1000)

    @model_validator(mode="after")
    def ordered(self):
        if not self.conservative <= self.baseline <= self.optimistic:
            raise ValueError("源/目标耗时倍率须满足：保守 ≤ 基准 ≤ 乐观")
        return self


class EmpiricalAssumptions(StrictModel):
    enabled: bool = True
    cube_compute_fraction: float = Field(default=.5, ge=0, le=1)
    short_kernel_us: float = Field(default=5., ge=0, le=10000)
    source_matrix_mode: Literal['fp16','bf16','fp32','tf32','hf32'] = 'fp16'
    target_matrix_mode: Literal['fp16','bf16','fp32','tf32','hf32'] = 'fp16'
    matrix_ratio: float | None = Field(default=None, gt=0, le=1000)
    bandwidth_ratio: float | None = Field(default=None, gt=0, le=1000)
    hardware_factors: ScenarioRatios = Field(default_factory=ScenarioRatios)
    missing_spec_ratio: ScenarioRatios = Field(default_factory=ScenarioRatios)
    fused_ratio: ScenarioRatios = Field(default_factory=ScenarioRatios)
    short_ratio: ScenarioRatios = Field(default_factory=lambda: ScenarioRatios(conservative=.5,baseline=1,optimistic=1.3))
    transfer_ratio: ScenarioRatios = Field(default_factory=ScenarioRatios)
    communication_ratio: ScenarioRatios = Field(default_factory=ScenarioRatios)
    unknown_ratio: ScenarioRatios = Field(default_factory=lambda: ScenarioRatios(conservative=.5,baseline=1,optimistic=1.5))
    gap_ratio: ScenarioRatios = Field(default_factory=lambda: ScenarioRatios(conservative=1,baseline=1,optimistic=1))


class AnalysisRequest(StrictModel):
    name: str = Field(default="新建建模分析", min_length=1, max_length=120)
    source_platform_id: str
    source_platform_revision: int = Field(ge=1)
    target_platform_id: str
    target_platform_revision: int = Field(ge=1)
    scope: Scope = Field(default_factory=Scope)
    assumptions: Assumptions = Field(default_factory=Assumptions)
    empirical: EmpiricalAssumptions = Field(default_factory=EmpiricalAssumptions)
    calibrations: list[Calibration] = Field(default_factory=list, max_length=1000)
    workload: str = Field(default="", max_length=4000, description="Batch、序列长度、训练/推理与软件条件")

    @model_validator(mode="after")
    def unique_calibrations(self):
        keys = [c.signature for c in self.calibrations]
        if len(keys) != len(set(keys)):
            raise ValueError("同一调用签名只能提供一个校准记录")
        return self
