"""Additive, one-time device presets. Never overwrite user revisions or revive trash."""
import json
import uuid

from .schema import Hardware, Platform

ASCEND = 'https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/昇腾950%20NPU架构白皮书.pdf'
A100 = 'https://www.nvidia.com/en-us/data-center/a100/'
A100_40 = 'https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf'
AMPERE = 'https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/'
SOURCE_SHEET = '用户提供《推荐硬件对比.xlsx》Sheet1!F2:F8（2026-10-26核对）'


def metric(value, unit, source, *, dense=None, note='', evidence='official'):
    return dict(value=value,unit=unit,source=source,dense=dense,note=note,evidence=evidence if value is not None else 'unknown')


def device_presets():
    for capacity,form,bw in [(40,'PCIe',1555),(40,'SXM',1555),(80,'PCIe',1935),(80,'SXM',2039)]:
        name=f'NVIDIA A100 {capacity}GB {form}'
        source=A100 if capacity==80 else A100_40
        hw=dict(name=name,vendor='NVIDIA',metrics={
            'matrix_fp16':metric(312,'TFLOPS',source,dense=True,note='稠密 Tensor Core；不是含2:4稀疏的624 TFLOPS'),
            'matrix_bf16':metric(312,'TFLOPS',source,dense=True),
            'matrix_tf32':metric(156,'TFLOPS',source,dense=True),
            'vector_fp32':metric(19.5,'TFLOPS',source,note='非Tensor FP32峰值'),
            'vector_fp16':metric(78,'TFLOPS',AMPERE,note='非Tensor FP16指令峰值，不代表所有逐元素算子吞吐'),
            'vector_bf16':metric(39,'TFLOPS',AMPERE,note='非Tensor BF16指令峰值'),
            'memory_capacity':metric(capacity,'GB',source),
            'memory_bandwidth':metric(bw,'GB/s',source)},
            note='完整GPU规格，不适用于MIG切片。单卡峰值不等于实测吞吐；FP32矩阵不能直接使用TF32档位。')
        platform=dict(name=f'A100 {capacity}GB {form} 单设备建模平台',
                      note='公开设备规格模板。CPU、内存条、软件和实装设备数量未确认，不从原表未注明SKU的A100主机自动套用。建模范围为一颗GPU。')
        yield f'a100-{capacity}-{form.lower()}-v1',hw,platform
    for cores,fp16,tf32,vec16,vec32 in [(28,378,189,47,23),(32,432,216,54,27),(36,486,243,60,30)]:
        hw=dict(name=f'950DT {cores} Cube',vendor='Ascend',metrics={
            'matrix_fp16':metric(fp16,'TFLOPS',ASCEND+'#page=14',dense=True),
            'matrix_bf16':metric(fp16,'TFLOPS',ASCEND+'#page=14',dense=True),
            'matrix_hf32':metric(tf32,'TFLOPS',ASCEND+'#page=14',dense=True,note='白皮书表3-1标为TF32；本工具沿用CANN HF32矩阵模式字段，需确认实际数学模式与精度'),
            'vector_fp16':metric(vec16,'TFLOPS',ASCEND+'#page=14'),
            'vector_bf16':metric(vec16,'TFLOPS',ASCEND+'#page=14'),
            'vector_fp32':metric(vec32,'TFLOPS',ASCEND+'#page=14'),
            'cube_count':metric(cores,'count',ASCEND+'#page=13'),
            'vector_count':metric(cores*2,'count',ASCEND+'#page=13'),
            'memory_bandwidth':metric(4000,'GB/s',ASCEND+'#page=14',note='4 TB/s按十进制换算；不是互联带宽'),
            'memory_capacity':metric(None,'GB',ASCEND+'#page=14',note='白皮书列出144/96GB，但未给出与各Cube档位一一对应关系，按实际SKU补齐')},
            note='白皮书表3-1独立Cube/Vector峰值，不使用Cube+Vector合计算力。具体SKU、内存容量与运行频率需结合实机核验。')
        platform=dict(name=f'950DT {cores} Cube 单设备建模平台',note='单颗NPU建模范围，不代表整机配置；主机信息未确认。')
        if cores==28:
            platform.update(name='950DT Server 28 Cube 单设备建模平台',cpu_model='Kunpeng 950 7592C',cpu_count=2,
                cores_per_cpu=64,cpu_mhz=2300,dimm_count=24,dimm_gb=64,memory_rated_mts=4800,
                note=SOURCE_SHEET+'。原表378 TFLOPS矩阵档位对应28 Cube参考配置，不代表所有950DT Server SKU；Vector FP32采用白皮书23而非原表23.5。card_count=1仅为当前单设备建模范围。')
        yield f'950dt-{cores}-v1',hw,platform


def install_presets(store):
    """Install a hardware/platform pair atomically; a seed key survives deletion/rename."""
    from .store import now
    for key,hardware,platform in device_presets():
        with store.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            if c.execute('SELECT 1 FROM builtin_seeds WHERE key=?',(key,)).fetchone():
                continue
            hwid=uuid.uuid5(uuid.NAMESPACE_URL,'workload-modeler/hardware/'+key).hex
            pid=uuid.uuid5(uuid.NAMESPACE_URL,'workload-modeler/platform/'+key).hex
            hw=Hardware.model_validate(hardware).model_dump()
            p=Platform.model_validate(dict(platform,hardware_id=hwid,hardware_revision=1)).model_dump()
            for kind,entry_id,value in [('hardware',hwid,hw),('platform',pid,p)]:
                c.execute('INSERT INTO revisions VALUES(?,?,?,?,?)',(kind,entry_id,1,now(),json.dumps(value,ensure_ascii=False)))
            c.execute('INSERT INTO builtin_seeds VALUES(?)',(key,))
