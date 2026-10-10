from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from profiling_tools.compare import CompareConfig, compare_profiles
from profiling_tools.manual_mapping import apply_links, catalog, rank_candidate
from profiling_tools.matcher import match_ops
from profiling_tools.report import build_npu_kernel_gpu_mapping_rows
from profiling_tools.tests.test_compare import complete
from profiling_tools.tests.test_logical_calls import op, profile
from profiling_tools.tests.test_scope_recovery import strong


def nested_fixture(root, count=1):
    root = Path(root)
    for side, pid in [('gpu',1),('npu',3)]:
        events = [complete('ProfilerStep#1','user_annotation',0,100*count,pid,pid),
                  complete('nn.Module: IndexBlock_0','python_function',1,100*count-10,pid,pid)]
        for i in range(count):
            args = {'External id':2*i+1,'Input Dims':[[4],[],[4],[4],[]],
                    'Input type':['float','Scalar','long','float','Scalar']}
            events.append(complete('aten::index_add_','cpu_op',100*i+10,10,pid,pid,args))
            if side == 'npu':
                events.append(complete('aten::index_add','cpu_op',100*i+12,5,pid,pid,{**args,'External id':2*i+2}))
            events.append(complete('index_kernel','kernel',100*i+30,4 if side=='npu' else 2,pid+1,7,
                                  {'External id':2*i+(2 if side=='npu' else 1),'stream':7}))
        (root/f'{side}.json').write_text(json.dumps({'traceEvents':events}),encoding='utf-8')
    (root/'details.csv').write_text('Name,Type,Duration(us),Start Time(us),Stream ID,Device_id,Step ID\n'+
                                  ''.join(f'index_kernel,IndexAdd,4,{100*i+30},7,0,1\n' for i in range(count)),encoding='utf-8')
    return CompareConfig(str(root/'gpu.json'),str(root/'npu.json'),str(root/'details.csv'),str(root/'out'),gpu_step='1',npu_step='1')


class ManualMappingTest(unittest.TestCase):
    def data(self):
        gpu = profile('gpu',[op('aten::index_add_',0)],[(0,'indexKernel')])
        npu = profile('npu',[op('aten::index_add_',0),op('aten::index_add',1,parent=0)],[(1,'IndexAdd'),(1,'IndexAdd')])
        strong(gpu,npu)
        matches,_,_ = match_ops(gpu,npu,{})
        rows = build_npu_kernel_gpu_mapping_rows(gpu,npu,matches)
        return gpu,npu,matches,rows

    def test_nested_parent_is_top_candidate_and_owner_is_preserved(self):
        g,n,m,rows = self.data()
        c = catalog(g,n,m,rows)
        result = rank_candidate(c['targets'][0],c['candidates'][0])
        self.assertEqual(result['rank_score'],100)
        apply_links(g,n,rows,[{'id':'manual1','npu_indices':[0,1],'gpu_indices':[0],'reason':'verified parent'}])
        self.assertTrue(all(r['Comparable'] and not r['Direct Kernel Pair'] for r in rows))
        self.assertTrue(all(r['NPU CPU OP Index']==1 for r in rows))
        self.assertEqual(catalog(g,n,m,rows),{'targets':[],'candidates':[]})

    def test_reject_invalid_duplicate_mixed_or_occupied(self):
        for variant in ('invalid','duplicate','mixed','reused','mapped','device'):
            g,n,m,rows = self.data()
            link = {'id':'m','npu_indices':[0,1],'gpu_indices':[0],'reason':'verified'}
            links = [link]
            if variant=='invalid': link['gpu_indices']=[999]
            if variant=='duplicate': link['npu_indices']=[0,0]
            if variant=='mixed': rows[1]['OP Type']='Cast'
            if variant=='reused': links.append({**link,'id':'m2'})
            if variant=='mapped': rows[0]['Comparable']=True
            if variant=='device':
                import copy
                g.kernels.append(copy.deepcopy(g.kernels[0]));g.kernels[1].event.pid='9';link['gpu_indices']=[0,1]
            with self.subTest(variant=variant), self.assertRaises(ValueError): apply_links(g,n,rows,links)

    def test_full_reports_validate_and_source_changes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = nested_fixture(tmp)
            before = compare_profiles(cfg,manual_enabled=True)
            base = json.loads((Path(cfg.output_dir)/'run_manifest.json').read_text(encoding='utf-8'))
            c = json.loads((Path(cfg.output_dir)/'manual_catalog.json').read_text(encoding='utf-8'))
            self.assertEqual(rank_candidate(c['targets'][0],c['candidates'][0])['rank_score'],100)
            link={'id':'m','npu_indices':c['targets'][0]['indices'],'gpu_indices':c['candidates'][0]['indices'],'reason':'verified'}
            after=compare_profiles(replace(cfg,output_dir=str(Path(tmp)/'manual')),manual_enabled=True,manual_links=[link],manual_base=base)
            self.assertEqual(after['validation']['status'],'valid')
            self.assertEqual(after['gpu_accounting']['gpu_total_time_us'],before['gpu_accounting']['gpu_total_time_us'])
            self.assertEqual(after['gpu_accounting']['npu_total_time_us'],before['gpu_accounting']['npu_total_time_us'])
            self.assertEqual(after['comparison']['npu_first_comparable_kernel_count'],1)
            self.assertEqual(before['comparison']['npu_first_comparable_kernel_count'],0)
            base['producer']['source_sha256']='changed'
            with self.assertRaises(ValueError):
                compare_profiles(replace(cfg,output_dir=str(Path(tmp)/'bad')),manual_links=[link],manual_base=base)
