import json
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from profiling_web.ai_backends import AIConfig, BackendError, prepare_command, run_backend, session_result
from profiling_web.ai_evidence import build_evidence, validate_analysis, prompt_for
from profiling_web.tests.test_manual_links import ManualWebTest


def response(evidence):
    pair = evidence['candidate_pairs'][0]
    return json.dumps({'summary': '外层 CPU 调用需要人工核对。',
        'findings': [{'title': '嵌套调用差异', 'detail': '按相同外层调用位置核对 shape 和工作量。', 'evidence_ids': pair['evidence_ids']}],
        'proposals': [{k: pair[k] for k in ('target_id', 'candidate_id', 'evidence_ids')} |
                      {'reason': '核对外层 index_add_ 调用', 'confidence': 'medium'}],
        'limitations': ['样本不代表全部调用，AI 不证明语义等价。']}, ensure_ascii=False)


class AIWebTest(unittest.TestCase):
    setUp = ManualWebTest.setUp
    tearDown = ManualWebTest.tearDown
    new = ManualWebTest.new
    upload = ManualWebTest.upload
    inspect = ManualWebTest.inspect
    complete = ManualWebTest.complete
    execute = ManualWebTest.execute

    def setup_job(self):
        self.ident = self.complete()
        self.job = self.store.get(self.ident)
        self.base = f'/api/jobs/{self.ident}'
        self.ai = self.app.state.ai
        self.ai.config = AIConfig(True, {'codex': [sys.executable], 'deepseek': [sys.executable]}, {'codex': '', 'deepseek': ''}, 10)
        self.row = self.client.get(self.base+'/results').json()['items'][0]
        self.body = {'attempt': self.job['attempt'], 'provider': 'codex', 'question': '排查低覆盖率',
                     'share_evidence': True, 'scope': {k: self.row[k] for k in ('op_type','device_id','core_type')}}

    def finished(self, rid):
        deadline = time.monotonic()+10
        while time.monotonic()<deadline:
            result = self.client.get(self.base+'/ai-runs/'+rid).json()
            if result['status'] not in {'queued', 'running'}:
                return result
            time.sleep(.03)
        self.fail('AI queue did not finish')

    def start_fake(self, function=None):
        with patch.dict(os.environ, {'CODEX_API_KEY':'test-not-a-real-key'}), patch('profiling_web.ai_service.run_backend',
                side_effect=function or (lambda cfg,p,q,e,*_: response(e))):
            result=self.client.post(self.base+'/ai-runs',json=self.body)
            self.assertEqual(result.status_code,202,result.text)
            return self.finished(result.json()['id'])

    def test_analysis_batch_confirm_conservation_provenance_undo_and_stale(self):
        self.call_count=3
        self.setup_job()
        with patch('profiling_tools.parser._load_trace_events', side_effect=AssertionError('AI must use indexes')):
            result=self.start_fake()
        self.assertEqual(result['status'],'succeeded',result)
        self.assertEqual(self.store.get(self.ident)['manual'],{})
        proposal=result['result']['proposals'][0]
        seed={k:proposal[k] for k in ('target_id','candidate_id')}
        seed.update(attempt=self.job['attempt'],ai_run_id=result['id'],ai_proposal_id=proposal['id'])
        forged=self.client.post(self.base+'/manual-batch/preview',json={**seed,'candidate_id':'invented-id'})
        self.assertEqual(forged.status_code,422)
        self.assertEqual(self.client.post(self.base+'/manual-batch/preview',json={**seed,'ai_proposal_id':None}).status_code,422)
        preview=self.client.post(self.base+'/manual-batch/preview',json=seed)
        self.assertEqual(preview.status_code,200,preview.text)
        self.assertEqual(preview.json()['pair_count'],3)
        applied=self.client.post(self.base+'/manual-batch',json={**seed,'preview_token':preview.json()['preview_token'],'reason':'人工核对三次调用一致'})
        self.assertEqual(applied.status_code,202,applied.text)
        self.execute()
        now=self.store.get(self.ident)
        self.assertEqual(now['status'],'succeeded',now['error'])
        for field in ('gpu_total_time_us','npu_total_time_us'):
            self.assertEqual(self.job['summary']['gpu_accounting'][field],now['summary']['gpu_accounting'][field])
        links=now['manual']['links']
        self.assertTrue(all(l['ai_run_id']==result['id'] for l in links))
        self.assertTrue(self.client.get(self.base+'/ai-runs/'+result['id']).json()['stale'])
        self.assertEqual(self.client.post(self.base+'/manual-batch/preview',json=seed).status_code,409)
        self.assertEqual(self.client.post(self.base+f"/manual-links/{links[0]['batch_id']}/undo",json={'attempt':now['attempt']}).status_code,202)
        self.execute()
        self.assertEqual(self.store.get(self.ident)['manual']['links'],[])

    def test_validation_auth_failures_and_bounded_evidence(self):
        self.setup_job()
        self.assertEqual(self.client.get('/api/ai/config',headers={'Authorization':''}).status_code,401)
        self.assertEqual(self.client.post(self.base+'/ai-runs',json={**self.body,'share_evidence':False}).status_code,422)
        self.assertEqual(self.client.post(self.base+'/ai-runs',json={**self.body,'command':['cmd']}).status_code,422)
        self.ai.config.enabled=False
        self.assertEqual(self.client.post(self.base+'/ai-runs',json=self.body).status_code,503)
        self.ai.config.enabled=True
        result=self.start_fake(lambda *args: '{"secret":"invalid"}')
        self.assertEqual(result['status'],'failed')
        self.assertNotIn('secret',result['error'])
        self.assertIsNone(result['result'])
        evidence=result['evidence']
        self.assertLess(len(prompt_for(self.body['question'],evidence).encode()),130000)
        good=json.loads(response(evidence))
        good['proposals'][0]['candidate_id']='not-in-evidence'
        with self.assertRaises(ValueError):validate_analysis(json.dumps(good),evidence)
        good=json.loads(response(evidence));good['findings'][0]['evidence_ids']=['E9999']
        with self.assertRaises(ValueError):validate_analysis(json.dumps(good),evidence)
        good=json.loads(response(evidence));good['proposals'][0]['evidence_ids']=['E1']
        with self.assertRaises(ValueError):validate_analysis(json.dumps(good),evidence)

    def test_cancel_does_not_change_job_or_accept_late_result(self):
        self.setup_job()
        entered=threading.Event()
        def waiting(cfg,p,q,e,scratch,cancelled):
            entered.set();cancelled.wait(5)
            return response(e)
        with patch.dict(os.environ, {'CODEX_API_KEY':'test-not-a-real-key'}),patch('profiling_web.ai_service.run_backend',side_effect=waiting):
            created=self.client.post(self.base+'/ai-runs',json=self.body)
            rid=created.json()['id']
            self.assertTrue(entered.wait(5))
            self.assertEqual(self.client.post(self.base+'/ai-runs',json=self.body).status_code,409)
            self.assertEqual(self.client.delete(self.base+'?confirm=true').status_code,409)
            cancelled=self.client.post(self.base+'/ai-runs/'+rid+'/cancel')
            self.assertEqual(cancelled.json()['status'],'cancelled')
            self.ai.queue.join()
        self.assertIsNone(self.client.get(self.base+'/ai-runs/'+rid).json()['result'])
        self.assertEqual(self.store.get(self.ident)['attempt'],self.job['attempt'])
        other=self.new()
        self.assertEqual(self.client.get(f'/api/jobs/{other}/ai-runs/{rid}').status_code,404)
        self.assertEqual(self.client.delete(self.base+'/ai-runs/'+rid).status_code,200)

    def test_followup_context_and_exact_exclusion_totals(self):
        self.setup_job()
        first=self.start_fake()
        self.body['parent_id']=first['id']
        self.body['question']='继续分析 shape 的影响'
        second=self.start_fake()
        self.assertEqual(second['status'],'succeeded',second)
        self.assertEqual(second['evidence']['conversation'][0]['answer'],first['result']['summary'])
        plan=next(e['data'] for e in second['evidence']['items'] if e['kind']=='query_plan')
        self.assertEqual(plan['mode'],'shape')
        index=self.store.path(self.ident)/'attempts'/self.job['attempt']/'index.sqlite'
        evidence=build_evidence(self.job,index,{},exclude_types=['IndexAdd'])
        totals=next(e['data'] for e in evidence['items'] if e['kind']=='scope_totals')
        self.assertGreater(totals['before']['npu_time_us'],0)
        self.assertEqual(totals['after_exclusions']['npu_time_us'],0)
        other=self.new()
        self.assertEqual(self.client.post(f'/api/jobs/{other}/ai-runs',json=self.body).status_code,404)
        self.store.update(self.ident,attempt='f'*32)
        self.assertEqual(self.client.post(self.base+'/ai-runs',json=self.body).status_code,409)

    def test_trace_opt_in_reuse_invalidation_and_cleanup(self):
        self.setup_job()
        preview=self.client.get(self.base+'/ai-trace',params={'attempt':self.job['attempt']})
        self.assertEqual(preview.status_code,200,preview.text)
        self.assertEqual(len(preview.json()['files']),2)
        sent=[]
        def fake(cfg,p,q,e,scratch,cancelled,session=None,attachments=()):
            sent.append((session,attachments))
            (session['path']/'home').mkdir(exist_ok=True)
            return {'text':response(e),'backend_id':'session-test'}
        self.body['trace_mode']='attach'
        first=self.start_fake(fake)
        self.assertEqual(first['status'],'succeeded',first)
        self.assertTrue(first['trace_context']['reusable'])
        self.assertEqual(len(sent[0][1]),2)
        self.assertEqual(first['evidence']['trace']['attached_bytes'],preview.json()['total_bytes'])
        self.assertNotIn('traceEvents',json.dumps(first))
        self.body.update(parent_id=first['id'],trace_mode='reuse')
        second=self.start_fake(fake)
        self.assertEqual(second['status'],'succeeded',second)
        self.assertEqual(sent[1][1],())
        self.assertEqual(sent[1][0]['backend_id'],'session-test')
        self.assertEqual(sent[0][0]['path'],sent[1][0]['path'])
        self.assertEqual(second['evidence']['trace']['attached_bytes'],0)
        self.assertFalse(self.client.get(self.base+'/ai-runs/'+first['id']).json()['trace_context']['reusable'])
        with patch.dict(os.environ,{'CODEX_API_KEY':'fake','DEEPSEEK_API_KEY':'fake'}):
            self.assertEqual(self.client.post(self.base+'/ai-runs',json=self.body).status_code,409) # old parent
            self.body['parent_id']=second['id']
            self.assertEqual(self.client.post(self.base+'/ai-runs',json={**self.body,'provider':'deepseek'}).status_code,409)
        # Refusing Trace must not resume a context which contains it.
        self.body['trace_mode']='none'
        third=self.start_fake()
        self.assertEqual(third['status'],'succeeded',third)
        self.assertNotIn('trace',third['evidence'])
        # Mutation is detected before backend invocation; no silent resend.
        path=self.store.path(self.ident)/'attempts'/self.job['attempt']/'reports/gpu_trace.json'
        path.write_bytes(path.read_bytes()+b' ')
        self.body['trace_mode']='reuse'
        failed=self.start_fake(lambda *a,**k: self.fail('changed Trace must not be sent'))
        self.assertEqual(failed['status'],'failed')
        self.assertIn('内容已变化',failed['error'])
        for run in (first,second):
            self.assertEqual(self.client.delete(self.base+'/ai-runs/'+run['id']).status_code,200)
        self.assertFalse(sent[0][0]['path'].exists())

    def test_trace_limits_missing_and_invalid_sessions(self):
        self.setup_job()
        self.assertEqual(self.client.post(self.base+'/ai-runs',json={**self.body,'trace_mode':'reuse'}).status_code,422)
        self.assertEqual(self.client.post(self.base+'/ai-runs',json={**self.body,'trace_mode':'attach','share_evidence':False}).status_code,422)
        with patch.dict(os.environ,{'PROFILING_AI_TRACE_MAX_BYTES':'1024'}):
            self.assertEqual(self.client.post(self.base+'/ai-runs',json={**self.body,'trace_mode':'attach'}).status_code,413)
        path=self.store.path(self.ident)/'attempts'/self.job['attempt']/'reports/gpu_trace.json'
        path.unlink()
        self.assertEqual(self.client.get(self.base+'/ai-trace',params={'attempt':self.job['attempt']}).status_code,409)
        self.assertEqual(self.start_fake()['status'],'succeeded') # old results still support statistics

    def test_failed_trace_turn_cannot_be_resumed(self):
        self.setup_job()
        self.body['trace_mode']='attach'
        def fake(cfg,p,q,e,scratch,cancelled,session=None,attachments=()):
            (session['path']/'home').mkdir(exist_ok=True)
            return {'text':response(e),'backend_id':'session-test'}
        first=self.start_fake(fake)
        self.body.update(trace_mode='reuse',parent_id=first['id'])
        def failing(*args,**kwargs):
            raise BackendError('会话失败')
        failed=self.start_fake(failing)
        self.assertEqual(failed['status'],'failed',failed)
        parent=self.client.get(self.base+'/ai-runs/'+first['id']).json()
        self.assertFalse(parent['trace_context']['reusable'])


class BackendTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.config=AIConfig(True,{'codex':[sys.executable],'deepseek':[sys.executable]}, {'codex':'test-model','deepseek':'test-model'},10)

    def tearDown(self):self.temp.cleanup()

    def test_commands_are_fixed_private_and_strip_unrelated_secrets(self):
        for provider in ('codex','deepseek'):
            work=self.root/provider;work.mkdir()
            with patch.dict(os.environ,{'PROFILING_WEB_TOKEN':'private','AWS_SECRET_ACCESS_KEY':'private','CODEX_API_KEY':'fake','DEEPSEEK_API_KEY':'fake'}):
                command,env,answer=prepare_command(self.config,provider,work)
            self.assertNotIn('PROFILING_WEB_TOKEN',env)
            self.assertNotIn('AWS_SECRET_ACCESS_KEY',env)
            self.assertIn('read-only',command)
            self.assertEqual(command[-1],'-')
            if provider=='deepseek':
                self.assertIn('ctx.tools.guard',(work/'analysis.patch.yml').read_text())
                self.assertNotIn('CODEX_API_KEY',env)
            else:
                self.assertIn('features.shell_tool=false',command)
                self.assertNotIn('DEEPSEEK_API_KEY',env)

    def test_subprocess_stdin_output_cleanup_timeout_and_cancel(self):
        script=self.root/'fake.py'
        script.write_text("import sys,json\nprompt=sys.stdin.read()\nprint(json.dumps({'received':prompt.endswith('EVIDENCE_JSON:\\n{}')}))\n",encoding='utf-8')
        self.config.commands['deepseek']=[sys.executable,str(script)]
        output=run_backend(self.config,'deepseek','question',{},self.root,threading.Event())
        self.assertTrue(json.loads(output)['received'])
        self.assertEqual(list(self.root.glob('ai-*')),[])
        script.write_text('import time\ntime.sleep(60)\n',encoding='utf-8')
        event=threading.Event();event.set()
        with self.assertRaisesRegex(BackendError,'取消'):run_backend(self.config,'deepseek','q',{},self.root,event)
        self.config.timeout=.1
        with self.assertRaisesRegex(BackendError,'超时'):run_backend(self.config,'deepseek','q',{},self.root,threading.Event())
        self.assertEqual(list(self.root.glob('ai-*')),[])

    def test_persistent_commands_and_event_validation(self):
        for provider in ('codex','deepseek'):
            work=self.root/provider;work.mkdir()
            session={'path':work,'backend_id':'session-test'}
            command,env,answer=prepare_command(self.config,provider,work,session)
            self.assertNotIn('--ephemeral',command)
            self.assertIn('--json',command)
            self.assertIn('session-test',command)
            if provider=='codex':
                self.assertIn('resume',command)
                self.assertIn('features.shell_tool=false',command)
            else:
                self.assertIn('--session-id',command)
                self.assertIn('sessionId: !!js ctx.headlessStartup.sessionId',(work/'analysis.patch.yml').read_text())
        ds=json.dumps({'type':'final','sessionId':'session-test','reason':{'kind':'completed'},'text':'{}'})
        self.assertEqual(session_result('deepseek',ds,None,'session-test')['text'],'{}')
        with self.assertRaises(BackendError):session_result('deepseek',ds,None,'wrong-session')
        with self.assertRaises(BackendError):session_result('deepseek',ds.replace('completed','error'),None)
        cx='{"type":"thread.started","thread_id":"session-test"}\n{"type":"turn.completed"}'
        self.assertEqual(session_result('codex',cx,'{}')['backend_id'],'session-test')
        with self.assertRaises(BackendError):session_result('codex',cx.replace('turn.completed','turn.failed'),'{}')

    def test_persistent_process_sends_attachment_once_and_cleans_auth(self):
        script=self.root/'session_fake.py'
        script.write_text('''import json,sys,os
from pathlib import Path
prompt=sys.stdin.read()
result=json.dumps({'attached':prompt.count('TRACE_ATTACHMENT_JSON gpu_trace.json:'),'cwd':os.getcwd()})
if '-o' in sys.argv:
    Path(sys.argv[sys.argv.index('-o')+1]).write_text(result)
    print(json.dumps({'type':'thread.started','thread_id':'session-test'}))
    print(json.dumps({'type':'turn.completed'}))
else:
    print(json.dumps({'type':'final','sessionId':'session-test','reason':{'kind':'completed'},'text':result}))
''',encoding='utf-8')
        trace=self.root/'gpu_trace.json';raw=b'{"traceEvents":[{"name":"private-event"}]}'
        trace.write_bytes(raw)
        meta={'name':trace.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        for provider in ('codex','deepseek'):
            self.config.commands[provider]=[sys.executable,str(script)]
            path=self.root/provider;path.mkdir();(path/'home').mkdir()
            (path/'home'/'auth.json').write_text('{}')
            session={'path':path,'backend_id':None}
            first=run_backend(self.config,provider,'q',{},self.root,threading.Event(),session,[(trace,meta)])
            session['backend_id']=first['backend_id']
            second=run_backend(self.config,provider,'q2',{},self.root,threading.Event(),session)
            self.assertEqual(json.loads(first['text'])['attached'],1)
            self.assertEqual(json.loads(second['text'])['attached'],0)
            self.assertEqual(json.loads(first['text'])['cwd'],json.loads(second['text'])['cwd'])
            self.assertFalse((path/'home'/'auth.json').exists())
        trace.write_bytes(raw+b' ')
        with self.assertRaisesRegex(BackendError,'已变化'):
            run_backend(self.config,'deepseek','q',{},self.root,threading.Event(),session,[(trace,meta)])
