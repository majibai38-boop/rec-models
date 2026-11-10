import argparse
import os
import uvicorn
from .app import create_app


def main():
    parser=argparse.ArgumentParser(description="单服务器性能预测工作台")
    parser.add_argument("--port",type=int,default=8770)
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--auth",choices=("auto","local","token","none"),default=os.environ.get("WORKLOAD_MODELER_AUTH","auto"))
    parser.add_argument("--allowed-hosts",default=os.environ.get("WORKLOAD_MODELER_HOSTS"))
    parser.add_argument("--data-dir",default=None)
    parser.add_argument("--upload-limit-mb",type=int,default=2048,help="单文件上传上限（MiB）")
    parser.add_argument("--expanded-limit-mb",type=int,default=4096,help="Trace解压/CSV上限（MiB）")
    parser.add_argument("--memory-mb",type=int,default=4096,help="分析进程内存软限制（MiB）")
    parser.add_argument("--timeout",type=int,default=600,help="分析超时（秒）")
    args=parser.parse_args()
    local=args.host in {"127.0.0.1","localhost","::1"}
    token=os.environ.get("WORKLOAD_MODELER_TOKEN","")
    mode=args.auth if args.auth!="auto" else "local" if local and not token else "token"
    if mode=="local" and not local:
        parser.error("local模式只能监听回环地址；局域网免令牌请明确指定--auth none")
    if mode=="token" and len(token)<24:
        parser.error("请设置至少24字符的WORKLOAD_MODELER_TOKEN")
    hosts=args.allowed_hosts.split(',') if args.allowed_hosts else None if local else ['*']
    if not 1<=args.port<=65535:
        parser.error("port必须为1–65535")
    if min(args.upload_limit_mb,args.expanded_limit_mb,args.memory_mb,args.timeout)<=0:
        parser.error("资源限制必须为正整数")
    uvicorn.run(create_app(args.data_dir,upload_limit_mb=args.upload_limit_mb,
        expanded_limit_mb=args.expanded_limit_mb,memory_mb=args.memory_mb,timeout=args.timeout,
        auth_mode=mode,token=token,allowed_hosts=hosts),
        host=args.host,port=args.port,proxy_headers=False)


if __name__=="__main__":
    main()
