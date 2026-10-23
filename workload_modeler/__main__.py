import argparse
import uvicorn
from .app import create_app


def main():
    parser=argparse.ArgumentParser(description="本机性能预测工作台")
    parser.add_argument("--port",type=int,default=8770)
    parser.add_argument("--data-dir",default=None)
    parser.add_argument("--upload-limit-mb",type=int,default=2048,help="单文件上传上限（MiB）")
    parser.add_argument("--expanded-limit-mb",type=int,default=4096,help="Trace解压/CSV上限（MiB）")
    parser.add_argument("--memory-mb",type=int,default=4096,help="分析进程内存软限制（MiB）")
    parser.add_argument("--timeout",type=int,default=600,help="分析超时（秒）")
    args=parser.parse_args()
    if not 1<=args.port<=65535:
        parser.error("port必须为1–65535")
    if min(args.upload_limit_mb,args.expanded_limit_mb,args.memory_mb,args.timeout)<=0:
        parser.error("资源限制必须为正整数")
    uvicorn.run(create_app(args.data_dir,upload_limit_mb=args.upload_limit_mb,
        expanded_limit_mb=args.expanded_limit_mb,memory_mb=args.memory_mb,timeout=args.timeout),
        host="127.0.0.1",port=args.port,proxy_headers=False)


if __name__=="__main__":
    main()
