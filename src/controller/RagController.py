# 导入必要的库
from fastapi.responses import StreamingResponse

from fastapi import FastAPI, File, UploadFile, APIRouter
# import uvicorn
from dotenv import load_dotenv
import os
import sys

# 添加 src 目录到 Python 路径（解决模块导入问题）
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from service.RagService import RagService
load_dotenv()

# 创建路由器
upload_router = APIRouter(prefix="/upload", tags=["文件上传"])
answer_router = APIRouter(prefix="/ask", tags=["问答"])
token_router = APIRouter(prefix="/token", tags=["统计使用的token数量"])

ragService = RagService()

@answer_router.get("/")
async def answer(question: str, session_id: str = None):
    return StreamingResponse(
        ragService.rag_qa(question, session_id),
        media_type="text/event-stream"
    )


@token_router.get("/")
async def token():
    return ragService.get_completion_token()


@upload_router.post("/")
async def upload_single_file(file: UploadFile = File(...)):
    return await ragService.rag_file_upload(file)


# 创建FastAPI应用
app = FastAPI(title="rag项目", version="1.0.0")
app.include_router(upload_router)
app.include_router(answer_router)
app.include_router(token_router)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
