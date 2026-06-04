
FROM python:3.13-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建目录
RUN mkdir -p logs uploads

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "src/main.py"]
