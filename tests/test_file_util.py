"""
FileUtil 工具函数单元测试
"""
import os
import sys
import tempfile
import pytest

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from util.FileUtil import clean_text, get_unique_filename, load_text_file_with_encoding, doc_clean_split


class TestCleanText:
    """文本清洗测试"""
    
    def test_remove_special_chars(self):
        """测试去除特殊字符"""
        # 使用足够长的文本（clean_text 会过滤掉长度<=10 的行）
        text = "Hello World! @#$测试文本内容%^&* 这是一段足够长的测试数据"
        result = clean_text(text)
        # 特殊字符应该被移除
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
        assert "%" not in result
        assert "^" not in result
        assert "*" not in result
        # 字母和中文应该保留
        assert len(result) > 0
        print("成功")
    
    def test_merge_spaces(self):
        """测试合并连续空格"""
        text = "hello    world"
        result = clean_text(text)
        assert "  " not in result
        assert "hello world" in result
    
    def test_empty_input(self):
        """测试空输入"""
        result = clean_text("")
        assert result == ""
    
    def test_chinese_text(self):
        """测试中文文本"""
        # 使用足够长的文本（clean_text 会过滤掉长度<=10 的行）
        text = "你好世界！这是一段测试文本@#数据内容 12345"
        result = clean_text(text)
        # 特殊字符应该被移除
        assert "@" not in result
        assert "#" not in result
        # 中文应该保留
        assert len(result) > 0
        # 结果应该只包含中文、字母、数字和空格
        import re
        assert re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9\s]*$', result)


class TestGetUniqueFilename:
    """唯一文件名生成测试"""
    
    def test_adds_timestamp(self):
        """测试添加时间戳"""
        filename = "test.txt"
        result = get_unique_filename(filename)
        assert "test_" in result
        assert result.endswith(".txt")
    
    def test_preserves_extension(self):
        """测试保留文件扩展名"""
        filename = "document.pdf"
        result = get_unique_filename(filename)
        assert result.endswith(".pdf")
    
    def test_different_files_different_results(self):
        """测试不同文件名生成不同结果"""
        result1 = get_unique_filename("file1.txt")
        result2 = get_unique_filename("file2.txt")
        assert "file1" in result1
        assert "file2" in result2


class TestLoadTextFileWithEncoding:
    """文件加载测试"""
    
    def test_load_utf8_file(self):
        """测试加载 UTF-8 文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("测试内容\nHello World")
            temp_path = f.name
        
        try:
            content = load_text_file_with_encoding(temp_path)
            assert "测试内容" in content
            assert "Hello World" in content
        finally:
            os.unlink(temp_path)
    
    def test_file_not_exists(self):
        """测试文件不存在"""
        with pytest.raises(Exception):
            load_text_file_with_encoding("/nonexistent/file.txt")


class TestDocCleanSplit:
    """文档切分测试"""
    
    def test_split_long_text(self):
        """测试长文本切分"""
        # 创建一个临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            # 写入足够长的文本以触发切分
            long_text = "这是一段测试文本。" * 100
            f.write(long_text)
            temp_path = f.name
        
        try:
            # 设置环境变量
            os.environ['CHUNK_SIZE'] = '200'
            os.environ['CHUNK_OVERLAP'] = '50'
            
            result = doc_clean_split(temp_path)
            
            # 应该返回多个文档片段
            assert isinstance(result, list)
            assert len(result) > 0
            
            # 每个片段都应该是 Document 对象
            for doc in result:
                assert hasattr(doc, 'page_content')
                assert hasattr(doc, 'metadata')
        finally:
            os.unlink(temp_path)
    
    def test_empty_file(self):
        """测试空文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("")
            temp_path = f.name
        
        try:
            os.environ['CHUNK_SIZE'] = '1000'
            os.environ['CHUNK_OVERLAP'] = '0'
            
            result = doc_clean_split(temp_path)
            assert isinstance(result, list)
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
