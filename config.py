"""
基于BERT的多源数据情感分析系统配置文件
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """基本配置类"""
    # 应用配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # 数据目录配置
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    VISUALIZATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/visualizations')
    
    # 爬虫配置
    CRAWL_DELAY = 2  # 爬虫请求间隔（秒）
    DEFAULT_CRAWL_PAGES = 5  # 默认爬取页数
    
    # 模型配置
    BERT_MODEL_NAME = 'bert-base-chinese'  # 使用的BERT模型名称
    SENTIMENT_CLASSES = 3  # 情感分类数（3：负面、中性、正面）
    MAX_LENGTH = 128  # 最大序列长度
    BATCH_SIZE = 16  # 批处理大小
    EPOCHS = 4  # 训练轮数
    LEARNING_RATE = 2e-5  # 学习率
    
    # 数据预处理配置
    STOPWORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/stopwords.txt')
    
    # 可视化配置
    VISUALIZATION_DPI = 300  # 图片DPI
    
    @staticmethod
    def init_app(app):
        """初始化应用"""
        # 确保目录存在
        for dir_path in [Config.DATA_DIR, Config.MODELS_DIR, Config.UPLOAD_FOLDER, Config.VISUALIZATIONS_DIR]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True

class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    # 生产环境可以使用更保守的爬虫延迟
    CRAWL_DELAY = 3

class TestingConfig(Config):
    """测试环境配置"""
    TESTING = True
    # 使用内存数据库进行测试
    WTF_CSRF_ENABLED = False

# 配置字典
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# 默认停用词列表（如果没有外部文件）
DEFAULT_STOPWORDS = [
    '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', 
    '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '啊', '吧', '把', 
    '但', '但是', '并', '个', '给', '过', '还', '还是', '还有', '其', '其实', '其中', '几', '可', '可以', 
    '可是', '么', '没', '什么', '什么样', '这样', '那样', '之', '之一', '只', '只是', '只要', '只有', 
    '就是', '就是说', '打', '呢', '来', '来说', '来自', '哪儿', '哪里', '如', '如果', '如何', '如此', 
    '对', '对于', '比', '多', '多少', '而', '而且', '被', '该', '应', '应该', '一些', '这些', '那些', 
    '任何', '这个', '那个', '从', '现在', '因为', '所以', '所有', '这里', '那里', '一下', '一定', 
    '已经', '出来', '过来', '起来', '这么', '那么', '最', '这次', '那次'
]
