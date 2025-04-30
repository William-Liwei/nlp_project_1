# '''
# 基于BERT的多源数据情感分析系统
# 功能模块：
# 1. 多平台数据爬取模块(豆瓣、电商、网易云音乐等)
# 2. 数据预处理与清洗模块
# 3. BERT模型训练与优化模块
# 4. 情感分析与分类模块
# 5. 结果可视化与分析展示模块
# '''

# import os
# import re
# import json
# import time
# import random
# import jieba
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
# import seaborn as sns
# from wordcloud import WordCloud
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# import torch
# from torch import nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
# from torch.optim import AdamW

# import requests
# from bs4 import BeautifulSoup
# import urllib.parse
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm

# # 确保所需目录存在
# os.makedirs('data', exist_ok=True)
# os.makedirs('models', exist_ok=True)
# os.makedirs('uploads', exist_ok=True)
# os.makedirs('static/visualizations', exist_ok=True)
# os.makedirs('static/fonts', exist_ok=True)

# # 设置中文字体
# try:
#     # 尝试在项目目录中查找字体
#     font_paths = [
#         'static/fonts/SimHei.ttf',
#         'SimHei.ttf',
#         '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # Linux常见路径
#     ]
    
#     font_found = False
#     for path in font_paths:
#         if os.path.exists(path):
#             font = FontProperties(fname=path)
#             font_found = True
#             break
    
#     if not font_found:
#         font = FontProperties(family=['sans'])
#         print("警告: 找不到中文字体文件，将使用系统默认字体")
# except:
#     font = FontProperties(family=['sans'])
#     print("警告: 设置中文字体时出错，将使用系统默认字体")

# # 尝试设置matplotlib中文支持
# try:
#     plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
#     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# except:
#     print("警告: 设置matplotlib中文支持时出错")

# # ====================== 1. 爬虫模块 ======================

# class MultiSourceCrawler:
#     """多源数据爬取类，包含豆瓣电影、京东商品、网易云音乐评论爬取"""
    
#     def __init__(self):
#         self.headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'
#         }
#         # 创建保存数据的目录
#         os.makedirs('data', exist_ok=True)
    
#     def random_sleep(self):
#         """随机休眠，避免被反爬"""
#         time.sleep(random.uniform(1, 3))
    
#     def crawl_douban_movie_reviews(self, movie_id, pages=10):
#         """爬取豆瓣电影评论
        
#         Args:
#             movie_id: 豆瓣电影ID
#             pages: 爬取页数
            
#         Returns:
#             评论数据列表
#         """
#         print(f"开始爬取豆瓣电影 {movie_id} 的评论...")
#         reviews = []
        
#         for page in range(pages):
#             url = f"https://movie.douban.com/subject/{movie_id}/comments?start={page*20}&limit=20&sort=new_score&status=P"
            
#             try:
#                 response = requests.get(url, headers=self.headers)
#                 if response.status_code == 200:
#                     soup = BeautifulSoup(response.text, 'html.parser')
#                     comment_items = soup.select('.comment-item')
                    
#                     for item in comment_items:
#                         try:
#                             comment = item.select_one('.comment p').text.strip()
#                             rating = item.select_one('.rating')
#                             score = 0
#                             if rating:
#                                 score_class = rating.get('class')
#                                 if score_class:
#                                     for cls in score_class:
#                                         if 'allstar' in cls:
#                                             score = int(cls.replace('allstar', '')) // 10
#                                             break
                            
#                             # 获取评论者信息
#                             user_info = item.select_one('.comment-info a').text.strip() if item.select_one('.comment-info a') else "未知用户"
#                             comment_time = item.select_one('.comment-time').text.strip() if item.select_one('.comment-time') else ""
                            
#                             reviews.append({
#                                 'platform': '豆瓣电影',
#                                 'content': comment,
#                                 'score': score,
#                                 'user': user_info,
#                                 'time': comment_time,
#                                 'movie_id': movie_id
#                             })
#                         except Exception as e:
#                             print(f"处理豆瓣评论项时出错: {e}")
#                             continue
                    
#                     print(f"已爬取豆瓣电影第 {page+1} 页评论")
#                     self.random_sleep()
#                 else:
#                     print(f"请求失败，状态码: {response.status_code}")
#                     break
#             except Exception as e:
#                 print(f"爬取豆瓣电影评论出错: {e}")
#                 continue
        
#         # 保存数据
#         if reviews:
#             df = pd.DataFrame(reviews)
#             output_path = f'data/douban_movie_{movie_id}_reviews.csv'
#             df.to_csv(output_path, index=False, encoding='utf-8')
#             print(f"豆瓣电影 {movie_id} 的评论已保存到 {output_path}，共 {len(reviews)} 条")
#         else:
#             print(f"未获取到豆瓣电影 {movie_id} 的评论数据")
        
#         return reviews
    
#     def crawl_jd_product_reviews(self, product_id, pages=10):
#         """爬取京东商品评论
        
#         Args:
#             product_id: 京东商品ID
#             pages: 爬取页数
            
#         Returns:
#             评论数据列表
#         """
#         print(f"开始爬取京东商品 {product_id} 的评论...")
#         reviews = []
        
#         for page in range(pages):
#             url = f"https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId={product_id}&score=0&sortType=5&page={page}&pageSize=10&isShadowSku=0&fold=1"
            
#             try:
#                 response = requests.get(url, headers=self.headers)
#                 if response.status_code == 200:
#                     # 处理jsonp格式数据
#                     text = response.text
#                     try:
#                         json_str = re.search(r'fetchJSON_comment98\((.*)\);', text).group(1)
#                         data = json.loads(json_str)
                        
#                         comment_list = data.get('comments', [])
#                         for comment in comment_list:
#                             reviews.append({
#                                 'platform': '京东商品',
#                                 'content': comment.get('content', ''),
#                                 'score': comment.get('score', 0),
#                                 'user': comment.get('nickname', ''),
#                                 'time': comment.get('creationTime', ''),
#                                 'product_id': product_id
#                             })
#                     except Exception as e:
#                         print(f"解析京东评论JSON数据出错: {e}")
#                         continue
                    
#                     print(f"已爬取京东商品第 {page+1} 页评论")
#                     self.random_sleep()
#                 else:
#                     print(f"请求失败，状态码: {response.status_code}")
#                     break
#             except Exception as e:
#                 print(f"爬取京东商品评论出错: {e}")
#                 continue
        
#         # 保存数据
#         if reviews:
#             df = pd.DataFrame(reviews)
#             output_path = f'data/jd_product_{product_id}_reviews.csv'
#             df.to_csv(output_path, index=False, encoding='utf-8')
#             print(f"京东商品 {product_id} 的评论已保存到 {output_path}，共 {len(reviews)} 条")
#         else:
#             print(f"未获取到京东商品 {product_id} 的评论数据")
        
#         return reviews
    
#     def crawl_netease_music_comments(self, song_id, pages=10):
#         """爬取网易云音乐评论
        
#         Args:
#             song_id: 歌曲ID
#             pages: 爬取页数
            
#         Returns:
#             评论数据列表
#         """
#         print(f"开始爬取网易云音乐 {song_id} 的评论...")
#         reviews = []
        
#         # 注意：网易云音乐评论API需要特殊处理，这里使用简化模拟
#         for page in range(pages):
#             url = f"https://music.163.com/api/v1/resource/comments/R_SO_4_{song_id}?limit=20&offset={page*20}"
            
#             try:
#                 response = requests.get(url, headers=self.headers)
#                 if response.status_code == 200:
#                     try:
#                         data = response.json()
#                         comments = data.get('comments', [])
                        
#                         for comment in comments:
#                             content = comment.get('content', '')
#                             user = comment.get('user', {}).get('nickname', '')
#                             time_str = comment.get('time', 0)
#                             time_fmt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_str/1000)) if time_str else ''
                            
#                             # 点赞数作为情感参考
#                             like_count = comment.get('likedCount', 0)
#                             # 简化处理：点赞数0-10为1分，10-50为2分，50-200为3分，200-1000为4分，1000以上为5分
#                             score = 1
#                             if like_count > 10:
#                                 score = 2
#                             if like_count > 50:
#                                 score = 3
#                             if like_count > 200:
#                                 score = 4
#                             if like_count > 1000:
#                                 score = 5
                            
#                             reviews.append({
#                                 'platform': '网易云音乐',
#                                 'content': content,
#                                 'score': score,
#                                 'user': user,
#                                 'time': time_fmt,
#                                 'song_id': song_id
#                             })
#                     except Exception as e:
#                         print(f"解析网易云音乐评论数据出错: {e}")
#                         continue
                    
#                     print(f"已爬取网易云音乐第 {page+1} 页评论")
#                     self.random_sleep()
#                 else:
#                     print(f"请求失败，状态码: {response.status_code}")
#                     break
#             except Exception as e:
#                 print(f"爬取网易云音乐评论出错: {e}")
#                 continue
        
#         # 保存数据
#         if reviews:
#             df = pd.DataFrame(reviews)
#             output_path = f'data/netease_music_{song_id}_reviews.csv'
#             df.to_csv(output_path, index=False, encoding='utf-8')
#             print(f"网易云音乐 {song_id} 的评论已保存到 {output_path}，共 {len(reviews)} 条")
#         else:
#             print(f"未获取到网易云音乐 {song_id} 的评论数据")
        
#         return reviews
    
#     def crawl_all_sources(self, movie_ids=None, product_ids=None, song_ids=None, pages=5):
#         """爬取所有数据源的评论
        
#         Args:
#             movie_ids: 豆瓣电影ID列表
#             product_ids: 京东商品ID列表
#             song_ids: 网易云音乐歌曲ID列表
#             pages: 每个ID爬取的页数
        
#         Returns:
#             所有评论数据的DataFrame
#         """
#         all_reviews = []
        
#         # 豆瓣电影评论
#         if movie_ids:
#             for movie_id in movie_ids:
#                 reviews = self.crawl_douban_movie_reviews(movie_id, pages)
#                 all_reviews.extend(reviews)
        
#         # 京东商品评论
#         if product_ids:
#             for product_id in product_ids:
#                 reviews = self.crawl_jd_product_reviews(product_id, pages)
#                 all_reviews.extend(reviews)
        
#         # 网易云音乐评论
#         if song_ids:
#             for song_id in song_ids:
#                 reviews = self.crawl_netease_music_comments(song_id, pages)
#                 all_reviews.extend(reviews)
        
#         # 合并所有数据
#         if all_reviews:
#             df = pd.DataFrame(all_reviews)
#             output_path = 'data/all_reviews.csv'
#             df.to_csv(output_path, index=False, encoding='utf-8')
#             print(f"所有评论数据已保存到 {output_path}，共 {len(all_reviews)} 条")
#             return df
#         else:
#             print("未获取到任何评论数据")
#             return pd.DataFrame()

# # ====================== 2. 数据预处理模块 ======================

# class TextPreprocessor:
#     """文本预处理类，包含文本清洗、分词、去停用词等功能"""
    
#     def __init__(self, stopwords_path=None):
#         """初始化预处理器
        
#         Args:
#             stopwords_path: 停用词表路径，不提供则使用默认的停用词列表
#         """
#         # 加载停用词表
#         self.stopwords = set()
#         if stopwords_path and os.path.exists(stopwords_path):
#             with open(stopwords_path, 'r', encoding='utf-8') as f:
#                 self.stopwords = set([line.strip() for line in f.readlines()])
#         else:
#             # 使用默认的简单停用词表
#             self.stopwords = set(['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', 
#                 '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '啊', '吧', '把', 
#                 '但', '但是', '并', '个', '给', '过', '还', '还是', '还有', '其', '其实', '其中', '几', '可', '可以', 
#                 '可是', '么', '没', '什么', '什么样', '这样', '那样', '之', '之一', '只', '只是', '只要', '只有', 
#                 '就是', '就是说', '打', '呢', '来', '来说', '来自', '哪儿', '哪里', '如', '如果', '如何', '如此', 
#                 '对', '对于', '比', '多', '多少', '而', '而且', '被', '该', '应', '应该'])
        
#         # 确保jieba加载成功
#         try:
#             jieba.initialize()
#             print("成功初始化jieba分词")
#         except Exception as e:
#             print(f"初始化jieba分词出错: {e}")
        
#         # 加载结巴分词用户词典
#         # jieba.load_userdict('user_dict.txt')  # 如果有自定义词典，可以取消注释
    
#     def clean_text(self, text):
#         """清洗文本
        
#         Args:
#             text: 原始文本
            
#         Returns:
#             清洗后的文本
#         """
#         if not isinstance(text, str) or not text:
#             return ''
        
#         try:
#             # 去除HTML标签
#             text = re.sub(r'<[^>]+>', '', text)
#             # 去除网址
#             text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
#             # 去除特殊字符和标点符号，但保留中文标点
#             text = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', '', text)
#             # 去除数字
#             text = re.sub(r'\d+', '', text)
#             # 去除多余空白字符
#             text = re.sub(r'\s+', ' ', text).strip()
            
#             return text
#         except Exception as e:
#             print(f"清洗文本出错: {e}")
#             return text if isinstance(text, str) else ''
    
#     def segment(self, text):
#         """中文分词
        
#         Args:
#             text: 清洗后的文本
            
#         Returns:
#             分词结果列表
#         """
#         if not text:
#             return []
            
#         try:
#             # 使用结巴分词
#             words = jieba.lcut(text)
#             # 去除停用词
#             words = [word for word in words if len(word) > 1 and word not in self.stopwords]
            
#             return words
#         except Exception as e:
#             print(f"分词出错: {e}")
#             return []
    
#     def process_dataframe(self, df, content_col='content'):
#         """处理DataFrame中的文本数据
        
#         Args:
#             df: 包含文本数据的DataFrame
#             content_col: 文本内容的列名
            
#         Returns:
#             处理后的DataFrame，增加了清洗文本和分词结果列
#         """
#         # 复制一份，避免修改原始数据
#         result_df = df.copy()
        
#         # 确保content_col存在
#         if content_col not in result_df.columns:
#             print(f"错误: 列 '{content_col}' 不存在于数据中")
#             if not result_df.empty:
#                 print(f"可用列: {result_df.columns.tolist()}")
#             return result_df
        
#         print(f"开始处理 {len(result_df)} 条文本数据...")
        
#         # 清洗文本
#         result_df['clean_content'] = result_df[content_col].apply(self.clean_text)
        
#         # 分词
#         result_df['words'] = result_df['clean_content'].apply(self.segment)
        
#         # 计算分词后的词数
#         result_df['word_count'] = result_df['words'].apply(len)
        
#         print(f"文本处理完成。")
#         return result_df
    
#     def save_processed_data(self, df, output_path='data/processed_reviews.csv'):
#         """保存处理后的数据
        
#         Args:
#             df: 处理后的DataFrame
#             output_path: 输出文件路径
#         """
#         try:
#             # 确保数据目录存在
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
#             # 将words列表转换为字符串，方便存储
#             df['words_str'] = df['words'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
            
#             # 保存到CSV
#             df.to_csv(output_path, index=False, encoding='utf-8')
#             print(f"处理后的数据已保存到 {output_path}")
#         except Exception as e:
#             print(f"保存处理后的数据出错: {e}")

#     def extract_linguistic_features(self, text):
#         """提取高级语言学特征用于学术NLP分析"""
#         features = {}
#         # 情感词统计（使用预定义词典）
#         pos_words = ['优秀', '精彩', '满意', '喜欢', '推荐', '好', '棒', '赞', '华丽', '精致', 
#                     '震撼', '感人', '完美', '出色', '卓越', '超值', '高效', '实用', '方便']
#         neg_words = ['差', '糟糕', '失望', '浪费', '后悔', '烂', '坑', '骗', '无用', '过时',
#                     '难用', '枯燥', '乏味', '无聊', '劣质', '粗糙', '虚假', '敷衍']
        
#         # 分词结果
#         words = self.segment(text)
        
#         # 词频特征
#         features['正面词数量'] = sum(1 for word in words if word in pos_words)
#         features['负面词数量'] = sum(1 for word in words if word in neg_words)
#         features['平均词长'] = np.mean([len(word) for word in words]) if words else 0
#         features['句子数量'] = len(re.split(r'[。！？]', text))
        
#         # 情感强度词统计
#         intensifiers = ['很', '非常', '太', '真是', '极其', '特别', '相当', '十分', '尤其', '格外']
#         features['情感强度词数量'] = sum(1 for word in words if word in intensifiers)
        
#         # 转折词统计
#         transitions = ['但是', '然而', '却', '不过', '尽管', '虽然', '反而', '相反', '只是']
#         features['转折词数量'] = sum(1 for i in range(len(text)-1) if text[i:i+2] in transitions)
        
#         # 否定词统计
#         negations = ['不', '没', '不是', '没有', '别', '莫', '未', '无']
#         features['否定词数量'] = sum(1 for word in words if word in negations)
        
#         return features

#     def load_pretrained_dataset(self, dataset_name='ChnSentiCorp'):
#         """加载标准中文情感数据集以获得更好的性能"""
#         print(f"加载预训练数据集: {dataset_name}")
        
#         if dataset_name == 'ChnSentiCorp':
#             # 这个数据集包含约12k中文评论，带有二元情感标注
#             try:
#                 df = pd.read_csv('data/ChnSentiCorp_htl_all.csv')
#                 df['sentiment_label'] = df['label'].map({0: 0, 1: 2})  # 映射到三分类系统
#                 df['sentiment_text'] = df['sentiment_label'].map({0: '负面', 1: '中性', 2: '正面'})
#                 return df
#             except:
#                 print("无法加载ChnSentiCorp数据集，请确保文件已下载到data目录")
        
#         elif dataset_name == 'online_shopping_10_cats':
#             # 一个更大的数据集，包含跨10个产品类别的60k+评论
#             try:
#                 df = pd.read_csv('data/online_shopping_10_cats.csv')
#                 # 将5星评分转换为情感类别
#                 df['sentiment_label'] = df['rating'].apply(
#                     lambda x: 0 if x <= 2 else (1 if x == 3 else 2)
#                 )
#                 df['sentiment_text'] = df['sentiment_label'].map({0: '负面', 1: '中性', 2: '正面'})
#                 return df
#             except:
#                 print("无法加载online_shopping_10_cats数据集，请确保文件已下载到data目录")
        
#         # 如果没有请求的数据集，或加载失败，则创建并加载自带的小型示例数据集
#         print("尝试加载示例数据集...")
#         if not os.path.exists('data/sample_reviews.csv'):
#             self._create_sample_dataset()
            
#         try:
#             df = pd.read_csv('data/sample_reviews.csv')
#             print(f"加载示例数据集，共{len(df)}条评论")
#             return df
#         except:
#             print("无法加载任何数据集，请检查data目录是否存在")
#             return pd.DataFrame()

#     def _create_sample_dataset(self):
#         """创建示例数据集保存到CSV"""
#         data = {
#             'content': [
#                 "这部电影太棒了，演员演技很好，情节扣人心弦，强烈推荐！",
#                 "画面精美，但是剧情有点拖沓，人物塑造不够丰满。",
#                 "这是我看过最差的电影，浪费时间和金钱，剧情混乱，演技尴尬。",
#                 "音效不错，但是台词很生硬，整体感觉一般。",
#                 "特效炸裂，剧情紧凑，节奏把握得很好，是一部不可多得的佳作。",
#                 "虽然有些桥段不太合理，但整体来说是部不错的电影。",
#                 "剧情老套，演技浮夸，毫无创新可言，很失望。",
#                 "影片节奏紧凑，情节设计巧妙，结局出人意料，非常精彩。",
#                 "特效做得不错，但是剧情实在太过牵强，人物刻画也很单薄。",
#                 "音乐配得很到位，渲染了整个影片的氛围，值得一看。"
#             ],
#             'score': [5, 3, 1, 3, 5, 4, 2, 5, 2, 4],
#             'platform': ['豆瓣电影'] * 10,
#             'time': ['2023-01-01'] * 10
#         }
        
#         df = pd.DataFrame(data)
#         os.makedirs('data', exist_ok=True)
#         df.to_csv('data/sample_reviews.csv', index=False, encoding='utf-8')
#         print(f"示例数据集已保存到 data/sample_reviews.csv，共{len(df)}条评论")

# # ====================== 3. BERT模型训练与情感分析模块 ======================

# class SentimentDataset(Dataset):
#     """情感分析数据集类"""
    
#     def __init__(self, texts, labels, tokenizer, max_length=128):
#         """初始化数据集
        
#         Args:
#             texts: 文本列表
#             labels: 标签列表
#             tokenizer: BERT tokenizer
#             max_length: 最大序列长度
#         """
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length
    
#     def __len__(self):
#         return len(self.texts)
    
#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         label = self.labels[idx]
        
#         # 使用BERT tokenizer编码文本
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
        
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.tensor(label, dtype=torch.long)
#         }

# class BertSentimentClassifier(nn.Module):
#     """基于BERT的情感分类模型"""
    
#     def __init__(self, bert_model_name, num_classes):
#         """初始化模型
        
#         Args:
#             bert_model_name: BERT预训练模型名称
#             num_classes: 类别数量
#         """
#         super(BertSentimentClassifier, self).__init__()
        
#         try:
#             self.bert = BertModel.from_pretrained(bert_model_name)
#             self.dropout = nn.Dropout(0.1)
#             self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
#             print(f"成功初始化BERT模型: {bert_model_name}")
#         except Exception as e:
#             print(f"初始化BERT模型出错: {e}")
#             # 创建一个空壳模型，避免程序崩溃
#             self.bert = None
#             self.dropout = nn.Dropout(0.1)
#             self.fc = nn.Linear(768, num_classes)  # 使用默认BERT隐藏大小
    
#     def forward(self, input_ids, attention_mask):
#         """前向传播
        
#         Args:
#             input_ids: 输入ID序列
#             attention_mask: 注意力掩码
            
#         Returns:
#             logits: 分类logits
#         """
#         if self.bert is None:
#             # 如果BERT初始化失败，返回随机输出
#             batch_size = input_ids.shape[0]
#             return torch.randn(batch_size, self.fc.out_features)
        
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
        
#         pooled_output = outputs.pooler_output
#         pooled_output = self.dropout(pooled_output)
#         logits = self.fc(pooled_output)
        
#         return logits

# class SentimentAnalyzer:
#     """情感分析器类，包含模型训练和预测功能"""
    
#     def __init__(self, bert_model_name='bert-base-chinese', num_classes=3, device=None):
#         """初始化情感分析器
        
#         Args:
#             bert_model_name: BERT预训练模型名称
#             num_classes: 情感类别数，默认为3（负面、中性、正面）
#             device: 运行设备，默认自动选择
#         """
#         # 设置运行设备
#         self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"使用设备: {self.device}")
        
#         # 加载BERT tokenizer
#         try:
#             self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
#             print(f"成功加载BERT tokenizer: {bert_model_name}")
#         except Exception as e:
#             print(f"加载BERT tokenizer出错: {e}")
#             # 尝试加载本地保存的tokenizer
#             try:
#                 if os.path.exists('models/tokenizer'):
#                     self.tokenizer = BertTokenizer.from_pretrained('models/tokenizer')
#                     print("已加载本地tokenizer")
#                 else:
#                     print("警告: 无法加载BERT tokenizer，某些功能可能无法正常工作")
#                     self.tokenizer = None
#             except:
#                 print("警告: 无法加载BERT tokenizer，某些功能可能无法正常工作")
#                 self.tokenizer = None
        
#         # 初始化BERT情感分类模型
#         self.model = BertSentimentClassifier(bert_model_name, num_classes)
#         self.model.to(self.device)
        
#         # 模型保存路径
#         self.model_save_path = 'models/bert_sentiment_model.pth'
#         os.makedirs('models', exist_ok=True)
        
#         # 类别数
#         self.num_classes = num_classes
    
#     def prepare_data(self, df, text_col='clean_content', label_col='sentiment_label', test_size=0.2):
#         """准备训练和测试数据
        
#         Args:
#             df: 包含文本和情感标签的DataFrame
#             text_col: 文本列名
#             label_col: 标签列名
#             test_size: 测试集比例
            
#         Returns:
#             训练和测试数据加载器
#         """
#         # 检查必要的列是否存在
#         if text_col not in df.columns:
#             print(f"错误: 文本列 '{text_col}' 不存在于数据中")
#             print(f"可用列: {df.columns.tolist()}")
#             return None, None
        
#         if label_col not in df.columns:
#             print(f"错误: 标签列 '{label_col}' 不存在于数据中")
#             print(f"可用列: {df.columns.tolist()}")
#             return None, None
        
#         # 确保tokenizer已加载
#         if self.tokenizer is None:
#             print("错误: BERT tokenizer未加载，无法准备数据")
#             return None, None
        
#         texts = df[text_col].fillna('').tolist()
#         labels = df[label_col].fillna(0).astype(int).tolist()
        
#         # 检查标签是否有效
#         if min(labels) < 0 or max(labels) >= self.num_classes:
#             print(f"警告: 标签范围应为0-{self.num_classes-1}，但实际范围为{min(labels)}-{max(labels)}")
#             # 修正标签
#             labels = [max(0, min(l, self.num_classes-1)) for l in labels]
        
#         # 划分训练集和测试集
#         try:
#             X_train, X_test, y_train, y_test = train_test_split(
#                 texts, labels, test_size=test_size, random_state=42, stratify=labels
#             )
            
#             print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
            
#             # 创建数据集
#             train_dataset = SentimentDataset(X_train, y_train, self.tokenizer)
#             test_dataset = SentimentDataset(X_test, y_test, self.tokenizer)
            
#             # 创建数据加载器
#             train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#             test_dataloader = DataLoader(test_dataset, batch_size=16)
            
#             return train_dataloader, test_dataloader
#         except Exception as e:
#             print(f"准备数据时出错: {e}")
#             return None, None
    
#     def train(self, train_dataloader, epochs=4, learning_rate=2e-5):
#         """训练模型
        
#         Args:
#             train_dataloader: 训练数据加载器
#             epochs: 训练轮数
#             learning_rate: 学习率
#         """
#         if train_dataloader is None:
#             print("错误: 训练数据加载器为空，无法进行训练")
#             return
        
#         # 优化器
#         optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
#         # 学习率调度器
#         total_steps = len(train_dataloader) * epochs
#         scheduler = get_linear_schedule_with_warmup(
#             optimizer, 
#             num_warmup_steps=0,
#             num_training_steps=total_steps
#         )
        
#         # 损失函数
#         criterion = nn.CrossEntropyLoss()
        
#         # 训练循环
#         self.model.train()
        
#         try:
#             for epoch in range(epochs):
#                 print(f"Epoch {epoch+1}/{epochs}")
#                 total_loss = 0
                
#                 # 进度条
#                 progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
                
#                 for batch in progress_bar:
#                     # 准备数据
#                     input_ids = batch['input_ids'].to(self.device)
#                     attention_mask = batch['attention_mask'].to(self.device)
#                     labels = batch['labels'].to(self.device)
                    
#                     # 清零梯度
#                     optimizer.zero_grad()
                    
#                     # 前向传播
#                     outputs = self.model(input_ids, attention_mask)
                    
#                     # 计算损失
#                     loss = criterion(outputs, labels)
#                     total_loss += loss.item()
                    
#                     # 反向传播
#                     loss.backward()
                    
#                     # 梯度裁剪
#                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
#                     # 更新参数
#                     optimizer.step()
#                     scheduler.step()
                    
#                     # 更新进度条
#                     progress_bar.set_postfix({'loss': loss.item()})
                
#                 avg_loss = total_loss / len(train_dataloader)
#                 print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
#             # 保存模型
#             torch.save(self.model.state_dict(), self.model_save_path)
#             print(f"模型已保存到 {self.model_save_path}")
            
#         except Exception as e:
#             print(f"训练过程中出错: {e}")
    
#     def evaluate(self, test_dataloader):
#         """评估模型
        
#         Args:
#             test_dataloader: 测试数据加载器
            
#         Returns:
#             评估结果（准确率、分类报告、混淆矩阵）
#         """
#         if test_dataloader is None:
#             print("错误: 测试数据加载器为空，无法进行评估")
#             return None, None, None
        
#         self.model.eval()
        
#         # 存储预测结果和真实标签
#         predictions = []
#         true_labels = []
        
#         try:
#             with torch.no_grad():
#                 for batch in tqdm(test_dataloader, desc="Evaluating"):
#                     # 准备数据
#                     input_ids = batch['input_ids'].to(self.device)
#                     attention_mask = batch['attention_mask'].to(self.device)
#                     labels = batch['labels'].to(self.device)
                    
#                     # 前向传播
#                     outputs = self.model(input_ids, attention_mask)
                    
#                     # 获取预测结果
#                     _, preds = torch.max(outputs, 1)
                    
#                     # 保存结果
#                     predictions.extend(preds.cpu().tolist())
#                     true_labels.extend(labels.cpu().tolist())
            
#             # 计算准确率
#             accuracy = accuracy_score(true_labels, predictions)
#             print(f"测试集准确率: {accuracy:.4f}")
            
#             # 生成分类报告
#             if self.num_classes == 3:
#                 target_names = ['负面', '中性', '正面']
#             elif self.num_classes == 5:
#                 target_names = ['非常负面', '负面', '中性', '正面', '非常正面']
#             else:
#                 target_names = [str(i) for i in range(self.num_classes)]
            
#             report = classification_report(true_labels, predictions, target_names=target_names)
#             print("分类报告:")
#             print(report)
            
#             # 混淆矩阵
#             cm = confusion_matrix(true_labels, predictions)
            
#             return accuracy, report, cm
        
#         except Exception as e:
#             print(f"评估过程中出错: {e}")
#             return None, None, None
    
#     def load_model(self):
#         """加载已保存的模型"""
#         if os.path.exists(self.model_save_path):
#             try:
#                 self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
#                 self.model.eval()
#                 print(f"已加载模型 {self.model_save_path}")
#                 return True
#             except Exception as e:
#                 print(f"加载模型出错: {e}")
#                 return False
#         else:
#             print(f"模型文件 {self.model_save_path} 不存在")
#             return False
    
#     def predict(self, texts):
#         """预测文本情感
        
#         Args:
#             texts: 文本列表或单个文本
            
#         Returns:
#             预测结果（情感标签和概率）
#         """
#         # 确保tokenizer已加载
#         if self.tokenizer is None:
#             print("错误: BERT tokenizer未加载，无法进行预测")
#             return pd.DataFrame({'text': texts if isinstance(texts, list) else [texts],
#                                 'sentiment_label': [0] * (len(texts) if isinstance(texts, list) else 1),
#                                 'probability': [0.0] * (len(texts) if isinstance(texts, list) else 1)})
        
#         # 确保模型处于评估模式
#         self.model.eval()
        
#         # 处理单个文本的情况
#         if isinstance(texts, str):
#             texts = [texts]
        
#         results = []
        
#         try:
#             with torch.no_grad():
#                 for text in texts:
#                     if not text or not isinstance(text, str):
#                         # 处理空文本或非字符串
#                         results.append({
#                             'text': text if isinstance(text, str) else '',
#                             'sentiment_label': 0,  # 默认为中性
#                             'probability': 0.0
#                         })
#                         continue
                        
#                     # 使用tokenizer处理文本
#                     encoding = self.tokenizer.encode_plus(
#                         text,
#                         add_special_tokens=True,
#                         max_length=128,
#                         padding='max_length',
#                         truncation=True,
#                         return_attention_mask=True,
#                         return_tensors='pt'
#                     )
                    
#                     # 将数据移到设备
#                     input_ids = encoding['input_ids'].to(self.device)
#                     attention_mask = encoding['attention_mask'].to(self.device)
                    
#                     # 前向传播
#                     outputs = self.model(input_ids, attention_mask)
                    
#                     # 获取预测概率
#                     probs = torch.softmax(outputs, dim=1)
                    
#                     # 获取预测标签和概率
#                     label_id = torch.argmax(probs, dim=1).item()
#                     probability = probs[0, label_id].item()
                    
#                     # 添加结果
#                     results.append({
#                         'text': text,
#                         'sentiment_label': label_id,
#                         'probability': probability
#                     })
#         except Exception as e:
#             print(f"预测过程中出错: {e}")
#             # 返回默认结果
#             results = [{
#                 'text': text if isinstance(text, str) else '',
#                 'sentiment_label': 0,  # 默认为中性
#                 'probability': 0.0
#             } for text in texts]
        
#         # 转换为DataFrame
#         result_df = pd.DataFrame(results)
        
#         # 添加情感文本标签
#         if self.num_classes == 3:
#             sentiment_map = {0: '负面', 1: '中性', 2: '正面'}
#         elif self.num_classes == 5:
#             sentiment_map = {0: '非常负面', 1: '负面', 2: '中性', 3: '正面', 4: '非常正面'}
#         else:
#             sentiment_map = {i: str(i) for i in range(self.num_classes)}
        
#         result_df['sentiment_text'] = result_df['sentiment_label'].map(sentiment_map)
        
#         return result_df
    
#     def predict_dataframe(self, df, text_col='clean_content'):
#         """对DataFrame中的文本进行情感预测
        
#         Args:
#             df: 包含文本的DataFrame
#             text_col: 文本列名
            
#         Returns:
#             添加了情感预测结果的DataFrame
#         """
#         # 检查文本列是否存在
#         if text_col not in df.columns:
#             print(f"错误: 文本列 '{text_col}' 不存在于数据中")
#             print(f"可用列: {df.columns.tolist()}")
#             # 创建一个空的结果列
#             result_df = df.copy()
#             result_df['sentiment_label'] = 0
#             result_df['sentiment_prob'] = 0.0
#             result_df['sentiment_text'] = '中性'
#             return result_df
        
#         # 复制DataFrame
#         result_df = df.copy()
        
#         # 获取文本列表
#         texts = result_df[text_col].fillna('').tolist()
        
#         # 批量预测
#         print(f"开始预测 {len(texts)} 条文本的情感...")
#         predictions = self.predict(texts)
        
#         # 将预测结果添加到DataFrame
#         result_df['sentiment_label'] = predictions['sentiment_label']
#         result_df['sentiment_prob'] = predictions['probability']
#         result_df['sentiment_text'] = predictions['sentiment_text']
        
#         print("情感预测完成。")
#         return result_df
    
#     def convert_score_to_sentiment(self, df, score_col='score', score_mapping=None):
#         """将评分转换为情感标签
        
#         Args:
#             df: 包含评分的DataFrame
#             score_col: 评分列名
#             score_mapping: 评分到情感标签的映射字典，默认为None
            
#         Returns:
#             添加了情感标签的DataFrame
#         """
#         # 检查评分列是否存在
#         if score_col not in df.columns:
#             print(f"错误: 评分列 '{score_col}' 不存在于数据中")
#             print(f"可用列: {df.columns.tolist()}")
#             # 创建一个空的结果列
#             result_df = df.copy()
#             result_df['sentiment_label'] = 0
#             result_df['sentiment_text'] = '中性'
#             return result_df
        
#         # 复制DataFrame
#         result_df = df.copy()
        
#         # 默认评分映射（5分制）
#         if score_mapping is None:
#             if self.num_classes == 3:
#                 # 3分类：1-2分为负面，3分为中性，4-5分为正面
#                 score_mapping = {
#                     1: 0, 2: 0,  # 负面
#                     3: 1,        # 中性
#                     4: 2, 5: 2   # 正面
#                 }
#             elif self.num_classes == 5:
#                 # 5分类：直接映射
#                 score_mapping = {
#                     1: 0,  # 非常负面
#                     2: 1,  # 负面
#                     3: 2,  # 中性
#                     4: 3,  # 正面
#                     5: 4   # 非常正面
#                 }
        
#         # 应用映射
#         result_df['sentiment_label'] = result_df[score_col].map(score_mapping)
        
#         # 处理可能的NaN值
#         result_df['sentiment_label'] = result_df['sentiment_label'].fillna(1).astype(int)  # 默认为中性(1)
        
#         # 添加情感文本标签
#         if self.num_classes == 3:
#             sentiment_map = {0: '负面', 1: '中性', 2: '正面'}
#         elif self.num_classes == 5:
#             sentiment_map = {0: '非常负面', 1: '负面', 2: '中性', 3: '正面', 4: '非常正面'}
#         else:
#             sentiment_map = {i: str(i) for i in range(self.num_classes)}
        
#         result_df['sentiment_text'] = result_df['sentiment_label'].map(sentiment_map)
        
#         return result_df

# # ====================== 4. 可视化模块 ======================

# class SentimentVisualizer:
#     """情感分析可视化类"""
    
#     def __init__(self, font_path=None):
#         """初始化可视化器
        
#         Args:
#             font_path: 中文字体路径
#         """
#         # 尝试设置中文字体
#         try:
#             # 尝试在项目目录中查找字体
#             font_paths = [
#                 'static/fonts/SimHei.ttf',
#                 'SimHei.ttf',
#                 '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # Linux常见路径
#             ]
            
#             font_found = False
#             for path in font_paths:
#                 if os.path.exists(path):
#                     self.font = FontProperties(fname=path)
#                     font_found = True
#                     break
            
#             if not font_found:
#                 font = FontProperties(family=['sans'])
#                 print("警告: 找不到中文字体文件，可视化图表中的中文可能无法正确显示")
#         except:
#             font = FontProperties(family=['sans'])
#             print("警告: 设置中文字体时出错，可视化图表中的中文可能无法正确显示")
            
#         plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
#         plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
#         # 保存图表的目录
#         os.makedirs('static/visualizations', exist_ok=True)
    
#     def plot_sentiment_distribution(self, df, sentiment_col='sentiment_text', platform_col=None, 
#                                    title='情感分布', figsize=(10, 6), save_path=None):
#         """绘制情感分布图
        
#         Args:
#             df: 包含情感标签的DataFrame
#             sentiment_col: 情感文本标签列名
#             platform_col: 平台列名，用于按平台分组
#             title: 图表标题
#             figsize: 图表大小
#             save_path: 保存路径
#         """
#         try:
#             # 检查必要的列是否存在
#             if sentiment_col not in df.columns:
#                 print(f"错误: 情感列 '{sentiment_col}' 不存在于数据中")
#                 return
                
#             if platform_col and platform_col not in df.columns:
#                 print(f"警告: 平台列 '{platform_col}' 不存在于数据中，将绘制整体情感分布")
#                 platform_col = None
            
#             plt.figure(figsize=figsize)
            
#             if platform_col and platform_col in df.columns:
#                 # 按平台分组统计情感分布
#                 sentiment_counts = df.groupby([platform_col, sentiment_col]).size().unstack(fill_value=0)
#                 sentiment_counts.plot(kind='bar', stacked=True, colormap='viridis')
#                 plt.xlabel('平台', fontproperties=self.font)
#             else:
#                 # 统计整体情感分布
#                 sentiment_counts = df[sentiment_col].value_counts()
#                 sentiment_counts.plot(kind='bar', color=sns.color_palette('viridis', len(sentiment_counts)))
#                 plt.xlabel('情感类别', fontproperties=self.font)
            
#             plt.ylabel('评论数量', fontproperties=self.font)
#             plt.title(title, fontproperties=self.font)
#             plt.xticks(rotation=45)
#             plt.tight_layout()
            
#             if save_path:
#                 plt.savefig(save_path, dpi=300)
#                 print(f"情感分布图已保存到 {save_path}")
            
#             plt.close()
#         except Exception as e:
#             print(f"绘制情感分布图出错: {e}")
    
#     def plot_sentiment_pie(self, df, sentiment_col='sentiment_text', title='情感占比', 
#                           figsize=(8, 8), save_path=None):
#         """绘制情感占比饼图
        
#         Args:
#             df: 包含情感标签的DataFrame
#             sentiment_col: 情感文本标签列名
#             title: 图表标题
#             figsize: 图表大小
#             save_path: 保存路径
#         """
#         try:
#             # 检查必要的列是否存在
#             if sentiment_col not in df.columns:
#                 print(f"错误: 情感列 '{sentiment_col}' 不存在于数据中")
#                 return
                
#             plt.figure(figsize=figsize)
            
#             # 统计情感占比
#             sentiment_counts = df[sentiment_col].value_counts()
            
#             # 绘制饼图
#             plt.pie(
#                 sentiment_counts, 
#                 labels=sentiment_counts.index,
#                 autopct='%1.1f%%',
#                 startangle=90,
#                 colors=sns.color_palette('viridis', len(sentiment_counts))
#             )
            
#             plt.title(title, fontproperties=self.font)
#             plt.axis('equal')  # 确保饼图是圆形的
            
#             if save_path:
#                 plt.savefig(save_path, dpi=300)
#                 print(f"情感占比饼图已保存到 {save_path}")
            
#             plt.close()
#         except Exception as e:
#             print(f"绘制情感占比饼图出错: {e}")
    
#     def plot_platform_comparison(self, df, sentiment_col='sentiment_text', platform_col='platform',
#                                 title='不同平台情感对比', figsize=(12, 6), save_path=None):
#         """绘制不同平台情感对比图
        
#         Args:
#             df: 包含情感标签和平台的DataFrame
#             sentiment_col: 情感文本标签列名
#             platform_col: 平台列名
#             title: 图表标题
#             figsize: 图表大小
#             save_path: 保存路径
#         """
#         try:
#             # 检查必要的列是否存在
#             if sentiment_col not in df.columns:
#                 print(f"错误: 情感列 '{sentiment_col}' 不存在于数据中")
#                 return
                
#             if platform_col not in df.columns:
#                 print(f"错误: 平台列 '{platform_col}' 不存在于数据中")
#                 return
            
#             # 检查是否有足够的数据
#             if len(df[platform_col].unique()) < 2:
#                 print(f"警告: 数据中只有一个平台，无法进行平台对比")
#                 return
                
#             plt.figure(figsize=figsize)
            
#             # 按平台和情感分组计算百分比
#             platform_sentiment = df.groupby([platform_col, sentiment_col]).size().unstack(fill_value=0)
#             platform_sentiment_pct = platform_sentiment.div(platform_sentiment.sum(axis=1), axis=0) * 100
            
#             # 绘制堆叠条形图
#             platform_sentiment_pct.plot(kind='bar', stacked=True, colormap='viridis')
            
#             plt.xlabel('平台', fontproperties=self.font)
#             plt.ylabel('占比(%)', fontproperties=self.font)
#             plt.title(title, fontproperties=self.font)
#             plt.xticks(rotation=45)
#             plt.legend(title='情感类别', prop=self.font)
#             plt.tight_layout()
            
#             if save_path:
#                 plt.savefig(save_path, dpi=300)
#                 print(f"平台情感对比图已保存到 {save_path}")
            
#             plt.close()
#         except Exception as e:
#             print(f"绘制平台情感对比图出错: {e}")
    
#     def plot_wordcloud(self, df, words_col='words', sentiment_col='sentiment_text', 
#                       sentiment_value=None, title='词云图', figsize=(10, 8), 
#                       max_words=100, save_path=None):
#         """绘制词云图
        
#         Args:
#             df: 包含分词结果和情感标签的DataFrame
#             words_col: 分词结果列名
#             sentiment_col: 情感文本标签列名
#             sentiment_value: 指定情感类别，为None则使用所有评论
#             title: 图表标题
#             figsize: 图表大小
#             max_words: 最大词数
#             save_path: 保存路径
#         """
#         try:
#             # 检查必要的列是否存在
#             if words_col not in df.columns:
#                 print(f"错误: 分词列 '{words_col}' 不存在于数据中")
#                 return
            
#             if sentiment_value is not None and sentiment_col not in df.columns:
#                 print(f"错误: 情感列 '{sentiment_col}' 不存在于数据中")
#                 return
                
#             plt.figure(figsize=figsize)
            
#             # 筛选特定情感的评论
#             if sentiment_value is not None:
#                 filtered_df = df[df[sentiment_col] == sentiment_value]
#                 if filtered_df.empty:
#                     print(f"警告: 没有'{sentiment_value}'情感的评论")
#                     return
#                 title = f"{sentiment_value}评论 - {title}"
#             else:
#                 filtered_df = df
            
#             # 合并所有词
#             all_words = []
#             for words in filtered_df[words_col]:
#                 if isinstance(words, list):
#                     all_words.extend(words)
#                 elif isinstance(words, str):
#                     # 处理存储为字符串的情况
#                     words_list = words.split()
#                     all_words.extend(words_list)
            
#             # 统计词频
#             word_freq = {}
#             for word in all_words:
#                 if isinstance(word, str) and len(word) > 1:  # 忽略单字词
#                     word_freq[word] = word_freq.get(word, 0) + 1
            
#             # 创建词云
#             if word_freq:
#                 # 查找字体
#                 font_paths = [
#                     'static/fonts/SimHei.ttf',
#                     'SimHei.ttf',
#                     '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # Linux常见路径
#                 ]
                
#                 font_path = None
#                 for path in font_paths:
#                     if os.path.exists(path):
#                         font_path = path
#                         break
                
#                 wordcloud = WordCloud(
#                     font_path=font_path,
#                     max_words=max_words,
#                     width=800,
#                     height=600,
#                     background_color='white'
#                 ).generate_from_frequencies(word_freq)
                
#                 plt.imshow(wordcloud, interpolation='bilinear')
#                 plt.axis('off')
#                 plt.title(title, fontproperties=self.font)
                
#                 if save_path:
#                     plt.savefig(save_path, dpi=300)
#                     print(f"词云图已保存到 {save_path}")
                
#                 plt.close()
#             else:
#                 print("没有足够的词生成词云")
#         except Exception as e:
#             print(f"绘制词云图出错: {e}")
    
#     def plot_sentiment_trend(self, df, time_col='time', sentiment_col='sentiment_label', 
#                             platform_col=None, title='情感趋势分析', figsize=(12, 6), 
#                             save_path=None):
#         """绘制情感趋势图
        
#         Args:
#             df: 包含时间和情感标签的DataFrame
#             time_col: 时间列名
#             sentiment_col: 情感标签列名
#             platform_col: 平台列名，用于按平台分组
#             title: 图表标题
#             figsize: 图表大小
#             save_path: 保存路径
#         """
#         try:
#             # 检查必要的列是否存在
#             if time_col not in df.columns:
#                 print(f"错误: 时间列 '{time_col}' 不存在于数据中")
#                 return
                
#             if sentiment_col not in df.columns:
#                 print(f"错误: 情感列 '{sentiment_col}' 不存在于数据中")
#                 return
                
#             if platform_col and platform_col not in df.columns:
#                 print(f"警告: 平台列 '{platform_col}' 不存在于数据中，将绘制整体情感趋势")
#                 platform_col = None
                
#             plt.figure(figsize=figsize)
            
#             # 确保时间列是datetime类型
#             df = df.copy()
#             if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
#                 df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
#             # 过滤掉无效的时间
#             df = df.dropna(subset=[time_col])
            
#             if df.empty:
#                 print("警告: 过滤无效时间后，没有足够的数据绘制情感趋势图")
#                 return
            
#             # 按日期分组计算平均情感得分
#             df['date'] = df[time_col].dt.date
            
#             if platform_col and platform_col in df.columns:
#                 # 按平台和日期分组
#                 sentiment_trend = df.groupby([platform_col, 'date'])[sentiment_col].mean().unstack(0, fill_value=None)
#                 sentiment_trend.plot(marker='o', linestyle='-')
#                 plt.legend(title='平台')
#             else:
#                 # 按日期分组
#                 sentiment_trend = df.groupby('date')[sentiment_col].mean()
#                 sentiment_trend.plot(marker='o', linestyle='-', color='blue')
            
#             plt.xlabel('日期', fontproperties=self.font)
#             plt.ylabel('平均情感得分', fontproperties=self.font)
#             plt.title(title, fontproperties=self.font)
#             plt.grid(True, linestyle='--', alpha=0.7)
#             plt.tight_layout()
            
#             if save_path:
#                 plt.savefig(save_path, dpi=300)
#                 print(f"情感趋势图已保存到 {save_path}")
            
#             plt.close()
#         except Exception as e:
#             print(f"绘制情感趋势图出错: {e}")
    
#     def plot_confusion_matrix(self, cm, class_names, title='混淆矩阵', 
#                              figsize=(8, 6), save_path=None):
#         """绘制混淆矩阵热力图
        
#         Args:
#             cm: 混淆矩阵
#             class_names: 类别名称
#             title: 图表标题
#             figsize: 图表大小
#             save_path: 保存路径
#         """
#         try:
#             plt.figure(figsize=figsize)
            
#             # 计算准确率
#             cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
#             # 绘制热力图
#             sns.heatmap(
#                 cm_normalized, 
#                 annot=True, 
#                 cmap='Blues', 
#                 fmt='.2f',
#                 xticklabels=class_names,
#                 yticklabels=class_names
#             )
            
#             plt.xlabel('预测标签', fontproperties=self.font)
#             plt.ylabel('真实标签', fontproperties=self.font)
#             plt.title(title, fontproperties=self.font)
#             plt.tight_layout()
            
#             if save_path:
#                 plt.savefig(save_path, dpi=300)
#                 print(f"混淆矩阵热力图已保存到 {save_path}")
            
#             plt.close()
#         except Exception as e:
#             print(f"绘制混淆矩阵热力图出错: {e}")

# # ====================== 5. 主程序 ======================

# def main():
#     """主程序"""
#     print("="*50)
#     print("基于BERT的多源数据情感分析系统")
#     print("="*50)
    
#     # 1. 数据爬取
#     print("\n1. 数据爬取阶段")
#     crawler = MultiSourceCrawler()
    
#     # 设置要爬取的ID
#     movie_ids = ['35267208']  # 电影《流浪地球2》的豆瓣ID
#     product_ids = ['100035955048']  # 京东商品ID
#     song_ids = ['1999869183']  # 网易云音乐歌曲ID
    
#     # 爬取所有来源的评论
#     reviews_df = crawler.crawl_all_sources(
#         movie_ids=movie_ids,
#         product_ids=product_ids,
#         song_ids=song_ids,
#         pages=3  # 每个ID爬取3页
#     )
    
#     # 2. 数据预处理
#     print("\n2. 数据预处理阶段")
#     preprocessor = TextPreprocessor()
#     processed_df = preprocessor.process_dataframe(reviews_df)
#     preprocessor.save_processed_data(processed_df)
    
#     # 3. 情感分析
#     print("\n3. 情感分析阶段")
    
#     # 将评分转换为情感标签（用于训练）
#     analyzer = SentimentAnalyzer(num_classes=3)  # 3分类：负面、中性、正面
#     labeled_df = analyzer.convert_score_to_sentiment(processed_df)
    
#     # 划分训练和测试数据
#     train_loader, test_loader = analyzer.prepare_data(labeled_df)
    
#     # 训练模型
#     analyzer.train(train_loader, epochs=2)  # 实际应用中可以增加轮数
    
#     # 评估模型
#     accuracy, report, cm = analyzer.evaluate(test_loader)
    
#     # 对所有评论进行情感预测
#     prediction_df = analyzer.predict_dataframe(processed_df)
    
#     # 4. 可视化结果
#     print("\n4. 可视化结果阶段")
#     visualizer = SentimentVisualizer()
    
#     # 情感分布饼图
#     visualizer.plot_sentiment_pie(
#         prediction_df, 
#         save_path='static/visualizations/sentiment_pie.png'
#     )
    
#     # 不同平台情感对比
#     visualizer.plot_platform_comparison(
#         prediction_df,
#         save_path='static/visualizations/platform_comparison.png'
#     )
    
#     # 情感词云
#     visualizer.plot_wordcloud(
#         prediction_df, 
#         sentiment_value='正面',
#         save_path='static/visualizations/positive_wordcloud.png'
#     )
    
#     visualizer.plot_wordcloud(
#         prediction_df, 
#         sentiment_value='负面',
#         save_path='static/visualizations/negative_wordcloud.png'
#     )
    
#     # 情感趋势
#     visualizer.plot_sentiment_trend(
#         prediction_df,
#         save_path='static/visualizations/sentiment_trend.png'
#     )
    
#     # 混淆矩阵
#     if cm is not None:
#         visualizer.plot_confusion_matrix(
#             cm, 
#             class_names=['负面', '中性', '正面'],
#             save_path='static/visualizations/confusion_matrix.png'
#         )
    
#     print("\n分析完成！结果已保存到对应目录。")

# if __name__ == '__main__':
#     main()


'''
基于BERT的多源数据情感分析系统 - 增强版
功能模块：
1. 多平台数据爬取模块(豆瓣、电商、网易云音乐等)
2. 数据预处理与清洗模块
3. BERT模型训练与优化模块
4. 情感分析与分类模块
5. 结果可视化与分析展示模块
6. 命名实体识别模块
7. 依存句法分析模块
8. 关键词抽取模块
9. 文本摘要模块
10. 主题建模模块
11. 方面级情感分析模块
'''

import os
import re
import json
import time
import random
import jieba
import jieba.posseg as pseg
from jieba.analyse import textrank, extract_tags
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter, defaultdict
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertModel, BertForTokenClassification, 
    BertForSequenceClassification, BertForQuestionAnswering,
    pipeline, AutoTokenizer, AutoModel,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

import requests
from bs4 import BeautifulSoup
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 尝试加载HanLP (如果可用)
try:
    import hanlp
    HANLP_AVAILABLE = True
except ImportError:
    HANLP_AVAILABLE = False
    print("HanLP未安装，部分高级功能将不可用。可使用 pip install hanlp 安装")

# 尝试加载LAC (如果可用)
try:
    from LAC import LAC
    LAC_AVAILABLE = True
except ImportError:
    LAC_AVAILABLE = False
    print("LAC未安装，部分功能将使用替代实现。可使用 pip install lac 安装")

# 确保所需目录存在
os.makedirs('data', exist_ok=True)
os.makedirs('data/advanced', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/visualizations', exist_ok=True)
os.makedirs('static/visualizations/advanced', exist_ok=True)
os.makedirs('static/fonts', exist_ok=True)

# 设置中文字体
try:
    # 尝试在项目目录中查找字体
    font_paths = [
        'static/fonts/SimHei.ttf',
        'SimHei.ttf',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # Linux常见路径
    ]
    
    font_found = False
    for path in font_paths:
        if os.path.exists(path):
            font = FontProperties(fname=path)
            font_found = True
            break
    
    if not font_found:
        font = FontProperties(family=['sans'])
        print("警告: 找不到中文字体文件，将使用系统默认字体")
except:
    font = FontProperties(family=['sans'])
    print("警告: 设置中文字体时出错，将使用系统默认字体")

# 尝试设置matplotlib中文支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    print("警告: 设置matplotlib中文支持时出错")

# ====================== 1. 爬虫模块 ======================

class MultiSourceCrawler:
    """多源数据爬取类，包含豆瓣电影、京东商品、网易云音乐评论爬取"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'
        }
        # 创建保存数据的目录
        os.makedirs('data', exist_ok=True)
    
    def random_sleep(self):
        """随机休眠，避免被反爬"""
        time.sleep(random.uniform(1, 3))
    
    def crawl_douban_movie_reviews(self, movie_id, pages=10):
        """爬取豆瓣电影评论
        
        Args:
            movie_id: 豆瓣电影ID
            pages: 爬取页数
            
        Returns:
            评论数据列表
        """
        print(f"开始爬取豆瓣电影 {movie_id} 的评论...")
        reviews = []
        
        for page in range(pages):
            url = f"https://movie.douban.com/subject/{movie_id}/comments?start={page*20}&limit=20&sort=new_score&status=P"
            
            try:
                response = requests.get(url, headers=self.headers)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    comment_items = soup.select('.comment-item')
                    
                    for item in comment_items:
                        try:
                            comment = item.select_one('.comment p').text.strip()
                            rating = item.select_one('.rating')
                            score = 0
                            if rating:
                                score_class = rating.get('class')
                                if score_class:
                                    for cls in score_class:
                                        if 'allstar' in cls:
                                            score = int(cls.replace('allstar', '')) // 10
                                            break
                            
                            # 获取评论者信息
                            user_info = item.select_one('.comment-info a').text.strip() if item.select_one('.comment-info a') else "未知用户"
                            comment_time = item.select_one('.comment-time').text.strip() if item.select_one('.comment-time') else ""
                            
                            reviews.append({
                                'platform': '豆瓣电影',
                                'content': comment,
                                'score': score,
                                'user': user_info,
                                'time': comment_time,
                                'movie_id': movie_id
                            })
                        except Exception as e:
                            print(f"处理豆瓣评论项时出错: {e}")
                            continue
                    
                    print(f"已爬取豆瓣电影第 {page+1} 页评论")
                    self.random_sleep()
                else:
                    print(f"请求失败，状态码: {response.status_code}")
                    break
            except Exception as e:
                print(f"爬取豆瓣电影评论出错: {e}")
                continue
        
        # 保存数据
        if reviews:
            df = pd.DataFrame(reviews)
            output_path = f'data/douban_movie_{movie_id}_reviews.csv'
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"豆瓣电影 {movie_id} 的评论已保存到 {output_path}，共 {len(reviews)} 条")
        else:
            print(f"未获取到豆瓣电影 {movie_id} 的评论数据")
        
        return reviews
    
    def crawl_jd_product_reviews(self, product_id, pages=10):
        """爬取京东商品评论"""
        print(f"开始爬取京东商品 {product_id} 的评论...")
        reviews = []
        
        for page in range(pages):
            # 更新URL结构，确保获取JSON数据
            url = f"https://club.jd.com/comment/productPageComments.action?productId={product_id}&score=0&sortType=5&page={page}&pageSize=10&isShadowSku=0&fold=1"
            
            try:
                # 添加更多headers模拟真实浏览器
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
                    'Referer': f'https://item.jd.com/{product_id}.html',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'Connection': 'keep-alive',
                    'Cache-Control': 'max-age=0'
                }
                
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    try:
                        # 尝试直接作为JSON解析
                        data = response.json()
                    except:
                        # 如果不是纯JSON，尝试提取JSONP中的JSON部分
                        text = response.text
                        json_str = re.search(r'fetchJSON_comment\w*\((.*)\);', text)
                        if json_str:
                            data = json.loads(json_str.group(1))
                        else:
                            print(f"无法解析京东评论数据，页面 {page}")
                            continue
                    
                    comment_list = data.get('comments', [])
                    for comment in comment_list:
                        reviews.append({
                            'platform': '京东商品',
                            'content': comment.get('content', ''),
                            'score': comment.get('score', 0),
                            'user': comment.get('nickname', ''),
                            'time': comment.get('creationTime', ''),
                            'product_id': product_id
                        })
                    
                    print(f"已爬取京东商品第 {page+1} 页评论，获得 {len(comment_list)} 条")
                    # 随机延迟，防止被反爬
                    sleep_time = random.uniform(1.5, 3.5)
                    print(f"随机等待 {sleep_time:.2f} 秒...")
                    time.sleep(sleep_time)
                else:
                    print(f"请求失败，状态码: {response.status_code}")
                    # 返回请求内容以便调试
                    print(f"响应内容: {response.text[:200]}...")
                    break
            except Exception as e:
                print(f"爬取京东商品评论出错: {e}")
                continue
        
        # 保存数据
        if reviews:
            df = pd.DataFrame(reviews)
            output_path = f'data/jd_product_{product_id}_reviews.csv'
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"京东商品 {product_id} 的评论已保存到 {output_path}，共 {len(reviews)} 条")
        else:
            print(f"未获取到京东商品 {product_id} 的评论数据")
        
        return reviews
    
    def crawl_netease_music_comments(self, song_id, pages=10):
        """爬取网易云音乐评论
        
        Args:
            song_id: 歌曲ID
            pages: 爬取页数
            
        Returns:
            评论数据列表
        """
        print(f"开始爬取网易云音乐 {song_id} 的评论...")
        reviews = []
        
        # 注意：网易云音乐评论API需要特殊处理，这里使用简化模拟
        for page in range(pages):
            url = f"https://music.163.com/api/v1/resource/comments/R_SO_4_{song_id}?limit=20&offset={page*20}"
            
            try:
                response = requests.get(url, headers=self.headers)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        comments = data.get('comments', [])
                        
                        for comment in comments:
                            content = comment.get('content', '')
                            user = comment.get('user', {}).get('nickname', '')
                            time_str = comment.get('time', 0)
                            time_fmt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_str/1000)) if time_str else ''
                            
                            # 点赞数作为情感参考
                            like_count = comment.get('likedCount', 0)
                            # 简化处理：点赞数0-10为1分，10-50为2分，50-200为3分，200-1000为4分，1000以上为5分
                            score = 1
                            if like_count > 10:
                                score = 2
                            if like_count > 50:
                                score = 3
                            if like_count > 200:
                                score = 4
                            if like_count > 1000:
                                score = 5
                            
                            reviews.append({
                                'platform': '网易云音乐',
                                'content': content,
                                'score': score,
                                'user': user,
                                'time': time_fmt,
                                'song_id': song_id
                            })
                    except Exception as e:
                        print(f"解析网易云音乐评论数据出错: {e}")
                        continue
                    
                    print(f"已爬取网易云音乐第 {page+1} 页评论")
                    self.random_sleep()
                else:
                    print(f"请求失败，状态码: {response.status_code}")
                    break
            except Exception as e:
                print(f"爬取网易云音乐评论出错: {e}")
                continue
        
        # 保存数据
        if reviews:
            df = pd.DataFrame(reviews)
            output_path = f'data/netease_music_{song_id}_reviews.csv'
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"网易云音乐 {song_id} 的评论已保存到 {output_path}，共 {len(reviews)} 条")
        else:
            print(f"未获取到网易云音乐 {song_id} 的评论数据")
        
        return reviews
    
    def crawl_all_sources(self, movie_ids=None, product_ids=None, song_ids=None, pages=5):
        """爬取所有数据源的评论
        
        Args:
            movie_ids: 豆瓣电影ID列表
            product_ids: 京东商品ID列表
            song_ids: 网易云音乐歌曲ID列表
            pages: 每个ID爬取的页数
        
        Returns:
            所有评论数据的DataFrame
        """
        all_reviews = []
        
        # 豆瓣电影评论
        if movie_ids:
            for movie_id in movie_ids:
                reviews = self.crawl_douban_movie_reviews(movie_id, pages)
                all_reviews.extend(reviews)
        
        # 京东商品评论
        if product_ids:
            for product_id in product_ids:
                reviews = self.crawl_jd_product_reviews(product_id, pages)
                all_reviews.extend(reviews)
        
        # 网易云音乐评论
        if song_ids:
            for song_id in song_ids:
                reviews = self.crawl_netease_music_comments(song_id, pages)
                all_reviews.extend(reviews)
        
        # 合并所有数据
        if all_reviews:
            df = pd.DataFrame(all_reviews)
            output_path = 'data/all_reviews.csv'
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"所有评论数据已保存到 {output_path}，共 {len(all_reviews)} 条")
            return df
        else:
            print("未获取到任何评论数据")
            return pd.DataFrame()

# ====================== 2. 数据预处理模块 ======================

class TextPreprocessor:
    """文本预处理类，包含文本清洗、分词、去停用词等功能"""
    
    def __init__(self, stopwords_path=None):
        """初始化预处理器
        
        Args:
            stopwords_path: 停用词表路径，不提供则使用默认的停用词列表
        """
        # 加载停用词表
        self.stopwords = set()
        if stopwords_path and os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                self.stopwords = set([line.strip() for line in f.readlines()])
        else:
            # 使用默认的简单停用词表
            self.stopwords = set(['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', 
                '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '啊', '吧', '把', 
                '但', '但是', '并', '个', '给', '过', '还', '还是', '还有', '其', '其实', '其中', '几', '可', '可以', 
                '可是', '么', '没', '什么', '什么样', '这样', '那样', '之', '之一', '只', '只是', '只要', '只有', 
                '就是', '就是说', '打', '呢', '来', '来说', '来自', '哪儿', '哪里', '如', '如果', '如何', '如此', 
                '对', '对于', '比', '多', '多少', '而', '而且', '被', '该', '应', '应该'])
        
        # 确保jieba加载成功
        try:
            jieba.initialize()
            print("成功初始化jieba分词")
        except Exception as e:
            print(f"初始化jieba分词出错: {e}")
        
        # 加载结巴分词用户词典
        # jieba.load_userdict('user_dict.txt')  # 如果有自定义词典，可以取消注释
    
    def clean_text(self, text):
        """清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not isinstance(text, str) or not text:
            return ''
        
        try:
            # 去除HTML标签
            text = re.sub(r'<[^>]+>', '', text)
            # 去除网址
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            # 去除特殊字符和标点符号，但保留中文标点
            text = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', '', text)
            # 去除数字
            text = re.sub(r'\d+', '', text)
            # 去除多余空白字符
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            print(f"清洗文本出错: {e}")
            return text if isinstance(text, str) else ''
    
    def segment(self, text):
        """中文分词
        
        Args:
            text: 清洗后的文本
            
        Returns:
            分词结果列表
        """
        if not text:
            return []
            
        try:
            # 使用结巴分词
            words = jieba.lcut(text)
            # 去除停用词
            words = [word for word in words if len(word) > 1 and word not in self.stopwords]
            
            return words
        except Exception as e:
            print(f"分词出错: {e}")
            return []
    
    def process_dataframe(self, df, content_col='content'):
        """处理DataFrame中的文本数据
        
        Args:
            df: 包含文本数据的DataFrame
            content_col: 文本内容的列名
            
        Returns:
            处理后的DataFrame，增加了清洗文本和分词结果列
        """
        # 复制一份，避免修改原始数据
        result_df = df.copy()
        
        # 确保content_col存在
        if content_col not in result_df.columns:
            print(f"错误: 列 '{content_col}' 不存在于数据中")
            if not result_df.empty:
                print(f"可用列: {result_df.columns.tolist()}")
            return result_df
        
        print(f"开始处理 {len(result_df)} 条文本数据...")
        
        # 清洗文本
        result_df['clean_content'] = result_df[content_col].apply(self.clean_text)
        
        # 分词
        result_df['words'] = result_df['clean_content'].apply(self.segment)
        
        # 计算分词后的词数
        result_df['word_count'] = result_df['words'].apply(len)
        
        print(f"文本处理完成。")
        return result_df
    
    def save_processed_data(self, df, output_path='data/processed_reviews.csv'):
        """保存处理后的数据
        
        Args:
            df: 处理后的DataFrame
            output_path: 输出文件路径
        """
        try:
            # 确保数据目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 将words列表转换为字符串，方便存储
            df['words_str'] = df['words'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
            
            # 保存到CSV
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"处理后的数据已保存到 {output_path}")
        except Exception as e:
            print(f"保存处理后的数据出错: {e}")

    def extract_linguistic_features(self, text):
        """提取高级语言学特征用于学术NLP分析"""
        features = {}
        # 情感词统计（使用预定义词典）
        pos_words = ['优秀', '精彩', '满意', '喜欢', '推荐', '好', '棒', '赞', '华丽', '精致', 
                    '震撼', '感人', '完美', '出色', '卓越', '超值', '高效', '实用', '方便']
        neg_words = ['差', '糟糕', '失望', '浪费', '后悔', '烂', '坑', '骗', '无用', '过时',
                    '难用', '枯燥', '乏味', '无聊', '劣质', '粗糙', '虚假', '敷衍']
        
        # 分词结果
        words = self.segment(text)
        
        # 词频特征
        features['正面词数量'] = sum(1 for word in words if word in pos_words)
        features['负面词数量'] = sum(1 for word in words if word in neg_words)
        features['平均词长'] = np.mean([len(word) for word in words]) if words else 0
        features['句子数量'] = len(re.split(r'[。！？]', text))
        
        # 情感强度词统计
        intensifiers = ['很', '非常', '太', '真是', '极其', '特别', '相当', '十分', '尤其', '格外']
        features['情感强度词数量'] = sum(1 for word in words if word in intensifiers)
        
        # 转折词统计
        transitions = ['但是', '然而', '却', '不过', '尽管', '虽然', '反而', '相反', '只是']
        features['转折词数量'] = sum(1 for i in range(len(text)-1) if text[i:i+2] in transitions)
        
        # 否定词统计
        negations = ['不', '没', '不是', '没有', '别', '莫', '未', '无']
        features['否定词数量'] = sum(1 for word in words if word in negations)
        
        return features

    def load_pretrained_dataset(self, dataset_name='ChnSentiCorp'):
        """加载标准中文情感数据集以获得更好的性能"""
        print(f"加载预训练数据集: {dataset_name}")
        
        if dataset_name == 'ChnSentiCorp':
            # 这个数据集包含约12k中文评论，带有二元情感标注
            try:
                df = pd.read_csv('data/ChnSentiCorp_htl_all.csv')
                df['sentiment_label'] = df['label'].map({0: 0, 1: 2})  # 映射到三分类系统
                df['sentiment_text'] = df['sentiment_label'].map({0: '负面', 1: '中性', 2: '正面'})
                return df
            except:
                print("无法加载ChnSentiCorp数据集，请确保文件已下载到data目录")
        
        elif dataset_name == 'online_shopping_10_cats':
            # 一个更大的数据集，包含跨10个产品类别的60k+评论
            try:
                df = pd.read_csv('data/online_shopping_10_cats.csv')
                # 将5星评分转换为情感类别
                df['sentiment_label'] = df['rating'].apply(
                    lambda x: 0 if x <= 2 else (1 if x == 3 else 2)
                )
                df['sentiment_text'] = df['sentiment_label'].map({0: '负面', 1: '中性', 2: '正面'})
                return df
            except:
                print("无法加载online_shopping_10_cats数据集，请确保文件已下载到data目录")
        
        # 如果没有请求的数据集，或加载失败，则创建并加载自带的小型示例数据集
        print("尝试加载示例数据集...")
        if not os.path.exists('data/sample_reviews.csv'):
            self._create_sample_dataset()
            
        try:
            df = pd.read_csv('data/sample_reviews.csv')
            print(f"加载示例数据集，共{len(df)}条评论")
            return df
        except:
            print("无法加载任何数据集，请检查data目录是否存在")
            return pd.DataFrame()

    def _create_sample_dataset(self):
        """创建示例数据集保存到CSV"""
        data = {
            'content': [
                "这部电影太棒了，演员演技很好，情节扣人心弦，强烈推荐！",
                "画面精美，但是剧情有点拖沓，人物塑造不够丰满。",
                "这是我看过最差的电影，浪费时间和金钱，剧情混乱，演技尴尬。",
                "音效不错，但是台词很生硬，整体感觉一般。",
                "特效炸裂，剧情紧凑，节奏把握得很好，是一部不可多得的佳作。",
                "虽然有些桥段不太合理，但整体来说是部不错的电影。",
                "剧情老套，演技浮夸，毫无创新可言，很失望。",
                "影片节奏紧凑，情节设计巧妙，结局出人意料，非常精彩。",
                "特效做得不错，但是剧情实在太过牵强，人物刻画也很单薄。",
                "音乐配得很到位，渲染了整个影片的氛围，值得一看。"
            ],
            'score': [5, 3, 1, 3, 5, 4, 2, 5, 2, 4],
            'platform': ['豆瓣电影'] * 10,
            'time': ['2023-01-01'] * 10
        }
        
        df = pd.DataFrame(data)
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/sample_reviews.csv', index=False, encoding='utf-8')
        print(f"示例数据集已保存到 data/sample_reviews.csv，共{len(df)}条评论")

# ====================== 3. BERT模型训练与情感分析模块 ======================

class SentimentDataset(Dataset):
    """情感分析数据集类"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: BERT tokenizer
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 使用BERT tokenizer编码文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertSentimentClassifier(nn.Module):
    """基于BERT的情感分类模型"""
    
    def __init__(self, bert_model_name, num_classes):
        """初始化模型
        
        Args:
            bert_model_name: BERT预训练模型名称
            num_classes: 类别数量
        """
        super(BertSentimentClassifier, self).__init__()
        
        try:
            self.bert = BertModel.from_pretrained(bert_model_name)
            self.dropout = nn.Dropout(0.1)
            self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
            print(f"成功初始化BERT模型: {bert_model_name}")
        except Exception as e:
            print(f"初始化BERT模型出错: {e}")
            # 创建一个空壳模型，避免程序崩溃
            self.bert = None
            self.dropout = nn.Dropout(0.1)
            self.fc = nn.Linear(768, num_classes)  # 使用默认BERT隐藏大小
    
    def forward(self, input_ids, attention_mask):
        """前向传播
        
        Args:
            input_ids: 输入ID序列
            attention_mask: 注意力掩码
            
        Returns:
            logits: 分类logits
        """
        if self.bert is None:
            # 如果BERT初始化失败，返回随机输出
            batch_size = input_ids.shape[0]
            return torch.randn(batch_size, self.fc.out_features)
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        
        return logits

class SentimentAnalyzer:
    """情感分析器类，包含模型训练和预测功能"""
    
    def __init__(self, bert_model_name='bert-base-chinese', num_classes=3, device=None):
        """初始化情感分析器
        
        Args:
            bert_model_name: BERT预训练模型名称
            num_classes: 情感类别数，默认为3（负面、中性、正面）
            device: 运行设备，默认自动选择
        """
        # 设置运行设备
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载BERT tokenizer
        try:
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            print(f"成功加载BERT tokenizer: {bert_model_name}")
        except Exception as e:
            print(f"加载BERT tokenizer出错: {e}")
            # 尝试加载本地保存的tokenizer
            try:
                if os.path.exists('models/tokenizer'):
                    self.tokenizer = BertTokenizer.from_pretrained('models/tokenizer')
                    print("已加载本地tokenizer")
                else:
                    print("警告: 无法加载BERT tokenizer，某些功能可能无法正常工作")
                    self.tokenizer = None
            except:
                print("警告: 无法加载BERT tokenizer，某些功能可能无法正常工作")
                self.tokenizer = None
        
        # 初始化BERT情感分类模型
        self.model = BertSentimentClassifier(bert_model_name, num_classes)
        self.model.to(self.device)
        
        # 模型保存路径
        self.model_save_path = 'models/bert_sentiment_model.pth'
        os.makedirs('models', exist_ok=True)
        
        # 类别数
        self.num_classes = num_classes
    
    def prepare_data(self, df, text_col='clean_content', label_col='sentiment_label', test_size=0.2):
        """准备训练和测试数据
        
        Args:
            df: 包含文本和情感标签的DataFrame
            text_col: 文本列名
            label_col: 标签列名
            test_size: 测试集比例
            
        Returns:
            训练和测试数据加载器
        """
        # 检查必要的列是否存在
        if text_col not in df.columns:
            print(f"错误: 文本列 '{text_col}' 不存在于数据中")
            print(f"可用列: {df.columns.tolist()}")
            return None, None
        
        if label_col not in df.columns:
            print(f"错误: 标签列 '{label_col}' 不存在于数据中")
            print(f"可用列: {df.columns.tolist()}")
            return None, None
        
        # 确保tokenizer已加载
        if self.tokenizer is None:
            print("错误: BERT tokenizer未加载，无法准备数据")
            return None, None
        
        texts = df[text_col].fillna('').tolist()
        labels = df[label_col].fillna(0).astype(int).tolist()
        
        # 检查标签是否有效
        if min(labels) < 0 or max(labels) >= self.num_classes:
            print(f"警告: 标签范围应为0-{self.num_classes-1}，但实际范围为{min(labels)}-{max(labels)}")
            # 修正标签
            labels = [max(0, min(l, self.num_classes-1)) for l in labels]
        
        # 划分训练集和测试集
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=42, stratify=labels
            )
            
            print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
            
            # 创建数据集
            train_dataset = SentimentDataset(X_train, y_train, self.tokenizer)
            test_dataset = SentimentDataset(X_test, y_test, self.tokenizer)
            
            # 创建数据加载器
            train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=16)
            
            return train_dataloader, test_dataloader
        except Exception as e:
            print(f"准备数据时出错: {e}")
            return None, None
    
    def train(self, train_dataloader, epochs=4, learning_rate=2e-5):
        """训练模型
        
        Args:
            train_dataloader: 训练数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
        """
        if train_dataloader is None:
            print("错误: 训练数据加载器为空，无法进行训练")
            return
        
        # 优化器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # 学习率调度器
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        self.model.train()
        
        try:
            for epoch in range(epochs):
                print(f"Epoch {epoch+1}/{epochs}")
                total_loss = 0
                
                # 进度条
                progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
                
                for batch in progress_bar:
                    # 准备数据
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # 清零梯度
                    optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = self.model(input_ids, attention_mask)
                    
                    # 计算损失
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    
                    # 反向传播
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # 更新参数
                    optimizer.step()
                    scheduler.step()
                    
                    # 更新进度条
                    progress_bar.set_postfix({'loss': loss.item()})
                
                avg_loss = total_loss / len(train_dataloader)
                print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            # 保存模型
            torch.save(self.model.state_dict(), self.model_save_path)
            print(f"模型已保存到 {self.model_save_path}")
            
        except Exception as e:
            print(f"训练过程中出错: {e}")
    
    def evaluate(self, test_dataloader):
        """评估模型
        
        Args:
            test_dataloader: 测试数据加载器
            
        Returns:
            评估结果（准确率、分类报告、混淆矩阵）
        """
        if test_dataloader is None:
            print("错误: 测试数据加载器为空，无法进行评估")
            return None, None, None
        
        self.model.eval()
        
        # 存储预测结果和真实标签
        predictions = []
        true_labels = []
        
        try:
            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc="Evaluating"):
                    # 准备数据
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # 前向传播
                    outputs = self.model(input_ids, attention_mask)
                    
                    # 获取预测结果
                    _, preds = torch.max(outputs, 1)
                    
                    # 保存结果
                    predictions.extend(preds.cpu().tolist())
                    true_labels.extend(labels.cpu().tolist())
            
            # 计算准确率
            accuracy = accuracy_score(true_labels, predictions)
            print(f"测试集准确率: {accuracy:.4f}")
            
            # 生成分类报告
            if self.num_classes == 3:
                target_names = ['负面', '中性', '正面']
            elif self.num_classes == 5:
                target_names = ['非常负面', '负面', '中性', '正面', '非常正面']
            else:
                target_names = [str(i) for i in range(self.num_classes)]
            
            report = classification_report(true_labels, predictions, target_names=target_names)
            print("分类报告:")
            print(report)
            
            # 混淆矩阵
            cm = confusion_matrix(true_labels, predictions)
            
            return accuracy, report, cm
        
        except Exception as e:
            print(f"评估过程中出错: {e}")
            return None, None, None
    
    def load_model(self):
        """加载已保存的模型"""
        if os.path.exists(self.model_save_path):
            try:
                self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
                self.model.eval()
                print(f"已加载模型 {self.model_save_path}")
                return True
            except Exception as e:
                print(f"加载模型出错: {e}")
                return False
        else:
            print(f"模型文件 {self.model_save_path} 不存在")
            return False
    
    def predict(self, texts):
        """预测文本情感
        
        Args:
            texts: 文本列表或单个文本
            
        Returns:
            预测结果（情感标签和概率）
        """
        # 确保tokenizer已加载
        if self.tokenizer is None:
            print("错误: BERT tokenizer未加载，无法进行预测")
            return pd.DataFrame({'text': texts if isinstance(texts, list) else [texts],
                                'sentiment_label': [0] * (len(texts) if isinstance(texts, list) else 1),
                                'probability': [0.0] * (len(texts) if isinstance(texts, list) else 1)})
        
        # 确保模型处于评估模式
        self.model.eval()
        
        # 处理单个文本的情况
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        
        try:
            with torch.no_grad():
                for text in texts:
                    if not text or not isinstance(text, str):
                        # 处理空文本或非字符串
                        results.append({
                            'text': text if isinstance(text, str) else '',
                            'sentiment_label': 0,  # 默认为中性
                            'probability': 0.0
                        })
                        continue
                        
                    # 使用tokenizer处理文本
                    encoding = self.tokenizer.encode_plus(
                        text,
                        add_special_tokens=True,
                        max_length=128,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                    )
                    
                    # 将数据移到设备
                    input_ids = encoding['input_ids'].to(self.device)
                    attention_mask = encoding['attention_mask'].to(self.device)
                    
                    # 前向传播
                    outputs = self.model(input_ids, attention_mask)
                    
                    # 获取预测概率
                    probs = torch.softmax(outputs, dim=1)
                    
                    # 获取预测标签和概率
                    label_id = torch.argmax(probs, dim=1).item()
                    probability = probs[0, label_id].item()
                    
                    # 添加结果
                    results.append({
                        'text': text,
                        'sentiment_label': label_id,
                        'probability': probability
                    })
        except Exception as e:
            print(f"预测过程中出错: {e}")
            # 返回默认结果
            results = [{
                'text': text if isinstance(text, str) else '',
                'sentiment_label': 0,  # 默认为中性
                'probability': 0.0
            } for text in texts]
        
        # 转换为DataFrame
        result_df = pd.DataFrame(results)
        
        # 添加情感文本标签
        if self.num_classes == 3:
            sentiment_map = {0: '负面', 1: '中性', 2: '正面'}
        elif self.num_classes == 5:
            sentiment_map = {0: '非常负面', 1: '负面', 2: '中性', 3: '正面', 4: '非常正面'}
        else:
            sentiment_map = {i: str(i) for i in range(self.num_classes)}
        
        result_df['sentiment_text'] = result_df['sentiment_label'].map(sentiment_map)
        
        return result_df
    
    def predict_dataframe(self, df, text_col='clean_content', progress_callback=None):
        """对DataFrame中的文本进行情感预测
        
        Args:
            df: 包含文本的DataFrame
            text_col: 文本列名
            progress_callback: 进度回调函数，接收 (current, total, message) 参数
            
        Returns:
            添加了情感预测结果的DataFrame
        """
        # 检查文本列是否存在
        if text_col not in df.columns:
            print(f"错误: 列 '{text_col}' 不存在于数据中")
            print(f"可用列: {df.columns.tolist()}")
            # 创建一个空的结果列
            result_df = df.copy()
            result_df['sentiment_label'] = 0
            result_df['sentiment_prob'] = 0.0
            result_df['sentiment_text'] = '中性'
            return result_df
        
        # 复制DataFrame
        result_df = df.copy()
        
        # 获取文本列表
        texts = result_df[text_col].fillna('').tolist()
        total = len(texts)
        
        print(f"开始预测 {total} 条文本的情感...")
        
        # 批量预测，使用分批处理避免内存溢出
        batch_size = 32
        all_predictions = []
        
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i+batch_size]
            predictions = self.predict(batch_texts)
            all_predictions.append(predictions)
            
            # 更新进度
            if progress_callback:
                current = min(i + batch_size, total)
                progress_callback(current, total, f'已分析 {current}/{total} 条文本...')
        
        # 合并所有预测结果
        predictions = pd.concat(all_predictions, ignore_index=True)
        
        # 将预测结果添加到DataFrame
        result_df['sentiment_label'] = predictions['sentiment_label']
        result_df['sentiment_prob'] = predictions['probability']
        result_df['sentiment_text'] = predictions['sentiment_text']
        
        # 计算准确率（如果有真实标签）
        if 'label' in result_df.columns:
            # 将标签转换为对应的模型预测标签
            # 假设label=1是正面(sentiment_label=2)，label=0是负面(sentiment_label=0)
            label_mapping = {1: 2, 0: 0}  # 1->正面(2), 0->负面(0)
            true_labels = result_df['label'].map(label_mapping).fillna(1).astype(int)  # 未知映射到中性(1)
            
            # 计算准确率
            matches = (true_labels == result_df['sentiment_label']).sum()
            accuracy = matches / len(result_df)
            result_df['accuracy'] = accuracy
            print(f"情感分析准确率: {accuracy:.4f}")
        
        print("情感预测完成。")
        return result_df
    
    def convert_score_to_sentiment(self, df, score_col='score', score_mapping=None):
        """将评分转换为情感标签
        
        Args:
            df: 包含评分的DataFrame
            score_col: 评分列名
            score_mapping: 评分到情感标签的映射字典，默认为None
            
        Returns:
            添加了情感标签的DataFrame
        """
        # 检查评分列是否存在
        if score_col not in df.columns:
            print(f"错误: 评分列 '{score_col}' 不存在于数据中")
            print(f"可用列: {df.columns.tolist()}")
            # 创建一个空的结果列
            result_df = df.copy()
            result_df['sentiment_label'] = 0
            result_df['sentiment_text'] = '中性'
            return result_df
        
        # 复制DataFrame
        result_df = df.copy()
        
        # 默认评分映射（5分制）
        if score_mapping is None:
            if self.num_classes == 3:
                # 3分类：1-2分为负面，3分为中性，4-5分为正面
                score_mapping = {
                    1: 0, 2: 0,  # 负面
                    3: 1,        # 中性
                    4: 2, 5: 2   # 正面
                }
            elif self.num_classes == 5:
                # 5分类：直接映射
                score_mapping = {
                    1: 0,  # 非常负面
                    2: 1,  # 负面
                    3: 2,  # 中性
                    4: 3,  # 正面
                    5: 4   # 非常正面
                }
        
        # 应用映射
        result_df['sentiment_label'] = result_df[score_col].map(score_mapping)
        
        # 处理可能的NaN值
        result_df['sentiment_label'] = result_df['sentiment_label'].fillna(1).astype(int)  # 默认为中性(1)
        
        # 添加情感文本标签
        if self.num_classes == 3:
            sentiment_map = {0: '负面', 1: '中性', 2: '正面'}
        elif self.num_classes == 5:
            sentiment_map = {0: '非常负面', 1: '负面', 2: '中性', 3: '正面', 4: '非常正面'}
        else:
            sentiment_map = {i: str(i) for i in range(self.num_classes)}
        
        result_df['sentiment_text'] = result_df['sentiment_label'].map(sentiment_map)
        
        return result_df

# ====================== 4. 可视化模块 ======================

class SentimentVisualizer:
    """情感分析可视化类"""
    
    def __init__(self, font_path=None):
        """初始化可视化器
        
        Args:
            font_path: 中文字体路径
        """
        # 尝试设置中文字体
        try:
            # 尝试在项目目录中查找字体
            font_paths = [
                'static/fonts/SimHei.ttf',
                'SimHei.ttf',
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # Linux常见路径
            ]
            
            font_found = False
            for path in font_paths:
                if os.path.exists(path):
                    self.font = FontProperties(fname=path)
                    font_found = True
                    break
            
            if not font_found:
                font = FontProperties(family=['sans'])
                print("警告: 找不到中文字体文件，可视化图表中的中文可能无法正确显示")
        except:
            font = FontProperties(family=['sans'])
            print("警告: 设置中文字体时出错，可视化图表中的中文可能无法正确显示")
            
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 保存图表的目录
        os.makedirs('static/visualizations', exist_ok=True)
    
    def plot_sentiment_distribution(self, df, sentiment_col='sentiment_text', platform_col=None, 
                                   title='情感分布', figsize=(10, 6), save_path=None):
        """绘制情感分布图
        
        Args:
            df: 包含情感标签的DataFrame
            sentiment_col: 情感文本标签列名
            platform_col: 平台列名，用于按平台分组
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        try:
            # 检查必要的列是否存在
            if sentiment_col not in df.columns:
                print(f"错误: 情感列 '{sentiment_col}' 不存在于数据中")
                return
                
            if platform_col and platform_col not in df.columns:
                print(f"警告: 平台列 '{platform_col}' 不存在于数据中，将绘制整体情感分布")
                platform_col = None
            
            plt.figure(figsize=figsize)
            
            if platform_col and platform_col in df.columns:
                # 按平台分组统计情感分布
                sentiment_counts = df.groupby([platform_col, sentiment_col]).size().unstack(fill_value=0)
                sentiment_counts.plot(kind='bar', stacked=True, colormap='viridis')
                plt.xlabel('平台', fontproperties=self.font)
            else:
                # 统计整体情感分布
                sentiment_counts = df[sentiment_col].value_counts()
                sentiment_counts.plot(kind='bar', color=sns.color_palette('viridis', len(sentiment_counts)))
                plt.xlabel('情感类别', fontproperties=self.font)
            
            plt.ylabel('评论数量', fontproperties=self.font)
            plt.title(title, fontproperties=self.font)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300)
                print(f"情感分布图已保存到 {save_path}")
            
            plt.close()
        except Exception as e:
            print(f"绘制情感分布图出错: {e}")
    
    def plot_sentiment_pie(self, df, sentiment_col='sentiment_text', title='情感占比', 
                          figsize=(8, 8), save_path=None):
        """绘制情感占比饼图
        
        Args:
            df: 包含情感标签的DataFrame
            sentiment_col: 情感文本标签列名
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        try:
            # 检查必要的列是否存在
            if sentiment_col not in df.columns:
                print(f"错误: 情感列 '{sentiment_col}' 不存在于数据中")
                return
                
            plt.figure(figsize=figsize)
            
            # 统计情感占比
            sentiment_counts = df[sentiment_col].value_counts()
            
            # 绘制饼图
            plt.pie(
                sentiment_counts, 
                labels=sentiment_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=sns.color_palette('viridis', len(sentiment_counts))
            )
            
            plt.title(title, fontproperties=self.font)
            plt.axis('equal')  # 确保饼图是圆形的
            
            if save_path:
                plt.savefig(save_path, dpi=300)
                print(f"情感占比饼图已保存到 {save_path}")
            
            plt.close()
        except Exception as e:
            print(f"绘制情感占比饼图出错: {e}")
    
    def plot_platform_comparison(self, df, sentiment_col='sentiment_text', platform_col='platform',
                                title='不同平台情感对比', figsize=(12, 6), save_path=None):
        """绘制不同平台情感对比图
        
        Args:
            df: 包含情感标签和平台的DataFrame
            sentiment_col: 情感文本标签列名
            platform_col: 平台列名
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        try:
            # 检查必要的列是否存在
            if sentiment_col not in df.columns:
                print(f"错误: 情感列 '{sentiment_col}' 不存在于数据中")
                return
                
            if platform_col not in df.columns:
                print(f"错误: 平台列 '{platform_col}' 不存在于数据中")
                return
            
            # 检查是否有足够的数据
            if len(df[platform_col].unique()) < 2:
                print(f"警告: 数据中只有一个平台，无法进行平台对比")
                return
                
            plt.figure(figsize=figsize)
            
            # 按平台和情感分组计算百分比
            platform_sentiment = df.groupby([platform_col, sentiment_col]).size().unstack(fill_value=0)
            platform_sentiment_pct = platform_sentiment.div(platform_sentiment.sum(axis=1), axis=0) * 100
            
            # 绘制堆叠条形图
            platform_sentiment_pct.plot(kind='bar', stacked=True, colormap='viridis')
            
            plt.xlabel('平台', fontproperties=self.font)
            plt.ylabel('占比(%)', fontproperties=self.font)
            plt.title(title, fontproperties=self.font)
            plt.xticks(rotation=45)
            plt.legend(title='情感类别', prop=self.font)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300)
                print(f"平台情感对比图已保存到 {save_path}")
            
            plt.close()
        except Exception as e:
            print(f"绘制平台情感对比图出错: {e}")
    
    def plot_wordcloud(self, df, words_col='words', sentiment_col='sentiment_text', 
                      sentiment_value=None, title='词云图', figsize=(10, 8), 
                      max_words=100, save_path=None):
        """绘制词云图
        
        Args:
            df: 包含分词结果和情感标签的DataFrame
            words_col: 分词结果列名
            sentiment_col: 情感文本标签列名
            sentiment_value: 指定情感类别，为None则使用所有评论
            title: 图表标题
            figsize: 图表大小
            max_words: 最大词数
            save_path: 保存路径
        """
        try:
            # 检查必要的列是否存在
            if words_col not in df.columns:
                print(f"错误: 分词列 '{words_col}' 不存在于数据中")
                return
            
            if sentiment_value is not None and sentiment_col not in df.columns:
                print(f"错误: 情感列 '{sentiment_col}' 不存在于数据中")
                return
                
            plt.figure(figsize=figsize)
            
            # 筛选特定情感的评论
            if sentiment_value is not None:
                filtered_df = df[df[sentiment_col] == sentiment_value]
                if filtered_df.empty:
                    print(f"警告: 没有'{sentiment_value}'情感的评论")
                    return
                title = f"{sentiment_value}评论 - {title}"
            else:
                filtered_df = df
            
            # 合并所有词
            all_words = []
            for words in filtered_df[words_col]:
                if isinstance(words, list):
                    all_words.extend(words)
                elif isinstance(words, str):
                    # 处理存储为字符串的情况
                    words_list = words.split()
                    all_words.extend(words_list)
            
            # 统计词频
            word_freq = {}
            for word in all_words:
                if isinstance(word, str) and len(word) > 1:  # 忽略单字词
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # 创建词云
            if word_freq:
                # 查找字体
                font_paths = [
                    'static/fonts/SimHei.ttf',
                    'SimHei.ttf',
                    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # Linux常见路径
                ]
                
                font_path = None
                for path in font_paths:
                    if os.path.exists(path):
                        font_path = path
                        break
                
                wordcloud = WordCloud(
                    font_path=font_path,
                    max_words=max_words,
                    width=800,
                    height=600,
                    background_color='white'
                ).generate_from_frequencies(word_freq)
                
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(title, fontproperties=self.font)
                
                if save_path:
                    plt.savefig(save_path, dpi=300)
                    print(f"词云图已保存到 {save_path}")
                
                plt.close()
            else:
                print("没有足够的词生成词云")
        except Exception as e:
            print(f"绘制词云图出错: {e}")
    
    def plot_sentiment_trend(self, df, time_col='time', sentiment_col='sentiment_label', 
                            platform_col=None, title='情感趋势分析', figsize=(12, 6), 
                            save_path=None):
        """绘制情感趋势图
        
        Args:
            df: 包含时间和情感标签的DataFrame
            time_col: 时间列名
            sentiment_col: 情感标签列名
            platform_col: 平台列名，用于按平台分组
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        try:
            # 检查必要的列是否存在
            if time_col not in df.columns:
                print(f"错误: 时间列 '{time_col}' 不存在于数据中")
                return
                
            if sentiment_col not in df.columns:
                print(f"错误: 情感列 '{sentiment_col}' 不存在于数据中")
                return
                
            if platform_col and platform_col not in df.columns:
                print(f"警告: 平台列 '{platform_col}' 不存在于数据中，将绘制整体情感趋势")
                platform_col = None
                
            plt.figure(figsize=figsize)
            
            # 确保时间列是datetime类型
            df = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # 过滤掉无效的时间
            df = df.dropna(subset=[time_col])
            
            if df.empty:
                print("警告: 过滤无效时间后，没有足够的数据绘制情感趋势图")
                return
            
            # 按日期分组计算平均情感得分
            df['date'] = df[time_col].dt.date
            
            if platform_col and platform_col in df.columns:
                # 按平台和日期分组
                sentiment_trend = df.groupby([platform_col, 'date'])[sentiment_col].mean().unstack(0, fill_value=None)
                sentiment_trend.plot(marker='o', linestyle='-')
                plt.legend(title='平台')
            else:
                # 按日期分组
                sentiment_trend = df.groupby('date')[sentiment_col].mean()
                sentiment_trend.plot(marker='o', linestyle='-', color='blue')
            
            plt.xlabel('日期', fontproperties=self.font)
            plt.ylabel('平均情感得分', fontproperties=self.font)
            plt.title(title, fontproperties=self.font)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300)
                print(f"情感趋势图已保存到 {save_path}")
            
            plt.close()
        except Exception as e:
            print(f"绘制情感趋势图出错: {e}")
    
    def plot_confusion_matrix(self, cm, class_names, title='混淆矩阵', 
                             figsize=(8, 6), save_path=None):
        """绘制混淆矩阵热力图
        
        Args:
            cm: 混淆矩阵
            class_names: 类别名称
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        try:
            plt.figure(figsize=figsize)
            
            # 计算准确率
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # 绘制热力图
            sns.heatmap(
                cm_normalized, 
                annot=True, 
                cmap='Blues', 
                fmt='.2f',
                xticklabels=class_names,
                yticklabels=class_names
            )
            
            plt.xlabel('预测标签', fontproperties=self.font)
            plt.ylabel('真实标签', fontproperties=self.font)
            plt.title(title, fontproperties=self.font)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300)
                print(f"混淆矩阵热力图已保存到 {save_path}")
            
            plt.close()
        except Exception as e:
            print(f"绘制混淆矩阵热力图出错: {e}")

# ====================== 5. 命名实体识别模块 ======================

class NamedEntityRecognition:
    """命名实体识别类，识别文本中的实体"""
    
    def __init__(self, model_type='default'):
        """初始化NER模型
        
        Args:
            model_type: 模型类型，可选 'default'(基于jieba), 'bert', 'hanlp', 'lac'
        """
        self.model_type = model_type
        self.model = None
        self.entity_colors = {
            'PERSON': '#FF9999',     # 人名 - 浅红色
            'LOCATION': '#99CCFF',   # 地名 - 浅蓝色
            'ORGANIZATION': '#FFCC99', # 组织机构 - 浅橙色
            'TIME': '#CCFF99',       # 时间 - 浅绿色
            'PRODUCT': '#CC99FF',    # 产品 - 浅紫色
            'BRAND': '#FF99CC',      # 品牌 - 粉色
            'MOVIE': '#99FFCC',      # 电影 - 薄荷色
            'MUSIC': '#FFFF99',      # 音乐 - 浅黄色
            'OTHER': '#CCCCCC'       # 其他 - 灰色
        }
        
        # 初始化模型
        if model_type == 'bert':
            try:
                # 使用Hugging Face的NER pipeline
                self.model = pipeline('ner', model='bert-base-chinese', tokenizer='bert-base-chinese')
                print("已加载BERT NER模型")
            except Exception as e:
                print(f"加载BERT NER模型失败: {e}")
                self.model_type = 'default'
        
        elif model_type == 'hanlp':
            if HANLP_AVAILABLE:
                try:
                    # 加载HanLP NER模型
                    self.model = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
                    print("已加载HanLP NER模型")
                except Exception as e:
                    print(f"加载HanLP NER模型失败: {e}")
                    self.model_type = 'default'
            else:
                print("HanLP未安装，切换为默认NER模型")
                self.model_type = 'default'
        
        elif model_type == 'lac':
            if LAC_AVAILABLE:
                try:
                    # 加载百度LAC模型
                    self.model = LAC(mode='lac')
                    print("已加载百度LAC NER模型")
                except Exception as e:
                    print(f"加载百度LAC NER模型失败: {e}")
                    self.model_type = 'default'
            else:
                print("LAC未安装，切换为默认NER模型")
                self.model_type = 'default'
        
        # 默认使用jieba分词和词性标注
        if self.model_type == 'default':
            print("使用jieba词性标注进行命名实体识别")
            # 加载用户词典（如果需要）
            # jieba.load_userdict("user_dict.txt")
    
    def recognize(self, text):
        """识别文本中的命名实体
        
        Args:
            text: 待分析的文本
            
        Returns:
            entities: 实体列表，每个实体为(词, 类型, 开始位置, 结束位置)
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        entities = []
        
        try:
            if self.model_type == 'bert':
                # 使用BERT NER模型
                ner_results = self.model(text)
                for result in ner_results:
                    word = result['word']
                    entity_type = result['entity'].split('-')[-1]  # 取出B-PER或I-PER中的PER
                    start = result['start']
                    end = result['end']
                    entities.append((word, entity_type, start, end))
            
            elif self.model_type == 'hanlp':
                # 使用HanLP NER模型
                result = self.model(text)
                ner_tags = result['ner/msra']
                for i, (word, tag) in enumerate(zip(result['tok/fine'], ner_tags)):
                    if tag != 'O':  # 'O'表示非实体
                        entity_type = tag.split('-')[-1]  # 取出B-PER或I-PER中的PER
                        # 计算位置（粗略估计）
                        start = text.find(word)
                        end = start + len(word)
                        entities.append((word, entity_type, start, end))
            
            elif self.model_type == 'lac':
                # 使用百度LAC模型
                result = self.model.run(text)
                words, tags = result
                for i, (word, tag) in enumerate(zip(words, tags)):
                    if tag != 'O' and tag != 'PER' and tag != 'LOC' and tag != 'ORG':
                        # 计算位置（粗略估计）
                        start = text.find(word)
                        end = start + len(word)
                        # 转换标签
                        entity_type = 'PERSON' if tag == 'PER' else ('LOCATION' if tag == 'LOC' else ('ORGANIZATION' if tag == 'ORG' else 'OTHER'))
                        entities.append((word, entity_type, start, end))
            
            else:  # default
                # 使用jieba词性标注
                words = pseg.cut(text)
                start = 0
                for word, flag in words:
                    length = len(word)
                    # 根据词性判断实体类型
                    entity_type = None
                    if flag == 'nr':  # 人名
                        entity_type = 'PERSON'
                    elif flag == 'ns':  # 地名
                        entity_type = 'LOCATION'
                    elif flag == 'nt':  # 机构团体
                        entity_type = 'ORGANIZATION'
                    elif flag == 't':  # 时间
                        entity_type = 'TIME'
                    elif flag in ['nz', 'x']:  # 其他专名
                        # 根据关键词判断是否为产品、品牌、电影或音乐
                        if '电影' in text[max(0, start-10):start+length+10] or '片' in text[max(0, start-5):start+length+5]:
                            entity_type = 'MOVIE'
                        elif '歌' in text[max(0, start-10):start+length+10] or '曲' in text[max(0, start-5):start+length+5]:
                            entity_type = 'MUSIC'
                        elif '牌' in text[max(0, start-10):start+length+10]:
                            entity_type = 'BRAND'
                        elif '产品' in text[max(0, start-10):start+length+10] or '款' in text[max(0, start-5):start+length+5]:
                            entity_type = 'PRODUCT'
                        else:
                            entity_type = 'OTHER'
                    
                    if entity_type:
                        entities.append((word, entity_type, start, start + length))
                    
                    start += length
            
            return entities
        
        except Exception as e:
            print(f"命名实体识别出错: {e}")
            return []
    
    def process_batch(self, texts):
        """批量处理多个文本
        
        Args:
            texts: 文本列表
            
        Returns:
            results: 每个文本的实体列表
        """
        results = []
        for text in texts:
            entities = self.recognize(text)
            results.append(entities)
        return results
    
    def process_dataframe(self, df, text_col='clean_content'):
        """处理DataFrame中的文本数据
        
        Args:
            df: 包含文本数据的DataFrame
            text_col: 文本内容的列名
            
        Returns:
            df_with_entities: 添加了实体信息的DataFrame
        """
        # 复制一份，避免修改原始数据
        df_with_entities = df.copy()
        
        # 确保text_col存在
        if text_col not in df_with_entities.columns:
            print(f"错误: 列 '{text_col}' 不存在于数据中")
            return df_with_entities
        
        # 应用NER到每一行
        print(f"开始处理 {len(df_with_entities)} 条文本进行命名实体识别...")
        
        # 创建实体列表
        df_with_entities['entities'] = df_with_entities[text_col].apply(self.recognize)
        
        # 提取各类实体
        df_with_entities['person_entities'] = df_with_entities['entities'].apply(
            lambda x: [item[0] for item in x if item[1] == 'PERSON'])
        df_with_entities['location_entities'] = df_with_entities['entities'].apply(
            lambda x: [item[0] for item in x if item[1] == 'LOCATION'])
        df_with_entities['organization_entities'] = df_with_entities['entities'].apply(
            lambda x: [item[0] for item in x if item[1] == 'ORGANIZATION'])
        df_with_entities['product_entities'] = df_with_entities['entities'].apply(
            lambda x: [item[0] for item in x if item[1] in ['PRODUCT', 'BRAND']])
        
        print("命名实体识别完成。")
        return df_with_entities
    
    def highlight_entities_html(self, text, entities):
        """生成带有实体高亮的HTML文本
        
        Args:
            text: 原始文本
            entities: 实体列表 [(词, 类型, 开始位置, 结束位置), ...]
            
        Returns:
            highlighted_text: 带有高亮标记的HTML文本
        """
        if not text or not entities:
            return text
        
        # 按照位置排序实体
        sorted_entities = sorted(entities, key=lambda x: x[2])
        
        # 构建高亮文本
        result = []
        last_end = 0
        
        for word, entity_type, start, end in sorted_entities:
            # 添加实体前的文本
            if start > last_end:
                result.append(text[last_end:start])
            
            # 添加带有高亮的实体
            color = self.entity_colors.get(entity_type, self.entity_colors['OTHER'])
            result.append(f'<span style="background-color: {color};" title="{entity_type}">{text[start:end]}</span>')
            
            last_end = end
        
        # 添加最后剩余的文本
        if last_end < len(text):
            result.append(text[last_end:])
        
        return ''.join(result)
    
    def generate_entity_summary(self, df, entity_col='entities'):
        """生成实体摘要统计
        
        Args:
            df: 包含实体信息的DataFrame
            entity_col: 实体列名
            
        Returns:
            summary: 实体摘要统计信息
        """
        if entity_col not in df.columns:
            print(f"错误: 列 '{entity_col}' 不存在于数据中")
            return {}
        
        # 统计各类实体
        entity_counts = defaultdict(Counter)
        total_entities = 0
        
        for entities in df[entity_col]:
            for word, entity_type, _, _ in entities:
                entity_counts[entity_type][word] += 1
                total_entities += 1
        
        # 生成摘要
        summary = {
            'total_count': total_entities,
            'type_distribution': {entity_type: sum(counts.values()) for entity_type, counts in entity_counts.items()},
            'top_entities': {
                entity_type: counts.most_common(10)
                for entity_type, counts in entity_counts.items()
            }
        }
        
        return summary
    
    def plot_entity_distribution(self, summary, title='实体类型分布', figsize=(10, 6), save_path=None):
        """绘制实体类型分布图
        
        Args:
            summary: 实体摘要统计
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        if not summary or 'type_distribution' not in summary:
            print("错误: 无效的实体摘要数据")
            return
        
        type_dist = summary['type_distribution']
        if not type_dist:
            print("错误: 没有实体类型分布数据")
            return
        
        plt.figure(figsize=figsize)
        
        # 准备数据
        entity_types = list(type_dist.keys())
        counts = list(type_dist.values())
        colors = [self.entity_colors.get(t, '#CCCCCC') for t in entity_types]
        
        # 绘制条形图
        bars = plt.bar(entity_types, counts, color=colors)
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}',
                    ha='center', va='bottom', fontproperties=font)
        
        plt.xlabel('实体类型', fontproperties=font)
        plt.ylabel('数量', fontproperties=font)
        plt.title(title, fontproperties=font)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"实体类型分布图已保存到 {save_path}")
        
        plt.close()
    
    def plot_top_entities(self, summary, entity_type, top_n=10, title=None, figsize=(12, 6), save_path=None):
        """绘制特定类型的Top N实体
        
        Args:
            summary: 实体摘要统计
            entity_type: 实体类型
            top_n: 显示前N个实体
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        if not summary or 'top_entities' not in summary:
            print("错误: 无效的实体摘要数据")
            return
        
        if entity_type not in summary['top_entities']:
            print(f"错误: 没有类型为'{entity_type}'的实体数据")
            return
        
        # 取Top N实体
        top_entities = summary['top_entities'][entity_type][:top_n]
        if not top_entities:
            print(f"错误: 没有足够的'{entity_type}'类型实体")
            return
        
        plt.figure(figsize=figsize)
        
        # 准备数据
        names = [e[0] for e in top_entities]
        counts = [e[1] for e in top_entities]
        color = self.entity_colors.get(entity_type, '#CCCCCC')
        
        # 绘制水平条形图
        bars = plt.barh(names, counts, color=color)
        
        # 添加数据标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                    f'{width}',
                    ha='left', va='center', fontproperties=font)
        
        plt.xlabel('出现次数', fontproperties=font)
        plt.ylabel('实体名称', fontproperties=font)
        
        if title:
            plt.title(title, fontproperties=font)
        else:
            plt.title(f'Top {top_n} {entity_type} 实体', fontproperties=font)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Top实体图已保存到 {save_path}")
        
        plt.close()

# ====================== 6. 依存句法分析模块 ======================

class DependencyParser:
    """依存句法分析器类"""
    
    def __init__(self, model_type='default'):
        """初始化依存句法分析器
        
        Args:
            model_type: 模型类型，可选 'default', 'hanlp', 'ltp'
        """
        self.model_type = model_type
        self.model = None
        
        # 初始化模型
        if model_type == 'hanlp':
            if HANLP_AVAILABLE:
                try:
                    # 加载HanLP依存句法分析模型
                    self.model = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
                    print("已加载HanLP依存句法分析模型")
                except Exception as e:
                    print(f"加载HanLP依存句法分析模型失败: {e}")
                    self.model_type = 'default'
            else:
                print("HanLP未安装，切换为默认依存句法分析模型")
                self.model_type = 'default'
                
        # 其他模型类型可以在此添加
        
        if self.model_type == 'default':
            print("使用简化实现进行依存句法分析")
    
    def parse(self, text):
        """对文本进行依存句法分析
        
        Args:
            text: 待分析的文本
            
        Returns:
            result: 依存句法分析结果
        """
        if not isinstance(text, str) or not text.strip():
            return {'words': [], 'postags': [], 'arcs': []}
        
        try:
            if self.model_type == 'hanlp':
                # 使用HanLP进行依存句法分析
                result = self.model(text)
                words = result['tok/fine']
                postags = result['pos/pku']
                deps = result['dep/ctb']
                
                # 转换为标准格式
                arcs = []
                for i, (head, relation) in enumerate(deps):
                    arcs.append({
                        'head': head,      # 依存弧的父节点
                        'relation': relation, # 依存关系
                        'id': i+1         # 当前词的索引
                    })
                
                return {
                    'words': words,
                    'postags': postags,
                    'arcs': arcs
                }
            
            else:  # default
                # 使用jieba分词和词性标注，简化的依存分析
                words = list(jieba.cut(text))
                postags = [p for w, p in pseg.cut(text)]
                
                # 简化的依存分析规则
                arcs = []
                root_idx = 0  # 假设句子的根节点
                for i, (word, pos) in enumerate(zip(words, postags)):
                    # 简单规则：大部分词依赖于前一个词，标点符号依赖于前一个词
                    if i == root_idx:
                        # 根节点
                        head = 0
                        relation = 'HED'  # 核心词
                    elif pos in ['wp', 'w']:  # 标点符号
                        head = i
                        relation = 'WP'
                    elif i > 0 and words[i-1] in ['的', '地', '得']:
                        # 的/地/得后面的词通常是前面词的修饰语
                        head = i - 2 if i >= 2 else 0
                        relation = 'ATT'  # 定语
                    elif pos.startswith('v'):  # 动词
                        # 找前面最近的名词作为主语
                        for j in range(i-1, -1, -1):
                            if postags[j].startswith('n'):
                                head = j + 1
                                relation = 'SBV'  # 主语
                                break
                        else:
                            head = i
                            relation = 'HED'
                    else:
                        # 默认依赖于前一个词
                        head = i
                        relation = 'DEP'  # 未知依存关系
                    
                    arcs.append({
                        'head': head,
                        'relation': relation,
                        'id': i + 1
                    })
                
                return {
                    'words': words,
                    'postags': postags,
                    'arcs': arcs
                }
        
        except Exception as e:
            print(f"依存句法分析出错: {e}")
            return {'words': [], 'postags': [], 'arcs': []}
    
    def process_batch(self, texts):
        """批量处理多个文本
        
        Args:
            texts: 文本列表
            
        Returns:
            results: 每个文本的依存句法分析结果
        """
        results = []
        for text in texts:
            parse_result = self.parse(text)
            results.append(parse_result)
        return results
    
    def plot_dependency_tree(self, parse_result, title='依存句法树', figsize=(12, 8), save_path=None):
        """绘制依存句法树
        
        Args:
            parse_result: 依存句法分析结果
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        if not parse_result or 'words' not in parse_result or not parse_result['words']:
            print("错误: 无效的依存句法分析结果")
            return
        
        words = parse_result['words']
        arcs = parse_result['arcs']
        
        plt.figure(figsize=figsize)
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点
        G.add_node(0, label='ROOT')  # 根节点
        for i, word in enumerate(words):
            G.add_node(i+1, label=word)
        
        # 添加边
        for arc in arcs:
            head = arc['head']
            child = arc['id']
            relation = arc['relation'] if 'relation' in arc else 'DEP'  # 默认关系
            G.add_edge(head, child, label=relation)
        
        # 设置节点位置 (分层布局)
        pos = nx.spring_layout(G)
        
        # 绘制节点
        node_labels = {node: data['label'] for node, data in G.nodes(data=True)}
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_family='SimHei')
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, arrowsize=20, width=1.5)
        edge_labels = {(head, child): data['label'] for head, child, data in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        
        plt.title(title, fontproperties=font)
        plt.axis('off')  # 不显示坐标轴
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"依存句法树已保存到 {save_path}")
        
        plt.close()
    
    def generate_syntax_html(self, parse_result):
        """生成可视化的HTML依存句法分析结果
        
        Args:
            parse_result: 依存句法分析结果
            
        Returns:
            html: HTML格式的依存句法分析结果
        """
        if not parse_result or 'words' not in parse_result or not parse_result['words']:
            return "<p>无效的依存句法分析结果</p>"
        
        words = parse_result['words']
        postags = parse_result.get('postags', [''] * len(words))
        arcs = parse_result['arcs']
        
        # 生成HTML
        html = """
        <div style="font-family: Arial, sans-serif; margin: 20px;">
            <h3 style="color: #333;">依存句法分析结果</h3>
            <div style="margin-bottom: 15px;">
                <b>原始文本:</b> {text}
            </div>
            <table style="width: 100%; border-collapse: collapse; border: 1px solid #ccc;">
                <tr style="background-color: #f0f0f0;">
                    <th style="padding: 8px; border: 1px solid #ccc;">序号</th>
                    <th style="padding: 8px; border: 1px solid #ccc;">词</th>
                    <th style="padding: 8px; border: 1px solid #ccc;">词性</th>
                    <th style="padding: 8px; border: 1px solid #ccc;">依存弧</th>
                    <th style="padding: 8px; border: 1px solid #ccc;">关系</th>
                </tr>
        """.format(text=''.join(words))
        
        # 添加每个词的依存信息
        for i, (word, pos, arc) in enumerate(zip(words, postags, arcs)):
            head = arc['head']
            relation = arc['relation'] if 'relation' in arc else 'DEP'
            head_word = 'ROOT' if head == 0 else words[head-1]
            
            html += """
                <tr>
                    <td style="padding: 8px; border: 1px solid #ccc; text-align: center;">{id}</td>
                    <td style="padding: 8px; border: 1px solid #ccc;">{word}</td>
                    <td style="padding: 8px; border: 1px solid #ccc;">{pos}</td>
                    <td style="padding: 8px; border: 1px solid #ccc;">{head} ({head_word})</td>
                    <td style="padding: 8px; border: 1px solid #ccc;">{relation}</td>
                </tr>
            """.format(
                id=i+1,
                word=word,
                pos=pos,
                head=head,
                head_word=head_word,
                relation=relation
            )
        
        html += """
            </table>
        </div>
        """
        
        return html

# ====================== 7. 关键词抽取模块 ======================

class KeywordExtractor:
    """关键词抽取类"""
    
    def __init__(self, method='tfidf', stopwords_path=None):
        """初始化关键词抽取器
        
        Args:
            method: 抽取方法，可选 'tfidf', 'textrank'
            stopwords_path: 停用词表路径
        """
        self.method = method
        
        # 加载停用词表
        self.stopwords = set()
        if stopwords_path and os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                self.stopwords = set([line.strip() for line in f.readlines()])
        else:
            # 使用默认的简单停用词表
            self.stopwords = set(['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', 
                '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '啊', '吧', '把', 
                '但', '但是', '并', '个', '给', '过', '还', '还是', '还有', '其', '其实', '其中', '几', '可', '可以', 
                '可是', '么', '没', '什么', '什么样', '这样', '那样', '之', '之一', '只', '只是', '只要', '只有', 
                '就是', '就是说', '打', '呢', '来', '来说', '来自', '哪儿', '哪里', '如', '如果', '如何', '如此'])
    
    def extract(self, text, top_k=10):
        """从文本中抽取关键词
        
        Args:
            text: 待分析的文本
            top_k: 返回前k个关键词
            
        Returns:
            keywords: 关键词及其权重列表 [(词, 权重), ...]
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        # 使用不同的方法抽取关键词
        try:
            if self.method == 'textrank':
                # 使用TextRank算法
                keywords = textrank(text, topK=top_k, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
                return keywords
            else:  # tfidf
                # 使用TF-IDF算法
                keywords = extract_tags(text, topK=top_k, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
                return keywords
        except Exception as e:
            print(f"关键词抽取出错: {e}")
            return []
    
    def extract_batch(self, texts, top_k=10):
        """批量抽取关键词
        
        Args:
            texts: 文本列表
            top_k: 每个文本返回前k个关键词
            
        Returns:
            results: 每个文本的关键词列表
        """
        results = []
        for text in texts:
            keywords = self.extract(text, top_k)
            results.append(keywords)
        return results
    
    def extract_from_df(self, df, text_col='clean_content', top_k=10):
        """从DataFrame抽取关键词
        
        Args:
            df: 包含文本的DataFrame
            text_col: 文本列名
            top_k: 每个文本返回前k个关键词
            
        Returns:
            df_with_keywords: 添加了关键词的DataFrame
        """
        # 复制一份，避免修改原始数据
        df_with_keywords = df.copy()
        
        # 确保text_col存在
        if text_col not in df_with_keywords.columns:
            print(f"错误: 列 '{text_col}' 不存在于数据中")
            return df_with_keywords
        
        print(f"开始从 {len(df_with_keywords)} 条文本中抽取关键词...")
        
        # 抽取关键词
        df_with_keywords['keywords'] = df_with_keywords[text_col].apply(
            lambda x: self.extract(x, top_k))
        
        # 提取关键词单词列表（不含权重）
        df_with_keywords['keyword_list'] = df_with_keywords['keywords'].apply(
            lambda x: [word for word, _ in x])
        
        print("关键词抽取完成。")
        return df_with_keywords
    
    def aggregate_keywords(self, df, keyword_col='keywords', weight_col=None, top_k=30):
        """聚合DataFrame中的关键词
        
        Args:
            df: 包含关键词的DataFrame
            keyword_col: 关键词列名
            weight_col: 权重列名，用于加权，如情感强度
            top_k: 返回前k个关键词
            
        Returns:
            aggregated: 聚合后的关键词及其权重
        """
        if keyword_col not in df.columns:
            print(f"错误: 列 '{keyword_col}' 不存在于数据中")
            return []
        
        # 聚合所有关键词
        keyword_weights = {}
        
        for i, row in df.iterrows():
            keywords = row[keyword_col]
            weight_multiplier = 1.0
            
            # 如果有权重列，使用权重调整关键词重要性
            if weight_col and weight_col in df.columns:
                weight_multiplier = row[weight_col]
                if not isinstance(weight_multiplier, (int, float)):
                    weight_multiplier = 1.0
            
            for keyword, weight in keywords:
                adjusted_weight = weight * weight_multiplier
                if keyword in keyword_weights:
                    keyword_weights[keyword] += adjusted_weight
                else:
                    keyword_weights[keyword] = adjusted_weight
        
        # 排序并返回前k个
        sorted_keywords = sorted(keyword_weights.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_k]
    
    def plot_keyword_cloud(self, keywords, title='关键词云', figsize=(12, 8), save_path=None):
        """绘制关键词词云
        
        Args:
            keywords: 关键词及其权重列表 [(词, 权重), ...]
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        if not keywords:
            print("错误: 没有关键词数据")
            return
        
        plt.figure(figsize=figsize)
        
        # 创建词频字典
        word_freq = {word: weight for word, weight in keywords}
        
        # 查找字体
        font_paths = [
            'static/fonts/SimHei.ttf',
            'SimHei.ttf',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
        ]
        
        font_path = None
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                break
        
        # 创建词云
        wordcloud = WordCloud(
            font_path=font_path,
            width=800,
            height=600,
            background_color='white',
            max_words=100
        ).generate_from_frequencies(word_freq)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontproperties=font)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"关键词云已保存到 {save_path}")
        
        plt.close()
    
    def plot_top_keywords(self, keywords, top_n=15, title='Top关键词', figsize=(12, 6), save_path=None):
        """绘制Top N关键词条形图
        
        Args:
            keywords: 关键词及其权重列表 [(词, 权重), ...]
            top_n: 显示前N个关键词
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        if not keywords:
            print("错误: 没有关键词数据")
            return
        
        # 取Top N关键词
        keywords = keywords[:top_n]
        
        plt.figure(figsize=figsize)
        
        # 准备数据
        words = [keyword for keyword, _ in keywords]
        weights = [weight for _, weight in keywords]
        
        # 绘制水平条形图
        bars = plt.barh(words[::-1], weights[::-1], color='skyblue')
        
        # 添加数据标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{width:.3f}',
                    ha='left', va='center', fontproperties=font)
        
        plt.xlabel('权重', fontproperties=font)
        plt.ylabel('关键词', fontproperties=font)
        plt.title(title, fontproperties=font)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Top关键词图已保存到 {save_path}")
        
        plt.close()

# ====================== 8. 文本摘要模块 ======================

class TextSummarizer:
    """文本摘要生成类"""
    
    def __init__(self, method='textrank', model_name=None):
        """初始化文本摘要生成器
        
        Args:
            method: 摘要方法，可选 'textrank', 'bert'
            model_name: 预训练模型名称，用于bert方法
        """
        self.method = method
        self.model = None
        self.tokenizer = None
        
        # 初始化BERT摘要模型
        if method == 'bert':
            try:
                model_name = model_name or 'bert-base-chinese'
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                print(f"已加载BERT摘要模型: {model_name}")
            except Exception as e:
                print(f"加载BERT摘要模型失败: {e}")
                self.method = 'textrank'
    
    def summarize(self, text, ratio=0.2, max_length=3):
        """生成文本摘要
        
        Args:
            text: 待摘要的文本
            ratio: 摘要比例，仅用于textrank方法
            max_length: 最大摘要句子数量
            
        Returns:
            summary: 摘要文本
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # 将文本分割成句子
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return ""
        
        try:
            if self.method == 'bert' and self.model and self.tokenizer:
                # 使用BERT方法
                sentence_embeddings = []
                
                with torch.no_grad():
                    for sentence in sentences:
                        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
                        outputs = self.model(**inputs)
                        # 使用[CLS]的输出作为句子表示
                        embedding = outputs.last_hidden_state[:, 0, :].numpy()
                        sentence_embeddings.append(embedding.flatten())
                
                # 计算句子之间的相似度
                similarity_matrix = np.zeros((len(sentences), len(sentences)))
                for i in range(len(sentences)):
                    for j in range(len(sentences)):
                        if i != j:
                            # 计算余弦相似度
                            similarity_matrix[i][j] = np.dot(sentence_embeddings[i], sentence_embeddings[j]) / (
                                np.linalg.norm(sentence_embeddings[i]) * np.linalg.norm(sentence_embeddings[j]))
                
                # 将相似度矩阵转换为图
                nx_graph = nx.from_numpy_array(similarity_matrix)
                scores = nx.pagerank(nx_graph)
                
                # 根据分数排序句子
                ranked_sentences = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
                
                # 选择分数最高的句子作为摘要
                top_sentence_indices = sorted([ranked_sentences[i][1] for i in range(min(max_length, len(ranked_sentences)))])
                summary = '。'.join([sentences[i] for i in top_sentence_indices]) + '。'
                
                return summary
            
            else:  # textrank
                # 使用TextRank方法
                # 计算句子权重
                keywords = textrank(text, topK=min(10, max(3, int(len(text) / 100))), withWeight=True)
                keyword_set = {word for word, _ in keywords}
                
                # 计算每个句子包含关键词的数量
                sentence_scores = []
                for i, sentence in enumerate(sentences):
                    words = jieba.lcut(sentence)
                    score = sum(1 for word in words if word in keyword_set)
                    sentence_scores.append((score, i, len(words)))
                
                # 按分数和句子长度排序
                sentence_scores.sort(key=lambda x: (x[0], -x[2]), reverse=True)
                
                # 选择得分最高的句子
                top_sentence_indices = [score[1] for score in sentence_scores[:max_length]]
                top_sentence_indices.sort()  # 按原文顺序排列
                
                summary = '。'.join([sentences[i] for i in top_sentence_indices]) + '。'
                
                return summary
        
        except Exception as e:
            print(f"生成摘要出错: {e}")
            # 返回前几个句子作为摘要
            return '。'.join(sentences[:max_length]) + '。'
    
    def summarize_batch(self, texts, ratio=0.2, max_length=3):
        """批量生成摘要
        
        Args:
            texts: 文本列表
            ratio: 摘要比例
            max_length: 最大摘要句子数量
            
        Returns:
            summaries: 摘要列表
        """
        summaries = []
        for text in texts:
            summary = self.summarize(text, ratio, max_length)
            summaries.append(summary)
        return summaries
    
    def summarize_from_df(self, df, text_col='clean_content', ratio=0.2, max_length=3):
        """从DataFrame生成摘要
        
        Args:
            df: 包含文本的DataFrame
            text_col: 文本列名
            ratio: 摘要比例
            max_length: 最大摘要句子数量
            
        Returns:
            df_with_summary: 添加了摘要的DataFrame
        """
        # 复制一份，避免修改原始数据
        df_with_summary = df.copy()
        
        # 确保text_col存在
        if text_col not in df_with_summary.columns:
            print(f"错误: 列 '{text_col}' 不存在于数据中")
            return df_with_summary
        
        print(f"开始为 {len(df_with_summary)} 条文本生成摘要...")
        
        # 生成摘要
        df_with_summary['summary'] = df_with_summary[text_col].apply(
            lambda x: self.summarize(x, ratio, max_length))
        
        print("摘要生成完成。")
        return df_with_summary
    
    def plot_length_reduction(self, df, text_col='clean_content', summary_col='summary', 
                             title='文本长度减少情况', figsize=(10, 6), save_path=None):
        """绘制摘要前后文本长度对比图
        
        Args:
            df: 包含原文和摘要的DataFrame
            text_col: 原文列名
            summary_col: 摘要列名
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        # 检查必要的列是否存在
        if text_col not in df.columns or summary_col not in df.columns:
            print(f"错误: 列 '{text_col}' 或 '{summary_col}' 不存在于数据中")
            return
        
        plt.figure(figsize=figsize)
        
        # 计算原文和摘要的长度
        df_len = df.copy()
        # Filter out rows with missing data
        df_len = df_len.dropna(subset=[text_col, summary_col])
        # Handle NaN values safely
        df_len['original_length'] = df_len[text_col].apply(lambda x: len(x) if isinstance(x, str) else 0)
        df_len['summary_length'] = df_len[summary_col].apply(lambda x: len(x) if isinstance(x, str) else 0)
        df_len['reduction_ratio'] = df_len['summary_length'] / df_len['original_length']
        
        # 计算平均减少比例
        avg_reduction = 1 - df_len['reduction_ratio'].mean()
        
        # 绘制散点图
        plt.scatter(df_len['original_length'], df_len['summary_length'], 
                  alpha=0.6, color='blue', edgecolors='navy')
        
        # 添加对角线（原文=摘要）
        max_len = max(df_len['original_length'].max(), df_len['summary_length'].max())
        plt.plot([0, max_len], [0, max_len], 'r--', alpha=0.5, label='原文=摘要')
        
        # 添加回归线
        x = df_len['original_length']
        y = df_len['summary_length']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), 'g-', alpha=0.8, label=f'平均减少: {avg_reduction:.1%}')
        
        plt.xlabel('原文长度', fontproperties=font)
        plt.ylabel('摘要长度', fontproperties=font)
        plt.title(title, fontproperties=font)
        plt.legend(prop=font)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"文本长度减少情况图已保存到 {save_path}")
        
        plt.close()

# ====================== 9. 主题建模模块 ======================

class TopicModeler:
    """主题建模类"""
    
    def __init__(self, n_topics=5, model_type='lda', max_features=5000, stopwords_path=None):
        """初始化主题建模器
        
        Args:
            n_topics: 主题数量
            model_type: 模型类型，可选 'lda', 'nmf'
            max_features: 最大特征数（词汇表大小）
            stopwords_path: 停用词表路径
        """
        self.n_topics = n_topics
        self.model_type = model_type
        self.max_features = max_features
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        
        # 加载停用词表
        self.stopwords = set()
        if stopwords_path and os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                self.stopwords = set([line.strip() for line in f.readlines()])
        else:
            # 使用默认的简单停用词表
            self.stopwords = set(['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', 
                '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '啊', '吧', '把', 
                '但', '但是', '并', '个', '给', '过', '还', '还是', '还有', '其', '其实', '其中', '几', '可', '可以', 
                '可是', '么', '没', '什么', '什么样', '这样', '那样', '之', '之一', '只', '只是', '只要', '只有', 
                '就是', '就是说', '打', '呢', '来', '来说', '来自', '哪儿', '哪里', '如', '如果', '如何', '如此'])
    
    def fit(self, texts):
        """训练主题模型
        
        Args:
            texts: 文本列表
            
        Returns:
            self: 返回自身实例
        """
        if not texts:
            print("错误: 没有提供训练文本")
            return self
        
        # 将文本分词并合并
        tokenized_texts = []
        for text in texts:
            if isinstance(text, str) and text.strip():
                words = [w for w in jieba.lcut(text) if w not in self.stopwords and len(w) > 1]
                tokenized_texts.append(' '.join(words))
            else:
                tokenized_texts.append('')
        
        try:
            # 使用TF-IDF向量化文本
            if self.model_type == 'nmf':
                # 对于NMF，使用TF-IDF
                self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words=list(self.stopwords))
            else:  # lda
                # 对于LDA，使用词频计数
                self.vectorizer = CountVectorizer(max_features=self.max_features, stop_words=list(self.stopwords))
            
            X = self.vectorizer.fit_transform(tokenized_texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
            
            # 训练主题模型
            if self.model_type == 'nmf':
                self.model = NMF(n_components=self.n_topics, random_state=42)
            else:  # lda
                self.model = LatentDirichletAllocation(n_components=self.n_topics, random_state=42)
            
            self.model.fit(X)
            
            print(f"主题模型训练完成，主题数量: {self.n_topics}")
            return self
        
        except Exception as e:
            print(f"训练主题模型出错: {e}")
            return self
    
    def transform(self, texts):
        """将文本转换为主题分布
        
        Args:
            texts: 文本列表
            
        Returns:
            topic_distributions: 主题分布矩阵
        """
        if not self.model or not self.vectorizer:
            print("错误: 模型未训练")
            return np.zeros((len(texts), self.n_topics))
        
        try:
            # 将文本分词并合并
            tokenized_texts = []
            for text in texts:
                if isinstance(text, str) and text.strip():
                    words = [w for w in jieba.lcut(text) if w not in self.stopwords and len(w) > 1]
                    tokenized_texts.append(' '.join(words))
                else:
                    tokenized_texts.append('')
            
            # 向量化文本
            X = self.vectorizer.transform(tokenized_texts)
            
            # 转换为主题分布
            topic_distributions = self.model.transform(X)
            
            return topic_distributions
        
        except Exception as e:
            print(f"转换文本为主题分布出错: {e}")
            return np.zeros((len(texts), self.n_topics))
    
    def get_topics(self, n_top_words=10):
        """获取主题关键词
        
        Args:
            n_top_words: 每个主题返回的关键词数量
            
        Returns:
            topics: 主题关键词列表，每个主题为(主题ID, [(词, 权重), ...])
        """
        if not self.model:
            print("错误: 模型未训练")
            return []
        
        try:
            topics = []
            for topic_idx, topic in enumerate(self.model.components_):
                top_words_idx = topic.argsort()[:-n_top_words-1:-1]
                top_words = [(self.feature_names[i], topic[i]) for i in top_words_idx]
                topics.append((topic_idx, top_words))
            
            return topics
        
        except Exception as e:
            print(f"获取主题关键词出错: {e}")
            return []
    
    def fit_transform_from_df(self, df, text_col='clean_content'):
        """从DataFrame训练模型并转换为主题分布
        
        Args:
            df: 包含文本的DataFrame
            text_col: 文本列名
            
        Returns:
            df_with_topics: 添加了主题分布的DataFrame
        """
        # 复制一份，避免修改原始数据
        df_with_topics = df.copy()
        
        # 确保text_col存在
        if text_col not in df_with_topics.columns:
            print(f"错误: 列 '{text_col}' 不存在于数据中")
            return df_with_topics
        
        print(f"开始从 {len(df_with_topics)} 条文本中建模...")
        
        # 训练模型
        texts = df_with_topics[text_col].tolist()
        self.fit(texts)
        
        # 转换为主题分布
        topic_distributions = self.transform(texts)
        
        # 添加主题分布列
        for i in range(self.n_topics):
            df_with_topics[f'topic_{i}'] = topic_distributions[:, i]
        
        # 添加主导主题列
        df_with_topics['dominant_topic'] = np.argmax(topic_distributions, axis=1)
        
        print("主题建模完成。")
        return df_with_topics
    
    def plot_topic_distribution(self, df, title='文档主题分布', figsize=(10, 6), save_path=None):
        """绘制文档主题分布图
        
        Args:
            df: 包含主题分布的DataFrame
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        # 检查主题列是否存在
        topic_cols = [f'topic_{i}' for i in range(self.n_topics)]
        if not all(col in df.columns for col in topic_cols):
            print(f"错误: 主题分布列不存在于数据中")
            return
        
        plt.figure(figsize=figsize)
        
        # 统计每个主题的文档数量
        dominant_topic_counts = df['dominant_topic'].value_counts().sort_index()
        
        # 绘制条形图
        bars = plt.bar(range(self.n_topics), dominant_topic_counts, color='skyblue')
        
        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}',
                    ha='center', va='bottom', fontproperties=font)
        
        plt.xlabel('主题ID', fontproperties=font)
        plt.ylabel('文档数量', fontproperties=font)
        plt.title(title, fontproperties=font)
        plt.xticks(range(self.n_topics), [f'主题{i}' for i in range(self.n_topics)])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"文档主题分布图已保存到 {save_path}")
        
        plt.close()
    
    def plot_topic_keywords(self, topics=None, title='主题关键词', figsize=(15, 10), save_path=None):
        """绘制主题关键词图
        
        Args:
            topics: 主题关键词列表，如果为None则使用get_topics()获取
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        if not topics:
            topics = self.get_topics()
        
        if not topics:
            print("错误: 没有主题数据")
            return
        
        plt.figure(figsize=figsize)
        
        # 设置子图布局
        n_cols = min(2, self.n_topics)
        n_rows = (self.n_topics + n_cols - 1) // n_cols
        
        # 绘制每个主题的关键词
        for i, (topic_id, top_words) in enumerate(topics):
            plt.subplot(n_rows, n_cols, i + 1)
            
            words = [word for word, _ in top_words]
            weights = [weight for _, weight in top_words]
            
            # 归一化权重，使其更适合可视化
            weights = [w / max(weights) for w in weights]
            
            # 绘制水平条形图
            bars = plt.barh(range(len(words)), weights, color='skyblue')
            
            plt.yticks(range(len(words)), words, fontproperties=font)
            plt.xlabel('相对权重', fontproperties=font)
            plt.title(f'主题 {topic_id}', fontproperties=font)
            plt.gca().invert_yaxis()  # 按从上到下的顺序显示
            plt.tight_layout()
        
        plt.suptitle(title, fontproperties=font, fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"主题关键词图已保存到 {save_path}")
        
        plt.close()
    
    def plot_topic_heatmap(self, topic_distributions, title='文档-主题分布热图', figsize=(12, 8), 
                          sample_size=50, save_path=None):
        """绘制文档-主题分布热图
        
        Args:
            topic_distributions: 主题分布矩阵
            title: 图表标题
            figsize: 图表大小
            sample_size: 采样文档数量
            save_path: 保存路径
        """
        if topic_distributions.shape[0] == 0:
            print("错误: 主题分布矩阵为空")
            return
        
        plt.figure(figsize=figsize)
        
        # 如果文档数量过多，进行采样
        if topic_distributions.shape[0] > sample_size:
            indices = np.random.choice(topic_distributions.shape[0], sample_size, replace=False)
            topic_dist_sample = topic_distributions[indices]
        else:
            topic_dist_sample = topic_distributions
        
        # 绘制热图
        sns.heatmap(
            topic_dist_sample, 
            cmap='YlGnBu', 
            linewidths=0.1,
            vmin=0, vmax=1,
            xticklabels=[f'主题{i}' for i in range(self.n_topics)],
            yticklabels=False
        )
        
        plt.xlabel('主题', fontproperties=font)
        plt.ylabel('文档', fontproperties=font)
        plt.title(title, fontproperties=font)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"文档-主题分布热图已保存到 {save_path}")
        
        plt.close()

# ====================== 10. 方面级情感分析模块 ======================

class AspectBasedSentimentAnalyzer:
    """方面级情感分析器类"""
    
    def __init__(self, aspect_lexicon_path=None, sentiment_lexicon_path=None, dependency_parser=None):
        """初始化方面级情感分析器
        
        Args:
            aspect_lexicon_path: 方面词典路径
            sentiment_lexicon_path: 情感词典路径
            dependency_parser: 依存句法分析器实例
        """
        self.dependency_parser = dependency_parser
        
        # 加载方面词典
        self.aspect_lexicon = set()
        if aspect_lexicon_path and os.path.exists(aspect_lexicon_path):
            with open(aspect_lexicon_path, 'r', encoding='utf-8') as f:
                self.aspect_lexicon = set([line.strip() for line in f.readlines()])
        else:
            # 使用默认的方面词典（示例）
            self.aspect_lexicon = set([
                # 电影相关方面
                '剧情', '故事', '情节', '结局', '节奏', '台词', '对白',
                '演技', '表演', '演员', '配音', '角色', '主角', '配角',
                '画面', '特效', '场景', '视觉', '镜头', '摄影', '视效',
                '音乐', '配乐', '音效', '歌曲', '主题曲', '插曲',
                '导演', '编剧', '制作', '拍摄', '改编', '原著',
                # 商品相关方面
                '质量', '做工', '材质', '耐用', '外观', '设计', '款式',
                '价格', '性价比', '便宜', '划算', '优惠', '贵',
                '物流', '配送', '快递', '包装', '送货',
                '服务', '售后', '客服', '店家', '卖家',
                '功能', '性能', '效果', '实用', '好用', '操作',
                # 音乐相关方面
                '歌词', '旋律', '编曲', '制作', '和声', '节奏', '曲风',
                '嗓音', '唱法', '技巧', '高音', '低音', '现场', '专辑',
                '创作', '作词', '作曲', '编曲', '制作人',
                # 通用方面
                '体验', '感受', '推荐', '评价', '整体', '总体'
            ])
        
        # 加载情感词典
        self.sentiment_lexicon = {}
        if sentiment_lexicon_path and os.path.exists(sentiment_lexicon_path):
            with open(sentiment_lexicon_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        word, score = parts[0], float(parts[1])
                        self.sentiment_lexicon[word] = score
        else:
            # 使用默认的情感词典（示例）
            # 正面词汇：1.0，负面词汇：-1.0
            pos_words = ['好', '优秀', '精彩', '完美', '出色', '满意', '喜欢', '推荐',
                        '强烈', '震撼', '感人', '有趣', '幽默', '华丽', '精致', '细腻',
                        '悦耳', '动听', '热血', '感动', '超值', '不错', '高效', '实用',
                        '方便', '快捷', '合理', '耐用', '舒适', '惊艳', '美观', '专业',
                        '热情', '周到', '耐心', '迅速', '准时', '可靠', '丰富', '创新']
            
            neg_words = ['差', '糟糕', '失望', '难看', '无聊', '浪费', '后悔', '枯燥',
                        '乏味', '俗套', '敷衍', '尴尬', '虚假', '过时', '劣质', '粗糙',
                        '廉价', '难听', '刺耳', '干燥', '噪音', '难用', '复杂', '繁琐',
                        '迟缓', '昂贵', '虚高', '漫长', '延迟', '缓慢', '刻板', '单调',
                        '枯燥', '无味', '塑料', '破损', '生硬', '敷衍', '冷漠', '敷衍']
            
            self.sentiment_lexicon = {word: 1.0 for word in pos_words}
            self.sentiment_lexicon.update({word: -1.0 for word in neg_words})
    
    def extract_aspects(self, text, mode='rule'):
        """从文本中抽取方面词
        
        Args:
            text: 待分析的文本
            mode: 抽取模式，'rule'使用规则，'dependency'使用依存句法分析
            
        Returns:
            aspects: 方面词列表 [(方面词, 位置), ...]
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        aspects = []
        
        try:
            if mode == 'dependency' and self.dependency_parser:
                # 使用依存句法分析
                parse_result = self.dependency_parser.parse(text)
                words = parse_result['words']
                postags = parse_result['postags']
                arcs = parse_result['arcs']
                
                # 查找名词并检查是否为方面词
                for i, (word, pos) in enumerate(zip(words, postags)):
                    if pos.startswith('n') and (word in self.aspect_lexicon or len(word) > 1):
                        # 检查是否有情感词依赖于该名词
                        has_sentiment = False
                        for j, (other_word, other_pos) in enumerate(zip(words, postags)):
                            if j != i and other_word in self.sentiment_lexicon:
                                # 检查依存关系
                                if arcs[j]['head'] == i + 1:  # 索引从1开始
                                    has_sentiment = True
                                    break
                        
                        if has_sentiment or word in self.aspect_lexicon:
                            # 计算位置
                            pos = text.find(word)
                            if pos >= 0:
                                aspects.append((word, pos))
            
            else:  # rule
                # 使用规则方法
                # 分词和词性标注
                words_pos = pseg.cut(text)
                
                # 查找名词并检查是否为方面词
                for word, pos in words_pos:
                    if (pos.startswith('n') or pos == 'l') and (word in self.aspect_lexicon or len(word) > 1):
                        # 计算位置
                        pos = text.find(word)
                        if pos >= 0:
                            aspects.append((word, pos))
            
            return aspects
        
        except Exception as e:
            print(f"方面抽取出错: {e}")
            return []
    
    def analyze_aspect_sentiment(self, text, aspects=None, window_size=5):
        """分析方面级情感
        
        Args:
            text: 待分析的文本
            aspects: 方面词列表 [(方面词, 位置), ...]，如果为None则自动抽取
            window_size: 上下文窗口大小
            
        Returns:
            results: 方面级情感分析结果 [(方面词, 情感极性, 情感强度, 相关评价), ...]
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        # 如果没有提供方面词，则自动抽取
        if aspects is None:
            aspects = self.extract_aspects(text)
        
        results = []
        
        try:
            # 对每个方面词进行情感分析
            for aspect, aspect_pos in aspects:
                # 截取上下文窗口
                start = max(0, aspect_pos - window_size)
                end = min(len(text), aspect_pos + len(aspect) + window_size)
                context = text[start:end]
                
                # 在上下文中查找情感词
                words = jieba.lcut(context)
                sentiment_score = 0.0
                sentiment_words = []
                
                for word in words:
                    if word in self.sentiment_lexicon:
                        score = self.sentiment_lexicon[word]
                        sentiment_score += score
                        sentiment_words.append((word, score))
                
                # 确定情感极性和强度
                if sentiment_score > 0:
                    polarity = '正面'
                    intensity = min(1.0, sentiment_score / 3)  # 限制在[0,1]范围内
                elif sentiment_score < 0:
                    polarity = '负面'
                    intensity = min(1.0, abs(sentiment_score) / 3)  # 限制在[0,1]范围内
                else:
                    polarity = '中性'
                    intensity = 0.0
                
                results.append((aspect, polarity, intensity, context, sentiment_words))
        
        except Exception as e:
            print(f"方面级情感分析出错: {e}")
        
        return results
    
    def analyze_from_df(self, df, text_col='clean_content'):
        """对DataFrame中的文本进行方面级情感分析
        
        Args:
            df: 包含文本的DataFrame
            text_col: 文本列名
            
        Returns:
            df_with_aspects: 添加了方面级情感分析结果的DataFrame
        """
        # 复制一份，避免修改原始数据
        df_with_aspects = df.copy()
        
        # 确保text_col存在
        if text_col not in df_with_aspects.columns:
            print(f"错误: 列 '{text_col}' 不存在于数据中")
            return df_with_aspects
        
        print(f"开始对 {len(df_with_aspects)} 条文本进行方面级情感分析...")
        
        # 进行方面级情感分析
        df_with_aspects['aspect_results'] = df_with_aspects[text_col].apply(
            lambda x: self.analyze_aspect_sentiment(x))
        
        # 提取常见方面词
        all_aspects = []
        for results in df_with_aspects['aspect_results']:
            all_aspects.extend([aspect for aspect, _, _, _, _ in results])
        
        # 获取前10个最常见的方面词
        top_aspects = [aspect for aspect, _ in Counter(all_aspects).most_common(10)]
        
        # 为每个常见方面词创建情感极性列
        for aspect in top_aspects:
            col_name = f'aspect_{aspect}'
            df_with_aspects[col_name] = df_with_aspects['aspect_results'].apply(
                lambda results: next((polarity for a, polarity, _, _, _ in results if a == aspect), '无')
            )
        
        print("方面级情感分析完成。")
        return df_with_aspects
    
    def get_aspect_summary(self, df, aspect_col='aspect_results'):
        """生成方面级情感分析摘要
        
        Args:
            df: 包含方面级情感分析结果的DataFrame
            aspect_col: 方面级情感分析结果列名
            
        Returns:
            summary: 方面级情感分析摘要
        """
        if aspect_col not in df.columns:
            print(f"错误: 列 '{aspect_col}' 不存在于数据中")
            return {}
        
        # 统计各方面的情感极性
        aspect_stats = defaultdict(lambda: {'正面': 0, '中性': 0, '负面': 0, '总数': 0, '样例': {}})
        
        for results in df[aspect_col]:
            for aspect, polarity, intensity, context, sentiment_words in results:
                aspect_stats[aspect][polarity] += 1
                aspect_stats[aspect]['总数'] += 1
                
                # 保存高强度的样例
                if intensity > 0.5 and len(aspect_stats[aspect]['样例']) < 3:
                    aspect_stats[aspect]['样例'][context] = {'极性': polarity, '强度': intensity, '情感词': sentiment_words}
        
        # 计算方面情感比例
        for aspect, stats in aspect_stats.items():
            total = stats['总数']
            if total > 0:
                stats['正面比例'] = stats['正面'] / total
                stats['中性比例'] = stats['中性'] / total
                stats['负面比例'] = stats['负面'] / total
        
        # 按总数排序
        sorted_aspects = sorted(aspect_stats.items(), key=lambda x: x[1]['总数'], reverse=True)
        
        return dict(sorted_aspects)
    
    def plot_aspect_sentiment(self, summary, top_n=10, title='方面级情感分析', figsize=(12, 8), save_path=None):
        """绘制方面级情感分析结果
        
        Args:
            summary: 方面级情感分析摘要
            top_n: 显示前N个方面词
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        if not summary:
            print("错误: 方面级情感分析摘要为空")
            return
        
        # 获取前N个方面词
        top_aspects = list(summary.keys())[:top_n]
        
        plt.figure(figsize=figsize)
        
        # 准备数据
        aspects = []
        pos_ratio = []
        neu_ratio = []
        neg_ratio = []
        total_counts = []
        
        for aspect in top_aspects:
            stats = summary[aspect]
            aspects.append(aspect)
            pos_ratio.append(stats.get('正面比例', 0) * 100)
            neu_ratio.append(stats.get('中性比例', 0) * 100)
            neg_ratio.append(stats.get('负面比例', 0) * 100)
            total_counts.append(stats['总数'])
        
        # 绘制堆叠条形图
        width = 0.8
        bar_pos = np.arange(len(aspects))
        
        plt.bar(bar_pos, pos_ratio, width, label='正面', color='#5cb85c')
        plt.bar(bar_pos, neu_ratio, width, bottom=pos_ratio, label='中性', color='#f0ad4e')
        plt.bar(bar_pos, neg_ratio, width, bottom=[p+n for p, n in zip(pos_ratio, neu_ratio)], label='负面', color='#d9534f')
        
        # 添加数据标签
        for i, count in enumerate(total_counts):
            plt.text(i, 101, f'n={count}', ha='center', va='bottom', fontproperties=font)
        
        plt.xlabel('方面词', fontproperties=font)
        plt.ylabel('比例(%)', fontproperties=font)
        plt.title(title, fontproperties=font)
        plt.xticks(bar_pos, aspects, rotation=45, ha='right', fontproperties=font)
        plt.ylim(0, 110)  # 留出空间显示数量标签
        plt.legend(prop=font)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"方面级情感分析图已保存到 {save_path}")
        
        plt.close()
    
    def plot_aspect_distribution(self, summary, title='方面词分布', figsize=(12, 6), save_path=None):
        """绘制方面词分布图
        
        Args:
            summary: 方面级情感分析摘要
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        if not summary:
            print("错误: 方面级情感分析摘要为空")
            return
        
        plt.figure(figsize=figsize)
        
        # 准备数据
        aspects = list(summary.keys())
        counts = [stats['总数'] for stats in summary.values()]
        
        # 按数量降序排序
        sorted_indices = np.argsort(counts)[::-1]
        sorted_aspects = [aspects[i] for i in sorted_indices]
        sorted_counts = [counts[i] for i in sorted_indices]
        
        # 绘制水平条形图
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(sorted_aspects)))
        bars = plt.barh(range(len(sorted_aspects)), sorted_counts, color=colors)
        
        # 添加数据标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                    f'{width}',
                    ha='left', va='center', fontproperties=font)
        
        plt.xlabel('出现次数', fontproperties=font)
        plt.ylabel('方面词', fontproperties=font)
        plt.title(title, fontproperties=font)
        plt.yticks(range(len(sorted_aspects)), sorted_aspects, fontproperties=font)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"方面词分布图已保存到 {save_path}")
        
        plt.close()

# ====================== 11. 集成分析函数 ======================

def perform_comprehensive_analysis(df, content_col='content'):
    """对数据进行综合分析，集成所有NLP功能
    
    Args:
        df: 包含文本数据的DataFrame
        content_col: 文本内容列名
        
    Returns:
        results_df: 添加了各种分析结果的DataFrame
        analysis_results: 包含各种分析摘要的字典
    """
    # 1. 数据预处理
    print("第1步: 文本预处理...")
    preprocessor = TextPreprocessor()
    processed_df = preprocessor.process_dataframe(df, content_col=content_col)
    
    # 2. 情感分析
    print("\n第2步: 情感分析...")
    analyzer = SentimentAnalyzer(num_classes=3)
    sentiment_df = analyzer.convert_score_to_sentiment(processed_df)
    # 如果有训练好的模型，则加载模型
    if os.path.exists(analyzer.model_save_path):
        analyzer.load_model()
        sentiment_df = analyzer.predict_dataframe(sentiment_df, text_col='clean_content')
    
    # 3. 命名实体识别
    print("\n第3步: 命名实体识别...")
    ner = NamedEntityRecognition()
    ner_df = ner.process_dataframe(sentiment_df, text_col='clean_content')
    entity_summary = ner.generate_entity_summary(ner_df)
    
    # 4. 依存句法分析（示例分析）
    print("\n第4步: 依存句法分析...")
    parser = DependencyParser()
    # 为了效率，只对前5条记录进行依存分析
    sample_texts = sentiment_df['clean_content'].head(5).tolist()
    parsing_results = []
    for text in sample_texts:
        parsing_results.append(parser.parse(text))
    
    # 5. 关键词抽取
    print("\n第5步: 关键词抽取...")
    keyword_extractor = KeywordExtractor()
    keyword_df = keyword_extractor.extract_from_df(sentiment_df, text_col='clean_content')
    
    # 聚合关键词
    aggregated_keywords = keyword_extractor.aggregate_keywords(keyword_df)
    
    # 6. 文本摘要
    print("\n第6步: 文本摘要...")
    summarizer = TextSummarizer()
    # 为较长的文本生成摘要
    keyword_df['is_long'] = keyword_df['clean_content'].apply(lambda x: len(x) > 100)
    long_texts_df = keyword_df[keyword_df['is_long']]
    if not long_texts_df.empty:
        summary_df = summarizer.summarize_from_df(long_texts_df, text_col='clean_content')
        # 将摘要合并回原始DataFrame
        keyword_df = keyword_df.join(summary_df[['summary']], how='left')
    
    # 7. 主题建模
    print("\n第7步: 主题建模...")
    topic_modeler = TopicModeler(n_topics=5)
    if len(keyword_df) >= 20:  # 确保有足够的数据进行主题建模
        topic_df = topic_modeler.fit_transform_from_df(keyword_df, text_col='clean_content')
        topics = topic_modeler.get_topics(n_top_words=8)
    else:
        topic_df = keyword_df
        topics = []
        print("数据量不足，跳过主题建模")
    
    # 8. 方面级情感分析
    print("\n第8步: 方面级情感分析...")
    aspect_analyzer = AspectBasedSentimentAnalyzer(dependency_parser=parser)
    aspect_df = aspect_analyzer.analyze_from_df(topic_df, text_col='clean_content')
    aspect_summary = aspect_analyzer.get_aspect_summary(aspect_df)
    
    # 9. 生成可视化
    print("\n第9步: 生成可视化...")
    visualizer = SentimentVisualizer()
    
    # 确保可视化目录存在
    os.makedirs('static/visualizations', exist_ok=True)
    os.makedirs('static/visualizations/advanced', exist_ok=True)
    
    # 情感分布饼图
    visualizer.plot_sentiment_pie(
        aspect_df, 
        save_path='static/visualizations/sentiment_pie.png'
    )
    
    # 如果有平台信息，生成平台比较
    if 'platform' in aspect_df.columns:
        visualizer.plot_platform_comparison(
            aspect_df,
            save_path='static/visualizations/platform_comparison.png'
        )
    
    # 词云图
    keyword_extractor.plot_keyword_cloud(
        aggregated_keywords,
        save_path='static/visualizations/keyword_cloud.png'
    )
    
    # 关键词条形图
    keyword_extractor.plot_top_keywords(
        aggregated_keywords,
        save_path='static/visualizations/top_keywords.png'
    )
    
    # 实体分布图
    ner.plot_entity_distribution(
        entity_summary,
        save_path='static/visualizations/advanced/entity_distribution.png'
    )
    
    # 主题关键词图
    if topics:
        topic_modeler.plot_topic_keywords(
            topics,
            save_path='static/visualizations/advanced/topic_keywords.png'
        )
    
    # 方面情感分析图
    if aspect_summary:
        aspect_analyzer.plot_aspect_sentiment(
            aspect_summary,
            save_path='static/visualizations/advanced/aspect_sentiment.png'
        )
    
    # 10. 收集分析结果
    analysis_results = {
        'sentiment_distribution': aspect_df['sentiment_text'].value_counts().to_dict(),
        'entity_summary': entity_summary,
        'top_keywords': aggregated_keywords[:20],
        'topics': topics,
        'aspect_summary': aspect_summary
    }
    
    # 如果有平台信息，添加平台统计
    if 'platform' in aspect_df.columns:
        analysis_results['platform_distribution'] = aspect_df['platform'].value_counts().to_dict()
    
    print("\n综合分析完成！")
    return aspect_df, analysis_results

# ====================== 12. 主函数 ======================

def main():
    """主程序入口"""
    print("="*50)
    print("基于BERT的多源数据情感分析系统 - 增强版")
    print("="*50)
    
    try:
        # 1. 数据爬取（示例）
        print("\n1. 数据爬取示例")
        crawler = MultiSourceCrawler()
        
        # 设置要爬取的ID
        movie_ids = ['35267208']  # 电影《流浪地球2》的豆瓣ID
        product_ids = ['100035955048']  # 京东商品ID
        song_ids = ['1999869183']  # 网易云音乐歌曲ID
        
        # 检查是否有已爬取的数据
        data_file = 'data/all_reviews.csv'
        if not os.path.exists(data_file):
            print(f"未找到已爬取的数据，开始爬取...")
            # 爬取所有来源的评论
            reviews_df = crawler.crawl_all_sources(
                movie_ids=movie_ids,
                product_ids=product_ids,
                song_ids=song_ids,
                pages=3  # 每个ID爬取3页
            )
        else:
            print(f"加载已有数据: {data_file}")
            reviews_df = pd.read_csv(data_file)
        
        # 2. 综合分析
        results_df, analysis_results = perform_comprehensive_analysis(reviews_df)
        
        # 3. 展示分析结果
        print("\n分析结果摘要:")
        print(f"- 总评论数: {len(results_df)}")
        print(f"- 情感分布: {analysis_results['sentiment_distribution']}")
        print(f"- 实体统计: 共识别 {analysis_results['entity_summary']['total_count']} 个实体")
        print(f"- 主题数量: {len(analysis_results['topics'])}")
        print(f"- 识别方面词: {len(analysis_results['aspect_summary'])} 个")
        print(f"- 前5个关键词: {[word for word, _ in analysis_results['top_keywords'][:5]]}")
        
        # 4. 保存结果
        results_df.to_csv('data/analysis_results.csv', index=False, encoding='utf-8')
        print("\n分析结果已保存到 data/analysis_results.csv")
        print("可视化结果已保存到 static/visualizations 目录")
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
        
