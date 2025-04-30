'''
高级NLP分析模块
实现更多自然语言处理任务:
1. 命名实体识别(NER)
2. 依存句法分析
3. 关键词抽取
4. 文本摘要
5. 主题建模
6. 方面级情感分析
'''

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter, defaultdict
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# 文本处理工具库
import jieba
import jieba.posseg as pseg
from jieba.analyse import textrank, extract_tags

# 引入更多深度学习模型支持
import torch
from transformers import (
    BertTokenizer, BertModel, BertForTokenClassification, 
    BertForSequenceClassification, BertForQuestionAnswering,
    pipeline, AutoTokenizer, AutoModel
)

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
os.makedirs('data/advanced', exist_ok=True)
os.makedirs('static/visualizations/advanced', exist_ok=True)

# 设置中文字体
try:
    font_paths = [
        'static/fonts/SimHei.ttf',
        'SimHei.ttf',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
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
        df_len['original_length'] = df_len[text_col].apply(len)
        df_len['summary_length'] = df_len[summary_col].apply(len)
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
                self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words=self.stopwords)
            else:  # lda
                # 对于LDA，使用词频计数
                self.vectorizer = CountVectorizer(max_features=self.max_features, stop_words=self.stopwords)
            
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

def analyze_text_with_all_modules(text):
    """使用所有高级NLP模块分析一段文本，用于示例和测试
    
    Args:
        text: 待分析的文本
        
    Returns:
        results: 各模块的分析结果
    """
    results = {}
    
    try:
        print("开始全面分析文本...")
        
        # 1. 命名实体识别
        ner = NamedEntityRecognition()
        entities = ner.recognize(text)
        results['entities'] = entities
        
        # 2. 依存句法分析
        parser = DependencyParser()
        parse_result = parser.parse(text)
        results['dependency'] = parse_result
        
        # 3. 关键词抽取
        keyword_extractor = KeywordExtractor()
        keywords = keyword_extractor.extract(text, top_k=10)
        results['keywords'] = keywords
        
        # 4. 文本摘要
        summarizer = TextSummarizer()
        summary = summarizer.summarize(text, max_length=2)
        results['summary'] = summary
        
        # 5. 方面级情感分析
        aspect_analyzer = AspectBasedSentimentAnalyzer()
        aspect_results = aspect_analyzer.analyze_aspect_sentiment(text)
        results['aspect_sentiment'] = aspect_results
        
        print("文本分析完成。")
        return results
    
    except Exception as e:
        print(f"文本分析过程中出错: {e}")
        return {'error': str(e)}

# 测试代码
if __name__ == "__main__":
    test_text = "这部电影的情节非常紧凑，演员的表演很出色，尤其是主角张三的演技真的太棒了，但是特效做得一般，音乐很好听。"
    result = analyze_text_with_all_modules(test_text)
    print(result)
