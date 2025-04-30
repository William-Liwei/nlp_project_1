# '''
# 基于BERT的多源数据情感分析系统 - 用户界面
# 使用Flask框架实现Web界面，展示情感分析结果
# '''

# import os
# import pandas as pd
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # 防止没有GUI环境时报错
# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
# from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
# from werkzeug.utils import secure_filename
# import json
# import time
# from datetime import datetime

# # 导入我们的情感分析模块 - 修正导入路径
# from sentiment_analysis import MultiSourceCrawler, TextPreprocessor, SentimentAnalyzer, SentimentVisualizer

# # 确保所需目录存在
# os.makedirs('uploads', exist_ok=True)
# os.makedirs('static/visualizations', exist_ok=True)
# os.makedirs('static/fonts', exist_ok=True)
# os.makedirs('data', exist_ok=True)
# os.makedirs('models', exist_ok=True)

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

# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# app = Flask(__name__)

# # 配置上传文件目录
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# # 初始化模块
# crawler = MultiSourceCrawler()
# preprocessor = TextPreprocessor()
# analyzer = SentimentAnalyzer(num_classes=3)  # 3分类：负面、中性、正面
# visualizer = SentimentVisualizer()

# # 全局变量存储当前数据
# current_data = None

# def allowed_file(filename):
#     """检查文件类型是否允许上传"""
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/')
# def index():
#     """主页"""
#     return render_template('index.html')

# @app.route('/crawl', methods=['GET', 'POST'])
# def crawl_data():
#     """爬取数据页面及处理"""
#     if request.method == 'POST':
#         try:
#             platform = request.form.get('platform')
#             ids = request.form.get('ids', '').split(',')
#             ids = [id.strip() for id in ids if id.strip()]  # 过滤空ID
#             pages = int(request.form.get('pages', 1))
            
#             result_df = None
            
#             if not ids and platform != 'all':
#                 return render_template('crawl.html', error="请输入至少一个ID")
            
#             if platform == 'douban':
#                 # 爬取豆瓣电影评论
#                 all_reviews = []
#                 for movie_id in ids:
#                     reviews = crawler.crawl_douban_movie_reviews(movie_id, pages)
#                     all_reviews.extend(reviews)
#                 result_df = pd.DataFrame(all_reviews) if all_reviews else None
            
#             elif platform == 'jd':
#                 # 爬取京东商品评论
#                 all_reviews = []
#                 for product_id in ids:
#                     reviews = crawler.crawl_jd_product_reviews(product_id, pages)
#                     all_reviews.extend(reviews)
#                 result_df = pd.DataFrame(all_reviews) if all_reviews else None
            
#             elif platform == 'netease':
#                 # 爬取网易云音乐评论
#                 all_reviews = []
#                 for song_id in ids:
#                     reviews = crawler.crawl_netease_music_comments(song_id, pages)
#                     all_reviews.extend(reviews)
#                 result_df = pd.DataFrame(all_reviews) if all_reviews else None
            
#             elif platform == 'all':
#                 # 爬取所有平台
#                 movie_ids = request.form.get('movie_ids', '').split(',')
#                 movie_ids = [id.strip() for id in movie_ids if id.strip()]
                
#                 product_ids = request.form.get('product_ids', '').split(',')
#                 product_ids = [id.strip() for id in product_ids if id.strip()]
                
#                 song_ids = request.form.get('song_ids', '').split(',')
#                 song_ids = [id.strip() for id in song_ids if id.strip()]
                
#                 if not movie_ids and not product_ids and not song_ids:
#                     return render_template('crawl.html', error="请至少输入一个平台的ID")
                
#                 result_df = crawler.crawl_all_sources(
#                     movie_ids=movie_ids if movie_ids else None,
#                     product_ids=product_ids if product_ids else None,
#                     song_ids=song_ids if song_ids else None,
#                     pages=pages
#                 )
            
#             if result_df is not None and not result_df.empty:
#                 # 预处理数据
#                 global current_data
#                 current_data = preprocessor.process_dataframe(result_df)
                
#                 # 保存为临时CSV
#                 timestamp = int(time.time())
#                 temp_file = f'uploads/temp_data_{timestamp}.csv'
#                 current_data.to_csv(temp_file, index=False, encoding='utf-8')
                
#                 return redirect(url_for('analysis', file=os.path.basename(temp_file)))
#             else:
#                 return render_template('crawl.html', error="爬取数据失败，请检查ID是否正确或网络连接是否正常")
                
#         except Exception as e:
#             return render_template('crawl.html', error=f"处理过程中出错: {str(e)}")
    
#     return render_template('crawl.html')

# @app.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     """上传文件页面及处理"""
#     if request.method == 'POST':
#         # 检查是否有文件
#         if 'file' not in request.files:
#             return render_template('upload.html', error="没有选择文件")
        
#         file = request.files['file']
        
#         # 检查文件名
#         if file.filename == '':
#             return render_template('upload.html', error="没有选择文件")
        
#         if file and allowed_file(file.filename):
#             try:
#                 filename = secure_filename(file.filename)
#                 filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#                 file.save(filepath)
                
#                 # 根据文件类型读取数据
#                 if filename.endswith('.csv'):
#                     df = pd.read_csv(filepath, encoding='utf-8')
#                 else:  # Excel文件
#                     df = pd.read_excel(filepath)
                
#                 # 检查必要的列
#                 required_cols = ['content']
#                 missing_cols = [col for col in required_cols if col not in df.columns]
                
#                 if missing_cols:
#                     return render_template('upload.html', error=f"文件缺少必要的列: {', '.join(missing_cols)}")
                
#                 # 预处理数据
#                 global current_data
#                 current_data = preprocessor.process_dataframe(df)
                
#                 # 保存为临时CSV
#                 timestamp = int(time.time())
#                 temp_file = f'uploads/temp_data_{timestamp}.csv'
#                 current_data.to_csv(temp_file, index=False, encoding='utf-8')
                
#                 return redirect(url_for('analysis', file=os.path.basename(temp_file)))
            
#             except Exception as e:
#                 return render_template('upload.html', error=f"处理文件时出错: {str(e)}")
#         else:
#             return render_template('upload.html', error="不支持的文件类型，请上传CSV或Excel文件")
    
#     return render_template('upload.html')

# @app.route('/analysis')
# def analysis():
#     """情感分析页面"""
#     file = request.args.get('file')
    
#     if not file:
#         return redirect(url_for('index'))
    
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
    
#     if not os.path.exists(filepath):
#         return redirect(url_for('index'))
    
#     # 读取数据
#     try:
#         global current_data
#         current_data = pd.read_csv(filepath, encoding='utf-8')
        
#         # 如果已经有情感分析结果，则直接展示
#         if 'sentiment_label' in current_data.columns:
#             return render_template('analysis.html', file=file, has_result=True)
        
#         return render_template('analysis.html', file=file, has_result=False)
#     except Exception as e:
#         return render_template('upload.html', error=f"读取数据文件时出错: {str(e)}")

# @app.route('/process', methods=['POST'])
# def process_data():
#     """处理数据并进行情感分析"""
#     file = request.form.get('file')
#     method = request.form.get('method')
    
#     if not file:
#         return jsonify({'status': 'error', 'message': '未指定文件'})
    
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
    
#     if not os.path.exists(filepath):
#         return jsonify({'status': 'error', 'message': '文件不存在'})
    
#     # 读取数据
#     try:
#         global current_data
#         current_data = pd.read_csv(filepath, encoding='utf-8')
        
#         # 根据选择的方法进行处理
#         if method == 'score':
#             # 使用评分转换为情感标签
#             if 'score' in current_data.columns:
#                 current_data = analyzer.convert_score_to_sentiment(current_data)
#             else:
#                 return jsonify({'status': 'error', 'message': '数据中没有score列，无法使用评分转换'})
#         else:  # method == 'bert'
#             # 使用BERT模型预测
#             try:
#                 if not os.path.exists(analyzer.model_save_path):
#                     # 如果模型不存在，先用评分数据训练一个简单模型
#                     if 'score' in current_data.columns:
#                         train_df = analyzer.convert_score_to_sentiment(current_data)
#                         train_loader, test_loader = analyzer.prepare_data(train_df)
#                         analyzer.train(train_loader, epochs=1)  # 实际应用中应增加轮数
#                     else:
#                         return jsonify({'status': 'error', 'message': '模型不存在且无法训练，请先上传带评分的数据'})
                
#                 # 加载模型
#                 analyzer.load_model()
                
#                 # 预测情感
#                 current_data = analyzer.predict_dataframe(current_data)
#             except Exception as e:
#                 return jsonify({'status': 'error', 'message': f'BERT分析失败: {str(e)}'})
        
#         # 保存结果
#         current_data.to_csv(filepath, index=False, encoding='utf-8')
        
#         # 生成可视化
#         try:
#             generate_visualizations(current_data)
#         except Exception as e:
#             print(f"生成可视化时出错: {e}")
#             # 即使可视化生成失败也继续返回成功
        
#         return jsonify({'status': 'success'})
    
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': f'处理数据时出错: {str(e)}'})

# def generate_visualizations(df):
#     """生成各种可视化图表"""
#     timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
#     # 情感分布饼图
#     visualizer.plot_sentiment_pie(
#         df, 
#         save_path=f'static/visualizations/sentiment_pie_{timestamp}.png'
#     )
    
#     # 如果有平台信息，生成平台对比图
#     if 'platform' in df.columns:
#         visualizer.plot_platform_comparison(
#             df,
#             save_path=f'static/visualizations/platform_comparison_{timestamp}.png'
#         )
    
#     # 情感词云
#     if 'words' in df.columns:
#         visualizer.plot_wordcloud(
#             df, 
#             sentiment_value='正面',
#             save_path=f'static/visualizations/positive_wordcloud_{timestamp}.png'
#         )
        
#         visualizer.plot_wordcloud(
#             df, 
#             sentiment_value='负面',
#             save_path=f'static/visualizations/negative_wordcloud_{timestamp}.png'
#         )
    
#     # 情感趋势（如果有时间信息）
#     if 'time' in df.columns:
#         visualizer.plot_sentiment_trend(
#             df,
#             save_path=f'static/visualizations/sentiment_trend_{timestamp}.png'
#         )

# @app.route('/results')
# def results():
#     """展示分析结果"""
#     file = request.args.get('file')
    
#     if not file:
#         return redirect(url_for('index'))
    
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
    
#     if not os.path.exists(filepath):
#         return redirect(url_for('index'))
    
#     # 读取数据
#     try:
#         global current_data
#         current_data = pd.read_csv(filepath, encoding='utf-8')
        
#         # 确保数据已经进行了情感分析
#         if 'sentiment_label' not in current_data.columns:
#             return redirect(url_for('analysis', file=file))
        
#         # 获取可视化图表
#         visualizations = []
#         for filename in os.listdir('static/visualizations'):
#             if filename.endswith('.png'):
#                 visualizations.append(filename)
        
#         # 数据摘要
#         if 'platform' in current_data.columns:
#             platform_counts = current_data['platform'].value_counts().to_dict()
#         else:
#             platform_counts = {}
        
#         sentiment_counts = current_data['sentiment_text'].value_counts().to_dict()
        
#         # 示例评论
#         positive_examples = []
#         negative_examples = []
#         neutral_examples = []
        
#         if 'sentiment_text' in current_data.columns and 'content' in current_data.columns:
#             for sentiment in ['正面', '负面', '中性']:
#                 filtered = current_data[current_data['sentiment_text'] == sentiment]
#                 if not filtered.empty:
#                     samples = filtered.sample(min(3, len(filtered)))
                    
#                     for _, row in samples.iterrows():
#                         example = {
#                             'content': row['content'],
#                             'platform': row.get('platform', '未知'),
#                             'score': row.get('score', '未知'),
#                             'probability': row.get('sentiment_prob', 0.0)
#                         }
                        
#                         if sentiment == '正面':
#                             positive_examples.append(example)
#                         elif sentiment == '负面':
#                             negative_examples.append(example)
#                         else:
#                             neutral_examples.append(example)
        
#         # 返回结果页面
#         return render_template(
#             'results.html', 
#             file=file,
#             visualizations=visualizations,
#             platform_counts=platform_counts,
#             sentiment_counts=sentiment_counts,
#             positive_examples=positive_examples,
#             negative_examples=negative_examples,
#             neutral_examples=neutral_examples,
#             total_count=len(current_data)
#         )
#     except Exception as e:
#         return render_template('upload.html', error=f"读取分析结果时出错: {str(e)}")

# @app.route('/analyze_text', methods=['GET', 'POST'])
# def analyze_text():
#     """单条文本情感分析"""
#     if request.method == 'POST':
#         text = request.form.get('text')
        
#         if not text:
#             return render_template('analyze_text.html', error="请输入文本")
        
#         try:
#             # 加载模型
#             if not os.path.exists(analyzer.model_save_path):
#                 return render_template('analyze_text.html', error="模型不存在，请先训练模型")
            
#             analyzer.load_model()
            
#             # 预处理文本
#             clean_text = preprocessor.clean_text(text)
#             words = preprocessor.segment(clean_text)
            
#             # 预测情感
#             result = analyzer.predict(clean_text)
            
#             if not result.empty:
#                 sentiment = result['sentiment_text'].iloc[0]
#                 probability = result['probability'].iloc[0]
                
#                 # 返回结果
#                 return render_template(
#                     'analyze_text.html', 
#                     text=text,
#                     clean_text=clean_text,
#                     words=' '.join(words),
#                     sentiment=sentiment,
#                     probability=f"{probability:.2f}"
#                 )
#             else:
#                 return render_template('analyze_text.html', error="情感分析失败")
        
#         except Exception as e:
#             return render_template('analyze_text.html', error=f"情感分析出错: {str(e)}")
    
#     return render_template('analyze_text.html')

# @app.route('/about')
# def about():
#     """关于页面"""
#     return render_template('about.html')

# @app.route('/api/data')
# def api_data():
#     """提供数据API"""
#     file = request.args.get('file')
    
#     if not file:
#         return jsonify({'status': 'error', 'message': '未指定文件'})
    
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
    
#     if not os.path.exists(filepath):
#         return jsonify({'status': 'error', 'message': '文件不存在'})
    
#     try:
#         # 读取数据
#         df = pd.read_csv(filepath, encoding='utf-8')
        
#         # 转换为JSON格式
#         data = df.to_dict(orient='records')
        
#         return jsonify({'status': 'success', 'data': data})
    
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': f'读取数据出错: {str(e)}'})

# @app.route('/favicon.ico')
# def favicon():
#     """提供favicon"""
#     return send_from_directory(os.path.join(app.root_path, 'static'),
#                                'favicon.ico', mimetype='image/vnd.microsoft.icon')

# if __name__ == '__main__':
#     # 首次运行时创建必要的目录
#     for dir_path in ['data', 'models', 'uploads', 'static/visualizations', 'static/fonts']:
#         os.makedirs(dir_path, exist_ok=True)
    
#     # 运行Flask应用
#     app.run(debug=True, host='0.0.0.0', port=5000)


'''
基于BERT的多源数据情感分析系统 - 用户界面
使用Flask框架实现Web界面，展示情感分析结果
'''

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 防止没有GUI环境时报错
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import json
import time
import random
from datetime import datetime

# 导入我们的情感分析模块
from sentiment_analysis import (
    MultiSourceCrawler, TextPreprocessor, SentimentAnalyzer, SentimentVisualizer,
    NamedEntityRecognition, DependencyParser, KeywordExtractor, 
    TextSummarizer, TopicModeler, AspectBasedSentimentAnalyzer, perform_comprehensive_analysis
)

# 检查HanLP和LAC是否可用
try:
    import hanlp
    HANLP_AVAILABLE = True
except ImportError:
    HANLP_AVAILABLE = False

try:
    from LAC import LAC
    LAC_AVAILABLE = True
except ImportError:
    LAC_AVAILABLE = False

# 确保所需目录存在
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/visualizations', exist_ok=True)
os.makedirs('static/visualizations/advanced', exist_ok=True)
os.makedirs('static/fonts', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('data/advanced', exist_ok=True)
os.makedirs('models', exist_ok=True)

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

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

app = Flask(__name__)

# 配置上传文件目录
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 初始化模块
crawler = MultiSourceCrawler()
preprocessor = TextPreprocessor()
analyzer = SentimentAnalyzer(num_classes=3)  # 3分类：负面、中性、正面
visualizer = SentimentVisualizer()

# 全局变量存储当前数据
current_data = None
current_advanced_results = None

# Global variables to track progress
analysis_progress = {
    'current': 0,
    'total': 0,
    'status': 'idle',
    'message': ''
}

advanced_analysis_progress = {
    'current': 0,
    'total': 0,
    'status': 'idle',
    'message': '',
    'current_module': ''
}

def reset_progress(progress_dict):
    """Reset progress tracker"""
    progress_dict['current'] = 0
    progress_dict['total'] = 0
    progress_dict['status'] = 'idle'
    progress_dict['message'] = ''
    if 'current_module' in progress_dict:
        progress_dict['current_module'] = ''

# Add a route to get progress
@app.route('/progress', methods=['GET'])
def get_progress():
    """Get current progress of analysis"""
    analysis_type = request.args.get('type', 'basic')
    
    if analysis_type == 'advanced':
        return jsonify(advanced_analysis_progress)
    else:
        return jsonify(analysis_progress)

def allowed_file(filename):
    """检查文件类型是否允许上传"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page that also lists available results"""
    # List all files in uploads directory
    upload_files = os.listdir(app.config['UPLOAD_FOLDER'])
    
    # Filter to get only the result files (exclude temp files)
    result_files = []
    for file in upload_files:
        if file.startswith('temp_data_') and file.endswith('.csv'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
            
            # Check if this file has analysis results
            has_results = False
            has_advanced_results = False
            
            try:
                df = pd.read_csv(file_path)
                if 'sentiment_label' in df.columns:
                    has_results = True
                
                # Check if advanced results exist
                advanced_results_path = f"{file_path.rsplit('.', 1)[0]}_advanced_results.json"
                if os.path.exists(advanced_results_path):
                    has_advanced_results = True
            except:
                continue
            
            # Get file creation time and format it
            creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
            creation_time_str = creation_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add to result files list
            result_files.append({
                'filename': file,
                'creation_time': creation_time_str,
                'has_results': has_results,
                'has_advanced_results': has_advanced_results,
                'size': f"{os.path.getsize(file_path) / (1024 * 1024):.2f} MB"
            })
    
    # Sort by creation time (newest first)
    result_files.sort(key=lambda x: x['creation_time'], reverse=True)
    
    return render_template('index.html', result_files=result_files)

@app.route('/crawl', methods=['GET', 'POST'])
def crawl_data():
    """爬取数据页面及处理"""
    if request.method == 'POST':
        try:
            platform = request.form.get('platform')
            ids = request.form.get('ids', '').split(',')
            ids = [id.strip() for id in ids if id.strip()]  # 过滤空ID
            pages = int(request.form.get('pages', 1))
            
            result_df = None
            
            if not ids and platform != 'all':
                return render_template('crawl.html', error="请输入至少一个ID")
            
            if platform == 'douban':
                # 爬取豆瓣电影评论
                all_reviews = []
                for movie_id in ids:
                    reviews = crawler.crawl_douban_movie_reviews(movie_id, pages)
                    all_reviews.extend(reviews)
                result_df = pd.DataFrame(all_reviews) if all_reviews else None
            
            elif platform == 'jd':
                # 爬取京东商品评论
                all_reviews = []
                for product_id in ids:
                    reviews = crawler.crawl_jd_product_reviews(product_id, pages)
                    all_reviews.extend(reviews)
                result_df = pd.DataFrame(all_reviews) if all_reviews else None
            
            elif platform == 'netease':
                # 爬取网易云音乐评论
                all_reviews = []
                for song_id in ids:
                    reviews = crawler.crawl_netease_music_comments(song_id, pages)
                    all_reviews.extend(reviews)
                result_df = pd.DataFrame(all_reviews) if all_reviews else None
            
            elif platform == 'all':
                # 爬取所有平台
                movie_ids = request.form.get('movie_ids', '').split(',')
                movie_ids = [id.strip() for id in movie_ids if id.strip()]
                
                product_ids = request.form.get('product_ids', '').split(',')
                product_ids = [id.strip() for id in product_ids if id.strip()]
                
                song_ids = request.form.get('song_ids', '').split(',')
                song_ids = [id.strip() for id in song_ids if id.strip()]
                
                if not movie_ids and not product_ids and not song_ids:
                    return render_template('crawl.html', error="请至少输入一个平台的ID")
                
                result_df = crawler.crawl_all_sources(
                    movie_ids=movie_ids if movie_ids else None,
                    product_ids=product_ids if product_ids else None,
                    song_ids=song_ids if song_ids else None,
                    pages=pages
                )
            
            if result_df is not None and not result_df.empty:
                # 预处理数据
                global current_data
                current_data = preprocessor.process_dataframe(result_df)
                
                # 保存为临时CSV
                timestamp = int(time.time())
                temp_file = f'uploads/temp_data_{timestamp}.csv'
                current_data.to_csv(temp_file, index=False, encoding='utf-8')
                
                return redirect(url_for('analysis', file=os.path.basename(temp_file)))
            else:
                return render_template('crawl.html', error="爬取数据失败，请检查ID是否正确或网络连接是否正常")
                
        except Exception as e:
            return render_template('crawl.html', error=f"处理过程中出错: {str(e)}")
    
    return render_template('crawl.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """上传文件页面及处理"""
    if request.method == 'POST':
        # 检查是否有文件
        if 'file' not in request.files:
            return render_template('upload.html', error="没有选择文件")
        
        file = request.files['file']
        
        # 检查文件名
        if file.filename == '':
            return render_template('upload.html', error="没有选择文件")
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # 根据文件类型读取数据
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath, encoding='utf-8')
                else:  # Excel文件
                    df = pd.read_excel(filepath)
                
                # 检查必要的列
                required_cols = ['content']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    return render_template('upload.html', error=f"文件缺少必要的列: {', '.join(missing_cols)}")
                
                # 预处理数据
                global current_data
                current_data = preprocessor.process_dataframe(df)
                
                # 保存为临时CSV
                timestamp = int(time.time())
                temp_file = f'uploads/temp_data_{timestamp}.csv'
                current_data.to_csv(temp_file, index=False, encoding='utf-8')
                
                return redirect(url_for('analysis', file=os.path.basename(temp_file)))
            
            except Exception as e:
                return render_template('upload.html', error=f"处理文件时出错: {str(e)}")
        else:
            return render_template('upload.html', error="不支持的文件类型，请上传CSV或Excel文件")
    
    return render_template('upload.html')

@app.route('/analysis')
def analysis():
    """情感分析页面"""
    file = request.args.get('file')
    
    if not file:
        return redirect(url_for('index'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
    
    if not os.path.exists(filepath):
        return redirect(url_for('index'))
    
    # 读取数据
    try:
        global current_data
        current_data = pd.read_csv(filepath, encoding='utf-8')
        
        # 如果已经有情感分析结果，则直接展示
        if 'sentiment_label' in current_data.columns:
            return render_template('analysis.html', file=file, has_result=True)
        
        return render_template('analysis.html', file=file, has_result=False)
    except Exception as e:
        return render_template('upload.html', error=f"读取数据文件时出错: {str(e)}")

@app.route('/process', methods=['POST'])
def process_data():
    """处理数据并进行情感分析"""
    global analysis_progress
    
    file = request.form.get('file')
    method = request.form.get('method')
    
    if not file:
        return jsonify({'status': 'error', 'message': '未指定文件'})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
    
    if not os.path.exists(filepath):
        return jsonify({'status': 'error', 'message': '文件不存在'})
    
    # 重置进度
    reset_progress(analysis_progress)
    analysis_progress['status'] = 'processing'
    analysis_progress['message'] = '正在加载数据...'
    
    # 读取数据
    try:
        global current_data
        current_data = pd.read_csv(filepath, encoding='utf-8')
        
        # 设置总进度
        analysis_progress['total'] = len(current_data) + 2  # 数据加载 + 处理 + 保存
        analysis_progress['current'] = 1
        analysis_progress['message'] = '正在分析情感...'
        
        # 根据选择的方法进行处理
        if method == 'score':
            # 使用评分转换为情感标签
            if 'score' in current_data.columns:
                current_data = analyzer.convert_score_to_sentiment(current_data)
            else:
                analysis_progress['status'] = 'error'
                analysis_progress['message'] = '数据中没有score列，无法使用评分转换'
                return jsonify({'status': 'error', 'message': '数据中没有score列，无法使用评分转换'})
        else:  # method == 'bert'
            # 使用BERT模型预测
            try:
                if not os.path.exists(analyzer.model_save_path):
                    # 如果模型不存在，先用评分数据训练一个简单模型
                    analysis_progress['message'] = '模型不存在，训练中...'
                    if 'score' in current_data.columns:
                        train_df = analyzer.convert_score_to_sentiment(current_data)
                        train_loader, test_loader = analyzer.prepare_data(train_df)
                        analyzer.train(train_loader, epochs=1)  # 实际应用中应增加轮数
                    else:
                        analysis_progress['status'] = 'error'
                        analysis_progress['message'] = '模型不存在且无法训练，请先上传带评分的数据'
                        return jsonify({'status': 'error', 'message': '模型不存在且无法训练，请先上传带评分的数据'})
                
                # 加载模型
                analysis_progress['message'] = '加载BERT模型...'
                analyzer.load_model()
                
                # 预测情感
                analysis_progress['message'] = '使用BERT分析情感...'
                
                # 创建一个进度更新的回调函数
                def progress_callback(current, total, message=''):
                    analysis_progress['current'] = current + 1  # +1 是因为数据加载已经占用1个进度
                    if message:
                        analysis_progress['message'] = message
                
                current_data = analyzer.predict_dataframe(current_data, progress_callback=progress_callback)
            except Exception as e:
                analysis_progress['status'] = 'error'
                analysis_progress['message'] = f'BERT分析失败: {str(e)}'
                return jsonify({'status': 'error', 'message': f'BERT分析失败: {str(e)}'})
        
        # 更新进度
        analysis_progress['current'] = analysis_progress['total'] - 1
        analysis_progress['message'] = '正在保存结果...'
        
        # 保存结果
        current_data.to_csv(filepath, index=False, encoding='utf-8')
        
        # 生成可视化
        try:
            generate_visualizations(current_data)
        except Exception as e:
            print(f"生成可视化时出错: {e}")
            # 即使可视化生成失败也继续返回成功
        
        # 完成进度
        analysis_progress['current'] = analysis_progress['total']
        analysis_progress['status'] = 'completed'
        analysis_progress['message'] = '分析完成！'
        
        return jsonify({'status': 'success'})
    
    except Exception as e:
        analysis_progress['status'] = 'error'
        analysis_progress['message'] = f'处理数据时出错: {str(e)}'
        return jsonify({'status': 'error', 'message': f'处理数据时出错: {str(e)}'})

@app.route('/advanced_analysis')
def advanced_analysis():
    """高级分析页面"""
    file = request.args.get('file')
    
    if not file:
        return render_template('advanced_analysis.html', file=None)
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
    
    if not os.path.exists(filepath):
        return redirect(url_for('index'))
    
    # 读取数据
    try:
        # 检查数据是否已经进行过基础情感分析
        df = pd.read_csv(filepath, encoding='utf-8')
        if 'sentiment_label' not in df.columns:
            # 如果还没有进行情感分析，先进行情感分析
            return redirect(url_for('analysis', file=file))
        
        return render_template('advanced_analysis.html', file=file, 
                              hanlp_available=HANLP_AVAILABLE, 
                              lac_available=LAC_AVAILABLE)
    except Exception as e:
        return render_template('upload.html', error=f"读取数据文件时出错: {str(e)}")

@app.route('/process_advanced', methods=['POST'])
def process_advanced():
    """处理数据并进行高级分析"""
    global advanced_analysis_progress
    
    file = request.form.get('file')
    
    if not file:
        return jsonify({'status': 'error', 'message': '未指定文件'})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
    
    if not os.path.exists(filepath):
        return jsonify({'status': 'error', 'message': '文件不存在'})
    
    # 获取分析类型和参数
    analysis_types = request.form.getlist('analysis_types')
    topic_num = int(request.form.get('topic_num', 5))
    summary_length = int(request.form.get('summary_length', 3))
    keyword_num = int(request.form.get('keyword_num', 20))
    ner_model = request.form.get('ner_model', 'default')
    
    # 重置进度
    reset_progress(advanced_analysis_progress)
    advanced_analysis_progress['status'] = 'processing'
    advanced_analysis_progress['message'] = '正在加载数据...'
    
    # 读取数据
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        
        # 计算总进度步骤数
        total_steps = 1  # 数据加载
        if 'ner' in analysis_types:
            total_steps += 2  # NER处理和可视化
        if 'dependency' in analysis_types:
            total_steps += 1  # 依存句法分析
        if 'keywords' in analysis_types:
            total_steps += 2  # 关键词提取和可视化
        if 'summary' in analysis_types:
            total_steps += 2  # 摘要生成和可视化
        if 'topic' in analysis_types:
            total_steps += 2  # 主题建模和可视化
        if 'aspect' in analysis_types:
            total_steps += 2  # 方面级情感分析和可视化
        total_steps += 1  # 保存结果
        
        advanced_analysis_progress['total'] = total_steps
        advanced_analysis_progress['current'] = 1
        
        # 执行高级分析
        # 1. 初始化各分析模块
        current_step = 1
        
        # 2. 分析结果存储
        results_dict = {'analysis_types': analysis_types}
        
        # 3. 执行选定的分析
        if 'ner' in analysis_types:
            # 命名实体识别
            advanced_analysis_progress['current'] = current_step
            advanced_analysis_progress['current_module'] = '命名实体识别'
            advanced_analysis_progress['message'] = '正在初始化NER模型...'
            
            ner = NamedEntityRecognition(model_type=ner_model)
            
            advanced_analysis_progress['message'] = f'正在处理{len(df)}条文本进行命名实体识别...'
            df = ner.process_dataframe(df, text_col='clean_content')
            
            current_step += 1
            advanced_analysis_progress['current'] = current_step
            advanced_analysis_progress['message'] = '生成实体统计和可视化...'
            
            # 生成实体摘要
            entity_summary = ner.generate_entity_summary(df)
            results_dict['ner_stats'] = entity_summary
            
            # 生成实体可视化
            ner.plot_entity_distribution(
                entity_summary,
                save_path='static/visualizations/advanced/entity_distribution.png'
            )
            
            # 生成实体示例
            ner_samples = []
            if len(df) > 0:
                sample_indices = random.sample(range(len(df)), min(3, len(df)))
                for idx in sample_indices:
                    text = df.iloc[idx]['clean_content']
                    entities = df.iloc[idx]['entities']
                    html = ner.highlight_entities_html(text, entities)
                    ner_samples.append((text, html))
            results_dict['ner_samples'] = ner_samples
            
            current_step += 1
        
        if 'dependency' in analysis_types:
            # 依存句法分析
            advanced_analysis_progress['current'] = current_step
            advanced_analysis_progress['current_module'] = '依存句法分析'
            advanced_analysis_progress['message'] = '初始化依存句法分析模型...'
            
            parser = DependencyParser()
            
            advanced_analysis_progress['message'] = '进行依存句法分析...'
            
            # 依存句法分析（只对部分示例进行分析，避免过多计算）
            dependency_samples = []
            if len(df) > 0:
                sample_indices = random.sample(range(len(df)), min(3, len(df)))
                for idx in sample_indices:
                    text = df.iloc[idx]['clean_content']
                    # 限制文本长度，避免处理过长文本
                    if len(text) > 100:
                        text = text[:100] + '...'
                    parse_result = parser.parse(text)
                    html = parser.generate_syntax_html(parse_result)
                    dependency_samples.append((text, html))
                    # 生成依存句法树可视化
                    if len(dependency_samples) == 1:  # 只为第一个示例生成图
                        parser.plot_dependency_tree(
                            parse_result,
                            save_path='static/visualizations/advanced/dependency_tree.png'
                        )
            results_dict['dependency_samples'] = dependency_samples
            
            current_step += 1
        
        if 'keywords' in analysis_types:
            # 关键词抽取
            advanced_analysis_progress['current'] = current_step
            advanced_analysis_progress['current_module'] = '关键词抽取'
            advanced_analysis_progress['message'] = '初始化关键词抽取模型...'
            
            keyword_extractor = KeywordExtractor()
            
            advanced_analysis_progress['message'] = f'从{len(df)}条文本中抽取关键词...'
            
            df = keyword_extractor.extract_from_df(df, text_col='clean_content', top_k=keyword_num)
            
            current_step += 1
            advanced_analysis_progress['current'] = current_step
            advanced_analysis_progress['message'] = '聚合关键词并生成可视化...'
            
            # 聚合关键词
            aggregated_keywords = keyword_extractor.aggregate_keywords(df)
            results_dict['keywords'] = aggregated_keywords
            
            # 生成关键词可视化
            keyword_extractor.plot_keyword_cloud(
                aggregated_keywords,
                save_path='static/visualizations/keyword_cloud.png'
            )
            keyword_extractor.plot_top_keywords(
                aggregated_keywords,
                save_path='static/visualizations/top_keywords.png'
            )
            
            # 生成关键词示例
            keyword_samples = []
            if len(df) > 0:
                sample_indices = random.sample(range(len(df)), min(3, len(df)))
                for idx in sample_indices:
                    text = df.iloc[idx]['clean_content']
                    keywords = df.iloc[idx]['keywords']
                    keyword_samples.append((text, keywords))
            results_dict['keyword_samples'] = keyword_samples
            
            current_step += 1
        
        if 'summary' in analysis_types:
            # 文本摘要
            advanced_analysis_progress['current'] = current_step
            advanced_analysis_progress['current_module'] = '文本摘要'
            advanced_analysis_progress['message'] = '初始化文本摘要模型...'
            
            summarizer = TextSummarizer()
            
            advanced_analysis_progress['message'] = '生成文本摘要...'
            
            # 文本摘要（只为较长的文本生成摘要）
            df['is_long'] = df['clean_content'].apply(lambda x: len(x) > 100 if isinstance(x, str) else False)
            long_texts_df = df[df['is_long']]
            
            if not long_texts_df.empty:
                summary_df = summarizer.summarize_from_df(
                    long_texts_df, 
                    text_col='clean_content',
                    max_length=summary_length
                )
                
                # 将摘要合并回原始DataFrame
                if 'summary' in df.columns:
                    df = df.drop(columns=['summary'])
                df = df.join(summary_df[['summary']], how='left')
                
                current_step += 1
                advanced_analysis_progress['current'] = current_step
                advanced_analysis_progress['message'] = '生成摘要统计和可视化...'
                
                # 生成摘要统计信息
                summary_stats = {
                    'count': len(long_texts_df),
                    'avg_reduction': (1 - df['summary'].str.len().mean() / df['clean_content'].str.len().mean()) * 100
                }
                results_dict['summary_stats'] = summary_stats
                
                # 生成摘要可视化
                summarizer.plot_length_reduction(
                    df,
                    text_col='clean_content',
                    summary_col='summary',
                    save_path='static/visualizations/advanced/summary_length_reduction.png'
                )
                
                # 生成摘要示例
                summary_samples = []
                sample_indices = random.sample(range(len(long_texts_df)), min(3, len(long_texts_df)))
                for idx in sample_indices:
                    original = long_texts_df.iloc[idx]['clean_content']
                    summary = summary_df.iloc[idx]['summary']
                    summary_samples.append((original, summary))
                results_dict['summary_samples'] = summary_samples
            else:
                results_dict['summary_stats'] = None
                current_step += 2  # 跳过两个步骤
            
        
        if 'topic' in analysis_types:
            # 主题建模
            advanced_analysis_progress['current'] = current_step
            advanced_analysis_progress['current_module'] = '主题建模'
            advanced_analysis_progress['message'] = '初始化主题模型...'
            
            topic_modeler = TopicModeler(n_topics=topic_num)
            
            if True or len(df) >= 20:  # 确保有足够的数据进行主题建模
                advanced_analysis_progress['message'] = f'在{len(df)}条文本中进行主题建模...'
                
                df = topic_modeler.fit_transform_from_df(df, text_col='clean_content')
                
                current_step += 1
                advanced_analysis_progress['current'] = current_step
                advanced_analysis_progress['message'] = '生成主题可视化...'
                
                topics = topic_modeler.get_topics(n_top_words=8)
                results_dict['topics'] = topics
                
                # 生成主题可视化
                topic_modeler.plot_topic_keywords(
                    topics,
                    save_path='static/visualizations/advanced/topic_keywords.png'
                )
                topic_modeler.plot_topic_distribution(
                    df,
                    save_path='static/visualizations/advanced/topic_distribution.png'
                )
                
                # 生成主题示例
                topic_samples = {}
                for topic_id in range(topic_modeler.n_topics):
                    # 获取主题概率最高的文本
                    topic_col = f'topic_{topic_id}'
                    if topic_col in df.columns:
                        top_docs = df.nlargest(3, topic_col)
                        topic_samples[topic_id] = [(row['clean_content'], row[topic_col]) 
                                                for _, row in top_docs.iterrows()]
                results_dict['topic_samples'] = topic_samples
            else:
                results_dict['topics'] = None
                advanced_analysis_progress['message'] = '数据量不足，跳过主题建模'
                current_step += 2  # 跳过两个步骤
        
        if 'aspect' in analysis_types:
            # 方面级情感分析
            advanced_analysis_progress['current'] = current_step
            advanced_analysis_progress['current_module'] = '方面级情感分析'
            advanced_analysis_progress['message'] = '初始化方面级情感分析模型...'
            
            aspect_analyzer = AspectBasedSentimentAnalyzer(dependency_parser=parser if 'dependency' in analysis_types else None)
            
            advanced_analysis_progress['message'] = f'对{len(df)}条文本进行方面级情感分析...'
            
            df = aspect_analyzer.analyze_from_df(df, text_col='clean_content')
            
            current_step += 1
            advanced_analysis_progress['current'] = current_step
            advanced_analysis_progress['message'] = '生成方面级情感分析统计和可视化...'
            
            aspect_summary = aspect_analyzer.get_aspect_summary(df)
            results_dict['aspect_summary'] = aspect_summary
            
            if aspect_summary:
                # 生成方面级情感可视化
                aspect_analyzer.plot_aspect_sentiment(
                    aspect_summary,
                    save_path='static/visualizations/advanced/aspect_sentiment.png'
                )
                aspect_analyzer.plot_aspect_distribution(
                    aspect_summary,
                    save_path='static/visualizations/advanced/aspect_distribution.png'
                )
            
            current_step += 1
        
        # 保存分析结果
        advanced_analysis_progress['current'] = total_steps - 1
        advanced_analysis_progress['current_module'] = '完成'
        advanced_analysis_progress['message'] = '保存分析结果...'
        
        # 1. 保存DataFrame
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        # 2. 保存分析结果字典
        global current_advanced_results
        current_advanced_results = results_dict
        
        # 保存结果到JSON文件，方便后续加载
        results_json_path = f"{filepath.rsplit('.', 1)[0]}_advanced_results.json"
        # 将不可序列化的对象转换为字符串
        serializable_dict = {}
        for key, value in results_dict.items():
            if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                serializable_dict[key] = value
            else:
                serializable_dict[key] = str(value)
        
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_dict, f, ensure_ascii=False, indent=2)
        
        # 完成分析
        advanced_analysis_progress['current'] = total_steps
        advanced_analysis_progress['status'] = 'completed'
        advanced_analysis_progress['message'] = '高级分析完成！'
        
        return jsonify({'status': 'success'})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        advanced_analysis_progress['status'] = 'error'
        advanced_analysis_progress['message'] = f'高级分析过程中出错: {str(e)}'
        return jsonify({'status': 'error', 'message': f'高级分析过程中出错: {str(e)}'})

@app.route('/advanced_results')
def advanced_results():
    """展示高级分析结果"""
    file = request.args.get('file')
    
    if not file:
        return redirect(url_for('index'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
    
    if not os.path.exists(filepath):
        return redirect(url_for('index'))
    
    # 读取数据和分析结果
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        
        # 尝试加载保存的分析结果
        results_json_path = f"{filepath.rsplit('.', 1)[0]}_advanced_results.json"
        if os.path.exists(results_json_path):
            with open(results_json_path, 'r', encoding='utf-8') as f:
                results_dict = json.load(f)
        else:
            # 如果没有保存的结果，使用全局变量中的结果
            global current_advanced_results
            if current_advanced_results is None:
                return redirect(url_for('advanced_analysis', file=file))
            results_dict = current_advanced_results
        
        # 添加总数据量
        total_count = len(df)
        
        # 返回结果页面
        return render_template(
            'advanced_results.html',
            file=file,
            total_count=total_count,
            **results_dict
        )
    except Exception as e:
        return render_template('upload.html', error=f"读取高级分析结果时出错: {str(e)}")

def generate_visualizations(df):
    """生成各种可视化图表"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # 情感分布饼图
    visualizer.plot_sentiment_pie(
        df, 
        save_path=f'static/visualizations/sentiment_pie_{timestamp}.png'
    )
    
    # 如果有平台信息，生成平台对比图
    if 'platform' in df.columns:
        visualizer.plot_platform_comparison(
            df,
            save_path=f'static/visualizations/platform_comparison_{timestamp}.png'
        )
    
    # 情感词云
    if 'words' in df.columns:
        visualizer.plot_wordcloud(
            df, 
            sentiment_value='正面',
            save_path=f'static/visualizations/positive_wordcloud_{timestamp}.png'
        )
        
        visualizer.plot_wordcloud(
            df, 
            sentiment_value='负面',
            save_path=f'static/visualizations/negative_wordcloud_{timestamp}.png'
        )
    
    # 情感趋势（如果有时间信息）
    if 'time' in df.columns:
        visualizer.plot_sentiment_trend(
            df,
            save_path=f'static/visualizations/sentiment_trend_{timestamp}.png'
        )

@app.route('/results')
def results():
    """展示分析结果"""
    file = request.args.get('file')
    
    if not file:
        return redirect(url_for('index'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
    
    if not os.path.exists(filepath):
        return redirect(url_for('index'))
    
    # 读取数据
    try:
        global current_data
        current_data = pd.read_csv(filepath, encoding='utf-8')
        
        # 确保数据已经进行了情感分析
        if 'sentiment_label' not in current_data.columns:
            return redirect(url_for('analysis', file=file))
        
        # 获取可视化图表
        visualizations = []
        for filename in os.listdir('static/visualizations'):
            if filename.endswith('.png'):
                visualizations.append(filename)
        
        # 数据摘要
        if 'platform' in current_data.columns:
            platform_counts = current_data['platform'].value_counts().to_dict()
        else:
            platform_counts = {}
        
        sentiment_counts = current_data['sentiment_text'].value_counts().to_dict()
        
        # 添加准确率（如果有）
        if 'accuracy' in current_data.columns:
            sentiment_counts['accuracy'] = current_data['accuracy'].iloc[0]
        
        # 示例评论
        positive_examples = []
        negative_examples = []
        neutral_examples = []
        
        if 'sentiment_text' in current_data.columns and 'content' in current_data.columns:
            for sentiment in ['正面', '负面', '中性']:
                filtered = current_data[current_data['sentiment_text'] == sentiment]
                if not filtered.empty:
                    samples = filtered.sample(min(3, len(filtered)))
                    
                    for _, row in samples.iterrows():
                        example = {
                            'content': row['content'],
                            'platform': row.get('platform', '未知'),
                            'score': row.get('score', '未知'),
                            'label': row.get('label', '未知'),
                            'probability': row.get('sentiment_prob', 0.0)
                        }
                        
                        if sentiment == '正面':
                            positive_examples.append(example)
                        elif sentiment == '负面':
                            negative_examples.append(example)
                        else:
                            neutral_examples.append(example)
        
        # 返回结果页面
        return render_template(
            'results.html', 
            file=file,
            visualizations=visualizations,
            platform_counts=platform_counts,
            sentiment_counts=sentiment_counts,
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            neutral_examples=neutral_examples,
            total_count=len(current_data)
        )
    except Exception as e:
        return render_template('upload.html', error=f"读取分析结果时出错: {str(e)}")

@app.route('/analyze_text', methods=['GET', 'POST'])
def analyze_text():
    """单条文本情感分析"""
    if request.method == 'POST':
        text = request.form.get('text')
        
        if not text:
            return render_template('analyze_text.html', error="请输入文本")
        
        try:
            # 加载模型
            if not os.path.exists(analyzer.model_save_path):
                return render_template('analyze_text.html', error="模型不存在，请先训练模型")
            
            analyzer.load_model()
            
            # 预处理文本
            clean_text = preprocessor.clean_text(text)
            words = preprocessor.segment(clean_text)
            
            # 预测情感
            result = analyzer.predict(clean_text)
            
            if not result.empty:
                sentiment = result['sentiment_text'].iloc[0]
                probability = result['probability'].iloc[0]
                
                # 返回结果
                return render_template(
                    'analyze_text.html', 
                    text=text,
                    clean_text=clean_text,
                    words=' '.join(words),
                    sentiment=sentiment,
                    probability=f"{probability:.2f}"
                )
            else:
                return render_template('analyze_text.html', error="情感分析失败")
        
        except Exception as e:
            return render_template('analyze_text.html', error=f"情感分析出错: {str(e)}")
    
    return render_template('analyze_text.html')

@app.route('/about')
def about():
    """关于页面"""
    return render_template('about.html')

@app.route('/api/data')
def api_data():
    """提供数据API"""
    file = request.args.get('file')
    
    if not file:
        return jsonify({'status': 'error', 'message': '未指定文件'})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
    
    if not os.path.exists(filepath):
        return jsonify({'status': 'error', 'message': '文件不存在'})
    
    try:
        # 读取数据
        df = pd.read_csv(filepath, encoding='utf-8')
        
        # 转换为JSON格式
        data = df.to_dict(orient='records')
        
        return jsonify({'status': 'success', 'data': data})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'读取数据出错: {str(e)}'})

@app.route('/favicon.ico')
def favicon():
    """提供favicon"""
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    # 首次运行时创建必要的目录
    for dir_path in ['data', 'models', 'uploads', 'static/visualizations', 'static/fonts']:
        os.makedirs(dir_path, exist_ok=True)
    
    # 运行Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000)
