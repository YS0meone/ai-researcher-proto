"""
简化的数据加载脚本 - 单进程版本（适用于Windows）
"""
import json
from tqdm import tqdm
from app.core.config import settings
from app.services.elasticsearch import ElasticsearchService
from app.db.schema import ArxivPaper

def main():
    print("="*60)
    print("🔬 Simple Paper Loader (Single Process)")
    print("="*60)
    
    # 初始化Elasticsearch服务
    es_service = ElasticsearchService(settings.elasticsearch_config)
    
    # CS类别
    cs_categories = [
        'cs.AI', 'cs.AR', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.CL', 'cs.CR', 'cs.CV',
        'cs.CY', 'cs.DB', 'cs.DC', 'cs.DL', 'cs.DM', 'cs.DS', 'cs.ET', 'cs.FL',
        'cs.GL', 'cs.GR', 'cs.GT', 'cs.HC', 'cs.IR', 'cs.IT', 'cs.LG', 'cs.LO',
        'cs.MA', 'cs.MM', 'cs.MS', 'cs.NA', 'cs.NE', 'cs.NI', 'cs.OH', 'cs.OS',
        'cs.PF', 'cs.PL', 'cs.RO', 'cs.SC', 'cs.SD', 'cs.SE', 'cs.SI', 'cs.SY'
    ]
    cs_set = set(cs_categories)
    
    # 打开文件
    data_path = settings.paper_loader_config.arxiv_metadata_path
    print(f"\n📁 Reading from: {data_path}")
    print(f"🏷️  Looking for CS categories: {len(cs_categories)} types\n")
    
    processed = 0
    added = 0
    limit = 50000  # 加载50000篇论文，预计能得到~2000篇CS论文
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=limit, desc="Processing papers"):
            if processed >= limit:
                break
            
            try:
                data = json.loads(line)
                processed += 1
                
                # 检查是否是CS类别
                categories = data.get('categories', '')
                if not categories:
                    continue
                
                cat_list = categories.split()
                if not any(cat in cs_set for cat in cat_list):
                    continue
                
                # 创建ArxivPaper对象
                paper = ArxivPaper(
                    id=data['id'],
                    title=data.get('title', ''),
                    abstract=data.get('abstract', ''),
                    authors=data.get('authors'),
                    submitter=data.get('submitter'),
                    comments=data.get('comments'),
                    journal_ref=data.get('journal-ref'),
                    doi=data.get('doi'),
                    report_no=data.get('report-no'),
                    categories=categories,
                    license=data.get('license')
                )
                
                # 添加到Elasticsearch
                if es_service.add_paper(paper):
                    added += 1
                
            except Exception as e:
                print(f"\n❌ Error processing line: {e}")
                continue
    
    print(f"\n{'='*60}")
    print(f"📈 Final Statistics")
    print(f"{'='*60}")
    print(f"✅ Processed: {processed:,} lines")
    print(f"✅ Added: {added:,} CS papers")
    print(f"📊 Match rate: {(added/processed*100):.1f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
