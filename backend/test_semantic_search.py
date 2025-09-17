#!/usr/bin/env python3
"""Test script for semantic search functionality."""

import asyncio
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
os.sys.path.insert(0, str(project_root))

from app.services.embedding import embedding_service


async def test_embedding_service():
    """Test the embedding service."""
    print("🧪 Testing Embedding Service")
    print("=" * 50)
    
    # Test single text encoding
    test_text = "machine learning neural networks deep learning"
    print(f"Input text: {test_text}")
    
    try:
        embedding = await embedding_service.encode_async(test_text)
        print(f"✅ Generated embedding with {len(embedding)} dimensions")
        print(f"First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return False
    
    # Test batch encoding
    test_texts = [
        "natural language processing",
        "computer vision image recognition",
        "reinforcement learning algorithms"
    ]
    
    try:
        embeddings = await embedding_service.encode_batch_async(test_texts)
        print(f"✅ Generated {len(embeddings)} embeddings for batch")
        for i, emb in enumerate(embeddings):
            print(f"  Text {i+1}: {len(emb)} dimensions")
    except Exception as e:
        print(f"❌ Error generating batch embeddings: {e}")
        return False
    
    # Test paper embedding creation
    try:
        paper_embedding = await embedding_service.create_paper_embedding_async(
            title="Attention Is All You Need",
            abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
            full_text="This paper introduces the Transformer architecture..."
        )
        print(f"✅ Generated paper embedding with {len(paper_embedding)} dimensions")
    except Exception as e:
        print(f"❌ Error generating paper embedding: {e}")
        return False
    
    return True


async def test_tools_import():
    """Test that the search tools can be imported."""
    print("\n🔧 Testing Tools Import")
    print("=" * 50)
    
    try:
        from app.tools.search import search_papers, semantic_search_papers, hybrid_search_papers
        print("✅ Successfully imported search_papers")
        print("✅ Successfully imported semantic_search_papers")
        print("✅ Successfully imported hybrid_search_papers")
        
        # Check tool metadata
        print(f"\nTool descriptions:")
        print(f"- search_papers: {search_papers.description}")
        print(f"- semantic_search_papers: {semantic_search_papers.description}")
        print(f"- hybrid_search_papers: {hybrid_search_papers.description}")
        
        return True
    except Exception as e:
        print(f"❌ Error importing tools: {e}")
        return False


def test_model_import():
    """Test that the updated model can be imported."""
    print("\n🗃️  Testing Model Import")
    print("=" * 50)
    
    try:
        from app.db.models import Paper
        print("✅ Successfully imported Paper model")
        
        # Check if the model has the new methods
        if hasattr(Paper, 'semantic_search'):
            print("✅ Paper.semantic_search method exists")
        else:
            print("❌ Paper.semantic_search method not found")
            
        if hasattr(Paper, 'hybrid_search'):
            print("✅ Paper.hybrid_search method exists")
        else:
            print("❌ Paper.hybrid_search method not found")
            
        return True
    except Exception as e:
        print(f"❌ Error importing model: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Testing Semantic Search Implementation")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test embedding service
    if await test_embedding_service():
        tests_passed += 1
    
    # Test tools import
    if await test_tools_import():
        tests_passed += 1
    
    # Test model import
    if test_model_import():
        tests_passed += 1
    
    print(f"\n📊 Test Results")
    print("=" * 50)
    print(f"Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Your semantic search setup is ready.")
        print("\n🎯 Next Steps:")
        print("1. Start your PostgreSQL database")
        print("2. Run the migration: uv run alembic upgrade head")
        print("3. Populate some papers with embeddings")
        print("4. Start the LangGraph server: uv run langgraph dev")
        print("5. Test with the frontend!")
    else:
        print("❌ Some tests failed. Please fix the issues above.")


if __name__ == "__main__":
    asyncio.run(main())
