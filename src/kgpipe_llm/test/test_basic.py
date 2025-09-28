# """
# Basic tests for the refactored kgpipe_llm module.
# """

# import sys
# import os

# # Add the src directory to the path for imports
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# def test_imports():
#     """Test that all modules can be imported successfully."""
#     try:
#         # Test importing the main modules
#         import kgpipe_llm.core
#         import kgpipe_llm.models
#         import kgpipe_llm.config
#         import kgpipe_llm.tasks
#         import kgpipe_llm.rdf_tasks
#         import kgpipe_llm.examples
#         import kgpipe_llm.llm_tasks
        
#         print("✓ All modules can be imported")
#         return True
#     except ImportError as e:
#         print(f"✗ Import error: {e}")
#         return False

# def test_models():
#     """Test that Pydantic models can be instantiated."""
#     try:
#         from kgpipe_llm.models import Triplet, Triplets
        
#         # Test basic triplet creation
#         triplet = Triplet(
#             subject="Inception",
#             predicate="director",
#             object="Christopher Nolan"
#         )
#         assert triplet.subject == "Inception"
#         assert triplet.predicate == "director"
#         assert triplet.object == "Christopher Nolan"
        
#         # Test triplets container
#         triplets = Triplets(triplets=[triplet])
#         assert len(triplets.triplets) == 1
        
#         print("✓ Pydantic models work correctly")
#         return True
#     except Exception as e:
#         print(f"✗ Model test failed: {e}")
#         return False

# def test_config():
#     """Test configuration management."""
#     try:
#         from kgpipe_llm.config import config_manager, COMMON_NAMESPACES
        
#         # Test config manager
#         assert config_manager is not None
#         assert hasattr(config_manager, 'llm_config')
#         assert hasattr(config_manager, 'task_config')
        
#         # Test namespace prefixes
#         assert 'rdf' in COMMON_NAMESPACES
#         assert 'schema' in COMMON_NAMESPACES
        
#         print("✓ Configuration management works")
#         return True
#     except Exception as e:
#         print(f"✗ Config test failed: {e}")
#         return False

# def test_backward_compatibility():
#     """Test backward compatibility with original interface."""
#     try:
#         from kgpipe_llm.llm_tasks import (
#             JSON, ONTOLOGY
#         )
        
#         # Test that legacy constants exist
#         assert JSON is not None
#         assert ONTOLOGY is not None
        
#         print("✓ Backward compatibility maintained")
#         return True
#     except Exception as e:
#         print(f"✗ Backward compatibility test failed: {e}")
#         return False

# def test_module_structure():
#     """Test that the module structure is correct."""
#     try:
#         # Test that __init__.py exports the right things
#         from kgpipe_llm import (
#             Triplet, Triplets,
#             TransformationMapping, TransformationMappings
#         )
        
#         # Test that we can access the modules
#         import kgpipe_llm
#         assert hasattr(kgpipe_llm, 'core')
#         assert hasattr(kgpipe_llm, 'models')
#         assert hasattr(kgpipe_llm, 'tasks')
#         assert hasattr(kgpipe_llm, 'rdf_tasks')
#         assert hasattr(kgpipe_llm, 'config')
        
#         print("✓ Module structure is correct")
#         return True
#     except Exception as e:
#         print(f"✗ Module structure test failed: {e}")
#         return False

# def run_all_tests():
#     """Run all basic tests."""
#     print("Running basic tests for kgpipe_llm module...")
#     print("=" * 50)
    
#     tests = [
#         test_imports,
#         test_models,
#         test_config,
#         test_backward_compatibility,
#         test_module_structure
#     ]
    
#     passed = 0
#     total = len(tests)
    
#     for test in tests:
#         if test():
#             passed += 1
#         print()
    
#     print("=" * 50)
#     print(f"Tests passed: {passed}/{total}")
    
#     if passed == total:
#         print("✓ All tests passed!")
#         return True
#     else:
#         print("✗ Some tests failed!")
#         return False

# if __name__ == "__main__":
#     success = run_all_tests()
#     sys.exit(0 if success else 1) 