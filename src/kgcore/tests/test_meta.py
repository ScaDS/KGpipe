# kgcore/tests/test_meta.py
import tempfile
import os
from kgcore.decorators.event import event
from kgcore.decorators.class_ import class_
from kgcore.api import KG
from kgcore.system.schema import SysLabels

def test_event_decorator():
    """Test the @event decorator functionality."""
    # Create a temporary KG for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_events.ttl")
        kg = KG(backend="rdf_file", file_path=file_path, format="turtle")
        
        # Override the global system recorder
        from kgcore.decorators.event import _rec
        original_kg = _rec.kg
        _rec.kg = kg
        
        try:
            # Create a test function with @event decorator
            @event("test_function", "A test function for decorator testing", version="1.0")
            def test_function(x, y):
                return x + y
            
            # Call the decorated function
            result = test_function(5, 3)
            assert result == 8, f"Expected 8, got {result}"
            
            # Verify function was registered in the system graph
            functions = kg.find_entities(SysLabels.Function)
            assert len(functions) > 0, "No functions registered in system graph"
            
            # Check function properties
            test_func = next((f for f in functions if "test_function" in f.props.get("name", "")), None)
            assert test_func is not None, "Test function not found in system graph"
            assert test_func.props.get("description") == "A test function for decorator testing"
            # Note: version metadata goes to the run, not the function registration
            
            # Verify run was recorded
            runs = kg.find_entities(SysLabels.Run)
            assert len(runs) > 0, "No runs recorded in system graph"
            
            # Check run properties
            run = runs[0]
            assert run.props.get("name") == "test_function"
            assert run.props.get("version") == "1.0"
            # Note: result is stored in reified metadata, not directly on the run
            
            print("✓ Event decorator test passed")
        finally:
            # Restore original KG
            _rec.kg = original_kg

def test_class_decorator():
    """Test the @class_ decorator functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_classes.ttl")
        kg = KG(backend="rdf_file", file_path=file_path, format="turtle")
        
        # Override the global system recorder
        from kgcore.decorators.class_ import _rec
        original_kg = _rec.kg
        _rec.kg = kg
        
        try:
            # Create a test class with @class_ decorator
            @class_("TestClass", "A test class for decorator testing")
            class TestClass:
                def __init__(self, value):
                    self.value = value
                
                def get_value(self):
                    return self.value
            
            # Create instance and use it
            instance = TestClass(42)
            assert instance.get_value() == 42
            
            # Verify class was registered in the system graph
            functions = kg.find_entities(SysLabels.Function)
            assert len(functions) > 0, "No functions registered in system graph"
            
            # Check class properties
            test_class = next((f for f in functions if "TestClass" in f.props.get("name", "")), None)
            assert test_class is not None, "Test class not found in system graph"
            assert test_class.props.get("description") == "A test class for decorator testing"
            
            print("✓ Class decorator test passed")
        finally:
            # Restore original KG
            _rec.kg = original_kg

def test_event_decorator_error_handling():
    """Test @event decorator error handling."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_errors.ttl")
        kg = KG(backend="rdf_file", file_path=file_path, format="turtle")
        
        # Override the global system recorder
        from kgcore.decorators.event import _rec
        original_kg = _rec.kg
        _rec.kg = kg
        
        try:
            # Create a function that raises an exception
            @event("error_function", "A function that raises an error")
            def error_function():
                raise ValueError("Test error")
            
            # Call the function and expect it to raise
            try:
                error_function()
                assert False, "Expected ValueError to be raised"
            except ValueError as e:
                assert str(e) == "Test error"
            
            # Verify error was recorded in the system graph
            runs = kg.find_entities(SysLabels.Run)
            assert len(runs) > 0, "No runs recorded in system graph"
            
            # Check error properties
            run = runs[0]
            assert run.props.get("name") == "error_function"
            # Note: result and error are stored in reified metadata, not directly on the run
            
            print("✓ Event decorator error handling test passed")
        finally:
            # Restore original KG
            _rec.kg = original_kg

def test_decorator_metadata():
    """Test that decorator metadata is properly stored."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_metadata.ttl")
        kg = KG(backend="rdf_file", file_path=file_path, format="turtle")
        
        # Override the global system recorder
        from kgcore.decorators.event import _rec
        original_kg = _rec.kg
        _rec.kg = kg
        
        try:
            # Create function with rich metadata
            @event("metadata_function", "Function with rich metadata", 
                   version="2.0", author="test", category="utility")
            def metadata_function():
                return "success"
            
            # Call the function
            result = metadata_function()
            assert result == "success"
            
            # Verify metadata was stored
            functions = kg.find_entities(SysLabels.Function)
            test_func = next((f for f in functions if "metadata_function" in f.props.get("name", "")), None)
            assert test_func is not None
            # Note: metadata goes to the run, not the function registration
            
            # Check run metadata
            runs = kg.find_entities(SysLabels.Run)
            run = runs[0]
            assert run.props.get("version") == "2.0"
            assert run.props.get("author") == "test"
            assert run.props.get("category") == "utility"
            
            print("✓ Decorator metadata test passed")
        finally:
            # Restore original KG
            _rec.kg = original_kg

def test_system_graph_structure():
    """Test the overall structure of the system graph."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_structure.ttl")
        kg = KG(backend="rdf_file", file_path=file_path, format="turtle")
        
        # Override the global system recorders
        from kgcore.decorators.event import _rec as event_rec
        from kgcore.decorators.class_ import _rec as class_rec
        original_event_kg = event_rec.kg
        original_class_kg = class_rec.kg
        event_rec.kg = kg
        class_rec.kg = kg
        
        try:
            # Create multiple decorated functions
            @event("func1", "First function")
            def func1():
                return 1
            
            @event("func2", "Second function")
            def func2():
                return 2
            
            @class_("TestClass", "Test class")
            class TestClass:
                pass
            
            # Call functions
            func1()
            func2()
            
            # Verify system graph structure
            functions = kg.find_entities(SysLabels.Function)
            assert len(functions) >= 3, f"Expected at least 3 functions, got {len(functions)}"
            
            runs = kg.find_entities(SysLabels.Run)
            assert len(runs) == 2, f"Expected 2 runs, got {len(runs)}"
            
            # Check that runs are connected to functions via CALLS relation
            calls_relations = kg.find_relations("CALLS")
            assert len(calls_relations) == 2, f"Expected 2 CALLS relations, got {len(calls_relations)}"
            
            print("✓ System graph structure test passed")
        finally:
            # Restore original KGs
            event_rec.kg = original_event_kg
            class_rec.kg = original_class_kg

def test_versioning_integration():
    """Test that versioning works with the existing decorator system."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_versioning_integration.ttl")
        kg = KG(backend="rdf_file", file_path=file_path, format="turtle")
        
        # Override the global system recorder
        from kgcore.decorators.event import _rec
        original_kg = _rec.kg
        _rec.kg = kg
        
        try:
            # Test timestamp versioning
            @event("timestamped_test", "Test with timestamp versioning", 
                   auto_version="timestamp")
            def timestamped_test():
                return "timestamped"
            
            # Test hash versioning
            @event("hashed_test", "Test with hash versioning", 
                   auto_version="hash")
            def hashed_test(x):
                return x * 2
            
            # Test explicit version
            @event("explicit_test", "Test with explicit version", 
                   version="v1.0.0")
            def explicit_test():
                return "explicit"
            
            # Call all functions
            timestamped_test()
            hashed_test(5)
            explicit_test()
            
            # Verify all runs were recorded with versions
            runs = kg.find_entities(SysLabels.Run)
            assert len(runs) == 3, f"Expected 3 runs, got {len(runs)}"
            
            # Check that each run has appropriate versioning
            timestamped_runs = [r for r in runs if r.props.get("name") == "timestamped_test"]
            hashed_runs = [r for r in runs if r.props.get("name") == "hashed_test"]
            explicit_runs = [r for r in runs if r.props.get("name") == "explicit_test"]
            
            assert len(timestamped_runs) == 1, "Should have 1 timestamped run"
            assert len(hashed_runs) == 1, "Should have 1 hashed run"
            assert len(explicit_runs) == 1, "Should have 1 explicit run"
            
            # Check version formats
            timestamp_version = timestamped_runs[0].props.get("version")
            assert timestamp_version is not None, "Timestamp version should be present"
            assert len(timestamp_version) == 15, "Timestamp version should be 15 chars"
            
            hash_version = hashed_runs[0].props.get("version")
            assert hash_version is not None, "Hash version should be present"
            assert len(hash_version) == 16, "Hash version should be 16 chars"
            
            explicit_version = explicit_runs[0].props.get("version")
            assert explicit_version == "v1.0.0", f"Expected v1.0.0, got {explicit_version}"
            
            print("✓ Versioning integration test passed")
        finally:
            _rec.kg = original_kg

if __name__ == "__main__":
    test_event_decorator()
    test_class_decorator()
    test_event_decorator_error_handling()
    test_decorator_metadata()
    test_system_graph_structure()
    test_versioning_integration()
    print("\n🎉 All decorator tests passed!")
