# kgcore/tests/test_versioning.py
import tempfile
import os
import time
from kgcore.decorators.event import event, _compute_function_hash, _get_timestamp_version
from kgcore.api import KG
from kgcore.system.schema import SysLabels

def test_timestamp_versioning():
    """Test timestamp-based automatic versioning."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_timestamp.ttl")
        kg = KG(backend="rdf_file", file_path=file_path, format="turtle")
        
        # Override the global system recorder
        from kgcore.decorators.event import _rec
        original_kg = _rec.kg
        _rec.kg = kg
        
        try:
            @event("timestamped_function", "A function with timestamp versioning", 
                   auto_version="timestamp")
            def timestamped_function():
                return "success"
            
            # Call the function
            result = timestamped_function()
            assert result == "success"
            
            # Verify version was set
            runs = kg.find_entities(SysLabels.Run)
            assert len(runs) > 0, "No runs recorded"
            
            run = runs[0]
            version = run.props.get("version")
            assert version is not None, "No version found in run metadata"
            
            # Check that version looks like a timestamp (YYYYMMDD_HHMMSS)
            assert len(version) == 15, f"Version length should be 15, got {len(version)}"
            assert version[8] == "_", "Version should have underscore separator"
            assert version[:8].isdigit(), "Date part should be digits"
            assert version[9:].isdigit(), "Time part should be digits"
            
            print("✓ Timestamp versioning test passed")
        finally:
            _rec.kg = original_kg

def test_hash_versioning():
    """Test hash-based automatic versioning."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_hash.ttl")
        kg = KG(backend="rdf_file", file_path=file_path, format="turtle")
        
        # Override the global system recorder
        from kgcore.decorators.event import _rec
        original_kg = _rec.kg
        _rec.kg = kg
        
        try:
            @event("hashed_function", "A function with hash versioning", 
                   auto_version="hash")
            def hashed_function(x, y=10):
                return x + y
            
            # Call the function
            result = hashed_function(5)
            assert result == 15
            
            # Verify version was set
            runs = kg.find_entities(SysLabels.Run)
            assert len(runs) > 0, "No runs recorded"
            
            run = runs[0]
            version = run.props.get("version")
            assert version is not None, "No version found in run metadata"
            
            # Check that version looks like a hash (16 hex characters)
            assert len(version) == 16, f"Version length should be 16, got {len(version)}"
            assert all(c in "0123456789abcdef" for c in version), "Version should be hex"
            
            print("✓ Hash versioning test passed")
        finally:
            _rec.kg = original_kg

def test_explicit_version():
    """Test explicit version overrides automatic versioning."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_explicit.ttl")
        kg = KG(backend="rdf_file", file_path=file_path, format="turtle")
        
        # Override the global system recorder
        from kgcore.decorators.event import _rec
        original_kg = _rec.kg
        _rec.kg = kg
        
        try:
            @event("explicit_function", "A function with explicit version", 
                   version="v2.1.0", auto_version="hash")
            def explicit_function():
                return "success"
            
            # Call the function
            result = explicit_function()
            assert result == "success"
            
            # Verify explicit version was used
            runs = kg.find_entities(SysLabels.Run)
            assert len(runs) > 0, "No runs recorded"
            
            run = runs[0]
            version = run.props.get("version")
            assert version == "v2.1.0", f"Expected v2.1.0, got {version}"
            
            print("✓ Explicit version test passed")
        finally:
            _rec.kg = original_kg

def test_hash_consistency():
    """Test that function hash is consistent for unchanged functions."""
    # Test the hash function directly
    def test_func(x, y=5):
        return x + y
    
    hash1 = _compute_function_hash(test_func)
    hash2 = _compute_function_hash(test_func)
    
    assert hash1 == hash2, "Hash should be consistent for the same function"
    assert len(hash1) == 16, "Hash should be 16 characters"
    
    # Test with a different function
    def different_func(x, y=10):
        return x * y
    
    hash3 = _compute_function_hash(different_func)
    assert hash1 != hash3, "Different functions should have different hashes"
    
    print("✓ Hash consistency test passed")

def test_hash_change_detection():
    """Test that function hash changes when function is modified."""
    # This test simulates function modification by creating similar but different functions
    def original_func(x):
        return x + 1
    
    def modified_func(x):
        return x + 2  # Changed the constant
    
    hash_original = _compute_function_hash(original_func)
    hash_modified = _compute_function_hash(modified_func)
    
    assert hash_original != hash_modified, "Modified function should have different hash"
    
    print("✓ Hash change detection test passed")

def test_timestamp_uniqueness():
    """Test that timestamps are unique (with small delay)."""
    timestamp1 = _get_timestamp_version()
    time.sleep(1.1)  # Wait for at least 1 second to ensure different timestamps
    timestamp2 = _get_timestamp_version()
    
    assert timestamp1 != timestamp2, f"Timestamps should be unique: {timestamp1} == {timestamp2}"
    assert len(timestamp1) == 15, "Timestamp should be 15 characters"
    assert len(timestamp2) == 15, "Timestamp should be 15 characters"
    
    print("✓ Timestamp uniqueness test passed")

def test_version_metadata_combination():
    """Test that versioning works with other metadata."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_combined.ttl")
        kg = KG(backend="rdf_file", file_path=file_path, format="turtle")
        
        # Override the global system recorder
        from kgcore.decorators.event import _rec
        original_kg = _rec.kg
        _rec.kg = kg
        
        try:
            @event("combined_function", "A function with version and other metadata", 
                   auto_version="timestamp", author="test", category="demo")
            def combined_function():
                return "success"
            
            # Call the function
            result = combined_function()
            assert result == "success"
            
            # Verify all metadata is present
            runs = kg.find_entities(SysLabels.Run)
            assert len(runs) > 0, "No runs recorded"
            
            run = runs[0]
            assert run.props.get("version") is not None, "Version should be present"
            assert run.props.get("author") == "test", "Author should be present"
            assert run.props.get("category") == "demo", "Category should be present"
            
            print("✓ Version metadata combination test passed")
        finally:
            _rec.kg = original_kg

def test_no_auto_version():
    """Test that no automatic versioning works when not specified."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_no_version.ttl")
        kg = KG(backend="rdf_file", file_path=file_path, format="turtle")
        
        # Override the global system recorder
        from kgcore.decorators.event import _rec
        original_kg = _rec.kg
        _rec.kg = kg
        
        try:
            @event("no_version_function", "A function without versioning")
            def no_version_function():
                return "success"
            
            # Call the function
            result = no_version_function()
            assert result == "success"
            
            # Verify no version was set
            runs = kg.find_entities(SysLabels.Run)
            assert len(runs) > 0, "No runs recorded"
            
            run = runs[0]
            version = run.props.get("version")
            assert version is None, f"Version should be None, got {version}"
            
            print("✓ No auto version test passed")
        finally:
            _rec.kg = original_kg

if __name__ == "__main__":
    test_timestamp_versioning()
    test_hash_versioning()
    test_explicit_version()
    test_hash_consistency()
    test_hash_change_detection()
    test_timestamp_uniqueness()
    test_version_metadata_combination()
    test_no_auto_version()
    print("\n🎉 All versioning tests passed!")
