#!/usr/bin/env python3
"""
Validation script to check all improvements
"""

import os
import sys
from pathlib import Path

def check_file(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_no_hardcoded_paths():
    """Check that hardcoded paths have been removed"""
    print("\n🔍 Checking for hardcoded paths...")
    
    files_to_check = [
        "app.py",
        "ingest.py",
        "ingest_site.py",
        "image_processor.py",
        "check_index.py"
    ]
    
    issues = []
    
    for filename in files_to_check:
        if not os.path.exists(filename):
            continue
            
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for hardcoded Windows paths
            if 'D:\\' in content or 'C:\\' in content:
                issues.append(f"{filename}: Contains hardcoded Windows path")
            
            # Check for keys.env
            if 'keys.env' in content and '.env' not in content.replace('keys.env', ''):
                issues.append(f"{filename}: Still uses 'keys.env' instead of '.env'")
    
    if issues:
        for issue in issues:
            print(f"  ❌ {issue}")
        return False
    else:
        print("  ✅ No hardcoded paths found")
        return True

def check_environment_variables():
    """Check that environment variables are used"""
    print("\n🔍 Checking environment variable usage...")
    
    required_patterns = {
        "app.py": ["os.getenv", "GEMINI_API_KEY"],
        "ingest.py": ["os.getenv", "GEMINI_API_KEY"],
        "ingest_site.py": ["os.getenv", "CHROME_DRIVER_PATH"],
    }
    
    all_ok = True
    
    for filename, patterns in required_patterns.items():
        if not os.path.exists(filename):
            continue
            
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
            for pattern in patterns:
                if pattern in content:
                    print(f"  ✅ {filename} uses {pattern}")
                else:
                    print(f"  ❌ {filename} missing {pattern}")
                    all_ok = False
    
    return all_ok

def check_error_handling():
    """Check for improved error handling"""
    print("\n🔍 Checking error handling...")
    
    files_to_check = [
        "app.py",
        "ingest.py",
        "ingest_site.py",
    ]
    
    all_ok = True
    
    for filename in files_to_check:
        if not os.path.exists(filename):
            continue
            
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Check for try-except blocks
            try_count = content.count('try:')
            except_count = content.count('except')
            
            if try_count > 0 and except_count > 0:
                print(f"  ✅ {filename} has {try_count} try-except blocks")
            else:
                print(f"  ⚠️  {filename} has limited error handling")
                all_ok = False
    
    return all_ok

def check_unused_code():
    """Check that unused code has been removed"""
    print("\n🔍 Checking for unused code...")
    
    if not os.path.exists("app.py"):
        return False
    
    with open("app.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for unused prompt templates
    if 'PROMPT_TEMPLATE_OLD' in content or 'PROMPT_TEMPLATE_OTIMIZED' in content:
        print("  ❌ Unused prompt templates still present")
        return False
    else:
        print("  ✅ Unused code removed")
        return True

def main():
    """Main validation function"""
    print("=" * 70)
    print("  🔍 Validating GeneXus AI Assistant Improvements")
    print("=" * 70)
    
    results = []
    
    # Check essential files
    print("\n📁 Checking essential files...")
    results.append(check_file("requirements.txt", "Dependencies file"))
    results.append(check_file(".env.example", "Environment template"))
    results.append(check_file(".gitignore", "Git ignore file"))
    results.append(check_file("README_IMPROVED.md", "Improved README"))
    results.append(check_file("setup.py", "Setup utility"))
    
    # Check directories
    print("\n📂 Checking directories...")
    results.append(check_file("docs/", "Documents directory"))
    results.append(check_file("processed_text/", "Processed text directory"))
    
    # Check for improvements
    results.append(check_no_hardcoded_paths())
    results.append(check_environment_variables())
    results.append(check_error_handling())
    results.append(check_unused_code())
    
    # Summary
    print("\n" + "=" * 70)
    print("  📊 Validation Summary")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    percentage = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n  ✅ Passed: {passed}/{total} ({percentage:.1f}%)")
    
    if passed == total:
        print("\n  🎉 All validations passed! Improvements successfully implemented.")
    elif percentage >= 80:
        print("\n  ⚠️  Most validations passed. Minor issues remain.")
    else:
        print("\n  ❌ Several validations failed. Please review the issues above.")
    
    print("\n" + "=" * 70 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
