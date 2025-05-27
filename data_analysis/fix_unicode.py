#!/usr/bin/env python3
"""
Fix Unicode emoji characters in Python files for Windows compatibility
"""

import os
import re

def fix_unicode_in_file(filepath):
    """Replace Unicode emoji characters with ASCII equivalents"""
      # Mapping of Unicode characters to ASCII equivalents
    unicode_map = {
        '🚀': '>>',
        '✅': '✓',
        '📊': '[CHART]',
        '📝': '[REPORT]', 
        '🎉': '[SUCCESS]',
        '📁': '[FOLDER]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '🔄': '[PROCESSING]',
        '🌐': '[WEB]',
        '🖼️': '[IMAGE]',
        '💡': '[TIP]',
        '📖': '[READ]',
        '📄': '[FILE]',
        '⏰': '[TIME]',
        '📋': '[LIST]',
        '🛠️': '[TOOLS]',
        '🧠': '[BRAIN]',
        '📂': '[FOLDER]',
        '\u2713': 'OK',  # ✓ checkmark
        '📈': '[STATS]',
        '📉': '[TREND]',
        '📌': '[PIN]',
        '🔍': '[SEARCH]',
        '📐': '[MEASURE]',
        '🎯': '[TARGET]',
        '🔗': '[LINK]',
        '📊': '[CHART]'
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace Unicode characters
        for unicode_char, ascii_replacement in unicode_map.items():
            content = content.replace(unicode_char, ascii_replacement)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Fixed Unicode characters in {filepath}")
        return True
        
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Fix Unicode characters in Python files"""
    files_to_fix = [
        'analyze_single_environment.py',
        'process_data.py',
        'run_analysis.py',
        'generate_dashboard.py'
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for filename in files_to_fix:
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            fix_unicode_in_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    main()
