import os
import tokenize
import io
import re

def remove_python_comments(source):
    """
    Removes comments and docstrings from Python source code.
    """
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    
    try:
        tokens = tokenize.generate_tokens(io_obj.readline)
        for toktype, ttext, (slineno, scol), (elineno, ecol), ltext in tokens:
            if slineno > last_lineno:
                last_col = 0
            if scol > last_col:
                out += " " * (scol - last_col)
            
            if toktype == tokenize.COMMENT:
                pass
            elif toktype == tokenize.STRING:
                if prev_toktype == tokenize.INDENT or prev_toktype == tokenize.NEWLINE or prev_toktype == tokenize.NL:
                    # Likely a docstring
                    pass
                else:
                    out += ttext
            else:
                out += ttext
            
            prev_toktype = toktype
            last_lineno = elineno
            last_col = ecol
        return out
    except Exception:
        # Fallback to simple regex if tokenize fails
        return re.sub(r'#.*', '', source)

def cleanup_yaml(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if '#' in line:
            # Keep if it contains "Options:"
            if 'Options:' in line:
                new_lines.append(line)
            else:
                # Remove the comment part
                part = line.split('#')[0].rstrip()
                if part:
                    new_lines.append(part + '\n')
        else:
            new_lines.append(line)
            
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def main():
    # 1. Process Python files
    py_files = [
        r"scripts/train.py",
        r"scripts/setup_autodl.py",
        r"scripts/inference_notebook.py",
        r"src/utils/check_anno.py",
        r"src/utils/check_labs.py",
        r"src/training/losses.py",
        r"src/training/trainer.py",
        r"src/utils/check_tracking.py",
        r"src/inference/tta.py",
        r"src/models/fusion_model.py",
        r"src/postprocessing/notebook_logic.py",
        r"src/postprocessing/optimization.py",
        r"src/models/components/lca.py",
        r"src/models/encoders/topology.py",
        r"src/models/encoders/temporal.py",
        r"src/models/encoders/spatial.py"
    ]
    
    for py_file in py_files:
        abs_path = os.path.abspath(py_file)
        if os.path.exists(abs_path):
            print(f"Processing {py_file}...")
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            clean_content = remove_python_comments(content)
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(clean_content)
                
    # 2. Process YAML
    yaml_path = os.path.abspath("configs/base_config.yaml")
    if os.path.exists(yaml_path):
        print("Processing base_config.yaml...")
        cleanup_yaml(yaml_path)

if __name__ == "__main__":
    main()
