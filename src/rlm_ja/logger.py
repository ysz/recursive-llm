import logging
import sys
import os
from typing import Optional
from pathlib import Path
from datetime import datetime

def get_log_file_path() -> str:
    """Generate log file path based on main script name and timestamp."""
    # Get the main script name
    try:
        main_script = sys.modules['__main__'].__file__
        if main_script:
            script_name = Path(main_script).stem
        else:
            script_name = "interactive"
    except (AttributeError, KeyError):
        script_name = "unknown"

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory if not exists
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    return str(logs_dir / f"{script_name}_{timestamp}.log")

def get_logger(
    name: str = "rlm_ja",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # If logger already has handlers, return it
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = get_log_file_path()
        
    file_path = Path(log_file)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
        
    return logger
