import logging
import os
from datetime import datetime
from typing import Optional

def setup_logging(log_level: str = 'INFO', 
                  log_dir: str = 'logs',
                  log_to_file: bool = True,
                  log_to_console: bool = True,
                  include_timestamp: bool = True) -> logging.Logger:
    """
    Setup centralized logging configuration with timestamped log files.
    
    Parameters:
    -----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_dir : str
        Directory to store log files
    log_to_file : bool
        Whether to log to file
    log_to_console : bool
        Whether to log to console
    include_timestamp : bool
        Whether to include timestamp in log filename
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if requested
    if log_to_file:
        if include_timestamp:
            # Create timestamped filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'ring_partition_{timestamp}.log')
        else:
            log_file = os.path.join(log_dir, 'ring_partition.log')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Log the log file location
        print(f"📝 Logging to: {log_file}")
    
    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def setup_component_logging(component_name: str,
                           log_level: str = 'INFO',
                           log_dir: str = 'logs',
                           include_timestamp: bool = True) -> logging.Logger:
    """
    Setup logging for a specific component with its own log file.
    
    Parameters:
    -----------
    component_name : str
        Name of the component (e.g., 'mesh', 'optimization', 'tests')
    log_level : str
        Logging level
    log_dir : str
        Directory to store log files
    include_timestamp : bool
        Whether to include timestamp in log filename
        
    Returns:
    --------
    logging.Logger
        Configured logger instance for the component
    """
    # Create component-specific log directory
    component_log_dir = os.path.join(log_dir, component_name)
    os.makedirs(component_log_dir, exist_ok=True)
    
    # Get logger for this component
    logger = logging.getLogger(component_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler
    if include_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(component_log_dir, f'{component_name}_{timestamp}.log')
    else:
        log_file = os.path.join(component_log_dir, f'{component_name}.log')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    print(f"📝 {component_name} logging to: {log_file}")
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Parameters:
    -----------
    name : str
        Logger name (usually __name__)
        
    Returns:
    --------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)

def log_performance_conditional(func_name: str = None, logger: Optional[logging.Logger] = None, enabled: bool = True):
    """
    Conditional decorator to log function performance.
    
    Parameters:
    -----------
    func_name : str, optional
        Name of the function being monitored (defaults to function name)
    logger : logging.Logger, optional
        Logger instance to use
    enabled : bool
        Whether to enable performance logging (default: True)
    """
    def decorator(func):
        if not enabled:
            return func  # Return original function without decoration
        
        def wrapper(*args, **kwargs):
            import time
            import psutil
            
            # Use provided name or function name
            monitor_name = func_name or func.__name__
            
            # Get logger
            if logger is None:
                log = logging.getLogger(func.__module__)
            else:
                log = logger
            
            # Record start time and memory
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            log.info(f"Starting {monitor_name}")
            
            try:
                result = func(*args, **kwargs)
                
                # Record end time and memory
                elapsed_time = time.time() - start_time
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_used = end_memory - start_memory
                
                log.info(f"{monitor_name} completed: {elapsed_time:.3f}s, memory: {memory_used:.2f}MB")
                return result
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                log.error(f"{monitor_name} failed after {elapsed_time:.3f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator

def log_performance(func_name: str, logger: Optional[logging.Logger] = None):
    """
    Decorator to log function performance.
    
    Parameters:
    -----------
    func_name : str
        Name of the function being monitored
    logger : logging.Logger, optional
        Logger instance to use
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            import psutil
            
            # Get logger
            if logger is None:
                log = logging.getLogger(func.__module__)
            else:
                log = logger
            
            # Record start time and memory
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            log.info(f"Starting {func_name}")
            
            try:
                result = func(*args, **kwargs)
                
                # Record end time and memory
                elapsed_time = time.time() - start_time
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_used = end_memory - start_memory
                
                log.info(f"{func_name} completed: {elapsed_time:.3f}s, memory: {memory_used:.2f}MB")
                return result
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                log.error(f"{func_name} failed after {elapsed_time:.3f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator

def cleanup_old_logs(log_dir: str = 'logs', max_age_days: int = 30):
    """
    Clean up old log files to prevent disk space issues.
    
    Parameters:
    -----------
    log_dir : str
        Directory containing log files
    max_age_days : int
        Maximum age of log files to keep (in days)
    """
    import glob
    from datetime import timedelta
    
    if not os.path.exists(log_dir):
        return
    
    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    
    # Find all log files
    log_pattern = os.path.join(log_dir, '**', '*.log')
    log_files = glob.glob(log_pattern, recursive=True)
    
    deleted_count = 0
    for log_file in log_files:
        file_time = datetime.fromtimestamp(os.path.getctime(log_file))
        if file_time < cutoff_date:
            try:
                os.remove(log_file)
                deleted_count += 1
                print(f"🗑️  Deleted old log file: {log_file}")
            except OSError as e:
                print(f"⚠️  Could not delete {log_file}: {e}")
    
    if deleted_count > 0:
        print(f"🧹 Cleaned up {deleted_count} old log files") 