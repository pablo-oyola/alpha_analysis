import logging
import sys


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds ANSI color codes to log output."""
    
    # ANSI color codes
    COLORS = {
        'ERROR': '\033[91m',      # Red
        'WARNING': '\033[93m',    # Yellow
        'INFO': '\033[92m',       # Green
        'DEBUG': '\033[94m',      # Blue
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors and custom format."""
        # Get the color for this level
        level_name = record.levelname
        color = self.COLORS.get(level_name, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format: [LEVEL] file:line - Message
        formatted_level = f"{color}[{level_name}]{reset}"
        formatted_msg = f"{formatted_level} {record.filename}:{record.lineno} - {record.getMessage()}"
        
        return formatted_msg


def get_logger(name):
    """Get a configured logger for the given name.
    
    Parameters
    ----------
    name : str
        Logger name, typically __name__ from the calling module.
    
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Only add handler if logger doesn't already have one
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        
        # Create and set formatter
        formatter = ColoredFormatter()
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger
