# scripts/logger.py
import logging
import os
from datetime import datetime

def setup_logger():
    """
    Set up the logger to record activities both in the notebook (console)
    and to a file. The log file is saved with a timestamp in the 'logs' directory.
    
    Returns:
    --------
    logger : logging.Logger
        Configured logger object.
    """
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('../logs'):
        os.makedirs('../logs')
    
    # Generate log file name based on current date
    log_file_name = f"../logs/notebook_activity_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Configure the logger
    logger = logging.getLogger('NotebookLogger')
    logger.setLevel(logging.DEBUG)  # Capture all log levels (DEBUG and above)

    # Create a file handler to save logs to a file
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler to display logs in the notebook (console output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to both handlers (file and console)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# Example Usage in Notebook:
# logger = setup_logger()
# logger.info("Starting the notebook activity logging.")
# logger.error("Error encountered during execution.")