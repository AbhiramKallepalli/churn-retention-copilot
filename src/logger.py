import logging
import os
from datetime import datetime

def setup_logger():
    """
    Set up logging to file and console
    """
    # Create outputs directory if not exists
    os.makedirs('outputs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('churn_copilot')
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler - writes to session.log
    log_file = 'outputs/session.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler - shows in terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    # Format for log messages
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_query(customer_id, risk_category, 
              churn_probability, response_time,
              llm_valid=True):
    """
    Log a customer query to session.log
    """
    logger = setup_logger()
    
    logger.info(
        f"customer_id={customer_id} | "
        f"risk_category={risk_category} | "
        f"churn_probability={churn_probability}% | "
        f"response_time={response_time:.2f}s | "
        f"llm_valid={llm_valid}"
    )


def log_error(error_message):
    """
    Log an error to session.log
    """
    logger = setup_logger()
    logger.error(f"ERROR | {error_message}")


def log_session_start():
    """
    Log when a new session starts
    """
    logger = setup_logger()
    logger.info("=" * 50)
    logger.info("NEW SESSION STARTED")
    logger.info("=" * 50)


def log_summary_query(total, high, medium, low):
    """
    Log a summary query
    """
    logger = setup_logger()
    logger.info(
        f"SUMMARY QUERY | "
        f"total={total} | "
        f"high={high} | "
        f"medium={medium} | "
        f"low={low}"
    )


# Test it
if __name__ == "__main__":
    # Test logging
    log_session_start()
    log_query(
        customer_id="7590-VHVEG",
        risk_category="MEDIUM RISK",
        churn_probability=65.0,
        response_time=2.5,
        llm_valid=True
    )
    log_query(
        customer_id="9497-QCMMS",
        risk_category="HIGH RISK",
        churn_probability=95.4,
        response_time=3.1,
        llm_valid=True
    )
    log_error("Test error message")
    log_summary_query(7043, 592, 2021, 4430)
    
    print("Logging test complete!")
    print("Check outputs/session.log for results")