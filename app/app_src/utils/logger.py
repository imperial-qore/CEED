import os
import logging


class Logger:

    def __init__(self, log_file_path, log_format=None, log_console=True):
        """
        Initialize the CustomLogger.

        :param log_file_path: Path to save the log file.
        :param log_format: Custom log format. If None, will use a default format.
        """

        self.log_file_path = log_file_path

        # Ensure the directory exists
        try:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        except OSError:
            pass

        # Set default format if none is provided
        if log_format is None:
            log_format = '%(asctime)s - %(levelname)s - %(message)s'

        self.logger = logging.getLogger(log_file_path)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(log_format)

        # File Handler for logging
        if not self.logger.handlers:
            file_handler = logging.FileHandler(self.log_file_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            if log_console:
                # Console Handler for logging
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                console_handler.setLevel(logging.DEBUG)
                self.logger.addHandler(console_handler)

        self.clear()

    def log(self, level, message):
        """
        Log a message.

        :param level: Logging level. E.g., "INFO", "DEBUG", "ERROR", etc.
        :param message: The message to be logged.
        """
        if level.upper() == "INFO":
            self.logger.info(message)
        elif level.upper() == "DEBUG":
            self.logger.debug(message)
        elif level.upper() == "WARNING":
            self.logger.warning(message)
        elif level.upper() == "ERROR":
            self.logger.error(message)
        elif level.upper() == "CRITICAL":
            self.logger.critical(message)
        else:
            self.logger.info(message)

    def clear(self):
        """
        Clear the log file.
        """
        open(self.log_file_path, 'w').close()
