import logging

from rich.logging import RichHandler


class DNALogging:

    @staticmethod
    def config_logging():
        """
        Set the logging format.
        :return: None
        """
        handler = RichHandler()
        FORMAT = "%(message)s"
        logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[handler])
