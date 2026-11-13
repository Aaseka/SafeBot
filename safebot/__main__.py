import logging

from . import config
from .bot import SafeBot


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = config.load()
    SafeBot(settings).run(settings.token, log_handler=None)


if __name__ == "__main__":
    main()
