
class TTSBase:
    def __init__(self, log: logging.Logger = None):
        self.log = log or logging.getLogger(__name__)
