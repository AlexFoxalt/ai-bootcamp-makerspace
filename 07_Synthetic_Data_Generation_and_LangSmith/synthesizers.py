class BaseSynthesizer:
    def __init__(self, percentage: int) -> None:
        self.percentage = percentage

    @property
    def name(self):
        return self.__class__.__name__


class SimpleEvolutionQuerySynthesizer(BaseSynthesizer):
    ...


class MultiContextQuerySynthesizer(BaseSynthesizer):
    ...


class ReasoningQuerySynthesizer(BaseSynthesizer):
    ...
