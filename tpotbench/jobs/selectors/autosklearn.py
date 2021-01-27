import pickle

from .selector_job import SelectorJob
from ...models.selectors.autosklearn import AutoSklearnSelectorModel


class AutoSklearnSelectorJob(SelectorJob):

    @classmethod
    def algo_type(cls) -> str:
        return 'autosklearn'

    def model(self) -> AutoSklearnSelectorModel:
        automodel = None
        with open(self.paths()['files']['model'], 'rb') as f:
            automodel = pickle.load(f)

        classifier_models = [clf.model() for clf in self.classifiers]

        return AutoSklearnSelectorModel(self.name(), automodel, classifier_models)
