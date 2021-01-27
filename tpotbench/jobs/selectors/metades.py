import pickle

from .selector_job import SelectorJob
from ...models.selectors.metades import METADESSelectorModel


class METADESSelectorJob(SelectorJob):

    @classmethod
    def algo_type(cls) -> str:
        return 'metades'

    def model(self) -> METADESSelectorModel:
        with open(self.paths()['files']['model'], 'rb') as f:
            selector_model = pickle.load(f)

        classifier_models = [clf.model() for clf in self.classifiers]

        return METADESSelectorModel(self.name(), selector_model, classifier_models)
