"""Autonomous search checks for fold-local screening and backward selection."""

from sklearn.linear_model import LogisticRegression
from test_core import dataset, feature

from featune import AutonomousSampler, CVEvaluator, Proposal, create_study
from featune.samplers import BaseFeatureSampler, validate_params


class SequenceSampler(BaseFeatureSampler):
    """Exercise one useful proposal, one constant proposal and one deletion."""

    def sample(self, context):
        """Return a fixed add, screen and delete sequence."""
        if context.number == 1:
            return Proposal(features=[feature()])
        if context.number == 2:
            return Proposal(features=[feature("constant", "is_missing", ["loan"])])
        return Proposal(remove=["ratio"])


def test_autonomous_screening_and_backward_selection():
    """Screen constant columns on train folds and evaluate a deletion against its parent."""
    X, y, schema = dataset()
    study = create_study(search_strategy="autonomous", sampler=SequenceSampler(), cache=False)
    study.optimize(
        X,
        y,
        schema,
        evaluator=CVEvaluator(LogisticRegression(max_iter=1000), cv=2, importance_repeats=0),
        n_trials=3,
        refit=False,
    )
    assert study.trials[2].state == "SCREENED_OUT"
    assert study.trials[2].model_fits == 0
    assert study.trials[3].proposal["remove"] == ["ratio"]
    assert not study.trials[3].candidate_set["features"]
    assert study.best_trial.number == 1
    study.max_features = 1
    assert study._select_parent(4).number == 0


def test_autonomous_sampler_removes_only_leaves_and_params_are_type_strict():
    """Keep dependencies intact and enforce exact parameter value types."""
    _, _, schema = dataset()
    sampler = AutonomousSampler(remove_every=2)
    from featune.samplers import SearchContext

    context = SearchContext(
        schema, 2, [feature(), feature("squared", "square", ["ratio"])], [], "auc", "maximize"
    )
    assert sampler.sample(context).remove == ["squared"]
    full = SearchContext(schema, 3, [feature()], [], "auc", "maximize", remaining_features=0)
    assert sampler.sample(full).remove == ["ratio"]
    leaves = [feature(), feature("loan_abs", "abs", ["loan"])]
    tried = SearchContext(
        schema,
        4,
        leaves,
        [{"parent": 7, "proposal": {"remove": ["loan_abs"]}}],
        "auc",
        "maximize",
        remaining_features=0,
        parent_trial=7,
    )
    assert sampler.sample(tried).remove == ["ratio"]
    for value, allowed in [(1, [True]), (True, [1])]:
        try:
            validate_params({"flag": value}, {"flag": allowed})
        except ValueError:
            pass
        else:
            raise AssertionError("Parameter whitelist accepted a different type")
