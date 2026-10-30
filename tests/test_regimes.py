import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from fxml.regimes.evaluate import (
    agreement,
    regime_profile,
    selection_table,
    spells,
    transition_matrix,
    vote_k,
)
from fxml.regimes.models import RegimeModel, walk_forward_labels
from fxml.regimes.state import STATE_COLUMNS, market_state

from .conftest import PAIRS

# Three persistent regimes; the calm one is listed last so its raw cluster number is unlikely to
# be 0, which tests the reordering.
MEANS = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],     # normal
    [3.0, 2.5, 1.0, 2.0, 2.5],     # turbulent
    [-3.0, -2.5, -1.0, -2.0, -2.5],  # calm
])
CALM_TO_TURBULENT = [2, 0, 1]


def synthetic_state(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.empty(n, dtype=int)
    truth[0] = 0
    for t in range(1, n):
        truth[t] = truth[t - 1] if rng.random() < 0.97 else rng.integers(3)
    values = MEANS[truth] + rng.normal(0, 0.5, (n, 5))
    dates = pd.bdate_range("2008-01-01", periods=n, name="date")
    return pd.DataFrame(values, index=dates, columns=STATE_COLUMNS), pd.Series(truth, index=dates)


def test_market_state_is_causal(synthetic_table):
    """Dropping every day after a cut-off leaves the state up to it unchanged."""
    full = market_state(synthetic_table, PAIRS)
    cut = pd.Timestamp("2011-03-15")
    dates = synthetic_table.index.get_level_values("date")
    short = market_state(synthetic_table[dates <= cut], PAIRS)
    pd.testing.assert_frame_equal(full.loc[:cut], short)
    assert list(full.columns) == STATE_COLUMNS
    assert full.loc["2010":].notna().all().all()


@pytest.mark.parametrize("method", ["kmeans", "ward", "gmm"])
def test_regimes_are_found_and_ordered_calm_to_turbulent(method):
    state, truth = synthetic_state()
    model = RegimeModel(method, 3).fit(state)
    expected = truth.map({raw: new for new, raw in enumerate(CALM_TO_TURBULENT)})
    assert (model.labels_ == expected).mean() > 0.99
    # the reordering does not depend on how the training rows happen to be ordered
    shuffled = state.sample(frac=1, random_state=1)
    again = RegimeModel(method, 3).fit(shuffled)
    assert (again.predict(state) == model.predict(state)).mean() > 0.99


def test_out_of_sample_rules():
    state, _ = synthetic_state()
    for method in ["ward", "gmm", "dbscan"]:
        model = RegimeModel(method, 3).fit(state)
        assert (model.predict(state) == model.labels_).mean() > 0.97
    gmm = RegimeModel("gmm", 3).fit(state)
    proba = gmm.predict_proba(state)
    assert proba.sum(axis=1) == pytest.approx(1.0)
    assert (proba.argmax(axis=1) == gmm.predict(state)).all()
    # a day far from anything seen in training is noise for DBSCAN
    dbscan = RegimeModel("dbscan").fit(state)
    far = pd.DataFrame([[20.0] * 5], columns=STATE_COLUMNS)
    assert dbscan.predict(far)[0] == -1
    with pytest.raises(ValueError):
        RegimeModel("kmeans", 3).fit(state).predict_proba(state)


def test_walk_forward_labels_never_use_the_future():
    state, _ = synthetic_state()
    labels = walk_forward_labels(state, "gmm", 3, "2010-01-01")
    assert labels.index.equals(state.index)
    assert labels["in_sample"].eq(labels.index < "2010-01-01").all()
    assert {"p0", "p1", "p2"} <= set(labels.columns)
    cut = pd.Timestamp("2011-12-31")
    truncated = walk_forward_labels(state.loc[:cut], "gmm", 3, "2010-01-01")
    pd.testing.assert_frame_equal(labels.loc[:cut], truncated)
    # changing later days changes nothing before them
    altered = state.copy()
    altered.loc[altered.index > cut] *= -1
    pd.testing.assert_frame_equal(
        walk_forward_labels(altered, "gmm", 3, "2010-01-01").loc[:cut], truncated
    )


def test_selection_prefers_the_true_number_of_regimes():
    state, _ = synthetic_state()
    table = selection_table(state, ks=range(2, 6))
    assert set(table["method"]) == {"kmeans", "ward", "gmm"}
    assert table.loc[table["method"] != "gmm", "bic"].isna().all()
    k, votes = vote_k(table)
    assert k == 3
    assert len(votes) == 7  # silhouette and Davies-Bouldin for each method, plus BIC
    with pytest.raises(ValueError):
        vote_k(table, min_share=0.9)


def test_spells_and_transitions():
    dates = pd.bdate_range("2020-01-01", periods=6)
    labels = pd.Series([0, 0, 1, 1, 1, 0], index=dates)
    runs = spells(labels)
    assert runs["regime"].tolist() == [0, 1, 0]
    assert runs["length"].tolist() == [2, 3, 1]
    assert runs["start"].tolist() == [dates[0], dates[2], dates[5]]
    moves = transition_matrix(labels)
    assert moves.loc[0].tolist() == pytest.approx([0.5, 0.5])
    assert moves.loc[1].tolist() == pytest.approx([1 / 3, 2 / 3])


def test_regime_profile_and_agreement():
    state, truth = synthetic_state()
    rv = pd.Series(np.exp(state["vol_short"]) * 1e-5, index=state.index)
    stress = pd.Series(truth == 1, index=state.index)
    profile = regime_profile(truth, rv, stress, state)
    assert profile["share_of_days"].sum() == pytest.approx(1.0)
    assert profile["stress_capture"].sum() == pytest.approx(1.0)
    assert profile.set_index("regime").loc[1, "stress_share"] == 1.0
    assert (profile["mean_spell_days"] > 10).all()
    assert profile["mean_vol_short"].tolist() == pytest.approx(MEANS[:, 0], abs=0.1)
    labels = pd.DataFrame({"truth": truth, "renamed": truth.map({0: 2, 1: 0, 2: 1})})
    ari = agreement(labels)
    assert ari.loc["truth", "renamed"] == pytest.approx(1.0)
    assert adjusted_rand_score(truth, truth) == 1.0
