"""Heterogeneous action space: ask the patient, query the KG, verify a herb, retrieve text.

This is the component that separates the approach from MAI-DxO and AgentClinic,
whose agents choose among homogeneous information requests (ask / order a test).
Here the agent must also decide *which knowledge source is worth consulting*,
trading expected information against a source-specific cost.

The unifying abstraction: **every action is an observation channel.** An action
is fully described by a likelihood vector `p1[d] = P(observation = 1 | condition d)`
plus a scalar cost. Once expressed that way, mutual information is computed
identically for all four action types and they compete on a single scale:

    score(a) = I(D ; o_a) / cost(a)              [efficiency form]
    score(a) = I(D ; o_a) - lambda * cost(a)     [Lagrangian form]

The four channels differ in what they can observe and how reliably:

  ask_patient    any of the 977 attributes, highest fidelity, highest cost -
                 it consumes consultation time and patient patience
  query_kg       only dosha / prakriti attributes, near-authoritative (the graph
                 encodes them directly), very cheap
  retrieve_text  any attribute, but noisy - a corpus statement about a condition
                 is weaker evidence than the patient's own report
  verify_herb    a confirmatory test on the leading hypothesis: is this herb
                 pharmacologically compatible with condition d? Cheap, and
                 informative exactly when the top hypotheses disagree about it
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ayur.data.prep import ConditionSpace
from ayur.data.schema import DOSHAS


class ActionKind(str, Enum):
    ASK_PATIENT = "ask_patient"
    QUERY_KG = "query_kg"
    RETRIEVE_TEXT = "retrieve_text"
    VERIFY_HERB = "verify_herb"
    DIAGNOSE = "diagnose"
    ABSTAIN = "abstain"


#: Default costs, in "one patient question" units. Relative ordering is the
#: design claim, not the absolute numbers: consulting a local index is far
#: cheaper than consuming consultation time, and these are swept in the paper.
DEFAULT_COSTS = {
    ActionKind.ASK_PATIENT: 1.00,
    ActionKind.QUERY_KG: 0.10,
    ActionKind.RETRIEVE_TEXT: 0.30,
    ActionKind.VERIFY_HERB: 0.40,
}

#: Observation noise per channel. The patient is the ground truth about
#: themselves; the corpus is a weaker signal about the same attribute.
DEFAULT_CHANNEL_NOISE = {
    ActionKind.ASK_PATIENT: 0.05,
    ActionKind.QUERY_KG: 0.02,      # the graph states dosha involvement directly
    ActionKind.RETRIEVE_TEXT: 0.25,  # lexical match is a poor proxy for presence
    ActionKind.VERIFY_HERB: 0.10,
}

# --- channel scope: who can answer what -------------------------------------
#
# Two models of the action space, and the difference decides whether *source
# selection* is a real problem at all.
#
# REDUNDANT (the original design): every channel can observe every feature, at
#   its own fidelity. Under this model an agent given unlimited observations
#   should simply use the highest-fidelity channel for everything, and the
#   matched-observation control confirms it does (results/matched_budget.json:
#   0/5 budgets favour multi-source). Selection is genuinely worthless here -
#   the channels are duplicates.
#
# EXCLUSIVE (this model): each channel owns the information it can actually
#   supply, mirroring a real consultation.
#     * A patient reports symptoms, history, lifestyle, environment, diet.
#     * A patient CANNOT reliably self-report their Prakriti or dosha state -
#       in Ayurveda that is a practitioner's assessment, and in AyurGenixAI the
#       `Doshas` / `Constitution/Prakriti` columns are clinical judgements, not
#       self-reports. Only the knowledge graph supplies them.
#     * A patient has no access whatever to herb pharmacology; only the
#       pharmacopoeia can answer whether a herb suits their dosha profile.
#     * The corpus can speak to any attribute, but weakly, and it is the only
#       source that remains when nothing else covers a question.
#
# Under EXCLUSIVE access, asking the wrong source wastes the observation
# entirely, so choosing a source is a genuine planning decision.

#: Columns only the knowledge graph can answer (clinical assessments).
KG_EXCLUSIVE_COLUMNS = ("Doshas", "Constitution/Prakriti")

#: Columns the patient is the authority on (self-reportable facts).
PATIENT_COLUMNS = (
    "Symptoms", "Symptom Severity", "Medical History", "Current Medications",
    "Risk Factors", "Environmental Factors", "Sleep Patterns", "Stress Levels",
    "Physical Activity Levels", "Family History", "Dietary Habits",
    "Allergies (Food/Env)", "Seasonal Variation", "Age Group", "Gender",
    "Occupation and Lifestyle", "Cultural Preferences",
)


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    target: int          # feature index, or herb index for VERIFY_HERB
    cost: float
    label: str

    def __str__(self) -> str:
        return f"{self.kind.value}({self.label}) cost={self.cost:.2f}"


class ActionSpace:
    """Builds the likelihood matrix for every available action.

    `p1` has shape (n_actions, n_conditions): row a is P(o_a = 1 | d).
    """

    def __init__(
        self,
        space: ConditionSpace,
        kg=None,
        costs: dict | None = None,
        channel_noise: dict | None = None,
        enable: set[ActionKind] | None = None,
        exclusive: bool = False,
    ):
        self.space = space
        self.costs = {**DEFAULT_COSTS, **(costs or {})}
        self.noise = {**DEFAULT_CHANNEL_NOISE, **(channel_noise or {})}
        #: See the CHANNEL SCOPE note above. False reproduces the original
        #: redundant-channel design; True partitions access by who can actually
        #: answer, which is both more realistic and the only model under which
        #: source selection is a meaningful decision.
        self.exclusive = exclusive
        self.enable = enable or {
            ActionKind.ASK_PATIENT,
            ActionKind.QUERY_KG,
            ActionKind.RETRIEVE_TEXT,
            ActionKind.VERIFY_HERB,
        }

        self.actions: list[Action] = []
        rows: list[np.ndarray] = []
        #: For channels that observe a patient attribute, which feature it is.
        #: -1 for channels that do not (VERIFY_HERB).
        self.feature_of: list[int] = []

        m = space.matrix.astype(np.float64)

        # --- ask_patient: what a patient can actually report --------------------
        if ActionKind.ASK_PATIENT in self.enable:
            eps = self.noise[ActionKind.ASK_PATIENT]
            for k in range(space.n_features):
                if self.exclusive and space.feature_column(k) not in PATIENT_COLUMNS:
                    continue   # patients cannot self-report dosha or prakriti
                self.actions.append(Action(
                    ActionKind.ASK_PATIENT, k,
                    self.costs[ActionKind.ASK_PATIENT], space.features[k]))
                rows.append(np.where(m[:, k] > 0, 1 - eps, eps))
                self.feature_of.append(k)

        # --- query_kg: dosha and prakriti attributes only ----------------------
        if ActionKind.QUERY_KG in self.enable:
            eps = self.noise[ActionKind.QUERY_KG]
            for k in range(space.n_features):
                if space.feature_column(k) not in KG_EXCLUSIVE_COLUMNS:
                    continue
                self.actions.append(Action(
                    ActionKind.QUERY_KG, k,
                    self.costs[ActionKind.QUERY_KG], space.features[k]))
                rows.append(np.where(m[:, k] > 0, 1 - eps, eps))
                self.feature_of.append(k)

        # --- retrieve_text: any attribute, noisier -----------------------------
        # Under exclusive access the corpus is the fallback for attributes no
        # other channel owns; it is deliberately still allowed to overlap the
        # patient's scope, because in practice literature and patient report do
        # speak to the same things - just at different reliability.
        if ActionKind.RETRIEVE_TEXT in self.enable:
            eps = self.noise[ActionKind.RETRIEVE_TEXT]
            for k in range(space.n_features):
                self.actions.append(Action(
                    ActionKind.RETRIEVE_TEXT, k,
                    self.costs[ActionKind.RETRIEVE_TEXT], space.features[k]))
                rows.append(np.where(m[:, k] > 0, 1 - eps, eps))
                self.feature_of.append(k)

        # --- verify_herb: confirmatory test on pharmacological compatibility ---
        self.herbs: list[str] = []
        if ActionKind.VERIFY_HERB in self.enable and kg is not None:
            eps = self.noise[ActionKind.VERIFY_HERB]
            compat = self._herb_compatibility(kg)
            self._herb_compat_cache = compat
            for j, herb in enumerate(self.herbs):
                col = compat[:, j]
                if col.max() == col.min():
                    continue  # uninformative: agrees with every condition
                self.actions.append(Action(
                    ActionKind.VERIFY_HERB, j,
                    self.costs[ActionKind.VERIFY_HERB], herb))
                rows.append(np.where(col > 0, 1 - eps, eps))
                self.feature_of.append(-1)

        #: Cached (n_conditions, n_herbs) compatibility, needed at answer time.
        if not hasattr(self, "_herb_compat_cache"):
            self._herb_compat_cache = np.zeros((space.n_conditions, 0), dtype=np.int8)

        self.p1 = np.vstack(rows) if rows else np.zeros((0, space.n_conditions))
        self.cost = np.array([a.cost for a in self.actions], dtype=np.float64)
        self.kind = np.array([a.kind.value for a in self.actions])
        self._log_p1 = np.log(self.p1)
        self._log_p0 = np.log1p(-self.p1)

        # Precomputed reverse index: feature -> every action observing it.
        # Scanning `feature_of` on each observation was the dominant cost
        # (2,319 actions x ~25 observations x cases).
        self.actions_for_feature: dict[int, np.ndarray] = {}
        for i, f in enumerate(self.feature_of):
            if f >= 0:
                self.actions_for_feature.setdefault(f, []).append(i)
        self.actions_for_feature = {
            f: np.array(v, dtype=int) for f, v in self.actions_for_feature.items()
        }
        #: action label -> index, for trajectory replay without a linear scan.
        self.index_of_label = {str(a): i for i, a in enumerate(self.actions)}

    def block_feature(self, taken: np.ndarray, feature: int) -> None:
        """Mark every channel observing `feature` as spent."""
        idx = self.actions_for_feature.get(feature)
        if idx is not None:
            taken[idx] = True

    def _herb_compatibility(self, kg) -> np.ndarray:
        """(n_conditions, n_herbs) - 1 if the herb pacifies the condition's doshas."""
        herbs = kg.herbs()
        self.herbs = herbs
        herb_pacifies = [set(kg.objects(h, "pacifies")) for h in herbs]

        compat = np.zeros((self.space.n_conditions, len(herbs)), dtype=np.int8)
        for i, doshas in enumerate(self.space.dosha):
            target = {d for d in doshas if d in DOSHAS}
            if not target:
                continue
            for j, pac in enumerate(herb_pacifies):
                if target & pac:
                    compat[i, j] = 1
        return compat

    def __len__(self) -> int:
        return len(self.actions)

    def summary(self) -> dict:
        from collections import Counter

        counts = Counter(a.kind.value for a in self.actions)
        return {
            "n_actions": len(self.actions),
            "by_kind": dict(counts),
            "costs": {k.value: v for k, v in self.costs.items()},
            "channel_noise": {k.value: v for k, v in self.noise.items()},
            "n_herbs_usable": sum(1 for a in self.actions
                                  if a.kind is ActionKind.VERIFY_HERB),
        }


def action_mutual_information(space: ActionSpace, belief: np.ndarray) -> np.ndarray:
    """I(D ; o_a) for every action, in nats. Vectorised over the whole space."""
    p1 = np.clip(space.p1 @ belief, 1e-12, 1 - 1e-12)
    h_marginal = -(p1 * np.log(p1) + (1 - p1) * np.log1p(-p1))

    p = space.p1
    h_per_condition = -(p * np.log(p) + (1 - p) * np.log1p(-p))
    h_conditional = h_per_condition @ belief

    return h_marginal - h_conditional
